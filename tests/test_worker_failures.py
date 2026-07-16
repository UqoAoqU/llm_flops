from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from benchmark_engine.execution import StageTimeouts, WorkerController, WorkerOutcome
from benchmark_engine.execution.protocol import WorkerResponse, split_event_attempts
from benchmark_engine.execution.isolation import process_exists

from tests.worker_fixtures import FIXTURE_ROOT, make_job


class WorkerFailureTests(unittest.TestCase):
    def run_fixture(
        self,
        fixture: str,
        *,
        timeout: float = 1.0,
        controller: WorkerController | None = None,
    ):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        output = Path(temporary.name) / "evaluation"
        response = (controller or WorkerController()).run(
            make_job(output, fixture),
            spec_entrypoint="spec:spec",
            timeouts=StageTimeouts(timeout, timeout, timeout, timeout),
        )
        return response, output

    def test_import_exception_oom_and_unsupported_are_distinct(self) -> None:
        expectations = {
            "import_error": WorkerOutcome.ERROR,
            "oom": WorkerOutcome.OOM,
            "unsupported": WorkerOutcome.UNSUPPORTED,
        }
        for fixture, expected in expectations.items():
            with self.subTest(fixture=fixture):
                response, output = self.run_fixture(fixture)
                self.assertIs(response.outcome, expected)
                self.assertEqual(response.stage.value, "import")
                self.assertIsNotNone(response.diagnostic_path)
                self.assertTrue((output / response.diagnostic_path).is_file())
                self.assertLessEqual(len(response.error_message or ""), 512)

    @unittest.skipUnless(os.name == "posix", "signals/process groups require POSIX")
    def test_segfault_is_crashed_without_terminating_controller(self) -> None:
        # This host can take slightly over one second to report a real SIGSEGV
        # after the fixture has disabled core files.  The explicit finite
        # budget accommodates that kernel/coredump-helper exit-reporting delay;
        # it is not a heartbeat-based extension of the controller deadline.
        started = time.monotonic()
        response, output = self.run_fixture("segfault", timeout=3.0)
        elapsed = time.monotonic() - started
        self.assertLess(elapsed, 3.0)
        self.assertIs(response.outcome, WorkerOutcome.CRASHED)
        self.assertLess(response.exit_code or 0, 0)
        self.assertIsNotNone(response.diagnostic_path)
        self.assertTrue((output / response.diagnostic_path).is_file())

    @unittest.skipUnless(os.name == "posix", "process-tree assertion requires Linux")
    def test_import_timeout_kills_worker_and_spawned_child(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "evaluation"
            child_marker = Path(temporary) / "child.pid"
            with patch.dict(
                os.environ,
                {
                    "BENCHMARK_ENGINE_CHILD_PID_FILE": str(child_marker),
                    "BENCHMARK_ENGINE_HEARTBEAT_INTERVAL_S": "0.05",
                },
                clear=False,
            ):
                response = WorkerController(poll_interval_s=0.01).run(
                    make_job(output, "spawn_child_hang"),
                    spec_entrypoint="spec:spec",
                    timeouts=StageTimeouts(0.3, 1, 1, 1),
                )
            self.assertIs(response.outcome, WorkerOutcome.TIMEOUT)
            self.assertEqual(response.stage.value, "import")
            self.assertTrue(child_marker.is_file())
            child_pid = int(child_marker.read_text(encoding="utf-8"))
            deadline = time.monotonic() + 2
            while process_exists(child_pid) and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertFalse(process_exists(response.worker_pid))
            self.assertFalse(process_exists(child_pid))

    def test_output_flood_is_drained_and_capped_with_marker(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "evaluation"
            first_job = make_job(output, "flood")
            second_job = replace(
                first_job, result_id="res_22222222222222222222222222222222"
            )
            controller = WorkerController(
                log_limit_bytes=4096, summary_limit_bytes=1024
            )
            for job in (first_job, second_job):
                response = controller.run(
                    job,
                    spec_entrypoint="spec:spec",
                    timeouts=StageTimeouts(3, 3, 3, 3),
                )
                self.assertIs(response.outcome, WorkerOutcome.SUCCESS)
            for name in ("stdout.log", "stderr.log"):
                data = (output / "logs" / name).read_bytes()
                self.assertLessEqual(len(data), 4096)
                self.assertEqual(data.count(b"log truncated"), 1)

    @unittest.skipUnless(os.name == "posix", "interrupt cleanup requires POSIX")
    def test_keyboard_interrupt_cleans_worker_and_maps_exit_130(self) -> None:
        calls = 0

        def interrupt_poll(_seconds: float) -> None:
            nonlocal calls
            calls += 1
            if calls == 1:
                time.sleep(0.05)
                raise KeyboardInterrupt

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "evaluation"
            response = WorkerController(
                poll_interval_s=0.01, poll_sleep=interrupt_poll
            ).run(make_job(output, "hang"), spec_entrypoint="spec:spec")
            self.assertIs(response.outcome, WorkerOutcome.INTERRUPTED)
            self.assertEqual(response.exit_code, 130)
            self.assertFalse(process_exists(response.worker_pid))
            from benchmark_engine.execution.event_log import EventLog
            from benchmark_engine.execution import validate_event_sequence

            validate_event_sequence(EventLog(output / "logs" / "worker.jsonl").read())

    @unittest.skipUnless(os.name == "posix", "stale crash response test requires POSIX")
    def test_repeated_result_never_consumes_a_stale_success_response(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "evaluation"
            first_job = make_job(output)
            controller = WorkerController(poll_interval_s=0.01)
            first = controller.run(first_job, spec_entrypoint="spec:spec")
            crashing_job = replace(
                first_job,
                candidate=replace(
                    first_job.candidate, root=FIXTURE_ROOT / "segfault"
                ),
            )
            second = controller.run(
                crashing_job,
                spec_entrypoint="spec:spec",
                timeouts=StageTimeouts(3, 1, 1, 1),
            )
            self.assertIs(first.outcome, WorkerOutcome.SUCCESS)
            self.assertIs(second.outcome, WorkerOutcome.CRASHED)
            self.assertLess(second.exit_code or 0, 0)
            response_paths = sorted(
                (output / "diagnostics").glob("*.response.json")
            )
            self.assertEqual(len(response_paths), 2)
            stored = [
                WorkerResponse.from_json(path.read_text(encoding="utf-8"))
                for path in response_paths
            ]
            self.assertEqual(
                {response.outcome for response in stored},
                {WorkerOutcome.SUCCESS, WorkerOutcome.CRASHED},
            )
            from benchmark_engine.execution.event_log import EventLog

            attempts = split_event_attempts(
                EventLog(output / "logs" / "worker.jsonl").read()
            )
            self.assertEqual(len(attempts), 2)


if __name__ == "__main__":
    unittest.main()
