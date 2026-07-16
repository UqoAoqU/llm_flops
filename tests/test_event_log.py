from __future__ import annotations

import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark_engine.execution import (
    EventName,
    StageTimeouts,
    WorkerController,
    WorkerOutcome,
    validate_event_sequence,
)
from benchmark_engine.execution.event_log import EventLog

from tests.worker_fixtures import make_job
from tests.worker_fixtures import FIXTURE_ROOT


class EventLogTests(unittest.TestCase):
    def test_all_stages_are_paired_and_unimplemented_stages_are_skipped(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "evaluation"
            response = WorkerController().run(
                make_job(output), spec_entrypoint="spec:spec"
            )
            self.assertIs(response.outcome, WorkerOutcome.SUCCESS)
            events = EventLog(output / "logs" / "worker.jsonl").read()
            validate_event_sequence(events)
            by_name = {event.event: event for event in events}
            for started, finished in (
                (EventName.IMPORT_STARTED, EventName.IMPORT_FINISHED),
                (EventName.BUILD_STARTED, EventName.BUILD_FINISHED),
                (EventName.CORRECTNESS_STARTED, EventName.CORRECTNESS_FINISHED),
                (EventName.WARMUP_STARTED, EventName.WARMUP_FINISHED),
                (EventName.SAMPLING_STARTED, EventName.SAMPLING_FINISHED),
            ):
                self.assertIn(started, by_name)
                self.assertIn(finished, by_name)
            self.assertEqual(by_name[EventName.CORRECTNESS_FINISHED].status, "skipped")
            self.assertEqual(by_name[EventName.SAMPLING_FINISHED].status, "skipped")

    @unittest.skipUnless(os.name == "posix", "hard timeout test requires POSIX")
    def test_heartbeats_report_stage_but_do_not_reset_hard_timeout(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "evaluation"
            started = time.monotonic()
            with patch.dict(
                os.environ,
                {"BENCHMARK_ENGINE_HEARTBEAT_INTERVAL_S": "0.05"},
                clear=False,
            ):
                response = WorkerController(poll_interval_s=0.01).run(
                    make_job(output, "hang"),
                    spec_entrypoint="spec:spec",
                    timeouts=StageTimeouts(0.3, 1, 1, 1),
                )
            elapsed = time.monotonic() - started
            self.assertIs(response.outcome, WorkerOutcome.TIMEOUT)
            self.assertLess(elapsed, 2)
            events = EventLog(output / "logs" / "worker.jsonl").read()
            heartbeats = [event for event in events if event.event is EventName.HEARTBEAT]
            self.assertTrue(heartbeats)
            self.assertTrue(all(event.stage.value == "import" for event in heartbeats))
            self.assertTrue(all(event.pid == response.worker_pid for event in heartbeats))
            self.assertTrue(all(len(event.log_tail.encode()) <= 16 * 1024 for event in heartbeats))
            validate_event_sequence(events)

    @unittest.skipUnless(os.name == "posix", "build process-tree test requires POSIX")
    def test_build_timeout_heartbeat_reports_children_and_reaps_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "evaluation"
            marker = Path(temporary) / "build-children.txt"
            with patch.dict(
                os.environ,
                {
                    "BENCHMARK_ENGINE_BUILD_CHILD_PID_FILE": str(marker),
                    "BENCHMARK_ENGINE_HEARTBEAT_INTERVAL_S": "0.05",
                },
                clear=False,
            ):
                response = WorkerController(poll_interval_s=0.01).run(
                    make_job(output),
                    spec_entrypoint="spec:spec",
                    build_argv=(
                        sys.executable,
                        str(FIXTURE_ROOT / "normal" / "build_spawn_hang.py"),
                    ),
                    timeouts=StageTimeouts(1, 0.35, 1, 1),
                )
            self.assertIs(response.outcome, WorkerOutcome.TIMEOUT)
            self.assertEqual(response.stage.value, "build")
            events = EventLog(output / "logs" / "worker.jsonl").read()
            heartbeats = [
                event
                for event in events
                if event.event is EventName.HEARTBEAT and event.stage.value == "build"
            ]
            self.assertTrue(heartbeats)
            observed = {pid for event in heartbeats for pid in event.child_pids}
            declared = {int(line) for line in marker.read_text().splitlines()}
            self.assertTrue(observed.intersection(declared))
            from benchmark_engine.execution.isolation import process_exists

            deadline = time.monotonic() + 2
            while any(process_exists(pid) for pid in declared) and time.monotonic() < deadline:
                time.sleep(0.02)
            self.assertFalse(process_exists(response.worker_pid))
            self.assertTrue(all(not process_exists(pid) for pid in declared))
            validate_event_sequence(events)


if __name__ == "__main__":
    unittest.main()
