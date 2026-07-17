from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark_engine.execution import (
    EventName,
    StageTimeouts,
    WorkerController,
    WorkerOutcome,
    WorkerStage,
    validate_event_sequence,
)
from benchmark_engine.execution.event_log import EventEmitter, EventLog

from tests.worker_fixtures import make_job
from tests.worker_fixtures import FIXTURE_ROOT


class EventLogTests(unittest.TestCase):
    def test_heartbeat_append_is_atomic_with_stage_finish(self) -> None:
        class MemoryEventLog:
            def __init__(self) -> None:
                self.events = []

            def append(self, event) -> None:
                self.events.append(event)

        job = make_job(Path.cwd() / "event-race-evaluation")
        request = type(
            "Request",
            (),
            {"identity": job.identity, "result_id": job.result_id},
        )()
        log = MemoryEventLog()
        emitter = EventEmitter(log, request)
        emitter.emit(EventName.DISCOVERED, WorkerStage.STARTUP, "discovered")
        emitter.emit(EventName.WORKER_STARTED, WorkerStage.STARTUP, "started")
        emitter.emit(EventName.IMPORT_STARTED, WorkerStage.IMPORT, "started")

        original_emit = emitter.emit
        heartbeat_entered = threading.Event()
        release_heartbeat = threading.Event()
        finished_started = threading.Event()
        thread_errors = []

        def coordinated_emit(event, stage, status, **kwargs):
            if event is EventName.HEARTBEAT:
                heartbeat_entered.set()
                if not release_heartbeat.wait(2):
                    raise TimeoutError("test did not release heartbeat")
            return original_emit(event, stage, status, **kwargs)

        emitter.emit = coordinated_emit

        def heartbeat_target() -> None:
            try:
                emitter.heartbeat("still importing")
            except BaseException as error:
                thread_errors.append(error)

        def finished_target() -> None:
            try:
                finished_started.set()
                original_emit(
                    EventName.IMPORT_FINISHED,
                    WorkerStage.IMPORT,
                    "success",
                )
            except BaseException as error:
                thread_errors.append(error)

        with patch(
            "benchmark_engine.execution.event_log.linux_descendant_pids",
            return_value=(),
        ):
            heartbeat = threading.Thread(target=heartbeat_target)
            finished = threading.Thread(target=finished_target)
            heartbeat.start()
            self.assertTrue(heartbeat_entered.wait(1))
            finished.start()
            self.assertTrue(finished_started.wait(1))
            # The FINISHED call has a full scheduling window. With the old
            # Lock/check-then-emit implementation it completes here; with the
            # RLock atomic section it is blocked until heartbeat is appended.
            finished.join(timeout=0.2)
            self.assertTrue(finished.is_alive())
            release_heartbeat.set()
            heartbeat.join(timeout=2)
            finished.join(timeout=2)

        self.assertFalse(heartbeat.is_alive())
        self.assertFalse(finished.is_alive())
        self.assertFalse(thread_errors, thread_errors)
        for started, finished_event, stage in (
            (EventName.BUILD_STARTED, EventName.BUILD_FINISHED, WorkerStage.BUILD),
            (EventName.CORRECTNESS_STARTED, EventName.CORRECTNESS_FINISHED, WorkerStage.CORRECTNESS),
            (EventName.WARMUP_STARTED, EventName.WARMUP_FINISHED, WorkerStage.WARMUP),
            (EventName.SAMPLING_STARTED, EventName.SAMPLING_FINISHED, WorkerStage.SAMPLING),
        ):
            original_emit(started, stage, "started")
            original_emit(finished_event, stage, "success")
        original_emit(EventName.REPORT_WRITTEN, WorkerStage.REPORT, "success")
        original_emit(EventName.WORKER_EXITED, WorkerStage.COMPLETE, "success")
        self.assertEqual(
            [event.event for event in log.events[2:5]],
            [
                EventName.IMPORT_STARTED,
                EventName.HEARTBEAT,
                EventName.IMPORT_FINISHED,
            ],
        )
        validate_event_sequence(tuple(log.events))

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
