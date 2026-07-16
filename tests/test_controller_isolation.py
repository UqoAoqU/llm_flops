from __future__ import annotations

import os
import sys
import tempfile
import unittest
import gc
import json
import warnings
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from benchmark_engine.execution import (
    StageTimeouts,
    WorkerController,
    WorkerOutcome,
    validate_event_sequence,
)
from benchmark_engine.execution.event_log import EventLog
from benchmark_engine.execution.isolation import BoundedLogDrain
from benchmark_engine.execution.protocol import split_event_attempts

from tests.worker_fixtures import make_job


class ControllerIsolationTests(unittest.TestCase):
    def test_controller_never_imports_candidate_and_worker_load_order_is_stable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "order.txt"
            before = frozenset(sys.modules)
            with patch.dict(
                os.environ,
                {"BENCHMARK_ENGINE_IMPORT_ORDER": str(marker)},
                clear=False,
            ):
                response = WorkerController().run(
                    make_job(root / "evaluation"),
                    spec_entrypoint="spec:spec",
                    timeouts=StageTimeouts(2, 2, 2, 2),
                )
            self.assertIs(response.outcome, WorkerOutcome.SUCCESS)
            self.assertEqual(
                marker.read_text(encoding="utf-8").splitlines(),
                ["spec", "reference", "candidate"],
            )
            added = frozenset(sys.modules).difference(before)
            self.assertFalse(
                any(name.startswith("_benchmark_engine_worker_candidate") for name in added)
            )
            self.assertNotIn("implementation", sys.modules)

    def test_normal_worker_is_a_distinct_process_and_leaves_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "evaluation"
            response = WorkerController().run(
                make_job(output), spec_entrypoint="spec:spec"
            )
            self.assertNotEqual(response.worker_pid, os.getpid())
            self.assertTrue((output / "logs" / "stdout.log").is_file())
            self.assertTrue((output / "logs" / "stderr.log").is_file())
            self.assertTrue((output / "logs" / "worker.jsonl").is_file())
            self.assertTrue(response.stage_elapsed_s["import"] >= 0)

    def test_multiple_results_append_evaluation_logs_and_keep_responses(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "evaluation"
            first_job = make_job(output)
            second_job = replace(
                first_job,
                result_id="res_11111111111111111111111111111111",
                case=replace(first_job.case, case_id="second"),
            )
            controller = WorkerController()
            first = controller.run(first_job, spec_entrypoint="spec:spec")
            events_after_first = (output / "logs" / "worker.jsonl").read_bytes()
            second = controller.run(second_job, spec_entrypoint="spec:spec")
            self.assertIs(first.outcome, WorkerOutcome.SUCCESS)
            self.assertIs(second.outcome, WorkerOutcome.SUCCESS)
            all_event_bytes = (output / "logs" / "worker.jsonl").read_bytes()
            self.assertTrue(all_event_bytes.startswith(events_after_first))
            events = EventLog(output / "logs" / "worker.jsonl").read()
            validate_event_sequence(events)
            self.assertEqual({event.result_id for event in events}, {first_job.result_id, second_job.result_id})
            controller_lines = (output / "logs" / "controller.jsonl").read_text().splitlines()
            self.assertEqual(len(controller_lines), 2)
            self.assertEqual(
                {json.loads(line)["result_id"] for line in controller_lines},
                {first_job.result_id, second_job.result_id},
            )
            self.assertEqual(len(tuple((output / "diagnostics").glob("*.response.json"))), 2)

    def test_repeated_result_creates_independent_attempt_transcripts_and_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "evaluation"
            job = make_job(output)
            controller = WorkerController()
            first = controller.run(job, spec_entrypoint="spec:spec")
            first_event_bytes = (output / "logs" / "worker.jsonl").read_bytes()
            second = controller.run(job, spec_entrypoint="spec:spec")
            self.assertIs(first.outcome, WorkerOutcome.SUCCESS)
            self.assertIs(second.outcome, WorkerOutcome.SUCCESS)
            event_bytes = (output / "logs" / "worker.jsonl").read_bytes()
            self.assertTrue(event_bytes.startswith(first_event_bytes))
            events = EventLog(output / "logs" / "worker.jsonl").read()
            attempts = split_event_attempts(events)
            self.assertEqual(len(attempts), 2)
            for attempt in attempts:
                self.assertEqual({event.result_id for event in attempt}, {job.result_id})
                validate_event_sequence(attempt)
            validate_event_sequence(events)
            diagnostics = output / "diagnostics"
            self.assertEqual(len(tuple(diagnostics.glob("*.request.json"))), 2)
            self.assertEqual(len(tuple(diagnostics.glob("*.response.json"))), 2)

    def test_worker_pipes_are_closed_without_resource_warnings(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            drains: list[BoundedLogDrain] = []

            class RecordingDrain(BoundedLogDrain):
                def __init__(self, *args, **kwargs):
                    super().__init__(*args, **kwargs)
                    drains.append(self)

            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter("always", ResourceWarning)
                with patch(
                    "benchmark_engine.execution.controller.BoundedLogDrain",
                    RecordingDrain,
                ):
                    WorkerController().run(
                        make_job(Path(temporary) / "evaluation"),
                        spec_entrypoint="spec:spec",
                    )
                gc.collect()
            self.assertEqual(len(drains), 2)
            self.assertTrue(all(drain.source.closed for drain in drains))
            self.assertFalse(
                [warning for warning in observed if issubclass(warning.category, ResourceWarning)]
            )


if __name__ == "__main__":
    unittest.main()
