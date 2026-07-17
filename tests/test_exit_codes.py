from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark_engine.cli import main
from benchmark_engine.engine import RunOutcome
from benchmark_engine.execution import WorkerOutcome, WorkerResponse, WorkerStage

from tests.test_correctness_cli_e2e import OPERATOR, PASS


ROOT = Path(__file__).resolve().parents[1]


class ExitCodeTests(unittest.TestCase):
    def run_pass_candidate(self, output: Path) -> int:
        return main(
            (
                "run", "--mode", "correctness", "--operator", OPERATOR,
                "--candidate", PASS, "--case", "tiny",
                "--output-root", str(output),
            ),
            repository_root=ROOT,
        )

    def test_run_outcome_exit_code_contract_covers_all_terminal_classes(self):
        self.assertEqual(RunOutcome("run", (), 1, 0, 0).exit_code, 0)
        self.assertEqual(RunOutcome("run", (), 0, 1, 0).exit_code, 1)
        self.assertEqual(RunOutcome("run", (), 0, 0, 1).exit_code, 3)
        self.assertEqual(RunOutcome("run", (), 0, 0, 0, interrupted=True).exit_code, 130)

    def test_selector_errors_exit_two(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "results"
            self.assertEqual(
                main(
                    ("run", "--operator", "missing", "--output-root", str(output)),
                    repository_root=ROOT,
                ),
                2,
            )

    def test_raw_artifact_oserror_is_infrastructure_exit_3(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "benchmark_engine.cli.execute_plan",
                side_effect=OSError("artifact disk unavailable"),
            ):
                self.assertEqual(
                    self.run_pass_candidate(Path(temporary) / "results"), 3
                )

    def test_keyboard_interrupt_at_planning_boundary_is_exit_130(self):
        with tempfile.TemporaryDirectory() as temporary:
            with patch(
                "benchmark_engine.cli.build_execution_plan",
                side_effect=KeyboardInterrupt,
            ):
                self.assertEqual(
                    self.run_pass_candidate(Path(temporary) / "results"), 130
                )

    def test_worker_without_payload_preserves_infrastructure_and_gate_classes(self):
        class FakeController:
            outcome = WorkerOutcome.ERROR
            stage = WorkerStage.IMPORT

            def __init__(self, **_kwargs):
                pass

            def run(self, job, **_kwargs):
                if _kwargs.get("cuda_devices") != ():
                    raise AssertionError("CPU manifest must pass no CUDA generator devices")
                error_fields = {
                    WorkerOutcome.ERROR: ("ImportError", "reference import failed"),
                    WorkerOutcome.CRASHED: ("WorkerCrash", "worker exited without payload"),
                    WorkerOutcome.TIMEOUT: ("StageTimeout", "correctness stage exceeded 1s"),
                    WorkerOutcome.OOM: ("MemoryError", "worker exhausted memory"),
                    WorkerOutcome.UNSUPPORTED: (
                        "NotImplementedError",
                        "backend unsupported",
                    ),
                }
                error_type, error_message = error_fields[self.outcome]
                return WorkerResponse(
                    identity=job.identity,
                    result_id=job.result_id,
                    outcome=self.outcome,
                    stage=self.stage,
                    worker_pid=123,
                    started_at_utc="2026-07-16T12:00:00Z",
                    finished_at_utc="2026-07-16T12:00:01Z",
                    elapsed_s=1.0,
                    stage_elapsed_s={},
                    error_type=error_type,
                    error_message=error_message,
                    exit_code=1,
                )

        for outcome, stage, expected, correctness_status in (
            (WorkerOutcome.ERROR, WorkerStage.IMPORT, 3, None),
            (WorkerOutcome.CRASHED, WorkerStage.CORRECTNESS, 3, None),
            (WorkerOutcome.TIMEOUT, WorkerStage.CORRECTNESS, 1, "timeout"),
            (WorkerOutcome.OOM, WorkerStage.CORRECTNESS, 1, "oom"),
            (WorkerOutcome.UNSUPPORTED, WorkerStage.CORRECTNESS, 1, "unsupported"),
        ):
            with self.subTest(outcome=outcome), tempfile.TemporaryDirectory() as temporary:
                FakeController.outcome = outcome
                FakeController.stage = stage
                output = Path(temporary) / "results"
                with patch("benchmark_engine.engine.WorkerController", FakeController):
                    self.assertEqual(
                        self.run_pass_candidate(output), expected
                    )
                if correctness_status is None:
                    continue
                evaluations = [
                    path
                    for path in output.glob(f"{OPERATOR}/{PASS}/*")
                    if path.is_dir()
                ]
                self.assertEqual(len(evaluations), 1)
                evaluation = evaluations[0]
                with (evaluation / "results.csv").open(
                    newline="", encoding="utf-8"
                ) as stream:
                    rows = list(csv.DictReader(stream))
                self.assertEqual(len(rows), 1)
                row = rows[0]
                self.assertEqual(row["correctness_status"], correctness_status)
                self.assertEqual(row["performance_status"], "skipped")
                expected_error_type, expected_error_message = {
                    WorkerOutcome.TIMEOUT: (
                        "StageTimeout",
                        "correctness stage exceeded 1s",
                    ),
                    WorkerOutcome.OOM: ("MemoryError", "worker exhausted memory"),
                    WorkerOutcome.UNSUPPORTED: (
                        "NotImplementedError",
                        "backend unsupported",
                    ),
                }[outcome]
                self.assertEqual(row["error_type"], expected_error_type)
                self.assertEqual(row["error_message"], expected_error_message)
                manifest = json.loads(
                    (evaluation / "evaluation_manifest.json").read_text(
                        encoding="utf-8"
                    )
                )
                self.assertEqual(manifest["status"], "complete")


if __name__ == "__main__":
    unittest.main()
