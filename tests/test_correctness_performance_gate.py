import csv
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from benchmark_engine.cli import main
from benchmark_engine.execution import WorkerOutcome, WorkerResponse, WorkerStage
from tests.test_correctness_cli_e2e import NUMERIC, OPERATOR, PASS, ROOT


class CorrectnessPerformanceGateTests(unittest.TestCase):
    def test_unsupported_policy_controls_exit_but_never_ranking(self):
        class UnsupportedController:
            def __init__(self, **kwargs): pass
            def run(self, job, **kwargs):
                return WorkerResponse(identity=job.identity, result_id=job.result_id,
                    outcome=WorkerOutcome.SUCCESS, stage=WorkerStage.COMPLETE,
                    worker_pid=1, started_at_utc="2026-07-17T00:00:00Z",
                    finished_at_utc="2026-07-17T00:00:01Z", elapsed_s=1,
                    stage_elapsed_s={}, result_payload={"status": "pass",
                    "case_hash": "hash", "input_summary": {},
                    "comparison": {"metrics": {}}, "output_contracts": {},
                    "performance": {"status": "unsupported", "reason": "not supported"}})
        for policy, expected, expected_status, expected_gate in (
            ("fail", 1, "unsupported", "failed"),
            ("allow", 0, "passed", "passed"),
        ):
            with self.subTest(policy=policy), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "results"
                with patch("benchmark_engine.engine.WorkerController", UnsupportedController):
                    code = main(("run", "--mode", "performance", "--operator", OPERATOR,
                        "--candidate", PASS, "--case", "tiny", "--timer", "wall_clock",
                        "--unsupported-policy", policy, "--output-root", str(output)),
                        repository_root=ROOT)
                self.assertEqual(code, expected)
                evaluation = next(path for path in output.glob(f"{OPERATOR}/{PASS}/*") if path.is_dir())
                with (evaluation / "results.csv").open(newline="", encoding="utf-8") as stream:
                    row = next(csv.DictReader(stream))
                self.assertEqual(row["performance_status"], "unsupported")
                self.assertEqual(row["status"], expected_status)
                self.assertEqual(row["performance_gate_status"], expected_gate)
                self.assertEqual(row["ranking_eligible"], "false")
    def test_opt_in_measures_but_is_permanently_nonformal(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "results"
            code = main(("run", "--mode", "performance", "--operator", OPERATOR,
                "--candidate", NUMERIC, "--case", "tiny", "--timer", "wall_clock",
                "--samples", "2", "--inner-iterations", "1", "--warmup", "0",
                "--perf-on-correctness-fail", "--output-root", str(output)),
                repository_root=ROOT)
            self.assertEqual(code, 1)
            evaluation = next(path for path in output.glob(f"{OPERATOR}/{NUMERIC}/*") if path.is_dir())
            with (evaluation / "performance_samples.csv").open(newline="", encoding="utf-8") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 4)
            with (evaluation / "results.csv").open(newline="", encoding="utf-8") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["correctness_status"], "failed")
            self.assertEqual(row["performance_formal"], "false")
            self.assertEqual(row["ranking_eligible"], "false")
            self.assertEqual(row["perf_on_correctness_fail"], "true")


if __name__ == "__main__": unittest.main()
