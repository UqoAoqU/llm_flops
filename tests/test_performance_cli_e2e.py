from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

from benchmark_engine.cli import main
from benchmark_engine.execution import WorkerOutcome, WorkerResponse, WorkerStage

from tests.test_correctness_cli_e2e import NUMERIC, OPERATOR, PASS, ROOT


class PerformanceCliE2ETests(unittest.TestCase):
    def invoke(self, *args: str):
        stdout = io.StringIO()
        stderr = io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = main(args, repository_root=ROOT)
        return code, stdout.getvalue(), stderr.getvalue()

    def run_candidate(self, output: Path, candidate: str, *extra: str):
        return self.invoke(
            "run",
            "--mode",
            "performance",
            "--operator",
            OPERATOR,
            "--candidate",
            candidate,
            "--case",
            "tiny",
            "--timer",
            "wall_clock",
            "--warmup",
            "1",
            "--samples",
            "2",
            "--inner-iterations",
            "2",
            "--output-root",
            str(output),
            *extra,
        )

    @staticmethod
    def evaluation(output: Path, candidate: str) -> Path:
        return next(path for path in output.glob(f"{OPERATOR}/{candidate}/*") if path.is_dir())

    def test_cpu_performance_writes_both_roles_and_real_stage_events(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "results"
            code, stdout, stderr = self.run_candidate(output, PASS)
            self.assertEqual((code, stderr), (1, ""))
            evaluation = self.evaluation(output, PASS)
            with (evaluation / "results.csv").open(newline="", encoding="utf-8") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["correctness_status"], "passed")
            self.assertEqual(rows[0]["performance_status"], "unstable")
            self.assertEqual(rows[0]["status"], "failed")
            self.assertEqual(rows[0]["performance_gate_status"], "failed")
            self.assertEqual(rows[0]["ranking_eligible"], "false")
            self.assertEqual(rows[0]["requested_timer"], "wall_clock")
            self.assertEqual(rows[0]["effective_timer"], "wall_clock")
            manifest = json.loads(
                (evaluation / "evaluation_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                rows[0]["torch_version"],
                manifest["environment"]["snapshot"]["packages"]["torch"]["version"],
            )
            self.assertNotEqual(rows[0]["first_call_ms"], "")
            self.assertNotEqual(rows[0]["steady_state_ms"], "")
            with (evaluation / "performance_samples.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                samples = list(csv.DictReader(stream))
            self.assertEqual(len(samples), 4)
            self.assertEqual(
                {row["implementation_role"] for row in samples},
                {"reference", "candidate"},
            )
            self.assertEqual({row["inner_iterations"] for row in samples}, {"2"})
            self.assertEqual({row["effective_timer"] for row in samples}, {"wall_clock"})
            for sample in samples:
                reference_dtypes = json.loads(sample["reference_dtype"])
                candidate_dtypes = json.loads(sample["candidate_dtype"])
                reference_shapes = json.loads(sample["reference_shape"])
                candidate_shapes = json.loads(sample["candidate_shape"])
                self.assertEqual(reference_dtypes, candidate_dtypes)
                self.assertEqual(reference_shapes, candidate_shapes)
                self.assertTrue(reference_dtypes)
                self.assertEqual(set(reference_dtypes.values()), {"float64"})
                self.assertEqual(
                    set(reference_dtypes), set(reference_shapes)
                )
            events = [
                json.loads(line)["event"]
                for line in (evaluation / "logs" / "worker.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            self.assertLess(events.index("WARMUP_STARTED"), events.index("WARMUP_FINISHED"))
            self.assertLess(events.index("WARMUP_FINISHED"), events.index("SAMPLING_STARTED"))
            self.assertLess(events.index("SAMPLING_STARTED"), events.index("SAMPLING_FINISHED"))
            self.assertIn("run_id:", stdout)

    def test_correctness_failure_dominates_and_writes_no_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "results"
            code, _, _ = self.run_candidate(output, NUMERIC)
            self.assertEqual(code, 1)
            evaluation = self.evaluation(output, NUMERIC)
            with (evaluation / "performance_samples.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                self.assertEqual(list(csv.DictReader(stream)), [])
            with (evaluation / "results.csv").open(newline="", encoding="utf-8") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["performance_status"], "skipped")
            self.assertEqual(row["skip_reason"], "correctness_gate_failed")

    def test_controller_discards_samples_attached_to_failed_correctness(self):
        class MaliciousController:
            def __init__(self, **kwargs):
                pass

            def run(self, job, **kwargs):
                measurement = {
                    "role": "candidate",
                    "selection": {
                        "requested_timer": "wall_clock",
                        "effective_timer": "wall_clock",
                        "fallback_reason": None,
                    },
                    "first_call_ms": 1.0,
                    "warmup_ms": 1.0,
                    "graph_capture_ms": 0.0,
                    "steady_state_ms": 1.0,
                    "samples": [
                        {
                            "sample_index": 0,
                            "inner_iterations": 1,
                            "elapsed_ms": 1.0,
                            "per_call_ms": 1.0,
                        }
                    ],
                    "statistics": {"median_ms": 1.0},
                }
                return WorkerResponse(
                    identity=job.identity,
                    result_id=job.result_id,
                    outcome=WorkerOutcome.SUCCESS,
                    stage=WorkerStage.COMPLETE,
                    worker_pid=1,
                    started_at_utc="2026-07-16T12:00:00Z",
                    finished_at_utc="2026-07-16T12:00:01Z",
                    elapsed_s=1.0,
                    stage_elapsed_s={},
                    result_payload={
                        "status": "fail",
                        "case_hash": "hash",
                        "input_summary": {},
                        "comparison": {"metrics": {"mismatch_count": 1}},
                        "output_contracts": {},
                        "performance": {
                            "status": "pass",
                            "measurements": {
                                "reference": {**measurement, "role": "reference"},
                                "candidate": measurement,
                            },
                            "cost": {},
                        },
                    },
                )

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "results"
            with patch("benchmark_engine.engine.WorkerController", MaliciousController):
                code, _, _ = self.run_candidate(output, PASS, "--samples", "1")
            self.assertEqual(code, 1)
            evaluation = self.evaluation(output, PASS)
            with (evaluation / "performance_samples.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                self.assertEqual(list(csv.DictReader(stream)), [])

    def test_completed_performance_failure_resume_keeps_exit_one_without_worker(self):
        class FailureController:
            calls = 0

            def __init__(self, **kwargs):
                pass

            def run(self, job, **kwargs):
                type(self).calls += 1
                return WorkerResponse(
                    identity=job.identity,
                    result_id=job.result_id,
                    outcome=WorkerOutcome.ERROR,
                    stage=WorkerStage.SAMPLING,
                    worker_pid=1,
                    started_at_utc="2026-07-16T12:00:00Z",
                    finished_at_utc="2026-07-16T12:00:01Z",
                    elapsed_s=1.0,
                    stage_elapsed_s={"sampling": 1.0},
                    error_type="RuntimeError",
                    error_message="sampling failed",
                    result_payload={
                        "status": "pass",
                        "case_hash": "hash",
                        "input_summary": {},
                        "comparison": {"metrics": {}},
                        "output_contracts": {},
                        "performance": {"status": "error", "reason": "sampling failed"},
                    },
                )

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "results"
            with patch("benchmark_engine.engine.WorkerController", FailureController):
                code, stdout, _ = self.run_candidate(output, PASS, "--samples", "1")
            self.assertEqual(code, 1)
            evaluation = self.evaluation(output, PASS)
            with (evaluation / "results.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["performance_status"], "error")
            self.assertEqual(row["status"], "error")
            run_id = next(
                line.split(":", 1)[1].strip()
                for line in stdout.splitlines()
                if line.startswith("run_id:")
            )
            calls = FailureController.calls

            class MustNotRun:
                def __init__(self, **kwargs):
                    pass

                def run(self, *args, **kwargs):
                    raise AssertionError("completed result was repeated")

            with patch("benchmark_engine.engine.WorkerController", MustNotRun):
                code, _, stderr = self.invoke(
                    "run", "--resume", run_id, "--output-root", str(output)
                )
            self.assertEqual((code, stderr), (1, ""))
            self.assertEqual(FailureController.calls, calls)

    def test_hard_sampling_timeout_without_payload_is_durable_and_resumable(self):
        class HardTimeoutController:
            calls = 0

            def __init__(self, **kwargs):
                pass

            def run(self, job, **kwargs):
                type(self).calls += 1
                return WorkerResponse(
                    identity=job.identity,
                    result_id=job.result_id,
                    outcome=WorkerOutcome.TIMEOUT,
                    stage=WorkerStage.SAMPLING,
                    worker_pid=1,
                    started_at_utc="2026-07-16T12:00:00Z",
                    finished_at_utc="2026-07-16T12:00:01Z",
                    elapsed_s=1.0,
                    stage_elapsed_s={"sampling": 1.0},
                    error_type="StageTimeout",
                    error_message="sampling stage exceeded 1s",
                    result_payload=None,
                )

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "results"
            with patch("benchmark_engine.engine.WorkerController", HardTimeoutController):
                code, stdout, stderr = self.run_candidate(output, PASS, "--samples", "1")
            self.assertEqual((code, stderr), (1, ""))
            evaluation = self.evaluation(output, PASS)
            with (evaluation / "results.csv").open(newline="", encoding="utf-8") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["correctness_status"], "passed")
            self.assertEqual(row["performance_status"], "timeout")
            self.assertEqual(row["status"], "timeout")
            self.assertEqual(row["error_type"], "StageTimeout")
            with (evaluation / "performance_samples.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                self.assertEqual(list(csv.DictReader(stream)), [])
            run_id = next(
                line.split(":", 1)[1].strip()
                for line in stdout.splitlines()
                if line.startswith("run_id:")
            )
            calls = HardTimeoutController.calls

            class MustNotRun:
                def __init__(self, **kwargs):
                    pass

                def run(self, *args, **kwargs):
                    raise AssertionError("completed hard timeout was repeated")

            with patch("benchmark_engine.engine.WorkerController", MustNotRun):
                code, _, stderr = self.invoke(
                    "run", "--resume", run_id, "--output-root", str(output)
                )
            self.assertEqual((code, stderr), (1, ""))
            self.assertEqual(HardTimeoutController.calls, calls)

    def test_hard_warmup_crash_without_payload_is_a_performance_failure(self):
        class HardCrashController:
            def __init__(self, **kwargs):
                pass

            def run(self, job, **kwargs):
                return WorkerResponse(
                    identity=job.identity,
                    result_id=job.result_id,
                    outcome=WorkerOutcome.CRASHED,
                    stage=WorkerStage.WARMUP,
                    worker_pid=1,
                    started_at_utc="2026-07-16T12:00:00Z",
                    finished_at_utc="2026-07-16T12:00:01Z",
                    elapsed_s=1.0,
                    stage_elapsed_s={"warmup": 1.0},
                    error_type="WorkerCrash",
                    error_message="worker exited during warmup",
                    exit_code=-11,
                )

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "results"
            with patch("benchmark_engine.engine.WorkerController", HardCrashController):
                code, _, stderr = self.run_candidate(output, PASS, "--samples", "1")
            self.assertEqual((code, stderr), (1, ""))
            evaluation = self.evaluation(output, PASS)
            with (evaluation / "results.csv").open(newline="", encoding="utf-8") as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["correctness_status"], "passed")
            self.assertEqual(row["performance_status"], "crashed")
            self.assertEqual(row["status"], "crashed")

    def test_hard_sampling_oom_without_payload_preserves_oom_status(self):
        class HardOomController:
            def __init__(self, **kwargs):
                pass

            def run(self, job, **kwargs):
                return WorkerResponse(
                    identity=job.identity,
                    result_id=job.result_id,
                    outcome=WorkerOutcome.OOM,
                    stage=WorkerStage.SAMPLING,
                    worker_pid=1,
                    started_at_utc="2026-07-16T12:00:00Z",
                    finished_at_utc="2026-07-16T12:00:01Z",
                    elapsed_s=1.0,
                    stage_elapsed_s={"sampling": 1.0},
                    error_type="MemoryError",
                    error_message="sampling exhausted device memory",
                )

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "results"
            with patch("benchmark_engine.engine.WorkerController", HardOomController):
                code, _, stderr = self.run_candidate(output, PASS, "--samples", "1")
            self.assertEqual((code, stderr), (1, ""))
            evaluation = self.evaluation(output, PASS)
            with (evaluation / "results.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["correctness_status"], "passed")
            self.assertEqual(row["performance_status"], "oom")
            self.assertEqual(row["status"], "oom")


if __name__ == "__main__":
    unittest.main()
