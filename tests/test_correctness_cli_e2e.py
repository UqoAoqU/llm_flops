from __future__ import annotations

import csv
import io
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

from benchmark_engine.cli import main


ROOT = Path(__file__).resolve().parents[1]
OPERATOR = "example_cpu_add"
PASS = "quickstart__20260716T120000Z__4279e756"
NUMERIC = "numeric_bad__20260716T120100Z__c1ac82e6"
EXCEPTION = "exception_demo__20260716T120200Z__1bd9c56f"
TIMEOUT = "timeout_demo__20260716T120300Z__60c187ff"


class CorrectnessCliE2ETests(unittest.TestCase):
    def invoke(self, *args: str):
        output = io.StringIO()
        errors = io.StringIO()
        with redirect_stdout(output), redirect_stderr(errors):
            code = main(args, repository_root=ROOT)
        return code, output.getvalue(), errors.getvalue()

    def run_candidate(self, output: Path, candidate: str, *extra: str):
        return self.invoke(
            "run",
            "--mode",
            "correctness",
            "--operator",
            OPERATOR,
            "--candidate",
            candidate,
            "--output-root",
            str(output),
            *extra,
        )

    def rows(self, output: Path):
        result = next(output.glob(f"{OPERATOR}/*/*/results.csv"))
        with result.open(newline="", encoding="utf-8") as stream:
            return list(csv.DictReader(stream)), result.parent

    def test_correct_candidate_runs_real_workers_and_writes_mirrored_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "results"
            code, stdout, stderr = self.run_candidate(output, PASS)
            self.assertEqual((code, stderr), (0, ""))
            rows, evaluation = self.rows(output)
            self.assertEqual(len(rows), 2)
            self.assertEqual({row["correctness_status"] for row in rows}, {"passed"})
            self.assertEqual({row["performance_status"] for row in rows}, {"skipped"})
            self.assertEqual({float(row["rel_l2"]) for row in rows}, {0.0})
            with (evaluation / "correctness_outputs.csv").open(
                newline="", encoding="utf-8"
            ) as stream:
                outputs = list(csv.DictReader(stream))
            self.assertEqual(len(outputs), 5)
            self.assertEqual({row["reference_dtype"] for row in outputs}, {"float64"})
            self.assertEqual({row["candidate_dtype"] for row in outputs}, {"float64"})
            self.assertEqual({float(row["rel_l2"]) for row in outputs}, {0.0})
            self.assertIn("run_id:", stdout)

    def test_numeric_exception_and_hard_timeout_are_gate_failures(self):
        for candidate, extra in (
            (NUMERIC, ("--case", "tiny")),
            (EXCEPTION, ("--case", "tiny")),
            (TIMEOUT, ("--case", "tiny", "--timeout-s", "0.2")),
        ):
            with self.subTest(candidate=candidate), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / "results"
                code, _, _ = self.run_candidate(output, candidate, *extra)
                self.assertEqual(code, 1)
                rows, evaluation = self.rows(output)
                self.assertNotEqual(rows[0]["correctness_status"], "passed")
                self.assertEqual(rows[0]["performance_status"], "skipped")
                self.assertTrue((evaluation / "diagnostics").is_dir())
                if candidate == EXCEPTION:
                    self.assertEqual(rows[0]["error_type"], "RuntimeError")
                    self.assertIn(
                        "intentional quick-start candidate exception",
                        rows[0]["error_message"],
                    )
                if candidate == NUMERIC:
                    expected_rel_l2 = 0.20751149978670042
                    self.assertAlmostEqual(
                        float(rows[0]["rel_l2"]), expected_rel_l2, places=15
                    )
                    with (evaluation / "correctness_outputs.csv").open(
                        newline="", encoding="utf-8"
                    ) as stream:
                        output_rows = list(csv.DictReader(stream))
                    self.assertEqual(len(output_rows), 1)
                    self.assertAlmostEqual(
                        float(output_rows[0]["rel_l2"]),
                        expected_rel_l2,
                        places=15,
                    )
                    self.assertEqual(
                        output_rows[0]["rel_l2"], rows[0]["rel_l2"]
                    )

    def test_default_continues_and_fail_fast_stops_after_first_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "all"
            code, _, _ = self.run_candidate(output, NUMERIC)
            self.assertEqual(code, 1)
            self.assertEqual(len(self.rows(output)[0]), 2)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "fast"
            code, _, _ = self.run_candidate(output, NUMERIC, "--fail-fast")
            self.assertEqual(code, 1)
            self.assertEqual(len(self.rows(output)[0]), 1)


if __name__ == "__main__":
    unittest.main()
