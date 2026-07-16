from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from benchmark_engine.cli import main
from benchmark_engine.reporting.summary import summarize_evaluation
from tests.test_correctness_cli_e2e import NUMERIC, OPERATOR, ROOT


class SummaryTests(unittest.TestCase):
    def test_summary_is_read_only_and_contains_reproduction(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "results"
            code = main(
                (
                    "run", "--mode", "correctness", "--operator", OPERATOR,
                    "--candidate", NUMERIC, "--case", "tiny",
                    "--output-root", str(output),
                ),
                repository_root=ROOT,
            )
            self.assertEqual(code, 1)
            evaluation = next(
                path for path in output.glob(f"{OPERATOR}/*/*") if path.is_dir()
            )
            before = {path: path.stat().st_mtime_ns for path in evaluation.rglob("*") if path.is_file()}
            summary = summarize_evaluation(evaluation)
            csv_summary = summarize_evaluation(evaluation / "results.csv")
            after = {path: path.stat().st_mtime_ns for path in evaluation.rglob("*") if path.is_file()}
            self.assertEqual(before, after)
            self.assertEqual(summary, csv_summary)
            self.assertIn("Reproduce:", summary)
            self.assertIn("performance_not_implemented", summary)

    def test_missing_and_bad_artifacts_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaises(ValueError):
                summarize_evaluation(root / "missing.csv")
            bad = root / "results.csv"
            bad.write_text("not,the,schema\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                summarize_evaluation(bad)


if __name__ == "__main__":
    unittest.main()
