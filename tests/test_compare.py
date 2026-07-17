import json
import tempfile
import unittest
from pathlib import Path

from benchmark_engine.reporting.artifact_writer import RUN_INDEX_SCHEMA
from benchmark_engine.reporting.compare import CompareCompatibilityError, compare_artifacts
from benchmark_engine.reporting.csv_writer import AtomicCsvTable, RESULTS_SCHEMA
from tests.reporting_fixtures import results_row


class CompareTests(unittest.TestCase):
    def write(self, root, evaluation, run, candidate, median):
        relative = f"test_operator/{candidate}/{evaluation}"
        path = root / relative
        row = results_row()
        row.update(run_id=run, evaluation_id=evaluation, candidate_id=candidate,
                   mode="performance", performance_status="passed",
                   requested_timer="wall_clock", effective_timer="wall_clock",
                   reference_effective_timer="wall_clock", candidate_median_ms=median,
                   performance_formal=True, ranking_eligible=True,
                   performance_gate_status="passed", performance_gate_reasons="[]",
                   perf_on_correctness_fail=False, gate_unsupported_policy="fail")
        AtomicCsvTable(path / "results.csv", RESULTS_SCHEMA).append(row)
        AtomicCsvTable(root / "run_index.csv", RUN_INDEX_SCHEMA).append({
            "run_id": run, "operator_id": "test_operator", "candidate_id": candidate,
            "evaluation_id": evaluation, "relative_path": relative})

    def test_evaluation_and_run_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write(root, "base", "run_base", "base", 2.0)
            self.write(root, "now", "run_now", "now", 1.0)
            result = json.loads(compare_artifacts(root, "now", "base"))
            self.assertEqual(result["cases"][0]["speedup"], 2.0)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write(root, "base", "run_base", "same", 2.0)
            self.write(root, "now", "run_now", "same", 1.0)
            # Run comparison uses mirrored run_index paths and never creates a run dir.
            self.assertIn("cases", json.loads(compare_artifacts(root, "run_now", "run_base", run=True)))
            self.assertFalse((root / "run_now").exists())

    def test_incompatible_environment_refuses_speedup(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.write(root, "base", "run_base", "base", 2.0)
            self.write(root, "now", "run_now", "now", 1.0)
            table = AtomicCsvTable(root / "test_operator/now/now/results.csv", RESULTS_SCHEMA)
            row = dict(table.read_rows()[0])
            # Rewrite a typed row through a separate artifact to avoid mutating tables.
            row["environment_fingerprint"] = "different"
            other = root / "other/results.csv"
            typed = {column.name: (None if row[column.name] == "" else
                    int(row[column.name]) if column.kind == "integer" else
                    float(row[column.name]) if column.kind == "number" else
                    row[column.name] == "true" if column.kind == "boolean" else row[column.name])
                    for column in RESULTS_SCHEMA.columns}
            AtomicCsvTable(other, RESULTS_SCHEMA).append(typed)
            with self.assertRaises(CompareCompatibilityError):
                compare_artifacts(root, str(other.parent), "base")


if __name__ == "__main__": unittest.main()
