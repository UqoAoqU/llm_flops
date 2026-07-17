import csv
import tempfile
import unittest
from pathlib import Path

from benchmark_engine.reporting.csv_writer import (
    AtomicCsvTable,
    PERFORMANCE_SAMPLES_SCHEMA,
    PERFORMANCE_SAMPLES_SCHEMA_V1,
    RESULTS_SCHEMA,
    RESULTS_SCHEMA_V1,
)


def result_row(result_id="res_old"):
    return {
        "run_id": "run_20260716T120000Z_abcdef123456",
        "evaluation_id": "env_20260716T120000Z_abcdef12",
        "timestamp_utc": "2026-07-16T12:00:00Z",
        "suite_id": "smoke",
        "mode": "performance",
        "result_id": result_id,
        "operator_id": "operator",
        "contract_version": 1,
        "candidate_id": "task__20260716T120000Z__abcdef12",
        "reference_id": "reference",
        "candidate_source_hash": "a" * 64,
        "reference_source_hash": "b" * 64,
        "environment_fingerprint": "c" * 64,
        "case_id": "case",
        "case_hash": "d" * 64,
        "seed": 7,
        "status": "passed",
        "correctness_status": "passed",
        "performance_status": "passed",
        "correctness_pass": True,
        "timer": "cuda_event",
        "candidate_median_ms": 1.25,
    }


def sample_row(result_id="res_old", sample_index=0):
    return {
        "result_id": result_id,
        "implementation_role": "candidate",
        "sample_index": sample_index,
        "inner_iterations": 2,
        "elapsed_ms": 2.5,
        "per_call_ms": 1.25,
        "order_index": sample_index,
    }


class PerformanceCsvTests(unittest.TestCase):
    def test_nonempty_v1_results_read_and_first_append_atomically_upgrade(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.csv"
            AtomicCsvTable(path, RESULTS_SCHEMA_V1).append(result_row())
            table = AtomicCsvTable(path, RESULTS_SCHEMA)
            rows = table.read_rows()
            self.assertEqual(rows[0]["schema_version"], "2")
            self.assertEqual(rows[0]["seed"], "7")
            self.assertEqual(rows[0]["correctness_pass"], "true")
            self.assertEqual(rows[0]["requested_timer"], "cuda_event")
            self.assertEqual(rows[0]["effective_timer"], "cuda_event")
            self.assertEqual(
                rows[0]["timer_fallback_reason"], "not_recorded_in_schema_v1"
            )
            new = result_row("res_new")
            new.update(
                requested_timer="cuda_event",
                effective_timer="cuda_event",
                timer_fallback_reason=None,
            )
            table.append(new)
            with path.open(newline="", encoding="utf-8") as stream:
                stored = list(csv.DictReader(stream))
            self.assertEqual(stored[0]["schema_version"], "2")
            self.assertEqual(len(stored), 2)
            self.assertEqual(stored[0]["candidate_median_ms"], "1.25")

    def test_nonempty_v1_samples_read_and_first_append_preserves_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "performance_samples.csv"
            AtomicCsvTable(path, PERFORMANCE_SAMPLES_SCHEMA_V1).append(sample_row())
            table = AtomicCsvTable(path, PERFORMANCE_SAMPLES_SCHEMA)
            rows = table.read_rows()
            self.assertEqual(rows[0]["schema_version"], "2")
            self.assertEqual(rows[0]["sample_index"], "0")
            self.assertEqual(rows[0]["per_call_ms"], "1.25")
            self.assertEqual(rows[0]["requested_timer"], "legacy_unknown")
            self.assertEqual(
                rows[0]["fallback_reason"], "not_recorded_in_schema_v1"
            )
            new = sample_row("res_new", 1)
            new.update(
                requested_timer="auto",
                effective_timer="cuda_event",
                fallback_reason="GraphCaptureError: dynamic shape",
            )
            table.append(new)
            stored = table.read_rows()
            self.assertEqual(len(stored), 2)
            self.assertEqual(stored[1]["effective_timer"], "cuda_event")

    def test_raw_rows_recompute_summary_without_hidden_samples(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "performance_samples.csv"
            table = AtomicCsvTable(path, PERFORMANCE_SAMPLES_SCHEMA)
            for index, value in enumerate((1.0, 2.0, 3.0)):
                row = sample_row("res", index)
                row.update(
                    elapsed_ms=value * 2,
                    per_call_ms=value,
                    requested_timer="auto",
                    effective_timer="cuda_event",
                    fallback_reason="capture failed",
                )
                table.append(row)
            values = [float(row["per_call_ms"]) for row in table.read_rows()]
            self.assertEqual(values, [1.0, 2.0, 3.0])
            self.assertEqual(sum(values) / len(values), 2.0)


if __name__ == "__main__":
    unittest.main()
