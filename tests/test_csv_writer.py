from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from benchmark_engine.reporting import (
    AtomicCsvTable,
    CsvColumn,
    CsvConflictError,
    CsvContractError,
    CsvSchema,
    HISTORY_SCHEMA,
    PERFORMANCE_SAMPLES_SCHEMA,
    RESULTS_SCHEMA,
)
from tests.reporting_fixtures import results_row


class AtomicCsvTableTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.schema = CsvSchema(
            "data.csv",
            (
                CsvColumn("schema_version", "integer", False),
                CsvColumn("row_id", "string", False),
                CsvColumn("message", "string", True),
                CsvColumn("count", "integer", True),
            ),
            ("row_id",),
        )

    def test_rfc4180_quoting_crlf_and_stable_column_order(self) -> None:
        table = AtomicCsvTable(self.root / "data.csv", self.schema)
        table.append(
            {"schema_version": 1, "row_id": "one", "message": 'a,"b"\nnext'}
        )
        payload = (self.root / "data.csv").read_bytes()
        self.assertTrue(payload.startswith(b"schema_version,row_id,message,count\r\n"))
        self.assertIn(b'"a,""b""\nnext"', payload)
        self.assertNotIn(b"\n", payload.replace(b"\r\n", b"").replace(b"\nnext", b""))
        with (self.root / "data.csv").open(
            "r", encoding="utf-8", newline=""
        ) as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(rows[0]["message"], 'a,"b"\nnext')
        self.assertEqual(rows[0]["count"], "")

    def test_append_is_idempotent_and_conflicts_do_not_overwrite(self) -> None:
        table = AtomicCsvTable(self.root / "data.csv", self.schema)
        row = {"schema_version": 1, "row_id": "one", "message": "old"}
        self.assertTrue(table.append(row))
        self.assertFalse(table.append(row))
        before = (self.root / "data.csv").read_bytes()
        with self.assertRaisesRegex(CsvConflictError, "different data"):
            table.append(
                {"schema_version": 1, "row_id": "one", "message": "new"}
            )
        self.assertEqual((self.root / "data.csv").read_bytes(), before)

    def test_batch_deduplicates_and_rejects_unknown_or_wrong_types(self) -> None:
        table = AtomicCsvTable(self.root / "data.csv", self.schema)
        self.assertEqual(
            table.append_many(
                (
                    {"row_id": "one", "count": 1},
                    {"row_id": "two", "count": 2},
                    {"row_id": "one", "count": 1},
                )
            ),
            2,
        )
        self.assertEqual(len(table.read_rows()), 2)
        with self.assertRaises(CsvContractError):
            table.append({"row_id": "three", "count": "3"})
        with self.assertRaises(CsvContractError):
            table.append({"row_id": "three", "unknown": "x"})

    def test_replace_failure_keeps_previous_csv_and_cleans_temporary(self) -> None:
        path = self.root / "data.csv"
        stable = AtomicCsvTable(path, self.schema)
        stable.append({"row_id": "one", "message": "stable"})
        before = path.read_bytes()

        def fail_replace(source: object, target: object) -> None:
            raise OSError("injected CSV replace failure")

        failing = AtomicCsvTable(path, self.schema, replace=fail_replace)
        with self.assertRaisesRegex(OSError, "injected CSV"):
            failing.append({"row_id": "two", "message": "new"})
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(stable.read_rows(), [stable.normalise({"row_id": "one", "message": "stable"})])
        self.assertEqual(list(self.root.glob(".data.csv.*.tmp")), [])

    def test_results_v1_starts_with_schema_and_has_result_primary_key(self) -> None:
        self.assertEqual(RESULTS_SCHEMA.version, 1)
        self.assertEqual(RESULTS_SCHEMA.fieldnames[0], "schema_version")
        self.assertEqual(RESULTS_SCHEMA.primary_key, ("result_id",))
        self.assertEqual(RESULTS_SCHEMA.filename, "results.csv")

    def test_enum_columns_reject_invalid_values_before_creating_tables(self) -> None:
        result = results_row()
        for field in (
            "mode",
            "status",
            "correctness_status",
            "performance_status",
        ):
            with self.subTest(table="results", field=field):
                path = self.root / field / RESULTS_SCHEMA.filename
                invalid = dict(result)
                invalid[field] = "nonsense"
                with self.assertRaises(CsvContractError):
                    AtomicCsvTable(path, RESULTS_SCHEMA).append(invalid)
                self.assertFalse(path.exists())

        performance = {
            "result_id": result["result_id"],
            "implementation_role": "baseline_typo",
            "sample_index": 0,
            "inner_iterations": 1,
            "elapsed_ms": 1.0,
            "per_call_ms": 1.0,
            "order_index": 0,
        }
        performance_path = self.root / "performance" / PERFORMANCE_SAMPLES_SCHEMA.filename
        with self.assertRaises(CsvContractError):
            AtomicCsvTable(performance_path, PERFORMANCE_SAMPLES_SCHEMA).append(
                performance
            )
        self.assertFalse(performance_path.exists())

        history = self._history_row(status="failed")
        history_path = self.root / "history" / HISTORY_SCHEMA.filename
        with self.assertRaises(CsvContractError):
            AtomicCsvTable(history_path, HISTORY_SCHEMA).append(history)
        self.assertFalse(history_path.exists())

    def test_legal_enum_values_are_persisted(self) -> None:
        result = results_row()
        result_table = AtomicCsvTable(self.root / "results.csv", RESULTS_SCHEMA)
        self.assertTrue(result_table.append(result))
        self.assertEqual(result_table.read_rows()[0]["status"], "passed")

        sample = {
            "result_id": result["result_id"],
            "implementation_role": "reference",
            "sample_index": 0,
            "inner_iterations": 1,
            "elapsed_ms": 1.0,
            "per_call_ms": 1.0,
            "order_index": 0,
        }
        sample_table = AtomicCsvTable(
            self.root / "samples" / PERFORMANCE_SAMPLES_SCHEMA.filename,
            PERFORMANCE_SAMPLES_SCHEMA,
        )
        self.assertTrue(sample_table.append(sample))

        history_table = AtomicCsvTable(
            self.root / "candidate" / HISTORY_SCHEMA.filename, HISTORY_SCHEMA
        )
        self.assertTrue(history_table.append(self._history_row(status="complete")))

    def test_reader_rejects_existing_row_with_invalid_enum(self) -> None:
        path = self.root / "malformed" / RESULTS_SCHEMA.filename
        table = AtomicCsvTable(path, RESULTS_SCHEMA)
        row = table.normalise(results_row())
        row["status"] = "nonsense"
        path.parent.mkdir(parents=True)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=RESULTS_SCHEMA.fieldnames,
                lineterminator="\r\n",
            )
            writer.writeheader()
            writer.writerow(row)
        with self.assertRaisesRegex(CsvContractError, "status"):
            table.read_rows()

    @staticmethod
    def _history_row(*, status: str) -> dict[str, object]:
        result = results_row()
        return {
            "evaluation_id": result["evaluation_id"],
            "run_id": result["run_id"],
            "operator_id": result["operator_id"],
            "candidate_id": result["candidate_id"],
            "status": status,
            "completed_at_utc": "2026-07-16T12:11:00Z",
            "relative_path": "test_operator/candidate/evaluation",
            "suite_id": result["suite_id"],
            "mode": result["mode"],
            "reference_source_hash": result["reference_source_hash"],
            "candidate_source_hash": result["candidate_source_hash"],
            "environment_fingerprint": result["environment_fingerprint"],
        }


if __name__ == "__main__":
    unittest.main()
