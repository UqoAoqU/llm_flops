import csv
import json
import os
import tempfile
import unittest
from pathlib import Path

from deepseek_v4_benchmark import adapter_io_shapes, prefill_adapters

from benchmark_engine.reporting import (
    AtomicCsvTable,
    CompareCompatibilityError,
    LegacyImportConflictError,
    LegacyImportError,
    MODEL_PROJECTION_SCHEMA,
    PERFORMANCE_SAMPLES_SCHEMA,
    RESULTS_SCHEMA,
    compare_artifacts,
    convert_legacy_csv,
    summarize_evaluation,
)
from benchmark_engine.reporting.legacy_import import LEGACY_HEADER
from benchmark_engine.reporting.csv_writer import RESULTS_SCHEMA_V3

from tests.reporting_fixtures import results_row


class LegacyImportTests(unittest.TestCase):
    def write_legacy(self, path: Path, *, mutate=None) -> None:
        rows = []
        for adapter in prefill_adapters("fp8_mxfp8"):
            input_shape, output_shape = adapter_io_shapes(adapter, 1024, 65536)
            row = {
                "phase": "prefill",
                "quant_profile": "fp8_mxfp8",
                "environment_fingerprint": "4286ebdeb11b",
                "m": "1024",
                "context": "65536",
                "operator": adapter.name,
                "backend": adapter.backend,
                "instances": str(adapter.instances),
                "call_ms": "0.100000",
                "model_ms": f"{adapter.instances * 0.1:.6f}",
                "pct": "1.0000",
                "status": "executed",
                "input_shape": input_shape,
                "output_shape": output_shape,
                "error": "",
            }
            rows.append(row)
        if mutate is not None:
            mutate(rows)
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=LEGACY_HEADER, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

    def test_import_is_mirrored_non_rankable_and_content_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first_csv = root / "first.csv"
            second_csv = root / "renamed.csv"
            self.write_legacy(first_csv)
            second_csv.write_bytes(first_csv.read_bytes())
            output = root / "results"

            report = convert_legacy_csv(first_csv, output, "legacy_control")
            self.assertGreater(report.created, 0)
            self.assertEqual(report.reused, 0)
            self.assertFalse((output / "run_index.csv").exists())
            all_projections = []
            for relative in report.directories:
                evaluation = output / relative
                self.assertEqual(relative.split("/")[1], "legacy_control")
                self.assertFalse((evaluation / "evaluation_manifest.json").exists())
                manifest = json.loads((evaluation / "legacy_import_manifest.json").read_text())
                self.assertTrue(manifest["imported_legacy"])
                self.assertIsNone(manifest["availability"]["candidate_source_hash"])
                self.assertEqual(
                    set(manifest["artifacts"]),
                    {"results.csv", "correctness_outputs.csv", "performance_samples.csv",
                     "model_projection.csv", "summary.md"},
                )
                rows = AtomicCsvTable(evaluation / "results.csv", RESULTS_SCHEMA).read_rows()
                self.assertTrue(rows)
                self.assertTrue(all(row["imported_legacy"] == "true" for row in rows))
                self.assertTrue(all(row["candidate_source_hash"] == "" for row in rows))
                self.assertTrue(all(row["candidate_median_ms"] == "" for row in rows))
                self.assertTrue(all(row["legacy_graph_ms"] == "0.1" for row in rows))
                self.assertTrue(all(row["status"] == "skipped" for row in rows))
                self.assertTrue(all(row["performance_status"] == "skipped" for row in rows))
                projections = AtomicCsvTable(
                    evaluation / "model_projection.csv", MODEL_PROJECTION_SCHEMA
                ).read_rows()
                self.assertTrue(projections)
                all_projections.extend(projections)
                for projection in projections:
                    shape = json.loads(projection["legacy_shape"])
                    self.assertTrue(shape["input_shape"])
                    self.assertTrue(shape["output_shape"])
                self.assertFalse(
                    AtomicCsvTable(
                        evaluation / "performance_samples.csv", PERFORMANCE_SAMPLES_SCHEMA
                    ).read_rows()
                )
                self.assertIn("Imported legacy: true", summarize_evaluation(evaluation))
                self.assertIn(
                    "Imported legacy: true", summarize_evaluation(evaluation / "results.csv")
                )
                with self.assertRaisesRegex(CompareCompatibilityError, "non-rankable"):
                    compare_artifacts(output, str(evaluation), str(evaluation))
                marker = evaluation / "legacy_import_manifest.json"
                marker_content = marker.read_bytes()
                marker.unlink()
                try:
                    with self.assertRaisesRegex(CompareCompatibilityError, "non-rankable"):
                        compare_artifacts(output, str(evaluation), str(evaluation))
                    recovered_summary = summarize_evaluation(evaluation / "results.csv")
                    self.assertIn("Imported legacy: true", recovered_summary)
                    self.assertIn(
                        "Correctness/source hashes/raw samples/CV/speedup: unavailable",
                        recovered_summary,
                    )
                finally:
                    marker.write_bytes(marker_content)

            shape_free = [
                json.loads(projection["legacy_shape"])
                for projection in all_projections
                if projection["display_name"] == "C4 Indexer FP8 Quant"
            ]
            self.assertEqual(
                shape_free,
                [{
                    "input_shape": "q=(1024,64,128)",
                    "logical_shape": None,
                    "output_shape": "q_fp8=(1024,64,128)",
                }],
            )

            repeated = convert_legacy_csv(second_csv, output, "legacy_control")
            self.assertEqual(repeated.run_id, report.run_id)
            self.assertEqual(repeated.evaluation_id, report.evaluation_id)
            self.assertEqual(repeated.created, 0)
            self.assertEqual(repeated.reused, report.created)

    def test_conflict_header_backend_shape_and_casefold_are_stable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "legacy.csv"
            self.write_legacy(source)
            output = root / "results"
            report = convert_legacy_csv(source, output, "legacy_control")
            target = output / report.directories[0] / "legacy_import_manifest.json"
            manifest = json.loads(target.read_text())
            manifest["source_csv"]["sha256"] = "0" * 64
            target.write_text(json.dumps(manifest), encoding="utf-8")
            with self.assertRaises(LegacyImportConflictError):
                convert_legacy_csv(source, output, "legacy_control")

            artifact_output = root / "artifact-tamper"
            artifact_report = convert_legacy_csv(source, artifact_output, "candidate")
            artifact = artifact_output / artifact_report.directories[0] / "results.csv"
            artifact.write_bytes(artifact.read_bytes() + b"\n")
            with self.assertRaisesRegex(LegacyImportConflictError, "content hash mismatch"):
                convert_legacy_csv(source, artifact_output, "candidate")

            bad_header = root / "header.csv"
            bad_header.write_text("bad,header\n1,2\n", encoding="utf-8")
            with self.assertRaisesRegex(LegacyImportError, "header is incompatible"):
                convert_legacy_csv(bad_header, root / "header-out", "candidate")

            for field, value, message in (
                ("backend", "wrong", "backend mismatch"),
                ("input_shape", "x=(1,2)", "shape mismatch"),
                ("output_shape", "", "shape mismatch"),
            ):
                path = root / f"bad-{field}.csv"
                self.write_legacy(path, mutate=lambda rows, f=field, v=value: rows[0].__setitem__(f, v))
                with self.assertRaisesRegex(LegacyImportError, message):
                    convert_legacy_csv(path, root / f"out-{field}", "candidate")

            collision_output = root / "collision"
            first_operator = report.directories[0].split("/")[0]
            (collision_output / first_operator / "LegacyControl").mkdir(parents=True)
            with self.assertRaisesRegex(LegacyImportConflictError, "case-insensitively collides"):
                convert_legacy_csv(source, collision_output, "legacycontrol")
            identity_probe = convert_legacy_csv(source, root / "identity-probe", "candidate")
            evaluation_collision = root / "evaluation-collision"
            operator_id = identity_probe.directories[0].split("/")[0]
            (evaluation_collision / operator_id / "candidate" / identity_probe.evaluation_id.upper()).mkdir(parents=True)
            with self.assertRaisesRegex(LegacyImportConflictError, "evaluation_id case-insensitively"):
                convert_legacy_csv(source, evaluation_collision, "candidate")
            with self.assertRaises(ValueError):
                convert_legacy_csv(source, root / "traversal", "../escape")

    def test_unavailable_legacy_row_has_no_latency_or_formal_status(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "legacy.csv"

            def unavailable(rows):
                rows[-1].update(
                    status="unavailable", call_ms="", model_ms="", pct="0.0000",
                    error="legacy backend unavailable",
                )

            self.write_legacy(source, mutate=unavailable)
            report = convert_legacy_csv(source, root / "results", "candidate")
            all_rows = []
            for relative in report.directories:
                evaluation = root / "results" / relative
                all_rows.extend(
                    AtomicCsvTable(evaluation / "results.csv", RESULTS_SCHEMA).read_rows()
                )
            unavailable_rows = [row for row in all_rows if row["error_type"] == "legacy_unavailable"]
            self.assertEqual(len(unavailable_rows), 1)
            row = unavailable_rows[0]
            self.assertEqual(row["legacy_graph_ms"], "")
            self.assertEqual(row["status"], "skipped")
            self.assertEqual(row["performance_status"], "skipped")
            self.assertEqual(row["ranking_eligible"], "false")

    def test_directory_publish_failure_leaves_no_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "legacy.csv"
            self.write_legacy(source)

            def fail_replace(_source, _target):
                raise OSError("injected directory rename failure")

            with self.assertRaisesRegex(OSError, "injected"):
                convert_legacy_csv(
                    source, root / "results", "candidate", replace_directory=fail_replace
                )
            self.assertFalse(list((root / "results").glob("*/*/*")))

    def test_multi_directory_publish_failure_rerun_converges(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "legacy.csv"
            self.write_legacy(source)
            output = root / "results"
            calls = 0

            def fail_second(source_path, target_path):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("injected second directory rename failure")
                os.replace(source_path, target_path)

            with self.assertRaisesRegex(OSError, "second directory"):
                convert_legacy_csv(
                    source, output, "candidate", replace_directory=fail_second
                )
            complete = [
                path for path in output.glob("*/*/*")
                if (path / "legacy_import_manifest.json").is_file()
            ]
            self.assertEqual(len(complete), 1)
            self.assertFalse(list(output.glob("*/*/.*")))
            report = convert_legacy_csv(source, output, "candidate")
            self.assertEqual(report.reused, 1)
            self.assertEqual(report.created + report.reused, len(report.directories))
            self.assertEqual(len(list(output.glob("*/*/*/legacy_import_manifest.json"))), len(report.directories))


class ResultsV4ContractTests(unittest.TestCase):
    def test_nonlegacy_legacy_graph_ms_is_rejected_on_append_and_stored_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            table = AtomicCsvTable(path, RESULTS_SCHEMA)
            normal = results_row()
            with self.assertRaisesRegex(ValueError, "legacy_graph_ms empty"):
                table.append(dict(normal, legacy_graph_ms=0.1))

            stored = table.normalise(normal)
            stored["legacy_graph_ms"] = "0.1"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=RESULTS_SCHEMA.fieldnames)
                writer.writeheader()
                writer.writerow(stored)
            with self.assertRaisesRegex(ValueError, "legacy_graph_ms empty"):
                table.read_rows()

    def test_normal_v3_reads_and_first_append_atomically_upgrades_to_v4(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            normal = results_row()
            old = {
                name: (3 if name == "schema_version" else normal.get(name))
                for name in RESULTS_SCHEMA_V3.fieldnames
            }
            AtomicCsvTable(path, RESULTS_SCHEMA_V3).append(old)
            current = AtomicCsvTable(path, RESULTS_SCHEMA)
            migrated = current.read_rows()
            self.assertEqual(migrated[0]["imported_legacy"], "false")
            self.assertEqual(migrated[0]["legacy_graph_ms"], "")
            current.append(
                results_row("res_33333333333333333333333333333333")
            )
            with path.open("r", encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                rows = list(reader)
            self.assertEqual(reader.fieldnames, list(RESULTS_SCHEMA.fieldnames))
            self.assertTrue(all(row["schema_version"] == "4" for row in rows))

    def test_normal_and_legacy_semantics_apply_on_append_and_read(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "results.csv"
            table = AtomicCsvTable(path, RESULTS_SCHEMA)
            normal = results_row()
            table.append(normal)
            broken = dict(normal, result_id="res_11111111111111111111111111111111", candidate_source_hash="")
            with self.assertRaisesRegex(ValueError, "non-legacy results require provenance"):
                table.append(broken)
            legacy = dict(
                normal,
                result_id="res_22222222222222222222222222222222",
                timestamp_utc="",
                reference_id="",
                candidate_source_hash="",
                reference_source_hash="",
                case_hash="",
                imported_legacy=True,
                correctness_status="skipped",
                performance_status="skipped",
                status="skipped",
                suite_id="legacy_import",
                mode="performance",
                skip_reason="imported_legacy_without_correctness_or_raw_samples",
                timer="legacy_cuda_graph",
                requested_timer="legacy_cuda_graph",
                effective_timer="legacy_cuda_graph",
                legacy_graph_ms=0.1,
                correctness_pass=None,
                performance_formal=False,
                ranking_eligible=False,
                performance_gate_status="skipped",
                performance_gate_reasons='["imported_legacy_non_rankable"]',
                perf_on_correctness_fail=False,
                gate_unsupported_policy="fail",
                candidate_median_ms=None,
            )
            table.append_many([legacy])
            for field, value in (
                ("candidate_median_ms", 0.1),
                ("import_ms", 1.0),
                ("tflops", 2.0),
                ("peak_memory_bytes", 4096),
                ("error_type", "arbitrary"),
            ):
                with self.subTest(forbidden_legacy_field=field):
                    with self.assertRaises(ValueError):
                        table.normalise(dict(legacy, **{field: value}))
            with self.assertRaises(ValueError):
                table.normalise(dict(legacy, status="passed"))
            with path.open("r", encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                stored = list(reader)
                fieldnames = reader.fieldnames
            stored[-1]["imported_legacy"] = "false"
            with path.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(stored)
            with self.assertRaises(ValueError):
                table.read_rows()


if __name__ == "__main__":
    unittest.main()
