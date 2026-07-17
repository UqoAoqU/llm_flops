import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from benchmark_engine.registry import FilesystemRegistry
from benchmark_engine.registry.validation import (
    ManifestValidationError,
    parse_operator_manifest,
)

from tests.registry_fixtures import (
    OPERATOR_ID,
    write_candidate,
    write_reference,
    write_registry,
)


class FilesystemRegistryTest(unittest.TestCase):
    def test_performance_optional_fields_resolve_defaults_and_explicit_values(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = write_reference(Path(temporary)) / "operator.yaml"
            performance = parse_operator_manifest(manifest).performance
            self.assertEqual(performance.max_cv, 0.1)
            self.assertIsNone(performance.min_speedup)
            self.assertIsNone(performance.max_candidate_median_ms)
            self.assertIsNone(performance.max_memory_bytes)

            manifest.write_text(
                manifest.read_text(encoding="utf-8").replace(
                    "  regression_threshold_pct: 5.0\n",
                    "  regression_threshold_pct: 5.0\n"
                    "  min_speedup: 1.2\n"
                    "  max_candidate_median_ms: 3.5\n"
                    "  max_cv: 0.025\n"
                    "  max_memory_bytes: 4096\n",
                ),
                encoding="utf-8",
            )
            performance = parse_operator_manifest(manifest).performance
            self.assertEqual(performance.min_speedup, 1.2)
            self.assertEqual(performance.max_candidate_median_ms, 3.5)
            self.assertEqual(performance.max_cv, 0.025)
            self.assertEqual(performance.max_memory_bytes, 4096)

    def test_performance_max_cv_explicit_null_disables_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest = write_reference(Path(temporary)) / "operator.yaml"
            manifest.write_text(
                manifest.read_text(encoding="utf-8").replace(
                    "  regression_threshold_pct: 5.0\n",
                    "  regression_threshold_pct: 5.0\n  max_cv: null\n",
                ),
                encoding="utf-8",
            )
            self.assertIsNone(parse_operator_manifest(manifest).performance.max_cv)

    def test_performance_max_cv_rejects_invalid_values(self):
        for value in ("not-a-number", "-0.1", ".nan", ".inf"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as temporary:
                manifest = write_reference(Path(temporary)) / "operator.yaml"
                manifest.write_text(
                    manifest.read_text(encoding="utf-8").replace(
                        "  regression_threshold_pct: 5.0\n",
                        f"  regression_threshold_pct: 5.0\n  max_cv: {value}\n",
                    ),
                    encoding="utf-8",
                )
                with self.assertRaises(ManifestValidationError) as raised:
                    parse_operator_manifest(manifest)
                self.assertEqual(raised.exception.field, "performance.max_cv")

    def test_valid_reference_and_two_candidates_are_immutable_and_sorted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            expected = sorted(write_registry(root))
            registry = FilesystemRegistry(root)
            snapshot = registry.discover()
            self.assertTrue(snapshot.is_valid, snapshot.issues)
            self.assertEqual(list(snapshot.references), [OPERATOR_ID])
            self.assertEqual(
                [item.implementation_id for item in registry.get_candidates(OPERATOR_ID)],
                expected,
            )
            self.assertEqual(
                registry.get_reference(OPERATOR_ID).entrypoint,
                "implementation:operator",
            )
            metadata = registry.get_operator_spec(OPERATOR_ID)
            self.assertEqual(metadata.entrypoint, "spec:SPEC")
            with self.assertRaises(TypeError):
                snapshot.references["other"] = registry.get_reference(OPERATOR_ID)
            with self.assertRaises(FrozenInstanceError):
                snapshot.issues = ()

    def test_discovery_never_imports_candidate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            marker = root / "imported.marker"
            write_reference(root)
            _, candidate_id = write_candidate(
                root,
                "dangerous",
                "20260716T081500Z",
                with_manifest=False,
                import_marker=marker,
            )
            snapshot = FilesystemRegistry(root).discover()
            self.assertTrue(snapshot.is_valid, snapshot.issues)
            self.assertEqual(
                snapshot.candidates[OPERATOR_ID][0].implementation_id, candidate_id
            )
            self.assertFalse(marker.exists())

    def test_orphan_bad_yaml_unknown_schema_and_missing_entrypoint_are_issues(self):
        scenarios = (
            ("orphan", "registry.orphan_candidate"),
            ("bad_yaml", "manifest.yaml"),
            ("unknown_schema", "manifest.schema_version"),
            ("missing_entrypoint", "entrypoint.missing"),
        )
        for scenario, expected_code in scenarios:
            with self.subTest(scenario=scenario), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                if scenario == "orphan":
                    write_candidate(
                        root,
                        "orphan",
                        "20260716T081500Z",
                        with_manifest=False,
                    )
                else:
                    reference = write_reference(root)
                    manifest = reference / "operator.yaml"
                    if scenario == "bad_yaml":
                        manifest.write_text("schema_version: [\n")
                    elif scenario == "unknown_schema":
                        manifest.write_text(
                            manifest.read_text().replace("schema_version: 1", "schema_version: 2")
                        )
                    else:
                        (reference / "implementation.py").unlink()
                snapshot = FilesystemRegistry(root).discover()
                self.assertIn(expected_code, {issue.code for issue in snapshot.issues})
                for issue in snapshot.issues:
                    self.assertTrue(issue.code)
                    self.assertTrue(issue.field)
                    self.assertIsInstance(issue.path, Path)

    def test_unknown_field_and_wrong_type_are_strict(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = write_reference(root)
            manifest = reference / "operator.yaml"
            manifest.write_text(manifest.read_text() + "surprise: true\n")
            snapshot = FilesystemRegistry(root).discover()
            issue = snapshot.issues[0]
            self.assertEqual(issue.code, "manifest.unknown_field")
            self.assertEqual(issue.field, "surprise")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            reference = write_reference(root)
            manifest = reference / "operator.yaml"
            manifest.write_text(
                manifest.read_text().replace("contract_version: 1", "contract_version: one")
            )
            self.assertIn(
                "manifest.type",
                {issue.code for issue in FilesystemRegistry(root).discover().issues},
            )

    def test_candidate_name_is_independent_of_recorded_source_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_reference(root)
            candidate, candidate_id = write_candidate(
                root,
                "mismatch",
                "20260716T081500Z",
                with_manifest=True,
            )
            source_hash = next(
                item.source_hash
                for item in FilesystemRegistry(root).discover().candidates[OPERATOR_ID]
            )
            arbitrary_id = "Human readable candidate"
            renamed = candidate.with_name(arbitrary_id)
            candidate.rename(renamed)
            manifest = renamed / "candidate.yaml"
            manifest.write_text(
                manifest.read_text().replace(candidate_id, arbitrary_id)
            )
            snapshot = FilesystemRegistry(root).discover()
            self.assertTrue(snapshot.is_valid, snapshot.issues)
            implementation = snapshot.candidates[OPERATOR_ID][0]
            self.assertEqual(implementation.implementation_id, arbitrary_id)
            self.assertEqual(implementation.source_hash, source_hash)

    def test_candidate_names_still_reject_case_insensitive_duplicates(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_reference(root)
            parent = root / "operators" / "candidates" / OPERATOR_ID
            first = parent / "Candidate"
            second = parent / "candidate"
            try:
                for path in (first, second):
                    path.mkdir(parents=True, exist_ok=False)
                    (path / "implementation.py").write_text(
                        "def operator(left, right):\n    return left + right\n"
                    )
            except FileExistsError:
                self.skipTest("filesystem is case-insensitive")
            issues = FilesystemRegistry(root).discover().issues
            self.assertIn("registry.case_collision", {issue.code for issue in issues})

    def test_case_insensitive_collisions_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_reference(root, "lowercase_op")
            try:
                write_reference(root, "LOWERCASE_OP")
            except FileExistsError:
                self.skipTest("filesystem is case-insensitive")
            issues = FilesystemRegistry(root).discover().issues
            self.assertIn("registry.case_collision", {issue.code for issue in issues})


if __name__ == "__main__":
    unittest.main()
