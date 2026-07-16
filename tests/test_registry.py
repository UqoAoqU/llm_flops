import tempfile
import unittest
from dataclasses import FrozenInstanceError
from pathlib import Path

from benchmark_engine.registry import FilesystemRegistry

from tests.registry_fixtures import (
    OPERATOR_ID,
    write_candidate,
    write_reference,
    write_registry,
)


class FilesystemRegistryTest(unittest.TestCase):
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

    def test_hash_mismatch_is_stable_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            write_reference(root)
            candidate, candidate_id = write_candidate(
                root,
                "mismatch",
                "20260716T081500Z",
                with_manifest=True,
            )
            wrong_id = candidate_id[:-8] + "00000000"
            candidate.rename(candidate.with_name(wrong_id))
            manifest = candidate.with_name(wrong_id) / "candidate.yaml"
            manifest.write_text(manifest.read_text().replace(candidate_id, wrong_id))
            issues = FilesystemRegistry(root).discover().issues
            self.assertIn("candidate.hash_mismatch", {issue.code for issue in issues})

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
