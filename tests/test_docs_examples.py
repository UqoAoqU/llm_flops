import shutil
import tempfile
import unittest
from pathlib import Path

from benchmark_engine.registry import FilesystemRegistry, compute_source_hash


ROOT = Path(__file__).resolve().parents[1]


class DocumentationExampleTest(unittest.TestCase):
    def test_cpu_examples_are_discoverable_when_copied_to_contract_paths(self):
        with tempfile.TemporaryDirectory() as temporary:
            repository = Path(temporary)
            reference = repository / "operators" / "references" / "minimal_cpu_add"
            shutil.copytree(ROOT / "docs" / "examples" / "minimal-operator", reference)

            source = ROOT / "docs" / "examples" / "minimal-candidate"
            building = repository / "operators" / "candidates" / "minimal_cpu_add" / "building"
            shutil.copytree(source, building)
            source_hash = compute_source_hash(building)
            candidate_id = f"docs_example__20260716T081500Z__{source_hash[:8]}"
            candidate = building.with_name(candidate_id)
            building.rename(candidate)

            snapshot = FilesystemRegistry(repository).discover()
            self.assertTrue(snapshot.is_valid, snapshot.issues)
            self.assertIn("minimal_cpu_add", snapshot.references)
            self.assertEqual(
                snapshot.candidates["minimal_cpu_add"][0].implementation_id,
                candidate_id,
            )


if __name__ == "__main__":
    unittest.main()
