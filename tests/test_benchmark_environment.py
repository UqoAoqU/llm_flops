import json
import tempfile
import unittest
from pathlib import Path

from benchmark_environment import (
    environment_fingerprint,
    load_lock,
    validate_environment,
)


class BenchmarkEnvironmentTest(unittest.TestCase):
    def setUp(self):
        self.lock = {
            "schema_version": 1,
            "python": "3.12",
            "cuda": "13.0",
            "gpu": {"capability": [10, 0], "name_contains": "B200"},
            "packages": {"torch": {"version": "2.11.0", "module": "torch"}},
            "required_symbols": ["torch.mm"],
        }
        self.observed = {
            "python": "3.12.3",
            "cuda": "13.0",
            "gpu": {"available": True, "capability": [10, 0], "name": "NVIDIA B200"},
            "packages": {"torch": {"version": "2.11.0", "module": "torch"}},
            "symbols": {"torch.mm": True},
            "import_paths": {"torch": "/tmp/one/torch/__init__.py"},
        }

    def test_load_lock_rejects_unknown_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lock.json"
            path.write_text(json.dumps({"schema_version": 2}))
            with self.assertRaisesRegex(ValueError, "schema_version"):
                load_lock(path)

    def test_matching_environment_is_valid(self):
        self.assertEqual(validate_environment(self.lock, self.observed), [])

    def test_validation_reports_every_mismatch(self):
        observed = json.loads(json.dumps(self.observed))
        observed["python"] = "3.11.9"
        observed["cuda"] = "12.8"
        observed["gpu"]["capability"] = [9, 0]
        observed["gpu"]["name"] = "NVIDIA H100"
        observed["packages"]["torch"]["version"] = "2.10.0"
        observed["symbols"]["torch.mm"] = False
        errors = validate_environment(self.lock, observed)
        self.assertEqual(len(errors), 6)
        self.assertTrue(any("Python" in error for error in errors))
        self.assertTrue(any("torch" in error for error in errors))
        self.assertTrue(any("torch.mm" in error for error in errors))

    def test_fingerprint_is_stable_and_path_independent(self):
        first = environment_fingerprint(self.observed)
        observed = json.loads(json.dumps(self.observed))
        observed["import_paths"]["torch"] = "/another/location/torch/__init__.py"
        self.assertEqual(environment_fingerprint(observed), first)
        self.assertRegex(first, r"^[0-9a-f]{12}$")


if __name__ == "__main__":
    unittest.main()
