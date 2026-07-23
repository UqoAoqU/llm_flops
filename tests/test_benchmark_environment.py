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
            "schema_version": 2,
            "python": "3.11",
            "accelerator": {"backend": "rocm", "runtime_prefix": "7.0"},
            "gpu": {
                "arch_prefix": "gfx942",
                "name_contains": "AMD Radeon Graphics",
                "count": 8,
            },
            "packages": {
                "torch": {"version": "2.10.0+rocm7.0", "module": "torch"}
            },
            "source": {
                "sglang": {
                    "root_env": "SGLANG_ROOT",
                    "commit": "85fd90072d1a9f2432842b03588f63b745e524e4",
                    "dirty": True,
                    "diff_sha256": "b" * 64,
                }
            },
            "required_symbols": ["torch.mm"],
        }
        self.observed = {
            "python": "3.11.4",
            "accelerator": {
                "backend": "rocm",
                "runtime": "7.0.51831",
                "arch": "gfx942:sramecc+:xnack-",
            },
            "cuda": None,
            "hip": "7.0.51831",
            "gpu": {
                "available": True,
                "count": 8,
                "capability": [9, 4],
                "name": "AMD Radeon Graphics",
                "arch": "gfx942:sramecc+:xnack-",
            },
            "packages": {
                "torch": {
                    "version": "2.10.0+rocm7.0",
                    "module": "torch",
                }
            },
            "source": {
                "sglang": {
                    "commit": "85fd90072d1a9f2432842b03588f63b745e524e4",
                    "dirty": True,
                    "diff_sha256": "b" * 64,
                    "root_env": "SGLANG_ROOT",
                }
            },
            "symbols": {"torch.mm": True},
            "import_paths": {"torch": "/tmp/one/torch/__init__.py"},
        }

    def test_load_lock_rejects_unknown_schema(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "lock.json"
            path.write_text(json.dumps({"schema_version": 1}))
            with self.assertRaisesRegex(ValueError, "schema_version"):
                load_lock(path)

    def test_matching_environment_is_valid(self):
        self.assertEqual(validate_environment(self.lock, self.observed), [])

    def test_validation_reports_every_mismatch(self):
        observed = json.loads(json.dumps(self.observed))
        observed["python"] = "3.12.0"
        observed["accelerator"]["backend"] = "cuda"
        observed["accelerator"]["runtime"] = "13.0"
        observed["gpu"]["arch"] = "sm100"
        observed["gpu"]["name"] = "NVIDIA B200"
        observed["gpu"]["count"] = 1
        observed["packages"]["torch"]["version"] = "2.11.0"
        observed["symbols"]["torch.mm"] = False
        observed["source"]["sglang"]["commit"] = "deadbeef"
        errors = validate_environment(self.lock, observed)
        self.assertEqual(len(errors), 9)
        self.assertTrue(any("Python" in error for error in errors))
        self.assertTrue(any("backend" in error for error in errors))
        self.assertTrue(any("torch.mm" in error for error in errors))

    def test_source_diff_changes_fingerprint(self):
        first = environment_fingerprint(self.observed)
        observed = json.loads(json.dumps(self.observed))
        observed["source"]["sglang"]["diff_sha256"] = "c" * 64
        self.assertNotEqual(environment_fingerprint(observed), first)

    def test_fingerprint_is_stable_and_path_independent(self):
        first = environment_fingerprint(self.observed)
        observed = json.loads(json.dumps(self.observed))
        observed["import_paths"]["torch"] = "/another/location/torch/__init__.py"
        self.assertEqual(environment_fingerprint(observed), first)
        self.assertRegex(first, r"^[0-9a-f]{12}$")


if __name__ == "__main__":
    unittest.main()
