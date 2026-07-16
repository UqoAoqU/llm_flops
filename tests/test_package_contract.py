import importlib
import subprocess
import sys
import tomllib
import unittest
from pathlib import Path

import benchmark_engine


ROOT = Path(__file__).resolve().parents[1]


class PackageContractTest(unittest.TestCase):
    def test_pyproject_declares_src_layout_and_console_entry(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())
        self.assertEqual(project["project"]["requires-python"], ">=3.12,<3.13")
        self.assertIn("PyYAML==6.0.3", project["project"]["dependencies"])
        self.assertEqual(
            project["project"]["scripts"]["bench"],
            "benchmark_engine.cli:main",
        )
        self.assertEqual(
            project["tool"]["setuptools"]["packages"]["find"]["where"], ["src"]
        )

    def test_public_version_and_package_imports(self):
        project = tomllib.loads((ROOT / "pyproject.toml").read_text())
        self.assertEqual(
            benchmark_engine.__version__,
            project["project"]["version"],
        )
        subpackages = (
            "registry",
            "execution",
            "correctness",
            "performance",
            "reporting",
            "environment",
            "projection",
        )
        for name in subpackages:
            with self.subTest(package=name):
                importlib.import_module(f"benchmark_engine.{name}")

    def test_module_entrypoint_has_only_help_and_version(self):
        help_result = subprocess.run(
            [sys.executable, "-m", "benchmark_engine", "--help"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertNotIn("{list,validate,run}", help_result.stdout)

        version_result = subprocess.run(
            [sys.executable, "-m", "benchmark_engine", "--version"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(version_result.returncode, 0, version_result.stderr)
        self.assertEqual(
            version_result.stdout.strip(),
            f"bench {benchmark_engine.__version__}",
        )


if __name__ == "__main__":
    unittest.main()
