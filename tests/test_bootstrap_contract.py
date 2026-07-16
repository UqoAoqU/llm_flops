import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class BootstrapContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.script = (ROOT / "bootstrap.sh").read_text()

    def test_is_strict_and_repository_relative(self):
        self.assertIn("set -euo pipefail", self.script)
        self.assertIn("BASH_SOURCE[0]", self.script)
        self.assertIn('unset PYTHONPATH', self.script)

    def test_builds_an_ignored_local_runtime(self):
        self.assertIn(".runtime/venv", self.script)
        self.assertIn(".runtime/logs/bootstrap.log", self.script)
        self.assertIn("UV_CACHE_DIR", self.script)
        self.assertIn(".runtime/", (ROOT / ".gitignore").read_text().splitlines())

    def test_installs_the_immutable_sglang_revision(self):
        lock = json.loads((ROOT / "requirements/benchmark-lock.json").read_text())
        source = lock["source"]["sglang"]
        self.assertEqual(
            source["commit"], "19593359971ebc3582a74f000bf285488d993362"
        )
        self.assertEqual(source["url"], "https://github.com/sgl-project/sglang.git")
        self.assertEqual(source["subdirectory"], "python")
        self.assertIn("source['commit']", self.script)
        self.assertIn("source['subdirectory']", self.script)

    def test_never_reuses_reference_environment_or_external_source(self):
        self.assertNotIn(".venv-sglang0515", self.script)
        self.assertNotIn("/home/claude-lsh", self.script)
        self.assertNotIn("/mnt/", self.script)

    def test_validates_before_marking_install_complete(self):
        validation = '"$VENV/bin/python" -m benchmark_environment --check'
        self.assertIn(validation, self.script)
        marker_write = "printf '%s\\n' \"$LOCK_HASH\" > \"$MARKER\""
        self.assertIn(marker_write, self.script)
        self.assertLess(self.script.rindex(validation), self.script.index(marker_write))


if __name__ == "__main__":
    unittest.main()
