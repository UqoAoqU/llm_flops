import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import benchmark_environment
from benchmark_engine.cli import main
from tests.registry_fixtures import OPERATOR_ID, write_candidate, write_reference


SUITE = """schema_version: 1
suite_id: smoke
operators:
  include: ['*']
  exclude: []
cases:
  tags: [smoke]
mode: correctness
correctness:
  seeds: [0, 1]
performance:
  samples: 3
  inner_iterations: 1
"""


class CliDryRunTest(unittest.TestCase):
    def call(self, root: Path, *arguments: str):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = main(arguments, repository_root=root)
        return code, stdout.getvalue(), stderr.getvalue()

    def repository(self, root: Path) -> str:
        write_reference(root)
        marker = root / "candidate-imported"
        _, candidate_id = write_candidate(root, "task", "20260716T081500Z", with_manifest=False, import_marker=marker)
        (root / "suites").mkdir()
        (root / "suites" / "smoke.yaml").write_text(SUITE, encoding="utf-8")
        (root / "requirements").mkdir()
        (root / "requirements" / "benchmark-lock.json").write_text('{"schema_version": 1}', encoding="utf-8")
        return candidate_id

    def test_dry_run_is_json_and_has_no_candidate_cuda_or_artifact_side_effect(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            candidate_id = self.repository(root)
            torch_before = "torch" in sys.modules
            with patch.object(benchmark_environment, "collect_environment", side_effect=AssertionError("collector called")):
                code, stdout, stderr = self.call(root, "run", "--suite", "smoke", "--dry-run", "--operator", OPERATOR_ID, "--candidate", candidate_id)
            self.assertEqual(code, 0, stderr)
            payload = json.loads(stdout)
            self.assertEqual(payload["fingerprint_kind"], "provisional/dry-run")
            self.assertEqual(len(payload["jobs"]), 2)
            self.assertFalse((root / "candidate-imported").exists())
            self.assertFalse((root / "results").exists())
            self.assertEqual("torch" in sys.modules, torch_before)

    def test_no_match_and_execution_unavailable_exit_two(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.repository(root)
            code, _, stderr = self.call(root, "run", "--suite", "smoke", "--dry-run", "--case", "missing")
            self.assertEqual(code, 2)
            self.assertIn("matched no cases", stderr)
            code, _, stderr = self.call(root, "run", "--suite", "smoke")
            self.assertEqual(code, 2)
            self.assertIn("execution not available", stderr)


if __name__ == "__main__":
    unittest.main()
