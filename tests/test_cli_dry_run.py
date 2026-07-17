import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

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

    def test_no_match_is_usage_error_and_performance_is_implemented(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            self.repository(root)
            code, _, stderr = self.call(root, "run", "--suite", "smoke", "--dry-run", "--case", "missing")
            self.assertEqual(code, 2)
            self.assertIn("matched no cases", stderr)
            code, stdout, stderr = self.call(
                root,
                "run",
                "--suite",
                "smoke",
                "--mode",
                "performance",
                "--dry-run",
            )
            self.assertEqual(code, 0, stderr)
            payload = json.loads(stdout)
            self.assertEqual({job["mode"] for job in payload["jobs"]}, {"performance"})

    def test_actual_run_preserves_suite_mode_unless_cli_overrides_it(self):
        outcome = MagicMock(run_id="run_0123456789abcdef", passed=0, failed=0,
                            infrastructure_failures=0, evaluation_paths=(), exit_code=0)
        plan, snapshot, environment = MagicMock(), MagicMock(), {}
        invocations = (
            (("--suite", "deepseek_v4_prefill"), None),
            (("--suite", "deepseek_v4_decode"), None),
            (("--suite", "smoke", "--mode", "correctness"), "correctness"),
        )
        for arguments, expected in invocations:
            with self.subTest(arguments=arguments), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                self.repository(root)
                with patch("benchmark_engine.cli.build_execution_plan",
                           return_value=(plan, snapshot, environment)) as build, \
                     patch("benchmark_engine.cli.execute_plan", return_value=outcome):
                    code, _, stderr = self.call(root, "run", *arguments)
                self.assertEqual((code, stderr), (0, ""))
                self.assertEqual(build.call_args.kwargs["mode"], expected)


if __name__ == "__main__":
    unittest.main()
