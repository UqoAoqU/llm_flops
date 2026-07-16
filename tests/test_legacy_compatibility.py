import subprocess
import tempfile
import unittest
from pathlib import Path

from benchmark_cli import GPU_COMMANDS, resolve_command


ROOT = Path(__file__).resolve().parents[1]
RUN_SH = ROOT / "run.sh"


class LegacyCompatibilityTest(unittest.TestCase):
    def test_top_level_commands_map_to_their_legacy_python_files(self):
        expected = {
            "prefill": "bench_deepseek_v4_prefill.py",
            "decode": "bench_deepseek_v4_decode.py",
            "compare": "print_deepseek_v4_quant_comparison.py",
            "check": "tools/check_environment.py",
            "smoke": "tools/smoke_test.py",
        }
        for command, script in expected.items():
            with self.subTest(command=command):
                self.assertEqual(
                    resolve_command([command], ROOT),
                    [str(ROOT / script)],
                )

    def test_operator_commands_map_to_their_legacy_python_files(self):
        expected = {
            "dsa_indexer": "dsa_indexer.py",
            "dsa_flashmla": "dsa_flashmla.py",
            "dsa_projection": "dsa_projection.py",
            "mla_flashmla": "mla_flashmla.py",
            "moe_deepgemm": "moe_deepgemm.py",
        }
        for operator, script in expected.items():
            with self.subTest(operator=operator):
                self.assertEqual(
                    resolve_command(["op", operator], ROOT),
                    [str(ROOT / script)],
                )

    def test_arguments_are_forwarded_verbatim_in_original_order(self):
        arguments = [
            "--m",
            "1024,2048",
            "--context=65536",
            "--csv",
            "result path.csv",
        ]
        top_level_scripts = {
            "prefill": "bench_deepseek_v4_prefill.py",
            "decode": "bench_deepseek_v4_decode.py",
            "compare": "print_deepseek_v4_quant_comparison.py",
            "check": "tools/check_environment.py",
            "smoke": "tools/smoke_test.py",
        }
        for command, script in top_level_scripts.items():
            with self.subTest(command=command):
                self.assertEqual(
                    resolve_command([command, *arguments], ROOT),
                    [str(ROOT / script), *arguments],
                )

        for operator, script in {
            "dsa_indexer": "dsa_indexer.py",
            "dsa_flashmla": "dsa_flashmla.py",
            "dsa_projection": "dsa_projection.py",
            "mla_flashmla": "mla_flashmla.py",
            "moe_deepgemm": "moe_deepgemm.py",
        }.items():
            with self.subTest(operator=operator):
                self.assertEqual(
                    resolve_command(["op", operator, *arguments], ROOT),
                    [str(ROOT / script), *arguments],
                )

    def test_gpu_precheck_command_set_is_unchanged(self):
        self.assertEqual(GPU_COMMANDS, {"op", "prefill", "decode", "smoke"})

    def test_run_sh_uses_the_repository_runtime_and_local_caches(self):
        launcher = RUN_SH.read_text()
        required_fragments = (
            'ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"',
            'PYTHON="$ROOT/.runtime/venv/bin/python"',
            "unset PYTHONPATH",
            'export UV_CACHE_DIR="$ROOT/.runtime/cache/uv"',
            'export TORCH_EXTENSIONS_DIR="$ROOT/.runtime/cache/torch_extensions"',
            'export FLASHINFER_WORKSPACE_BASE="$ROOT/.runtime/cache/flashinfer"',
            'export XDG_CACHE_HOME="$ROOT/.runtime/cache/xdg"',
            'exec "$PYTHON" "$ROOT/benchmark_cli.py" "$@"',
        )
        for fragment in required_fragments:
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, launcher)

    def test_run_sh_reports_bootstrap_when_repository_venv_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            launcher = Path(directory) / "run.sh"
            launcher.write_text(RUN_SH.read_text())
            result = subprocess.run(
                ["bash", str(launcher)],
                text=True,
                capture_output=True,
                check=False,
            )
        self.assertEqual(result.returncode, 2)
        self.assertEqual(
            result.stderr.strip(),
            "ERROR: benchmark runtime is missing; run ./bootstrap.sh first",
        )

    def test_invalid_legacy_commands_return_usage_errors_without_gpu_work(self):
        cases = (
            ([], "Usage: run.sh op <operator>"),
            (["unknown"], "Unknown command 'unknown'."),
            (["op"], "Unknown operator '<missing>'."),
            (["op", "unknown"], "Unknown operator 'unknown'."),
        )
        for arguments, expected_error in cases:
            with self.subTest(arguments=arguments):
                result = subprocess.run(
                    [str(RUN_SH), *arguments],
                    cwd=ROOT,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 2)
                self.assertIn(expected_error, result.stderr)
                self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
