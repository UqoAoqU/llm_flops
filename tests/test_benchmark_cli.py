import unittest
from pathlib import Path

from benchmark_cli import CommandError, resolve_command


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkCliTest(unittest.TestCase):
    def test_resolves_every_single_operator(self):
        expected = {
            "dsa_indexer": "dsa_indexer.py",
            "dsa_flashmla": "dsa_flashmla.py",
            "dsa_projection": "dsa_projection.py",
            "mla_flashmla": "mla_flashmla.py",
            "moe_deepgemm": "moe_deepgemm.py",
        }
        for name, script in expected.items():
            with self.subTest(name=name):
                self.assertEqual(resolve_command(["op", name], ROOT), [str(ROOT / script)])

    def test_forwards_profiling_arguments_unchanged(self):
        arguments = ["--m", "1024,2048", "--context", "65536"]
        self.assertEqual(
            resolve_command(["prefill", *arguments], ROOT),
            [str(ROOT / "bench_deepseek_v4_prefill.py"), *arguments],
        )
        self.assertEqual(
            resolve_command(["decode", *arguments], ROOT),
            [str(ROOT / "bench_deepseek_v4_decode.py"), *arguments],
        )

    def test_resolves_support_commands(self):
        self.assertEqual(
            resolve_command(["compare"], ROOT),
            [str(ROOT / "print_deepseek_v4_quant_comparison.py")],
        )
        self.assertEqual(
            resolve_command(["check"], ROOT),
            [str(ROOT / "tools/check_environment.py")],
        )
        self.assertEqual(
            resolve_command(["smoke"], ROOT),
            [str(ROOT / "tools/smoke_test.py")],
        )

    def test_unknown_operator_lists_valid_names(self):
        with self.assertRaisesRegex(CommandError, "dsa_indexer.*moe_deepgemm"):
            resolve_command(["op", "unknown"], ROOT)

    def test_missing_command_has_usage(self):
        with self.assertRaisesRegex(CommandError, "Usage"):
            resolve_command([], ROOT)


if __name__ == "__main__":
    unittest.main()
