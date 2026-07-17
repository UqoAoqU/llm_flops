from __future__ import annotations

import unittest

from benchmark_engine.cli import build_parser


class ReadmeCommandTests(unittest.TestCase):
    def test_phase8_quick_start_and_performance_commands_parse(self):
        parser = build_parser()
        commands = (
            ("validate", "--operator", "example_cpu_add"),
            ("list", "--operator", "example_cpu_add"),
            ("run", "--suite", "smoke"),
            (
                "run", "--mode", "correctness", "--operator", "example_cpu_add",
                "--candidate", "numeric_bad__20260716T120100Z__c1ac82e6", "--case", "tiny", "--seed", "1",
            ),
            ("run", "--resume", "run_0123456789abcdef0123456789abcdef"),
            (
                "run", "--mode", "performance", "--operator", "example_cpu_add",
                "--candidate", "quickstart__20260716T120000Z__4279e756",
                "--case", "tiny", "--timer", "wall_clock", "--warmup", "5",
                "--samples", "30", "--inner-iterations", "20",
            ),
            (
                "summarize", "--operator", "example_cpu_add", "--candidate",
                "quickstart__20260716T120000Z__4279e756", "--evaluation",
                "20260716T120000Z__abcdef123456__0123456789ab",
            ),
        )
        for command in commands:
            with self.subTest(command=command):
                parser.parse_args(command)

    def test_performance_guide_is_indexed_and_names_phase8_boundaries(self):
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        guide = (root / "docs" / "performance.md").read_text(encoding="utf-8")
        index = (root / "docs" / "index.md").read_text(encoding="utf-8")
        self.assertIn("performance.md", index)
        for phrase in (
            "CUDA Graph",
            "population standard deviation",
            "Type-7",
            "Phase 9",
            "performance_samples.csv",
        ):
            with self.subTest(phrase=phrase):
                self.assertIn(phrase, guide)


if __name__ == "__main__":
    unittest.main()
