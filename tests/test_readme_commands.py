from __future__ import annotations

import unittest

from benchmark_engine.cli import build_parser


class ReadmeCommandTests(unittest.TestCase):
    def test_phase7_quick_start_commands_parse(self):
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
                "summarize", "--operator", "example_cpu_add", "--candidate",
                "quickstart__20260716T120000Z__4279e756", "--evaluation",
                "20260716T120000Z__abcdef123456__0123456789ab",
            ),
        )
        for command in commands:
            with self.subTest(command=command):
                parser.parse_args(command)


if __name__ == "__main__":
    unittest.main()
