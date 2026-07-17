import unittest
from pathlib import Path

from benchmark_engine.ids import (
    IdentifierError,
    candidate_hash_suffix,
    validate_candidate_id,
    validate_evaluation_id,
    validate_operator_id,
)


class IdentifierTest(unittest.TestCase):
    def test_operator_id_contract(self):
        self.assertEqual(validate_operator_id("cpu_add"), "cpu_add")
        for value in ("ab", "CPU_add", "../cpu_add", "cpu/add", "/cpu_add"):
            with self.subTest(value=value), self.assertRaises(IdentifierError):
                validate_operator_id(value)

    def test_candidate_id_is_any_safe_unique_path_component(self):
        recommended = "task_42__20260716T081500Z__abcdef12"
        self.assertEqual(validate_candidate_id(recommended), recommended)
        self.assertEqual(candidate_hash_suffix(recommended), "abcdef12")
        accepted = (
            "task__20260230T081500Z__abcdef12",
            "task__20260716T081500+08__abcdef12",
            "task__20260716T081500Z__ABCDEF12",
            "task__20260716T081500Z__abcdef1",
            "test_impl",
            "Human readable candidate",
            "候选实现一",
            "draft..2",
        )
        for value in accepted:
            with self.subTest(value=value):
                self.assertEqual(validate_candidate_id(value), value)
        self.assertIsNone(candidate_hash_suffix("test_impl"))
        self.assertIsNone(
            candidate_hash_suffix("task__20260230T081500Z__abcdef12")
        )
        invalid = (
            "",
            "candidate\x00name",
            ".",
            "..",
            "../task__20260716T081500Z__abcdef12",
            "candidate/name",
            "candidate\\name",
            "/absolute-candidate",
        )
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(IdentifierError):
                validate_candidate_id(value)

    def test_evaluation_id_is_format_only_in_phase_two(self):
        value = "20260716T091500Z__01234567__run_42"
        self.assertEqual(validate_evaluation_id(value), value)
        with self.assertRaises(IdentifierError):
            validate_evaluation_id("20260716T171500+08__01234567__run_42")


if __name__ == "__main__":
    unittest.main()
