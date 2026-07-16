import tempfile
import unittest
from pathlib import Path

from benchmark_engine.ids import (
    IdentifierError,
    candidate_result_path,
    candidate_source_path,
    evaluation_result_path,
)


CANDIDATE = "task__20260716T081500Z__abcdef12"


class ResultPathTest(unittest.TestCase):
    def test_source_and_result_keys_are_mirrored(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = candidate_source_path(root, "cpu_add", CANDIDATE)
            result = candidate_result_path(root / "results", "cpu_add", CANDIDATE)
            self.assertEqual(source.parts[-2:], result.parts[-2:])
            self.assertEqual(source.parent.parent.name, "candidates")
            evaluation = evaluation_result_path(
                root / "results",
                "cpu_add",
                CANDIDATE,
                "20260716T090000Z__01234567__run_001",
            )
            self.assertEqual(evaluation.parent, result)

    def test_traversal_and_absolute_identifiers_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for operator in ("../cpu_add", "cpu/add", str(root.resolve())):
                with self.subTest(operator=operator), self.assertRaises(IdentifierError):
                    candidate_result_path(root, operator, CANDIDATE)
            with self.assertRaises(IdentifierError):
                evaluation_result_path(root, "cpu_add", CANDIDATE, "../evaluation")


if __name__ == "__main__":
    unittest.main()
