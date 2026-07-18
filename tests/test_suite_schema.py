import tempfile
import unittest
from pathlib import Path

from benchmark_engine.suite import SuiteValidationError, load_suite


VALID = """schema_version: 1
suite_id: smoke
operators:
  include: ['*']
  exclude: []
cases:
  tags: [smoke]
mode: correctness
correctness:
  seeds: [0, 7]
performance:
  samples: 3
  inner_iterations: 1
"""


class SuiteSchemaTest(unittest.TestCase):
    def parse(self, text: str):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "suite.yaml"
        path.write_text(text, encoding="utf-8")
        return load_suite(path)

    def test_strict_v1_rounds_to_immutable_values(self):
        suite = self.parse(VALID)
        self.assertEqual(suite.correctness_seeds, (0, 7))
        self.assertEqual(suite.operator_include, ("*",))
        self.assertIsNone(suite.performance_timer)
        self.assertIsNone(suite.performance_warmup)

        configured = self.parse(
            VALID.replace(
                "  samples: 3",
                "  timer: cuda_event\n  warmup: 0\n  samples: 3",
            )
        )
        self.assertEqual(configured.performance_timer, "cuda_event")
        self.assertEqual(configured.performance_warmup, 0)

    def test_unknown_semantic_fields_are_rejected(self):
        for field in ("shape: [2, 2]", "rtol: 1e-2", "reference: other"):
            with self.subTest(field=field), self.assertRaisesRegex(
                SuiteValidationError, "suite.unknown_field"
            ):
                self.parse(VALID.replace("  tags: [smoke]", f"  tags: [smoke]\n  {field}"))

    def test_type_and_version_errors_are_stable(self):
        with self.assertRaisesRegex(SuiteValidationError, "suite.schema_version"):
            self.parse(VALID.replace("schema_version: 1", "schema_version: 2"))
        with self.assertRaisesRegex(SuiteValidationError, "suite.type"):
            self.parse(VALID.replace("seeds: [0, 7]", "seeds: zero"))

    def test_performance_optional_fields_remain_strict(self):
        invalid = (
            (
                VALID.replace("  samples: 3", "  timer: stopwatch\n  samples: 3"),
                "suite.value",
            ),
            (
                VALID.replace("  samples: 3", "  warmup: -1\n  samples: 3"),
                "suite.value",
            ),
            (
                VALID.replace("  samples: 3", "  mystery: 4\n  samples: 3"),
                "suite.unknown_field",
            ),
        )
        for text, code in invalid:
            with self.subTest(code=code), self.assertRaisesRegex(
                SuiteValidationError, code
            ):
                self.parse(text)

    def test_empty_performance_inherits_operator_measurement_shape(self):
        inherited = self.parse(
            VALID.replace("performance:\n  samples: 3\n  inner_iterations: 1",
                          "performance: {}")
        )
        self.assertIsNone(inherited.performance_samples)
        self.assertIsNone(inherited.performance_inner_iterations)
        self.assertIsNone(inherited.performance_timer)


if __name__ == "__main__":
    unittest.main()
