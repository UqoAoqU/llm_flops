import math
import unittest

from benchmark_engine.correctness import (
    CalcDiffComparator,
    CorrectnessEvaluator,
    InputBundle,
    OracleGate,
    clone_input_bundle,
    normalize_output,
)
from benchmark_engine.models import CaseSpec


CASE = CaseSpec("oracle", {}, 7, frozenset({"smoke"}))


class Vector:
    device = "cpu"
    layout = "strided"
    dtype = "float32"

    def __init__(self, values):
        self.values = values
        self.shape = (len(values),)

    def stride(self):
        return (1,)

    def reshape(self, *_):
        return self

    def tolist(self):
        return self.values


class OracleSpec:
    operator_id = "oracle_test"

    def make_inputs(self, case, context):
        del case, context
        return InputBundle(args=([1.0, 2.0],))

    def clone_inputs(self, inputs):
        return clone_input_bundle(inputs)

    def normalize_output(self, output):
        return normalize_output(output)

    def comparator(self, case):
        raise AssertionError("legacy reference comparator must not run")

    def correctness_oracles(self, case):
        del case
        return {"math": lambda inputs: list(inputs.args[0])}

    def correctness_gates(self, case):
        del case
        return (
            OracleGate(
                gate_id="implementation",
                oracle_id="math",
                comparator=CalcDiffComparator(max_diff=1e-5, source="test"),
            ),
        )

    def cost_model(self, case):
        del case
        return None


class OracleCorrectnessContractTests(unittest.TestCase):
    def test_calc_diff_matches_upstream_formula_and_zero_rule(self):
        comparator = CalcDiffComparator(max_diff=0.2, source="deepgemm")
        reference = normalize_output(Vector([1.0, 2.0]))
        candidate = normalize_output(Vector([1.0, 3.0]))
        result = comparator.compare(reference, candidate)
        expected = 1.0 - 2.0 * (1.0 + 6.0) / (1.0 + 4.0 + 1.0 + 9.0)
        self.assertAlmostEqual(result.metrics["output"]["calc_diff"], expected)
        self.assertTrue(result.passed)

        zero = comparator.compare(normalize_output(Vector([0.0, 0.0])), normalize_output(Vector([0.0, 0.0])))
        self.assertTrue(zero.passed)
        self.assertEqual(zero.metrics["output"]["calc_diff"], 0.0)
        self.assertTrue(zero.metrics["output"]["zero_denominator"])

    def test_calc_diff_is_strict_and_rejects_nonfinite_values(self):
        comparator = CalcDiffComparator(max_diff=0.0, source="deepgemm")
        equal = comparator.compare(normalize_output(Vector([1.0])), normalize_output(Vector([1.0])))
        self.assertFalse(equal.passed)  # DeepGEMM uses `<`, not `<=`.
        nonfinite = comparator.compare(normalize_output(Vector([1.0])), normalize_output(Vector([math.inf])))
        self.assertFalse(nonfinite.passed)
        self.assertEqual(nonfinite.metrics["output"]["nonfinite_count"], 1)

    def test_calc_diff_compares_only_declared_output_paths(self):
        comparator = CalcDiffComparator(
            max_diff=1e-5,
            source="deepgemm",
            output_paths=("output",),
        )
        reference = normalize_output(
            Vector([1.0, 2.0]),
            {"private_oracle_input": Vector([1.0])},
        )
        candidate = normalize_output(
            Vector([1.0, 2.0]),
            {"private_oracle_input": Vector([99.0])},
        )

        result = comparator.compare(reference, candidate)

        self.assertTrue(result.passed)
        self.assertEqual(set(result.metrics), {"output"})

    def test_calc_diff_requires_every_declared_output_path(self):
        comparator = CalcDiffComparator(
            max_diff=1e-5,
            source="deepgemm",
            output_paths=("output",),
        )

        result = comparator.compare(
            normalize_output({"other": Vector([1.0])}),
            normalize_output({"other": Vector([1.0])}),
        )

        self.assertFalse(result.passed)
        self.assertEqual(result.comparator, "structure")
        self.assertEqual(result.failed_path, "output")

    def test_calc_diff_rejects_invalid_declared_output_paths(self):
        for output_paths in ((), ("",), ("output", "output"), ["output"]):
            with self.subTest(output_paths=output_paths):
                with self.assertRaisesRegex(
                    (TypeError, ValueError),
                    "calc_diff output_paths",
                ):
                    CalcDiffComparator(
                        max_diff=1e-5,
                        source="deepgemm",
                        output_paths=output_paths,
                    )

    def test_oracle_gate_requires_both_reference_and_candidate_to_pass(self):
        result = CorrectnessEvaluator(synchronizer=lambda *_: None).evaluate(
            spec=OracleSpec(),
            reference=lambda values: list(values),
            candidate=lambda values: list(values),
            case=CASE,
        )
        self.assertEqual(result.status, "pass")
        gate = result.comparison.metrics["gates"]["implementation"]
        self.assertTrue(gate["reference"]["passed"])
        self.assertTrue(gate["candidate"]["passed"])
        self.assertEqual(gate["oracle_id"], "math")

        broken_reference = CorrectnessEvaluator(synchronizer=lambda *_: None).evaluate(
            spec=OracleSpec(),
            reference=lambda values: [value * 2 for value in values],
            candidate=lambda values: list(values),
            case=CASE,
        )
        self.assertEqual(broken_reference.status, "fail")
        gate = broken_reference.comparison.metrics["gates"]["implementation"]
        self.assertFalse(gate["reference"]["passed"])
        self.assertTrue(gate["candidate"]["passed"])
