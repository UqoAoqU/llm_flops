import unittest

from benchmark_engine.correctness import InputBundle, clone_input_bundle
from benchmark_engine.models import CaseSpec
from benchmark_engine.performance import PerformanceConfig, PerformanceEvaluator


class StepClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        current = self.value
        self.value += 0.001
        return current


class Spec:
    def make_inputs(self, case, context):
        return InputBundle(args=([1.0, 2.0], [3.0, 4.0]))

    def clone_inputs(self, inputs):
        return clone_input_bundle(inputs)

    def cost_model(self, case):
        return {"flops": 2, "bytes": 32}


class PerformanceEvaluatorCpuTests(unittest.TestCase):
    def test_first_warmup_and_sampling_are_separate_for_both_roles(self):
        calls = {"reference": 0, "candidate": 0}

        def reference(left, right):
            calls["reference"] += 1
            return [a + b for a, b in zip(left, right)]

        def candidate(left, right):
            calls["candidate"] += 1
            return [a + b for a, b in zip(left, right)]

        clock = StepClock()
        result = PerformanceEvaluator(
            clock=clock, synchronizer=lambda output, inputs: None
        ).evaluate(
            spec=Spec(),
            reference=reference,
            candidate=candidate,
            case=CaseSpec("case", {}, 0, frozenset()),
            config=PerformanceConfig(
                requested_timer="wall_clock",
                warmup=1,
                samples=2,
                inner_iterations=2,
                minimum_stable_samples=2,
                maximum_cv=1.0,
            ),
        )
        # Per role: one first call, one warmup, and 2*2 steady calls.
        self.assertEqual(calls, {"reference": 6, "candidate": 6})
        for measurement in (result.reference, result.candidate):
            self.assertAlmostEqual(measurement.first_call_ms, 1.0)
            self.assertAlmostEqual(measurement.warmup_ms, 1.0)
            self.assertEqual(len(measurement.samples), 2)
            self.assertEqual(
                [round(sample.per_call_ms, 6) for sample in measurement.samples],
                [0.5, 0.5],
            )
            self.assertAlmostEqual(measurement.steady_state_ms, 2.0)
        self.assertEqual(result.status, "pass")
        self.assertTrue(result.cost.available)

    def test_prepare_does_not_create_steady_state_samples(self):
        calls = 0

        def operator(left, right):
            nonlocal calls
            calls += 1
            return left

        evaluator = PerformanceEvaluator(
            clock=StepClock(), synchronizer=lambda output, inputs: None
        )
        prepared = evaluator.prepare(
            spec=Spec(),
            reference=operator,
            candidate=operator,
            case=CaseSpec("case", {}, 0, frozenset()),
            config=PerformanceConfig(
                requested_timer="wall_clock", warmup=0, samples=3, inner_iterations=2
            ),
        )
        self.assertEqual(calls, 2)
        evaluator.sample(prepared)
        self.assertEqual(calls, 14)


if __name__ == "__main__":
    unittest.main()
