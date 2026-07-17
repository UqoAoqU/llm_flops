import unittest

from benchmark_engine.correctness.models import InputBundle
from benchmark_engine.models import CaseSpec
from benchmark_engine.performance.evaluator import PerformanceConfig, PerformanceEvaluator
from benchmark_engine.performance.timers import RawSample, TimerSelection


class Timer:
    def __init__(self, role, trace): self.role, self.trace, self.prepared = role, trace, False
    @property
    def selection(self): return TimerSelection("wall_clock", "wall_clock")
    def prepare(self, fn, config): self.prepared = True; self.trace.append(self.role + "-capture"); return 0
    def sample(self, fn, config):
        self.trace.append(self.role[0].upper())
        fn()
        return (RawSample(0, config.inner_iterations, 2.0, 1.0),)


class Spec:
    def make_inputs(self, case, context): return InputBundle(args=([1],))
    def clone_inputs(self, bundle): return InputBundle(args=(list(bundle.args[0]),))
    def cost_model(self, case): return None


class MeasurementOrderTests(unittest.TestCase):
    def test_exact_interleave_indices_and_independent_prepare(self):
        trace = []
        roles = iter(("reference", "candidate"))
        evaluator = PerformanceEvaluator(timer_factory=lambda requested, sync: Timer(next(roles), trace))
        result = evaluator.evaluate(spec=Spec(), reference=lambda x: x,
            candidate=lambda x: x, case=CaseSpec("tiny", {}, 0, frozenset()),
            config=PerformanceConfig(requested_timer="wall_clock", warmup=1,
                                     samples=4, inner_iterations=2))
        steady = [item for item in trace if item in {"R", "C"}]
        self.assertEqual(steady, list("RCCR" * 2))
        self.assertEqual([s.sample_index for s in result.reference.samples], list(range(4)))
        self.assertEqual([s.sample_index for s in result.candidate.samples], list(range(4)))
        merged = sorted((*result.reference.samples, *result.candidate.samples), key=lambda s: s.order_index)
        self.assertEqual([s.order_index for s in merged], list(range(8)))
        self.assertEqual(result.reference.statistics.median_ms, 1.0)


if __name__ == "__main__": unittest.main()
