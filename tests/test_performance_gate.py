import unittest

from benchmark_engine.performance.gate import PerformanceGateConfig, evaluate_performance_gate


def payload(**overrides):
    value = {"status": "pass", "formal": True, "speedup": 1.1,
             "slowdown_pct": -9.0, "peak_memory_allocated_bytes": 10,
             "measurements": {
                 "reference": {"selection": {"effective_timer": "wall_clock"}},
                 "candidate": {"selection": {"effective_timer": "wall_clock"},
                               "statistics": {"median_ms": 1.0, "cv": .01}},
             }}
    value.update(overrides)
    return value


class GateTests(unittest.TestCase):
    def test_pass_and_boundaries(self):
        gate = evaluate_performance_gate(payload(speedup=1.0, slowdown_pct=5.0),
            PerformanceGateConfig(max_slowdown_pct=5, min_speedup=1,
                                  max_candidate_median_ms=1, max_cv=.01,
                                  max_memory_bytes=10), correctness_pass=True)
        self.assertEqual(gate.status, "passed")
        self.assertTrue(gate.ranking_eligible)

    def test_each_failure_is_stable(self):
        gate = evaluate_performance_gate(payload(speedup=.5, slowdown_pct=6),
            PerformanceGateConfig(max_slowdown_pct=5, min_speedup=1,
                                  max_candidate_median_ms=.5, max_cv=.001,
                                  max_memory_bytes=5), correctness_pass=True)
        self.assertEqual(gate.status, "failed")
        self.assertEqual(gate.reasons, ("max_slowdown_exceeded", "min_speedup_not_met",
            "max_candidate_median_exceeded", "max_cv_exceeded", "max_memory_exceeded_or_unavailable"))

    def test_unstable_mismatch_and_nonformal_cannot_rank(self):
        value = payload(status="unstable", formal=False)
        value["measurements"]["candidate"]["selection"]["effective_timer"] = "cuda_event"
        gate = evaluate_performance_gate(value, PerformanceGateConfig(), correctness_pass=True)
        self.assertEqual(gate.status, "failed")
        self.assertIn("effective_timer_mismatch", gate.reasons)

    def test_unsupported_allow_never_ranks(self):
        gate = evaluate_performance_gate({"status": "unsupported"},
            PerformanceGateConfig(unsupported_policy="allow"), correctness_pass=True)
        self.assertEqual(gate.status, "passed")
        self.assertFalse(gate.ranking_eligible)

    def test_invalid_thresholds_rejected(self):
        for value in (float("nan"), float("inf"), -1):
            with self.assertRaises(ValueError): PerformanceGateConfig(max_cv=value)


if __name__ == "__main__": unittest.main()
