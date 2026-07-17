import math
import unittest

from benchmark_engine.performance import TheoreticalCost, evaluate_cost_model


class CostModelTests(unittest.TestCase):
    def test_theoretical_rates_use_milliseconds_and_decimal_units(self):
        result = evaluate_cost_model(
            TheoreticalCost(2.0e12, 4.0e9, throughput_units=1000), 2.0
        )
        self.assertTrue(result.available)
        self.assertTrue(result.theoretical)
        self.assertEqual(result.tflops, 1000.0)
        self.assertEqual(result.effective_bandwidth_gbps, 2000.0)
        self.assertEqual(result.arithmetic_intensity, 500.0)
        self.assertEqual(result.throughput, 500000.0)

    def test_none_is_explicitly_unavailable_not_zero(self):
        for latency in (0.0, 1.0, math.nan):
            with self.subTest(latency=latency):
                result = evaluate_cost_model(None, latency)
                self.assertFalse(result.available)
                self.assertIsNone(result.tflops)
                self.assertIn("unavailable", result.reason)

    def test_invalid_cost_or_latency_is_rejected(self):
        cost = TheoreticalCost(1, 1)
        for value in (0, -1, math.nan, math.inf):
            with self.subTest(latency=value), self.assertRaises((TypeError, ValueError)):
                evaluate_cost_model(cost, value)
        for cost in (
            {"flops": 0, "bytes": 1},
            {"flops": 1, "bytes": -1},
            {"flops": 1, "bytes": 1, "estimated_bytes": 1},
            {"flops": 1, "bytes": 1, "mystery": 2},
        ):
            with self.subTest(cost=cost), self.assertRaises((TypeError, ValueError)):
                evaluate_cost_model(cost, 1.0)


if __name__ == "__main__":
    unittest.main()
