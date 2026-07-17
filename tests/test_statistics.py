import math
import unittest

from benchmark_engine.performance import compute_statistics, percentile


class StatisticsTests(unittest.TestCase):
    def test_known_population_statistics_and_type7_percentiles(self):
        result = compute_statistics(
            [1.0, 2.0, 3.0, 4.0], minimum_stable_samples=4, maximum_cv=1.0
        )
        self.assertEqual(result.count, 4)
        self.assertEqual(result.mean_ms, 2.5)
        self.assertEqual(result.median_ms, 2.5)
        self.assertAlmostEqual(result.stddev_ms, math.sqrt(1.25))
        self.assertEqual(result.p50_ms, 2.5)
        self.assertAlmostEqual(result.p90_ms, 3.7)
        self.assertAlmostEqual(result.p95_ms, 3.85)
        self.assertAlmostEqual(result.p99_ms, 3.97)
        self.assertFalse(result.unstable)
        self.assertEqual(result.to_dict()["stddev_kind"], "population")

    def test_single_sample_is_defined_but_marked_unstable(self):
        result = compute_statistics([2.0])
        self.assertEqual(result.stddev_ms, 0.0)
        self.assertEqual(result.cv, 0.0)
        self.assertTrue(result.unstable)
        self.assertIn("sample_count", result.instability_reason)
        self.assertEqual(percentile([2.0], 0.99), 2.0)

    def test_high_cv_is_visible_and_samples_are_not_removed(self):
        result = compute_statistics(
            [1.0, 1.0, 10.0, 1.0, 1.0],
            minimum_stable_samples=5,
            maximum_cv=0.1,
        )
        self.assertTrue(result.unstable)
        self.assertIn("cv>", result.instability_reason)
        self.assertEqual(result.max_ms, 10.0)

    def test_empty_nonfinite_negative_and_bad_quantiles_are_rejected(self):
        for values in ([], [math.nan], [math.inf], [-1.0]):
            with self.subTest(values=values), self.assertRaises((TypeError, ValueError)):
                compute_statistics(values)
        for quantile in (-0.1, 1.1, math.nan):
            with self.subTest(quantile=quantile), self.assertRaises(ValueError):
                percentile([1.0], quantile)


if __name__ == "__main__":
    unittest.main()
