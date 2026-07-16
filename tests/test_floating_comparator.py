import json
import math
import unittest

from benchmark_engine.correctness import FloatingComparator, Tolerance, normalize_output, resolve_tolerance


class Array:
    device="cpu"; layout="strided"
    def __init__(self, values, dtype="float32"): self.values=values; self.dtype=dtype; self.shape=(len(values),)
    def stride(self): return (1,)
    def reshape(self,*_): return self
    def tolist(self): return self.values


class FloatingComparatorTests(unittest.TestCase):
    def test_metrics_and_error_threshold(self):
        comp=FloatingComparator(operator_default=Tolerance(0, .1), require_explicit=True)
        result=comp.compare(normalize_output(Array([0., 2., 4.])), normalize_output(Array([0., 2.05, 5.])))
        self.assertFalse(result.passed)
        metrics=result.metrics["output"]
        for name in ("max_abs_error","mean_abs_error","p50_abs_error","p95_abs_error","p99_abs_error","max_rel_error","rmse","relative_l2","cosine_similarity","mismatch_count","mismatch_rate"):
            self.assertIn(name, metrics)
        self.assertEqual(metrics["mismatch_count"], 1)

    def test_tolerance_priority_and_formal_requirement(self):
        fallback=resolve_tolerance("float32")
        self.assertEqual(fallback.source, "dtype_fallback")
        operator=Tolerance(.3,.3); output=Tolerance(.2,.2); case=Tolerance(.1,.1)
        self.assertEqual(resolve_tolerance("float32",operator_default=operator,output_override=output,case_override=case).rtol,.1)
        with self.assertRaisesRegex(ValueError,"explicit tolerance"):
            resolve_tolerance("float32", require_explicit=True)

    def test_tolerance_rejects_invalid_configuration_at_construction(self):
        for rtol,atol in ((-1.,0.),(0.,-1.),(math.nan,0.),(math.inf,0.),(True,0.)):
            with self.assertRaises(ValueError): Tolerance(rtol,atol)
        with self.assertRaises(TypeError): Tolerance(0.,0.,equal_nan=1)

    def test_nan_inf_zero_and_empty_semantics_are_json_safe(self):
        comp=FloatingComparator(operator_default=Tolerance(0,0,equal_nan=True))
        result=comp.compare(normalize_output(Array([math.nan, math.inf, -math.inf, 0.])), normalize_output(Array([math.nan, math.inf, math.inf, 0.])))
        self.assertFalse(result.passed); self.assertEqual(result.metrics["output"]["mismatch_count"],1)
        json.dumps(result.metrics, allow_nan=False)
        empty=comp.compare(normalize_output([]),normalize_output([]))
        self.assertTrue(empty.passed)

    def test_empty_zero_nonfinite_and_zero_denominator_semantics(self):
        comp=FloatingComparator(operator_default=Tolerance(0,10))
        empty=comp.compare(normalize_output(Array([])),normalize_output(Array([]))).metrics["output"]
        zeros=comp.compare(normalize_output(Array([0.,0.])),normalize_output(Array([0.,0.]))).metrics["output"]
        infinities=comp.compare(normalize_output(Array([math.inf,-math.inf])),normalize_output(Array([math.inf,-math.inf]))).metrics["output"]
        zero_reference=comp.compare(normalize_output(Array([0.,0.])),normalize_output(Array([1.,0.]))).metrics["output"]
        for metrics in (empty,infinities):
            for name in ("max_abs_error","mean_abs_error","p50_abs_error","p95_abs_error","p99_abs_error","max_rel_error","rmse","relative_l2","cosine_similarity"):
                self.assertIsNone(metrics[name])
        self.assertEqual(zeros["relative_l2"],0.0); self.assertEqual(zeros["cosine_similarity"],1.0)
        self.assertIsNone(zero_reference["relative_l2"]); self.assertIsNone(zero_reference["cosine_similarity"])
        json.dumps({"empty":empty,"zeros":zeros,"inf":infinities,"zero_reference":zero_reference},allow_nan=False)


if __name__ == "__main__": unittest.main()
