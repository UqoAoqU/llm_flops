import unittest

from benchmark_engine.correctness import QuantizationParameters, QuantizedComparator, Tolerance, normalize_output


class Array:
    device="cpu"; layout="packed"
    def __init__(self, values, shape=None): self.values=values; self.dtype="int8"; self.shape=shape or (len(values),)
    def stride(self): return (1,) if len(self.shape)==1 else (self.shape[1],1)
    def reshape(self,*_): return self
    def tolist(self): return self.values


class QuantizedComparatorTests(unittest.TestCase):
    def test_raw_code_is_exact(self):
        comp=QuantizedComparator(mode="raw_code")
        exact=comp.compare(normalize_output(Array([1,2])),normalize_output(Array([1,2])))
        self.assertTrue(exact.passed); self.assertEqual(exact.metrics["zero_rate"],0.0)
        self.assertFalse(comp.compare(normalize_output(Array([1,2])),normalize_output(Array([1,3]))).passed)

    def test_dequant_requires_explicit_metadata_and_reuses_float(self):
        with self.assertRaisesRegex(ValueError,"explicit scale"):
            QuantizedComparator(mode="dequant",tolerance=Tolerance(0,.2)).compare(normalize_output(Array([1])),normalize_output(Array([1])))
        comp=QuantizedComparator(mode="dequant",parameters=QuantizationParameters((.5,1.),(0,1),axis=1,layout="packed"),tolerance=Tolerance(0,.6))
        self.assertTrue(comp.compare(normalize_output(Array([2,2],(1,2))),normalize_output(Array([3,2],(1,2)))).passed)
        bad=QuantizedComparator(mode="dequant",parameters=QuantizationParameters(1.,layout="wrong"),tolerance=Tolerance(0,0))
        with self.assertRaisesRegex(ValueError,"layout mismatch"): bad.compare(normalize_output(Array([1])),normalize_output(Array([1])))

    def test_parameters_are_validated_at_construction(self):
        for scale in (True,0.,-1.,float("nan")):
            with self.assertRaises((TypeError,ValueError)): QuantizationParameters(scale)
        with self.assertRaises(TypeError): QuantizationParameters(1.,zero_point=True)
        with self.assertRaises(ValueError): QuantizationParameters((1.,2.))
        with self.assertRaises(TypeError): QuantizationParameters(1.,axis=True)
        with self.assertRaises(ValueError): QuantizationParameters(1.,quant_min=-8)
        with self.assertRaises(ValueError): QuantizationParameters(1.,quant_min=8,quant_max=8)

    def test_dequant_metrics_and_explicit_code_range(self):
        params=QuantizationParameters(0.5,quant_min=-2,quant_max=2)
        comp=QuantizedComparator(mode="dequant",parameters=params,tolerance=Tolerance(0,.6))
        compared=comp.compare(normalize_output(Array([-2,0,1])),normalize_output(Array([-2,0,2])))
        self.assertTrue(compared.passed)
        self.assertIn("dequant_error",compared.metrics); self.assertAlmostEqual(compared.metrics["zero_rate"],1/3)
        self.assertAlmostEqual(compared.metrics["saturation_rate"],2/3)
        failed=comp.compare(normalize_output(Array([0])),normalize_output(Array([3])))
        self.assertFalse(failed.passed); self.assertEqual(failed.comparator,"quantized_code_range")


if __name__ == "__main__": unittest.main()
