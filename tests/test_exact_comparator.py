import unittest
import json
import math

from benchmark_engine.correctness import ExactComparator, normalize_output


class Array:
    device="cpu"; layout="strided"
    def __init__(self, values, dtype="int64", shape=None, stride=None): self.values=values; self.dtype=dtype; self.shape=shape or (len(values),); self._stride=stride or (1,)
    def stride(self): return self._stride
    def reshape(self,*_): return self
    def tolist(self): return self.values


class ExactComparatorTests(unittest.TestCase):
    def test_exact_pass_and_value_failure(self):
        comparator=ExactComparator()
        self.assertTrue(comparator.compare(normalize_output(Array([1,2])), normalize_output(Array([1,2]))).passed)
        result=comparator.compare(normalize_output(Array([1,2])), normalize_output(Array([1,3])))
        self.assertFalse(result.passed); self.assertEqual(result.metrics["mismatch_count"], 1)

    def test_dtype_shape_and_layout_are_checked_before_values(self):
        for candidate in (Array([1,2], "int32"), Array([1,2], shape=(1,2), stride=(2,1)), Array([1,2], stride=(2,))):
            result=ExactComparator().compare(normalize_output(Array([1,2])), normalize_output(candidate))
            self.assertFalse(result.passed); self.assertEqual(result.comparator, "structure")

    def test_nonfinite_diagnostic_remains_json_safe(self):
        result=ExactComparator().compare(normalize_output(Array([math.nan],"float32")),normalize_output(Array([math.inf],"float32")))
        json.dumps(result.diagnostics,allow_nan=False)


if __name__ == "__main__": unittest.main()
