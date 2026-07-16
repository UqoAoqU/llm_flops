import dataclasses
import unittest

from benchmark_engine.correctness import OutputNormalizationError, normalize_output


class Array:
    dtype = "float32"; device = "cpu"; layout = "strided"
    def __init__(self, values, shape=(2,), stride=(1,)): self.values=values; self.shape=shape; self._stride=stride
    def stride(self): return self._stride
    def reshape(self, *_): return self
    def tolist(self): return self.values

@dataclasses.dataclass
class Result:
    logits: object
    count: int


class OutputNormalizationTests(unittest.TestCase):
    def test_nested_paths_and_contract(self):
        bundle = normalize_output(Result(Array([1.0, 2.0]), 2), {"cache": Array([3.0, 4.0], stride=(2,))})
        self.assertEqual([leaf.path for leaf in bundle.leaves], ["output.logits", "output.count", "state.cache"])
        self.assertEqual(bundle.leaves[0].dtype, "float32")
        self.assertEqual(bundle.leaves[2].stride, (2,))

    def test_mapping_order_is_deterministic(self):
        self.assertEqual([x.path for x in normalize_output({"z": 1, "a": 2}).leaves], ["output.a", "output.z"])

    def test_rejects_unstable_key_cycle_and_unsupported_type_with_path(self):
        with self.assertRaisesRegex(OutputNormalizationError, "stable identifiers at output"):
            normalize_output({"bad.key": 1})
        cycle=[]; cycle.append(cycle)
        with self.assertRaisesRegex(OutputNormalizationError, r"output\[0\]"):
            normalize_output(cycle)
        with self.assertRaisesRegex(OutputNormalizationError, "output.good"):
            normalize_output({"good": object()})

    def test_empty_sequence_is_a_valid_empty_structure(self):
        self.assertEqual(normalize_output([]).leaves, ())


if __name__ == "__main__": unittest.main()
