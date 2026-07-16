import unittest
import json
import math
from collections.abc import Mapping

from benchmark_engine.correctness import InputBundle, assert_input_isolation, case_fingerprint, clone_input_bundle, input_summary, make_generator_context
from benchmark_engine.models import CaseSpec


class Cloneable:
    shape = (2,)
    device = "cpu"
    def __init__(self, values): self.values = list(values)
    def clone(self): return Cloneable(self.values)
    def data_ptr(self): return id(self.values)


class InputBundleTests(unittest.TestCase):
    def test_clone_preserves_alias_inside_bundle_but_isolates_bundles(self):
        state = [1, 2]
        original = InputBundle(args=(state,), observed_state={"cache": state})
        left = clone_input_bundle(original); right = clone_input_bundle(original)
        self.assertIs(left.args[0], left.observed_state["cache"])
        assert_input_isolation(left, right)
        left.args[0].append(3)
        self.assertEqual(right.args[0], [1, 2])

    def test_shared_tensor_storage_is_rejected(self):
        tensor = Cloneable([1, 2])
        alias = Cloneable([])
        alias.values = tensor.values
        with self.assertRaisesRegex(ValueError, "share tensor storage"):
            assert_input_isolation(InputBundle(args=(tensor,)), InputBundle(args=(alias,)))

    def test_explicit_seed_is_repeatable(self):
        first = make_generator_context(42); second = make_generator_context(42)
        if hasattr(first.cpu, "initial_seed"):
            self.assertEqual(first.cpu.initial_seed(), second.cpu.initial_seed())
        else:
            self.assertEqual(first.cpu.random(), second.cpu.random())

    def test_empty_torch_tensors_do_not_false_positive(self):
        try: import torch
        except ImportError: self.skipTest("torch unavailable")
        assert_input_isolation(InputBundle(args=(torch.empty(0),)), InputBundle(args=(torch.empty(0),)))

    def test_numpy_views_are_rejected(self):
        try: import numpy
        except ImportError: self.skipTest("numpy unavailable")
        base=numpy.arange(16)
        with self.assertRaisesRegex(ValueError,"array buffer"):
            assert_input_isolation(InputBundle(args=(base[1:10],)),InputBundle(args=(base[3:12],)))

    def test_summary_is_bounded_nonfinite_and_hash_stable(self):
        deep: object = "leaf"
        for _ in range(20): deep=[deep]
        inputs=InputBundle(args=([math.nan,math.inf,-math.inf]+list(range(200)),deep,"x"*1000),kwargs={1:"integer","1":"string"})
        summary=input_summary(inputs)
        encoded=json.dumps(summary,sort_keys=True,allow_nan=False)
        self.assertIn('"NaN"',encoded); self.assertIn('"+Inf"',encoded); self.assertIn('"-Inf"',encoded)
        args_summary=summary["args"]
        large=args_summary["items"][0]
        self.assertEqual(large["size"],203); self.assertTrue(large["truncated"])
        key_tokens=[entry["key"] for entry in summary["kwargs"]["entries"]]
        self.assertEqual(len(key_tokens),len(set(key_tokens)))
        case=CaseSpec("bounded",{},3,frozenset())
        self.assertEqual(case_fingerprint(case,summary),case_fingerprint(case,input_summary(inputs)))

    def test_mapping_key_encoding_rejects_collisions(self):
        class Colliding(Mapping):
            def __len__(self): return 2
            def __iter__(self): return iter((1,1))
            def __getitem__(self,key): return "value"
            def items(self): return ((1,"left"),(1,"right"))
        with self.assertRaisesRegex(ValueError,"collide"):
            input_summary(InputBundle(kwargs={"nested":Colliding()}))


if __name__ == "__main__": unittest.main()
