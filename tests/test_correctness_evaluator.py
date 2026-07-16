import json
import unittest

from benchmark_engine.correctness import ComparisonResult, CorrectnessEvaluator, CorrectnessResult, ExactComparator, FloatingComparator, InputBundle, Tolerance, clone_input_bundle, normalize_output
from benchmark_engine.models import CaseSpec


CASE=CaseSpec("small",{},7,frozenset({"smoke"}))

class Spec:
    def cases(self): return (CASE,)
    def make_inputs(self,case,ctx):
        state=[case.seed]
        return InputBundle(args=(state,),observed_state={"cache":state})
    def clone_inputs(self,inputs): return clone_input_bundle(inputs)
    def normalize_output(self,output): return normalize_output(output)
    def comparator(self,case): return ExactComparator()
    def cost_model(self,case): return None


class CorrectnessEvaluatorTests(unittest.TestCase):
    def test_pass_state_mutation_and_reproducible_case_hash(self):
        def operator(cache): cache.append(1); return len(cache)
        evaluator=CorrectnessEvaluator(synchronizer=lambda *_:None)
        first=evaluator.evaluate(spec=Spec(),reference=operator,candidate=operator,case=CASE,operator_id="abc",candidate_id="task__run__deadbeef")
        second=evaluator.evaluate(spec=Spec(),reference=operator,candidate=operator,case=CASE,operator_id="abc",candidate_id="task__run__deadbeef")
        self.assertEqual(first.status,"pass"); self.assertEqual(first.case_hash,second.case_hash)
        json.dumps(first.to_dict(),allow_nan=False)

    def test_result_serialization_bounds_adversarial_nonfinite_metadata(self):
        comparison=ComparisonResult(False,"custom",{"nan":float("nan"),"inf":float("inf"),"large":list(range(1000))})
        result=CorrectnessResult("fail","small",7,"hash","version",{"text":"x"*10000},comparison,{"value":float("-inf")})
        encoded=json.dumps(result.to_dict(),allow_nan=False)
        self.assertIn('"NaN"',encoded); self.assertIn('"+Inf"',encoded); self.assertIn('"-Inf"',encoded)
        self.assertLess(len(encoded),20000)

    def test_value_and_state_failure(self):
        def ref(cache): cache.append(1); return 2
        def cand(cache): cache.append(2); return 3
        result=CorrectnessEvaluator(synchronizer=lambda *_:None).evaluate(spec=Spec(),reference=ref,candidate=cand,case=CASE)
        self.assertEqual(result.status,"fail"); self.assertIn(result.comparison.failed_path,{"output","state.cache[1]"})
        self.assertIn("--seed 7",result.diagnostic["reproduction_command"])

    def test_candidate_exception_timeout_and_async_sync_error(self):
        def ref(cache): return 1
        def fail(cache): raise RuntimeError("boom")
        self.assertEqual(CorrectnessEvaluator(synchronizer=lambda *_:None).evaluate(spec=Spec(),reference=ref,candidate=fail,case=CASE).status,"error")
        def timeout(cache): raise TimeoutError("late")
        self.assertEqual(CorrectnessEvaluator(synchronizer=lambda *_:None).evaluate(spec=Spec(),reference=ref,candidate=timeout,case=CASE).status,"timeout")
        calls=[]
        def sync(output,inputs):
            calls.append(output)
            if len(calls)==2: raise RuntimeError("asynchronous launch failed")
        result=CorrectnessEvaluator(synchronizer=sync).evaluate(spec=Spec(),reference=ref,candidate=ref,case=CASE)
        self.assertEqual(result.status,"error"); self.assertEqual(result.diagnostic["stage"],"candidate")

    def test_import_error_is_error_for_reference_and_candidate(self):
        def missing(cache): raise ImportError("dependency missing")
        evaluator=CorrectnessEvaluator(synchronizer=lambda *_:None)
        reference=evaluator.evaluate(spec=Spec(),reference=missing,candidate=lambda cache:1,case=CASE)
        candidate=evaluator.evaluate(spec=Spec(),reference=lambda cache:1,candidate=missing,case=CASE)
        self.assertEqual(reference.status,"error"); self.assertEqual(reference.diagnostic["stage"],"reference")
        self.assertEqual(candidate.status,"error"); self.assertEqual(candidate.diagnostic["stage"],"candidate")

    def test_nondeterminism_is_distinct(self):
        counter={"value":0}
        def ref(cache): return 0
        def cand(cache): counter["value"]+=1; return counter["value"]
        result=CorrectnessEvaluator(determinism_repeats=2,synchronizer=lambda *_:None).evaluate(spec=Spec(),reference=ref,candidate=cand,case=CASE)
        self.assertEqual(result.status,"nondeterministic")

    def test_bad_clone_is_rejected(self):
        class Bad(Spec):
            def clone_inputs(self,inputs): return inputs
        result=CorrectnessEvaluator(synchronizer=lambda *_:None).evaluate(spec=Bad(),reference=lambda x:1,candidate=lambda x:1,case=CASE)
        self.assertEqual(result.status,"error"); self.assertEqual(result.diagnostic["stage"],"input")

    def test_keyboard_interrupt_is_not_hidden(self):
        def interrupt(cache): raise KeyboardInterrupt()
        with self.assertRaises(KeyboardInterrupt):
            CorrectnessEvaluator(synchronizer=lambda *_:None).evaluate(spec=Spec(),reference=lambda cache:1,candidate=interrupt,case=CASE)

    def test_every_determinism_clone_is_isolated_from_canonical(self):
        class LateAlias(Spec):
            def __init__(self): self.canonical=None; self.calls=0
            def make_inputs(self,case,ctx):
                self.canonical=super().make_inputs(case,ctx); return self.canonical
            def clone_inputs(self,inputs):
                self.calls+=1
                return self.canonical if self.calls == 4 else clone_input_bundle(inputs)
        result=CorrectnessEvaluator(determinism_repeats=3,synchronizer=lambda *_:None).evaluate(spec=LateAlias(),reference=lambda cache:1,candidate=lambda cache:1,case=CASE)
        self.assertEqual(result.status,"error"); self.assertEqual(result.diagnostic["stage"],"candidate")

    def test_cuda_smoke_when_available(self):
        try: import torch
        except ImportError: self.skipTest("torch unavailable")
        if not torch.cuda.is_available(): self.skipTest("CUDA unavailable")
        class CudaSpec:
            def make_inputs(self,case,ctx):
                value=torch.rand((4,),device="cuda",generator=ctx.cuda["cuda:0"])
                return InputBundle(args=(value,))
            def clone_inputs(self,x): return clone_input_bundle(x)
            def normalize_output(self,x): return normalize_output(x)
            def comparator(self,c): return FloatingComparator(operator_default=Tolerance(0,0))
        result=CorrectnessEvaluator().evaluate(spec=CudaSpec(),reference=lambda x:x+1,candidate=lambda x:x+1,case=CASE,cuda_devices=("cuda:0",))
        self.assertEqual(result.status,"pass")


if __name__ == "__main__": unittest.main()
