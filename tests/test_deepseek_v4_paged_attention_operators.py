from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import json
import os
import sys
import types
import unittest
from unittest import mock
from pathlib import Path

from benchmark_engine.cli import main
from benchmark_engine.correctness import CorrectnessEvaluator
from benchmark_engine.correctness.models import OutputBundle, OutputLeaf
from benchmark_engine.ids import candidate_result_path, candidate_source_path
from benchmark_engine.performance import PerformanceConfig, PerformanceEvaluator
from benchmark_engine.performance.timers import AutoTimer, RawSample, TimerSelection, TimerUnsupportedError
from benchmark_engine.registry import FilesystemRegistry


ROOT = Path(__file__).resolve().parents[1]
OPERATORS = {
    "deepseek_v4_fp8_paged_mqa_logits": "reference_control__20260717T090000Z__0a12c4e8",
    "deepseek_v4_sparse_prefill_attention": "reference_control__20260717T090100Z__8c38e7a1",
    "deepseek_v4_sparse_decode_attention": "reference_control__20260717T090200Z__bb63c012",
    "deepseek_v4_dense_swa_attention": "reference_control__20260717T090300Z__1dc9f30a",
}


def load(path: Path, attribute: str):
    spec = importlib.util.spec_from_file_location(f"_phase12_{path.parent.name}_{path.stem}_{id(path)}", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, attribute)


SPECS = {operator_id: load(ROOT / "operators" / "references" / operator_id / "spec.py", "SPEC") for operator_id in OPERATORS}


class Phase12ContractTests(unittest.TestCase):
    def test_registry_safe_specs_strict_manifests_and_control_candidates(self):
        snapshot = FilesystemRegistry(ROOT).discover()
        issues = [issue for issue in snapshot.issues if any(operator_id in str(issue.path) for operator_id in OPERATORS)]
        self.assertFalse(issues, issues)
        for operator_id, candidate_id in OPERATORS.items():
            with self.subTest(operator=operator_id):
                reference_root = ROOT / "operators" / "references" / operator_id
                source = (reference_root / "spec.py").read_text(encoding="utf-8")
                imported = []
                for node in ast.walk(ast.parse(source)):
                    if isinstance(node, ast.Import):
                        imported.extend(alias.name for alias in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        imported.append(node.module)
                self.assertFalse(any(name == "torch" or name.startswith(("torch.", "sglang", "sgl_kernel", "deep_gemm")) for name in imported))
                manifest = snapshot.operator_specs[operator_id].manifest
                self.assertEqual((manifest.performance.timer, manifest.performance.graph_mode), ("auto", "auto"))
                candidate_root = candidate_source_path(ROOT, operator_id, candidate_id)
                self.assertIn(candidate_id, {candidate.implementation_id for candidate in snapshot.candidates[operator_id]})
                self.assertEqual((reference_root / "implementation.py").read_bytes(), (candidate_root / "implementation.py").read_bytes())
                self.assertEqual(candidate_root.parts[-2:], candidate_result_path(ROOT / "results", operator_id, candidate_id).parts[-2:])

    def test_case_layouts_cover_tail_minimum_c4_c128_and_are_phase_isolated(self):
        all_cases = {operator_id: spec.cases() for operator_id, spec in SPECS.items()}
        for operator_id, cases in all_cases.items():
            self.assertTrue(any("smoke" in case.tags for case in cases), operator_id)
            self.assertTrue(any("representative" in case.tags and case.symbols["raw_context"] == 65536 for case in cases), operator_id)
            self.assertTrue(any("tail_page" in case.tags for case in cases), operator_id)
            self.assertTrue(any("minimum" in case.tags for case in cases), operator_id)
            for case in cases:
                contract = SPECS[operator_id].layout_contract(case)
                self.assertEqual(contract["cache_mutation"], "forbidden")
                self.assertIn("workspace_lifecycle", contract)
                self.assertIsNone(SPECS[operator_id].workspace_bytes(case))
        prefill = all_cases["deepseek_v4_sparse_prefill_attention"]
        decode = all_cases["deepseek_v4_sparse_decode_attention"]
        self.assertTrue(all("decode" not in case.tags for case in prefill))
        self.assertTrue(all("prefill" not in case.tags for case in decode))
        self.assertEqual({case.symbols["compression_ratio"] for case in decode}, {4, 128})
        empty = next(case for case in prefill if "empty_index" in case.tags)
        short = next(case for case in prefill if "short_index" in case.tags)
        self.assertEqual((empty.symbols["topk"], short.symbols["topk"]), (0, 1))
        paged_smoke = all_cases["deepseek_v4_fp8_paged_mqa_logits"][0]
        self.assertEqual((paged_smoke.symbols["heads"], paged_smoke.symbols["head_dim"]), (64, 128))
        for operator_id in (
            "deepseek_v4_sparse_prefill_attention",
            "deepseek_v4_sparse_decode_attention",
            "deepseek_v4_dense_swa_attention",
        ):
            self.assertTrue(
                all((case.symbols["heads"], case.symbols["head_dim"]) == (128, 512) for case in all_cases[operator_id]),
                operator_id,
            )

    def test_paged_contract_exercises_q_offset_reordered_pages_and_tail(self):
        spec = SPECS["deepseek_v4_fp8_paged_mqa_logits"]
        smoke = next(case for case in spec.cases() if "smoke" in case.tags)
        contract = spec.layout_contract(smoke)
        self.assertEqual(smoke.symbols["batch"], 2)
        self.assertEqual(contract["compressed_context"], 65)
        self.assertEqual(contract["tail_tokens"], 1)
        source = (ROOT / "operators" / "references" / spec.operator_id / "spec.py").read_text(encoding="utf-8")
        self.assertIn("page_table[1] = page_table[1].flip(0)", source)
        self.assertIn("logical_kv[:, compressed:] = 4.0", source)
        implementation = (ROOT / "operators" / "references" / spec.operator_id / "implementation.py").read_text(encoding="utf-8")
        self.assertIn("q_offset=q_offset", implementation)

    def test_legacy_mapping_and_cost_are_complete(self):
        paged = SPECS["deepseek_v4_fp8_paged_mqa_logits"].legacy_mappings()
        prefill = SPECS["deepseek_v4_sparse_prefill_attention"].legacy_mappings()
        decode = SPECS["deepseek_v4_sparse_decode_attention"].legacy_mappings()
        dense = SPECS["deepseek_v4_dense_swa_attention"].legacy_mappings()
        self.assertEqual((paged["backend"], paged["instances"]), ("DeepGEMM fp8_paged_mqa_logits", 30))
        self.assertEqual((prefill["backend"], prefill["instances"]), ("sgl-kernel FlashMLA sparse_fwd", 60))
        self.assertEqual({item["compression_ratio"] for item in decode}, {4, 128})
        self.assertEqual({item["phase"] for item in dense}, {"decode"})
        representative_prefill = next(case for case in SPECS["deepseek_v4_sparse_prefill_attention"].cases() if "representative" in case.tags)
        self.assertEqual((representative_prefill.symbols["query_tokens"], representative_prefill.symbols["raw_context"]), (1024, 65536))
        for operator_id, spec in SPECS.items():
            for case in spec.cases():
                cost = spec.cost_model(case)
                if "unsupported" in case.tags:
                    self.assertIsNone(cost, (operator_id, case.case_id))
                    continue
                self.assertIsNotNone(cost, (operator_id, case.case_id))
                self.assertGreater(cost["flops"], 0, (operator_id, case.case_id))
                self.assertGreater(cost["estimated_bytes"], 0, (operator_id, case.case_id))

    def test_independent_cpu_semantic_oracles(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        paged_module = load(ROOT / "operators" / "references" / "deepseek_v4_fp8_paged_mqa_logits" / "spec.py", "_mqa_logits_oracle")
        q = torch.tensor([[[1.0, 2.0], [0.5, 1.0]]])
        kv = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [9.0, 9.0]]])
        weights = torch.tensor([[1.0, 2.0]])
        self.assertTrue(torch.equal(paged_module(torch, q, kv, weights, torch.tensor([2])), torch.tensor([[2.0, 4.0, 0.0]])))
        sparse_oracle = load(ROOT / "operators" / "references" / "deepseek_v4_sparse_prefill_attention" / "spec.py", "_sparse_oracle")
        q2 = torch.tensor([[[1.0, 0.0]]])
        kv2 = torch.tensor([[[1.0, 2.0]], [[0.0, 4.0]]])
        out = sparse_oracle(torch, q2, kv2, torch.tensor([[[0, 1]]], dtype=torch.int32), 1.0, 2)
        probabilities = torch.softmax(torch.tensor([1.0, 0.0]), dim=0)
        self.assertTrue(torch.allclose(out[0, 0], probabilities[0] * kv2[0, 0] + probabilities[1] * kv2[1, 0]))
        decode_oracle = load(ROOT / "operators" / "references" / "deepseek_v4_sparse_decode_attention" / "spec.py", "_selected_attention_oracle")
        out2 = decode_oracle(torch, q2.view(1, 1, 1, 2), kv2[:, 0], torch.tensor([[[0, 1]]]), torch.tensor([1]))
        self.assertTrue(torch.equal(out2[0, 0, 0], kv2[0, 0]))

    def test_sparse_prefill_wrapper_unpacks_list_and_tuple_without_cuda(self):
        state = {"result": None}
        flash_module = types.ModuleType("sgl_kernel.flash_mla")
        flash_module.flash_mla_sparse_fwd = lambda *_args, **_kwargs: state["result"]
        package = types.ModuleType("sgl_kernel")
        package.__path__ = []
        package.flash_mla = flash_module
        implementation = ROOT / "operators" / "references" / "deepseek_v4_sparse_prefill_attention" / "implementation.py"
        with mock.patch.dict(sys.modules, {"sgl_kernel": package, "sgl_kernel.flash_mla": flash_module}):
            operator = load(implementation, "operator")

        class FakeTensor:
            shape = (1,)
            def detach(self):
                return self

        output = FakeTensor()
        for result in ([output, "metadata"], (output, "metadata"), output):
            with self.subTest(container=type(result).__name__):
                state["result"] = result
                self.assertIs(operator(None, None, None, 1.0, 1), output)
        state["result"] = []
        with self.assertRaisesRegex(RuntimeError, "empty output sequence"):
            operator(None, None, None, 1.0, 1)
        state["result"] = [object()]
        with self.assertRaisesRegex(TypeError, "must be a tensor"):
            operator(None, None, None, 1.0, 1)

    def test_all_sampling_helpers_use_the_same_bounded_indices(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        source = torch.arange(1000, dtype=torch.float32)
        for operator_id in OPERATORS:
            root = ROOT / "operators" / "references" / operator_id / "spec.py"
            with self.subTest(operator=operator_id):
                sample_indices = load(root, "_sample_indices")
                bounded_output = load(root, "_bounded_output")
                indices = sample_indices(torch, source.numel(), source.device)
                normalized = bounded_output(source).by_path()["output"]
                self.assertEqual(indices.numel(), 256)
                self.assertEqual(len(normalized.value), 256)
                self.assertEqual(normalized.value, tuple(source[indices].tolist()))

    def test_sampling_indices_are_int64_monotonic_and_safe_above_float32_range(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        total = 67_108_864
        expected = None
        for operator_id in OPERATORS:
            root = ROOT / "operators" / "references" / operator_id / "spec.py"
            with self.subTest(operator=operator_id):
                sample_indices = load(root, "_sample_indices")
                indices = sample_indices(torch, total, torch.device("cpu"))
                self.assertEqual(indices.dtype, torch.int64)
                self.assertLessEqual(indices.numel(), 256)
                self.assertEqual(indices[0].item(), 0)
                self.assertEqual(indices[-1].item(), total - 1)
                self.assertTrue(torch.all(indices >= 0).item())
                self.assertTrue(torch.all(indices < total).item())
                self.assertTrue(torch.all(indices[1:] >= indices[:-1]).item())
                self.assertEqual(sample_indices(torch, 0, torch.device("cpu")).numel(), 0)
                self.assertEqual(sample_indices(torch, 1, torch.device("cpu")).tolist(), [0])
                values = indices.tolist()
                if expected is None:
                    expected = values
                else:
                    self.assertEqual(values, expected)

    def test_padded_decode_cache_clone_preserves_576_byte_stride(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        decode_root = ROOT / "operators" / "references" / "deepseek_v4_sparse_decode_attention" / "spec.py"
        make_decode_cache = load(decode_root, "_padded_cache")
        clone_decode_cache = load(decode_root, "_clone_padded_cache")
        for page_size in (2, 64, 128):
            with self.subTest(kind="decode", page_size=page_size):
                cache = make_decode_cache(torch, 2, page_size, torch.device("cpu"))
                clone = clone_decode_cache(torch, cache)
                self.assertEqual(cache.stride(0) % 576, 0)
                self.assertEqual(clone.stride(0) % 576, 0)
                self.assertNotEqual(cache.untyped_storage().data_ptr(), clone.untyped_storage().data_ptr())
        dense_root = ROOT / "operators" / "references" / "deepseek_v4_dense_swa_attention" / "spec.py"
        make_dense_cache = load(dense_root, "_padded_cache")
        clone_dense_cache = load(dense_root, "_clone_padded_cache")
        cache = make_dense_cache(torch, 2, torch.device("cpu"))
        clone = clone_dense_cache(torch, cache)
        self.assertEqual(cache.stride(0) % 576, 0)
        self.assertEqual(clone.stride(0) % 576, 0)
        self.assertNotEqual(cache.untyped_storage().data_ptr(), clone.untyped_storage().data_ptr())

    def test_short_and_empty_prefill_cases_are_stably_unsupported(self):
        spec = SPECS["deepseek_v4_sparse_prefill_attention"]
        unsupported = [case for case in spec.cases() if "unsupported" in case.tags]
        self.assertEqual({case.symbols["topk"] for case in unsupported}, {0, 1})
        context = type("Context", (), {"cuda": {}})()
        for case in unsupported:
            with self.subTest(case=case.case_id):
                with self.assertRaisesRegex(NotImplementedError, "divisible by 64"):
                    spec.make_inputs(case, context)
                result = CorrectnessEvaluator(synchronizer=lambda *_: None).evaluate(
                    spec=spec,
                    reference=lambda *_args, **_kwargs: None,
                    candidate=lambda *_args, **_kwargs: None,
                    case=case,
                )
                self.assertEqual(result.status, "unsupported", result.to_dict())

    def test_unsupported_and_oom_classification_helpers(self):
        for operator_id, spec in SPECS.items():
            case = spec.cases()[0]
            bad = dict(case.symbols)
            bad["head_dim"] = 64
            with self.assertRaises(NotImplementedError, msg=operator_id):
                spec._validate(bad)
            guard = load(ROOT / "operators" / "references" / operator_id / "spec.py", "_check_available_memory")
            with self.assertRaises(MemoryError):
                guard(91, 100)
            guard(90, 100)

    def test_observed_cache_mutation_is_a_correctness_failure(self):
        def leaf(path, values, shape):
            return OutputLeaf(path, tuple(values), "float32", shape, None, "contiguous", "cpu")
        common = (
            leaf("output", (0.0,), (1,)),
            leaf("state.semantic_oracle", (0.0,), (1,)),
            leaf("state.oracle_indices", (0,), (1,)),
        )
        reference = OutputBundle(common + (leaf("state.swa_cache_head", (0,), (1,)),))
        candidate = OutputBundle(common + (leaf("state.swa_cache_head", (1,), (1,)),))
        comparator = SPECS["deepseek_v4_sparse_decode_attention"].comparator(
            SPECS["deepseek_v4_sparse_decode_attention"].cases()[0]
        )
        result = comparator.compare(reference, candidate)
        self.assertFalse(result.passed)
        self.assertTrue(any("mutated" in item.get("error", "") for item in result.diagnostics))

    def test_auto_graph_capture_fallback_is_visible(self):
        class Graph:
            @property
            def selection(self):
                return TimerSelection("cuda_graph", "cuda_graph")
            def prepare(self, fn, config):
                raise TimerUnsupportedError("addresses are not stable")
            def sample(self, fn, config):
                raise AssertionError
        class Event:
            @property
            def selection(self):
                return TimerSelection("cuda_event", "cuda_event")
            def prepare(self, fn, config):
                return 0.0
            def sample(self, fn, config):
                return (RawSample(0, 1, 1.0, 1.0),)
        timer = AutoTimer(graph_factory=Graph, event_factory=Event)
        timer.prepare(lambda: None, type("Config", (), {"samples": 1, "inner_iterations": 1})())
        self.assertEqual(timer.selection.effective_timer, "cuda_event")
        self.assertIn("addresses are not stable", timer.selection.fallback_reason)

    def test_suite_and_direct_selectors_produce_jobs(self):
        for operator_id, candidate_id in OPERATORS.items():
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = main(("run", "--suite", "smoke", "--operator", operator_id, "--candidate", candidate_id, "--dry-run"), repository_root=ROOT)
            self.assertEqual(code, 0, stderr.getvalue())
            jobs = json.loads(stdout.getvalue())["jobs"]
            self.assertTrue(jobs, operator_id)
            self.assertTrue(all(job["identity"]["operator_id"] == operator_id for job in jobs))


@unittest.skipUnless(os.environ.get("BENCHMARK_ENGINE_RUN_GPU_INTEGRATION") == "1", "set BENCHMARK_ENGINE_RUN_GPU_INTEGRATION=1 on B200")
class Phase12GpuIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA unavailable")
        cls.loaded = {}
        for operator_id, candidate_id in OPERATORS.items():
            reference_root = ROOT / "operators" / "references" / operator_id
            cls.loaded[operator_id] = (
                SPECS[operator_id],
                load(reference_root / "implementation.py", "operator"),
                load(candidate_source_path(ROOT, operator_id, candidate_id) / "implementation.py", "operator"),
                candidate_id,
            )

    def test_smoke_correctness_and_mutation_detection(self):
        evaluator = CorrectnessEvaluator()
        for operator_id, (spec, reference, candidate, candidate_id) in self.loaded.items():
            with self.subTest(operator=operator_id):
                case = next(case for case in spec.cases() if "smoke" in case.tags)
                result = evaluator.evaluate(spec=spec, reference=reference, candidate=candidate, case=case, operator_id=operator_id, candidate_id=candidate_id, cuda_devices=("cuda:0",))
                self.assertEqual(result.status, "pass", result.to_dict())
        operator_id = "deepseek_v4_sparse_decode_attention"
        spec, reference, _, _ = self.loaded[operator_id]
        wrong = load(ROOT / "tests" / "fixtures" / "deepseek_v4_sparse_decode_attention_mutates_cache.py", "operator")
        case = next(case for case in spec.cases() if "smoke" in case.tags)
        result = evaluator.evaluate(spec=spec, reference=reference, candidate=wrong, case=case, operator_id=operator_id, candidate_id="mutates_cache", cuda_devices=("cuda:0",))
        self.assertEqual(result.status, "fail", result.to_dict())
        self.assertIn("observed", json.dumps(result.to_dict()))

    def test_representative_performance_is_staged_and_sampled(self):
        for operator_id, (spec, reference, candidate, _) in self.loaded.items():
            with self.subTest(operator=operator_id):
                case = next(case for case in spec.cases() if "representative" in case.tags)
                result = PerformanceEvaluator().evaluate(spec=spec, reference=reference, candidate=candidate, case=case, config=PerformanceConfig(requested_timer="auto", warmup=2, samples=5, inner_iterations=10, maximum_cv=0.2), cuda_devices=("cuda:0",))
                self.assertIn(result.status, {"pass", "unstable"})
                self.assertEqual((len(result.reference.samples), len(result.candidate.samples)), (5, 5))
                self.assertGreaterEqual(result.reference.first_call_ms, 0)
                self.assertGreaterEqual(result.candidate.graph_capture_ms, 0)
                self.assertIsNone(result.workspace_bytes)


if __name__ == "__main__":
    unittest.main()
