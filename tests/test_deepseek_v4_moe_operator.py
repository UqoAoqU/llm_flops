from __future__ import annotations

import ast
import contextlib
import importlib.util
import inspect
import io
import json
import os
import types
import unittest
from pathlib import Path
from unittest import mock

from benchmark_engine.cli import main
from benchmark_engine.correctness import CorrectnessEvaluator, assert_input_isolation
from benchmark_engine.execution.protocol import WorkerOutcome
from benchmark_engine.execution.worker import _classify
from benchmark_engine.ids import candidate_result_path, candidate_source_path
from benchmark_engine.performance import PerformanceConfig, PerformanceEvaluator
from benchmark_engine.projection import DEEPSEEK_V4_PROJECTION
from benchmark_engine.registry import FilesystemRegistry, compute_source_hash

from deepseek_v4_benchmark import decode_adapters, local_routed_pairs as legacy_local_pairs, prefill_adapters


ROOT = Path(__file__).resolve().parents[1]
OPERATOR = "deepseek_v4_trtllm_fp8_mxfp8_moe"
CANDIDATE = "reference_control__20260717T150000Z__d4f14e00"
REFERENCE_ROOT = ROOT / "operators" / "references" / OPERATOR


def load(path: Path, attribute: str):
    spec = importlib.util.spec_from_file_location(
        f"_phase14_{path.parent.name}_{path.stem}_{id(path)}", path
    )
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, attribute)


def _unbound_operator(function):
    """Store a module function on a unittest class without method binding."""

    return staticmethod(function)


SPEC_MODULE = load(REFERENCE_ROOT / "spec.py", "SPEC")
LOCAL_ROUTED_PAIRS = load(REFERENCE_ROOT / "spec.py", "local_routed_pairs")


class Phase14ContractTests(unittest.TestCase):
    def test_gpu_fixture_operator_callable_remains_unbound_with_ten_parameters(self):
        def operator(a, b, c, d, e, f, g, h, i, j):
            return (a, b, c, d, e, f, g, h, i, j)

        class Holder:
            pass

        Holder.operator = _unbound_operator(operator)
        resolved = Holder().operator
        self.assertIs(resolved, operator)
        self.assertEqual(len(inspect.signature(resolved).parameters), 10)

    def test_registry_safe_strict_manifest_mirror_and_byte_identical_control(self):
        source = (REFERENCE_ROOT / "spec.py").read_text(encoding="utf-8")
        imported = []
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        self.assertFalse(
            any(
                name == "torch"
                or name.startswith(("torch.", "flashinfer", "sglang"))
                for name in imported
            ),
            imported,
        )
        snapshot = FilesystemRegistry(ROOT).discover()
        issues = [issue for issue in snapshot.issues if OPERATOR in str(issue.path)]
        self.assertFalse(issues, issues)
        manifest = snapshot.operator_specs[OPERATOR].manifest
        self.assertEqual((manifest.performance.timer, manifest.performance.graph_mode), ("auto", "auto"))
        self.assertEqual(manifest.performance.timeout_s, 1800)
        candidate_root = candidate_source_path(ROOT, OPERATOR, CANDIDATE)
        self.assertIn(CANDIDATE, {item.implementation_id for item in snapshot.candidates[OPERATOR]})
        self.assertEqual(
            (REFERENCE_ROOT / "implementation.py").read_bytes(),
            (candidate_root / "implementation.py").read_bytes(),
        )
        self.assertEqual(
            candidate_root.parts[-2:],
            candidate_result_path(ROOT / "results", OPERATOR, CANDIDATE).parts[-2:],
        )
        self.assertEqual(len(compute_source_hash(candidate_root)), 64)

    def test_legacy_geometry_backend_and_local_pair_formula_are_lossless(self):
        mapping = SPEC_MODULE.legacy_mappings()
        self.assertEqual(mapping["shape"], (16, 7168, 3072))
        self.assertEqual((mapping["global_experts"], mapping["topk"], mapping["instances"]), (384, 6, 61))
        self.assertEqual(mapping["backend"], "FlashInfer TRTLLM FP8 weight + MXFP8 activation")
        adapters = [
            item
            for item in prefill_adapters("fp8_mxfp8") + decode_adapters("fp8_mxfp8")
            if item.kind == "moe_fp8_mxfp8"
        ]
        self.assertEqual(len(adapters), 2)
        for adapter in adapters:
            self.assertEqual(
                (adapter.name, adapter.backend, adapter.instances, adapter.shape),
                (mapping["adapter_name"], mapping["backend"], mapping["instances"], mapping["shape"]),
            )
        for m in (0, 1, 4, 16, 32, 1024, 2048, 4096):
            self.assertEqual(LOCAL_ROUTED_PAIRS(m), legacy_local_pairs(m))

    def test_cases_layout_routes_cost_and_projection_mapping(self):
        cases = SPEC_MODULE.cases()
        projection = [case for case in cases if "model_projection" in case.tags]
        self.assertEqual(len(projection), 5)
        self.assertEqual(
            {(case.symbols["phase"], case.symbols["model_input"]) for case in projection},
            {("prefill", 1024), ("prefill", 2048), ("prefill", 4096), ("decode", 16), ("decode", 32)},
        )
        for case in cases:
            layout = SPEC_MODULE.layout_contract(case)
            self.assertEqual(layout["w13_logical"], [16, 6144, 7168])
            self.assertEqual(layout["w2_logical"], [16, 7168, 3072])
            self.assertEqual(layout["routing_ids"], [case.symbols["m"], 6])
            self.assertEqual(layout["weight_scale_block"], 32)
            self.assertEqual(layout["output"], [case.symbols["m"], 7168])
            cost = SPEC_MODULE.cost_model(case)
            if case.symbols["route_pattern"] == "no_local":
                self.assertIsNone(cost)
                continue
            self.assertIsNotNone(cost)
            self.assertGreater(cost["estimated_bytes"], 0)
            self.assertGreaterEqual(cost["flops"], 0)
        for phase in ("prefill", "decode"):
            mapping = next(
                item
                for item in DEEPSEEK_V4_PROJECTION.mappings(phase, "fp8_mxfp8")
                if item.adapter_id == "routed_expert_fused_moe"
            )
            self.assertEqual((mapping.operator_id, mapping.instances, mapping.shape), (OPERATOR, 61, (16, 7168, 3072)))
        self.assertEqual(DEEPSEEK_V4_PROJECTION.mappings("prefill", "mxfp4"), ())

    def test_routing_boundaries_validation_and_small_independent_oracle(self):
        routing_values = load(REFERENCE_ROOT / "spec.py", "_routing_values")
        validate = load(REFERENCE_ROOT / "spec.py", "_validate_routing_values")
        ids, weights = routing_values(1, "no_local")
        validate(ids, weights)
        self.assertTrue(all(value >= 16 for value in ids[0]))
        ids, weights = routing_values(1, "repeated_local")
        validate(ids, weights)
        self.assertEqual(ids[0], (0, 0, 0, 0, 0, 0))
        ids, weights = routing_values(16, "legacy_uniform_ep0")
        validate(ids, weights)
        self.assertEqual(sum(value < 16 for row in ids for value in row), legacy_local_pairs(16))
        with self.assertRaisesRegex(ValueError, "outside"):
            validate(((384, 1),), ((0.5, 0.5),), topk=2)
        with self.assertRaisesRegex(ValueError, "sum to one"):
            validate(((0, 1),), ((0.4, 0.4),), topk=2)
        with self.assertRaisesRegex(ValueError, "finite and non-negative"):
            validate(((0, 1),), ((float("nan"), 1.0),), topk=2)

        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        oracle = load(REFERENCE_ROOT / "spec.py", "_routed_mlp_oracle")
        x = torch.tensor([[1.0, 2.0]])
        w13 = torch.zeros((2, 2, 2))
        w2 = torch.zeros((2, 2, 1))
        w13[0, 0] = torch.tensor([1.0, 0.0])
        w13[0, 1] = torch.tensor([0.0, 1.0])
        w2[0, :, 0] = torch.tensor([1.0, 2.0])
        local = oracle(torch, x, torch.tensor([[0, 0]]), torch.tensor([[0.25, 0.75]]), w13, w2)
        one = oracle(torch, x, torch.tensor([[0]]), torch.tensor([[1.0]]), w13, w2)
        remote = oracle(torch, x, torch.tensor([[9]]), torch.tensor([[1.0]]), w13, w2)
        self.assertTrue(torch.allclose(local, one))
        self.assertTrue(torch.equal(remote, torch.zeros_like(remote)))

    def test_invalid_layout_profile_and_memory_are_explicit(self):
        validate = load(REFERENCE_ROOT / "spec.py", "_validate_symbols")
        base = dict(next(case.symbols for case in SPEC_MODULE.cases() if "smoke" in case.tags))
        for field, value in (
            ("hidden", 4096),
            ("intermediate", 2048),
            ("local_experts", 8),
            ("topk", 4),
            ("weight_scale_block", 128),
            ("weight_layout", "row_major"),
            ("weight_dtype", "float8_e5m2"),
            ("activation_quantization", "mxfp4"),
        ):
            with self.subTest(field=field):
                invalid = dict(base, **{field: value})
                with self.assertRaisesRegex(NotImplementedError, "unsupported"):
                    validate(invalid)
        guard = load(REFERENCE_ROOT / "spec.py", "_check_available_memory")
        with self.assertRaises(MemoryError):
            guard(91, 100)
        guard(90, 100)

    def test_runtime_modules_import_pack_before_alignment_without_real_sglang(self):
        runtime_modules = load(REFERENCE_ROOT / "spec.py", "_runtime_modules")
        calls = []
        fake_torch = object()
        pack_module = types.SimpleNamespace(
            PackTopkIds=types.SimpleNamespace(vanilla=lambda ids, weights: (ids, weights))
        )
        align = lambda layer: layer
        align_module = types.SimpleNamespace(
            align_mxfp8_moe_weights_for_flashinfer_trtllm=align
        )
        modules = {
            "torch": fake_torch,
            "sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe": pack_module,
            "sglang.srt.layers.moe.moe_runner.flashinfer_trtllm": align_module,
        }

        def import_module(name):
            calls.append(name)
            return modules[name]

        with mock.patch("importlib.import_module", side_effect=import_module):
            resolved_torch, resolved_align, resolved_pack = runtime_modules()

        self.assertIs(resolved_torch, fake_torch)
        self.assertIs(resolved_align, align)
        self.assertIs(resolved_pack, pack_module.PackTopkIds.vanilla)
        self.assertEqual(
            calls,
            [
                "torch",
                "sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe",
                "sglang.srt.layers.moe.moe_runner.flashinfer_trtllm",
            ],
        )

    def test_clone_helper_isolates_every_tensor_and_rebuilds_packed_routes(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        clone = load(REFERENCE_ROOT / "spec.py", "_clone_with_packer")
        observed = load(REFERENCE_ROOT / "spec.py", "_observed_state")
        from benchmark_engine.correctness import InputBundle

        ids = torch.tensor([[0, 1]], dtype=torch.int32)
        weights = torch.tensor([[0.5, 0.5]], dtype=torch.float32)
        w13 = torch.ones((2, 2, 2))
        s13 = torch.ones((2, 2, 1), dtype=torch.uint8)
        w2 = torch.ones((2, 2, 1))
        s2 = torch.ones((2, 2, 1), dtype=torch.uint8)
        original = InputBundle(
            args=(torch.ones((1, 2)), (ids, weights), ids, weights, w13, s13, w2, s2, torch.empty((1, 2)), 1),
            observed_state=observed(ids, weights, w13, s13, w2, s2),
        )
        cloned = clone(original, lambda left, right: (left, right))
        assert_input_isolation(original, cloned)
        self.assertIsNot(original.args[1], cloned.args[1])
        self.assertNotEqual(original.args[4].untyped_storage().data_ptr(), cloned.args[4].untyped_storage().data_ptr())
        self.assertNotEqual(original.observed_state["routing_ids_head"].untyped_storage().data_ptr(), cloned.observed_state["routing_ids_head"].untyped_storage().data_ptr())

    def test_large_output_is_bounded_and_detects_a_wrong_value(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        tensor = torch.arange(4 * 7168, dtype=torch.bfloat16).reshape(4, 7168)
        reference = SPEC_MODULE.normalize_output(tensor)
        control = SPEC_MODULE.normalize_output(tensor.clone())
        wrong = tensor.clone()
        wrong.reshape(-1)[-1] = -1000.0
        wrong = SPEC_MODULE.normalize_output(wrong)
        self.assertLessEqual(sum(leaf.size for leaf in reference.leaves), 256)
        case = next(case for case in SPEC_MODULE.cases() if "smoke" in case.tags)
        self.assertTrue(SPEC_MODULE.comparator(case).compare(reference, control).passed)
        self.assertFalse(SPEC_MODULE.comparator(case).compare(reference, wrong).passed)

    def test_import_jit_and_error_classification_are_outside_timed_operator(self):
        source = (REFERENCE_ROOT / "implementation.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        operator = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "operator")
        operator_names = {
            node.func.id
            for node in ast.walk(operator)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        self.assertNotIn("import_module", operator_names)
        self.assertNotIn("_load_backend_symbols", operator_names)
        self.assertIn("_load_backend_symbols()", source)
        self.assertIn("unsupported: missing FlashInfer symbol(s)", source)
        self.assertIn("FlashInfer import/build error", source)
        self.assertEqual(_classify(NotImplementedError("missing symbol")), WorkerOutcome.UNSUPPORTED)
        self.assertEqual(_classify(RuntimeError("FlashInfer import/build error")), WorkerOutcome.ERROR)
        self.assertEqual(_classify(MemoryError("out of memory")), WorkerOutcome.OOM)

    def test_suite_dry_runs_add_exact_prefill_and_decode_jobs(self):
        for suite, expected, phase in (
            ("deepseek_v4_prefill", 17, "prefill"),
            ("deepseek_v4_decode", 17, "decode"),
        ):
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = main(("run", "--suite", suite, "--dry-run"), repository_root=ROOT)
            self.assertEqual(code, 0, stderr.getvalue())
            jobs = json.loads(stdout.getvalue())["jobs"]
            self.assertEqual(len(jobs), expected)
            moe_jobs = [
                job
                for job in jobs
                if job["identity"]["operator_id"]
                == "deepseek_v4_aiter_fp8_fused_moe"
            ]
            self.assertEqual(len(moe_jobs), 2)
            self.assertEqual({job["case"]["symbols"]["phase"] for job in moe_jobs}, {phase})


@unittest.skipUnless(
    os.environ.get("BENCHMARK_ENGINE_RUN_GPU_INTEGRATION") == "1",
    "set BENCHMARK_ENGINE_RUN_GPU_INTEGRATION=1 on B200",
)
class Phase14GpuIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA unavailable")
        capability = torch.cuda.get_device_capability(0)
        if capability < (10, 0):
            raise unittest.SkipTest(f"FlashInfer TRTLLM FP8/MXFP8 requires SM100, got {capability}")
        cls.reference = _unbound_operator(
            load(REFERENCE_ROOT / "implementation.py", "operator")
        )
        cls.candidate = _unbound_operator(
            load(
                candidate_source_path(ROOT, OPERATOR, CANDIDATE) / "implementation.py",
                "operator",
            )
        )

    def test_real_flashinfer_symbol_correctness_wrong_output_and_mutation(self):
        evaluator = CorrectnessEvaluator()
        smoke = next(case for case in SPEC_MODULE.cases() if "smoke" in case.tags)
        result = evaluator.evaluate(
            spec=SPEC_MODULE,
            reference=self.reference,
            candidate=self.candidate,
            case=smoke,
            operator_id=OPERATOR,
            candidate_id=CANDIDATE,
            cuda_devices=("cuda:0",),
        )
        self.assertEqual(result.status, "pass", result.to_dict())
        for fixture in (
            "deepseek_v4_trtllm_fp8_mxfp8_moe_wrong.py",
            "deepseek_v4_trtllm_fp8_mxfp8_moe_mutates_weight.py",
        ):
            wrong = load(ROOT / "tests" / "fixtures" / fixture, "operator")
            failed = evaluator.evaluate(
                spec=SPEC_MODULE,
                reference=self.reference,
                candidate=wrong,
                case=smoke,
                operator_id=OPERATOR,
                candidate_id="wrong_fixture",
                cuda_devices=("cuda:0",),
            )
            self.assertEqual(failed.status, "fail", failed.to_dict())

    def test_representative_cuda_graph_raw_samples_and_cost(self):
        case = next(
            case
            for case in SPEC_MODULE.cases()
            if case.symbols.get("phase") == "decode" and case.symbols.get("model_input") == 16
        )
        result = PerformanceEvaluator().evaluate(
            spec=SPEC_MODULE,
            reference=self.reference,
            candidate=self.candidate,
            case=case,
            config=PerformanceConfig(
                requested_timer="cuda_graph",
                warmup=2,
                samples=5,
                inner_iterations=1,
                maximum_cv=0.15,
            ),
            cuda_devices=("cuda:0",),
        )
        self.assertEqual(result.status, "pass", result.to_dict())
        self.assertEqual((len(result.reference.samples), len(result.candidate.samples)), (5, 5))
        self.assertEqual(result.reference.selection.effective_timer, "cuda_graph")
        self.assertEqual(result.candidate.selection.effective_timer, "cuda_graph")
        self.assertGreaterEqual(result.reference.first_call_ms, 0.0)
        self.assertGreaterEqual(result.candidate.first_call_ms, 0.0)
        self.assertGreaterEqual(result.reference.graph_capture_ms, 0.0)
        self.assertGreaterEqual(result.candidate.graph_capture_ms, 0.0)
        self.assertTrue(result.cost.available)
        self.assertGreater(result.cost.tflops, 0.0)


if __name__ == "__main__":
    unittest.main()
