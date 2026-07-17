from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import json
import os
import unittest
from pathlib import Path

from benchmark_engine.correctness import CorrectnessEvaluator
from benchmark_engine.cli import main
from benchmark_engine.ids import candidate_result_path, candidate_source_path
from benchmark_engine.operator_spec import load_operator_cases
from benchmark_engine.performance import PerformanceConfig, PerformanceEvaluator
from benchmark_engine.registry import FilesystemRegistry, compute_source_hash
from benchmark_engine.registry.validation import parse_operator_manifest

from deepseek_v4_benchmark import decode_adapters, graph_ms, prefill_adapters


ROOT = Path(__file__).resolve().parents[1]
OPERATOR_ID = "deepseek_v4_fp8_gemm_nt"
CANDIDATE_ID = "pytorch_dequant__20260717T040000Z__f2d56382"
CONTROL_CANDIDATE_ID = "test_impl__20260717T061620Z__cfb18306"
REFERENCE_ROOT = ROOT / "operators" / "references" / OPERATOR_ID
CANDIDATE_ROOT = ROOT / "operators" / "candidates" / OPERATOR_ID / CANDIDATE_ID
CONTROL_CANDIDATE_ROOT = (
    ROOT / "operators" / "candidates" / OPERATOR_ID / CONTROL_CANDIDATE_ID
)


def load(path: Path, attribute: str):
    module_spec = importlib.util.spec_from_file_location(
        f"_phase10_{path.stem}_{id(path)}", path
    )
    if module_spec is None or module_spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return getattr(module, attribute)


SPEC = load(REFERENCE_ROOT / "spec.py", "SPEC")
CANDIDATE = load(CANDIDATE_ROOT / "implementation.py", "operator")
WRONG = load(
    ROOT / "tests" / "fixtures" / "deepseek_v4_fp8_gemm_nt_wrong.py",
    "operator",
)


class DeepSeekV4Fp8GemmContractTests(unittest.TestCase):
    def test_registry_safe_spec_and_manifest_contract(self):
        tree = ast.parse((REFERENCE_ROOT / "spec.py").read_text(encoding="utf-8"))
        imported = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.append(node.module)
        self.assertFalse(
            any(name == "torch" or name.startswith("torch.") for name in imported)
        )
        self.assertFalse(
            any(
                name == "deep_gemm" or name.startswith("deep_gemm.")
                for name in imported
            )
        )

        manifest = parse_operator_manifest(REFERENCE_ROOT / "operator.yaml")
        self.assertEqual(manifest.operator_id, OPERATOR_ID)
        self.assertEqual(manifest.device_types, ("cuda",))
        self.assertEqual(manifest.performance.graph_mode, "enabled")
        self.assertEqual(manifest.performance.inner_iterations, 20)
        self.assertEqual(
            (manifest.correctness.rtol, manifest.correctness.atol), (0.01, 0.1)
        )
        snapshot = FilesystemRegistry(ROOT).discover()
        self.assertFalse(
            [issue for issue in snapshot.issues if OPERATOR_ID in str(issue.path)],
            snapshot.issues,
        )
        self.assertEqual(
            {case.case_id for case in load_operator_cases(snapshot, OPERATOR_ID)},
            {case.case_id for case in SPEC.cases()},
        )

    def test_cases_cover_smoke_boundary_and_real_legacy_shape(self):
        smoke_shapes = {
            tuple(case.symbols[name] for name in ("m", "k", "n"))
            for case in SPEC.cases()
            if "smoke" in case.tags
        }
        self.assertEqual(smoke_shapes, {(16, 128, 128), (16, 256, 256)})
        by_tag = {
            tag: case
            for case in SPEC.cases()
            for tag in case.tags
            if tag != "smoke"
        }
        self.assertEqual(
            tuple(by_tag["boundary"].symbols[name] for name in ("m", "k", "n")),
            (1, 128, 128),
        )
        self.assertEqual(
            tuple(
                by_tag["representative"].symbols[name]
                for name in ("m", "k", "n")
            ),
            (16, 7168, 2048),
        )

    def test_every_legacy_fp8_adapter_maps_losslessly(self):
        expected = []
        for phase, adapters, m_values in (
            ("prefill", prefill_adapters(), (1024, 2048, 4096)),
            ("decode", decode_adapters(), (16, 32)),
        ):
            for adapter in adapters:
                if adapter.kind == "fp8":
                    k, n = adapter.shape
                    expected.append(
                        {
                            "phase": phase,
                            "adapter_name": adapter.name,
                            "backend": adapter.backend,
                            "instances": adapter.instances,
                            "m_values": m_values,
                            "k": k,
                            "n": n,
                        }
                    )
        self.assertEqual(list(SPEC.legacy_mappings()), expected)
        self.assertEqual(
            {(row["k"], row["n"]) for row in expected},
            {
                (7168, 2048),
                (1536, 65536),
                (7168, 1024),
                (16384, 7168),
                (7168, 129280),
            },
        )

    def test_shape_scale_dtype_and_cost_contract(self):
        for case in SPEC.cases():
            m, k, n = (int(case.symbols[name]) for name in ("m", "k", "n"))
            self.assertEqual((k % 128, n % 128), (0, 0))
            cost = SPEC.cost_model(case)
            expected_bytes = (
                m * k
                + n * k
                + 4 * m * (k // 128)
                + 4 * (n // 128) * (k // 128)
                + 2 * m * n
            )
            self.assertEqual(cost["flops"], 2 * m * k * n)
            self.assertEqual(cost["estimated_bytes"], expected_bytes)
            self.assertEqual(cost["throughput_units"], m * n)
        source = (REFERENCE_ROOT / "spec.py").read_text(encoding="utf-8")
        self.assertIn('FP8_DTYPE = "float8_e4m3fn"', source)
        self.assertIn('OUTPUT_DTYPE = "bfloat16"', source)
        self.assertIn("get_mn_major_tma_aligned_tensor", source)

    def test_candidate_expands_logical_scales_without_deepgemm(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        activation = torch.ones((1, 128), dtype=torch.float32)
        weight = torch.ones((128, 128), dtype=torch.float32)
        activation_scale = torch.tensor([[2.0]], dtype=torch.float32)
        weight_scale = torch.tensor([[2.0]], dtype=torch.float32)
        output = torch.empty((1, 128), dtype=torch.bfloat16)
        observed = CANDIDATE(
            activation,
            activation_scale,
            activation_scale,
            weight,
            weight_scale,
            output,
        )
        self.assertEqual(observed.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(observed, torch.full_like(output, 512.0)))
        candidate_source = (CANDIDATE_ROOT / "implementation.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("deep_gemm", candidate_source)

    def test_formal_candidate_is_standalone_hashed_and_paths_mirror(self):
        candidate_root = (
            ROOT / "operators" / "candidates" / OPERATOR_ID / CANDIDATE_ID
        )
        candidate_source = (candidate_root / "implementation.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("deepseek_v4_benchmark", candidate_source)
        self.assertNotIn("bench_deepseek", candidate_source)
        self.assertNotIn("deep_gemm", candidate_source)
        reference_source = (REFERENCE_ROOT / "implementation.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("import deep_gemm", reference_source)
        reference_tree = ast.parse(reference_source)
        reference_operator = next(
            node
            for node in reference_tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "operator"
        )
        self.assertEqual(
            [type(statement) for statement in reference_operator.body],
            [ast.Expr, ast.Return],
        )
        source_hash = compute_source_hash(candidate_root)
        self.assertTrue(source_hash.startswith(CANDIDATE_ID.rsplit("__", 1)[1]))
        source = candidate_source_path(ROOT, OPERATOR_ID, CANDIDATE_ID)
        result = candidate_result_path(ROOT / "results", OPERATOR_ID, CANDIDATE_ID)
        self.assertEqual(source.parts[-2:], result.parts[-2:])
        self.assertEqual(candidate_root, source)

    def test_reference_copy_control_candidate_is_discoverable_and_mirrored(self):
        reference_source = (REFERENCE_ROOT / "implementation.py").read_bytes()
        candidate_source = (
            CONTROL_CANDIDATE_ROOT / "implementation.py"
        ).read_bytes()
        self.assertEqual(candidate_source, reference_source)
        snapshot = FilesystemRegistry(ROOT).discover()
        candidate = next(
            item
            for item in snapshot.candidates[OPERATOR_ID]
            if item.implementation_id == CONTROL_CANDIDATE_ID
        )
        self.assertEqual(
            candidate.source_hash, compute_source_hash(CONTROL_CANDIDATE_ROOT)
        )
        source = candidate_source_path(ROOT, OPERATOR_ID, CONTROL_CANDIDATE_ID)
        result = candidate_result_path(
            ROOT / "results", OPERATOR_ID, CONTROL_CANDIDATE_ID
        )
        self.assertEqual(source.parts[-2:], result.parts[-2:])

    def test_documented_cli_commands_have_nonempty_dry_run_plans(self):
        commands = (
            (
                "run", "--suite", "smoke", "--operator", OPERATOR_ID,
                "--candidate", "pytorch_dequant__*", "--dry-run",
            ),
            (
                "run", "--suite", "full", "--mode", "correctness",
                "--operator", OPERATOR_ID, "--candidate", "pytorch_dequant__*",
                "--dry-run",
            ),
            (
                "run", "--suite", "regression", "--mode", "all",
                "--operator", OPERATOR_ID, "--candidate", "pytorch_dequant__*",
                "--tag", "representative", "--dry-run",
            ),
        )
        expected_jobs = (2, 12, 3)
        for arguments, count in zip(commands, expected_jobs):
            with self.subTest(arguments=arguments):
                stdout, stderr = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    code = main(arguments, repository_root=ROOT)
                self.assertEqual(code, 0, stderr.getvalue())
                payload = json.loads(stdout.getvalue())
                self.assertEqual(len(payload["jobs"]), count)


@unittest.skipUnless(
    os.environ.get("BENCHMARK_ENGINE_RUN_GPU_INTEGRATION") == "1",
    "set BENCHMARK_ENGINE_RUN_GPU_INTEGRATION=1 on the B200 host",
)
class DeepSeekV4Fp8GemmGpuIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch

        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA unavailable")
        __import__("deep_gemm")
        cls.reference = staticmethod(
            load(REFERENCE_ROOT / "implementation.py", "operator")
        )
        cls.candidate = staticmethod(CANDIDATE)

    def test_runtime_input_shapes_scale_layout_and_dtypes(self):
        import torch

        case = next(case for case in SPEC.cases() if "smoke" in case.tags)
        generator = torch.Generator(device="cuda:0")
        generator.manual_seed(case.seed)
        context = type("Context", (), {"cuda": {"cuda:0": generator}})()
        bundle = SPEC.make_inputs(case, context)
        activation, logical, aligned, weight, weight_scale, output = bundle.args
        self.assertEqual((activation.shape, weight.shape, output.shape), ((16, 128), (128, 128), (16, 128)))
        self.assertEqual((logical.shape, weight_scale.shape), ((16, 1), (1, 1)))
        self.assertEqual(activation.dtype, torch.float8_e4m3fn)
        self.assertEqual(weight.dtype, torch.float8_e4m3fn)
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertEqual(logical.dtype, torch.float32)
        self.assertEqual(weight_scale.dtype, torch.float32)
        self.assertTrue(torch.equal(logical, aligned))
        self.assertNotEqual(logical.data_ptr(), aligned.data_ptr())
        scales = torch.cat((logical.reshape(-1), weight_scale.reshape(-1)))
        self.assertTrue(torch.isfinite(scales).all())
        self.assertTrue((scales > 0).all())
        mantissas, _ = torch.frexp(scales)
        self.assertTrue(torch.equal(mantissas, torch.full_like(mantissas, 0.5)))
        scale_bits = scales.contiguous().view(torch.int32)
        self.assertTrue(torch.equal(scale_bits & 0x007FFFFF, torch.zeros_like(scale_bits)))
        self.assertGreaterEqual(torch.unique(scales).numel(), 2)

    def test_optimized_baseline_candidate_and_wrong_fixture_on_all_cases(self):
        evaluator = CorrectnessEvaluator()
        for case in SPEC.cases():
            with self.subTest(case=case.case_id):
                result = evaluator.evaluate(
                    spec=SPEC,
                    reference=self.reference,
                    candidate=self.candidate,
                    case=case,
                    operator_id=OPERATOR_ID,
                    candidate_id=CANDIDATE_ID,
                    cuda_devices=("cuda:0",),
                )
                self.assertEqual(result.status, "pass", result.to_dict())

        smoke = next(case for case in SPEC.cases() if "smoke" in case.tags)
        wrong = evaluator.evaluate(
            spec=SPEC,
            reference=self.reference,
            candidate=WRONG,
            case=smoke,
            operator_id=OPERATOR_ID,
            candidate_id=CANDIDATE_ID,
            cuda_devices=("cuda:0",),
        )
        self.assertEqual(wrong.status, "fail")
        self.assertEqual(wrong.comparison.failed_path, "output")
        self.assertTrue(wrong.comparison.diagnostics)

    def test_cuda_graph_raw_samples_cost_rates_and_legacy_timer_parity(self):
        import torch

        case = next(case for case in SPEC.cases() if "representative" in case.tags)
        performance = PerformanceEvaluator().evaluate(
            spec=SPEC,
            reference=self.reference,
            candidate=self.candidate,
            case=case,
            config=PerformanceConfig(
                requested_timer="cuda_graph",
                warmup=2,
                samples=10,
                inner_iterations=20,
                maximum_cv=0.15,
            ),
            cuda_devices=("cuda:0",),
        )
        self.assertEqual(performance.status, "pass")
        self.assertTrue(performance.formal)
        self.assertEqual(len(performance.reference.samples), 10)
        self.assertEqual(len(performance.candidate.samples), 10)
        order = performance.reference.samples + performance.candidate.samples
        self.assertEqual(
            sorted(sample.order_index for sample in order), list(range(20))
        )
        self.assertEqual({sample.inner_iterations for sample in order}, {20})
        self.assertLessEqual(performance.reference.statistics.cv, 0.15)
        self.assertLessEqual(performance.candidate.statistics.cv, 0.15)
        self.assertIsNotNone(performance.speedup)
        self.assertTrue(performance.cost.available)
        self.assertGreater(performance.cost.tflops, 0.0)

        generator = torch.Generator(device="cuda:0")
        generator.manual_seed(case.seed)
        context = type("Context", (), {"cuda": {"cuda:0": generator}})()
        inputs = SPEC.clone_inputs(SPEC.make_inputs(case, context))
        legacy_ms = graph_ms(
            lambda: self.reference(*inputs.args), torch, warmup=2, runs=20
        )
        new_ms = performance.reference.statistics.median_ms
        relative_delta = abs(new_ms - legacy_ms) / legacy_ms
        evidence = {
            "phase10_timer_parity": {
                "case_id": case.case_id,
                "new_reference_cuda_graph_median_ms": new_ms,
                "legacy_graph_ms": legacy_ms,
                "relative_delta": relative_delta,
                "allowed_relative_delta": 0.35,
                "speedup": performance.speedup,
                "reference_cv": performance.reference.statistics.cv,
                "candidate_cv": performance.candidate.statistics.cv,
                "candidate_tflops": performance.cost.tflops,
            }
        }
        print(json.dumps(evidence, sort_keys=True))
        self.assertLessEqual(relative_delta, 0.35, evidence)


if __name__ == "__main__":
    unittest.main()
