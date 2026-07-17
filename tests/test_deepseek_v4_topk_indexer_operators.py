from __future__ import annotations

import ast
import contextlib
import importlib.util
import io
import json
import os
import unittest
from dataclasses import replace
from pathlib import Path

from benchmark_engine.cli import main
from benchmark_engine.correctness import CorrectnessEvaluator
from benchmark_engine.ids import candidate_result_path, candidate_source_path
from benchmark_engine.performance import PerformanceConfig, PerformanceEvaluator
from benchmark_engine.registry import FilesystemRegistry, compute_source_hash

from deepseek_v4_benchmark import decode_adapters, prefill_adapters


ROOT = Path(__file__).resolve().parents[1]
TOPK = "deepseek_v4_topk_transform"
QUANT = "deepseek_v4_indexer_fp8_quant"
TOPK_CANDIDATE = "reference_control__20260717T043000Z__50af9eec"
QUANT_CANDIDATE = "reference_control__20260717T043000Z__a39aa41c"


def load(path: Path, attribute: str):
    spec = importlib.util.spec_from_file_location(f"_phase11_{path.stem}_{id(path)}", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, attribute)


TOPK_ROOT = ROOT / "operators" / "references" / TOPK
QUANT_ROOT = ROOT / "operators" / "references" / QUANT
TOPK_SPEC = load(TOPK_ROOT / "spec.py", "SPEC")
QUANT_SPEC = load(QUANT_ROOT / "spec.py", "SPEC")


class Phase11ContractTests(unittest.TestCase):
    def test_registry_safe_specs_and_formal_manifests(self):
        for operator_id, root in ((TOPK, TOPK_ROOT), (QUANT, QUANT_ROOT)):
            with self.subTest(operator=operator_id):
                source = (root / "spec.py").read_text(encoding="utf-8")
                imported = []
                for node in ast.walk(ast.parse(source)):
                    if isinstance(node, ast.Import):
                        imported.extend(alias.name for alias in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module:
                        imported.append(node.module)
                self.assertFalse(any(name == "torch" or name.startswith("torch.") for name in imported))
                self.assertFalse(any(name == "sglang" or name.startswith("sglang.") for name in imported))
        snapshot = FilesystemRegistry(ROOT).discover()
        self.assertFalse(
            [issue for issue in snapshot.issues if TOPK in str(issue.path) or QUANT in str(issue.path)],
            snapshot.issues,
        )
        for operator_id in (TOPK, QUANT):
            manifest = snapshot.operator_specs[operator_id].manifest
            self.assertEqual(manifest.performance.graph_mode, "enabled")
            self.assertEqual(manifest.performance.timer, "cuda_graph")
            self.assertEqual(manifest.performance.regression_threshold_pct, 5.0)

    def test_candidate_hashes_and_mirrored_paths(self):
        for operator_id, candidate_id in ((TOPK, TOPK_CANDIDATE), (QUANT, QUANT_CANDIDATE)):
            root = candidate_source_path(ROOT, operator_id, candidate_id)
            self.assertTrue(compute_source_hash(root).startswith(candidate_id.rsplit("__", 1)[1]))
            result = candidate_result_path(ROOT / "results", operator_id, candidate_id)
            self.assertEqual(root.parts[-2:], result.parts[-2:])
            candidate_bytes = (root / "implementation.py").read_bytes()
            reference_bytes = (ROOT / "operators" / "references" / operator_id / "implementation.py").read_bytes()
            self.assertEqual(candidate_bytes, reference_bytes)

    def test_topk_cases_contract_cost_and_batch_safe_page_tables(self):
        cases = TOPK_SPEC.cases()
        tags = {tag for case in cases for tag in case.tags}
        self.assertTrue({"smoke", "boundary", "adversarial", "representative"} <= tags)
        tie = next(case for case in cases if case.symbols["pattern"] == "tie")
        self.assertGreater(int(tie.symbols["seq_len"]), int(tie.symbols["topk"]))
        expected_boundaries = {
            "boundary_k1": (68, 65),
            "boundary_cutoff_tie": (132, 129),
            "boundary_all_negative": (132, 129),
            "boundary_tail_page": (196, 191),
        }
        for case_id, (length, seq_len) in expected_boundaries.items():
            case = next(item for item in cases if item.case_id == case_id)
            self.assertEqual((case.symbols["length"], case.symbols["seq_len"]), (length, seq_len))
            self.assertEqual(length % 4, 0)
        representative = next(case for case in cases if "representative" in case.tags)
        self.assertEqual(
            tuple(representative.symbols[name] for name in ("batch", "length", "topk", "page_size")),
            (16, 65536, 1024, 64),
        )
        for case in cases:
            cost = TOPK_SPEC.cost_model(case)
            expected = int(case.symbols["batch"]) * int(case.symbols["seq_len"])
            self.assertEqual(cost["flops"], expected)
            self.assertGreater(cost["estimated_bytes"], 0)
        with self.assertRaisesRegex(ValueError, "seq_len"):
            TOPK_SPEC._validate({"batch": 1, "length": 8, "seq_len": -1, "topk": 1, "page_size": 4})

    def test_topk_legacy_mapping_is_lossless(self):
        mapping = TOPK_SPEC.legacy_mappings()
        self.assertEqual(mapping["backend"], "SGLang JIT topk_transform_512_v2")
        self.assertEqual(mapping["instances"], 30)
        self.assertEqual(mapping["prefill_m"], (1024, 2048, 4096))
        self.assertEqual(mapping["decode_m"], (16, 32))
        self.assertEqual((mapping["context"], mapping["topk"], mapping["page_size"]), (65536, 1024, 64))
        expected = [a for a in prefill_adapters() if a.kind == "topk"] + [a for a in decode_adapters() if a.kind == "topk"]
        self.assertEqual(len(expected), 2)
        self.assertTrue(all(a.name == mapping["adapter_name"] and a.backend == mapping["backend"] and a.instances == mapping["instances"] for a in expected))

    def test_indexer_cases_formula_cost_and_lossless_mapping(self):
        cases = QUANT_SPEC.cases()
        representative = next(case for case in cases if "representative" in case.tags)
        self.assertEqual((representative.symbols["batch"], representative.symbols["context"], representative.symbols["position"]), (16, 65536, 65535))
        self.assertTrue(any(case.symbols["pattern"] == "zero" for case in cases))
        self.assertTrue(any(case.symbols["weight_layout"] == "noncontiguous" for case in cases))
        for case in cases:
            cost = QUANT_SPEC.cost_model(case)
            self.assertGreater(cost["flops"], 0)
            self.assertGreater(cost["estimated_bytes"], 0)
        mapping = QUANT_SPEC.legacy_mappings()
        self.assertEqual(mapping["backend"], "SGLang fused RoPE/Hadamard FP8")
        self.assertEqual(mapping["instances"], 30)
        self.assertEqual((mapping["heads"], mapping["head_dim"], mapping["context"]), (64, 128, 65536))
        self.assertAlmostEqual(mapping["weight_scale"], 128**-0.5 * 64**-0.5)
        expected = [a for a in prefill_adapters("fp8_mxfp8") if a.kind == "fp8_quant"] + [a for a in decode_adapters("fp8_mxfp8") if a.kind == "fp8_quant"]
        self.assertEqual(len(expected), 2)
        self.assertTrue(all(a.name == mapping["adapter_name"] and a.backend == mapping["backend"] and a.instances == mapping["instances"] for a in expected))

    def test_jit_is_import_time_and_outside_timed_operator(self):
        topk_reference = (TOPK_ROOT / "implementation.py").read_text(encoding="utf-8")
        quant_reference = (QUANT_ROOT / "implementation.py").read_text(encoding="utf-8")
        self.assertIn("_jit_topk_v2_module()", topk_reference)
        self.assertIn("_jit_main_q_indexer_rope_hadamard_quant_module(torch.bfloat16)", quant_reference)
        self.assertIn("if not weight.is_contiguous()", quant_reference)
        self.assertIn("weight = weight.contiguous()", quant_reference)
        for source in (topk_reference, quant_reference):
            tree = ast.parse(source)
            operator = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "operator")
            self.assertFalse(any(isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id.startswith("_jit_") for node in ast.walk(operator)))

    def test_documented_commands_produce_nonempty_dry_run(self):
        for operator_id, candidate in ((TOPK, "reference_control__*"), (QUANT, "reference_control__*")):
            for arguments in (
                ("run", "--suite", "smoke", "--operator", operator_id, "--candidate", candidate, "--dry-run"),
                ("run", "--suite", "full", "--mode", "correctness", "--operator", operator_id, "--candidate", candidate, "--dry-run"),
                ("run", "--suite", "regression", "--mode", "all", "--operator", operator_id, "--candidate", candidate, "--tag", "representative", "--dry-run"),
            ):
                stdout, stderr = io.StringIO(), io.StringIO()
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    code = main(arguments, repository_root=ROOT)
                self.assertEqual(code, 0, stderr.getvalue())
                self.assertTrue(json.loads(stdout.getvalue())["jobs"])

        combined_commands = (
            (
                "run", "--suite", "full", "--mode", "correctness",
                "--operator", TOPK, "--operator", QUANT,
                "--candidate", "reference_control__*", "--dry-run",
            ),
            (
                "run", "--suite", "regression", "--mode", "all",
                "--operator", TOPK, "--operator", QUANT,
                "--candidate", "reference_control__*",
                "--tag", "representative", "--dry-run",
            ),
        )
        for arguments in combined_commands:
            stdout, stderr = io.StringIO(), io.StringIO()
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                code = main(arguments, repository_root=ROOT)
            self.assertEqual(code, 0, stderr.getvalue())
            jobs = json.loads(stdout.getvalue())["jobs"]
            self.assertTrue(jobs)
            self.assertEqual(
                {job["identity"]["operator_id"] for job in jobs},
                {TOPK, QUANT},
            )

    def test_noncontiguous_clone_preserves_contract(self):
        try:
            import torch
        except ImportError:
            self.skipTest("torch unavailable")
        backing = torch.arange(2 * 128, dtype=torch.bfloat16).view(2, 128)
        weight = backing[:, ::2]
        bundle = type("Bundle", (), {"args": (torch.zeros(2, 64, 128), weight, 1.0, torch.zeros(2, 32, dtype=torch.complex64), torch.zeros(2, dtype=torch.int32))})()
        clone = QUANT_SPEC.clone_inputs(bundle)
        self.assertFalse(clone.args[1].is_contiguous())
        self.assertEqual(clone.args[1].stride(), (128, 2))
        self.assertNotEqual(clone.args[1].untyped_storage().data_ptr(), weight.untyped_storage().data_ptr())


@unittest.skipUnless(os.environ.get("BENCHMARK_ENGINE_RUN_GPU_INTEGRATION") == "1", "set BENCHMARK_ENGINE_RUN_GPU_INTEGRATION=1 on B200")
class Phase11GpuIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        if not torch.cuda.is_available():
            raise unittest.SkipTest("CUDA unavailable")
        cls.operators = {}
        for operator_id, spec, root, candidate_id in (
            (TOPK, TOPK_SPEC, TOPK_ROOT, TOPK_CANDIDATE),
            (QUANT, QUANT_SPEC, QUANT_ROOT, QUANT_CANDIDATE),
        ):
            cls.operators[operator_id] = (
                spec,
                load(root / "implementation.py", "operator"),
                load(candidate_source_path(ROOT, operator_id, candidate_id) / "implementation.py", "operator"),
                candidate_id,
            )

    def test_all_cases_and_adversarial_wrong_fixtures(self):
        evaluator = CorrectnessEvaluator()
        for operator_id, (spec, reference, candidate, candidate_id) in self.operators.items():
            for case in spec.cases():
                result = evaluator.evaluate(spec=spec, reference=reference, candidate=candidate, case=case, operator_id=operator_id, candidate_id=candidate_id, cuda_devices=("cuda:0",))
                self.assertEqual(result.status, "pass", result.to_dict())
            wrong = load(ROOT / "tests" / "fixtures" / f"{operator_id}_wrong.py", "operator")
            smoke = next(case for case in spec.cases() if "smoke" in case.tags)
            result = evaluator.evaluate(spec=spec, reference=reference, candidate=wrong, case=smoke, operator_id=operator_id, candidate_id="wrong_fixture", cuda_devices=("cuda:0",))
            self.assertEqual(result.status, "fail", result.to_dict())

    def test_representative_cuda_graph_samples_and_cost(self):
        for operator_id, (spec, reference, candidate, _) in self.operators.items():
            case = next(case for case in spec.cases() if "representative" in case.tags)
            result = PerformanceEvaluator().evaluate(
                spec=spec,
                reference=reference,
                candidate=candidate,
                case=case,
                config=PerformanceConfig(requested_timer="cuda_graph", warmup=2, samples=10, inner_iterations=20, maximum_cv=0.15),
                cuda_devices=("cuda:0",),
            )
            self.assertEqual(result.status, "pass", result.to_dict())
            self.assertTrue(result.formal)
            self.assertEqual((len(result.reference.samples), len(result.candidate.samples)), (10, 10))
            self.assertLessEqual(result.reference.statistics.cv, 0.15)
            self.assertLessEqual(result.candidate.statistics.cv, 0.15)
            self.assertIsNotNone(result.speedup)
            self.assertGreaterEqual(result.speedup, 0.95)
            self.assertLessEqual(result.speedup, 1.05)
            self.assertTrue(result.cost.available)
            self.assertGreater(result.cost.tflops, 0.0)

    def test_indexer_noncontiguous_weight_seed_one_raw_codes_are_exact(self):
        spec, reference, candidate, candidate_id = self.operators[QUANT]
        declared = next(case for case in spec.cases() if case.case_id == "boundary_noncontiguous_weight")
        case = replace(declared, seed=1)
        result = CorrectnessEvaluator().evaluate(
            spec=spec,
            reference=reference,
            candidate=candidate,
            case=case,
            operator_id=QUANT,
            candidate_id=candidate_id,
            cuda_devices=("cuda:0",),
        )
        self.assertEqual(result.status, "pass", result.to_dict())
        self.assertEqual(result.comparison.metrics["raw_codes"]["mismatch_count"], 0)


if __name__ == "__main__":
    unittest.main()
