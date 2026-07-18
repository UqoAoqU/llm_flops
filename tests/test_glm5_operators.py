from __future__ import annotations

import ast
import importlib
import importlib.util
import random
import unittest
from pathlib import Path

import yaml

from benchmark_engine.operator_spec import load_operator_cases
from benchmark_engine.registry import FilesystemRegistry
from benchmark_engine.registry.validation import parse_operator_manifest


ROOT = Path(__file__).resolve().parents[1]
RUNNABLE = {
    "glm5_dsa_indexer": "reference_control__20260718T080000Z__16000001",
    "glm5_dsa_projection": "reference_control__20260718T080000Z__16000002",
    "glm5_dsa_sparse_attention": "reference_control__20260718T080000Z__16000003",
    "glm5_dense_prefill_attention": "reference_control__20260718T080000Z__16000004",
    "glm5_moe_grouped_gemm": "reference_control__20260718T080000Z__16000005",
    "glm5_dsa_index_score": "reference_control__20260718T080000Z__16000007",
    "glm5_moe_masked_grouped_gemm": "reference_control__20260718T080000Z__16000008",
    "glm5_dsa_unified_sparse_attention": "reference_control__20260718T080000Z__16000009",
}
DEEPEP = "glm5_deepep_dispatch"
CANDIDATES = dict(RUNNABLE, **{DEEPEP: "reference_control__20260718T080000Z__16000006"})
SUITE_OPERATORS = set(RUNNABLE) - {"glm5_dense_prefill_attention"}


def load(path: Path, attribute: str):
    spec = importlib.util.spec_from_file_location(f"_glm5_{path.parent.name}_{id(path)}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, attribute)


class Glm5OperatorContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.snapshot = FilesystemRegistry(ROOT).discover()

    def test_all_contracts_are_registry_safe_and_have_one_control_candidate(self):
        expected = set(CANDIDATES)
        discovered = {operator_id for operator_id in self.snapshot.references if operator_id.startswith("glm5_")}
        self.assertEqual(discovered, expected)
        self.assertFalse([issue for issue in self.snapshot.issues if "glm5_" in str(issue.path)], self.snapshot.issues)
        for operator_id in expected:
            with self.subTest(operator=operator_id):
                root = ROOT / "operators" / "references" / operator_id
                tree = ast.parse((root / "spec.py").read_text(encoding="utf-8"))
                imports = [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
                self.assertNotIn("torch", imports)
                manifest = parse_operator_manifest(root / "operator.yaml")
                self.assertEqual(manifest.operator_id, operator_id)
                self.assertEqual(manifest.device_types, ("cuda",))
                self.assertTrue(load_operator_cases(self.snapshot, operator_id))

    def test_control_candidates_are_byte_identical_to_references(self):
        for operator_id, candidate_id in CANDIDATES.items():
            with self.subTest(operator=operator_id):
                reference = ROOT / "operators" / "references" / operator_id / "implementation.py"
                candidate = ROOT / "operators" / "candidates" / operator_id / candidate_id / "implementation.py"
                self.assertEqual(candidate.read_bytes(), reference.read_bytes())

    def test_five_legacy_op_characterizations_are_lossless(self):
        expectations = {
            "glm5_dsa_indexer": ("dsa_indexer.py", "glm5_dsa_indexer_perf.csv", 48),
            "glm5_dsa_sparse_attention": ("dsa_flashmla.py", "glm5_sparse_prefill_perf.csv", 16),
            "glm5_dsa_projection": ("dsa_projection.py", "glm5_attention_gemm_perf.csv", 24),
            "glm5_dense_prefill_attention": ("mla_flashmla.py", "glm5_dense_prefill_perf.csv", 16),
            "glm5_moe_grouped_gemm": ("moe_deepgemm.py", "glm5_moe_deepgemm_perf.csv", 15),
        }
        for operator_id, (script, csv_name, full_count) in expectations.items():
            spec = load(ROOT / "operators" / "references" / operator_id / "spec.py", "SPEC")
            mapping = spec.legacy_mappings()
            self.assertEqual(mapping["script"], script)
            self.assertEqual(mapping["csv"], csv_name)
            self.assertEqual(sum("legacy_full" in case.tags for case in spec.cases()), full_count)
            self.assertTrue((ROOT / script).is_file())

    def test_moe_legacy_distributions_and_physical_cost_match_old_formula(self):
        spec = load(ROOT / "operators" / "references" / "glm5_moe_grouped_gemm" / "spec.py", "SPEC")
        legacy = [case for case in spec.cases() if "legacy_full" in case.tags]
        expected_counts = {
            0: (17, 14, 16, 21, 19, 20, 6, 15),
            1: (19, 13, 15, 19, 14, 15, 15, 18),
            2: (12, 20, 17, 16, 15, 16, 16, 16),
            3: (17, 13, 14, 21, 15, 11, 18, 19),
            4: (17, 14, 18, 19, 18, 19, 12, 11),
        }
        self.assertEqual(len(legacy), 15)
        for projection in ("moe_gate_proj", "moe_up_proj", "moe_down_proj"):
            cases = [case for case in legacy if case.symbols["projection"] == projection]
            self.assertEqual({case.symbols["dist_idx"] for case in cases}, set(range(5)))
            self.assertEqual(len({tuple(case.symbols["expert_counts"]) for case in cases}), 5)
            for case in cases:
                counts = tuple(case.symbols["expert_counts"])
                self.assertEqual(counts, expected_counts[case.symbols["dist_idx"]])
                self.assertEqual(sum(counts), 128)
                physical_rows = sum(((count + 127) // 128) * 128 for count in counts if count)
                k, n = case.symbols["k"], case.symbols["n"]
                cost = spec.cost_model(case)
                self.assertEqual(cost["flops"], 2 * physical_rows * k * n)
                self.assertEqual(cost["estimated_bytes"],
                                 physical_rows * k + 8 * n * k + 2 * physical_rows * n)
                self.assertEqual(cost["throughput_units"], physical_rows)
                self.assertGreater(physical_rows, sum(counts))

    def test_masked_moe_cost_uses_logical_assignments_not_contiguous_padding(self):
        spec = load(ROOT / "operators" / "references" / "glm5_moe_masked_grouped_gemm" / "spec.py", "SPEC")
        case = next(case for case in spec.cases() if case.case_id == "smoke_masked_gate_logical16")
        cost = spec.cost_model(case)
        k, n = case.symbols["k"], case.symbols["n"]
        self.assertEqual(cost["flops"], 2 * 16 * k * n)
        self.assertEqual(cost["estimated_bytes"], 16 * k + 8 * n * k + 2 * 16 * n)
        self.assertEqual(cost["throughput_units"], 16)

    def test_masked_moe_fixed_random_routing_and_safe_capacity(self):
        spec = load(ROOT / "operators" / "references" /
                    "glm5_moe_masked_grouped_gemm" / "spec.py", "SPEC")
        cases = [case for case in spec.cases() if "model_projection" in case.tags]
        self.assertEqual(len(cases), 27)
        self.assertEqual(len({case.symbols["distribution_seed"] for case in cases}), 27)
        distributions = set()
        for case in cases:
            seed = case.symbols["distribution_seed"]
            rows = case.symbols["total_rows"]
            rng = random.Random(seed)
            expected = [0] * 8
            for _ in range(rows):
                expected[rng.randint(0, 7)] += 1
            counts = tuple(case.symbols["expert_counts"])
            self.assertEqual(counts, tuple(expected))
            self.assertEqual(sum(counts), rows)
            self.assertGreater(len(set(counts)), 1)
            legacy_expected_m = (((rows + 7) // 8 + 127) // 128) * 128
            safe_expected_m = ((max(counts) + 127) // 128) * 128
            self.assertEqual(case.symbols["legacy_expected_m"], legacy_expected_m)
            self.assertEqual(case.symbols["safe_expected_m"], safe_expected_m)
            self.assertGreaterEqual(safe_expected_m, max(counts))
            distributions.add(counts)
        self.assertGreater(len(distributions), 9)
        reproduced = next(case for case in cases
                          if case.case_id == "prefill__moe_gate_proj__m1024__ctx65536")
        self.assertEqual(reproduced.symbols["distribution_seed"], 434010241)
        self.assertEqual(tuple(reproduced.symbols["expert_counts"]),
                         (951, 1056, 1037, 1074, 1010, 1020, 1041, 1003))
        self.assertEqual(reproduced.symbols["legacy_expected_m"], 1024)
        self.assertEqual(reproduced.symbols["safe_expected_m"], 1152)

    def test_indexer_cost_bytes_follow_backend_dtypes(self):
        spec = load(ROOT / "operators" / "references" / "glm5_dsa_indexer" / "spec.py", "SPEC")
        bf16 = next(case for case in spec.cases()
                    if case.case_id == "prefill__index_weights_proj__m1024__ctx65536")
        gm, k, n = (bf16.symbols[name] for name in ("gemm_m", "k", "n"))
        self.assertEqual(spec.cost_model(bf16)["estimated_bytes"],
                         2 * gm * k + 2 * n * k + 4 * gm * n)
        fp8 = next(case for case in spec.cases()
                   if case.case_id == "prefill__index_q_upproj__m1024__ctx65536")
        gm, k, n = (fp8.symbols[name] for name in ("gemm_m", "k", "n"))
        self.assertEqual(spec.cost_model(fp8)["estimated_bytes"],
                         gm * k + n * k + 2 * gm * n)

    def test_timer_contracts_preserve_legacy_and_unified_shapes(self):
        expected = {
            "glm5_dsa_indexer": ("cuda_graph", "enabled", 20, 5, 10),
            "glm5_dsa_projection": ("cuda_graph", "enabled", 20, 5, 10),
            "glm5_dsa_index_score": ("cuda_graph", "enabled", 20, 5, 10),
            "glm5_dsa_unified_sparse_attention": ("cuda_graph", "enabled", 20, 5, 10),
            "glm5_moe_masked_grouped_gemm": ("cuda_graph", "enabled", 20, 5, 10),
            "glm5_dsa_sparse_attention": ("cuda_graph", "enabled", 1, 5, 20),
            "glm5_dense_prefill_attention": ("cuda_graph", "enabled", 1, 5, 20),
            "glm5_moe_grouped_gemm": ("cuda_event", "disabled", 1, 5, 20),
        }
        for operator_id, contract in expected.items():
            with self.subTest(operator=operator_id):
                manifest = yaml.safe_load((ROOT / "operators" / "references" / operator_id /
                                           "operator.yaml").read_text(encoding="utf-8"))
                performance = manifest["performance"]
                actual = (performance["timer"], performance["graph_mode"],
                          performance["inner_iterations"], performance["warmup"],
                          performance["samples"])
                self.assertEqual(actual, contract)
        regression = yaml.safe_load((ROOT / "suites" / "glm5_regression.yaml").read_text(encoding="utf-8"))
        self.assertEqual(regression["performance"], {})

    def test_suite_tags_cover_each_supported_family_once(self):
        for tag in ("glm5_smoke", "glm5_regression"):
            selected = {}
            for operator_id in SUITE_OPERATORS:
                spec = load(ROOT / "operators" / "references" / operator_id / "spec.py", "SPEC")
                selected[operator_id] = [case.case_id for case in spec.cases() if tag in case.tags]
            self.assertEqual(set(selected), SUITE_OPERATORS)
            self.assertTrue(all(len(cases) == 1 for cases in selected.values()), selected)

    def test_deepep_is_complete_and_explicitly_unsupported(self):
        spec = load(ROOT / "operators" / "references" / DEEPEP / "spec.py", "SPEC")
        cases = spec.cases()
        self.assertEqual(len(cases), 6 * 4)
        self.assertTrue(all("unsupported" in case.tags and "multi_gpu" in case.tags for case in cases))
        mapping = spec.legacy_mappings()
        self.assertEqual(mapping["local_gpus"], 8)
        self.assertIn("excluded", mapping["unsupported_reason"])
        with self.assertRaisesRegex(NotImplementedError, "8 local GPUs"):
            spec.make_inputs(cases[0], object())

    def test_large_output_sampling_indices_stay_in_bounds(self):
        torch = importlib.import_module("torch")
        total = 33_554_432
        for operator_id in RUNNABLE:
            with self.subTest(operator=operator_id):
                sampler = load(
                    ROOT / "operators" / "references" / operator_id / "spec.py",
                    "_sample_indices",
                )
                indices = sampler(torch, total, "cpu")
                self.assertEqual(indices.numel(), 256)
                self.assertEqual(int(indices[0]), 0)
                self.assertEqual(int(indices[-1]), total - 1)
                self.assertTrue(bool(torch.all(indices[1:] > indices[:-1])))


if __name__ == "__main__":
    unittest.main()
