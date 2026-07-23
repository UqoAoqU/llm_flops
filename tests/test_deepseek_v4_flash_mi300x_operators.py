from __future__ import annotations

import importlib.util
import unittest
from collections import Counter
from pathlib import Path

import yaml

from benchmark_engine.registry import FilesystemRegistry


ROOT = Path(__file__).resolve().parents[1]
HIGH_PRIORITY = (
    "deepseek_v4_aiter_block_fp8_gemm",
    "deepseek_v4_fused_qk_norm_rope_store",
    "deepseek_v4_c4_c128_compressor",
    "deepseek_v4_aiter_c4_paged_mqa_logits",
    "deepseek_v4_tilelang_sparse_attention",
    "deepseek_v4_aiter_fp8_fused_moe",
)


def load_spec(operator_id):
    path = ROOT / "operators" / "references" / operator_id / "spec.py"
    spec = importlib.util.spec_from_file_location(
        f"test_{operator_id}_spec", path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.SPEC


class DeepSeekV4FlashMi300xContractTests(unittest.TestCase):
    def test_registry_has_six_references_and_mirrored_control_candidates(self):
        snapshot = FilesystemRegistry(ROOT).discover()
        self.assertFalse(snapshot.issues)
        for operator_id in HIGH_PRIORITY:
            with self.subTest(operator_id=operator_id):
                self.assertIn(operator_id, snapshot.references)
                candidates = snapshot.candidates.get(operator_id, ())
                self.assertEqual(len(candidates), 1)
                self.assertTrue(
                    candidates[0].implementation_id.startswith("reference_control__")
                )

    def test_specs_are_controller_safe_and_cover_both_phases(self):
        expected_counts = {
            "deepseek_v4_aiter_block_fp8_gemm": 4,
            "deepseek_v4_fused_qk_norm_rope_store": 4,
            "deepseek_v4_c4_c128_compressor": 8,
            "deepseek_v4_aiter_c4_paged_mqa_logits": 4,
            "deepseek_v4_tilelang_sparse_attention": 10,
            "deepseek_v4_aiter_fp8_fused_moe": 4,
        }
        for operator_id in HIGH_PRIORITY:
            with self.subTest(operator_id=operator_id):
                spec = load_spec(operator_id)
                cases = spec.cases()
                self.assertEqual(len(cases), expected_counts[operator_id])
                phases = Counter(case.symbols["phase"] for case in cases)
                self.assertEqual(set(phases), {"prefill", "decode"})
                self.assertTrue(any("oracle" in case.tags for case in cases))
                self.assertTrue(
                    any("performance_only" in case.tags for case in cases)
                )
                for case in cases:
                    cost = spec.cost_model(case)
                    self.assertGreater(cost["flops"], 0)
                    self.assertGreater(cost["estimated_bytes"], 0)
                    self.assertGreater(cost["throughput_units"], 0)
                    self.assertTrue(spec.layout_contract(case)["backend"])

    def test_suites_keep_only_confirmed_high_priority_operators(self):
        for phase in ("prefill", "decode"):
            path = ROOT / "suites" / f"deepseek_v4_{phase}.yaml"
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            self.assertEqual(tuple(data["operators"]["include"]), HIGH_PRIORITY)
            self.assertEqual(data["cases"]["tags"], [f"deepseek_v4_{phase}"])
            self.assertEqual(data["performance"]["warmup"], 5)
            self.assertEqual(data["performance"]["samples"], 30)
            self.assertEqual(data["performance"]["inner_iterations"], 20)
            self.assertNotIn("timer", data["performance"])

    def test_moe_disables_unsafe_hip_graph_probe(self):
        snapshot = FilesystemRegistry(ROOT).discover()
        moe = snapshot.operator_manifests[
            "deepseek_v4_aiter_fp8_fused_moe"
        ]
        self.assertEqual(moe.performance.timer, "cuda_event")
        self.assertEqual(moe.performance.graph_mode, "disabled")

    def test_attention_ratios_and_full_moe_geometry_match_v4_flash(self):
        attention = load_spec("deepseek_v4_tilelang_sparse_attention")
        for phase in ("prefill", "decode"):
            ratios = {
                int(case.symbols["compression_ratio"])
                for case in attention.cases()
                if case.symbols["phase"] == phase
            }
            self.assertEqual(ratios, {0, 4, 128})
        moe = load_spec("deepseek_v4_aiter_fp8_fused_moe")
        representative = next(
            case
            for case in moe.cases()
            if case.symbols["phase"] == "decode"
            and "representative" in case.tags
        )
        self.assertEqual(
            {
                key: representative.symbols[key]
                for key in ("hidden", "intermediate", "experts", "topk")
            },
            {"hidden": 4096, "intermediate": 2048, "experts": 256, "topk": 6},
        )

    def test_launchers_isolate_aiter_merged_config_cache(self):
        for filename in ("bootstrap.sh", "run.sh", "bench.sh"):
            source = (ROOT / filename).read_text(encoding="utf-8")
            self.assertIn("AITER_CONFIG_DIR", source)
            self.assertIn(".runtime/cache/aiter", source.replace('"$RUNTIME', '"$ROOT/.runtime'))


if __name__ == "__main__":
    unittest.main()
