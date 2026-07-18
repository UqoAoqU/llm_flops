import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
ROWS = (
    "fused_qkv_a_proj", "q_b_proj", "absorbed_W_UK", "absorbed_W_UV", "o_proj",
    "index_k_proj", "index_q_upproj", "index_weights_proj", "index_score",
    "moe_gate_proj", "moe_up_proj", "moe_down_proj",
)


class Phase16Glm5DocumentationTests(unittest.TestCase):
    def test_readme_and_index_link_the_complete_migration_guide(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        index = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")
        self.assertIn("glm5_smoke", readme)
        self.assertIn("glm5_regression", readme)
        self.assertIn("glm5-migration.md", readme)
        self.assertIn("glm5-migration.md", index)

    def test_coverage_guide_names_every_legacy_entry_and_unified_row(self):
        text = (ROOT / "docs" / "glm5-migration.md").read_text(encoding="utf-8")
        for script in ("dsa_indexer.py", "dsa_flashmla.py", "dsa_projection.py",
                       "mla_flashmla.py", "moe_deepgemm.py", "bench_glm5_prefill.py",
                       "bench_glm5_decode.py", "bench_glm5_deepep.py"):
            self.assertIn(script, text)
        for row in ROWS:
            self.assertGreaterEqual(text.count(f"`{row}`"), 2, row)
        self.assertIn("`dsa_prefill_attn`", text)
        self.assertIn("`dsa_decode_attn`", text)
        self.assertIn("glm5_dsa_unified_sparse_attention", text)
        self.assertIn("glm5_moe_masked_grouped_gemm", text)
        self.assertIn("15 CSV rows", text)
        self.assertIn("under-allocated", text)
        self.assertIn("safe_expected_m", text)
        self.assertNotIn("dense decode MLA", text)
        self.assertIn("dense paged prefill attention", text)
        self.assertIn("explicit unsupported", text.lower())

    def test_suites_are_strict_yaml_and_exclude_deepep(self):
        for name in ("glm5_smoke", "glm5_regression"):
            data = yaml.safe_load((ROOT / "suites" / f"{name}.yaml").read_text(encoding="utf-8"))
            self.assertEqual(data["schema_version"], 1)
            self.assertEqual(data["suite_id"], name)
            self.assertNotIn("glm5_deepep_dispatch", data["operators"]["include"])
            self.assertNotIn("glm5_dense_prefill_attention", data["operators"]["include"])
            self.assertEqual(set(data["operators"]["include"]), {
                "glm5_dsa_indexer", "glm5_dsa_projection", "glm5_dsa_index_score",
                "glm5_dsa_sparse_attention", "glm5_dsa_unified_sparse_attention",
                "glm5_moe_grouped_gemm", "glm5_moe_masked_grouped_gemm",
            })
            self.assertEqual(data["candidates"]["include"], ["reference_control__*"])


if __name__ == "__main__":
    unittest.main()
