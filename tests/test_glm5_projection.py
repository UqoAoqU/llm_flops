from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path

from benchmark_engine.engine import build_dry_run_plan
from benchmark_engine.projection import GLM5_PROJECTION, projection_for_case, projection_for_id
from benchmark_engine.reporting.summary import _projection_lines
from benchmark_engine.selectors import Selectors


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_ROWS = {
    "fused_qkv_a_proj", "q_b_proj", "absorbed_W_UK", "absorbed_W_UV", "o_proj",
    "index_k_proj", "index_q_upproj", "index_weights_proj", "index_score",
    "moe_gate_proj", "moe_up_proj", "moe_down_proj",
}


def load_spec(operator_id):
    path = ROOT / "operators" / "references" / operator_id / "spec.py"
    module_spec = importlib.util.spec_from_file_location(f"_glm5_projection_{operator_id}", path)
    assert module_spec is not None and module_spec.loader is not None
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module.SPEC


class Glm5ProjectionTests(unittest.TestCase):
    def test_prefill_and_decode_have_exactly_thirteen_unique_rows(self):
        self.assertIs(projection_for_id("glm5_operator_sum"), GLM5_PROJECTION)
        for phase in ("prefill", "decode"):
            mappings = GLM5_PROJECTION.mappings(phase, "glm5_fp8")
            self.assertEqual(len(mappings), 13)
            ids = {mapping.adapter_id for mapping in mappings}
            self.assertEqual(ids - {f"dsa_{phase}_attn"}, EXPECTED_ROWS)
            self.assertTrue(all(mapping.instances == 1 for mapping in mappings))
            self.assertTrue(all(mapping.operator_id for mapping in mappings))
        self.assertEqual(GLM5_PROJECTION.mappings("prefill", "mxfp4"), ())

    def test_every_projection_mapping_has_a_case_for_every_legacy_input(self):
        specs = {mapping.operator_id: load_spec(mapping.operator_id)
                 for phase in ("prefill", "decode")
                 for mapping in GLM5_PROJECTION.mappings(phase, "glm5_fp8")}
        inputs = {"prefill": (1024, 2048, 4096), "decode": (1, 4, 8, 16, 32, 64)}
        for phase, values in inputs.items():
            for mapping in GLM5_PROJECTION.mappings(phase, "glm5_fp8"):
                matching = [case for case in specs[mapping.operator_id].cases()
                            if case.symbols.get("phase") == phase
                            and case.symbols.get("projection_adapter_id") == mapping.adapter_id]
                self.assertEqual({case.symbols["model_input"] for case in matching}, set(values),
                                 (phase, mapping.adapter_id))
                for case in matching:
                    projection, resolved = projection_for_case(mapping.operator_id, case)
                    self.assertIs(projection, GLM5_PROJECTION)
                    self.assertEqual(resolved, mapping)
                    self.assertEqual(case.symbols["raw_context"], 65536)

    def test_glm5_suites_expand_one_control_job_per_supported_family(self):
        for suite in ("glm5_smoke", "glm5_regression"):
            with tempfile.TemporaryDirectory() as temporary:
                plan = build_dry_run_plan(ROOT, suite, Selectors(),
                                          output_root=Path(temporary) / "results")
                self.assertEqual(len(plan.jobs), 7)
                self.assertEqual(len({job.identity.operator_id for job in plan.jobs}), 7)
                self.assertNotIn("glm5_deepep_dispatch",
                                 {job.identity.operator_id for job in plan.jobs})

    def test_regression_inherits_each_operator_timing_shape(self):
        with tempfile.TemporaryDirectory() as temporary:
            plan = build_dry_run_plan(ROOT, "glm5_regression", Selectors(),
                                      output_root=Path(temporary) / "results")
        expected = {
            "glm5_dsa_indexer": ("cuda_graph", "enabled", 20, 5, 10),
            "glm5_dsa_projection": ("cuda_graph", "enabled", 20, 5, 10),
            "glm5_dsa_index_score": ("cuda_graph", "enabled", 20, 5, 10),
            "glm5_dsa_unified_sparse_attention": ("cuda_graph", "enabled", 20, 5, 10),
            "glm5_moe_masked_grouped_gemm": ("cuda_graph", "enabled", 20, 5, 10),
            "glm5_dsa_sparse_attention": ("cuda_graph", "enabled", 1, 5, 20),
            "glm5_moe_grouped_gemm": ("cuda_event", "disabled", 1, 5, 20),
        }
        self.assertEqual({job.identity.operator_id for job in plan.jobs}, set(expected))
        for job in plan.jobs:
            config = job.resolved_config
            self.assertEqual(
                (config.performance_timer, config.performance_graph_mode,
                 config.performance_inner_iterations, config.performance_warmup,
                 config.performance_samples),
                expected[job.identity.operator_id],
            )

    def test_summary_uses_glm5_projection_registry_and_all_thirteen_rows(self):
        rows = []
        for mapping in GLM5_PROJECTION.mappings("prefill", "glm5_fp8"):
            base = {
                "projection_id": "glm5_operator_sum", "phase": "prefill",
                "quant_profile": "glm5_fp8", "model_input": "1024",
                "raw_context": "65536", "adapter_id": mapping.adapter_id,
                "implementation_role": "candidate", "candidate_id": "control",
                "status": "measured", "reason": "", "per_call_ms": "1.0",
                "projected_model_ms": "1.0", "_seed": "0",
                "evaluation_id": "evaluation", "result_id": mapping.adapter_id,
            }
            rows.append(base)
            rows.append(dict(base, implementation_role="reference",
                             result_id=f"reference_{mapping.adapter_id}"))
        summary = "\n".join(_projection_lines(rows, complete_model=True))
        self.assertIn("## GLM-5 operator sum model projection", summary)
        self.assertNotIn("DeepSeek V4 Pro", summary)
        self.assertIn("Candidate measured partial total: 13.000000", summary)
        self.assertIn("Reference measured partial total: 13.000000", summary)
        self.assertIn("Missing operators: none", summary)
        self.assertEqual(summary.count("| control |"), 13)


if __name__ == "__main__":
    unittest.main()
