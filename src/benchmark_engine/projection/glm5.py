"""GLM-5 unified-script operator-sum projection.

The legacy scripts sum one call for each of thirteen rows.  They do not encode
a model layer multiplier, so this projection intentionally uses instances=1
and is named ``glm5_operator_sum`` rather than claiming full-model latency.
"""

from benchmark_engine.models import CaseSpec

from .base import ProjectionMapping


FP8_PROFILE = "glm5_fp8"
PREFILL_INPUTS = (1024, 2048, 4096)
DECODE_INPUTS = (1, 4, 8, 16, 32, 64)
RAW_CONTEXT = 65536


def _m(adapter_id, phase, name, backend, operator_id, kind, shape=None):
    return ProjectionMapping(adapter_id, phase, name, backend, 1, operator_id, kind, shape)


def _rows(phase):
    return (
        _m("fused_qkv_a_proj", phase, "fused_qkv_a_proj", "DeepGEMM fp8_gemm_nt", "glm5_dsa_projection", "fp8_gemm", (6144, 2624)),
        _m("q_b_proj", phase, "q_b_proj", "DeepGEMM fp8_gemm_nt", "glm5_dsa_projection", "fp8_gemm", (2048, 16384)),
        _m("absorbed_W_UK", phase, "absorbed_W_UK", "sgl-kernel bmm_fp8", "glm5_dsa_projection", "fp8_bmm", (64, 192, 512)),
        _m("absorbed_W_UV", phase, "absorbed_W_UV", "sgl-kernel bmm_fp8", "glm5_dsa_projection", "fp8_bmm", (64, 512, 256)),
        _m("o_proj", phase, "o_proj", "DeepGEMM fp8_gemm_nt", "glm5_dsa_projection", "fp8_gemm", (16384, 6144)),
        _m(f"dsa_{phase}_attn", phase, f"dsa_{phase}_attn", "sgl-kernel flash_mla_sparse_fwd", "glm5_dsa_unified_sparse_attention", "sparse_attention", (64, 576, 512, 2048)),
        _m("index_k_proj", phase, "index_k_proj", "DeepGEMM fp8_gemm_nt", "glm5_dsa_indexer", "fp8_gemm", (6144, 128)),
        _m("index_q_upproj", phase, "index_q_upproj", "DeepGEMM fp8_gemm_nt", "glm5_dsa_indexer", "fp8_gemm", (2048, 4096)),
        _m("index_weights_proj", phase, "index_weights_proj", "DeepGEMM bf16_gemm_nt", "glm5_dsa_indexer", "bf16_gemm", (6144, 32)),
        _m("index_score", phase, "index_score", "DeepGEMM fp8_mqa_logits" if phase == "prefill" else "DeepGEMM fp8_paged_mqa_logits", "glm5_dsa_index_score", "fp8_mqa", (32, 128, RAW_CONTEXT)),
        _m("moe_gate_proj", phase, "moe_gate_proj", "DeepGEMM fp8_m_grouped_gemm_nt_masked", "glm5_moe_masked_grouped_gemm", "grouped_fp8", (8, 6144, 2048)),
        _m("moe_up_proj", phase, "moe_up_proj", "DeepGEMM fp8_m_grouped_gemm_nt_masked", "glm5_moe_masked_grouped_gemm", "grouped_fp8", (8, 6144, 2048)),
        _m("moe_down_proj", phase, "moe_down_proj", "DeepGEMM fp8_m_grouped_gemm_nt_masked", "glm5_moe_masked_grouped_gemm", "grouped_fp8", (8, 2048, 6144)),
    )


_PREFILL = _rows("prefill")
_DECODE = _rows("decode")


class Glm5Projection:
    projection_id = "glm5_operator_sum"
    display_name = "GLM-5 operator sum"

    def mappings(self, phase: str, quant_profile: str) -> tuple[ProjectionMapping, ...]:
        if quant_profile != FP8_PROFILE:
            return ()
        return _PREFILL if phase == "prefill" else _DECODE if phase == "decode" else ()

    def mapping_for_case(self, operator_id: str, case: CaseSpec) -> ProjectionMapping | None:
        symbols = case.symbols
        adapter_id, phase, profile = (symbols.get(name) for name in
                                      ("projection_adapter_id", "phase", "quant_profile"))
        if not all(isinstance(value, str) for value in (adapter_id, phase, profile)):
            return None
        return next((mapping for mapping in self.mappings(phase, profile)
                     if mapping.adapter_id == adapter_id and mapping.operator_id == operator_id), None)


GLM5_PROJECTION = Glm5Projection()


__all__ = ["DECODE_INPUTS", "FP8_PROFILE", "GLM5_PROJECTION", "Glm5Projection",
           "PREFILL_INPUTS", "RAW_CONTEXT"]
