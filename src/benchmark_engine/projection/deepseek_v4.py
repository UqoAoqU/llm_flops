"""DeepSeek V4 Pro legacy-adapter to benchmark-engine projection mapping."""

from __future__ import annotations

from benchmark_engine.models import CaseSpec

from .base import ProjectionMapping


MODEL_LAYERS = 61
C4_LAYERS = 30
C128_LAYERS = 30
DENSE_LAYERS = 1
PREFILL_INPUTS = (1024, 2048, 4096)
DECODE_INPUTS = (16, 32)
RAW_CONTEXT = 65536
FP8_PROFILE = "fp8_mxfp8"


def _m(adapter_id, phase, name, backend, instances, operator_id, kind, shape=None):
    return ProjectionMapping(adapter_id, phase, name, backend, instances, operator_id, kind, shape)


_PREFILL = (
    _m("fused_wq_a_wkv", "prefill", "Fused WQ_A + WKV", "DeepGEMM fp8_gemm_nt", 61, "deepseek_v4_fp8_gemm_nt", "fp8", (7168, 2048)),
    _m("q_rmsnorm_wq_b", "prefill", "Q RMSNorm + WQ_B", "DeepGEMM fp8_gemm_nt", 61, "deepseek_v4_fp8_gemm_nt", "fp8", (1536, 65536)),
    _m("compressor_wkv_gate_c4", "prefill", "Compressor WKV-Gate GEMM (C4)", "DeepGEMM fp8_gemm_nt", 30, "deepseek_v4_fp8_gemm_nt", "fp8", (7168, 2048)),
    _m("compressor_wkv_gate_c128", "prefill", "Compressor WKV-Gate GEMM (C128)", "DeepGEMM fp8_gemm_nt", 30, "deepseek_v4_fp8_gemm_nt", "fp8", (7168, 1024)),
    _m("c4_indexer_q_projection", "prefill", "C4 Indexer Q Projection", "DeepGEMM fp8_gemm_nt", 30, "deepseek_v4_fp8_gemm_nt", "fp8", (1536, 65536)),
    _m("c4_indexer_head_weight", "prefill", "C4 Indexer Head Weight Projection", "cuBLAS BF16 GEMM", 30, None, "bf16", (7168, 64)),
    _m("c4_indexer_fp8_quant", "prefill", "C4 Indexer FP8 Quant", "SGLang fused RoPE/Hadamard FP8", 30, "deepseek_v4_indexer_fp8_quant", "fp8_quant"),
    _m("c4_fp8_paged_mqa_logits", "prefill", "C4 FP8 Paged MQA Logits", "DeepGEMM fp8_paged_mqa_logits", 30, "deepseek_v4_fp8_paged_mqa_logits", "fp8_logits"),
    _m("c4_topk_transform", "prefill", "C4 TopK Transform", "SGLang JIT topk_transform_512_v2", 30, "deepseek_v4_topk_transform", "topk"),
    _m("sparse_prefill_attention", "prefill", "Sparse Prefill Attention", "sgl-kernel FlashMLA sparse_fwd", 60, "deepseek_v4_sparse_prefill_attention", "prefill_attention"),
    _m("dense_swa_attention", "prefill", "Dense SWA Attention", "sgl-kernel FlashMLA SWA-only", 1, None, "dense_prefill_attention"),
    _m("wo_a_grouped_projection", "prefill", "WO_A Grouped Projection", "cuBLAS grouped BF16 GEMM", 61, None, "grouped_bf16", (16, 4096, 1024)),
    _m("wo_b_projection", "prefill", "WO_B Projection", "DeepGEMM fp8_gemm_nt", 61, "deepseek_v4_fp8_gemm_nt", "fp8", (16384, 7168)),
    _m("routed_expert_fused_moe", "prefill", "Routed Expert Fused MoE", "FlashInfer TRTLLM FP8 weight + MXFP8 activation", 61, "deepseek_v4_trtllm_fp8_mxfp8_moe", "moe_fp8_mxfp8", (16, 7168, 3072)),
    _m("lm_head", "prefill", "LM Head", "DeepGEMM FP8 full vocab", 1, "deepseek_v4_fp8_gemm_nt", "fp8", (7168, 129280)),
)

_DECODE = (
    _m("fused_wq_a_wkv", "decode", "Fused WQ_A + WKV", "DeepGEMM fp8_gemm_nt", 61, "deepseek_v4_fp8_gemm_nt", "fp8", (7168, 2048)),
    _m("q_rmsnorm_wq_b", "decode", "Q RMSNorm + WQ_B", "DeepGEMM fp8_gemm_nt", 61, "deepseek_v4_fp8_gemm_nt", "fp8", (1536, 65536)),
    _m("c4_indexer_q_projection", "decode", "C4 Indexer Q Projection", "DeepGEMM fp8_gemm_nt", 30, "deepseek_v4_fp8_gemm_nt", "fp8", (1536, 65536)),
    _m("c4_indexer_head_weight", "decode", "C4 Indexer Head Weight Projection", "cuBLAS BF16 GEMM", 30, None, "bf16", (7168, 64)),
    _m("c4_indexer_fp8_quant", "decode", "C4 Indexer FP8 Quant", "SGLang fused RoPE/Hadamard FP8", 30, "deepseek_v4_indexer_fp8_quant", "fp8_quant"),
    _m("c4_fp8_paged_mqa_logits", "decode", "C4 FP8 Paged MQA Logits", "DeepGEMM fp8_paged_mqa_logits", 30, "deepseek_v4_fp8_paged_mqa_logits", "fp8_logits"),
    _m("c4_topk_transform", "decode", "C4 TopK Transform", "SGLang JIT topk_transform_512_v2", 30, "deepseek_v4_topk_transform", "topk"),
    _m("sparse_decode_attention_c4", "decode", "Sparse Decode Attention C4", "sgl-kernel FlashMLA dual-cache C4", 30, "deepseek_v4_sparse_decode_attention", "decode_attention_c4"),
    _m("sparse_decode_attention_c128", "decode", "Sparse Decode Attention C128", "sgl-kernel FlashMLA dual-cache C128", 30, "deepseek_v4_sparse_decode_attention", "decode_attention_c128"),
    _m("dense_swa_attention", "decode", "Dense SWA Attention", "sgl-kernel FlashMLA SWA-only", 1, "deepseek_v4_dense_swa_attention", "dense_decode_attention"),
    _m("wo_a_grouped_projection", "decode", "WO_A Grouped Projection", "cuBLAS grouped BF16 GEMM", 61, None, "grouped_bf16", (16, 4096, 1024)),
    _m("wo_b_projection", "decode", "WO_B Projection", "DeepGEMM fp8_gemm_nt", 61, "deepseek_v4_fp8_gemm_nt", "fp8", (16384, 7168)),
    _m("routed_expert_fused_moe", "decode", "Routed Expert Fused MoE", "FlashInfer TRTLLM FP8 weight + MXFP8 activation", 61, "deepseek_v4_trtllm_fp8_mxfp8_moe", "moe_fp8_mxfp8", (16, 7168, 3072)),
    _m("lm_head", "decode", "LM Head", "DeepGEMM FP8 full vocab", 1, "deepseek_v4_fp8_gemm_nt", "fp8", (7168, 129280)),
)


class DeepSeekV4Projection:
    projection_id = "deepseek_v4_pro"
    display_name = "DeepSeek V4 Pro"

    def mappings(self, phase: str, quant_profile: str) -> tuple[ProjectionMapping, ...]:
        if quant_profile != FP8_PROFILE:
            return ()
        if phase == "prefill":
            return _PREFILL
        if phase == "decode":
            return _DECODE
        return ()

    def mapping_for_case(self, operator_id: str, case: CaseSpec) -> ProjectionMapping | None:
        symbols = case.symbols
        adapter_id = symbols.get("projection_adapter_id")
        phase = symbols.get("phase")
        profile = symbols.get("quant_profile")
        if not all(isinstance(value, str) for value in (adapter_id, phase, profile)):
            return None
        for mapping in self.mappings(phase, profile):
            if mapping.adapter_id == adapter_id and mapping.operator_id == operator_id:
                return mapping
        return None


DEEPSEEK_V4_PROJECTION = DeepSeekV4Projection()


__all__ = [
    "C4_LAYERS", "C128_LAYERS", "DECODE_INPUTS", "DEEPSEEK_V4_PROJECTION",
    "DENSE_LAYERS", "FP8_PROFILE", "MODEL_LAYERS", "PREFILL_INPUTS",
    "RAW_CONTEXT", "DeepSeekV4Projection",
]
