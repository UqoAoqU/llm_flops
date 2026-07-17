"""FlashInfer TRTLLM FP8-weight/MXFP8-activation routed-MoE baseline.

Backend imports and any import-time JIT happen while the worker records the
import stage.  Activation quantization and the fused kernel are the timed
operator, matching the legacy ``_moe_fp8_mxfp8_fn`` closure.  Weight creation,
layout alignment, routing construction and top-k packing are spec-side input
preparation and therefore never enter steady-state samples.
"""

from importlib import import_module

import torch


def _load_backend_symbols():
    """Resolve the exact legacy symbols with stable missing-symbol errors."""

    try:
        flashinfer = import_module("flashinfer")
        fused_moe = import_module("flashinfer.fused_moe")
    except (ImportError, OSError) as error:
        raise RuntimeError(f"FlashInfer import/build error: {error}") from error

    required = {
        "flashinfer.mxfp8_quantize": getattr(flashinfer, "mxfp8_quantize", None),
        "flashinfer.fused_moe.Fp8QuantizationType": getattr(
            fused_moe, "Fp8QuantizationType", None
        ),
        "flashinfer.fused_moe.trtllm_fp8_block_scale_routed_moe": getattr(
            fused_moe, "trtllm_fp8_block_scale_routed_moe", None
        ),
    }
    missing = tuple(name for name, value in required.items() if value is None)
    if missing:
        raise NotImplementedError(
            "unsupported: missing FlashInfer symbol(s): " + ", ".join(missing)
        )
    return flashinfer, required[
        "flashinfer.fused_moe.Fp8QuantizationType"
    ], required["flashinfer.fused_moe.trtllm_fp8_block_scale_routed_moe"]


_FLASHINFER, _FP8_QUANTIZATION_TYPE, _ROUTED_MOE = _load_backend_symbols()

GLOBAL_EXPERTS = 384
LOCAL_EXPERTS = 16
TOPK = 6
INTERMEDIATE = 3072


def operator(
    hidden_states,
    packed_topk,
    routing_ids,
    routing_weights,
    w13,
    w13_scale,
    w2,
    w2_scale,
    output,
    tune_max_num_tokens,
):
    # Raw routing tensors are part of the public/observed contract; the legacy
    # backend consumes the PackTopkIds object prepared from these same tensors.
    del routing_ids, routing_weights
    hidden_states_quant, hidden_states_scale = _FLASHINFER.mxfp8_quantize(
        hidden_states, False, backend="cute-dsl"
    )
    # The legacy path passes the scale bytes to TRTLLM after MXFP8 quantization.
    hidden_states_scale = hidden_states_scale.view(torch.uint8).reshape(
        hidden_states.shape[0], -1
    )
    _ROUTED_MOE(
        topk_ids=packed_topk,
        routing_bias=None,
        hidden_states=hidden_states_quant,
        hidden_states_scale=hidden_states_scale,
        gemm1_weights=w13,
        gemm1_weights_scale=w13_scale,
        gemm2_weights=w2,
        gemm2_weights_scale=w2_scale,
        num_experts=GLOBAL_EXPERTS,
        top_k=TOPK,
        n_group=None,
        topk_group=None,
        intermediate_size=INTERMEDIATE,
        local_expert_offset=0,
        local_num_experts=LOCAL_EXPERTS,
        routed_scaling_factor=1.0,
        routing_method_type=1,
        use_shuffled_weight=True,
        do_finalize=True,
        output=output,
        tune_max_num_tokens=tune_max_num_tokens,
        fp8_quantization_type=_FP8_QUANTIZATION_TYPE.MxFp8,
        activation_type=3,
    )
    return output
