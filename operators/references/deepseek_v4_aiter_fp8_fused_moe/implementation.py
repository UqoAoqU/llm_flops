"""AITER/Triton routed-expert FP8 fused MoE used by DeepSeek V4 Flash."""

import os

import torch

from aiter import ActivationType, QuantType
from aiter.fused_moe import fused_moe

_disable_aiter = False


def _triton_fallback(
    hidden_states,
    w13,
    w2,
    topk_weights,
    topk_ids,
    w13_scale,
    w2_scale,
):
    from types import SimpleNamespace

    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        fused_moe as triton_fused_moe,
    )
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.server_args import (
        get_global_server_args,
        set_global_server_args_for_scheduler,
    )

    try:
        get_global_server_args()
    except ValueError:
        set_global_server_args_for_scheduler(
            SimpleNamespace(
                enable_deterministic_inference=False,
                enable_fused_moe_sum_all_reduce=False,
            )
        )

    return triton_fused_moe(
        hidden_states=hidden_states,
        w1=w13,
        w2=w2,
        topk_output=StandardTopKOutput(topk_weights, topk_ids, None),
        moe_runner_config=MoeRunnerConfig(
            activation="silu",
            is_gated=True,
            inplace=False,
            swiglu_limit=10.0,
        ),
        use_fp8_w8a8=True,
        w1_scale=w13_scale,
        w2_scale=w2_scale,
        block_shape=[128, 128],
    )


def operator(
    hidden_states,
    w13,
    w2,
    w13_canonical,
    w2_canonical,
    topk_weights,
    topk_ids,
    w13_scale,
    w2_scale,
):
    global _disable_aiter
    use_aiter = (
        os.environ.get("LLM_FLOPS_DSV4_USE_AITER_MOE") == "1"
        and hidden_states.shape[0] > 32
        and hidden_states.shape[1] == 4096
        and w13.shape[0] == 256
        and w2.shape[-1] == 2048
        and topk_ids.shape[1] == 6
    )
    if not _disable_aiter and use_aiter:
        try:
            output = fused_moe(
                hidden_states=hidden_states,
                w1=w13,
                w2=w2,
                topk_weight=topk_weights,
                topk_ids=topk_ids,
                activation=ActivationType.Silu,
                quant_type=QuantType.per_128x128,
                w1_scale=w13_scale,
                w2_scale=w2_scale,
                dtype=torch.bfloat16,
                swiglu_limit=10.0,
                gate_mode="interleave",
            )
            # AITER launches parts of this one-stage path asynchronously while
            # allocating temporary sorting/quantization buffers inside
            # fused_moe().  The standalone benchmark discards those temporaries
            # at the Python call boundary, so close that lifetime explicitly.
            torch.cuda.synchronize(hidden_states.device)
            return output
        except RuntimeError as error:
            message = str(error)
            if "[aiter] build [" not in message or "failed" not in message:
                raise
            _disable_aiter = True
    return _triton_fallback(
        hidden_states,
        w13_canonical,
        w2_canonical,
        topk_weights,
        topk_ids,
        w13_scale,
        w2_scale,
    )
