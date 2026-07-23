"""SGLang TileLang DSV4 sparse-attention kernel for gfx942."""

import os

from sglang.srt.layers.attention.dsa.tilelang_kernel import (
    dpsk_v4_fp8_attention_fwd,
)

_use_triton_fallback = (
    os.environ.get("SGLANG_HACK_FLASHMLA_BACKEND", "triton").lower()
    == "triton"
)


def _kwargs(
    q,
    swa_cache,
    swa_indices,
    swa_lengths,
    extra_cache,
    extra_indices,
    extra_lengths,
    attention_sink,
):
    return {
        "q": q,
        "k_cache": swa_cache,
        "block_table": None,
        "cache_seqlens": None,
        "head_dim_v": 512,
        "tile_scheduler_metadata": None,
        "softmax_scale": 512**-0.5,
        "causal": False,
        "is_fp8_kvcache": True,
        "indices": swa_indices,
        "attn_sink": attention_sink,
        "extra_k_cache": extra_cache,
        "extra_indices_in_kvcache": extra_indices,
        "topk_length": swa_lengths,
        "extra_topk_length": extra_lengths,
    }


def operator(
    q,
    swa_cache,
    swa_indices,
    swa_lengths,
    extra_cache,
    extra_indices,
    extra_lengths,
    attention_sink,
):
    global _use_triton_fallback
    arguments = _kwargs(
        q,
        swa_cache,
        swa_indices,
        swa_lengths,
        extra_cache,
        extra_indices,
        extra_lengths,
        attention_sink,
    )
    if not _use_triton_fallback:
        try:
            result = dpsk_v4_fp8_attention_fwd(**arguments)
        except ValueError as error:
            if "software pipeline" not in str(error):
                raise
            _use_triton_fallback = True
    if _use_triton_fallback:
        from sglang.srt.layers.attention.nsa.triton_decode import (
            triton_fp8_attention_fwd,
        )

        result = triton_fp8_attention_fwd(**arguments)
    return result[0] if isinstance(result, (tuple, list)) else result
