"""Production fused Q/K norm, RoPE, and SWA-store kernel."""

import torch

from sglang.srt.layers.fused_qk_norm_rope_store import (
    fused_qk_norm_rope_swa_store,
)


def operator(
    q,
    kv,
    q_norm_weight,
    kv_norm_weight,
    cos_cache,
    sin_cache,
    positions,
    swa_cache,
    swa_loc,
    q_out,
):
    return fused_qk_norm_rope_swa_store(
        q,
        kv,
        q_norm_weight,
        kv_norm_weight,
        1.0e-6,
        1.0e-6,
        64,
        cos_cache,
        sin_cache,
        positions,
        swa_cache=swa_cache,
        swa_loc=swa_loc,
        swa_page_size=128,
        q_out=q_out,
        dtype=torch.bfloat16,
    )
