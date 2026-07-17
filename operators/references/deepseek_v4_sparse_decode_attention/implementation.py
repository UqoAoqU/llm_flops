"""SGL Kernel FlashMLA dual-cache decode baseline from the legacy benchmark."""

from sgl_kernel.flash_mla import flash_mla_with_kvcache


def operator(
    q,
    swa_cache,
    swa_indices,
    swa_lengths,
    extra_cache,
    extra_indices,
    extra_lengths,
    attention_sink,
    scheduler,
):
    result = flash_mla_with_kvcache(
        q=q,
        k_cache=swa_cache,
        block_table=None,
        cache_seqlens=None,
        head_dim_v=512,
        tile_scheduler_metadata=scheduler,
        softmax_scale=512**-0.5,
        is_fp8_kvcache=True,
        indices=swa_indices,
        attn_sink=attention_sink,
        topk_length=swa_lengths,
        extra_k_cache=extra_cache,
        extra_indices_in_kvcache=extra_indices,
        extra_topk_length=extra_lengths,
    )
    return result[0] if isinstance(result, tuple) else result
