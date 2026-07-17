"""SGL Kernel FlashMLA SWA-only baseline from the legacy benchmark."""

from sgl_kernel.flash_mla import flash_mla_with_kvcache


def operator(q, cache, indices, lengths, attention_sink, scheduler):
    result = flash_mla_with_kvcache(
        q=q,
        k_cache=cache,
        block_table=None,
        cache_seqlens=None,
        head_dim_v=512,
        tile_scheduler_metadata=scheduler,
        softmax_scale=512**-0.5,
        is_fp8_kvcache=True,
        indices=indices,
        attn_sink=attention_sink,
        topk_length=lengths,
        extra_k_cache=None,
        extra_indices_in_kvcache=None,
        extra_topk_length=None,
    )
    return result[0] if isinstance(result, tuple) else result
