"""SGL Kernel dense paged FlashMLA baseline."""

from sgl_kernel.flash_mla import flash_mla_with_kvcache


def operator(query, cache, block_table, cache_lengths, value_dim, scheduler, num_splits, softmax_scale):
    result = flash_mla_with_kvcache(
        query, cache, block_table, cache_lengths, value_dim,
        scheduler, num_splits, causal=False, softmax_scale=softmax_scale,
    )
    return result[0] if isinstance(result, tuple) else result
