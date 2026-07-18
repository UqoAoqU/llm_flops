"""Unified GLM-5 SGL Kernel FlashMLA sparse-forward reference."""

from sgl_kernel.flash_mla import flash_mla_sparse_fwd


def operator(query, cache, indices, softmax_scale, value_dim):
    result = flash_mla_sparse_fwd(query, cache, indices, softmax_scale, value_dim)
    return result[0] if isinstance(result, (tuple, list)) else result
