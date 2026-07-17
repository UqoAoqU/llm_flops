"""SGL Kernel FlashMLA sparse-prefill baseline from the legacy benchmark."""

from sgl_kernel.flash_mla import flash_mla_sparse_fwd


def _primary_output(result):
    if isinstance(result, (tuple, list)):
        if not result:
            raise RuntimeError("flash_mla_sparse_fwd returned an empty output sequence")
        result = result[0]
    if not hasattr(result, "shape") or not callable(getattr(result, "detach", None)):
        raise TypeError("flash_mla_sparse_fwd primary output must be a tensor")
    return result


def operator(q, kv, indices, softmax_scale, value_dim):
    result = flash_mla_sparse_fwd(q, kv, indices, softmax_scale, value_dim)
    return _primary_output(result)
