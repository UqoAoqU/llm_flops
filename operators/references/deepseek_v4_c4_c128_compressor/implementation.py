"""SGLang C4/C128 compressor primitive used on MI300X."""

from sglang.jit_kernel.dsv4 import compress_forward


def operator(state_pool, kv_score_input, ape, plan, ratio, head_dim):
    output = compress_forward(
        state_pool,
        kv_score_input,
        ape,
        plan,
        head_dim=head_dim,
        compress_ratio=ratio,
    )
    return output
