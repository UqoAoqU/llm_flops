"""Control candidate for AITER C4 preshuffled paged-MQA logits."""

from aiter.ops.triton.attention.pa_mqa_logits import (
    deepgemm_fp8_paged_mqa_logits,
)


def operator(
    q_fp8,
    packed_cache,
    weights,
    output,
    context_lens,
    page_table,
    max_seq_len,
    valid_width,
):
    deepgemm_fp8_paged_mqa_logits(
        q_fp8,
        packed_cache,
        weights,
        output,
        context_lens,
        page_table,
        max_seq_len,
        Preshuffle=True,
        KVBlockSize=64,
        ChunkK=128,
    )
    return output[:, :valid_width]
