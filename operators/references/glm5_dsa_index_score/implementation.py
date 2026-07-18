"""DeepGEMM MQA logits baselines from the GLM-5 unified scripts."""

import deep_gemm
from sglang.jit_kernel.dsa import deepgemm_paged_mqa_logits_split


def operator(phase, query, cache, cache_scale, weights, lengths_start, lengths_end,
             block_tables, schedule, max_context):
    if phase == "prefill":
        return deep_gemm.fp8_mqa_logits(
            query, (cache, cache_scale), weights, lengths_start, lengths_end,
            clean_logits=False,
        )
    if phase == "decode":
        result = deepgemm_paged_mqa_logits_split(
            deep_gemm.fp8_paged_mqa_logits,
            query, cache, weights, lengths_start, block_tables, schedule,
            max_context, q_offset=query.shape[0],
        )
        return result[0] if isinstance(result, tuple) else result
    raise ValueError(f"unknown index-score phase {phase!r}")
