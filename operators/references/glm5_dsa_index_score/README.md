# GLM-5 DSA index score

This separate contract preserves the two optimized unified-benchmark paths:
contiguous `deep_gemm.fp8_mqa_logits` for prefill and paged
`deep_gemm.fp8_paged_mqa_logits` for decode. They are not collapsed into the
generic cuBLAS `index_score` row from `dsa_indexer.py`; the coverage table keeps
both legacy semantics explicit.
The reference calls the original DeepGEMM symbols directly.
