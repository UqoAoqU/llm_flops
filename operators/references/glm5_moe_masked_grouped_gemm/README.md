# GLM-5 unified masked grouped MoE GEMM

This contract owns the `fp8_m_grouped_gemm_nt_masked` gate/up/down rows in
`bench_glm5_prefill.py` and `bench_glm5_decode.py`. It captures twenty kernel
calls per CUDA graph replay, matching `_cuda_graph_bench`. Logical routing
assignments drive the cost model; the per-expert allocation alignment is a
workspace/layout detail rather than extra model tokens.
The reference backend is DeepGEMM masked grouped FP8 GEMM.

The old scripts used unseeded per-assignment Python `randint` routing but
allocated each expert from the aligned average count. For non-balanced prefill
routing that allocation is too small; a fixed M=1024 reproduction produced
`max(counts)=1074` with legacy capacity 1024 and caused a CUDA launch failure.
Cases retain independently seeded random counts, record both
`legacy_expected_m` and `safe_expected_m`, and allocate the safe
`align128(max(counts))` capacity. Logical cost remains based on `M * top_k`.
