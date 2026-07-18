# GLM-5 unified DSA sparse attention

This contract owns `dsa_prefill_attn` and `dsa_decode_attn` from
`bench_glm5_prefill.py` and `bench_glm5_decode.py`. Their `_cuda_graph_bench`
captures twenty `flash_mla_sparse_fwd` calls per replay, represented by
`inner_iterations: 20`. The legacy 4x4 sparse sweep remains in the separate
`glm5_dsa_sparse_attention` contract with one call per replay.
The reference backend is `sgl_kernel.flash_mla.flash_mla_sparse_fwd`.
