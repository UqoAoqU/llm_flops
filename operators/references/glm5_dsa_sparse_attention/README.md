# GLM-5 legacy DSA sparse attention

This contract matches `dsa_flashmla.py`: one `flash_mla_sparse_fwd` call is
captured in a CUDA graph and twenty independent replay/event samples are
recorded. The complete 4x4 legacy sweep remains tagged `legacy_full`.
The reference backend is `sgl_kernel.flash_mla.flash_mla_sparse_fwd`.
Unified prefill/decode sparse rows use a separate
`glm5_dsa_unified_sparse_attention` contract because each graph replay contains
twenty kernel calls.
