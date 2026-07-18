# GLM-5 attention projections

This operator preserves the six rows emitted by `dsa_projection.py`. GEMMs use
DeepGEMM FP8 block scaling and per-head BMM rows use the same cuBLAS
`torch._scaled_mm` loop as the legacy script. The unified GLM-5 prefill/decode
`fused_qkv_a_proj` row is an additional explicit case. Quantization, scale
alignment and first-call/JIT work happen before steady-state sampling.
The reference is the optimized legacy backend selected for each case.
