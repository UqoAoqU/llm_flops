# DeepSeek V4 indexer FP8 quantization

The reference is the existing optimized SGLang fused kernel. It applies RoPE
to the trailing 64 dimensions, a normalized 128-point FWHT, and per-`(B,H)`
E4M3 quantization. The scale is `max(1e-4, amax(abs(x))) / 448`; the returned
weight is `weight * (128^-0.5 * 64^-0.5) * scale` with shape `[B,64,1]`.
Correctness requires byte-exact FP8 codes, explicitly bounded weight error,
and bounded downstream `fp8_value * weight` error. Dequantization is performed
by `spec.normalize_output` after the timed call. The private SGLang JIT cache is
materialized during worker import (`import_ms`), never during steady samples.
The Phase 11 control candidate is a byte-identical copy of this implementation,
so its expected performance ratio is approximately 1x.
