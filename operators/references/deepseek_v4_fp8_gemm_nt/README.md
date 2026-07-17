# DeepSeek V4 FP8 GEMM NT

This operator migrates every legacy `Adapter(kind="fp8")` call that reaches
`deep_gemm.fp8_gemm_nt`.  `SPEC.legacy_mappings()` preserves the legacy phase,
adapter name, backend label, model instance count, requested M values, K and N.
Repeated shapes are deliberately not collapsed because model projection needs
their distinct instance counts.

## Contract

- `activation`: CUDA FP8 E4M3, `[M, K]`.
- logical activation scale: FP32 `[M, K/128]`, one scale per token and K tile.
- TMA-aligned activation scale: the same logical values transformed by
  DeepGEMM's `get_mn_major_tma_aligned_tensor` before timing.
- `weight`: CUDA FP8 E4M3, `[N, K]`.
- weight scale: FP32 `[N/128, K/128]`, one scale per 128x128 weight tile.
- output: contiguous CUDA BF16 `[M, N]`.

The locked Blackwell DeepGEMM path aliases these scales as UE8M0 and therefore
requires each positive FP32 scale to be an exact power of two. Inputs choose
seeded exponents from `{-1, 0, 1}` and construct scales as `2**exponent`, so the
cases exercise `0.5`, `1.0`, and `2.0` rather than evading the layout contract
with constant scales. The TMA-aligned activation tensor is derived only after
the logical values have been created.

K and N are multiples of 128. The reference is the existing optimized
DeepGEMM implementation: it consumes the aligned activation scale and writes a
preallocated BF16 output. The repository contains two useful candidate styles:
an auditable PyTorch dequantize-and-matmul implementation, and a byte-identical
DeepGEMM control copy used to validate the framework. The former includes
dequantization in its timed body; the latter should pass correctness and
measure approximately 1x.

Correctness uses explicit `rtol=1e-2`, `atol=1e-1`, matching DeepGEMM's
Blackwell FP8 GEMM validation.  This tolerance accounts for FP8 inputs,
tile-scaled accumulation and BF16 output rounding; it is not inferred from a
dtype fallback.  The intentionally wrong zero-output fixture demonstrates that
the comparator still localizes material errors.

## Cost model

FLOPs are `2*M*K*N`.  Estimated bytes count FP8 activation and weight, logical
FP32 activation scales, FP32 weight scales and the BF16 output exactly once:

`M*K + N*K + 4*M*(K/128) + 4*(N/128)*(K/128) + 2*M*N`.

Throughput units are output elements (`M*N`), while the engine additionally
reports TFLOPS and effective bandwidth from the candidate median latency.

## Measurement and limitations

Input generation, scale transformation, output allocation, first call/JIT,
warmup and CUDA Graph capture happen before steady-state sampling. The
reference timed body only calls `fp8_gemm_nt` and returns its preallocated
output. Candidate dequantization, expanded-scale allocation and matmul are
intentionally part of candidate timing. Artifacts therefore compare the
existing optimized baseline against the implementation under evaluation.
Each raw CUDA Graph sample contains 20 calls because the reference latency is
only tens of microseconds; this amortizes event noise without mixing import,
first-call/JIT, warmup, or graph-build time into steady-state latency.

The preserved optimized legacy code is the reference/baseline, while new,
alternative, or control-copy code is a candidate. Consequently, the
reference-to-candidate speedup may be below one and the regression gate may
honestly reject the demonstrative PyTorch candidate even when correctness
passes.

The formal path requires CUDA, PyTorch FP8 E4M3, DeepGEMM and dimensions
supported by the installed DeepGEMM build.  Multi-GPU scheduling and profiler
integration are not implemented by the engine.

Validated command forms are:

```bash
CUDA_VISIBLE_DEVICES=0 ./bench.sh run --suite smoke \
  --operator deepseek_v4_fp8_gemm_nt --candidate 'pytorch_dequant__*'
CUDA_VISIBLE_DEVICES=0 ./bench.sh run --suite full --mode correctness \
  --operator deepseek_v4_fp8_gemm_nt --candidate 'pytorch_dequant__*'
CUDA_VISIBLE_DEVICES=0 ./bench.sh run --suite regression --mode all \
  --operator deepseek_v4_fp8_gemm_nt --candidate 'pytorch_dequant__*' \
  --tag representative
```
