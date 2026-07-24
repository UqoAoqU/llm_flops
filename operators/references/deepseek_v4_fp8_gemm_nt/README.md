# DeepSeek V4 FP8 GEMM NT

This operator migrates every legacy `Adapter(kind="fp8")` call that reaches
`deep_gemm.fp8_gemm_nt`.  `SPEC.legacy_mappings()` preserves the legacy phase,
adapter name, backend label, model instance count, requested M values, K and N.
Repeated shapes are deliberately not collapsed because model projection needs
their distinct instance counts.

## Contract

- `activation`: CUDA FP8 E4M3, `[M, K]`.
- activation scale: SGLang column-major/TMA-aligned packed UE8M0,
  `int32 [M, ceil((K/128)/4)]`. Each `int32` contains four UE8M0 exponent
  bytes; storage pads both M and the logical K-scale count to multiples of
  four.
- `weight`: CUDA FP8 E4M3, `[N, K]`.
- weight scale: SGLang/DeepGEMM packed UE8M0,
  `int32 [N, ceil((K/128)/4)]`. A logical 128x128 block scale is repeated
  across its 128 weight rows before four exponent bytes are packed into each
  `int32`.
- output: contiguous CUDA BF16 `[M, N]`.

This is the production SGLang ABI on Blackwell. Input construction calls
`sglang_per_token_group_quant_fp8` with `column_major_scales=True`,
`scale_tma_aligned=True`, and `scale_ue8m0=True`. Weight construction calls
SGLang `quant_weight_ue8m0` followed by `transform_scale_ue8m0`. The public
operator therefore has five arguments:

`operator(activation, activation_scale, weight, weight_scale, output)`.

There is no evaluator-only logical FP32 scale and no duplicate aligned-scale
argument. The spec fails explicitly if SGLang has not selected its packed
UE8M0 DeepGEMM path. In the pinned SGLang runtime that path is enabled for
SM100 (B200) and SM103 (B300).

K and N are multiples of 128. The reference is the existing optimized
DeepGEMM implementation: it consumes the aligned activation scale and writes a
preallocated BF16 output. The candidate is the independently hashed,
auditable PyTorch implementation: it decodes each packed UE8M0 exponent byte,
expands the production scales, explicitly dequantizes both FP8 inputs,
performs a PyTorch matrix multiplication, and converts the result to BF16. It
does not import DeepGEMM or a root benchmark module.

Correctness is not element-wise `allclose`.  The fixture retains the original
BF16 activation and weight in private oracle state, then derives the FP8
operands by dividing each quantization tile by its locked scale.  The formal
gate compares both the DeepGEMM reference and each candidate to the original
BF16 matrix product using upstream `calc_diff`:

`1 - 2 * sum(x*y) / sum(x*x + y*y) < 1e-3`.

The zero-denominator rule and strict `<` match the installed SGLang fork,
`sgl-project/DeepGEMM` revision
`731e7c7a97d269e4b9f482ea18d0e709a948f293`,
`tests/test_fp8_fp4.py::test_gemm`.  Thus the candidate cannot pass merely by
matching a faulty reference, and neither the threshold nor the oracle belongs
to KDA-Pilot or to a candidate source tree.

## Cost model

FLOPs are `2*M*K*N`. Estimated bytes count FP8 activation and weight, the
physical SGLang packed-scale storage including four-way TMA padding, and the
BF16 output exactly once. With `G=K/128`, `A4=ceil(M/4)*4`, and
`G4=ceil(G/4)*4`:

`M*K + N*K + A4*G4 + N*G4 + 2*M*N`.

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

This role convention applies to subsequent migrations: preserved optimized
legacy code is the reference/baseline, while new or alternative code is a
candidate. Consequently, the reference-to-candidate speedup may be below one
and the Phase 9 regression gate may honestly reject this demonstrative PyTorch
candidate even when correctness passes.

The formal path requires CUDA, PyTorch FP8 E4M3, DeepGEMM and dimensions
supported by the installed DeepGEMM build.  Multi-GPU scheduling and profiler
integration are intentionally outside this migration.

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
