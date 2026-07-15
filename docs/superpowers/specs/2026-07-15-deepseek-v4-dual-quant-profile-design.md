# DeepSeek V4 Pro Dual Quantization Profile Design

## Goal

Preserve the validated B200 benchmark as an explicitly named MXFP4 profile and
add a directly comparable FP8/MXFP8 profile. Both profiles use identical model
geometry, operator counts, parallelism, and workload cases.

## Fixed Workload

- GPU/backend target: NVIDIA B200 (SM100), SGLang 0.5.15.
- Prefill: `seq_Q=1024/2048/4096`, raw `seq_KV=65536`, no append.
- Decode: `batch=16/32`, raw `seq_KV=65536`.
- Compressed lengths: C4 `16384`, C128 `512`.
- Attention and Indexer use full logical shapes without TP slicing.
- MoE uses EP24 only: 384 global experts, 16 local experts, Top-6,
  intermediate size 3072.
- Timing remains warmup 5 and 20 CUDA Graph replays. JIT and static weight
  preparation are excluded.

## Profiles

### `mxfp4`

This freezes the current validated implementation and results:

- Dense linear GEMMs: DeepGEMM FP8.
- Indexer query/cache path: FP4 quantization and FP4 paged MQA logits.
- Routed expert: FlashInfer TRTLLM MXFP4 weights with MXFP8 activation.
- The MoE row remains fused and includes input quantization, Gate/Up, SwiGLU,
  Down, and weighted finalization.

Existing final CSVs are retained and copied to profile-explicit names. Their
measurements are not regenerated or relabeled as NVFP4.

### `fp8_mxfp8`

- Dense linear GEMMs remain DeepGEMM FP8.
- Indexer uses the SGLang/DeepGEMM FP8 quantization and
  `fp8_paged_mqa_logits` backend.
- Routed expert uses FlashInfer TRTLLM block-scaled FP8 weights and MXFP8
  activations through the SGLang 0.5.15 B200 path.
- MoE fusion boundaries and routing distribution remain identical to the
  `mxfp4` profile.

`WO_A` grouped GEMM and FlashMLA BF16 interfaces remain BF16 because the
selected SGLang B200 execution path has no equivalent profile-aligned FP8
replacement. The profile name describes quantized GEMM, Indexer, and MoE
backends rather than forcing unsupported conversions.

## Interface And Output

- Add `--quant-profile {mxfp4,fp8_mxfp8}` to both benchmark entry points.
- Adapter registration and input/output shape reporting become profile-aware.
- CSV output records the quantization profile in every row.
- Produce five FP8/MXFP8 CSVs and a CLI comparison containing absolute model
  time, per-operator time, percentage, and speedup relative to MXFP4.

## Verification

- Unit tests pin profile-specific adapter names, backends, shapes, C4 length,
  and fused MoE semantics.
- GPU smoke tests execute FP8 Indexer and FP8/MXFP8 fused MoE independently.
- Formal runs cover all five fixed workload cases.
- Every final CSV row must be `executed`, contain input/output shapes, and sum
  to approximately 100 percent per case.
