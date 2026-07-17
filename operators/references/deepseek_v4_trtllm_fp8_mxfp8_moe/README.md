# DeepSeek V4 TRTLLM FP8/MXFP8 routed MoE

This reference migrates the legacy FlashInfer
`trtllm_fp8_block_scale_routed_moe` path. It is a distinct FP8-weight,
MXFP8-activation contract; the legacy MXFP4 path is intentionally not aliased
to this operator.

The formal geometry is 384 global experts, 16 local experts on EP rank zero,
top-k 6, hidden size 7168 and intermediate size 3072. Logical weights are
`w13[16, 6144, 7168]` and `w2[16, 7168, 3072]`, then transformed by SGLang to
the shuffled TRTLLM block-FP8 layout with 32-element UE8M0 scale blocks.
Routing IDs are global, weights are finite/non-negative and each token row sums
to one. Duplicate experts and tokens with no local expert are valid boundaries.

Weight preparation, route packing and SGLang layout alignment occur while
inputs are prepared. Backend import/build is charged to the worker import
stage; first-call JIT, warmup, graph capture and steady-state remain separate.
The operator times MXFP8 activation quantization plus fused routed MoE, exactly
as the legacy benchmark does. Large output artifacts are deterministically
bounded to 256 values while retaining the full public dtype and shape.

```bash
./bench.sh validate --operator deepseek_v4_trtllm_fp8_mxfp8_moe
CUDA_VISIBLE_DEVICES=0 ./bench.sh run --mode correctness \
  --operator deepseek_v4_trtllm_fp8_mxfp8_moe --tag smoke
CUDA_VISIBLE_DEVICES=0 ./bench.sh run --mode all \
  --operator deepseek_v4_trtllm_fp8_mxfp8_moe --tag representative
```
