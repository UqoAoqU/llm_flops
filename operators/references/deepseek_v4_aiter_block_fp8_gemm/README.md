# DeepSeek V4 Flash AITER block-FP8 GEMM

This reference measures the production ROCm path
`aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale.gemm_a8w8_blockscale`.
Activations use per-row, per-128-element scales; weights use 128x128 scales;
the output is BF16. The representative cases cover the V4 Flash Q-LoRA and
wide WQB projection shapes on `gfx942`.

The smoke cases compare both measured implementations with an independent
PyTorch dequantize-and-matmul oracle. Representative cases avoid materializing
that oracle and retain the same reference/control kernel and effective timer.

Source evidence:

- SGLang: `python/sglang/srt/layers/quantization/fp8_utils.py`
- AITER: `aiter/ops/triton/gemm/basic/gemm_a8w8_blockscale.py`
