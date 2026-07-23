# DeepSeek V4 Flash AITER FP8 fused MoE

This reference measures the current SGLang AMD routed-expert path with dynamic
block-FP8 activations, E4M3FNUZ weights, float32 128x128 inverse scales, and
the DeepSeek V4 SwiGLU clamp of 10. The representative model shape is
`E=256, H=4096, I=2048, topK=6`.

The formal reference uses SGLang's Triton block-FP8 `fused_moe` path for both
prefill and decode. This is the stable single-GPU SGLang path in the pinned
environment: the AITER two-stage CK JIT does not compile against the pinned CK
headers, while repeated calls to its one-stage gfx942 kernel intermittently
raise a HIP illegal-address error. The one-stage path remains available only
for diagnosis with `LLM_FLOPS_DSV4_USE_AITER_MOE=1`; it is never selected
silently for a formal result.

This operator intentionally uses HIP event timing because MoE sorting and
workspace allocation are not graph-capture safe.

Small cases use scaled dimensions and an independent PyTorch routed-expert
oracle. Representative prefill/decode cases preserve the full model weight
shape and call the current SGLang Triton gfx942 implementation.

Source evidence:

- SGLang: `python/sglang/srt/layers/moe/moe_runner/aiter.py`
- SGLang: `python/sglang/srt/layers/quantization/fp8.py`
- AITER: `aiter/fused_moe.py`
- AITER: `op_tests/test_moe_blockscale.py`
