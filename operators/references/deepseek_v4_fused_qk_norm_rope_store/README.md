# DeepSeek V4 Flash fused Q/K norm, RoPE, and SWA store

This reference measures the SGLang Triton kernel that fuses per-head Q
RMSNorm, KV RMSNorm, GPT-J RoPE, KV in-place update, and the paged SWA write.
The V4 Flash layout is fixed at 64 query heads, head dimension 512, RoPE
dimension 64, and 128 tokens per SWA page.

Correctness cases use an independent PyTorch oracle and verify Q output, the
in-place KV result, FP8 cache bytes, BF16 RoPE bytes, and UE8M0 scale bytes.
Representative prefill/decode cases call the current SGLang kernel directly.

Source evidence:

- `python/sglang/srt/layers/fused_qk_norm_rope_store.py`
- `python/sglang/srt/models/deepseek_v4.py`
