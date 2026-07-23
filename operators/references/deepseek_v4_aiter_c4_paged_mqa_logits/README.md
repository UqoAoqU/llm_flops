# DeepSeek V4 Flash AITER C4 paged-MQA logits

This reference measures AITER's gfx942 preshuffled paged-MQA kernel used by the
DeepSeek V4 C4 indexer. Q is `[M, 1, 64, 128]` FP8, cache pages contain 64
tokens with packed FP8 values and FP32 scales, and logits are FP32.

Correctness cases use shuffled page tables, a tail page, variable sequence
lengths, and an independent PyTorch MQA oracle. Cache and page-table state are
observed to reject accidental mutation.

Source evidence:

- SGLang: `python/sglang/srt/layers/attention/dsv4/indexer.py`
- AITER: `aiter/ops/triton/attention/pa_mqa_logits.py`
- AITER benchmark: `op_tests/op_benchmarks/triton/bench_deepgemm_attention.py`
