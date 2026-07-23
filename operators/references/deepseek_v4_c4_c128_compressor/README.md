# DeepSeek V4 Flash C4/C128 compressor

This reference isolates `sglang.jit_kernel.dsv4.compress_forward`, the
stateful softmax-pooling primitive shared by the V4 Flash C4 and C128
compressors. C4 consumes an eight-token overlap/current window; C128 consumes
one 128-token window. Both prefill and decode mutate a per-request ring state.

Small cases compare output and ring-buffer writes with an independent FP64
PyTorch semantic oracle. Representative cases retain the exact SGLang plan
objects and call the current gfx942 implementation.

Source evidence:

- `python/sglang/jit_kernel/dsv4/compress.py`
- `python/sglang/srt/layers/attention/dsv4/compressor_v2.py`
- `python/sglang/jit_kernel/tests/deepseek_v4/test_c4_v2.py`
- `python/sglang/jit_kernel/tests/deepseek_v4/test_c128_v2.py`
