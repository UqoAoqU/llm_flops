# DeepSeek V4 sparse prefill attention

This operator preserves the legacy `flash_mla_sparse_fwd(q, kv, indices,
scale, d_v)` call. C4/C128 context relationships are explicit case metadata;
the representative legacy case uses M=1024 and the uncompressed 65,536-token
cache.

Small cases are checked against an independent PyTorch gather/softmax/value
oracle. The representative performance-only input is zero-valued and retains
only 256 deterministic zero-oracle samples, avoiding construction of a full
1024x128x512 golden tensor. The oracle is not passed to the implementation, so
a candidate cannot edit it. Small cache/index views are observed to detect
in-place mutation. Empty indices are classified as `unsupported` rather than
silently rewritten.

The current SM100 kernel requires a non-empty `topk` divisible by 64. The
executable smoke case uses `topk=64`; explicit `topk=0` and `topk=1`
boundaries are stable unsupported cases.
