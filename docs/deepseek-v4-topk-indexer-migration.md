# DeepSeek V4 TopK and indexer FP8 migration

SGLang's `topk_transform_512_v2` and
`fused_q_indexer_rope_hadamard_quant` are the validated optimized references.
For Phase 11, each formal candidate is a byte-identical copy of its reference
`implementation.py`. These control candidates deliberately remove an operator
implementation delta so the acceptance run isolates benchmark-engine behavior:
discovery and hashing, strict correctness, JIT staging, CUDA Graph timing,
mirrored artifacts and performance gates. Speedup is expected to be about 1x.

## TopK selection and page transform

`plan_topk_v2(seq_lens, static_threshold=0)` prepares `[B+1,2]` int32 metadata
before timing. For every batch row, valid scores are `scores[:seq_len]`.
Selection is descending by score; a cutoff tie prefers the smaller raw index.
The selected raw index `i` becomes
`page_table[i // page_size] * page_size + i % page_size`.

Kernel output is int32 `[B,K]`; scores are not output. Comparison is unordered
within each row because optimized selection order is not an API promise, and
is never flattened across rows. Test page tables use disjoint physical ranges
per batch so a row mix-up cannot pass as the same global set. Formal cases use
`seq_len >= K`; the documented kernel behavior for `seq_len <= K` is increasing
valid raw indices followed by `-1`, while negative lengths are rejected.

The legacy map retains K=1024, page size 64, context 65536, 30 C4 layers,
prefill M 1024/2048/4096 and decode M 16/32. Cases also cover K=1, cutoff ties,
all-negative scores, non-aligned lengths and a tail page.

## Indexer RoPE, FWHT and E4M3

The independent indexer contract fixes H=64 and D=128. RoPE multiplies the
trailing 64 dimensions as 32 complex pairs by `freqs_cis[position]`. A
128-point FWHT runs stages 1,2,4,...,64 and divides by `sqrt(128)`:

```text
scale = max(1e-4, amax(abs(hadamard))) / 448
inv_scale = 1.0 / scale
q_fp8 = E4M3(hadamard * inv_scale)
weights_out = weight * (128^-0.5 * 64^-0.5) * scale
```

Because the control candidate executes the same SGLang kernel, byte-exact FP8
codes validate comparator and artifact plumbing without conflating framework
arithmetic-rounding differences with engine correctness.

Outputs are `q_fp8[B,64,128]` and `weights_out[B,64,1]`. Normalization after
the timed call exposes uint8 FP8 codes, decoded values, weights and their
effective product. `QuantizedComparator` requires codes in 0..255 to match
exactly. Explicit tolerances are rtol=1e-5/atol=1e-6 for weights and
rtol=1e-4/atol=1e-5 for the effective product. Wrong fixtures prove both gates
diagnose errors. Cases cover zero clamp, non-contiguous weights, position 65535
and the legacy B=16/context=65536 decode shape.

## JIT and steady state

Reference modules call `_jit_topk_v2_module()` and
`_jit_main_q_indexer_rope_hadamard_quant_module(torch.bfloat16)` during worker
entrypoint import. Cold Ninja/PTXAS work is charged to `import_ms` while the
heartbeat is active. Input generation and TopK planning happen before timing.
First call, warmup, CUDA Graph build and steady raw samples remain separate.
Graph preparation includes one synchronized, untimed replay so CUDA's lazy
first-replay instantiation is charged to `graph_capture_ms`, never sample zero.

Use explicit suites because tag selection is suite-bounded:

```bash
CUDA_VISIBLE_DEVICES=0 ./bench.sh run --suite full --mode correctness \
  --operator deepseek_v4_topk_transform \
  --operator deepseek_v4_indexer_fp8_quant \
  --candidate 'reference_control__*'
CUDA_VISIBLE_DEVICES=0 ./bench.sh run --suite regression --mode all \
  --operator deepseek_v4_topk_transform \
  --operator deepseek_v4_indexer_fp8_quant \
  --candidate 'reference_control__*' \
  --tag representative
```

Results mirror sources under
`results/<operator_id>/<candidate_id>/<evaluation_id>/`. Legacy
`./run.sh prefill` and `./run.sh decode` behavior is unchanged.
