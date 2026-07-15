# DeepSeek V4 Pro Dual Quantization Profile Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Freeze the validated MXFP4 benchmark and add a shape-identical FP8/MXFP8 profile with B200 SGLang 0.5.15 backends and five-case comparison output.

**Architecture:** Keep one benchmark engine and make adapter registration, fixture construction, CSV output, and weight caches profile-aware. The `mxfp4` path remains behaviorally unchanged; `fp8_mxfp8` substitutes only the Indexer quant/logits and routed-expert backend, so totals and per-operator speedups remain directly comparable.

**Tech Stack:** Python 3.12, PyTorch 2.11 CUDA 13.0, SGLang 0.5.15, DeepGEMM, FlashInfer TRTLLM fused MoE, unittest, NVIDIA B200.

## Global Constraints

- Prefill uses `seq_Q=1024/2048/4096`, raw `seq_KV=65536`, and no append.
- Decode uses batch `16/32` and raw `seq_KV=65536`.
- C4 length is `16384`; C128 length is `512`.
- Attention and Indexer use full logical shapes without TP slicing.
- MoE uses EP24, 384 global experts, 16 local experts, Top-6, and intermediate size 3072.
- MoE stays fused; do not split Gate/Up, SwiGLU, Down, or weighted finalization.
- Static weight preparation and JIT compilation are excluded from timing.
- Preserve unrelated dirty-worktree changes.

---

### Task 1: Profile-Aware Registry And Output Contract

**Files:**
- Modify: `deepseek_v4_benchmark.py`
- Modify: `tests/test_deepseek_v4_benchmark.py`

**Interfaces:**
- Produces: `QUANT_PROFILES`, `prefill_adapters(profile: str)`, `decode_adapters(profile: str)`, `run_adapter(..., profile: str)`, and `--quant-profile`.
- Produces: CSV column `quant_profile` on every result row.

- [ ] **Step 1: Write failing registry tests**

```python
def test_quant_profiles_use_distinct_indexer_and_moe_backends(self):
    mx = {a.name: a for a in prefill_adapters("mxfp4")}
    fp8 = {a.name: a for a in prefill_adapters("fp8_mxfp8")}
    self.assertEqual(mx["C4 Indexer FP4 Quant"].kind, "fp4_quant")
    self.assertEqual(fp8["C4 Indexer FP8 Quant"].kind, "fp8_quant")
    self.assertEqual(mx["Routed Expert Fused MoE"].kind, "moe_mxfp4")
    self.assertEqual(fp8["Routed Expert Fused MoE"].kind, "moe_fp8_mxfp8")

def test_unknown_quant_profile_is_rejected(self):
    with self.assertRaisesRegex(ValueError, "unknown quant profile"):
        prefill_adapters("invalid")
```

- [ ] **Step 2: Run tests and verify RED**

Run: `.venv-sglang0515/bin/python -m unittest tests/test_deepseek_v4_benchmark.py -v`

Expected: failure because adapter factories do not accept a profile.

- [ ] **Step 3: Add profile validation and profile-specific adapters**

```python
QUANT_PROFILES = ("mxfp4", "fp8_mxfp8")

def validate_quant_profile(profile):
    if profile not in QUANT_PROFILES:
        raise ValueError(f"unknown quant profile: {profile}")
    return profile

def prefill_adapters(profile="mxfp4"):
    validate_quant_profile(profile)
    indexer = (
        [Adapter("C4 Indexer FP4 Quant", "SGLang fused RoPE/Hadamard FP4", C4_LAYERS, kind="fp4_quant"),
         Adapter("C4 FP4 Paged MQA Logits", "DeepGEMM fp8_fp4_paged_mqa_logits", C4_LAYERS, kind="fp4_logits")]
        if profile == "mxfp4"
        else
        [Adapter("C4 Indexer FP8 Quant", "SGLang/torch FP8 E4M3", C4_LAYERS, kind="fp8_quant"),
         Adapter("C4 FP8 Paged MQA Logits", "DeepGEMM fp8_paged_mqa_logits", C4_LAYERS, kind="fp8_logits")]
    )
    # Insert indexer into the unchanged common registry and select
    # kind="moe_mxfp4" or kind="moe_fp8_mxfp8" for the fused MoE row.
```

Add `--quant-profile` with choices from `QUANT_PROFILES`, pass it through adapter construction and `run_adapter`, print it in the case header, and add it to each CSV row.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `.venv-sglang0515/bin/python -m unittest tests/test_deepseek_v4_benchmark.py -v`

Expected: all profile and legacy default-profile tests pass.

- [ ] **Step 5: Commit profile contract**

```bash
git add deepseek_v4_benchmark.py tests/test_deepseek_v4_benchmark.py
git commit -m "feat: add DeepSeek V4 quant profiles"
```

### Task 2: FP8 Indexer Fixture

**Files:**
- Modify: `deepseek_v4_benchmark.py`
- Modify: `tests/test_deepseek_v4_benchmark.py`

**Interfaces:**
- Produces: `_fp8_quant_fn(batch, context, torch)` and `_fp8_logits_fn(batch, c4_context, torch)`.
- Consumes: profile-specific kinds `fp8_quant` and `fp8_logits` from Task 1.

- [ ] **Step 1: Write failing FP8 shape tests**

```python
def test_fp8_indexer_shapes_use_c4_context(self):
    adapters = {a.name: a for a in prefill_adapters("fp8_mxfp8")}
    self.assertEqual(
        adapter_io_shapes(adapters["C4 FP8 Paged MQA Logits"], 1024, 65536),
        ("q=(1024,1,64,128); raw_kv=65536; c4_kv=16384",
         "logits=(1024,16384)"),
    )
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `.venv-sglang0515/bin/python -m unittest tests.test_deepseek_v4_benchmark.DeepSeekV4BenchmarkTest.test_fp8_indexer_shapes_use_c4_context -v`

Expected: failure because the FP8 kinds have no shape contract.

- [ ] **Step 3: Implement the FP8 fixtures using the SGLang test-proven layout**

```python
def _fp8_logits_fn(batch, context, torch):
    import deep_gemm
    from sglang.jit_kernel.dsa import deepgemm_paged_mqa_logits_split
    from sglang.srt.layers.attention.dsa.utils import (
        fp8_mqa_logits_ceil_to_ue8m0,
        fp8_mqa_logits_make_fused_kv,
    )

    blocks_per_sequence = (context + PAGE_SIZE - 1) // PAGE_SIZE
    num_blocks = batch * blocks_per_sequence
    page_table = torch.arange(num_blocks, dtype=torch.int32, device="cuda").view(batch, blocks_per_sequence)
    context_lens = torch.full((batch,), context, dtype=torch.int32, device="cuda")
    q_fp8 = torch.randn(batch, INDEX_HEADS, INDEX_HEAD_DIM, device="cuda").to(torch.float8_e4m3fn)
    kv = torch.randn(num_blocks, PAGE_SIZE, INDEX_HEAD_DIM, device="cuda")
    kv_amax = kv.abs().float().amax(dim=-1, keepdim=True).clamp(1.0e-4)
    kv_scale = fp8_mqa_logits_ceil_to_ue8m0(kv_amax / 448.0).squeeze(-1)
    kv_fp8 = (kv / kv_scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    kv_fused = fp8_mqa_logits_make_fused_kv(kv_fp8, kv_scale, PAGE_SIZE, INDEX_HEAD_DIM)
    weights = torch.randn(batch, INDEX_HEADS, device="cuda", dtype=torch.float32)
    metadata = deep_gemm.get_paged_mqa_logits_metadata(
        context_lens.unsqueeze(-1), PAGE_SIZE, deep_gemm.get_num_sms()
    )
    return lambda: deepgemm_paged_mqa_logits_split(
        deep_gemm.fp8_paged_mqa_logits, q_fp8, kv_fused, weights,
        context_lens.unsqueeze(-1), page_table, metadata, context, q_offset=batch,
    )
```

Implement `_fp8_quant_fn` as E4M3 conversion of `[M,64,128]`, and route both new adapter kinds in `run_adapter`. Keep TopK unchanged because it consumes the same `[M,16384]` logits.

- [ ] **Step 4: Run unit tests and GPU smoke**

Run unit tests, then run one adapter each at `M=16` and `M=1024`, `raw_kv=65536` with warmup 1/runs 2.

Expected: executed rows with FP8 input/output shapes and no unavailable status.

- [ ] **Step 5: Commit FP8 Indexer**

```bash
git add deepseek_v4_benchmark.py tests/test_deepseek_v4_benchmark.py
git commit -m "feat: benchmark FP8 DeepSeek V4 indexer"
```

### Task 3: FlashInfer TRTLLM FP8 Weight And MXFP8 Activation MoE

**Files:**
- Modify: `deepseek_v4_benchmark.py`
- Modify: `tests/test_deepseek_v4_benchmark.py`

**Interfaces:**
- Produces: `_get_fp8_mxfp8_moe_weights(torch)` and `_moe_fp8_mxfp8_fn(m, torch)`.
- Consumes: `moe_fp8_mxfp8` adapter kind and the existing deterministic EP-local packed TopK distribution.

- [ ] **Step 1: Write failing fused-MoE profile tests**

```python
def test_fp8_mxfp8_moe_is_one_fused_row(self):
    rows = [a for a in decode_adapters("fp8_mxfp8") if "MoE" in a.name]
    self.assertEqual(len(rows), 1)
    self.assertEqual(rows[0].backend, "FlashInfer TRTLLM FP8 weight + MXFP8 activation")
    self.assertEqual(rows[0].shape, (16, 7168, 3072))
```

- [ ] **Step 2: Run the focused test and verify RED**

Run: `.venv-sglang0515/bin/python -m unittest tests.test_deepseek_v4_benchmark.DeepSeekV4BenchmarkTest.test_fp8_mxfp8_moe_is_one_fused_row -v`

Expected: backend/kind mismatch until the FP8/MXFP8 adapter is implemented.

- [ ] **Step 3: Implement SGLang-aligned shuffled FP8 weights and MXFP8 activation**

Create FP8 E4M3 weights `[16,6144,7168]` and `[16,7168,3072]`, uint8 E8M0 block-32 scales, and apply the row and scale shuffles used by `align_mxfp8_moe_weights_for_flashinfer_trtllm`. Reuse the current deterministic packed TopK IDs so local pairs remain `M/4`.

```python
def run():
    from flashinfer import mxfp8_quantize
    from flashinfer.fused_moe import Fp8QuantizationType
    from sglang.srt.layers.moe.flashinfer_trtllm_moe import (
        trtllm_fp8_block_scale_routed_moe_wrapper,
    )

    a_q, a_sf = mxfp8_quantize(x, False, backend="cute-dsl")
    a_sf = a_sf.view(torch.uint8).reshape(m, -1)
    return trtllm_fp8_block_scale_routed_moe_wrapper(
        topk_ids=packed_topk, routing_bias=None,
        hidden_states=a_q, hidden_states_scale=a_sf,
        gemm1_weights=w13, gemm1_weights_scale=s13,
        gemm2_weights=w2, gemm2_weights_scale=s2,
        num_experts=E_GLOBAL, top_k=MOE_TOPK,
        n_group=None, topk_group=None,
        intermediate_size=MOE_INTERMEDIATE,
        local_expert_offset=0, local_num_experts=E_LOCAL,
        routed_scaling_factor=1.0, routing_method_type=1,
        use_shuffled_weight=True,
        tune_max_num_tokens=1 << (m - 1).bit_length(),
        fp8_quantization_type=int(Fp8QuantizationType.MxFp8),
    )
```

- [ ] **Step 4: Run unit tests and B200 fused-MoE smoke**

Run the full unit suite and direct `M=16`/`M=1024` MoE adapters with warmup 1/runs 2.

Expected: both rows execute, retain one fused call, and report local pairs 4/256.

- [ ] **Step 5: Commit FP8/MXFP8 MoE**

```bash
git add deepseek_v4_benchmark.py tests/test_deepseek_v4_benchmark.py
git commit -m "feat: benchmark FP8 MXFP8 routed MoE"
```

### Task 4: Freeze MXFP4 Results And Run Comparison Matrix

**Files:**
- Create: `results/deepseek_v4_pro_mxfp4_*_kv65536.csv` by copying the five validated final CSVs.
- Create: `results/deepseek_v4_pro_fp8_mxfp8_*_kv65536.csv` from formal runs.
- Create: `print_deepseek_v4_quant_comparison.py`
- Modify: `tests/test_deepseek_v4_benchmark.py`

**Interfaces:**
- Produces: terminal comparison with profile totals, percentages, absolute deltas, and `mxfp4_total / fp8_mxfp8_total` speedup.

- [ ] **Step 1: Add a failing comparison-data test**

```python
def test_comparison_requires_matching_operator_sets(self):
    comparison = compare_profile_rows(mx_rows, fp8_rows)
    self.assertEqual(comparison.case, ("prefill", 1024, 65536))
    self.assertGreater(comparison.speedup, 0.0)
```

- [ ] **Step 2: Implement CSV loading and comparison validation**

Require matching phase/M/context, matching instance counts, and semantic mapping of FP4 versus FP8 Indexer row names. Fail with a clear error rather than comparing mismatched cases.

- [ ] **Step 3: Freeze the MXFP4 files and run formal FP8/MXFP8 cases**

```bash
env -u PYTHONPATH .venv-sglang0515/bin/python -u bench_deepseek_v4_prefill.py \
  --quant-profile fp8_mxfp8 --m 1024,2048,4096 --context 65536 \
  --warmup 5 --runs 20 --csv results/deepseek_v4_pro_fp8_mxfp8_prefill_kv65536.csv
env -u PYTHONPATH .venv-sglang0515/bin/python -u bench_deepseek_v4_decode.py \
  --quant-profile fp8_mxfp8 --m 16,32 --context 65536 \
  --warmup 5 --runs 20 --csv results/deepseek_v4_pro_fp8_mxfp8_decode_kv65536.csv
```

- [ ] **Step 4: Verify final artifacts**

Run py_compile and all unit tests. Audit ten CSVs: every row is `executed`, context is 65536, C4 shapes contain 16384, shapes are nonempty, percentages sum within 0.01 of 100, and the two profiles have matching model geometry.

- [ ] **Step 5: Print the comparison to CLI**

Run: `.venv-sglang0515/bin/python print_deepseek_v4_quant_comparison.py`

Expected: separate Prefill and Decode tables showing absolute total ms, percentage distribution, and FP8/MXFP8 speedup relative to frozen MXFP4.

- [ ] **Step 6: Commit comparison tooling**

```bash
git add deepseek_v4_benchmark.py tests/test_deepseek_v4_benchmark.py \
  print_deepseek_v4_quant_comparison.py
git commit -m "feat: compare DeepSeek V4 quant profiles"
```
