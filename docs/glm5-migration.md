# GLM-5 migration and coverage

GLM-5 is available through the same Registry, worker isolation, correctness
gate, timer, raw-sample and mirrored-artifact path as DeepSeek V4. The
optimized legacy kernel is the reference. Each migrated operator has a
byte-equivalent control candidate; it is useful for validating the framework,
not as a speedup claim.

## Commands

```bash
./bench.sh validate --operator 'glm5_*'
CUDA_VISIBLE_DEVICES=0 ./bench.sh run --suite glm5_smoke
CUDA_VISIBLE_DEVICES=0 ./bench.sh run --suite glm5_regression
```

`glm5_smoke` runs one bounded correctness case per supported B200 single-GPU
timing contract. `glm5_regression` runs correctness plus steady-state sampling
while inheriting each operator's timer, graph shape, warmup and sample count.
Full old sweeps are retained as `legacy_full` cases and
must be selected deliberately; they are not silently substituted by smaller
shapes. Results use
`results/<operator_id>/<candidate_id>/<evaluation_id>/`.

The GLM-5 projection is `glm5_operator_sum`. The old unified scripts sum one
call from each of thirteen rows and do not declare a layer multiplier, so every
mapping uses `instances=1`. This is an operator-sum projection, not a claim of
full-model latency.

## Five `run.sh op` entries

| Legacy command/script | New operator contract | Shape/case coverage | Backend/timer | Status |
|---|---|---|---|---|
| `run.sh op dsa_indexer` / `dsa_indexer.py` | `glm5_dsa_indexer` | four rows x M `16,256,512,1024` x S `65536,131072,262144` | cuBLAS `torch._scaled_mm`, CUDA graph, 5 warmup/20 runs | migrated; old CSV unchanged |
| `run.sh op dsa_flashmla` / `dsa_flashmla.py` | `glm5_dsa_sparse_attention` | Q and KV cartesian product `16384,32768,65536,131072`, top-k 2048 | `flash_mla_sparse_fwd`, graph with one call, 20 replay/event samples | migrated |
| `run.sh op dsa_projection` / `dsa_projection.py` | `glm5_dsa_projection` | six projection rows at M `1024,4096,16384,65536` | DeepGEMM GEMM plus per-head cuBLAS BMM, CUDA graph | migrated |
| `run.sh op mla_flashmla` / `mla_flashmla.py` | `glm5_dense_prefill_attention` | Q and KV cartesian product `16384,32768,65536,131072` | dense paged `flash_mla_with_kvcache`, CUDA graph | contract/cases migrated; installed backend is SM90a-only and explicit unsupported on B200 |
| `run.sh op moe_deepgemm` / `moe_deepgemm.py` | `glm5_moe_grouped_gemm` | gate/up/down, 8 local experts, 128 logical token assignments, five distributions each (15 CSV rows) | contiguous DeepGEMM grouped FP8, direct CUDA events | migrated; fixed seeds replace legacy unseeded routing |

The legacy MoE script assigns 128 logical tokens, then pads every non-empty
expert count to 128 rows. Its measured/costed physical `total_m` is therefore
derived from the distribution and is commonly 1024, not 128. The migrated
cases record `dist_idx`, seed and all eight counts; they use
`random.Random(seed).randint(0, 7)` once per logical assignment.

The original scripts and `benchmark_cli.py` are intentionally unchanged. Their
CSV headers remain respectively:

- `glm5_dsa_indexer_perf.csv`: `name,M,S,gemm_M,K,gemm_N,avg_ms,tflops,tbps,flops_per_byte`;
- `glm5_sparse_prefill_perf.csv`: `s_q,s_kv,topk,h_q,d_qk,d_v,avg_ms,min_ms,max_ms,tflops,tbps,flops_per_byte`;
- `glm5_attention_gemm_perf.csv`: `name,type,M,shape,avg_ms,tflops,tbps,flops_per_byte`;
- `glm5_dense_prefill_perf.csv`: `s_q,s_kv,h_q,d_qk,d_v,avg_ms,min_ms,max_ms,tflops,tbps,flops_per_byte`;
- `glm5_moe_deepgemm_perf.csv`: `proj,dist_idx,total_m,K,N,m_min,m_max,m_avg,avg_ms,min_ms,max_ms,tflops,tbps,flops_per_byte,m_per_expert`.

## Unified prefill: all 13 rows from `bench_glm5_prefill.py`

Legacy M is `1024,2048,4096`, context is 65536, and timing is five warmups
plus twenty CUDA-graph replays.

| # | Legacy row | New operator / case adapter | Reference backend |
|---:|---|---|---|
| 1 | `fused_qkv_a_proj` | `glm5_dsa_projection` / `fused_qkv_a_proj` | DeepGEMM FP8 GEMM |
| 2 | `q_b_proj` | `glm5_dsa_projection` / `q_b_proj` | DeepGEMM FP8 GEMM |
| 3 | `absorbed_W_UK` | `glm5_dsa_projection` / `absorbed_W_UK` | `sgl_kernel.bmm_fp8` |
| 4 | `absorbed_W_UV` | `glm5_dsa_projection` / `absorbed_W_UV` | `sgl_kernel.bmm_fp8` |
| 5 | `o_proj` | `glm5_dsa_projection` / `o_proj` | DeepGEMM FP8 GEMM |
| 6 | `dsa_prefill_attn` | `glm5_dsa_unified_sparse_attention` / `dsa_prefill_attn` | FlashMLA sparse forward |
| 7 | `index_k_proj` | `glm5_dsa_indexer` / `index_k_proj` | DeepGEMM FP8 GEMM |
| 8 | `index_q_upproj` | `glm5_dsa_indexer` / `index_q_upproj` | DeepGEMM FP8 GEMM |
| 9 | `index_weights_proj` | `glm5_dsa_indexer` / `index_weights_proj` | DeepGEMM BF16-to-F32 GEMM |
| 10 | `index_score` | `glm5_dsa_index_score` / `index_score` | DeepGEMM contiguous FP8 MQA logits |
| 11 | `moe_gate_proj` | `glm5_moe_masked_grouped_gemm` / `moe_gate_proj` | DeepGEMM masked grouped FP8 |
| 12 | `moe_up_proj` | `glm5_moe_masked_grouped_gemm` / `moe_up_proj` | DeepGEMM masked grouped FP8 |
| 13 | `moe_down_proj` | `glm5_moe_masked_grouped_gemm` / `moe_down_proj` | DeepGEMM masked grouped FP8 |

## Unified decode: all 13 rows from `bench_glm5_decode.py`

Legacy batch is `1,4,8,16,32,64`, context is 65536, and timing is five
warmups plus twenty CUDA-graph replays.

| # | Legacy row | New operator / case adapter | Reference backend |
|---:|---|---|---|
| 1 | `fused_qkv_a_proj` | `glm5_dsa_projection` / `fused_qkv_a_proj` | DeepGEMM FP8 GEMM |
| 2 | `q_b_proj` | `glm5_dsa_projection` / `q_b_proj` | DeepGEMM FP8 GEMM |
| 3 | `absorbed_W_UK` | `glm5_dsa_projection` / `absorbed_W_UK` | `sgl_kernel.bmm_fp8` |
| 4 | `absorbed_W_UV` | `glm5_dsa_projection` / `absorbed_W_UV` | `sgl_kernel.bmm_fp8` |
| 5 | `o_proj` | `glm5_dsa_projection` / `o_proj` | DeepGEMM FP8 GEMM |
| 6 | `dsa_decode_attn` | `glm5_dsa_unified_sparse_attention` / `dsa_decode_attn` | FlashMLA sparse forward |
| 7 | `index_k_proj` | `glm5_dsa_indexer` / `index_k_proj` | DeepGEMM FP8 GEMM |
| 8 | `index_q_upproj` | `glm5_dsa_indexer` / `index_q_upproj` | DeepGEMM FP8 GEMM |
| 9 | `index_weights_proj` | `glm5_dsa_indexer` / `index_weights_proj` | DeepGEMM BF16-to-F32 GEMM |
| 10 | `index_score` | `glm5_dsa_index_score` / `index_score` | DeepGEMM paged FP8 MQA logits |
| 11 | `moe_gate_proj` | `glm5_moe_masked_grouped_gemm` / `moe_gate_proj` | DeepGEMM masked grouped FP8 |
| 12 | `moe_up_proj` | `glm5_moe_masked_grouped_gemm` / `moe_up_proj` | DeepGEMM masked grouped FP8 |
| 13 | `moe_down_proj` | `glm5_moe_masked_grouped_gemm` / `moe_down_proj` | DeepGEMM masked grouped FP8 |

The unified scripts capture twenty calls per graph replay. Their sparse and
masked MoE rows therefore live in `glm5_dsa_unified_sparse_attention` and
`glm5_moe_masked_grouped_gemm`, both with `inner_iterations=20`. The legacy
sparse contract captures one call and records twenty independent replays; the
legacy contiguous MoE contract uses direct CUDA events. Separate operator IDs
prevent a suite override from silently changing these timing shapes.

The old unified masked-MoE scripts generated unseeded random routing counts
but allocated per-expert capacity as `align128(ceil(total_rows / 8))`. This is
under-allocated for every non-balanced prefill distribution when M is already
128-aligned. A fixed M=1024 reproduction (seed `434010241`) produced counts
`951,1056,1037,1074,1010,1020,1041,1003`: legacy capacity 1024 versus a maximum
count of 1074, followed by DeepGEMM assertions and a CUDA launch failure. The
migrated projection cases preserve independently seeded per-assignment random
routing, record both `legacy_expected_m` and `safe_expected_m`, and allocate
`safe_expected_m=align128(max(counts))`. This is an explicit safety correction;
logical FLOP/byte projection remains based on `M * top_k` assignments.

## DeepEP is explicit unsupported

`bench_glm5_deepep.py` always spawns eight local processes and requires NCCL
and DeepEP collective buffers. Multi-GPU scheduling is outside this engine
milestone. `glm5_deepep_dispatch` is nevertheless discoverable and contains
all six token counts x four routing scenarios; every case is tagged
`unsupported`, omitted from single-GPU suites, and raises a stable unsupported
result. It is never reported as pass or included in performance ranking. The
unchanged legacy program remains the only entry for an explicitly provisioned
eight-GPU environment.

The installed dense `flash_mla_with_kvcache` path used by `mla_flashmla.py`
also reports that dense paged prefill attention is SM90a-only. Its exact 4x4 contract remains
discoverable as `glm5_dense_prefill_attention`, but B200 suites exclude it and
direct execution returns a stable unsupported result. The legacy smoke helper
only exercises a small runner path and therefore is not evidence that the full
dense benchmark can run on SM100.

## Correctness and timing boundaries

Specs own deterministic inputs, physical clone isolation, normalized bounded
outputs, dtype/shape contracts and cost models. Candidate correctness runs
before formal performance. A correctness failure skips performance and makes a
row non-rankable. Quantization, scale alignment, FlashMLA scheduler metadata,
DeepGEMM JIT/first call, graph capture and warmup are outside steady-state raw
samples. System overlap or excessive CV remains visible and non-rankable; the
suite does not relax the common performance gate.
