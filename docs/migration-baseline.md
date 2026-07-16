# Legacy benchmark migration baseline

This document characterizes the legacy `llm_flops` benchmark at commit
`f2960df`. It is a migration reference for later refactoring, not the final
contract or schema of the future Benchmark Engine.

## Baseline identity

| Item | Baseline |
|---|---|
| Branch | `agent/develop-yusan` |
| Git commit | `f2960dfac0e42c23444364eebe212f25df0bc90f` |
| Fetch/push remote | `yusan` |
| Remote URL | `git@github.com:YusanXY/llm_flops.git` |
| Recorded | 2026-07-16 |

The working tree was clean before characterization. `origin` points to a
different repository and is not the fetch or push target for this baseline.

## Environment contract

The target environment declared by `requirements/benchmark-lock.json` is
distinct from the environment observed on the benchmark host:

| Component | Lock target | Observed on host 194 |
|---|---|---|
| Python | `3.12.x` | `3.12.13` |
| CUDA | `13.0` | PyTorch CUDA runtime `13.0`; `nvcc` is not on `PATH` |
| GPU | name contains `B200`, capability `[10, 0]` | 4 x NVIDIA B200, capability `(10, 0)`, driver `610.43.02` |
| PyTorch | `2.11.0` | distribution `2.11.0`, runtime build `2.11.0+cu130` |
| SGLang source | commit `19593359971ebc3582a74f000bf285488d993362` | same commit; package version `0.5.6.post3.dev7293+g195933599` |
| SGL Kernel | `0.4.4` | `0.4.4` |
| SGL DeepGEMM | `0.1.4` | `0.1.4` |
| FlashInfer Python | `0.6.12` | `0.6.12` |
| FlashInfer Cubin | `0.6.12` | `0.6.12` |
| NVIDIA CUTLASS DSL | `4.5.2` | `4.5.2` |

The observations came from `.runtime/venv/bin/python -m
benchmark_environment --json`, `.runtime/venv/bin/python --version`, PyTorch
runtime queries, `nvidia-smi`, and `nvcc --version`. The environment validator
reported fingerprint `4286ebdeb11b`, every required symbol present, and no
validation errors. CUDA `13.0` above is the runtime reported by PyTorch; the
standalone CUDA compiler was not observed.

## Legacy CLI dispatch

`run.sh` executes the repository-local `.runtime/venv/bin/python`, unsets
`PYTHONPATH`, and dispatches through `benchmark_cli.py`. The launcher keeps UV,
PyTorch extension, FlashInfer workspace, and XDG caches under `.runtime/cache`.
If the venv is missing it exits 2 and asks the user to run `./bootstrap.sh`.

| Legacy command | Python file selected | Environment precheck before dispatch |
|---|---|---|
| `./run.sh prefill` | `bench_deepseek_v4_prefill.py` | yes |
| `./run.sh decode` | `bench_deepseek_v4_decode.py` | yes |
| `./run.sh compare` | `print_deepseek_v4_quant_comparison.py` | no |
| `./run.sh check` | `tools/check_environment.py` | no CLI precheck; this command performs the check itself |
| `./run.sh smoke` | `tools/smoke_test.py` | yes |
| `./run.sh op dsa_indexer` | `dsa_indexer.py` | yes |
| `./run.sh op dsa_flashmla` | `dsa_flashmla.py` | yes |
| `./run.sh op dsa_projection` | `dsa_projection.py` | yes |
| `./run.sh op mla_flashmla` | `mla_flashmla.py` | yes |
| `./run.sh op moe_deepgemm` | `moe_deepgemm.py` | yes |

The current `GPU_COMMANDS` set is exactly `op`, `prefill`, `decode`, and
`smoke`. Arguments after a recognized command or operator are appended without
reordering or transformation. Missing commands, unknown commands, missing
operators, and unknown operators are usage errors with exit code 2; resolution
fails before any GPU environment check.

## DeepSeek V4 cases, profiles, and geometry

The no-argument CLI defaults are:

| Phase | `m` | Derived `context` | Quant profile | Warmup | Runs | Default CSV |
|---|---:|---:|---|---:|---:|---|
| prefill | `1024` | `1024` | `mxfp4` | 5 | 20 | `results/deepseek_v4_pro_prefill_local.csv` |
| decode | `16` | `16384` | `mxfp4` | 5 | 20 | `results/deepseek_v4_pro_decode_local.csv` |

An explicit `--context` overrides these derived values. The README's formal
benchmark matrix is prefill `m=1024,2048,4096` and decode `m=16,32`, both with
raw context `65536`; those are documentation examples with explicit arguments,
not the no-argument defaults. This mismatch is recorded as existing behavior
and is not corrected in Phase 0.

The supported quant profiles are exactly `mxfp4` and `fp8_mxfp8`. They differ
in C4 indexer quantization/logits and routed MoE backends: FP4 plus
FP8/FP4-paged logits and MXFP4 weights for `mxfp4`; FP8 plus FP8-paged logits
and FP8 weights/MXFP8 activations for `fp8_mxfp8`.

| Geometry | Value |
|---|---:|
| C4 layers | 30 |
| C128 layers | 30 |
| Dense/SWA layers | 1 |
| Model layers | 61 |
| Hidden size | 7168 |
| Q LoRA rank | 1536 |
| Attention heads / head dimension | 128 / 512 |
| Indexer heads / head dimension / TopK | 64 / 128 / 1024 |
| Global / local experts | 384 / 16 |
| Experts per token | 6 |
| MoE intermediate size | 3072 |
| Page size | 64 |

Adapter ordering, backend, instance count, kind, shape, and `m_override` are
locked by `tests/test_deepseek_v4_benchmark.py` for both phases and profiles.

## Legacy CSV and result semantics

The exact legacy CSV field order is:

```text
phase
quant_profile
environment_fingerprint
m
context
operator
backend
instances
call_ms
model_ms
pct
status
input_shape
output_shape
error
```

`executed` means the adapter ran and produced a per-call `call_ms`.
`unavailable` records an adapter that could not run and normally has blank
`call_ms` and `model_ms`, an explanatory `error`, and `pct` formatted as
`0.0000`. Summary totals include only `executed` rows with non-null `call_ms`;
`model_ms = call_ms * instances`, and percentages are computed against the
sum of those model times. A zero total produces no percentage entries. CSV
latencies use six decimal places and percentages use four.

A formal result exits 0 only when every row has status `executed`. Any
`unavailable` or other non-`executed` status makes `result_exit_code()` return
1. The legacy comparison reader expects matching phase, `m`, context, operator
set, and instance counts, and consumes numeric `model_ms` values. It does not
reinterpret unavailable rows; callers must not pass a blank `model_ms` into a
formal profile comparison.

`tests/fixtures/legacy_deepseek_result.csv` is a two-row compatibility fixture,
not a performance baseline or official benchmark result.

## Test baseline

The baseline command was:

```bash
.runtime/venv/bin/python -m unittest discover -s tests -v
```

It ran 33 tests in 0.003 seconds and all passed. The baseline tests were:

- `test_benchmark_cli`: `test_forwards_profiling_arguments_unchanged`,
  `test_missing_command_has_usage`, `test_resolves_every_single_operator`,
  `test_resolves_support_commands`, `test_unknown_operator_lists_valid_names`.
- `test_benchmark_environment`: `test_fingerprint_is_stable_and_path_independent`,
  `test_load_lock_rejects_unknown_schema`, `test_matching_environment_is_valid`,
  `test_source_commit_is_identity_when_generated_version_varies`,
  `test_validation_reports_every_mismatch`.
- `test_bootstrap_contract`: `test_builds_an_ignored_local_runtime`,
  `test_installs_the_immutable_sglang_revision`,
  `test_is_strict_and_repository_relative`,
  `test_never_reuses_reference_environment_or_external_source`,
  `test_validates_before_marking_install_complete`.
- `test_deepseek_v4_benchmark`:
  `test_comparison_requires_matching_cases_and_maps_indexer_names`,
  `test_decode_registry_uses_fp4_indexer_without_compression`,
  `test_ep_local_routed_pair_counts_match_requested_cases`,
  `test_fixed_raw_kv_derives_compressed_lengths`,
  `test_fp8_indexer_shapes_use_c4_context`,
  `test_fp8_mxfp8_moe_is_one_fused_row`,
  `test_prefill_context_defaults_to_sequence_length`,
  `test_prefill_lm_head_uses_original_full_token_style`,
  `test_prefill_registry_includes_indexer_selection_pipeline`,
  `test_quant_profiles_use_distinct_indexer_and_moe_backends`,
  `test_registry_matches_full_shape_v4_pro_geometry`,
  `test_result_csv_records_environment_fingerprint`,
  `test_shape_descriptions_cover_full_shape_and_fused_moe`,
  `test_summary_excludes_unavailable_rows`,
  `test_unavailable_row_fails_formal_result`,
  `test_unknown_quant_profile_is_rejected`.
- `test_smoke_test`: `test_every_case_has_a_callable_runner`,
  `test_plan_covers_every_single_operator_family`.

Phase 0 adds characterization tests around this baseline. It does not change
benchmark cases, tolerances, kernels, timing, production CSVs, or runtime
behavior. Future engine schemas and status taxonomies will deliberately evolve
beyond this legacy snapshot.
