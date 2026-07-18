# Migrating llm_flops results

The migration keeps the original launchers as acceptance baselines while making the
engine artifact contract explicit. It does not delete or silently reinterpret
legacy CSV files.

## Supported migration scope

The converter accepts the exact DeepSeek V4 `fp8_mxfp8` CSV header produced by
`deepseek_v4_benchmark.py`. It validates the full prefill or decode adapter set,
phase, profile, environment fingerprint, M, context, operator display name,
backend, instances, input/output shape, status, and `model_ms=call_ms*instances`.

MXFP4 is not migrated by the current engine and is rejected with
`unsupported quant_profile 'mxfp4'`. GLM-5 conversion belongs to its separate
operator migration. Do not describe either as a successful import.

## Import aggregate legacy data

Validate without publishing:

```bash
.runtime/venv/bin/python tools/convert_legacy_results.py \
  results/deepseek_v4_pro_fp8_mxfp8_prefill_kv65536_m1024.csv \
  --candidate-id legacy_control --dry-run
```

Remove `--dry-run` to atomically publish mirrored import directories. The same
CSV bytes and candidate ID always produce the same run/evaluation identity,
independent of input filename. Case-insensitive candidate collisions and path
traversal are rejected.

The import records the original aggregate as `legacy_graph_ms`. The old
`graph_ms` executes warmups, captures `runs` invocations in one graph, replays
once, and divides elapsed time by `runs`. It is not a raw sample distribution
or median. Therefore imported source hashes, correctness, raw samples, CV,
speedup, stage timings, cost, memory, and ranking eligibility remain empty.
Overall/correctness/performance statuses are `skipped`, even when an old
aggregate is present; `model_projection.csv` preserves the value with an
explicit non-formal reason. Its `legacy_shape` JSON always preserves the exact
CSV input/output shape strings; a current logical mapping shape is additional
nullable metadata rather than a substitute for the legacy facts.

Legacy CSV has no creation timestamp. Its deterministic evaluation ID therefore
uses `19700101T000000Z` as an explicit unknown-time sentinel plus content/run
hash components; `results.csv.timestamp_utc` remains empty rather than invented.

## Predeclared B200 regression

The acceptance comparison is defined before measurement:

- physical GPU 0, with one UUID recorded for the full run;
- no compute process on that UUID before legacy measurement, between
  frameworks, or after engine measurement;
- FP8/MXFP8, raw context 65536;
- prefill M=1024/2048/4096 or decode batch=16/32;
- warmup 5, runs/samples 20, one invocation per engine sample;
- legacy `graph_ms` versus engine candidate CUDA Graph median;
- pass per mapped operator/case when absolute delta is at most 0.02 ms **or**
  relative delta is at most 10%.

The tool rejects empty inputs, non-finite/negative timings, duplicate or
missing cases, multiple candidates for one operator, non-CUDA-Graph samples,
sample count/index/shape/dtype mismatches, missing source identity, failed
correctness, non-formal results, and failed ranking gates.

Inspect the immutable plan first:

```bash
.runtime/venv/bin/python tools/regress_deepseek_v4.py \
  --phase prefill --work-dir .runtime/regression/prefill --dry-run
.runtime/venv/bin/python tools/regress_deepseek_v4.py \
  --phase decode --work-dir .runtime/regression/decode --dry-run
```

Remove `--dry-run` only when physical GPU 0 is idle. The tool runs the original
launcher and engine sequentially, writes `regression_report.json`, and exits 1
for a threshold failure or 2 for invalid/incomplete evidence. Cold JIT/import,
build, first call, warmup, and graph capture are outside engine steady samples;
the legacy launcher also warms before capture.

Do not treat overlap/high-CV/system-jitter output as a successful regression.
If correctness and framework contracts pass but a repeated idle-GPU latency
threshold fails, record the data and investigate rather than changing shapes,
references, tolerances, gates, or thresholds after the fact.

## Recommendation boundary

The engine may be named the recommended performance entry for a particular
host/configuration only after both prefill and decode reports pass. Until then,
the original scripts remain the performance acceptance baseline and the engine
remains usable for correctness/framework validation.
