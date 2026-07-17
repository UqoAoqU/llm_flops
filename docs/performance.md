# Performance measurement

Performance measurement is correctness-gated and uses fair interleaving,
physical-GPU locking, explicit gates, and strict artifact comparison.
Nsight/NVTX, profiler backends, and multi-GPU scheduling are not implemented.

## CPU example

This repository includes a CPU-only operator and candidate, so the performance
path can be exercised without a CUDA installation:

```bash
./bench.sh run --mode performance \
  --operator example_cpu_add \
  --candidate quickstart__20260716T120000Z__4279e756 \
  --case tiny --timer wall_clock \
  --warmup 5 --samples 30 --inner-iterations 20
```

The correctness evaluator always runs first. A failed correctness result writes
`performance_status=skipped`, `skip_reason=correctness_gate_failed`, and no
performance samples. A hard timeout or crash after the worker has entered
warmup/sampling is instead persisted as a performance failure with zero raw
samples, so resume does not repeat it.

## Timers and selection

CUDA Event measures an ordinary eager launch with device events. CUDA Graph
captures the complete inner loop once and measures one replay per raw sample.
Immediately after capture, preparation performs one additional replay plus
synchronization so lazy graph instantiation is charged to `graph_capture_ms`
instead of the first steady-state sample.
These human-readable names correspond to the canonical timer identifiers below.

The canonical timer names are:

- `wall_clock`: end-to-end host elapsed time, synchronizing CUDA devices found
  in the actual input/output tensors before and after each sample;
- `cuda_event`: device elapsed time from a CUDA event pair per raw sample;
- `cuda_graph`: one captured graph contains the complete inner loop, one graph
  replay is measured per sample, then elapsed time is divided by the inner
  iteration count;
- `auto`: attempts `cuda_graph` and visibly falls back to `cuda_event` only when
  graph preparation is unsupported.

Explicit `cuda_graph` does not silently fall back. Every measured role records
`requested_timer`, `effective_timer`, and `fallback_reason`. Results using
different effective timers fail the formal performance gate.

## Stages and configuration

Import and candidate build are distinct stages. Runtime JIT commonly occurs
during the first call or graph capture; those durations are excluded from
steady state. The schema does not expose a separate `jit_ms` field. First call,
warmup, graph capture, and sampling are measured separately. Only steady-state
samples contribute to latency statistics. The stage totals are retained as `import_ms`, `build_ms`,
`first_call_ms`, `warmup_ms`, `graph_capture_ms`, and `steady_state_ms`, with
separate reference-prefixed preparation fields where applicable.

Engine defaults are 5 warmup calls, 30 raw samples, and 20 inner iterations.
Configuration resolves in this order: CLI override, suite value, operator
manifest value, engine default. Relevant CLI options include `--timer`,
`--warmup`, `--samples`, `--inner-iterations`, `--max-slowdown-pct`,
`--min-speedup`, `--max-cv`, and `--max-memory-bytes`.

## Raw data and statistics

After both roles complete independent first-call, warmup, and graph capture,
steady samples follow fixed `R-C-C-R` order. `order_index` is the global
execution order in this interleaving, not a rank. `performance_samples.csv`
schema v3 stores each reference/candidate sample
separately, including `elapsed_ms`, `per_call_ms`, `inner_iterations`, timer
provenance, a global order index, and compact output-path maps for both
implementations' dtype and shape. `results.csv` schema v3 stores the aggregate
mean, median, min, max, p50, p90, p95, p99, population standard deviation, and
coefficient of variation (CV) for both roles. Percentiles use linear Type-7
interpolation.

Fewer than five samples or CV above the resolved threshold marks a measurement
`unstable` and fails a formal performance request.

## GPU lock, telemetry, gates, and comparison

CUDA operators in `all` or `performance` mode acquire
`.runtime/locks/<GPU UUID>.lock` before worker launch, including when using
`wall_clock`. Lock metadata contains PID, run ID, UTC start, logical/visible
device, and UUID. Only a definitely absent owner PID is reclaimed; malformed
metadata, permission uncertainty, and possible PID reuse time out conservatively.

GPU identity, driver/CUDA version, `CUDA_VISIBLE_DEVICES`, and best-effort
other-compute-process state are recorded before sampling. Unknown telemetry is
empty, never fake zero. Competing compute activity makes a result non-formal.
Correctness failures normally produce no samples; `--perf-on-correctness-fail`
keeps diagnostic samples permanently non-formal.

Speedup is `reference_median/candidate_median`; zero or non-finite denominators
remain empty. Use `bench compare --result EVAL --baseline-result BASE` or
`bench compare --run RUN --baseline-run BASE_RUN`. Contract, case/seed,
environment, and timer provenance must match; incompatibility exits 2 without
calculating a misleading speedup.

## Theoretical cost model

`spec.py` 的 `cost_model(case)` may return theoretical FLOPs, estimated bytes,
and optional throughput units. With a positive candidate median latency the engine derives
TFLOP/s, effective decimal GB/s, arithmetic intensity, and throughput. Missing
cost data is recorded as unavailable, never as zero. These are theoretical
rates rather than profiler measurements. They appear in `results.csv` as
`flops`, `estimated_bytes`, `tflops`, `effective_bandwidth_gbps`,
`arithmetic_intensity`, and `throughput`; raw timing remains in `per_call_ms`
and is never multiplied or replaced by the cost model.

See [CSV schemas](csv-schema.md) for persisted fields and
[result/resume layout](result-layout.md) for artifact placement.
