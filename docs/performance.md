# Performance measurement

Phase 8 adds staged latency measurement after the correctness gate. It records
raw reference and candidate samples, but deliberately does **not** implement
GPU locking, interleaved fairness, regression gates, speedup ranking, or
cross-run comparison; those policies belong to Phase 9.

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
different effective timers are not directly comparable.

## Stages and configuration

Import and candidate build are distinct stages. Runtime JIT commonly occurs
during the first call or graph capture; those durations are excluded from
steady state, but Phase 8 does not claim a separate `jit_ms` field. First call,
warmup, graph capture, and sampling are measured separately. Only steady-state
samples contribute to latency statistics. The stage totals are retained as `import_ms`, `build_ms`,
`first_call_ms`, `warmup_ms`, `graph_capture_ms`, and `steady_state_ms`, with
separate reference-prefixed preparation fields where applicable.

Engine defaults are 5 warmup calls, 30 raw samples, and 20 inner iterations.
Configuration resolves in this order: CLI override, suite value, operator
manifest value, engine default. Relevant CLI options are `--timer`, `--warmup`,
`--samples`, and `--inner-iterations`.

## Raw data and statistics

`performance_samples.csv` schema v2 stores each reference/candidate sample
separately, including `elapsed_ms`, `per_call_ms`, `inner_iterations`, timer
provenance, and an order index. `results.csv` schema v2 stores the aggregate
mean, median, min, max, p50, p90, p95, p99, population standard deviation, and
coefficient of variation (CV) for both roles. Percentiles use linear Type-7
interpolation.

Fewer than five samples or CV above 0.1 marks a measurement `unstable`. That
status is diagnostic in Phase 8; it is not a performance gate and does not
create a ranking.

## Theoretical cost model

An operator may return theoretical FLOPs, estimated bytes, and optional
throughput units. With a positive candidate median latency the engine derives
TFLOP/s, effective decimal GB/s, arithmetic intensity, and throughput. Missing
cost data is recorded as unavailable, never as zero. These are theoretical
rates rather than profiler measurements.

See [CSV schemas](csv-schema.md) for persisted fields and
[result/resume layout](result-layout.md) for artifact placement.
