# Operator contract

Each reference lives at `operators/references/<operator_id>/`. The directory
name and the `operator_id` in `operator.yaml` must be identical and match
`^[a-z][a-z0-9_]{2,79}$`.

`operator.yaml` schema version 1 has these required fields:

```yaml
schema_version: 1
operator_id: minimal_cpu_add
contract_version: 1
description: Add two Python numbers on the CPU.
reference_entrypoint: implementation:operator
spec_entrypoint: spec:SPEC
device_types: [cpu]
tags: [example, cpu]
correctness:
  default_comparator: exact
  rtol: 0.0
  atol: 0.0
  equal_nan: false
  determinism_repeats: 1
performance:
  timer: wall_clock
  graph_mode: disabled
  warmup: 1
  samples: 3
  inner_iterations: 1
  timeout_s: 30
  regression_threshold_pct: 5.0
  min_speedup: 1.0
  max_candidate_median_ms: 10.0
  max_cv: 0.1
  max_memory_bytes: 1073741824
  unsupported_policy: fail
  gpu_lock_timeout_s: 600
cost_model: spec:cost_model
```

Unknown fields and unknown schema versions are errors. Entrypoints use
`<module>:<attribute>`. Discovery only checks that the corresponding `.py`
file is below the reference root; it never imports that module. See the
[copyable CPU example](examples/minimal-operator/operator.yaml).

The trusted `spec_entrypoint` implements `OperatorSpec` and declares its stable
`operator_id: str`: `cases()`,
`make_inputs()`, `clone_inputs()`, `normalize_output()`, `comparator()`, and
`cost_model()`. Controller-side planning imports only `cases()` metadata and
still rejects torch imports there. Runtime methods execute inside an isolated
worker; tensor objects never enter the JSON protocol.

An optional `workspace_bytes(case) -> int | None` hook reports candidate
workspace demand. Missing/unavailable values persist as empty fields, never zero.

`make_inputs(case, context)` receives explicit seeded CPU/CUDA generators and
returns `benchmark_engine.correctness.InputBundle`. `clone_inputs()` must
produce physically isolated tensor storage and mutable containers while
preserving aliases within one bundle. Any state changed in-place must be
registered in `observed_state`, otherwise it is outside the correctness
contract. `normalize_output()` maps implementation-specific return values to
the reference's semantic output before deterministic leaf normalization.

See [Correctness](correctness.md) for comparator and tolerance requirements.
