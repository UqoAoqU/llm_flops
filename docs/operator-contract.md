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
cost_model: spec:cost_model
```

Unknown fields and unknown schema versions are errors. Entrypoints use
`<module>:<attribute>`. Discovery only checks that the corresponding `.py`
file is below the reference root; it never imports that module. See the
[copyable CPU example](examples/minimal-operator/operator.yaml).

In Phase 3, the trusted `spec_entrypoint` object exposes
`cases() -> tuple[CaseSpec, ...]`. Cases are pure metadata: they declare case
IDs, symbols, default seeds, and tags, and must not import torch or construct
tensors. Shapes, tolerances, and reference semantics remain operator-owned and
cannot be redefined by a suite.
