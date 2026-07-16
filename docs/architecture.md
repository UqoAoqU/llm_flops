# Architecture

The engine uses a `src/benchmark_engine` package and keeps the legacy scripts
at the repository root until migration is complete.

The controller-facing data contracts live in `benchmark_engine.models`.
They are frozen dataclasses with explicit `to_dict()` and `from_dict()`
methods. Their wire forms contain JSON-safe metadata only: tensors, callables,
and arbitrary runtime objects are rejected. Paths, tuples, frozensets, and
stable status enums survive a JSON round trip.

The package boundaries are:

- `registry`: strict manifest discovery, static entrypoint checks, and source
  hashing. It returns immutable controller metadata and never imports modules.
- `execution`: future controller and subprocess worker infrastructure.
- `correctness`: future comparators and correctness gates.
- `performance`: future timers, sampling, statistics, and performance gates.
- `reporting`: future atomic artifacts and CSV indexes.
- `environment`: adapters that reuse the legacy collector and fingerprints.
- `projection`: optional per-call to model-level projections.

Phase 3 implements strict suites, selectors, resolved per-job configuration,
environment identity, and deterministic plans. Execution, artifacts, workers,
correctness, performance, profiling, and Nsight integration remain deferred.

The lifecycle is suite/config load -> static registry discovery and validation
-> trusted reference-spec metadata import -> selector expansion -> immutable
`EvaluationPlan`. Only the reference `spec_entrypoint` may be imported by the
controller; candidate modules are never imported. Each job carries its complete
JSON-safe resolved configuration and final result/evaluation identities.

`bench run --dry-run` hashes normalized lock-file contents to obtain a
`provisional/dry-run` planning fingerprint. It deliberately does not collect
the environment, import torch, initialize CUDA, create result directories, or
write formal artifacts. A later execution phase replaces that provisional
identity with the real fingerprint returned by the shared legacy collector.

Candidate source directories mirror result directories by operator and
candidate ID. Evaluation directories add a third validated identity component;
there is no run-ID result root.
