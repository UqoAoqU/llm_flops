# Architecture

The engine uses a `src/benchmark_engine` package and keeps the legacy scripts
at the repository root until migration is complete.

The controller-facing data contracts live in `benchmark_engine.models`.
They are frozen dataclasses with explicit `to_dict()` and `from_dict()`
methods. Their wire forms contain JSON-safe metadata only: tensors, callables,
and arbitrary runtime objects are rejected. Paths, tuples, frozensets, and
stable status enums survive a JSON round trip.

The package boundaries are:

- `registry`: future manifest discovery and validation.
- `execution`: future controller and subprocess worker infrastructure.
- `correctness`: future comparators and correctness gates.
- `performance`: future timers, sampling, statistics, and performance gates.
- `reporting`: future atomic artifacts and CSV indexes.
- `environment`: future environment snapshots and fingerprints.
- `projection`: optional per-call to model-level projections.

Only package boundaries exist in Phase 1. Registry behavior, execution,
profiling, and Nsight integration are intentionally deferred.

The planned lifecycle is discovery, validation, deterministic planning,
isolated correctness execution, gated performance execution, and atomic
reporting. Candidate source directories will mirror their result directories;
the approved design defines the complete contract.
