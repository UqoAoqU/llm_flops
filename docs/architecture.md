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
- `environment`: future environment snapshots and fingerprints.
- `projection`: optional per-call to model-level projections.

Phase 2 implements the registry boundary and safe source/result identity paths.
Execution, worker imports, planning, correctness, performance, profiling, and
Nsight integration remain deferred.

The lifecycle begins with `FilesystemRegistry` scanning references and
candidates. It parses YAML with `safe_load`, statically resolves entrypoint
module files, hashes source trees, and publishes a frozen snapshot. A later
worker will perform imports only after planning and isolation exist.

Candidate source directories mirror result directories by operator and
candidate ID. Evaluation directories add a third validated identity component;
there is no run-ID result root.
