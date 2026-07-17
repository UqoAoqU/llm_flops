# Architecture

Phase 8 lifecycle: Registry → runtime environment identity → deterministic plan
→ ArtifactWriter → isolated case worker → in-worker correctness gate → staged
performance evaluator → controller-only CSV projection → completion or resume.
The controller never imports candidate modules. Import, build/JIT, first call,
warmup, graph capture, and steady-state sampling remain distinct stages.

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
- `execution`: strict JSON protocol, managed subprocess worker, process-group
  cleanup, stage timeouts, bounded logs, and structured lifecycle events.
- `correctness`: runtime-only input bundles, output normalization, comparators,
  bounded diagnostics, and the reference/candidate evaluator.
- `performance`: wall-clock/CUDA timers, raw sampling, statistics, and
  theoretical cost rates. Fair scheduling and performance gates remain future work.
- `reporting`: versioned CSV tables, atomic artifacts, mirrored indexes, and
  resume discovery. The controller is the sole formal writer.
- `environment`: adapters that reuse the legacy collector and fingerprints.
- `projection`: optional per-call to model-level projections.

Phase 3 implements strict suites, selectors, resolved per-job configuration,
environment identity, and deterministic plans. Phase 4 adds the durable
reporting boundary. Phase 5 executes only the import/build skeleton in a
managed worker. Phase 6 implements correctness as an independent worker-local
library but deliberately does not change the wire schema or CLI; Phase 7
performs that integration. Phase 8 runs performance only after correctness and
persists hard performance failures without inventing samples.

Runtime `correctness.InputBundle` and normalized leaf values may contain
tensors and must remain in the worker. They are separate from the JSON-safe
metadata models in `benchmark_engine.models`. The evaluator constructs a
canonical seeded input once, creates storage-isolated reference/candidate
clones, synchronizes touched CUDA devices after each call, normalizes return
values and observed mutable state, then applies the reference-owned comparator.

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

The reporting lifecycle is initialization (`planned`, manifest, empty tables,
logs, run index), transition to `running`, then one terminal state. An
`interrupted` compatible evaluation may resume at `running`; `failed` and
`complete` remain terminal. Only `complete` publishes candidate `history.csv`
and monotonic `latest.json`. All formal
updates replace a flushed same-directory temporary file; duplicate row keys
are idempotent only when the complete row is identical. Resume validates the
durable manifest before returning the set of already completed `result_id`
values. The authoritative layouts and fields are in
[Result layout](result-layout.md) and [CSV schemas](csv-schema.md).

## Controller/worker boundary

The controller turns a prevalidated `EvaluationJob` into a schema-v1
`WorkerRequest`, starts `python -m benchmark_engine.execution.worker` with a
new POSIX session, continuously drains both output pipes, applies independent
import/build/correctness/performance hard deadlines, and consumes a strict
JSONL event stream. It never imports `spec.py`, reference implementation, or
candidate implementation. The worker validates every declared root and
entrypoint, then loads reference spec, reference implementation, and candidate
implementation in that order.

Requests contain identities, absolute validated roots, entrypoint strings,
`CaseSpec` metadata, resolved configuration, optional build argv, and timeout
values. They never contain pickle, callables, tensors, or shell command
strings. Responses contain a stable outcome (`success`, `error`, `timeout`,
`oom`, `crashed`, `unsupported`, or `interrupted`), failed stage, bounded error
summary, timings, and a diagnostic reference. Full tracebacks stay in
`diagnostics/`.

On timeout or interruption the controller sends TERM to the entire worker
process group, waits a short grace period, sends KILL, and reaps the leader.
It also removes descendants after normal worker exit, covering compiler and
build children such as Ninja, NVCC, and PTXAS. Heartbeats expose the current
stage, worker/known child PIDs, elapsed time, and a bounded recent log tail;
heartbeats never extend the hard stage deadline.

This subprocess boundary is failure isolation, not a security sandbox.
Candidate code still has the worker user's permissions. Untrusted code needs a
separate container/user/filesystem/network policy outside this engine.
