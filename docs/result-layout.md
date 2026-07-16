# Result layout

```text
results/<operator_id>/<candidate_id>/<evaluation_id>/
  evaluation_manifest.json
  results.csv
  correctness_outputs.csv
  performance_samples.csv
  summary.md
  diagnostics/
  logs/{controller.jsonl,worker.jsonl,stdout.log,stderr.log}
```

Each case atomically updates CSV artifacts. Performance samples have only a
schema header in Phase 7. History/latest publish only after every expected result
is terminal without infrastructure failure. Resume uses `run_index.csv` and
deduplicates by `result_id`.

Applies to manifest schema v1 and CSV schema v1.

Candidate source and result directories share the same two identity keys:

```text
operators/candidates/<operator_id>/<candidate_id>/
results/<operator_id>/<candidate_id>/
results/<operator_id>/<candidate_id>/<evaluation_id>/
```

The path API validates every identifier and verifies the resolved path remains
inside its configured root. Absolute identifiers, traversal, separators, and
invalid or non-UTC timestamps are rejected.

The evaluation ID format is
`<YYYYMMDDTHHMMSSZ>__<environment-short-hash>__<run-short-id>`. A global
`run_id` correlates evaluations in `results/run_index.csv`; it is never a
directory name.

```text
results/
├── run_index.csv
└── <operator_id>/<candidate_id>/
    ├── history.csv
    ├── latest.json
    └── <evaluation_id>/
        ├── evaluation_manifest.json
        ├── results.csv
        ├── correctness_outputs.csv
        ├── performance_samples.csv
        ├── summary.md
        ├── diagnostics/
        └── logs/
            ├── controller.jsonl
            ├── worker.jsonl
            ├── stdout.log
            └── stderr.log
```

Phase 5 stores the strict worker request/response and controller diagnostics
under `diagnostics/`. Full exception tracebacks are diagnostic text files;
`results.csv` and the response protocol contain only a short message plus a
relative diagnostic path. Worker failures do not rewrite a previously durable
result row.

`logs/worker.jsonl` records schema-v1 events in lifecycle order:
`DISCOVERED`, `WORKER_STARTED`, paired import/build/correctness/warmup/sampling
events, `REPORT_WRITTEN`, and `WORKER_EXITED`. Long active stages may insert
`HEARTBEAT` records. In Phase 5 the correctness, warmup, and sampling pairs end
with `status=skipped`; they must not be interpreted as measured results.
`controller.jsonl` records the supervisor outcome and truncation flags.
Both JSONL files append one result transcript/record within the evaluation;
running a later case must not erase an earlier case. Event sequence numbers are
continuous across the evaluation and each `result_id` still has a complete,
independently valid lifecycle.

Retries of the same `result_id` are separate attempts. Every `DISCOVERED`
record starts one attempt transcript, even when its identity and result ID
match an earlier transcript. Request/response/controller-diagnostic filenames
include a monotonically increasing per-result attempt number, so a crash can
never consume or overwrite a prior successful response.

Stdout and stderr are drained continuously to prevent a verbose candidate or
compiler from blocking on a full pipe. Each formal log has a configured byte
limit and receives an explicit truncation marker when exceeded. Console/in-
memory summaries have a smaller independent bound. The stdout/stderr byte cap
applies to the whole evaluation, not separately to each case, and the
truncation marker is written at most once.

Initialization records a mirrored evaluation in `run_index.csv`, allowing an
interrupted run to be found. `history.csv` receives exactly one row only after
the evaluation reaches `complete`; `latest.json` is atomically replaced last
and therefore never points at planned, running, failed, interrupted, or
half-written output. Failed and interrupted artifacts remain in place for
diagnosis and resume.

`evaluation_manifest.json` stores schema version, full identity, lifecycle
state, original command, resolved configuration, both source hashes, suite and
mode, and the environment snapshot/fingerprint. Allowed state changes are
strictly `planned -> running -> complete|failed|interrupted`, plus
`interrupted -> running` when the same compatible evaluation is resumed. A
resume clears the previous terminal reason. `failed` and `complete` remain
terminal. A repeated identical state update is idempotent; any other transition
is rejected.

`latest.json` is rebuilt from complete `history.csv` rows, ordered by canonical
`completed_at_utc` and then `evaluation_id`. Retrying publication of an older
complete evaluation can repair a missing latest file but cannot move latest
backwards.

Resume resolves a run through `run_index.csv`, checks that the indexed path is
the path implied by its identity, then validates manifest schema, identity,
sources, resolved config, suite/mode, and environment. Existing `result_id`
values in `results.csv` are returned as complete work and conflicting rows may
not replace them. See [CSV schema v1](csv-schema.md).

Formal text, JSON, and CSV mutations use a same-directory temporary file,
flush/fsync, and atomic replacement. The controller remains the only formal
artifact/CSV writer; workers write only their dedicated response, diagnostic,
and append-only event channels.
