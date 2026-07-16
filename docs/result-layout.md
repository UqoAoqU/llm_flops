# Result layout

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
flush/fsync, and atomic replacement. The controller is the only writer; Phase
4 intentionally does not add workers or concurrent evaluators.
