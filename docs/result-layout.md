# Result layout

Candidate source and result directories share the same two identity keys:

```text
operators/candidates/<operator_id>/<candidate_id>/
results/<operator_id>/<candidate_id>/
results/<operator_id>/<candidate_id>/<evaluation_id>/
```

The path API validates every identifier and verifies the resolved path remains
inside its configured root. Absolute identifiers, traversal, separators, and
invalid or non-UTC timestamps are rejected.

The reserved evaluation ID format is
`<YYYYMMDDTHHMMSSZ>__<environment-short-hash>__<run-short-id>`. Phase 2 only
validates this format; generation and result writing arrive in Phase 3 and
later. A global `run_id` may eventually correlate records, but there is no API
that constructs `results/<run_id>`.
