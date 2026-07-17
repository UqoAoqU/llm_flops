# Candidate guide

Place a candidate at
`operators/candidates/<operator_id>/<candidate_id>/`. A minimum candidate has
only `implementation.py` with an `operator` attribute; it therefore uses the
default `implementation:operator` entrypoint. The
[minimal candidate](examples/minimal-candidate/implementation.py) is copyable.

Candidate IDs may use any non-empty, path-safe directory name that is unique
under the same `operator_id`. Absolute paths, `.`/`..`, `/`, and `\\` are
rejected, and names that differ only by case are treated as collisions for
portable result layouts. The recommended, but optional, form is
`<task_identifier>__<YYYYMMDDTHHMMSSZ>__<8-or-more-lowercase-hex>`.

Complex candidates can add a strict `candidate.yaml`:

```yaml
schema_version: 1
operator_id: minimal_cpu_add
candidate_id: task_0042__20260716T081500Z__01234567
entrypoint: implementation:operator
framework: python
build:
  command: [python, build.py]
  timeout_s: 600
metadata:
  task_id: task_0042
```

Build commands are argv string arrays. Shell command strings are rejected,
and metadata must be JSON-safe. All source files, modes, and relative POSIX
paths affect the separately recorded source hash, except caches, bytecode,
build output, egg-info, and common editor temporary files. Symlinks are
rejected. The candidate name is not required to contain or match this hash.

`candidate.yaml` also affects the hash. Its YAML mapping is converted to sorted
compact JSON after removing only the `candidate_id` field, so a rename does not
change source identity. Every other manifest change changes the recorded source
hash. Choosing a new candidate name after a source change remains recommended
for readable history, but is not a Registry acceptance condition.
