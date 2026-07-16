# Candidate guide

Place a candidate at
`operators/candidates/<operator_id>/<candidate_id>/`. A minimum candidate has
only `implementation.py` with an `operator` attribute; it therefore uses the
default `implementation:operator` entrypoint. The
[minimal candidate](examples/minimal-candidate/implementation.py) is copyable.

Candidate IDs have the form
`<task_identifier>__<YYYYMMDDTHHMMSSZ>__<8-or-more-lowercase-hex>`. The final
component must be a prefix of the stable source SHA-256 computed by the
registry.

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
paths affect the source hash, except caches, bytecode, build output, egg-info,
and common editor temporary files. Symlinks are rejected.

`candidate.yaml` also affects the hash. To avoid an ID/hash circularity, its
YAML mapping is converted to sorted compact JSON after removing only the
`candidate_id` field. Consequently every other manifest change requires a new
candidate ID suffix.
