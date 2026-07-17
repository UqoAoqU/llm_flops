# Benchmark CLI

```text
bench run --suite smoke
bench run --mode correctness --operator OP --candidate CANDIDATE [--case CASE] [--seed N]
bench run --resume RUN_ID
bench summarize PATH
bench summarize --operator OP --candidate CANDIDATE --evaluation EVALUATION
bench compare --result EVALUATION --baseline-result BASELINE
bench compare --run RUN_ID --baseline-run BASELINE_RUN_ID
```

Default execution continues after gate failures. `--fail-fast` stops remaining
jobs while retaining completed rows. `--dry-run` keeps its Phase-3 JSON behavior
and creates no artifacts. Exit codes: 0 pass, 1 gate, 2 usage/config/registry,
3 infrastructure, 130 Ctrl-C.

Discovery commands remain available:

```bash
bench list
bench list --operator 'minimal_*' --candidate 'task_*'
bench validate
bench validate --operator 'minimal_*'
```

Selectors use case-sensitive shell-style globs. Output is sorted by operator
and candidate ID. An invalid registry or a selector with no matches exits with
status 2. Discovery checks Python file locations without importing modules.

Phase 3 adds environment inspection and artifact-free planning:

```bash
bench env
bench env --json
bench run --suite smoke --dry-run
bench run --suite regression --dry-run \
  --operator 'minimal_*' --candidate 'task_*' --case 'small*' \
  --tag representative --exclude-operator '*_experimental' --seed 7
```

`--operator`, `--candidate`, `--case`, `--tag`, `--exclude-operator`, and
`--seed` are repeatable. Values within one selector category are ORed;
different categories are ANDed; excludes apply last. `--mode` overrides the
suite. `--output-root`, `--evaluation-id` (exactly one selected candidate), and
`--resume` affect planning. An existing evaluation path is rejected unless
`--resume` is present. Phase 4 provides the durable run-index/manifest resume
reader used by the execution CLI. The Controller/Worker API isolates import,
build, correctness, warmup, and performance sampling.

Dry-run prints stable-key JSON and has no formal filesystem side effects. Its
environment fingerprint is explicitly marked `provisional/dry-run` and is a
SHA-256 of normalized dependency-lock contents. A non-dry run uses a runtime
fingerprint and durable artifacts.

Artifact writers and the managed worker back the correctness CLI. Their
state and compatibility rules are documented in [Result layout](result-layout.md)
and [CSV schemas](csv-schema.md); no CLI command bypasses those rules.

Performance threshold flags are `--max-slowdown-pct`, `--min-speedup`,
`--max-candidate-median-ms`, `--max-cv`, `--max-memory-bytes`, and
`--unsupported-policy fail|allow`. `--perf-on-correctness-fail` retains only
non-formal diagnostic samples. CLI overrides suite, operator, then defaults.
