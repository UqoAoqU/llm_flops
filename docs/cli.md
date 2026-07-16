# Benchmark CLI

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
reader used by the later execution CLI; because execution is still deferred,
non-dry `bench run` continues to exit 2 and does not mutate artifacts yet.

Dry-run prints stable-key JSON and has no formal filesystem side effects. Its
environment fingerprint is explicitly marked `provisional/dry-run` and is a
SHA-256 of normalized dependency-lock contents. A non-dry run exits 2 with
`execution not available until later phase`.

Artifact writers are currently a Python API for the future controller. Their
state and compatibility rules are documented in [Result layout](result-layout.md)
and [CSV schema v1](csv-schema.md); no Phase 4 CLI command bypasses those rules.
