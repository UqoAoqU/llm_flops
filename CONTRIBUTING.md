# Contributing

Benchmark-engine work must preserve the existing `run.sh` commands and the
DeepSeek V4 and GLM-5 benchmark behavior until their documented migration
phases.

Before sending a change:

1. Bootstrap with `./bootstrap.sh`.
2. Run the full unittest suite.
3. Run `./run.sh check`.
4. Run `git diff --check`.
5. Confirm that `.runtime/`, `logs/`, `results/`, caches, and GPU artifacts
   are not part of the change.

Cross-process contracts must remain explicit and JSON-safe. Do not use pickle
for candidates, tensors, or worker messages. New direct runtime dependencies
must be declared in `pyproject.toml` and pinned in
`requirements/benchmark-lock.json`.

Keep changes within the current implementation phase. Registry discovery,
worker execution, correctness, performance, reporting, and profiling should
arrive with their corresponding contracts and tests rather than as placeholder
commands.
