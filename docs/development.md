# Development and validation

## Environment

```bash
./bootstrap.sh
./bench.sh --help
```

The bootstrap command reuses the configured ROCm virtual environment through a
safe `.runtime/venv` symlink and validates the checked-in lock. It does not
install dependencies. Both launchers reset `PYTHONPATH` and keep uv, extension,
FlashInfer, Triton, and XDG caches under `.runtime/cache`.

## Required gates

Before committing an engine change, run:

```bash
bash -n bootstrap.sh bench.sh run.sh
.runtime/venv/bin/python -m unittest discover -s tests -v
./run.sh check
./bench.sh env --json
./bench.sh validate
git diff --check
```

For accelerator changes, also run `mi300x_smoke` on one idle physical MI300X:

```bash
ROCR_VISIBLE_DEVICES=0 HIP_VISIBLE_DEVICES=0 CUDA_VISIBLE_DEVICES=0 \
  ./bench.sh run --suite mi300x_smoke
```

Do not commit `.runtime/`, `results/`, logs, caches, bytecode, JIT output, build
output, or benchmark artifacts. Do not make an operator pass by weakening its
reference, cases, tolerance, performance gates, or tests.

## Test layers

- Registry, manifest, selector, and planning tests are accelerator-free.
- Controller/Worker tests use real subprocesses for protocols, timeout,
  signals, process-group cleanup, and bounded logs.
- Correctness tests cover input isolation, normalization, comparators,
  determinism, and diagnostics.
- Performance tests cover timer selection, sample ordering, statistics, cost
  models, physical GPU locking, and eligibility gates.
- Reporting tests cover strict schemas, migration, atomic writes, resume,
  summaries, and comparison.
- Operator tests cover manifest/spec contracts, semantic oracles, control
  candidates, and MI300X smoke execution.

Controller/Worker wire formats, manifests, and CSV files are versioned
protocols. Adding a field requires model, strict parser, writer/reader,
migration, documentation, and round-trip tests. Unknown fields must never be
silently ignored, and candidate code or tensors must never cross the worker
boundary through pickle.
