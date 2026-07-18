# Test tiers

The tier runner prints or executes stable command groups:

```bash
.runtime/venv/bin/python tools/run_test_tier.py cpu --dry-run
.runtime/venv/bin/python tools/run_test_tier.py gpu-smoke --dry-run
.runtime/venv/bin/python tools/run_test_tier.py b200-regression --dry-run
.runtime/venv/bin/python tools/run_test_tier.py full --dry-run
```

Remove `--dry-run` only when the required environment is available.

## CPU unit and contract

Runs full unittest discovery, legacy `run.sh check`, and Registry validation.
This tier covers static discovery, planning, schema migration, row semantics,
worker protocol/failure injection, artifact atomicity, resume, compare, docs,
and command contracts without requiring CUDA.

## GPU smoke

Runs `./run.sh smoke` and `./bench.sh run --suite smoke` with
`CUDA_VISIBLE_DEVICES=0`. Check `nvidia-smi` first and do not overlap another
formal benchmark.

## B200 regression

Runs the predeclared prefill and decode legacy/new comparisons. It requires an
idle physical GPU 0 and writes beneath `.runtime/regression/`, which is ignored.
The exact shapes, timers, evidence, and immutable thresholds are defined in
[the migration guide](migration-llm-flops.md).

## Full/manual/nightly

The full tier combines CPU, GPU smoke, both B200 regressions, the regression
suite, and DeepSeek prefill/decode suites. It is intentionally manual/nightly
because cold JIT and full model shapes are expensive.

Fault-injection coverage includes candidate exception/segfault, import/JIT
hang, timeout process-group cleanup, OOM classification, Ctrl-C cleanup,
interrupted resume, stale GPU lock recovery, atomic text/CSV replacement, and
multi-directory legacy import rename recovery. These tests must validate the
failure state; skipping/deleting them or relaxing contracts is not a fix.
