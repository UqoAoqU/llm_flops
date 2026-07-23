# Test tiers

The tier runner prints or executes stable command groups:

```bash
.runtime/venv/bin/python tools/run_test_tier.py cpu --dry-run
.runtime/venv/bin/python tools/run_test_tier.py mi300x-smoke --dry-run
.runtime/venv/bin/python tools/run_test_tier.py full --dry-run
```

Remove `--dry-run` only when the required environment is available.

## CPU unit and contract

Runs full unittest discovery, legacy `run.sh check`, and Registry validation.
This tier covers static discovery, planning, schema migration, row semantics,
worker protocol/failure injection, artifact atomicity, resume, compare, docs,
and command contracts without requiring an accelerator.

## MI300X smoke

Runs `./run.sh smoke` and `./bench.sh run --suite mi300x_smoke` with logical
device 0 selected consistently through `ROCR_VISIBLE_DEVICES`,
`HIP_VISIBLE_DEVICES`, and `CUDA_VISIBLE_DEVICES`. The suite validates the
BF16 GEMM control path with a smoke and representative shape. KFD topology
resolves the physical gfx942 device and its render minor before the
cross-process lock is acquired.

## Full/manual/nightly

The full tier combines the CPU/contract gates and the MI300X smoke suite. It
intentionally has no B200 regression or NVIDIA-only operator dependency.

Fault-injection coverage includes candidate exception/segfault, import/JIT
hang, timeout process-group cleanup, OOM classification, Ctrl-C cleanup,
interrupted resume, stale GPU lock recovery, atomic text/CSV replacement, and
multi-directory legacy import rename recovery. These tests must validate the
failure state; skipping/deleting them or relaxing contracts is not a fix.
