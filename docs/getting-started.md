# Getting started

## Prepare the MI300X runtime

The supported environment is Linux, Python 3.11, ROCm/HIP 7.0, PyTorch
`2.10.0+rocm7.0`, and `gfx942`. Configure `GPU_VENV`, `SGLANG_ROOT`, and
`AITER_ROOT` in `~/.config/agent4kernel/env.sh`, then run:

```bash
./bootstrap.sh
./run.sh check
./bench.sh --version
./bench.sh env --json
```

Bootstrap validates the lock and links `.runtime/venv` to the shared
`$GPU_VENV`. It performs no package download or GPU dependency installation.
See the [MI300X runtime guide](mi300x.md) for source fingerprints, cache
isolation, device mapping, and lock behavior.

## Discover and validate

```bash
./bench.sh list
./bench.sh list --operator 'example_*'
./bench.sh validate --operator example_rocm_bf16_gemm
./bench.sh validate
```

`list` and `validate` use static discovery and do not import candidate code.
Operator and candidate selectors are case-sensitive shell globs.

## Inspect a deterministic execution plan

```bash
./bench.sh run --suite mi300x_smoke --dry-run
./bench.sh run --mode correctness --dry-run \
  --operator example_rocm_bf16_gemm --case smoke --seed 0
```

Dry-run emits deterministic JSON without initializing the accelerator or
creating result directories.

## Run correctness and performance

Use all three visibility variables consistently because ROCm software in the
stack may inspect any one of them:

```bash
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
export CUDA_VISIBLE_DEVICES=0

./bench.sh run --mode correctness \
  --operator example_rocm_bf16_gemm --case smoke --seed 0

./bench.sh run --suite mi300x_smoke
```

The reference and candidate receive independent clones of the same deterministic
canonical input. Performance always runs correctness first. `auto` timing tries
HIP graph capture through `torch.cuda.CUDAGraph`; if graph preparation is
unsupported, it records an explicit fallback to HIP events. Both roles must use
the same effective timer for a formal result.

## Read, resume, and compare

```bash
./bench.sh summarize results/OPERATOR_ID/CANDIDATE_ID/EVALUATION_ID
./bench.sh run --resume RUN_ID
./bench.sh compare --result EVALUATION_ID \
  --baseline-result BASELINE_EVALUATION_ID
```

Resume runs only missing result IDs. Compare requires compatible contracts,
cases, seeds, source/environment fingerprints, accelerator provenance, and
timer provenance.

Exit codes are `0` for pass, `1` for a completed correctness/performance gate
failure, `2` for usage/configuration/registry errors, `3` for worker or artifact
infrastructure failure, and `130` after an interrupted worker is cleaned up.
