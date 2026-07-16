# Self-Contained Benchmark Environment Design

## Goal

Make `llm_flops` reproducible for other users on the same B200 server without
shipping the existing 11 GB virtual environment. Users must be able to test
individual operators first, then run the DeepSeek V4 Pro prefill and decode
profiling with the same model geometry, invocation counts, backends, and timing
method used to establish the project baseline.

The package may use the server's existing NVIDIA driver, CUDA development
toolchain, Python 3.12, and network access. All project changes are restricted
to the `llm_flops` directory. Dependency source repositories are fetched and
built but never modified by this project.

## Distribution Model

The repository contains benchmark sources, exact dependency manifests, setup
and launch scripts, and validation tools. It does not contain a prebuilt Python
environment or vendored dependency source trees.

`bootstrap.sh` creates a repository-local ignored `.runtime/` directory. It
downloads exact dependency revisions, builds required CUDA extensions, and
installs them into `.runtime/venv`. Temporary source checkouts and build output
also remain under `.runtime/`.

The backend manifest records:

- source URL and immutable commit or release tag;
- expected package version;
- build options that affect generated kernels;
- supported Python, CUDA, and GPU architecture;
- installed artifact hashes where they are stable enough to validate.

The current environment is the reference for resolving these values. In
particular, SGL Kernel is version 0.4.4 from the SGLang source revision
`195933599`. The top-level SGLang serving package is not installed unless a
benchmark directly imports it. DeepGEMM 0.1.4 provenance must be resolved to an
immutable source revision before the bootstrap is considered complete.

## User Interfaces

The original individual-operator scripts remain directly executable with the
repository-local Python environment. A common launcher provides stable names:

```bash
./run.sh op dsa_indexer
./run.sh op dsa_flashmla
./run.sh op dsa_projection
./run.sh op mla_flashmla
./run.sh op moe_deepgemm
```

The launcher also exposes the complete profiling entry points:

```bash
./run.sh prefill --quant-profile fp8_mxfp8 \
  --m 1024,2048,4096 --context 65536
./run.sh decode --quant-profile fp8_mxfp8 \
  --m 16,32 --context 65536
```

Arguments after the command are forwarded unchanged. The launcher clears an
external `PYTHONPATH`, selects `.runtime/venv`, sets repository-local cache and
build directories, and fails with an actionable bootstrap instruction when the
environment is absent or invalid.

## Baseline Contract

Environment reproducibility and benchmark reproducibility are separate checks.
The environment checker validates package versions, CUDA availability, B200
capability, required Python symbols, and loadability of compiled extensions.
The benchmark code remains the authority for DeepSeek V4 Pro geometry and
accounting:

- 61 transformer layers: 30 C4, 30 C128, and one ratio-0 layer;
- attention and indexer use full logical shapes without tensor-parallel slicing;
- MoE uses EP24, 384 global experts, 16 local experts, Top-6, and intermediate
  size 3072;
- prefill query lengths are 1024, 2048, and 4096 with raw KV length 65536 and
  no append;
- decode batches are 16 and 32 with raw KV length 65536;
- C4 and C128 derived KV lengths are 16384 and 512;
- timing uses five warmups and 20 CUDA Graph replays, excluding static weight
  preparation and JIT compilation.

Formal results record the resolved environment fingerprint alongside the CSV
or in a sidecar manifest. A mismatch does not silently produce an official
baseline result.

## Dependency Policy

Python dependencies are locked to exact versions. CUDA backend projects are
locked to immutable source revisions rather than mutable branches. The setup
script may compile extensions for SM100 but must not apply patches to dependency
sources. If the required revision cannot provide a symbol used by the current
benchmarks, bootstrap stops and reports the missing symbol instead of selecting
a different backend.

Network downloads may be cached under `.runtime/cache`. Re-running bootstrap is
idempotent: it reuses matching downloads and builds, while a changed lock file
creates or refreshes only the affected environment components.

## Validation

Validation has three levels:

1. Static unit tests verify lock parsing, command dispatch, environment
   fingerprints, baseline configuration, and error reporting without requiring
   a GPU.
2. Environment validation imports PyTorch, SGL Kernel, DeepGEMM, and FlashInfer;
   checks exact versions and required operator symbols; and verifies CUDA and
   SM100 availability.
3. GPU smoke tests run one minimal case for every individual-operator family,
   followed by reduced prefill and decode cases through the complete benchmark
   driver. Smoke tests validate execution and output structure, not performance
   thresholds.

The formal baseline commands remain separate from smoke tests so setup does not
accidentally overwrite benchmark result files.

## Failure Handling

Setup errors identify the failed dependency, source revision, build command,
and retained log path. Runtime validation reports the expected and observed
environment fingerprints. Unsupported hardware, missing CUDA tools, missing
backend symbols, and external environment contamination are hard failures for
formal profiling.

Partial operator results continue to use the benchmark's explicit `unavailable`
status, but the launcher returns a nonzero status when a formal baseline run has
any unavailable row.

## Repository Scope

Expected additions are setup and launch shell scripts, version and backend lock
files, small Python tools, tests, and usage documentation. Existing benchmark
scripts may receive narrowly scoped changes for stable command-line operation
and environment fingerprint output. No file outside `llm_flops` is edited, and
existing unrelated working-tree changes are left intact.
