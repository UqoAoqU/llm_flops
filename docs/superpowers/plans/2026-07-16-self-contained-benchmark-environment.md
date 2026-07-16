# Self-Contained Benchmark Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a small, reproducible distribution layer that installs the exact benchmark environment locally, preserves all individual-operator entry points, and runs DeepSeek V4 Pro profiling with a validated baseline fingerprint.

**Architecture:** A machine-readable lock file is the single source of truth for Python and CUDA backend versions. A Python environment module validates the installed runtime and computes its fingerprint; thin shell entry points create or select `.runtime/venv` and dispatch to testable Python CLI code. Existing benchmark kernels remain unchanged except for adding the environment fingerprint to formal result output and returning failure for unavailable rows.

**Tech Stack:** Bash, Python 3.12 standard library, uv 0.11+, PyTorch 2.11.0/CUDA 13, SGLang commit `19593359971ebc3582a74f000bf285488d993362`, SGL Kernel 0.4.4, SGL DeepGEMM 0.1.4, FlashInfer 0.6.12, unittest.

## Global Constraints

- Modify files only under `/home/claude-lsh/repo/llm_flops`; dependency source repositories must not be patched.
- Use the existing NVIDIA driver, CUDA development toolchain, `/usr/bin/python3.12`, and network access on the B200 server.
- Do not distribute `.runtime/`, downloaded sources, wheels, caches, JIT artifacts, or the existing `.venv-sglang0515`.
- Preserve direct execution of `dsa_indexer.py`, `dsa_flashmla.py`, `dsa_projection.py`, `mla_flashmla.py`, and `moe_deepgemm.py`.
- Preserve the DeepSeek V4 Pro 61-layer geometry, EP24/16-local-expert MoE configuration, KV 65536 cases, and five-warmup/20-replay formal timing contract.
- Leave unrelated dirty working-tree files and results untouched.

---

### Task 1: Runtime Lock And Environment Fingerprint

**Files:**
- Create: `requirements/benchmark-lock.json`
- Create: `benchmark_environment.py`
- Create: `tests/test_benchmark_environment.py`

**Interfaces:**
- Consumes: Python distribution metadata and `torch.cuda` runtime information.
- Produces: `load_lock(path: Path) -> dict`, `collect_environment(lock: dict, include_cuda: bool = True) -> dict`, `validate_environment(lock: dict, observed: dict) -> list[str]`, and `environment_fingerprint(observed: dict) -> str`.

- [ ] **Step 1: Write failing lock, validation, and deterministic fingerprint tests**

Test an exact lock containing Python 3.12, torch 2.11.0, SGLang source commit, sglang-kernel 0.4.4, sgl-deep-gemm 0.1.4, FlashInfer 0.6.12, and required symbols. Assert sorted JSON produces a stable 12-character SHA-256 fingerprint and that version/symbol mismatches return explicit errors.

- [ ] **Step 2: Run the focused test and verify it fails**

Run: `.venv-sglang0515/bin/python -m unittest tests.test_benchmark_environment -v`

Expected: FAIL because `benchmark_environment` does not exist.

- [ ] **Step 3: Implement the lock loader, collector, validator, and fingerprint**

Use `importlib.metadata.version`, `importlib.import_module`, attribute traversal for symbols, `platform.python_version`, and canonical `json.dumps(..., sort_keys=True, separators=(",", ":"))`. The collector must record package versions, import paths, required-symbol availability, CUDA runtime version, device name, and compute capability without importing the top-level SGLang server CLI.

- [ ] **Step 4: Run the focused tests**

Run: `.venv-sglang0515/bin/python -m unittest tests.test_benchmark_environment -v`

Expected: all environment tests PASS.

- [ ] **Step 5: Commit the runtime contract**

```bash
git add requirements/benchmark-lock.json benchmark_environment.py tests/test_benchmark_environment.py
git commit -m "feat: define benchmark runtime contract"
```

### Task 2: Reproducible Bootstrap

**Files:**
- Create: `bootstrap.sh`
- Create: `.gitignore`
- Create: `tests/test_bootstrap_contract.py`

**Interfaces:**
- Consumes: `requirements/benchmark-lock.json`, `/usr/bin/python3.12`, network access, and either `$UV` or `uv` on `PATH`.
- Produces: `.runtime/venv/bin/python`, `.runtime/logs/bootstrap.log`, and an environment that passes `python -m benchmark_environment --check`.

- [ ] **Step 1: Write failing static bootstrap contract tests**

Assert the script uses `set -euo pipefail`, resolves the repository directory from `${BASH_SOURCE[0]}`, creates `.runtime/venv`, installs exact locked packages, installs SGLang from the immutable Git commit, clears `PYTHONPATH`, records logs, and never references `.venv-sglang0515` or an external local source path.

- [ ] **Step 2: Run the bootstrap tests and verify failure**

Run: `.venv-sglang0515/bin/python -m unittest tests.test_bootstrap_contract -v`

Expected: FAIL because `bootstrap.sh` does not exist.

- [ ] **Step 3: Implement idempotent bootstrap**

Create the virtual environment with `uv venv --python /usr/bin/python3.12`. Install the exact backend packages from the lock and install SGLang from the immutable public Git commit with normal dependency resolution. Use `.runtime/cache` for `UV_CACHE_DIR`, keep a complete log, write the lock fingerprint into `.runtime/installed.lock`, and skip installation only when that fingerprint matches and environment validation succeeds.

- [ ] **Step 4: Run static bootstrap tests**

Run: `.venv-sglang0515/bin/python -m unittest tests.test_bootstrap_contract -v`

Expected: all bootstrap contract tests PASS.

- [ ] **Step 5: Commit bootstrap support**

```bash
git add bootstrap.sh .gitignore tests/test_bootstrap_contract.py
git commit -m "feat: bootstrap benchmark runtime"
```

### Task 3: Unified Single-Operator And Profiling Launcher

**Files:**
- Create: `benchmark_cli.py`
- Create: `run.sh`
- Create: `tests/test_benchmark_cli.py`

**Interfaces:**
- Consumes: validated `.runtime/venv`, command names `op`, `prefill`, `decode`, `compare`, `check`, and `smoke`.
- Produces: `resolve_command(argv: list[str], root: Path) -> list[str]` and subprocess exit status from the selected benchmark.

- [ ] **Step 1: Write failing command resolution tests**

Assert all five operator aliases map to their existing scripts, profiling arguments are forwarded byte-for-byte, unknown operators fail with the valid-name list, and `prefill`, `decode`, and `compare` map to their current scripts.

- [ ] **Step 2: Run launcher tests and verify failure**

Run: `.venv-sglang0515/bin/python -m unittest tests.test_benchmark_cli -v`

Expected: FAIL because `benchmark_cli` does not exist.

- [ ] **Step 3: Implement Python dispatch and thin shell launcher**

`run.sh` must select `.runtime/venv/bin/python`, unset `PYTHONPATH`, set repository-local `TORCH_EXTENSIONS_DIR`, `FLASHINFER_WORKSPACE_BASE`, and cache variables, then execute `benchmark_cli.py`. `benchmark_cli.py` validates the environment for every GPU command before replacing the process with the selected script.

- [ ] **Step 4: Run launcher tests and help commands**

Run:

```bash
.venv-sglang0515/bin/python -m unittest tests.test_benchmark_cli -v
bash -n run.sh bootstrap.sh
```

Expected: tests PASS and shell syntax checks return zero.

- [ ] **Step 5: Commit launcher support**

```bash
git add benchmark_cli.py run.sh tests/test_benchmark_cli.py
git commit -m "feat: add unified benchmark launcher"
```

### Task 4: Environment-Aware Formal Results And Smoke Tests

**Files:**
- Create: `tools/check_environment.py`
- Create: `tools/smoke_test.py`
- Modify: `deepseek_v4_benchmark.py`
- Modify: `tests/test_deepseek_v4_benchmark.py`
- Create: `tests/test_smoke_test.py`

**Interfaces:**
- Consumes: Task 1 environment functions and existing operator benchmark functions.
- Produces: a CLI environment report, minimal per-family GPU smoke execution, `environment_fingerprint` CSV column, and nonzero formal-run status if any row is unavailable.

- [ ] **Step 1: Write failing smoke-plan and CSV fingerprint tests**

Assert the smoke plan covers DSA indexer, sparse FlashMLA, projection GEMM, dense MLA, and grouped MoE; assert CSV headers and rows include the same fingerprint; assert an unavailable formal row yields failure after the CSV is written.

- [ ] **Step 2: Run focused tests and verify failure**

Run:

```bash
.venv-sglang0515/bin/python -m unittest tests.test_smoke_test tests.test_deepseek_v4_benchmark -v
```

Expected: new assertions FAIL.

- [ ] **Step 3: Implement validation report, smoke runner, and result fingerprinting**

The smoke runner uses one warmup and one measured replay, invokes the existing public helper for each operator family with the smallest valid SM100 shape, and writes no performance CSV. The formal benchmark prints the fingerprint in its case header and adds it to every CSV row. After writing results, `main` returns 1 when any row has status other than `executed`.

- [ ] **Step 4: Run focused and complete unit tests**

Run: `.venv-sglang0515/bin/python -m unittest discover -s tests -v`

Expected: all tests PASS.

- [ ] **Step 5: Commit validation and formal result metadata**

```bash
git add tools/check_environment.py tools/smoke_test.py deepseek_v4_benchmark.py \
  tests/test_deepseek_v4_benchmark.py tests/test_smoke_test.py
git commit -m "feat: validate formal benchmark runs"
```

### Task 5: Documentation And Clean-Runtime Verification

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: `bootstrap.sh`, `run.sh`, formal benchmark commands, and all validation tools.
- Produces: installation, single-operator, full profiling, result comparison, troubleshooting, and packaging instructions.

- [ ] **Step 1: Document the supported workflow**

Document `./bootstrap.sh`, all five `./run.sh op ...` commands, DeepSeek prefill/decode formal cases, `./run.sh compare`, `./run.sh check`, `./run.sh smoke`, `.runtime` exclusion, and a source-only archive command using `git archive` plus an optional patch file for intentionally uncommitted work.

- [ ] **Step 2: Build the repository-local runtime**

Run: `./bootstrap.sh`

Expected: exact dependency installation completes and environment validation prints a fingerprint with no mismatches.

- [ ] **Step 3: Run environment and GPU smoke validation**

Run:

```bash
./run.sh check
CUDA_VISIBLE_DEVICES=0 ./run.sh smoke
```

Expected: B200/SM100 is reported and every operator family executes.

- [ ] **Step 4: Run reduced end-to-end cases without overwriting formal results**

Run:

```bash
CUDA_VISIBLE_DEVICES=0 ./run.sh prefill --quant-profile fp8_mxfp8 \
  --m 16 --context 512 --warmup 1 --runs 1 \
  --csv /tmp/deepseek_v4_package_prefill.csv
CUDA_VISIBLE_DEVICES=0 ./run.sh decode --quant-profile fp8_mxfp8 \
  --m 1 --context 512 --warmup 1 --runs 1 \
  --csv /tmp/deepseek_v4_package_decode.csv
```

Expected: both commands return zero, all CSV rows are `executed`, and every row has the current environment fingerprint.

- [ ] **Step 5: Run final verification and inspect repository scope**

Run:

```bash
.runtime/venv/bin/python -m unittest discover -s tests -v
bash -n bootstrap.sh run.sh
git diff --check
git status --short
```

Expected: all tests and syntax checks pass; `.runtime` is absent from status; only intended project files and pre-existing unrelated changes remain.

- [ ] **Step 6: Commit documentation**

```bash
git add README.md
git commit -m "docs: document reproducible benchmark workflow"
```

