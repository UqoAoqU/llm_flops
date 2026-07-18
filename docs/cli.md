# CLI reference

Use `./bench.sh`; it selects the repository-local Python environment and cache
paths. All selectors are case-sensitive shell globs.

## Discovery and validation

```bash
./bench.sh list
./bench.sh list --operator 'deepseek_v4_*' --candidate 'reference_control__*'
./bench.sh validate
./bench.sh validate --operator deepseek_v4_fp8_gemm_nt
./bench.sh env --json
```

`list` and `validate` are static: they do not import implementations or
initialize CUDA. Registry/no-match errors exit 2.

## Run

```text
bench run [--suite ID] [--mode all|correctness|performance]
          [--operator GLOB ...] [--candidate GLOB ...]
          [--case GLOB ...] [--tag TAG ...] [--seed N ...]
```

Examples:

```bash
./bench.sh run --suite smoke --dry-run
./bench.sh run --mode correctness --operator OP --candidate CANDIDATE --case CASE
./bench.sh run --mode performance --operator OP --candidate CANDIDATE \
  --timer cuda_graph --warmup 5 --samples 20 --inner-iterations 1
./bench.sh run --mode all --operator OP --candidate CANDIDATE
```

Repeated values are ORed within a selector category and ANDed across
categories; exclusions apply last. CLI values override suite values, which
override operator defaults.

Important controls:

- `--dry-run`: emit a deterministic plan without formal environment/CUDA/result side effects;
- `--fail-fast`: stop scheduling after a failed result while retaining completed artifacts;
- `--timeout-s`: override worker stage timeout;
- `--output-root PATH`: replace the default `results/` root;
- `--evaluation-id ID`: explicit identity for one precisely selected candidate;
- `--resume RUN_ID`: verify manifest compatibility and execute missing result IDs only;
- `--timer auto|cuda_event|cuda_graph|wall_clock`;
- `--warmup`, `--samples`, `--inner-iterations`;
- performance gates: `--max-slowdown-pct`, `--min-speedup`,
  `--max-candidate-median-ms`, `--max-cv`, `--max-memory-bytes`;
- `--unsupported-policy fail|allow`, `--gpu-lock-timeout-s`;
- `--perf-on-correctness-fail`: retain explicitly non-formal diagnostic samples only.

`deepseek_v4_prefill` covers M=1024/2048/4096; `deepseek_v4_decode`
covers batch 16/32. Both use raw context 65536.

```bash
CUDA_VISIBLE_DEVICES=0 ./bench.sh run --suite deepseek_v4_prefill
CUDA_VISIBLE_DEVICES=0 ./bench.sh run --suite deepseek_v4_decode
```

## Summarize and compare

```bash
./bench.sh summarize results/OP/CANDIDATE/EVALUATION
./bench.sh summarize --operator OP --candidate CANDIDATE --evaluation EVALUATION
./bench.sh summarize --run RUN_ID
./bench.sh compare --result CURRENT --baseline-result BASELINE
./bench.sh compare --run CURRENT_RUN --baseline-run BASELINE_RUN
```

Summarize is read-only. Run-level forms resolve only `run_index.csv` entries.
Compare requires compatible contract/case/seed/source/environment/effective
timer and rankable inputs. A direct legacy-import summary is explicitly marked
non-formal; direct compare rejects it.

## Legacy tools

Legacy import and regression are explicit tools, not normal engine runs:

```bash
.runtime/venv/bin/python tools/convert_legacy_results.py INPUT.csv \
  --candidate-id legacy_control --dry-run
.runtime/venv/bin/python tools/regress_deepseek_v4.py \
  --phase prefill --work-dir .runtime/regression/prefill --dry-run
```

See [migration](migration-llm-flops.md) before removing `--dry-run`.

## Exit codes

- 0: requested gates passed;
- 1: completed correctness/performance gate failure;
- 2: usage/configuration/registry/compatibility error;
- 3: worker protocol or artifact infrastructure failure;
- 130: Ctrl-C after cleanup.
