# CLI 参考

使用 `./bench.sh` 或 `.runtime/venv/bin/bench`。前者会固定仓库运行环境，日常使用
优先选择它。

## `bench list`

静态列出 operator 和 candidate：

```bash
./bench.sh list
./bench.sh list --operator 'deepseek_v4_*' --candidate 'test_*'
```

`--operator`、`--candidate` 是区分大小写的 shell glob。无匹配或所选 Registry
无效时退出 2。该命令不导入 implementation。

## `bench validate`

```bash
./bench.sh validate
./bench.sh validate --operator deepseek_v4_fp8_gemm_nt
```

验证目录身份、manifest schema、entrypoint、source hash 输入和冲突，不运行算子。

## `bench env`

```bash
./bench.sh env
./bench.sh env --json
```

输出 Python、依赖、CUDA/GPU 和环境 fingerprint。JSON 形式适合归档或脚本读取。

## `bench run`

```text
bench run [--suite ID] [--mode all|correctness|performance]
          [--operator GLOB ...] [--candidate GLOB ...]
          [--case GLOB ...] [--tag TAG ...] [--seed N ...]
```

常用形式：

```bash
./bench.sh run --suite smoke
./bench.sh run --suite regression --dry-run
./bench.sh run --mode correctness \
  --operator OP --candidate CANDIDATE --case CASE --seed 0
./bench.sh run --mode performance \
  --operator OP --candidate CANDIDATE \
  --timer auto --warmup 5 --samples 30 --inner-iterations 20
./bench.sh run --resume RUN_ID
```

DeepSeek V4 model suites use explicit phase/profile/context case metadata:

```bash
./bench.sh run --suite deepseek_v4_prefill --dry-run
./bench.sh run --suite deepseek_v4_decode --dry-run
CUDA_VISIBLE_DEVICES=0 ./bench.sh run --suite deepseek_v4_prefill
CUDA_VISIBLE_DEVICES=0 ./bench.sh run --suite deepseek_v4_decode
```

选择器 `--operator`、`--candidate`、`--case`、`--tag`、`--exclude-operator` 和
`--seed` 可重复。同类值取 OR，不同类取 AND，exclude 最后执行。`--mode` 覆盖 suite
mode；CLI 性能参数覆盖 suite 和 operator manifest。

执行控制：

- `--dry-run`：只输出确定性计划；不收集正式环境、不初始化 CUDA、不写结果；
- `--fail-fast`：首个失败后不启动剩余 job，但保留已完成行；
- `--timeout-s`：覆盖 Worker 阶段超时；
- `--output-root PATH`：替换默认 `results/`；
- `--evaluation-id ID`：仅在精确选择一个 candidate 时使用；
- `--resume RUN_ID`：验证 manifest 兼容性后跳过已有 `result_id`。

性能参数：

- `--timer auto|cuda_event|cuda_graph|wall_clock`；
- `--warmup`、`--samples`、`--inner-iterations`；
- `--max-slowdown-pct`、`--min-speedup`、`--max-candidate-median-ms`；
- `--max-cv`、`--max-memory-bytes`；
- `--unsupported-policy fail|allow`；
- `--gpu-lock-timeout-s`；
- `--perf-on-correctness-fail`：仅保留非正式诊断样本，不能进入排名。

默认继续执行后续 job。Correctness 或 performance gate 失败退出 1；usage、配置、
Registry 或 compare incompatibility 退出 2；Worker/协议/artifact 基础设施失败退出 3；
Ctrl-C 退出 130。

## `bench summarize`

只读已有 artifact，不执行 operator：

```bash
./bench.sh summarize results/OP/CANDIDATE/EVALUATION
./bench.sh summarize --operator OP --candidate CANDIDATE --evaluation EVALUATION
./bench.sh summarize --run RUN_ID
```

Evaluation/path forms summarize one mirrored evaluation. `--run` resolves all
evaluations through `results/run_index.csv`; use it for a cross-operator model
projection and its partial total, missing, unavailable, and unsupported lists.

位置参数与三元身份参数二选一。输出 correctness、performance、gate、诊断路径和
复现信息。

## `bench compare`

```bash
./bench.sh compare --result EVALUATION --baseline-result BASELINE_EVALUATION
./bench.sh compare --run RUN_ID --baseline-run BASELINE_RUN_ID
```

`--result` 与 `--run` 互斥。比较前严格检查 contract version、case/seed、source、
environment 和 effective timer；不兼容的 artifact 不计算 speedup。

Artifact 和恢复规则见 [结果目录](result-layout.md)。
