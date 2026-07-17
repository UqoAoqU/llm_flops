# LLM Operator Benchmark Engine

本仓库提供面向 LLM CUDA 算子的可复现评测框架，同时保留原有 DeepSeek V4
和 GLM-5 benchmark 入口。框架把已经验证过的优化实现定义为 `reference`，把待评测
实现定义为 `candidate`，在同一组输入上先做正确性检查，再做隔离、可恢复的性能测量。

核心能力包括：

- 静态发现 reference、candidate 和 suite，不在 Controller 中导入 candidate；
- reference 与 candidate 输入存储隔离，支持输出与原地状态比较；
- wall clock、CUDA Event、CUDA Graph 计时，以及 import/build/first-call/warmup/
  graph-capture/steady-state 分阶段记录；
- 单 GPU 互斥锁、`R-C-C-R` 交错采样、稳定性和性能门禁；
- `results/<operator_id>/<candidate_id>/<evaluation_id>/` 镜像结果目录；
- 原子 CSV/JSON 写入、进程超时清理、中断恢复、结果汇总和兼容性比较。

## 安装

在仓库根目录执行：

```bash
./bootstrap.sh
./run.sh check
```

要求 Linux、`/usr/bin/python3.12` 和 `uv`；CUDA 算子还要求匹配的 NVIDIA
driver/toolkit。若依赖需要源码/JIT 编译，`rustc`/`cargo`、`ninja` 和 CUDA compiler
也必须位于 `PATH`。`bootstrap.sh` 创建 `.runtime/venv` 并安装锁定的 Python 依赖。
`bench.sh` 会自动使用该虚拟环境，无需手工 `activate`：

```bash
./bench.sh --help
./bench.sh env
```

## 五分钟 CPU 验证

内置 `example_cpu_add` 不依赖 CUDA，可用于确认 Registry、Worker、正确性、性能和
artifact 链路：

```bash
./bench.sh validate --operator example_cpu_add
./bench.sh list --operator example_cpu_add
./bench.sh run --suite smoke --operator example_cpu_add

./bench.sh run --mode performance \
  --operator example_cpu_add \
  --candidate quickstart__20260716T120000Z__4279e756 \
  --case tiny --timer wall_clock \
  --warmup 5 --samples 30 --inner-iterations 20
```

命令会打印 `run_id` 和 evaluation 路径。读取已有结果不会重新执行代码：

```bash
./bench.sh summarize \
  --operator example_cpu_add \
  --candidate quickstart__20260716T120000Z__4279e756 \
  --evaluation EVALUATION_ID

./bench.sh run --resume RUN_ID
./bench.sh compare --run RUN_ID --baseline-run BASELINE_RUN_ID
```

## DeepSeek V4 model suites

Model suites run migrated FP8 operators at legacy shapes while keeping raw
kernel timing separate from model projection:

```bash
CUDA_VISIBLE_DEVICES=0 ./bench.sh run --suite deepseek_v4_prefill
CUDA_VISIBLE_DEVICES=0 ./bench.sh run --suite deepseek_v4_decode
./bench.sh summarize --run RUN_ID
```

Prefill covers M=1024/2048/4096 and decode covers batch 16/32, both at raw
context 65536. Phase, quant profile, model input, context, and adapter identity
are explicit case metadata. Missing or unavailable legacy adapters are listed
and never counted as zero in the measured partial total.

## 评测一个 candidate

Candidate 放在：

```text
operators/candidates/<operator_id>/<candidate_id>/implementation.py
```

`candidate_id` 只需是同一 `operator_id` 下唯一、非空且路径安全的目录名。推荐使用
`<task_identifier>__<YYYYMMDDTHHMMSSZ>__<source-hash-prefix>`，但该格式不是强制
约束。为了只验证框架，允许直接复制 reference 的 `implementation.py` 作为 control
candidate；此时正确性应通过，性能比应接近 1。

以 DeepSeek V4 FP8 GEMM control candidate 为例：

```bash
CUDA_VISIBLE_DEVICES=0 ./bench.sh validate \
  --operator deepseek_v4_fp8_gemm_nt

CUDA_VISIBLE_DEVICES=0 ./bench.sh run --mode correctness \
  --operator deepseek_v4_fp8_gemm_nt \
  --candidate test_impl__20260717T061620Z__cfb18306 \
  --case 'smoke_*' --seed 0

CUDA_VISIBLE_DEVICES=0 ./bench.sh run --mode performance \
  --operator deepseek_v4_fp8_gemm_nt \
  --candidate test_impl__20260717T061620Z__cfb18306 \
  --case 'smoke_*' --timer cuda_graph \
  --warmup 5 --samples 20 --inner-iterations 100
```

性能模式仍会先执行正确性门禁。正确性失败默认不会产生正式性能样本，也不会进入
排名。CUDA 首次执行可能触发 Ninja/PTXAS JIT；编译时间归入 import/first-call/
graph-capture 等准备阶段，不计入 steady-state。

## 结果目录

```text
results/
├── run_index.csv
└── <operator_id>/<candidate_id>/
    ├── history.csv
    ├── latest.json
    └── <evaluation_id>/
        ├── evaluation_manifest.json
        ├── results.csv
        ├── correctness_outputs.csv
        ├── performance_samples.csv
        ├── model_projection.csv
        ├── summary.md
        ├── diagnostics/
        └── logs/
```

`results.csv` 是每个 case 的汇总；`correctness_outputs.csv` 是逐输出叶子的正确性
指标；`performance_samples.csv` 保存 reference/candidate 原始样本、采样顺序以及
两侧输出 dtype/shape。`order_index` 是交错采样的全局执行序号，不是性能排名。

## 文档

- [快速上手](docs/getting-started.md)：从发现到 correctness/performance/compare；
- [CLI 参考](docs/cli.md)：命令、选择器、覆盖规则和退出码；
- [Reference/Spec 接口](docs/operator-contract.md)：`operator.yaml`、`spec.py`、
  `implementation.py`；
- [Candidate 接入](docs/candidate-guide.md)：目录、命名、manifest 和 source hash；
- [架构](docs/architecture.md) 与 [代码实现](docs/implementation.md)：数据流、模块边界
  和扩展点；
- [正确性](docs/correctness.md)、[性能](docs/performance.md)、
  [CSV schema](docs/csv-schema.md) 与 [结果/恢复](docs/result-layout.md)；
- [旧入口](docs/legacy-launchers.md)：`run.sh`、DeepSeek V4 和 GLM-5 脚本；
- [开发与测试](docs/development.md)、[故障排查](docs/troubleshooting.md)。

完整导航见 [docs/index.md](docs/index.md)。贡献前请阅读
[CONTRIBUTING.md](CONTRIBUTING.md)。当前不包含 Nsight/NVTX、profiler backend 和
多 GPU 调度；这些能力不会在结果中被静默伪装为可用。
