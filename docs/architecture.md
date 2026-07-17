# 架构

Benchmark Engine 采用 Controller/Worker 两层结构。Controller 负责静态发现、计划、
调度、超时清理和唯一的正式 artifact 写入；Worker 才会导入 reference spec、reference
实现和 candidate 实现，并在同一隔离进程内执行正确性与性能测量。

```text
CLI
 └─ Registry ──> Suite/Selectors ──> EvaluationPlan
                                      │
                                      v
                              Controller + GPU Lock
                                      │ JSON-safe request/events/response
                                      v
                         Worker: import -> build -> correctness
                                             -> prepare -> R-C-C-R samples
                                      │
                                      v
                         ArtifactWriter -> mirrored results
                                      │
                         summarize / compare / resume
```

Model projection is a derived layer after kernel evaluation. The DeepSeek V4
mapping converts explicit case metadata and legacy adapter semantics into
instances, display/backend metadata, and `per_call_ms * instances`. It never
changes shapes, correctness, timers, JIT/build stages, or raw samples. Each
evaluation owns a recoverable `model_projection.csv`; run-level totals join all
mirrored evaluations through `run_index.csv`.

The DeepSeek routed-MoE mapping is a normal operator mapping, not a projection
special case. `deepseek_v4_trtllm_fp8_mxfp8_moe` owns the global/local expert,
top-k, shuffled FP8 weight and MXFP8 activation contract. Its per-call median
is multiplied by 61 only after correctness and the ordinary performance gate
succeed. Missing symbols, unsupported capability, import/build errors,
timeouts and OOM remain worker outcomes and therefore cannot enter a model
total as zero.

## 核心边界

- **Registry** 只解析 manifest、检查 entrypoint 文件和计算 source hash，不导入
  candidate。
- **Planning** 把 suite、选择器、case 和配置解析为不可变、可排序的 job；dry-run
  使用 provisional environment fingerprint 且没有正式文件副作用。
- **Controller** 启动独立进程组，持续排空 stdout/stderr，执行分阶段硬超时，清理
  Ninja/NVCC/PTXAS 等子进程，并持有物理 GPU UUID 锁。
- **Worker** 加载运行时代码，生成输入，执行正确性门禁，完成计时和内存采样，仅通过
  JSON-safe 协议返回结构化结果。
- **Reporting** 是 formal CSV/JSON 的唯一写入方；每个 case 完成后原子落盘，允许
  中断后按 `result_id` 恢复。
- **Readers**（summarize/compare）只读取 artifact，不执行 operator 代码。

## 执行生命周期

一个 job 按以下顺序推进：

1. 导入可信 spec、reference 和 candidate；
2. 执行可选 candidate build；
3. 由 spec 创建 canonical input，并为两侧生成物理隔离 clone；
4. 执行 reference/candidate、规范化输出、比较返回值和 observed state；
5. 正确性通过后，分别完成 first call、warmup 和 graph capture；
6. 以固定 `R-C-C-R` 次序交错采样并计算统计量；
7. Controller 重新验证样本序号、timer provenance 和性能门禁；
8. 原子追加 case 行并更新 evaluation 状态。

Import、build、first call、warmup、graph capture 和 steady state 是不同阶段。JIT
发生在哪个阶段就计入该阶段，不能通过预热隐藏进性能样本，也不能混入 steady-state。

## 数据与身份

Controller-facing 模型位于 `benchmark_engine.models`，使用 frozen dataclass 和显式
`to_dict()/from_dict()`；wire form 只能包含 JSON-safe 元数据，禁止 tensor、callable、
pickle 和 shell command string。运行时 `InputBundle`、tensor 和 normalized output
只存在于 Worker。

源码和结果共享两个身份键：

```text
operators/candidates/<operator_id>/<candidate_id>/
results/<operator_id>/<candidate_id>/<evaluation_id>/
```

`run_id` 关联一次命令产生的多个 evaluation，仅出现在 `run_index.csv`，不是目录名。
`evaluation_id` 包含 UTC 时间、环境短 hash 和 run 短 ID。

## 状态与恢复

Manifest 状态机是：

```text
planned -> running -> complete | failed | interrupted
                        ^
interrupted ------------|
```

只有兼容的 `interrupted` evaluation 可以恢复到 `running`。`complete` 和 `failed` 是
终态。只有完整完成的 evaluation 才发布 `history.csv` 和 `latest.json`；中断或失败的
目录保留用于诊断。

## 信任和安全

Reference spec 是仓库内可信代码；candidate 不可信于正确性，但仍以 Worker 用户权限
运行。进程组、路径验证、超时和有界日志提供故障隔离，不是安全沙箱。评测真正不可信
代码时，还需在引擎外配置独立容器、用户、文件系统和网络策略。

源码模块对应关系见 [代码实现导读](implementation.md)，文件协议见
[结果目录](result-layout.md) 与 [CSV Schema](csv-schema.md)。
