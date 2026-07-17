# 代码实现导读

生产代码位于 `src/benchmark_engine/`。本文按一次 `bench run` 的调用路径说明各模块，
方便新增 operator、定位失败或扩展通用能力。

## 入口与计划

| 模块 | 职责 |
|---|---|
| `cli.py` | argparse 命令、退出码和顶层错误映射 |
| `registry/` | 扫描 reference/candidate、严格解析 manifest、entrypoint 路径验证和 source hash |
| `suite.py` | 解析 `suites/*.yaml` 并拒绝未知字段 |
| `selectors.py` | operator/candidate/case/tag 的 glob 与排除语义 |
| `operator_spec.py` | Controller 只导入可信 `spec.py` 的纯 `cases()` 元数据；拒绝 torch import |
| `planning.py` | 构造排序稳定的 `EvaluationPlan`、job identity 和 resolved config |
| `environment/` | 收集运行环境并生成 fingerprint |

`engine.py` 是编排入口：构建 dry-run/execution/resume 计划，创建 artifact，逐 job
调用 Controller，并把 Worker 结果投影成 CSV 行。它不执行 operator 本体。

## Controller 与 Worker

`execution/protocol.py` 定义 schema-versioned request、event 和 response。请求包含已验证
的绝对 root、entrypoint、case 元数据、resolved config 和独立 timeout，不包含 tensor、
callable 或 pickle。

`execution/controller.py` 调用 `isolation.py` 创建新 POSIX session，持续读取两个输出
管道，验证 JSONL event 顺序，并在阶段超时或 Ctrl-C 时 TERM/KILL 整个进程组。
`gpu_lock.py` 用 GPU UUID 做跨进程互斥，并保守处理 stale/malformed lock。

`execution/worker.py` 是运行时代码唯一入口：

1. 重新验证 root 和 entrypoint；
2. 依次加载 spec、reference、candidate；
3. 运行可选 build；
4. 调用 correctness evaluator；
5. 仅在策略允许时调用 performance evaluator；
6. 写专属 response/diagnostic/event 文件后退出。

正式 `results.csv` 等文件仍由 Controller 写，Worker 不能直接修改它们。

## 正确性实现

`correctness/inputs.py` 创建显式 seeded CPU/CUDA generator，并深 clone tensor、容器和
observed state，同时保留单个 bundle 内的 alias。`normalization.py` 把返回值和状态
展平成带 `output_path`、dtype、shape、stride、layout 和 device 的叶子。

`comparators.py` 提供 exact、floating、TopK 和 quantized 比较器；`metrics.py` 计算误差
统计；`diagnostics.py` 生成有界的失败样本和复现命令；`evaluator.py` 负责执行顺序、
CUDA 同步、确定性重复和失败分类。

扩展新的比较语义时，应实现 Comparator 协议并由 reference `spec.comparator(case)`
显式选择，而不是在 candidate 中决定容差。

## 性能实现

`performance/timers.py` 实现 wall clock、CUDA Event 和 CUDA Graph。`evaluator.py`
准备两侧输入、执行 first-call/warmup/capture，并输出固定 `R-C-C-R` raw samples。
`statistics.py` 计算 Type-7 分位数、population standard deviation 和 CV；`gate.py`
检查 correctness、样本数、稳定性、timer 一致性、环境竞争、内存和 slowdown/speedup
阈值。

`cost_model.py` 把 `spec.cost_model(case)` 返回的 theoretical FLOPs/bytes/
throughput units 与 candidate median latency 组合，生成 `tflops`、
`effective_bandwidth_gbps`、`arithmetic_intensity` 和 `throughput`。它不改变原始计时，
也不是 profiler 测量。

新增 Timer 应保持 prepare/sample/cleanup 边界并记录 requested/effective/fallback；新增
门禁必须同时由 Worker 计算、Controller 可信重算，并更新 artifact schema 与测试。

## Reporting

`reporting/artifact_writer.py` 管理 evaluation manifest、run index、history/latest、
日志和原子替换。`csv_writer.py` 是 CSV 列顺序、schema 升级、枚举和主键冲突规则的
代码权威。`summary.py` 只读生成汇总，`compare.py` 先验证兼容性再计算性能差异。

`projection/base.py` 定义通用只读投影协议，`projection/deepseek_v4.py` 保存
legacy adapter 到稳定 operator/case 的映射。投影仅使用已持久化 per-call median
乘以 instances；跨 operator 的模型 partial total 由 `run_index.csv` 在 run 级聚合。

所有 artifact 写入遵守三个不变量：

- 同一主键和完全相同行可幂等重放；同键不同内容必须报错；
- 缺失值用空字段表示，不能伪造为 0/false；
- breaking change 提升 schema version，旧 header 只通过显式迁移读取。

## 新增 operator 的最小改动面

通常只需新增：

```text
operators/references/<operator_id>/operator.yaml
operators/references/<operator_id>/spec.py
operators/references/<operator_id>/implementation.py
operators/candidates/<operator_id>/<candidate_id>/implementation.py
tests/test_<operator_id>.py
```

只有当现有 `CaseSpec`、Comparator、Timer、CostModel 或 artifact 无法表达真实契约时，
才修改 `src/benchmark_engine/`。不要为了适配单个 kernel 在通用引擎中加入
operator-specific 分支。

完整接口见 [Operator contract](operator-contract.md)，验证命令见
[开发与测试](development.md)。
