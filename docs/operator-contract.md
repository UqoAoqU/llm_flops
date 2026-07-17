# Reference 与 Spec 接口

每个 reference 位于：

```text
operators/references/<operator_id>/
├── operator.yaml
├── spec.py
├── implementation.py
└── README.md                 # 推荐：记录算子特有语义
```

`operator_id` 必须与目录名相同，并匹配 `^[a-z][a-z0-9_]{2,79}$`。

## `operator.yaml`

Schema v1 示例：

```yaml
schema_version: 1
operator_id: minimal_cpu_add
contract_version: 1
description: Add two Python numbers on the CPU.
reference_entrypoint: implementation:operator
spec_entrypoint: spec:SPEC
device_types: [cpu]
tags: [example, cpu]
correctness:
  default_comparator: exact
  rtol: 0.0
  atol: 0.0
  equal_nan: false
  determinism_repeats: 1
performance:
  timer: wall_clock
  graph_mode: disabled
  warmup: 1
  samples: 3
  inner_iterations: 1
  timeout_s: 30
  regression_threshold_pct: 5.0
  min_speedup: 1.0
  max_candidate_median_ms: 10.0
  max_cv: 0.1
  max_memory_bytes: 1073741824
  unsupported_policy: fail
  gpu_lock_timeout_s: 600
cost_model: spec:cost_model
```

未知字段和未知 schema version 都是错误。Entrypoint 使用
`<module>:<attribute>`，文件必须在 reference root 内。Registry 仅检查文件位置；
Controller 只会导入可信 `spec.py` 的 case metadata，不导入 implementation。

## `spec.py` 的职责

`spec.py` 不是简单的数据格式转换器，而是 reference 拥有的完整评测语义：定义
case、构造和 clone 输入、声明可观察状态、规范化输出、选择比较器，并提供理论成本。

正式 `OperatorSpec` 方法如下：

| 成员 | 作用 | 执行位置 |
|---|---|---|
| `operator_id` | 与 manifest/request 绑定身份 | Controller + Worker |
| `cases()` | 返回非空、`case_id` 唯一的 `CaseSpec` | Controller 规划 + Worker 校验 |
| `make_inputs(case, context)` | 使用显式 seeded generator 构造 `InputBundle` | Worker |
| `clone_inputs(inputs)` | 为 reference/candidate 生成物理隔离输入 | Worker |
| `normalize_output(output)` | 把实现返回值映射为公开语义 | Worker |
| `comparator(case)` | 返回 reference 决定的 Comparator | Worker |
| `cost_model(case)` | 返回 FLOPs/bytes/throughput units 或 `None` | Worker |

一旦声明其中任意正式成员，就必须实现完整接口，避免“部分 spec”被静默接受。
Controller 导入 `cases()` 时会静态拒绝 `torch` import，因此 case metadata 必须轻量、
确定且不初始化 GPU；tensor 和实现特有对象只在 Worker 的运行时方法里出现。

`CaseSpec.symbols` 保存 shape/layout 等 JSON-safe 参数；`tags` 用于 smoke、boundary、
representative 等选择；`seed` 可被 CLI seed 覆盖。

## 输入与原地状态

`make_inputs()` 返回 `benchmark_engine.correctness.InputBundle(args, kwargs,
observed_state)`。所有会影响语义的原地修改都必须放入 `observed_state`，例如 KV cache、
block table 或输出 buffer；未声明的副作用不属于 correctness contract。

`clone_inputs()` 必须满足：

- reference/candidate 之间不共享 tensor storage 或可变容器；
- 同一个 bundle 内原有 alias 关系保持不变；
- 会被修改的 cache 和 metadata 被深复制。

可以使用 `clone_input_bundle()`，复杂 paged/cache 对象则实现显式 clone。

## 输出与 Comparator

`normalize_output()` 用于隐藏 workspace、临时 buffer 或实现特有返回结构，只暴露可
比较语义。框架随后检查 output path、dtype、shape、stride 和 layout，再执行 exact、
floating、TopK 或 quantized 比较器。容差由 reference/case 决定，candidate 不能修改。

## Cost model 与 workspace

`cost_model(case)` 可返回：

```python
TheoreticalCost(flops=..., estimated_bytes=..., throughput_units=...)
```

也可返回同字段 mapping 或 `None`。框架只用 candidate median latency 派生
`tflops`、`effective_bandwidth_gbps`、`arithmetic_intensity` 和 `throughput`；它不会
改变 `per_call_ms`。可选 `workspace_bytes(case) -> int | None` 声明 workspace 需求，
未知值写空字段而不是 0。

可复制示例位于 [docs/examples/minimal-operator](examples/minimal-operator/)。正确性细节见
[Correctness contract](correctness.md)。

## Routed MoE contract example

A routed-MoE spec must state both logical tensor geometry and backend layout:
global/local expert counts and offset, top-k, route-ID range, normalized route
weights, logical `w13[E,2I,H]`/`w2[E,H,I]`, scale block/dtype, activation
quantization and output dtype. Duplicate expert IDs and rows with no local
expert are valid semantic boundaries unless the operator explicitly says
otherwise. Routing tensors and representative weight/scale slices belong in
`observed_state` so an in-place candidate mutation is a correctness failure.

Large representative cases may persist a deterministic bounded sample rather
than a full golden tensor, but small CPU tests must retain an independent
routing/MLP oracle. Profile names are contracts: MXFP4 and FP8-weight/MXFP8-
activation kernels must not share one ambiguous runtime operator name.
