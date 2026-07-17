# 快速上手

## 1. 准备环境

基础要求是 Linux、Python 3.12 和 `uv`。GPU operator 需要匹配的 NVIDIA
driver/CUDA toolkit；源码/JIT 构建还需要 `rustc`/`cargo`、`ninja` 和 CUDA compiler
位于 `PATH`。

```bash
./bootstrap.sh
./run.sh check
./bench.sh --version
```

运行环境位于 `.runtime/venv`，缓存位于 `.runtime/cache`。`bench.sh` 会清理外部
`PYTHONPATH` 并使用仓库环境，通常不需要手工激活虚拟环境。

## 2. 发现和验证

```bash
./bench.sh list
./bench.sh list --operator 'deepseek_v4_*'
./bench.sh validate --operator deepseek_v4_fp8_gemm_nt
./bench.sh env --json
```

`list` 和 `validate` 只做静态发现，不导入 candidate。Operator/candidate 选择器是
区分大小写的 shell glob。

## 3. 先看执行计划

```bash
./bench.sh run --suite smoke --dry-run
./bench.sh run --mode correctness --dry-run \
  --operator deepseek_v4_fp8_gemm_nt \
  --candidate 'test_impl*' --case 'smoke_*' --seed 0
```

Dry-run 输出确定性的 JSON 计划，不初始化 CUDA、不创建 result 目录。新增 case、
candidate 或 seed 会改变 job 数，这是计划扩展的正常结果；测试应验证选中的身份和
shape，而不是依赖已经过时的固定数量。

## 4. 运行正确性

```bash
./bench.sh run --mode correctness \
  --operator example_cpu_add \
  --candidate quickstart__20260716T120000Z__4279e756 \
  --case tiny --seed 0
```

GPU operator 显式选择设备：

```bash
CUDA_VISIBLE_DEVICES=0 ./bench.sh run --mode correctness \
  --operator deepseek_v4_fp8_gemm_nt \
  --candidate 'test_impl*' --case 'smoke_*' --seed 0
```

Reference 的 `spec.py` 生成一次 canonical input，再分别 clone 给 reference 与
candidate。比较覆盖返回值和声明在 `observed_state` 中的原地状态。失败会写入
`correctness_outputs.csv` 和 `diagnostics/`。

## 5. 运行性能

```bash
CUDA_VISIBLE_DEVICES=0 ./bench.sh run --mode performance \
  --operator deepseek_v4_fp8_gemm_nt \
  --candidate 'test_impl*' --case 'smoke_*' \
  --timer cuda_graph --warmup 5 --samples 20 --inner-iterations 100
```

`performance` 模式仍先执行正确性。只有正确性通过、timer 可比、样本稳定、GPU
没有检测到竞争进程且满足阈值的结果才是 formal/rankable。使用 `--mode all` 可表达
同一意图；两者都不会绕过正确性门禁。

首次运行 CUDA reference 可能进行 Ninja/PTXAS 编译。等待 import/first-call/
graph-capture 阶段完成后再判断是否阻塞；steady-state 样本不包含这些一次性成本。

## 6. 使用 suite 与选择器

```bash
./bench.sh run --suite smoke
./bench.sh run --suite regression --operator 'deepseek_v4_*' --tag representative
./bench.sh run --mode correctness \
  --operator OP --candidate CANDIDATE --case CASE --seed 0 --seed 1
```

同类选择器多次出现时取并集，不同类别之间取交集，exclude 最后应用。内置 suite
只包含仓库维护的 candidate 模式；接入任意名称的新 candidate 时，可直接使用
`--mode` 加精确 `--operator/--candidate` 选择。

配置优先级是：CLI 覆盖 > suite > operator manifest > engine 默认值。

## 7. 汇总、恢复和比较

```bash
./bench.sh summarize results/OPERATOR_ID/CANDIDATE_ID/EVALUATION_ID
./bench.sh summarize --operator OP --candidate CANDIDATE --evaluation EVALUATION
./bench.sh run --resume RUN_ID

./bench.sh compare --result EVALUATION_ID \
  --baseline-result BASELINE_EVALUATION_ID
./bench.sh compare --run RUN_ID --baseline-run BASELINE_RUN_ID
```

Resume 只运行缺少 `result_id` 的 job，不覆盖已经持久化的行。Compare 仅比较 contract、
case/seed、环境和 timer provenance 兼容的结果；不兼容时返回配置错误，不计算误导性的
speedup。

退出码：`0` 通过，`1` correctness/performance 门禁失败，`2` 用法、配置或 Registry
错误，`3` Worker/引擎基础设施错误，`130` 为 Ctrl-C。
