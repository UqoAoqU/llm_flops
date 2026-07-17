# 开发与测试

## 环境

```bash
./bootstrap.sh
./bench.sh --help
```

`bootstrap.sh` 创建 `.runtime/venv`、安装锁定依赖、以 editable 方式安装当前仓库并验证
legacy GPU 环境。`bench.sh` 清理外部 `PYTHONPATH`，把 uv、extension、FlashInfer 和
XDG cache 固定到 `.runtime/cache`。

## 提交前门禁

每次变更至少运行：

```bash
.runtime/venv/bin/python -m unittest discover -s tests -v
./run.sh check
git diff --check
```

修改 CUDA operator 时，还要在 B200 上运行该 operator 的 correctness 和 performance
smoke。一次只使用一个正式 GPU benchmark 进程，并显式设置
`CUDA_VISIBLE_DEVICES=0`。

不得提交 `.runtime/`、`results/`、日志、缓存、bytecode 或 JIT/build 产物。通用引擎
变更不得改变旧 `run.sh` 语义；算子迁移也不得通过放宽 reference、case、容差、门禁或
测试来通过。

## 测试分层

- Registry/manifest/selector/planning：CPU 单元和 contract tests；
- Worker/Controller：真实子进程，覆盖协议、超时、signal、process group 和日志上限；
- Correctness：输入隔离、output normalization、各 Comparator、确定性与失败诊断；
- Performance：timer、采样次序、统计、cost model、GPU lock 和 gate；
- Reporting：schema、原子写入、主键冲突、resume、summary 和 compare；
- Operator：manifest/spec 契约、semantic oracle、control candidate 和 GPU smoke。

Linux-sensitive execution tests 必须使用短 timeout，并允许 Controller 完成
TERM/KILL/reap。Crash/hang fixtures 放在 `tests/fixtures/workers/`，不得放入生产
operator registry。

CPU correctness 专项可用：

```bash
.runtime/venv/bin/python -m unittest -v \
  tests/test_input_bundle.py \
  tests/test_output_normalization.py \
  tests/test_exact_comparator.py \
  tests/test_floating_comparator.py \
  tests/test_topk_comparator.py \
  tests/test_quantized_comparator.py \
  tests/test_correctness_evaluator.py
```

## 修改稳定协议

Controller/Worker wire、manifest 和 CSV 都是版本化协议。新增字段时同步修改模型、严格
解析、writer/reader、文档和 round-trip/legacy tests。未知字段不得被静默忽略；breaking
change 必须升级 schema version。禁止使用 pickle 传输 candidate、tensor 或 callable。

新增 operator 的推荐改动面见 [代码实现导读](implementation.md)。
