# 结果目录与恢复

Candidate source 和 result 共享 `<operator_id>/<candidate_id>`：

```text
operators/candidates/<operator_id>/<candidate_id>/
results/<operator_id>/<candidate_id>/<evaluation_id>/
```

完整布局：

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
        ├── summary.md
        ├── diagnostics/
        └── logs/
            ├── controller.jsonl
            ├── worker.jsonl
            ├── stdout.log
            └── stderr.log
```

`evaluation_id` 格式是
`<YYYYMMDDTHHMMSSZ>__<environment-short-hash>__<run-short-id>`。`run_id` 用于关联
一次命令产生的多个 evaluation，存放在 `results/run_index.csv`，从不作为结果根目录。
所有身份组件都经过路径验证；绝对路径、traversal 和分隔符会被拒绝。

## Evaluation manifest

`evaluation_manifest.json` 保存 schema version、完整 identity、生命周期状态、原始命令、
resolved configuration、双方 source hash、suite/mode 和 environment snapshot/fingerprint。

状态转换严格为：

```text
planned -> running -> complete | failed | interrupted
interrupted -> running          # 仅兼容 resume
```

相同状态更新可幂等重放；其他转换拒绝。Resume 会清除旧 interrupted reason；`failed`
和 `complete` 是终态。

## CSV 与发布索引

- `results.csv`：每个 `operator × candidate × case × seed` 一行汇总；
- `correctness_outputs.csv`：每个 normalized output path 一行；
- `performance_samples.csv`：reference/candidate 每个原始计时样本一行；
- `history.csv`：该 candidate 已完成 evaluation 的追加历史；
- `latest.json`：按 completion time 和 evaluation ID 指向最新 complete evaluation；
- `run_index.csv`：把 run ID 映射到一个或多个镜像 evaluation 路径。

只有 evaluation 的所有预期结果都终止且没有基础设施失败时，才发布 `history.csv` 和
`latest.json`。Failed/interrupted 目录保留诊断，但不会成为 latest。

## Worker 记录

`logs/worker.jsonl` 按 attempt 记录 `DISCOVERED`、`WORKER_STARTED`、各阶段 start/end、
`HEARTBEAT`、`REPORT_WRITTEN` 和 `WORKER_EXITED`。`controller.jsonl` 记录监督结果和
截断标记。Stdout/stderr 被持续排空并设总字节上限，达到上限后写一次显式 truncation
marker，仍继续排空以避免死锁。

同一 `result_id` 的重试使用递增 attempt 文件名；旧 response 不会被新 attempt 复用。
Full traceback 和协议诊断位于 `diagnostics/`，CSV 只保存短错误摘要和相对路径。

## 原子性和幂等

每个完成 case 都立即更新 artifact。正式 text/JSON/CSV 使用同目录临时文件，flush、
fsync 后 `os.replace()`。同一主键和完全相同行可幂等重放；同键不同内容是冲突，不能
覆盖已有事实。Controller 是唯一正式 writer，Worker 只写自己的 response、diagnostic
和 append-only event channel。

Resume 通过 `run_index.csv` 定位 evaluation，再验证路径、manifest schema、identity、
source、resolved config、suite/mode 和 environment。`results.csv` 中已有 `result_id`
被视为完成工作，只执行缺失部分。

字段的当前版本与精确列顺序见 [CSV Schema](csv-schema.md)。
