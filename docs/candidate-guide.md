# Candidate 接入指南

Candidate 目录与结果目录按同一身份镜像：

```text
operators/candidates/<operator_id>/<candidate_id>/
results/<operator_id>/<candidate_id>/<evaluation_id>/
```

最小 candidate 只有 `implementation.py`，默认入口是
`implementation:operator`。可直接复制 [最小 CPU candidate](examples/minimal-candidate/implementation.py)。

## 命名

`candidate_id` 只需满足：

- 在同一 `operator_id` 下唯一；
- 非空、不是绝对路径、不是 `.` 或 `..`；
- 不包含 `/`、`\` 或 NUL；
- 与同级其他名字不存在仅大小写不同的冲突。

推荐但不强制使用：

```text
<task_identifier>__<YYYYMMDDTHHMMSSZ>__<8-or-more-lowercase-hex>
```

目录名无需包含 source hash，也无需与 hash 前缀匹配。使用时间和 hash 仍有利于审计与
历史阅读，但 Registry 不以此拒绝名称。

## 实现角色

Reference 是已经验证过的优化 baseline；candidate 是被比较对象。框架验证阶段可以把
reference `implementation.py` 原样复制为 control candidate，用于证明 Registry、输入
隔离、正确性、JIT 分段、计时和 artifact 有效。这样的性能比预期接近 1，不代表存在
新的优化。

Candidate 必须遵守 reference `spec.py` 构造的调用签名和公开输出语义。它不能决定
case、容差、cost model 或正式性能门禁。

## 可选 `candidate.yaml`

复杂 candidate 可声明严格 schema v1：

```yaml
schema_version: 1
operator_id: minimal_cpu_add
candidate_id: my_candidate
entrypoint: implementation:operator
framework: python
build:
  command: [python, build.py]
  timeout_s: 600
metadata:
  task_id: task_0042
```

Build command 必须是 argv string array；shell command string 被拒绝。Metadata 必须
JSON-safe。不要在 import 或 timed operator 中偷偷执行构建；需要预构建时通过 manifest
声明，运行时 JIT 则由 first-call/graph-capture 阶段显式承担。

## Source hash

Hash 覆盖 source 文件内容、mode、相对 POSIX 路径和 `candidate.yaml` 的规范化内容；
缓存、bytecode、build output、egg-info 和常见编辑器临时文件被排除，symlink 被拒绝。
为了让重命名不改变源码身份，规范化 manifest 只移除 `candidate_id` 字段，其他修改都会
改变 hash。

Source hash 独立写入 evaluation manifest 和 `results.csv`。源码修改后换一个清晰的
candidate 名仍是推荐做法，但不是接受条件。

## 验证

```bash
./bench.sh validate --operator OPERATOR_ID
./bench.sh list --operator OPERATOR_ID --candidate CANDIDATE_ID
./bench.sh run --mode correctness --dry-run \
  --operator OPERATOR_ID --candidate CANDIDATE_ID
```

正式 GPU 运行前，先检查 dry-run 的 case、seed、shape 和 job identity。完整调用契约见
[Reference 与 Spec 接口](operator-contract.md)。
