# Benchmark Engine 文档

本文档集描述当前可使用的框架契约。开发阶段计划、阶段编号和临时迁移记录不属于
运行时规范；算子特有的 shape、layout 和容差说明放在对应
`operators/references/<operator_id>/README.md`。

## 开始使用

- [快速上手](getting-started.md)：安装、发现、dry-run、正确性、性能、恢复与比较；
- [CLI 参考](cli.md)：全部用户命令、选择器、配置优先级和退出码；
- [故障排查](troubleshooting.md)：JIT、超时、OOM、GPU 锁、非正式结果和日志。

## 接入算子

- [Reference 与 Spec 接口](operator-contract.md)：`operator.yaml`、`spec.py` 和
  reference `implementation.py`；
- [Candidate 指南](candidate-guide.md)：目录布局、可选 `candidate.yaml`、命名和
  source hash；
- [正确性契约](correctness.md)：输入隔离、observed state、输出规范化、Comparator
  与诊断；
- [性能测量](performance.md)：计时器、阶段、采样、GPU 锁、门禁和 cost model；
- [可复制的最小示例](examples/minimal-operator/operator.yaml)。

## 理解实现与产物

- [架构](architecture.md)：职责边界、执行生命周期和信任边界；
- [代码实现导读](implementation.md)：源码模块、关键数据模型和扩展路径；
- [结果目录与恢复](result-layout.md)：镜像目录、manifest、日志与原子写入；
- [CSV Schema](csv-schema.md)：`results.csv`、`correctness_outputs.csv`、
  `performance_samples.csv`、`model_projection.csv` 和索引表的稳定字段；
- [旧 benchmark 入口](legacy-launchers.md)：仍受支持的 `run.sh` 和 GLM-5 脚本；
- [开发与测试](development.md) 与 [贡献指南](../CONTRIBUTING.md)。

仓库入口和最短示例见 [README](../README.md)。
