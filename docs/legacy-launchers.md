# 旧 Benchmark 入口

新的 `bench.sh` 与原有脚本并行存在。旧入口仍用于迁移前的模型级汇总和回归对照；引擎
变更不应改变这些命令的参数或 CSV 语义。

## DeepSeek V4 Pro

环境检查和 smoke：

```bash
./bootstrap.sh
./run.sh check
CUDA_VISIBLE_DEVICES=0 ./run.sh smoke
```

Prefill：

```bash
./run.sh prefill --quant-profile fp8_mxfp8 \
  --m 1024,2048,4096 --context 65536 \
  --warmup 5 --runs 20 \
  --csv results/deepseek_v4_pro_fp8_mxfp8_prefill_kv65536.csv
```

Decode：

```bash
./run.sh decode --quant-profile fp8_mxfp8 \
  --m 16,32 --context 65536 \
  --warmup 5 --runs 20 \
  --csv results/deepseek_v4_pro_fp8_mxfp8_decode_kv65536.csv
```

`mxfp4` 使用 FP4 indexer 与 MXFP4 weight/MXFP8 activation MoE；`fp8_mxfp8`
使用 FP8 indexer 与 FP8 weight/MXFP8 activation MoE。原始 KV context 为 65536，
C4/C128 压缩 context 分别为 16384/512。`./run.sh compare` 比较已生成 profile CSV。

旧 CSV 字段为：`phase, quant_profile, environment_fingerprint, m, context,
operator, backend, instances, call_ms, model_ms, pct, status, input_shape,
output_shape, error`。它不同于 Benchmark Engine 的 schema-versioned artifact。

## GLM-5 脚本

| 脚本 | 作用 | 默认输出 |
|---|---|---|
| `bench_glm5_prefill.py` | Prefill 全算子汇总 | `glm5_unified_perf.csv` |
| `bench_glm5_decode.py` | Decode 全算子汇总 | `glm5_decode_perf.csv` |
| `bench_glm5_deepep.py` | DeepEP layout/dispatch/combine | `glm5_deepep_dispatch_perf.csv` |
| `dsa_flashmla.py` | Sparse prefill attention | `glm5_sparse_prefill_perf.csv` |
| `mla_flashmla.py` | Dense MLA prefill | `glm5_dense_prefill_perf.csv` |
| `dsa_indexer.py` | DSA indexer GEMM | `glm5_dsa_indexer_perf.csv` |
| `dsa_projection.py` | Attention GEMM/BMM | `glm5_attention_gemm_perf.csv` |
| `moe_deepgemm.py` | MoE grouped GEMM | `glm5_moe_deepgemm_perf.csv` |

单卡脚本直接在仓库环境运行，例如：

```bash
.runtime/venv/bin/python bench_glm5_prefill.py
.runtime/venv/bin/python bench_glm5_decode.py
NUM_RUNS=50 NUM_DISTRIBUTIONS=10 \
  .runtime/venv/bin/python moe_deepgemm.py
```

DeepEP 多进程/多节点需要在每个节点分别配置 `--nnodes`、`--node-rank`、
`--master-addr` 和相同 `--master-port`。Benchmark Engine 当前不提供多 GPU 调度，
因此这些通信脚本仍使用旧入口。

算子逐步进入新框架后，optimized legacy path 作为 reference，candidate 与其做正确性和
性能比较；模型级 instances 投影在完成迁移前仍以旧 CSV 为对照。
