# DeepSeek V4 Pro Operator Performance Benchmarks

DeepSeek V4 Pro Prefill、Decode 及各算子的 CUDA 性能测试。底层后端包括 DeepGEMM、SGL Kernel、FlashMLA 和 FlashInfer，计时使用 CUDA Graph。

## Benchmark Engine

> Phase 9 status: correctness-gated `R-C-C-R` measurement, physical-GPU
> locking, explicit performance gates, and strict artifact comparison are enabled.

### Five-minute CPU quick start

```bash
./bootstrap.sh
./bench.sh validate --operator example_cpu_add
./bench.sh list --operator example_cpu_add
./bench.sh run --suite smoke --operator example_cpu_add
```

The last command prints the `run_id` and mirrored result path. To inspect an
intentional numerical failure and its reproduction command:

```bash
./bench.sh run --mode correctness \
  --operator example_cpu_add \
  --candidate numeric_bad__20260716T120100Z__c1ac82e6 \
  --case tiny --seed 1
./bench.sh summarize results/example_cpu_add/numeric_bad__20260716T120100Z__c1ac82e6/<evaluation_id>
./bench.sh run --resume <run_id>
```

To run the included CPU performance example with the Phase 9 defaults
(warmup 5, samples 30, inner iterations 20):

```bash
./bench.sh run --mode performance \
  --operator example_cpu_add \
  --candidate quickstart__20260716T120000Z__4279e756 \
  --case tiny --timer wall_clock \
  --warmup 5 --samples 30 --inner-iterations 20
```

The mirrored evaluation directory contains `results.csv` schema v3 and
`performance_samples.csv` schema v2. Summary rows include requested/effective
timer, fallback reason, stage times, reference/candidate statistics, stability,
and optional theoretical cost rates. See the
[performance guide](docs/performance.md) for the field semantics and Phase 9
scope.

Compare compatible artifacts without creating a `results/<run_id>` directory:

```bash
./bench.sh compare --result <evaluation_id> --baseline-result <baseline_evaluation_id>
./bench.sh compare --run <run_id> --baseline-run <baseline_run_id>
```

Exit codes are 0 for pass, 1 for a correctness or performance failure, 2 for
usage or configuration, 3 for worker/engine infrastructure, and 130 for Ctrl-C. Every
completed case is durable before the next case starts; resume never reruns an
existing `result_id`. These commands need no GPU and do not change `run.sh`.

### B200 FP8 GEMM smoke

The CUDA smoke is deliberately separate from the five-minute CPU path because
its first run can compile DeepGEMM kernels:

```bash
CUDA_VISIBLE_DEVICES=0 ./bench.sh validate --operator deepseek_v4_fp8_gemm_nt
CUDA_VISIBLE_DEVICES=0 ./bench.sh run --suite smoke \
  --operator deepseek_v4_fp8_gemm_nt --candidate 'pytorch_dequant__*'
CUDA_VISIBLE_DEVICES=0 ./bench.sh run --suite full --mode correctness \
  --operator deepseek_v4_fp8_gemm_nt --candidate 'pytorch_dequant__*'
CUDA_VISIBLE_DEVICES=0 ./bench.sh run --suite regression --mode all \
  --operator deepseek_v4_fp8_gemm_nt --candidate 'pytorch_dequant__*' \
  --tag representative
```

The migrated contract, scale layouts, tolerance rationale, legacy adapter map,
and timer-parity procedure are documented in the
[FP8 GEMM migration note](docs/deepseek-v4-fp8-gemm-migration.md).
The existing DeepGEMM kernel is the optimized reference/baseline; the formal
candidate is an independent PyTorch dequantize-and-matmul implementation.
The microsecond-scale baseline uses 20 calls per captured graph sample to
amortize event noise; import, first-call/JIT and graph-build costs remain
separate artifact fields.

新的 benchmark engine 正在与现有 DeepSeek V4 和 GLM-5 入口并行建设。
当前 engine 提供可安装的 `benchmark_engine` 包、静态 Registry、确定性 dry-run
计划、可恢复 artifact，以及隔离 import/build 的 Controller/Worker 骨架：

~~~bash
./bootstrap.sh
.runtime/venv/bin/bench --help
./bench.sh --help
~~~

`bench list`、`bench validate`、`bench env`、`bench run --dry-run`、CPU
correctness 和 Phase 8 performance 执行均已可用。输入隔离、输出标准化、
Comparator、Timer 和 Evaluator API 均在隔离 worker 内运行；正确性失败会阻断
正式性能采样。
正确性契约见
[correctness guide](docs/correctness.md)。
架构与开发约定见 [文档索引](docs/index.md)，完整方案见
[批准的设计](design.md)，贡献要求见 [CONTRIBUTING.md](CONTRIBUTING.md)。
下文所有 legacy 用法保持不变。

## DeepSeek V4 Pro 配置

| 参数 | 值 |
|------|-----|
| Transformer layers | 61 |
| C4 layers | 30 |
| C128 layers | 30 |
| Dense/SWA layers | 1 |
| Hidden size | 7168 |
| Q LoRA rank | 1536 |
| Attention heads | 128 |
| Attention head dim | 512 |
| Indexer heads | 64 |
| Indexer head dim | 128 |
| Indexer TopK | 1024 |
| Global experts | 384 |
| Expert parallel size | 24 |
| Local experts | 16 |
| Experts per token | 6 |
| MoE intermediate size | 3072 |
| Attention/Indexer tensor parallel | 不切分 |

原始 KV 长度固定为 65536。C4 和 C128 层对应的 KV 长度分别为 16384 和 512。

## Prefill 阶段

Prefill 中 `M=seq_Q`，测试 case 为：

| `seq_Q` | 原始 `seq_KV` | C4 `seq_KV` | C128 `seq_KV` |
|---------|---------------|-------------|---------------|
| 1024 | 65536 | 16384 | 512 |
| 2048 | 65536 | 16384 | 512 |
| 4096 | 65536 | 16384 | 512 |

Prefill 不包含 KV append。执行链为：

1. Q/KV 投影。
2. C4/C128 Compressor。
3. C4 Indexer Q/Head 投影、量化、Paged MQA Logits 和 TopK。
4. 60 个稀疏 Attention 层和 1 个 Dense/SWA Attention 层。
5. WO_A、WO_B 输出投影。
6. 61 层 Routed Expert Fused MoE。
7. LM Head。

## Decode 阶段

Decode 中 `M=batch_size`，每个 request 处理一个 query token。测试 case 为：

| batch size | 原始 `seq_KV` | C4 `seq_KV` | C128 `seq_KV` |
|------------|---------------|-------------|---------------|
| 16 | 65536 | 16384 | 512 |
| 32 | 65536 | 16384 | 512 |

Decode 不执行 Compressor。执行链为：

1. Q/KV 投影。
2. C4 Indexer Q/Head 投影、量化、Paged MQA Logits 和 TopK。
3. 30 个 C4 dual-cache Attention、30 个 C128 dual-cache Attention 和 1 个 Dense/SWA Attention。
4. WO_A、WO_B 输出投影。
5. 61 层 Routed Expert Fused MoE。
6. LM Head。

## DeepSeek V4 Pro 算子测试

以下算子表对应 `fp8_mxfp8` quant profile。

### Prefill 算子

下表 shape 使用 `seq_Q=1024`、原始 `seq_KV=65536`：

| 算子 | 次数 | 后端 | 输入 | 输出 |
|------|-----:|------|------|------|
| Fused WQ_A + WKV | 61 | DeepGEMM FP8 GEMM | `x=(1024,7168)`, `w=(2048,7168)` | `(1024,2048)` |
| Q RMSNorm + WQ_B | 61 | DeepGEMM FP8 GEMM | `x=(1024,1536)`, `w=(65536,1536)` | `(1024,65536)` |
| Compressor C4 | 30 | DeepGEMM FP8 GEMM | `x=(1024,7168)`, `w=(2048,7168)` | `(1024,2048)` |
| Compressor C128 | 30 | DeepGEMM FP8 GEMM | `x=(1024,7168)`, `w=(1024,7168)` | `(1024,1024)` |
| C4 Indexer Q Projection | 30 | DeepGEMM FP8 GEMM | `x=(1024,1536)`, `w=(65536,1536)` | `(1024,65536)` |
| C4 Indexer Head Weight | 30 | cuBLAS BF16 GEMM | `x=(1024,7168)`, `w=(64,7168)` | `(1024,64)` |
| C4 Indexer FP8 Quant | 30 | SGLang fused RoPE/Hadamard FP8 | `q=(1024,64,128)` | `q_fp8=(1024,64,128)` |
| C4 FP8 Paged MQA Logits | 30 | DeepGEMM `fp8_paged_mqa_logits` | `q=(1024,1,64,128)`, `kv=16384` | `logits=(1024,16384)` |
| C4 TopK Transform | 30 | SGLang JIT | `scores=(1024,16384)` | `indices=(1024,1024)` |
| Sparse Prefill Attention | 60 | SGL Kernel FlashMLA | `q=(1024,128,512)`, `kv=(65536,1,512)`, `indices=(1024,1,1024)` | `(1024,128,512)` |
| Dense SWA Attention | 1 | SGL Kernel FlashMLA | `q=(1024,128,512)`, `kv=(65536,1,512)`, `indices=(1024,1,128)` | `(1024,128,512)` |
| WO_A Grouped Projection | 61 | cuBLAS BF16 BMM | `x=(1024,16,4096)`, `w=(16,4096,1024)` | `(1024,16,1024)` |
| WO_B Projection | 61 | DeepGEMM FP8 GEMM | `x=(1024,16384)`, `w=(7168,16384)` | `(1024,7168)` |
| Routed Expert Fused MoE | 61 | FlashInfer TRTLLM FP8/MXFP8 | `x=(1024,7168)`, `topk=(1024,6)`, `local_pairs=256`, `w13=(16,6144,7168)`, `w2=(16,7168,3072)` | `(1024,7168)` |
| LM Head | 1 | DeepGEMM FP8 GEMM | `x=(1024,7168)`, `w=(129280,7168)` | `(1024,129280)` |

### Decode 算子

下表 shape 使用 `batch_size=16`、原始 `seq_KV=65536`：

| 算子 | 次数 | 后端 | 输入 | 输出 |
|------|-----:|------|------|------|
| Fused WQ_A + WKV | 61 | DeepGEMM FP8 GEMM | `x=(16,7168)`, `w=(2048,7168)` | `(16,2048)` |
| Q RMSNorm + WQ_B | 61 | DeepGEMM FP8 GEMM | `x=(16,1536)`, `w=(65536,1536)` | `(16,65536)` |
| C4 Indexer Q Projection | 30 | DeepGEMM FP8 GEMM | `x=(16,1536)`, `w=(65536,1536)` | `(16,65536)` |
| C4 Indexer Head Weight | 30 | cuBLAS BF16 GEMM | `x=(16,7168)`, `w=(64,7168)` | `(16,64)` |
| C4 Indexer FP8 Quant | 30 | SGLang fused RoPE/Hadamard FP8 | `q=(16,64,128)` | `q_fp8=(16,64,128)` |
| C4 FP8 Paged MQA Logits | 30 | DeepGEMM `fp8_paged_mqa_logits` | `q=(16,1,64,128)`, `kv=16384` | `logits=(16,16384)` |
| C4 TopK Transform | 30 | SGLang JIT | `scores=(16,16384)` | `indices=(16,1024)` |
| Sparse Decode Attention C4 | 30 | SGL Kernel FlashMLA dual-cache | `q=(16,1,128,512)`, `context=65536` | `(16,1,128,512)` |
| Sparse Decode Attention C128 | 30 | SGL Kernel FlashMLA dual-cache | `q=(16,1,128,512)`, `context=65536` | `(16,1,128,512)` |
| Dense SWA Attention | 1 | SGL Kernel FlashMLA | `q=(16,1,128,512)`, `context=65536` | `(16,1,128,512)` |
| WO_A Grouped Projection | 61 | cuBLAS BF16 BMM | `x=(16,16,4096)`, `w=(16,4096,1024)` | `(16,16,1024)` |
| WO_B Projection | 61 | DeepGEMM FP8 GEMM | `x=(16,16384)`, `w=(7168,16384)` | `(16,7168)` |
| Routed Expert Fused MoE | 61 | FlashInfer TRTLLM FP8/MXFP8 | `x=(16,7168)`, `topk=(16,6)`, `local_pairs=4`, `w13=(16,6144,7168)`, `w2=(16,7168,3072)` | `(16,7168)` |
| LM Head | 1 | DeepGEMM FP8 GEMM | `x=(16,7168)`, `w=(129280,7168)` | `(16,129280)` |

### 环境版本

| 组件 | 版本 |
|------|------|
| GPU | NVIDIA B200 (SM100) |
| Python | 3.12 |
| CUDA | 13.0 |
| PyTorch | 2.11.0 |
| SGLang | commit `19593359971ebc3582a74f000bf285488d993362` |
| SGL Kernel | 0.4.4 |
| SGL DeepGEMM | 0.1.4 |
| FlashInfer Python/Cubin | 0.6.12 |
| NVIDIA CUTLASS DSL | 4.5.2 |

### 安装与检查

```bash
./bootstrap.sh
./run.sh check
CUDA_VISIBLE_DEVICES=0 ./run.sh smoke
```

环境目录：`.runtime/venv`

### Prefill 运行

```bash
./run.sh prefill --quant-profile fp8_mxfp8 \
  --m 1024,2048,4096 --context 65536 \
  --warmup 5 --runs 20 \
  --csv results/deepseek_v4_pro_fp8_mxfp8_prefill_kv65536.csv
```

### Decode 运行

```bash
./run.sh decode --quant-profile fp8_mxfp8 \
  --m 16,32 --context 65536 \
  --warmup 5 --runs 20 \
  --csv results/deepseek_v4_pro_fp8_mxfp8_decode_kv65536.csv
```

### Quant Profile

| Profile | Indexer | Routed MoE |
|---------|---------|------------|
| `mxfp4` | FP4 quant + FP8/FP4 paged MQA logits | MXFP4 weight + MXFP8 activation |
| `fp8_mxfp8` | FP8 quant + FP8 paged MQA logits | FP8 weight + MXFP8 activation |

### Profile 对比

```bash
./run.sh compare
```

### CSV 字段

`phase`, `quant_profile`, `environment_fingerprint`, `m`, `context`,
`operator`, `backend`, `instances`, `call_ms`, `model_ms`, `pct`, `status`,
`input_shape`, `output_shape`, `error`

## 附录：原 GLM-5 Benchmarks

以下内容保留仓库原有的 GLM-5 单算子和端到端测试说明，不属于上述 DeepSeek V4 Pro benchmark。

### GLM-5 模型参数

所有脚本使用统一的 GLM-5 模型配置：

| 参数 | 值 |
|------|-----|
| hidden_size | 6144 |
| q_lora_rank | 2048 |
| kv_lora_rank | 512 |
| qk_nope_head_dim | 192 |
| qk_rope_head_dim | 64 |
| num_attention_heads | 64 |
| v_head_dim | 256 |
| index_n_heads | 32 |
| index_head_dim | 128 |
| moe_intermediate_size | 2048 |
| n_routed_experts | 256 (全量) / 8 (单卡) |
| num_experts_per_tok | 8 |

---

### GLM-5 测试脚本

#### 1. bench_glm5_prefill.py — Prefill 阶段全算子性能

测试 sglang prefill 路径下的所有 GLM-5 算子，包括 Attention GEMM、DSA、DSA Indexer、MoE。

**覆盖算子：**
- Attention: `fused_qkv_a_proj`, `q_b_proj`, `absorbed_W_UK`, `absorbed_W_UV`, `o_proj`
- DSA: `flash_mla_sparse_fwd`（bf16 sparse attention，s_q=M，每 query gather topk=2048）
- DSA Indexer: `index_k_proj`, `index_q_upproj`, `index_weights_proj`, `index_score`（fp8_mqa_logits）
- MoE: `gate_proj`, `up_proj`, `down_proj`（fp8_m_grouped_gemm_nt_masked）

**参数：**
- `M`（输入 token 数）：默认 [1024, 2048, 4096]
- `S`（KV 上下文长度）：默认 [65536]

**运行：**
```bash
python bench_glm5_prefill.py
```

**输出：** 各算子延迟（ms），按耗时降序排列的 summary 表，CSV 保存到 `glm5_unified_perf.csv`。

---

#### 2. bench_glm5_decode.py — Decode 阶段全算子性能

测试 sglang decode 路径下的所有算子。与 prefill 的区别：M 为 batch_size（每请求 1 token），batch flatten 成 s_q=M，attention 同样使用 DSA sparse kernel（`flash_mla_sparse_fwd`），DSA Indexer 使用 `fp8_paged_mqa_logits`。

**覆盖算子：** 同 prefill，但 attention 为 decode 形态的 DSA sparse（s_q=batch），indexer score 使用 decode 版本的 `fp8_paged_mqa_logits`。

**参数：**
- `M`（batch_size）：默认 [1, 4, 8, 16, 32, 64]
- `S`（KV 上下文长度）：默认 [65536]

**运行：**
```bash
python bench_glm5_decode.py
```

**输出：** CSV 保存到 `glm5_decode_perf.csv`。

---

#### 3. bench_glm5_deepep.py — DeepEP All-to-All 通信性能

测试 MoE 的 expert parallel 通信开销：`get_dispatch_layout` + `dispatch`（发送 token 到专家所在 GPU）+ `combine`（收集结果）。

**支持多种负载均衡场景：**
- `balanced`：EPLB 均匀路由
- `mild` / `medium` / `heavy`：递增的路由倾斜

**参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--nnodes` | 节点数 | 1 |
| `--node-rank` | 当前节点编号 | 0 |
| `--master-addr` | 主节点 IP | 127.0.0.1 |
| `--master-port` | 端口 | 29500 |
| `--hidden` | hidden dim | 6144 |
| `--num-sms` | DeepEP 使用的 SM 数 | 24 |
| `--use-fp8` | 启用 FP8 传输 | True |
| `--scenario` | 测试场景 | all |

**运行：**
```bash
# 单节点 8 卡
python bench_glm5_deepep.py --nnodes 1 --m-per-gpu 4096

# 多节点（每个节点上执行，指定各自的 node-rank）
# Node 0:
python bench_glm5_deepep.py --nnodes 2 --node-rank 0 --master-addr 10.0.0.1 --m-per-gpu 4096
# Node 1:
python bench_glm5_deepep.py --nnodes 2 --node-rank 1 --master-addr 10.0.0.1 --m-per-gpu 4096

# 只测 balanced 场景
python bench_glm5_deepep.py --nnodes 1 --scenario balanced
```

**输出：** layout / dispatch / combine 延迟，expert 负载统计，CSV 保存到 `glm5_deepep_dispatch_perf.csv`。

---

#### 4. dsa_flashmla.py — FlashMLA Sparse Prefill 性能

测试 DSA（Dynamic Sparse Attention）中的 sparse prefill 算子 `flash_mla_sparse_fwd`，`s_q` 与 `s_kv` 独立配置并按笛卡尔积扫描。该算子对全量上下文做稀疏 prefill，不涉及 KV cache 命中率。

**场景：**
- `SQ_LIST = [16384, 32768, 65536, 131072]`（query token 数，独立）
- `SKV_LIST = [16384, 32768, 65536, 131072]`（KV cache 长度，独立）
- `topk = 2048`（固定）
- 每个 query 从 s_kv cache 中 gather topk 个 key

**运行：**
```bash
python dsa_flashmla.py
```

**输出：** 各 `(s_q, s_kv)` 组合下的延迟（ms）、TFlops、TB/s、计算访存比，CSV 保存到 `glm5_sparse_prefill_perf.csv`。

---

#### 5. mla_flashmla.py — FlashMLA Dense Prefill (传统 MLA) 性能

测试传统 MLA 的 dense prefill 算子 `flash_mla_with_kvcache`（paged KV cache，非稀疏）。与 `dsa_flashmla.py` 的区别：去掉稀疏（topk/indices），每个 query 对全部 s_kv 做 dense attention。除 topk 外的参数与 DSA 完全一致，`s_q`/`s_kv` 同样独立配置、笛卡尔积扫描，不涉及命中率。

**场景：**
- `SQ_LIST = [16384, 32768, 65536, 131072]`（query token 数，独立）
- `SKV_LIST = [16384, 32768, 65536, 131072]`（KV cache 长度，独立）
- dense，无 topk

**运行：**
```bash
python mla_flashmla.py
```

**输出：** 各 `(s_q, s_kv)` 组合下的延迟（ms）、TFlops、TB/s、计算访存比，CSV 保存到 `glm5_dense_prefill_perf.csv`。

---

#### 6. dsa_indexer.py — DSA Indexer GEMM (cuBLAS FP8) 性能

单独测试 DSA Indexer 的 4 个 GEMM 算子，使用 `torch._scaled_mm`（cuBLAS FP8）。

**覆盖算子：**
- `index_k_proj`：[S, 6144] × [6144, 128]
- `index_q_upproj`：[M, 2048] × [2048, 4096]
- `index_weights_proj`：[M, 6144] × [6144, 32]
- `index_score`：[32×M, 128] × [128, S]

**参数：**
- `M`：默认 [16, 256, 512, 1024]
- `S`：默认 [65536, 131072, 262144]

**运行：**
```bash
python dsa_indexer.py
```

**输出：** 延迟、TFlops、TB/s、计算访存比，CSV 保存到 `glm5_dsa_indexer_perf.csv`。

---

#### 7. dsa_projection.py — Attention GEMM/BMM (DeepGEMM FP8) 性能

单独测试 MLA attention 中的 6 个 GEMM/BMM 算子，使用 DeepGEMM FP8。

**覆盖算子：**
- `q_a_proj`：GEMM [M, 6144] × [6144, 2048]
- `q_b_proj`：GEMM [M, 2048] × [2048, 16384]
- `absorbed_W_UK`：BMM batch=64, [M, 192] × [192, 512]
- `kv_a_proj`：GEMM [M, 6144] × [6144, 576]
- `absorbed_W_UV`：BMM batch=64, [M, 512] × [512, 256]
- `o_proj`：GEMM [M, 16384] × [16384, 6144]

**参数：**
- `M`：默认 [1024, 4096, 16384, 65536]

**运行：**
```bash
python dsa_projection.py
```

**输出：** 延迟、TFlops、TB/s，CSV 保存到 `glm5_attention_gemm_perf.csv`。

---

#### 8. moe_deepgemm.py — MoE Grouped GEMM (DeepGEMM FP8) 性能

单独测试 MoE FFN 的 grouped GEMM（contiguous layout），使用 `deep_gemm.m_grouped_fp8_gemm_nt_contiguous`。测试多种随机 token 分布。

**覆盖算子：**
- `gate_proj`：K=6144, N=2048
- `up_proj`：K=6144, N=2048
- `down_proj`：K=2048, N=6144

**参数：**
- `TOTAL_TOKENS`：默认 128（16 × 8 experts）
- `NUM_DISTRIBUTIONS`：默认 5 种随机分布
- 可通过环境变量覆盖：`NUM_RUNS`、`NUM_WARMUP`、`NUM_DISTRIBUTIONS`

**运行：**
```bash
python moe_deepgemm.py

# 自定义
NUM_RUNS=50 NUM_DISTRIBUTIONS=10 python moe_deepgemm.py
```

**输出：** 各分布下的延迟、TFlops、TB/s，CSV 保存到 `glm5_moe_deepgemm_perf.csv`。

---

### GLM-5 脚本关系

| 脚本 | 定位 | 适用场景 |
|------|------|----------|
| `bench_glm5_prefill.py` | 端到端 prefill | 评估单层 prefill 总耗时和瓶颈 |
| `bench_glm5_decode.py` | 端到端 decode | 评估单层 decode 总耗时和瓶颈 |
| `bench_glm5_deepep.py` | 通信 | 评估 MoE EP 通信开销 |
| `dsa_flashmla.py` | 单算子 | 评估 sparse attention 随 (s_q, s_kv) 组合的变化 |
| `mla_flashmla.py` | 单算子 | 评估 dense MLA attention 随 (s_q, s_kv) 组合的变化 |
| `dsa_indexer.py` | 单算子 | 评估 DSA indexer 各 GEMM（cuBLAS） |
| `dsa_projection.py` | 单算子 | 评估 attention GEMM/BMM（DeepGEMM） |
| `moe_deepgemm.py` | 单算子 | 评估 MoE grouped GEMM 不同分布下性能 |
