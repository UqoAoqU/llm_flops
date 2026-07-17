# DeepSeek V4 FP8 GEMM migration

Phase 10 migrates the legacy `deep_gemm.fp8_gemm_nt` branch without changing
`deepseek_v4_benchmark.py` or `./run.sh prefill|decode`. The source and result
keys remain mirrored as:

```text
operators/candidates/deepseek_v4_fp8_gemm_nt/<candidate_id>/
results/deepseek_v4_fp8_gemm_nt/<candidate_id>/<evaluation_id>/
```

The reference-owned mapping table retains all twelve legacy FP8 adapter rows.
It covers five unique `(K,N)` shapes--`(7168,2048)`, `(1536,65536)`,
`(7168,1024)`, `(16384,7168)`, and `(7168,129280)`--for prefill M values
1024/2048/4096 and decode M values 16/32. Names, backend labels and instances
remain row-specific for later model projection.

The executable smoke/boundary/representative cases are intentionally bounded:
they cover tile-minimum M, a non-empty smoke batch, and the real decode
`(16,7168,2048)` projection. The other four production shapes remain explicit,
strictly tested mapping metadata rather than automatic correctness cases. This
avoids silently making `full` materialize multi-gigabyte normalized Python
outputs; each mapping can be promoted to a sampled case when projection-level
tests are introduced. No legacy shape is omitted from the metadata contract.

Logical activation scales are `[M,K/128]`; reference scales are transformed to
DeepGEMM's TMA-aligned MN-major layout before first call. Weight scales are
`[N/128,K/128]`. The locked Blackwell kernel interprets the FP32 scale bits as
UE8M0, so both logical scale tensors are seeded selections of exact powers of
two (`0.5`, `1.0`, `2.0`); arbitrary positive FP32 values violate the kernel's
layout assertion. Generation, FP8 conversion, scale alignment and BF16 output
allocation all happen in `make_inputs`. The reference module imports DeepGEMM
once at worker import, and its timed function contains only `fp8_gemm_nt` plus
return of the preallocated output. The standalone candidate imports no
DeepGEMM or root legacy module; its explicit scale expansion, dequantization,
allocation, PyTorch matmul and BF16 conversion are honestly timed. First
call/JIT, warmup and graph capture are reported separately from steady-state
samples.

The microsecond-scale optimized reference uses 20 calls inside every captured
graph sample. This amortizes CUDA event noise while preserving the stage
boundary: import, first-call/JIT, warmup, and graph capture remain separately
reported and are never included in steady-state per-call latency.

The migration role convention is normative for Phase 11 and later: the
existing optimized implementation is the reference/baseline and the newly
evaluated implementation is a candidate. Here the PyTorch candidate is
expected to be slower. `speedup = reference_median / candidate_median` can be
below one, and a formal `all` run may fail the performance gate without
invalidating a passing correctness result.

## B200 acceptance and timer parity

Run the opt-in integration test after the ordinary suite:

```bash
BENCHMARK_ENGINE_RUN_GPU_INTEGRATION=1 CUDA_VISIBLE_DEVICES=0 \
  .runtime/venv/bin/python -m unittest \
  tests.test_deepseek_v4_fp8_gemm_operator.DeepSeekV4Fp8GemmGpuIntegrationTests -v
```

The test prints one JSON `phase10_timer_parity` record containing the case,
new CUDA Graph reference median, legacy `graph_ms` latency, relative delta,
speedup, reference/candidate CV and candidate TFLOPS. The predeclared noise
allowance is 35%: both parity paths replay the same optimized DeepGEMM
reference kernel. The old helper times one graph containing 20 calls and
divides by `runs`; the engine records multiple R-C-C-R interleaved raw samples,
each of whose captured graphs also contains 20 calls and reports per-call
latency. The remaining difference is the interleaving and sampling structure,
not a one-call-versus-20-call comparison. A larger delta fails rather than
silently widening the limit.
The engine artifact is the durable evidence for paired raw samples and the
environment fingerprint; copy the printed parity JSON into the Phase 10
acceptance report alongside its evaluation path.

Correctness must pass smoke, boundary and representative cases at explicit
`rtol=1e-2`, `atol=1e-1` before formal performance is ranked. GPU lock,
telemetry and gate semantics are inherited unchanged from Phase 9.

The CLI selectors must name a suite containing the requested case tags:

```bash
CUDA_VISIBLE_DEVICES=0 ./bench.sh run --suite smoke \
  --operator deepseek_v4_fp8_gemm_nt --candidate 'pytorch_dequant__*'
CUDA_VISIBLE_DEVICES=0 ./bench.sh run --suite full --mode correctness \
  --operator deepseek_v4_fp8_gemm_nt --candidate 'pytorch_dequant__*'
CUDA_VISIBLE_DEVICES=0 ./bench.sh run --suite regression --mode all \
  --operator deepseek_v4_fp8_gemm_nt --candidate 'pytorch_dequant__*' \
  --tag representative
```
