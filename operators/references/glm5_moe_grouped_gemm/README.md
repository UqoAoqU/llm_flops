# GLM-5 legacy contiguous grouped MoE GEMM

This contract migrates `moe_deepgemm.py`. Its five formerly unseeded
distributions are represented by fixed per-token Python `randint` seeds for
each of gate/up/down. Per-expert logical counts are padded to 128 rows exactly
as in the legacy benchmark, and cost uses the padded physical `total_m`.
The reference backend is DeepGEMM contiguous grouped FP8 GEMM.
Unified masked MoE has a separate `glm5_moe_masked_grouped_gemm` contract
because it uses CUDA graphs instead of direct CUDA-event timing.
