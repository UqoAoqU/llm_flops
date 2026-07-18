"""cuBLAS/DeepGEMM indexer projections used by the legacy scripts."""

import deep_gemm
import torch


def operator(backend, activation, activation_scale, weight, weight_scale, output):
    if backend == "cublas_fp8":
        return torch._scaled_mm(activation, weight.t(), scale_a=activation_scale,
                                scale_b=weight_scale, out_dtype=torch.bfloat16)
    if backend == "deepgemm_fp8":
        deep_gemm.fp8_gemm_nt((activation, activation_scale), (weight, weight_scale), output)
        return output
    if backend == "deepgemm_bf16":
        deep_gemm.bf16_gemm_nt(activation, weight, output)
        return output
    raise ValueError(f"unknown indexer backend {backend!r}")
