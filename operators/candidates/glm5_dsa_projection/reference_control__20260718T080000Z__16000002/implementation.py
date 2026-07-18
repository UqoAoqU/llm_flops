"""DeepGEMM/cuBLAS projection baseline migrated from GLM-5 scripts."""

import deep_gemm
import torch
from sgl_kernel import bmm_fp8


def operator(kind, activation, activation_scale, weight, weight_scale, output):
    if kind == "gemm":
        deep_gemm.fp8_gemm_nt((activation, activation_scale), (weight, weight_scale), output)
        return output
    if kind == "bmm":
        values = []
        for head in range(activation.shape[0]):
            values.append(torch._scaled_mm(
                activation[head], weight[head].t(),
                scale_a=activation_scale[head], scale_b=weight_scale[head],
                out_dtype=torch.bfloat16,
            ))
        return torch.stack(values)
    if kind == "sgl_bmm":
        return bmm_fp8(activation, weight, activation_scale, weight_scale, torch.bfloat16)
    raise ValueError(f"unknown projection kind {kind!r}")
