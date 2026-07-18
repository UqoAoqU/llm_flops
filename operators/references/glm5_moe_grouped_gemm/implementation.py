"""DeepGEMM contiguous grouped-GEMM reference from ``moe_deepgemm.py``."""

import deep_gemm


def operator(activation, activation_scale, weight, weight_scale, output, routing):
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        (activation, activation_scale), (weight, weight_scale), output, routing
    )
    return output
