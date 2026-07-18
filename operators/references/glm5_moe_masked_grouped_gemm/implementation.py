"""DeepGEMM masked grouped-GEMM reference from the unified GLM-5 scripts."""

import deep_gemm


def operator(activation, activation_scale, weight, weight_scale, output, routing, expected_m):
    deep_gemm.fp8_m_grouped_gemm_nt_masked(
        (activation, activation_scale), (weight, weight_scale), output, routing, expected_m
    )
    return output
