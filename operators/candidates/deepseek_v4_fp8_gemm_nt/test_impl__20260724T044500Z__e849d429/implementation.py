"""Optimized DeepGEMM baseline migrated from the legacy benchmark."""

import deep_gemm


def operator(
    activation,
    activation_scale,
    weight,
    weight_scale,
    output,
):
    deep_gemm.fp8_gemm_nt(
        (activation, activation_scale),
        (weight, weight_scale),
        output,
    )
    return output
