"""AITER Triton block-FP8 GEMM used by DeepSeek V4 Flash on gfx942."""

import torch

from aiter.ops.triton.gemm.basic.gemm_a8w8_blockscale import (
    gemm_a8w8_blockscale,
)


def operator(x_fp8, weight_fp8, x_scale, weight_scale, output):
    return gemm_a8w8_blockscale(
        x_fp8,
        weight_fp8,
        x_scale,
        weight_scale,
        dtype=torch.bfloat16,
        y=output,
    )
