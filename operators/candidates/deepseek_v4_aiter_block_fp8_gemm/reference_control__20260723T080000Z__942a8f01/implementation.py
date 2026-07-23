"""Control candidate for the MI300X AITER block-FP8 GEMM reference."""

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
