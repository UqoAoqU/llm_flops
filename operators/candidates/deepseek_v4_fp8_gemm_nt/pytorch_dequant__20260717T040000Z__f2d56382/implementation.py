"""Auditable PyTorch dequantize-and-matmul candidate."""

import torch


def operator(
    activation,
    activation_scale,
    activation_scale_aligned,
    weight,
    weight_scale,
    output,
):
    del activation_scale_aligned, output
    block = 128
    activation_dequantized = activation.float() * activation_scale.repeat_interleave(
        block, dim=1
    )
    weight_dequantized = weight.float() * weight_scale.repeat_interleave(
        block, dim=0
    ).repeat_interleave(block, dim=1)
    return torch.matmul(
        activation_dequantized, weight_dequantized.transpose(0, 1)
    ).to(torch.bfloat16)
