"""Auditable PyTorch dequantize-and-matmul candidate."""

import torch


def _unpack_ue8m0(packed_scale, scale_groups):
    shifts = torch.arange(
        0,
        32,
        8,
        device=packed_scale.device,
        dtype=torch.int64,
    )
    unpacked_u8 = (
        (packed_scale.to(torch.int64).unsqueeze(-1) >> shifts) & 0xFF
    ).reshape(*packed_scale.shape[:-1], packed_scale.shape[-1] * 4)
    unpacked_fp32 = (unpacked_u8.to(torch.int32) << 23).view(torch.float32)
    return unpacked_fp32[..., :scale_groups]


def operator(
    activation,
    activation_scale,
    weight,
    weight_scale,
    output,
):
    del output
    block = 128
    scale_groups = activation.shape[-1] // block
    activation_scale = _unpack_ue8m0(activation_scale, scale_groups)
    weight_scale = _unpack_ue8m0(weight_scale, scale_groups)
    activation_dequantized = activation.float() * activation_scale.repeat_interleave(
        block, dim=-1
    )
    weight_dequantized = weight.float() * weight_scale.repeat_interleave(
        block, dim=-1
    )
    return torch.matmul(
        activation_dequantized, weight_dequantized.transpose(0, 1)
    ).to(torch.bfloat16)
