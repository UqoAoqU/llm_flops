"""PyTorch/ROCm BF16 GEMM reference."""

import torch


def operator(left, right):
    return torch.mm(left, right)
