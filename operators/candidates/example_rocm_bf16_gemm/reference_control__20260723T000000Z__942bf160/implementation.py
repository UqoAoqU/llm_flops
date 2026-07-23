"""Byte-equivalent control candidate for the MI300X framework probe."""

import torch


def operator(left, right):
    return torch.mm(left, right)
