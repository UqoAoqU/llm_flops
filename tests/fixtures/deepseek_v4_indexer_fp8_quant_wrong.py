"""Intentionally invalid indexer quant output used only by Phase 11 tests."""

import torch


def operator(q, weight, weight_scale, freqs_cis, positions):
    del weight_scale, freqs_cis, positions
    codes = torch.zeros_like(q, dtype=torch.float8_e4m3fn)
    weights = torch.zeros((*weight.shape, 1), device=weight.device, dtype=torch.float32)
    return codes, weights
