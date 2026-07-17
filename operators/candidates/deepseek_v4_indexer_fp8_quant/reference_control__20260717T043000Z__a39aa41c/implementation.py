"""Optimized SGLang fused RoPE/Hadamard/FP8 baseline."""

import torch

from sglang.jit_kernel.dsv4 import fused_q_indexer_rope_hadamard_quant
from sglang.jit_kernel.dsv4.elementwise import (
    _jit_main_q_indexer_rope_hadamard_quant_module,
)


_jit_main_q_indexer_rope_hadamard_quant_module(torch.bfloat16)


def operator(q, weight, weight_scale, freqs_cis, positions):
    # The public contract accepts strided head weights. The locked optimized
    # kernel requires stride(1)==1, so normalize only the exceptional boundary
    # input; production/representative contiguous inputs keep the exact legacy
    # steady-state path without an allocation.
    if not weight.is_contiguous():
        weight = weight.contiguous()
    return fused_q_indexer_rope_hadamard_quant(
        q, weight, weight_scale, freqs_cis, positions
    )
