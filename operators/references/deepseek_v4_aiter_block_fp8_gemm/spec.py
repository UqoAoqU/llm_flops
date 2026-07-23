"""Reference contract for the MI300X DeepSeek V4 block-FP8 GEMM."""

import importlib

from benchmark_engine.correctness import InputBundle, clone_input_bundle
from benchmark_engine.models import CaseSpec
from benchmark_engine.workloads.deepseek_v4_flash import (
    NumericPath,
    OracleStateComparator,
    normalize_named_tensors,
    sample_tensor,
)


BLOCK = 128


def _quantize_activation(torch, value):
    m, k = value.shape
    blocks = value.float().view(m, k // BLOCK, BLOCK)
    scale = blocks.abs().amax(dim=-1).clamp_min(1.0e-6) / 240.0
    quantized = (blocks / scale.unsqueeze(-1)).to(torch.float8_e4m3fnuz)
    return quantized.view(m, k), scale


def _quantize_weight(torch, value):
    n, k = value.shape
    blocks = value.float().view(n // BLOCK, BLOCK, k // BLOCK, BLOCK)
    scale = blocks.abs().amax(dim=(1, 3)).clamp_min(1.0e-6) / 240.0
    expanded = scale[:, None, :, None]
    quantized = (blocks / expanded).to(torch.float8_e4m3fnuz)
    return quantized.view(n, k), scale


def _dequantized_oracle(torch, x, weight, x_scale, weight_scale):
    x_value = x.float() * x_scale.repeat_interleave(BLOCK, dim=1)
    w_value = weight.float() * weight_scale.repeat_interleave(
        BLOCK, dim=0
    ).repeat_interleave(BLOCK, dim=1)
    return torch.mm(x_value, w_value.t())


class DeepSeekV4AiterBlockFp8GemmSpec:
    operator_id = "deepseek_v4_aiter_block_fp8_gemm"

    def cases(self):
        return (
            CaseSpec(
                "prefill_smoke_m16_k512_n512",
                {"phase": "prefill", "m": 16, "k": 512, "n": 512},
                4101,
                frozenset({"smoke", "oracle", "deepseek_v4_prefill"}),
                600,
            ),
            CaseSpec(
                "decode_smoke_m4_k512_n512",
                {"phase": "decode", "m": 4, "k": 512, "n": 512},
                4102,
                frozenset({"smoke", "oracle", "deepseek_v4_decode"}),
                600,
            ),
            CaseSpec(
                "prefill_q_lora_m256_k4096_n1024",
                {"phase": "prefill", "m": 256, "k": 4096, "n": 1024},
                4103,
                frozenset(
                    {"representative", "performance_only", "deepseek_v4_prefill"}
                ),
                1800,
            ),
            CaseSpec(
                "decode_wqb_m16_k1024_n32768",
                {"phase": "decode", "m": 16, "k": 1024, "n": 32768},
                4104,
                frozenset(
                    {"representative", "performance_only", "deepseek_v4_decode"}
                ),
                1800,
            ),
        )

    def make_inputs(self, case, context):
        torch = importlib.import_module("torch")
        m, k, n = (int(case.symbols[name]) for name in ("m", "k", "n"))
        if k % BLOCK or n % BLOCK:
            raise ValueError("K and N must be divisible by the 128-element scale block")
        device = next(iter(context.cuda), "cuda:0")
        generator = context.cuda.get(device)
        if generator is None:
            raise RuntimeError("block-FP8 GEMM requires a GPU generator")
        x = torch.randn(
            (m, k), device=device, dtype=torch.bfloat16, generator=generator
        ).clamp_(-2.0, 2.0)
        weight = torch.randn(
            (n, k), device=device, dtype=torch.bfloat16, generator=generator
        ).clamp_(-2.0, 2.0)
        x_fp8, x_scale = _quantize_activation(torch, x)
        weight_fp8, weight_scale = _quantize_weight(torch, weight)
        output = torch.empty((m, n), device=device, dtype=torch.bfloat16)
        observed = {
            "x_scale_head": x_scale[:1, : min(4, x_scale.shape[1])],
            "weight_scale_head": weight_scale[:1, : min(4, weight_scale.shape[1])],
        }
        if "performance_only" not in case.tags:
            oracle = _dequantized_oracle(
                torch, x_fp8, weight_fp8, x_scale, weight_scale
            )
            observed["semantic_oracle"] = sample_tensor(oracle)
        return InputBundle(
            args=(x_fp8, weight_fp8, x_scale, weight_scale, output),
            observed_state=observed,
        )

    def clone_inputs(self, inputs):
        return clone_input_bundle(inputs)

    def normalize_output(self, output):
        return normalize_named_tensors(output)

    def comparator(self, case):
        return OracleStateComparator(
            numeric_paths=(
                NumericPath(
                    "output",
                    atol=0.5,
                    rtol=0.02,
                    oracle_path="state.semantic_oracle",
                ),
            ),
            immutable_paths=("state.x_scale_head", "state.weight_scale_head"),
            require_oracle="performance_only" not in case.tags,
            name="block_fp8_gemm_oracle",
        )

    def cost_model(self, case):
        m, k, n = (int(case.symbols[name]) for name in ("m", "k", "n"))
        return {
            "flops": 2 * m * n * k,
            "estimated_bytes": (
                m * k
                + n * k
                + 4 * m * (k // BLOCK)
                + 4 * (n // BLOCK) * (k // BLOCK)
                + 2 * m * n
            ),
            "throughput_units": m * n,
        }

    def layout_contract(self, case):
        m, k, n = (int(case.symbols[name]) for name in ("m", "k", "n"))
        return {
            "x": [m, k],
            "weight": [n, k],
            "x_scale": [m, k // BLOCK],
            "weight_scale": [n // BLOCK, k // BLOCK],
            "output": [m, n],
            "dtype": "float8_e4m3fnuz -> bfloat16",
            "backend": "aiter Triton gemm_a8w8_blockscale",
        }


SPEC = DeepSeekV4AiterBlockFp8GemmSpec()


def cost_model(case):
    return SPEC.cost_model(case)
