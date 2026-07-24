"""Reference-owned contract and case metadata for DeepSeek V4 FP8 GEMM.

This module stays safe for controller-side registry discovery: heavyweight
CUDA modules are resolved only while runtime inputs are built or cloned.
"""

import importlib

from benchmark_engine.correctness import CalcDiffComparator, InputBundle, OracleGate
from benchmark_engine.models import CaseSpec


SCALE_BLOCK = 128
FP8_DTYPE = "float8_e4m3fn"
OUTPUT_DTYPE = "bfloat16"
UPSTREAM_CONTRACT = {
    "project": "sgl-project/DeepGEMM",
    "revision": "731e7c7a97d269e4b9f482ea18d0e709a948f293",
    "test": "tests/test_fp8_fp4.py::test_gemm",
    "metric": "calc_diff",
    "threshold": 1e-3,
}
PREFILL_M_VALUES = (1024, 2048, 4096)
DECODE_M_VALUES = (16, 32)

# This table is the lossless bridge from every legacy Adapter(kind="fp8") row
# to the new operator contract.  Repeated shapes remain separate because their
# model-projection instance counts and names are semantically significant.
LEGACY_ADAPTER_MAPPINGS = (
    {"phase": "prefill", "quant_profiles": ("mxfp4", "fp8_mxfp8"), "kind": "fp8", "adapter_name": "Fused WQ_A + WKV", "backend": "DeepGEMM fp8_gemm_nt", "instances": 61, "m_values": PREFILL_M_VALUES, "k": 7168, "n": 2048},
    {"phase": "prefill", "quant_profiles": ("mxfp4", "fp8_mxfp8"), "kind": "fp8", "adapter_name": "Q RMSNorm + WQ_B", "backend": "DeepGEMM fp8_gemm_nt", "instances": 61, "m_values": PREFILL_M_VALUES, "k": 1536, "n": 65536},
    {"phase": "prefill", "quant_profiles": ("mxfp4", "fp8_mxfp8"), "kind": "fp8", "adapter_name": "Compressor WKV-Gate GEMM (C4)", "backend": "DeepGEMM fp8_gemm_nt", "instances": 30, "m_values": PREFILL_M_VALUES, "k": 7168, "n": 2048},
    {"phase": "prefill", "quant_profiles": ("mxfp4", "fp8_mxfp8"), "kind": "fp8", "adapter_name": "Compressor WKV-Gate GEMM (C128)", "backend": "DeepGEMM fp8_gemm_nt", "instances": 30, "m_values": PREFILL_M_VALUES, "k": 7168, "n": 1024},
    {"phase": "prefill", "quant_profiles": ("mxfp4", "fp8_mxfp8"), "kind": "fp8", "adapter_name": "C4 Indexer Q Projection", "backend": "DeepGEMM fp8_gemm_nt", "instances": 30, "m_values": PREFILL_M_VALUES, "k": 1536, "n": 65536},
    {"phase": "prefill", "quant_profiles": ("mxfp4", "fp8_mxfp8"), "kind": "fp8", "adapter_name": "WO_B Projection", "backend": "DeepGEMM fp8_gemm_nt", "instances": 61, "m_values": PREFILL_M_VALUES, "k": 16384, "n": 7168},
    {"phase": "prefill", "quant_profiles": ("mxfp4", "fp8_mxfp8"), "kind": "fp8", "adapter_name": "LM Head", "backend": "DeepGEMM FP8 full vocab", "instances": 1, "m_values": PREFILL_M_VALUES, "k": 7168, "n": 129280},
    {"phase": "decode", "quant_profiles": ("mxfp4", "fp8_mxfp8"), "kind": "fp8", "adapter_name": "Fused WQ_A + WKV", "backend": "DeepGEMM fp8_gemm_nt", "instances": 61, "m_values": DECODE_M_VALUES, "k": 7168, "n": 2048},
    {"phase": "decode", "quant_profiles": ("mxfp4", "fp8_mxfp8"), "kind": "fp8", "adapter_name": "Q RMSNorm + WQ_B", "backend": "DeepGEMM fp8_gemm_nt", "instances": 61, "m_values": DECODE_M_VALUES, "k": 1536, "n": 65536},
    {"phase": "decode", "quant_profiles": ("mxfp4", "fp8_mxfp8"), "kind": "fp8", "adapter_name": "C4 Indexer Q Projection", "backend": "DeepGEMM fp8_gemm_nt", "instances": 30, "m_values": DECODE_M_VALUES, "k": 1536, "n": 65536},
    {"phase": "decode", "quant_profiles": ("mxfp4", "fp8_mxfp8"), "kind": "fp8", "adapter_name": "WO_B Projection", "backend": "DeepGEMM fp8_gemm_nt", "instances": 61, "m_values": DECODE_M_VALUES, "k": 16384, "n": 7168},
    {"phase": "decode", "quant_profiles": ("mxfp4", "fp8_mxfp8"), "kind": "fp8", "adapter_name": "LM Head", "backend": "DeepGEMM FP8 full vocab", "instances": 1, "m_values": DECODE_M_VALUES, "k": 7168, "n": 129280},
)


def _runtime_modules():
    torch = importlib.import_module("torch")
    fp8_kernel = importlib.import_module(
        "sglang.srt.layers.quantization.fp8_kernel"
    )
    fp8_utils = importlib.import_module(
        "sglang.srt.layers.quantization.fp8_utils"
    )
    deep_gemm_wrapper = importlib.import_module(
        "sglang.srt.layers.deep_gemm_wrapper"
    )
    return (
        torch,
        fp8_kernel.sglang_per_token_group_quant_fp8,
        fp8_utils.quant_weight_ue8m0,
        fp8_utils.transform_scale_ue8m0,
        deep_gemm_wrapper.DEEPGEMM_SCALE_UE8M0,
    )


def _quantize_production_inputs(activation, weight):
    (
        _,
        quant_activation,
        quant_weight,
        transform_weight_scale,
        scale_ue8m0,
    ) = _runtime_modules()
    if not scale_ue8m0:
        raise RuntimeError(
            "deepseek_v4_fp8_gemm_nt contract v3 requires the SGLang "
            "SM100/SM103 packed UE8M0 DeepGEMM path"
        )
    activation_fp8, activation_scale = quant_activation(
        activation,
        SCALE_BLOCK,
        column_major_scales=True,
        scale_tma_aligned=True,
        scale_ue8m0=True,
    )
    weight_fp8, weight_scale = quant_weight(
        weight,
        [SCALE_BLOCK, SCALE_BLOCK],
    )
    weight_scale = transform_weight_scale(
        weight_scale,
        mn=weight_fp8.shape[-2],
    )
    return activation_fp8, activation_scale, weight_fp8, weight_scale


class DeepSeekV4Fp8GemmNtSpec:
    operator_id = "deepseek_v4_fp8_gemm_nt"

    def cases(self):
        return (
            CaseSpec(
                case_id="smoke_m16_k128_n128",
                symbols={"m": 16, "k": 128, "n": 128, "scale_block": SCALE_BLOCK},
                seed=17,
                tags=frozenset({"smoke"}),
                timeout_s=600,
            ),
            CaseSpec(
                case_id="boundary_m1_k128_n128",
                symbols={"m": 1, "k": 128, "n": 128, "scale_block": SCALE_BLOCK},
                seed=23,
                tags=frozenset({"boundary"}),
                timeout_s=600,
            ),
            CaseSpec(
                case_id="representative_decode_fused_wqa_wkv",
                symbols={"m": 16, "k": 7168, "n": 2048, "scale_block": SCALE_BLOCK},
                seed=29,
                tags=frozenset({"representative", "legacy", "decode"}),
                timeout_s=1800,
            ),
        )

    def legacy_mappings(self):
        return LEGACY_ADAPTER_MAPPINGS

    def make_inputs(self, case, context):
        torch, *_ = _runtime_modules()
        m, k, n = (int(case.symbols[name]) for name in ("m", "k", "n"))
        if k % SCALE_BLOCK or n % SCALE_BLOCK:
            raise ValueError("K and N must be divisible by 128")
        device = next(iter(context.cuda), "cuda:0")
        generator = context.cuda.get(device)
        if generator is None:
            raise RuntimeError("deepseek_v4_fp8_gemm_nt requires a CUDA generator")
        activation_bf16 = torch.randn(
            (m, k), device=device, dtype=torch.bfloat16, generator=generator
        ).clamp_(-2.0, 2.0)
        weight_bf16 = torch.randn(
            (n, k), device=device, dtype=torch.bfloat16, generator=generator
        ).clamp_(-2.0, 2.0)
        activation, activation_scale, weight, weight_scale = (
            _quantize_production_inputs(
                activation_bf16,
                weight_bf16,
            )
        )
        output = torch.empty((m, n), device=device, dtype=torch.bfloat16)
        return InputBundle(
            args=(
                activation,
                activation_scale,
                weight,
                weight_scale,
                output,
            ),
            observed_state={
                "full_precision_activation": activation_bf16,
                "full_precision_weight": weight_bf16,
            },
        )

    def clone_inputs(self, inputs):
        full_activation = inputs.observed_state["full_precision_activation"]
        full_weight = inputs.observed_state["full_precision_weight"]
        activation, activation_scale, weight, weight_scale = (
            _quantize_production_inputs(
                full_activation.clone(),
                full_weight.clone(),
            )
        )
        output = inputs.args[-1]
        return InputBundle(
            args=(
                activation,
                activation_scale,
                weight,
                weight_scale,
                output.clone(),
            ),
            observed_state={
                "full_precision_activation": full_activation.clone(),
                "full_precision_weight": full_weight.clone(),
            },
        )

    def normalize_output(self, output):
        return output

    def comparator(self, case):
        del case
        raise RuntimeError("DeepSeek V4 FP8 GEMM requires its named oracle gate")

    @staticmethod
    def _full_precision_oracle(inputs):
        activation = inputs.observed_state["full_precision_activation"]
        weight = inputs.observed_state["full_precision_weight"]
        output = inputs.args[-1]
        return (activation.float() @ weight.float().t()).to(output.dtype)

    def correctness_oracles(self, case):
        del case
        return {"full_precision_bf16": self._full_precision_oracle}

    def correctness_gates(self, case):
        del case
        return (
            OracleGate(
                gate_id="end_to_end_full_precision",
                oracle_id="full_precision_bf16",
                comparator=CalcDiffComparator(
                    max_diff=UPSTREAM_CONTRACT["threshold"],
                    source=(
                        f'{UPSTREAM_CONTRACT["project"]}@'
                        f'{UPSTREAM_CONTRACT["revision"]}:'
                        f'{UPSTREAM_CONTRACT["test"]}'
                    ),
                    output_paths=("output",),
                ),
            ),
        )

    def cost_model(self, case):
        m, k, n = (int(case.symbols[name]) for name in ("m", "k", "n"))
        activation_bytes = m * k
        weight_bytes = n * k
        scale_groups = k // SCALE_BLOCK
        aligned_m = (m + 3) // 4 * 4
        aligned_scale_groups = (scale_groups + 3) // 4 * 4
        activation_scale_bytes = aligned_m * aligned_scale_groups
        weight_scale_bytes = n * aligned_scale_groups
        output_bytes = m * n * 2
        return {
            "flops": 2 * m * k * n,
            "estimated_bytes": (
                activation_bytes
                + weight_bytes
                + activation_scale_bytes
                + weight_scale_bytes
                + output_bytes
            ),
            "throughput_units": m * n,
        }


SPEC = DeepSeekV4Fp8GemmNtSpec()


def cost_model(case):
    return SPEC.cost_model(case)
