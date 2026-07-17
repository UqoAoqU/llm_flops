"""Reference-owned contract and case metadata for DeepSeek V4 FP8 GEMM.

This module stays safe for controller-side registry discovery: heavyweight
CUDA modules are resolved only while runtime inputs are built or cloned.
"""

import importlib

from benchmark_engine.correctness import FloatingComparator, InputBundle, Tolerance
from benchmark_engine.correctness.models import OutputBundle, OutputLeaf
from benchmark_engine.models import CaseSpec


SCALE_BLOCK = 128
FP8_DTYPE = "float8_e4m3fn"
OUTPUT_DTYPE = "bfloat16"
TOLERANCE = Tolerance(rtol=0.01, atol=0.1, source="deepgemm_blackwell_oracle")
PREFILL_M_VALUES = (1024, 2048, 4096)
DECODE_M_VALUES = (16, 32)

# This table is the lossless bridge from every legacy Adapter(kind="fp8") row
# to the new operator contract.  Repeated shapes remain separate because their
# model-projection instance counts and names are semantically significant.
LEGACY_ADAPTER_MAPPINGS = (
    {"phase": "prefill", "adapter_name": "Fused WQ_A + WKV", "backend": "DeepGEMM fp8_gemm_nt", "instances": 61, "m_values": PREFILL_M_VALUES, "k": 7168, "n": 2048},
    {"phase": "prefill", "adapter_name": "Q RMSNorm + WQ_B", "backend": "DeepGEMM fp8_gemm_nt", "instances": 61, "m_values": PREFILL_M_VALUES, "k": 1536, "n": 65536},
    {"phase": "prefill", "adapter_name": "Compressor WKV-Gate GEMM (C4)", "backend": "DeepGEMM fp8_gemm_nt", "instances": 30, "m_values": PREFILL_M_VALUES, "k": 7168, "n": 2048},
    {"phase": "prefill", "adapter_name": "Compressor WKV-Gate GEMM (C128)", "backend": "DeepGEMM fp8_gemm_nt", "instances": 30, "m_values": PREFILL_M_VALUES, "k": 7168, "n": 1024},
    {"phase": "prefill", "adapter_name": "C4 Indexer Q Projection", "backend": "DeepGEMM fp8_gemm_nt", "instances": 30, "m_values": PREFILL_M_VALUES, "k": 1536, "n": 65536},
    {"phase": "prefill", "adapter_name": "WO_B Projection", "backend": "DeepGEMM fp8_gemm_nt", "instances": 61, "m_values": PREFILL_M_VALUES, "k": 16384, "n": 7168},
    {"phase": "prefill", "adapter_name": "LM Head", "backend": "DeepGEMM FP8 full vocab", "instances": 1, "m_values": PREFILL_M_VALUES, "k": 7168, "n": 129280},
    {"phase": "decode", "adapter_name": "Fused WQ_A + WKV", "backend": "DeepGEMM fp8_gemm_nt", "instances": 61, "m_values": DECODE_M_VALUES, "k": 7168, "n": 2048},
    {"phase": "decode", "adapter_name": "Q RMSNorm + WQ_B", "backend": "DeepGEMM fp8_gemm_nt", "instances": 61, "m_values": DECODE_M_VALUES, "k": 1536, "n": 65536},
    {"phase": "decode", "adapter_name": "C4 Indexer Q Projection", "backend": "DeepGEMM fp8_gemm_nt", "instances": 30, "m_values": DECODE_M_VALUES, "k": 1536, "n": 65536},
    {"phase": "decode", "adapter_name": "WO_B Projection", "backend": "DeepGEMM fp8_gemm_nt", "instances": 61, "m_values": DECODE_M_VALUES, "k": 16384, "n": 7168},
    {"phase": "decode", "adapter_name": "LM Head", "backend": "DeepGEMM FP8 full vocab", "instances": 1, "m_values": DECODE_M_VALUES, "k": 7168, "n": 129280},
)

_PROJECTION_GEMMS = {
    "prefill": (("fused_wq_a_wkv", 7168, 2048), ("q_rmsnorm_wq_b", 1536, 65536),
                ("compressor_wkv_gate_c4", 7168, 2048), ("compressor_wkv_gate_c128", 7168, 1024),
                ("c4_indexer_q_projection", 1536, 65536), ("wo_b_projection", 16384, 7168),
                ("lm_head", 7168, 129280)),
    "decode": (("fused_wq_a_wkv", 7168, 2048), ("q_rmsnorm_wq_b", 1536, 65536),
               ("c4_indexer_q_projection", 1536, 65536), ("wo_b_projection", 16384, 7168),
               ("lm_head", 7168, 129280)),
}


def _projection_cases():
    values = []
    for phase, inputs in (("prefill", PREFILL_M_VALUES), ("decode", DECODE_M_VALUES)):
        for adapter_id, k, n in _PROJECTION_GEMMS[phase]:
            for m in inputs:
                values.append(CaseSpec(
                    f"{phase}__{adapter_id}__m{m}__ctx65536__fp8_mxfp8",
                    {"m": m, "k": k, "n": n, "scale_block": SCALE_BLOCK, "phase": phase,
                     "quant_profile": "fp8_mxfp8", "raw_context": 65536, "model_input": m,
                     "projection_adapter_id": adapter_id}, 211,
                    frozenset({"model_projection", f"deepseek_v4_{phase}", f"phase_{phase}",
                               "quant_profile_fp8_mxfp8", f"m_{m}", "context_65536",
                               "performance_only"}), 1800))
    return tuple(values)


def _runtime_modules():
    torch = importlib.import_module("torch")
    layout = importlib.import_module("deep_gemm.utils.layout")
    return torch, layout.get_mn_major_tma_aligned_tensor


def _power_of_two_scales(torch, shape, *, device, generator):
    """Generate seeded UE8M0-compatible positive FP32 scale values."""

    exponents = torch.randint(
        -1,
        2,
        shape,
        device=device,
        dtype=torch.int32,
        generator=generator,
    )
    if exponents.numel() >= 2:
        flattened = exponents.view(-1)
        flattened[0] = -1
        flattened[1] = 1
    return torch.exp2(exponents.to(torch.float32))


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
                case_id="smoke_m16_k256_n256",
                symbols={"m": 16, "k": 256, "n": 256, "scale_block": SCALE_BLOCK},
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
        ) + _projection_cases()

    def legacy_mappings(self):
        return LEGACY_ADAPTER_MAPPINGS

    def make_inputs(self, case, context):
        torch, align_scale = _runtime_modules()
        m, k, n = (int(case.symbols[name]) for name in ("m", "k", "n"))
        if k % SCALE_BLOCK or n % SCALE_BLOCK:
            raise ValueError("K and N must be divisible by 128")
        device = next(iter(context.cuda), "cuda:0")
        generator = context.cuda.get(device)
        if generator is None:
            raise RuntimeError("deepseek_v4_fp8_gemm_nt requires a CUDA generator")
        activation = torch.randn(
            (m, k), device=device, dtype=torch.bfloat16, generator=generator
        ).clamp_(-2.0, 2.0).to(torch.float8_e4m3fn)
        weight = torch.randn(
            (n, k), device=device, dtype=torch.bfloat16, generator=generator
        ).clamp_(-2.0, 2.0).to(torch.float8_e4m3fn)
        activation_scale = _power_of_two_scales(
            torch,
            (m, k // SCALE_BLOCK),
            device=device,
            generator=generator,
        )
        weight_scale = _power_of_two_scales(
            torch,
            (n // SCALE_BLOCK, k // SCALE_BLOCK),
            device=device,
            generator=generator,
        )
        combined_scales = torch.cat(
            (activation_scale.reshape(-1), weight_scale.reshape(-1))
        )
        if torch.unique(combined_scales).numel() < 2:
            activation_scale.reshape(-1)[0] = 0.5
            weight_scale.reshape(-1)[0] = 2.0
        activation_scale_aligned = align_scale(activation_scale.clone())
        output = torch.empty((m, n), device=device, dtype=torch.bfloat16)
        return InputBundle(
            args=(
                activation,
                activation_scale,
                activation_scale_aligned,
                weight,
                weight_scale,
                output,
            )
        )

    def clone_inputs(self, inputs):
        _, align_scale = _runtime_modules()
        activation, activation_scale, _, weight, weight_scale, output = inputs.args
        activation_clone = activation.clone()
        logical_scale_clone = activation_scale.clone()
        aligned_scale_clone = align_scale(logical_scale_clone.clone())
        return InputBundle(
            args=(
                activation_clone,
                logical_scale_clone,
                aligned_scale_clone,
                weight.clone(),
                weight_scale.clone(),
                output.clone(),
            )
        )

    def normalize_output(self, output):
        torch = importlib.import_module("torch")
        flat = output.detach().reshape(-1)
        count = min(flat.numel(), 256)
        if count <= 1:
            indices = torch.zeros(count, device=flat.device, dtype=torch.long)
        else:
            positions = torch.arange(count, device=flat.device, dtype=torch.long)
            span, intervals = flat.numel() - 1, count - 1
            indices = positions * (span // intervals) + positions * (span % intervals) // intervals
        values = tuple(flat[indices].float().cpu().tolist())
        return OutputBundle((OutputLeaf("output", values, str(output.dtype), tuple(output.shape),
                                       tuple(output.stride()), "strided", str(output.device)),))

    def comparator(self, case):
        del case
        return FloatingComparator(operator_default=TOLERANCE, require_explicit=True)

    def cost_model(self, case):
        m, k, n = (int(case.symbols[name]) for name in ("m", "k", "n"))
        activation_bytes = m * k
        weight_bytes = n * k
        activation_scale_bytes = m * (k // SCALE_BLOCK) * 4
        weight_scale_bytes = (n // SCALE_BLOCK) * (k // SCALE_BLOCK) * 4
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
