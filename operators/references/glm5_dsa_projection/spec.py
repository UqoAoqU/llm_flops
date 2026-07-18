"""GLM-5 attention projection shape, input and cost contracts."""

import importlib

from benchmark_engine.correctness import FloatingComparator, InputBundle, Tolerance, clone_input_bundle
from benchmark_engine.correctness.models import OutputBundle, OutputLeaf
from benchmark_engine.models import CaseSpec


HIDDEN = 6144
Q_LORA = 2048
KV_LORA = 512
QK_NOPE = 192
QK_ROPE = 64
QK_HEAD = 256
V_HEAD = 256
HEADS = 64
LEGACY_M_VALUES = (1024, 4096, 16384, 65536)
TOLERANCE = Tolerance(rtol=0.02, atol=0.2, source="glm5_projection_fp8_control")


def _sample_indices(torch, total, device, limit=256):
    count = min(int(total), int(limit))
    if count == 0:
        return torch.empty(0, device=device, dtype=torch.long)
    if count == 1:
        return torch.zeros(1, device=device, dtype=torch.long)
    positions = torch.arange(count, device=device, dtype=torch.long)
    span, intervals = int(total) - 1, count - 1
    return positions * (span // intervals) + positions * (span % intervals) // intervals

PROJECTIONS = {
    "q_a_proj": ("gemm", 1, HIDDEN, Q_LORA),
    "q_b_proj": ("gemm", 1, Q_LORA, HEADS * QK_HEAD),
    "absorbed_W_UK": ("bmm", HEADS, QK_NOPE, KV_LORA),
    "kv_a_proj": ("gemm", 1, HIDDEN, KV_LORA + QK_ROPE),
    "absorbed_W_UV": ("bmm", HEADS, KV_LORA, V_HEAD),
    "o_proj": ("gemm", 1, HEADS * V_HEAD, HIDDEN),
    # sglang's unified GLM-5 benchmark fuses q_a_proj and kv_a_proj.
    "fused_qkv_a_proj": ("gemm", 1, HIDDEN, Q_LORA + KV_LORA + QK_ROPE),
}


def _case(case_id, name, m, tags, phase=None, unified=False):
    kind, batch, k, n = PROJECTIONS[name]
    if unified and kind == "bmm":
        kind = "sgl_bmm"
    symbols = {"projection": name, "kind": kind, "batch": batch, "m": m, "k": k, "n": n}
    if phase:
        symbols.update({"phase": phase, "quant_profile": "glm5_fp8", "model_input": m,
                        "raw_context": 65536, "projection_adapter_id": name})
    return CaseSpec(case_id, symbols, 409, frozenset(tags), 1800)


def _legacy_cases():
    return tuple(
        _case(f"legacy__{name}__m{m}", name, m, {"legacy_full", "legacy_dsa_projection", name})
        for m in LEGACY_M_VALUES
        for name in ("q_a_proj", "q_b_proj", "absorbed_W_UK", "kv_a_proj", "absorbed_W_UV", "o_proj")
    )


def _projection_cases():
    cases = []
    for phase, values in (("prefill", (1024, 2048, 4096)), ("decode", (1, 4, 8, 16, 32, 64))):
        for m in values:
            for name in ("fused_qkv_a_proj", "q_b_proj", "absorbed_W_UK", "absorbed_W_UV", "o_proj"):
                tags = {"model_projection", f"glm5_{phase}", phase}
                if (phase, m, name) == ("prefill", 1024, "fused_qkv_a_proj"):
                    tags.add("glm5_regression")
                cases.append(_case(f"{phase}__{name}__m{m}__ctx65536", name, m, tags, phase, True))
    return tuple(cases)


class Glm5DsaProjectionSpec:
    operator_id = "glm5_dsa_projection"

    def cases(self):
        return (
            _case("smoke_q_a_proj_m16", "q_a_proj", 16, {"smoke", "glm5_smoke", "decode"}),
        ) + _legacy_cases() + _projection_cases()

    def legacy_mappings(self):
        return {"script": "dsa_projection.py", "csv": "glm5_attention_gemm_perf.csv",
                "m_values": LEGACY_M_VALUES, "operators": tuple(PROJECTIONS)[:-1],
                "timer": "cuda_graph", "warmup": 5, "runs": 20}

    def make_inputs(self, case, context):
        torch = importlib.import_module("torch")
        kind, batch, m, k, n = (case.symbols[name] for name in ("kind", "batch", "m", "k", "n"))
        batch, m, k, n = map(int, (batch, m, k, n))
        device = next(iter(context.cuda), "cuda:0")
        generator = context.cuda.get(device)
        if generator is None:
            raise RuntimeError("glm5_dsa_projection requires a CUDA generator")
        if kind == "gemm":
            layout = importlib.import_module("deep_gemm.utils.layout")
            activation = torch.randn((m, k), device=device, dtype=torch.bfloat16, generator=generator).clamp_(-2, 2).to(torch.float8_e4m3fn)
            weight = torch.randn((n, k), device=device, dtype=torch.bfloat16, generator=generator).clamp_(-2, 2).to(torch.float8_e4m3fn)
            activation_scale = layout.get_mn_major_tma_aligned_tensor(torch.ones((m, k // 128), device=device, dtype=torch.float32))
            weight_scale = torch.ones(((n + 127) // 128, k // 128), device=device, dtype=torch.float32)
            output = torch.empty((m, n), device=device, dtype=torch.bfloat16)
        elif kind in {"bmm", "sgl_bmm"}:
            activation = torch.randn((batch, m, k), device=device, dtype=torch.bfloat16, generator=generator).clamp_(-2, 2).to(torch.float8_e4m3fn)
            weight_shape = (batch, k, n) if kind == "sgl_bmm" else (batch, n, k)
            weight = torch.randn(weight_shape, device=device, dtype=torch.bfloat16, generator=generator).clamp_(-2, 2).to(torch.float8_e4m3fn)
            scale_shape = (1,) if kind == "sgl_bmm" else (batch, 1)
            activation_scale = torch.ones(scale_shape, device=device, dtype=torch.float32)
            weight_scale = torch.ones(scale_shape, device=device, dtype=torch.float32)
            output = torch.empty((batch, m, n), device=device, dtype=torch.bfloat16)
        else:
            raise ValueError(f"unknown projection kind {kind!r}")
        return InputBundle(args=(kind, activation, activation_scale, weight, weight_scale, output))

    def clone_inputs(self, inputs):
        return clone_input_bundle(inputs)

    def normalize_output(self, output):
        torch = importlib.import_module("torch")
        flat = output.detach().reshape(-1)
        indices = _sample_indices(torch, flat.numel(), flat.device)
        return OutputBundle((OutputLeaf("output", tuple(flat[indices].float().cpu().tolist()), str(output.dtype),
                                              tuple(output.shape), tuple(output.stride()), "strided", str(output.device)),))

    def comparator(self, case):
        if str(case.symbols["projection"]) not in PROJECTIONS:
            raise ValueError("unknown projection")
        return FloatingComparator(operator_default=TOLERANCE, require_explicit=True)

    def cost_model(self, case):
        batch, m, k, n = (int(case.symbols[name]) for name in ("batch", "m", "k", "n"))
        return {"flops": 2 * batch * m * k * n,
                "estimated_bytes": batch * (m * k + n * k + 2 * m * n),
                "throughput_units": batch * m * n}


SPEC = Glm5DsaProjectionSpec()


def cost_model(case):
    return SPEC.cost_model(case)
