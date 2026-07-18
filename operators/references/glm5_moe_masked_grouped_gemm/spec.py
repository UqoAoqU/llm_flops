"""Unified-script masked GLM-5 grouped FP8 GEMM contract."""

import importlib
import random

from benchmark_engine.correctness import FloatingComparator, InputBundle, Tolerance, clone_input_bundle
from benchmark_engine.correctness.models import OutputBundle, OutputLeaf
from benchmark_engine.models import CaseSpec


HIDDEN = 6144
INTERMEDIATE = 2048
EXPERTS = 8
TOPK = 8
ALIGNMENT = 128
TOLERANCE = Tolerance(rtol=0.02, atol=0.2, source="glm5_deepgemm_masked_control")
PROJECTIONS = {
    "moe_gate_proj": (HIDDEN, INTERMEDIATE),
    "moe_up_proj": (HIDDEN, INTERMEDIATE),
    "moe_down_proj": (INTERMEDIATE, HIDDEN),
}
PROJECTION_SEED_OFFSETS = {"moe_gate_proj": 1, "moe_up_proj": 2, "moe_down_proj": 3}
PHASE_SEED_OFFSETS = {"prefill": 1_000_000, "decode": 2_000_000}


def _sample_indices(torch, total, device, limit=256):
    count = min(int(total), int(limit))
    if count == 0:
        return torch.empty(0, device=device, dtype=torch.long)
    if count == 1:
        return torch.zeros(1, device=device, dtype=torch.long)
    positions = torch.arange(count, device=device, dtype=torch.long)
    span, intervals = int(total) - 1, count - 1
    return positions * (span // intervals) + positions * (span % intervals) // intervals


def _distribution_seed(phase, model_input, projection):
    return (433_000_000 + PHASE_SEED_OFFSETS[phase] + int(model_input) * 10
            + PROJECTION_SEED_OFFSETS[projection])


def _distribution_counts(seed, rows):
    generator = random.Random(int(seed))
    counts = [0] * EXPERTS
    for _ in range(int(rows)):
        counts[generator.randint(0, EXPERTS - 1)] += 1
    return tuple(counts)


def _legacy_expected_m(rows):
    average_ceil = (int(rows) + EXPERTS - 1) // EXPERTS
    return ((average_ceil + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT


def _safe_expected_m(counts):
    return ((max(counts) + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT


def _case(case_id, projection, total_rows, tags, distribution_seed,
          phase=None, model_input=None):
    k, n = PROJECTIONS[projection]
    counts = _distribution_counts(distribution_seed, total_rows)
    symbols = {
        "projection": projection,
        "layout": "masked",
        "experts": EXPERTS,
        "total_rows": total_rows,
        "k": k,
        "n": n,
        "distribution": "fixed_python_random_per_token",
        "distribution_seed": distribution_seed,
        "expert_counts": counts,
        "legacy_expected_m": _legacy_expected_m(total_rows),
        "safe_expected_m": _safe_expected_m(counts),
    }
    if phase:
        symbols.update({
            "phase": phase,
            "quant_profile": "glm5_fp8",
            "model_input": model_input,
            "raw_context": 65536,
            "projection_adapter_id": projection,
        })
    return CaseSpec(case_id, symbols, distribution_seed, frozenset(tags), 1800)


def _projection_cases():
    cases = []
    for phase, values in (("prefill", (1024, 2048, 4096)),
                          ("decode", (1, 4, 8, 16, 32, 64))):
        for model_input in values:
            for projection in PROJECTIONS:
                tags = {"model_projection", f"glm5_{phase}", phase}
                if (phase, model_input, projection) == ("prefill", 1024, "moe_gate_proj"):
                    tags.add("glm5_regression")
                cases.append(_case(
                    f"{phase}__{projection}__m{model_input}__ctx65536",
                    projection,
                    model_input * TOPK,
                    tags,
                    _distribution_seed(phase, model_input, projection),
                    phase,
                    model_input,
                ))
    return tuple(cases)


class Glm5MoeMaskedGroupedGemmSpec:
    operator_id = "glm5_moe_masked_grouped_gemm"

    def cases(self):
        return (
            _case("smoke_masked_gate_logical16", "moe_gate_proj", 16,
                  {"smoke", "glm5_smoke", "masked", "unified"}, 433_000_001),
        ) + _projection_cases()

    def legacy_mappings(self):
        return {
            "prefill_script": "bench_glm5_prefill.py",
            "decode_script": "bench_glm5_decode.py",
            "backend": "deep_gemm.fp8_m_grouped_gemm_nt_masked",
            "timer": "cuda_graph",
            "graph_inner_iterations": 20,
            "warmup_replays": 5,
            "legacy_expected_m": "align128(ceil(total_rows / 8))",
            "safe_expected_m": "align128(max(expert_counts))",
            "migration_safety": "fixed random routing retained; unsafe legacy under-allocation corrected",
        }

    @staticmethod
    def _validate(symbols):
        projection, layout = str(symbols["projection"]), str(symbols["layout"])
        experts, rows, k, n = (int(symbols[name]) for name in ("experts", "total_rows", "k", "n"))
        counts = tuple(int(value) for value in symbols["expert_counts"])
        legacy_expected_m = int(symbols["legacy_expected_m"])
        safe_expected_m = int(symbols["safe_expected_m"])
        if projection not in PROJECTIONS or PROJECTIONS[projection] != (k, n):
            raise ValueError("unknown projection or inconsistent K/N")
        if layout != "masked":
            raise ValueError("unified grouped-GEMM contract requires masked layout")
        if experts != EXPERTS or len(counts) != EXPERTS or any(value < 0 for value in counts):
            raise NotImplementedError("unsupported GLM-5 masked grouped-GEMM geometry")
        if rows <= 0 or sum(counts) != rows:
            raise ValueError("expert_counts must sum to total_rows")
        if legacy_expected_m != _legacy_expected_m(rows):
            raise ValueError("legacy_expected_m does not match the unified-script formula")
        if safe_expected_m != _safe_expected_m(counts) or safe_expected_m < max(counts):
            raise ValueError("safe_expected_m must cover every fixed routing count")
        return projection, rows, k, n, counts, legacy_expected_m, safe_expected_m

    def make_inputs(self, case, context):
        _, _, k, n, counts, _, safe_expected_m = self._validate(case.symbols)
        torch = importlib.import_module("torch")
        device = next(iter(context.cuda), "cuda:0")
        generator = context.cuda.get(device)
        if generator is None:
            raise RuntimeError("glm5_moe_masked_grouped_gemm requires a CUDA generator")
        expected_m = safe_expected_m
        n_ceil = (n + 127) // 128 * 128
        activation = torch.randn((EXPERTS, expected_m, k), device=device,
                                 dtype=torch.bfloat16, generator=generator).clamp_(-2, 2).to(torch.float8_e4m3fn)
        activation_scale = torch.ones((EXPERTS, expected_m, k // 128),
                                      device=device, dtype=torch.float32)
        weight = torch.randn((EXPERTS, n, k), device=device, dtype=torch.bfloat16,
                             generator=generator).clamp_(-2, 2).to(torch.float8_e4m3fn)
        weight_scale = torch.ones((EXPERTS, n_ceil // 128, k // 128),
                                  device=device, dtype=torch.float32)
        output = torch.zeros((EXPERTS, expected_m, n), device=device, dtype=torch.bfloat16)
        routing = torch.tensor(counts, device=device, dtype=torch.int32)
        return InputBundle(
            args=(activation, activation_scale, weight, weight_scale, output, routing, expected_m),
            observed_state={"routing": routing, "weight_head": weight[:1, :1, :16]},
        )

    def clone_inputs(self, inputs):
        return clone_input_bundle(inputs)

    def normalize_output(self, output):
        torch = importlib.import_module("torch")
        flat = output.detach().reshape(-1)
        indices = _sample_indices(torch, flat.numel(), flat.device)
        return OutputBundle((OutputLeaf(
            "output", tuple(flat[indices].float().cpu().tolist()), str(output.dtype),
            tuple(output.shape), tuple(output.stride()), "strided", str(output.device),
        ),))

    def comparator(self, case):
        self._validate(case.symbols)
        return FloatingComparator(operator_default=TOLERANCE, require_explicit=True)

    def cost_model(self, case):
        _, rows, k, n, _, _, _ = self._validate(case.symbols)
        return {
            "flops": 2 * rows * k * n,
            "estimated_bytes": rows * k + EXPERTS * n * k + 2 * rows * n,
            "throughput_units": rows,
        }


SPEC = Glm5MoeMaskedGroupedGemmSpec()


def cost_model(case):
    return SPEC.cost_model(case)
