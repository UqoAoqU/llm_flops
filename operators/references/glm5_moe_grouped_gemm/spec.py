"""Legacy contiguous GLM-5 grouped FP8 GEMM contract."""

import importlib
import random

from benchmark_engine.correctness import FloatingComparator, InputBundle, Tolerance, clone_input_bundle
from benchmark_engine.correctness.models import OutputBundle, OutputLeaf
from benchmark_engine.models import CaseSpec


HIDDEN = 6144
INTERMEDIATE = 2048
EXPERTS = 8
LEGACY_TOTAL_TOKENS = 128
ALIGNMENT = 128
LEGACY_DISTRIBUTION_SEEDS = (43101, 43102, 43103, 43104, 43105)
TOLERANCE = Tolerance(rtol=0.02, atol=0.2, source="glm5_deepgemm_grouped_control")
PROJECTIONS = {
    "moe_gate_proj": (HIDDEN, INTERMEDIATE),
    "moe_up_proj": (HIDDEN, INTERMEDIATE),
    "moe_down_proj": (INTERMEDIATE, HIDDEN),
}


def _sample_indices(torch, total, device, limit=256):
    count = min(int(total), int(limit))
    if count == 0:
        return torch.empty(0, device=device, dtype=torch.long)
    if count == 1:
        return torch.zeros(1, device=device, dtype=torch.long)
    positions = torch.arange(count, device=device, dtype=torch.long)
    span, intervals = int(total) - 1, count - 1
    return positions * (span // intervals) + positions * (span % intervals) // intervals


def _distribution_counts(seed, total_tokens=LEGACY_TOTAL_TOKENS):
    """Deterministic equivalent of the legacy per-token randint assignment."""

    generator = random.Random(int(seed))
    counts = [0] * EXPERTS
    for _ in range(int(total_tokens)):
        counts[generator.randint(0, EXPERTS - 1)] += 1
    return tuple(counts)


def _aligned_counts(counts):
    return tuple(((count + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT if count else 0
                 for count in counts)


def _case(case_id, projection, counts, tags, *, dist_idx=None, distribution_seed=None):
    counts = tuple(int(value) for value in counts)
    k, n = PROJECTIONS[projection]
    symbols = {
        "projection": projection,
        "layout": "contiguous",
        "experts": EXPERTS,
        "total_rows": sum(counts),
        "k": k,
        "n": n,
        "distribution": "python_random_per_token" if dist_idx is not None else "bounded_explicit",
        "expert_counts": counts,
    }
    if dist_idx is not None:
        symbols.update({"dist_idx": int(dist_idx), "distribution_seed": int(distribution_seed)})
    return CaseSpec(case_id, symbols, int(distribution_seed or 431), frozenset(tags), 1800)


def _legacy_cases():
    cases = []
    for projection in PROJECTIONS:
        for dist_idx, seed in enumerate(LEGACY_DISTRIBUTION_SEEDS):
            tags = {"legacy_full", "legacy_moe_deepgemm"}
            if projection == "moe_gate_proj" and dist_idx == 0:
                tags.add("glm5_regression")
            cases.append(_case(
                f"legacy__{projection}__dist{dist_idx}__logical128",
                projection,
                _distribution_counts(seed),
                tags,
                dist_idx=dist_idx,
                distribution_seed=seed,
            ))
    return tuple(cases)


class Glm5MoeGroupedGemmSpec:
    operator_id = "glm5_moe_grouped_gemm"

    def cases(self):
        return (
            _case("smoke_gate_logical16", "moe_gate_proj", (16, 0, 0, 0, 0, 0, 0, 0),
                  {"smoke", "glm5_smoke", "legacy_moe_deepgemm"}),
        ) + _legacy_cases()

    def legacy_mappings(self):
        return {
            "script": "moe_deepgemm.py",
            "csv": "glm5_moe_deepgemm_perf.csv",
            "projections": tuple(PROJECTIONS),
            "experts": EXPERTS,
            "logical_token_assignments": LEGACY_TOTAL_TOKENS,
            "distributions": len(LEGACY_DISTRIBUTION_SEEDS),
            "distribution_algorithm": "random.Random(seed).randint(0, 7) once per logical token",
            "distribution_seeds": LEGACY_DISTRIBUTION_SEEDS,
            "legacy_seed_status": "legacy Python random generator was unseeded",
            "physical_alignment_rows": ALIGNMENT,
            "timer": "cuda_event",
            "warmup": 5,
            "runs": 20,
        }

    @staticmethod
    def _validate(symbols):
        projection, layout = str(symbols["projection"]), str(symbols["layout"])
        experts, rows, k, n = (int(symbols[name]) for name in ("experts", "total_rows", "k", "n"))
        if projection not in PROJECTIONS or PROJECTIONS[projection] != (k, n):
            raise ValueError("unknown projection or inconsistent K/N")
        if layout != "contiguous":
            raise ValueError("legacy grouped-GEMM contract requires contiguous layout")
        counts = tuple(int(value) for value in symbols["expert_counts"])
        if experts != EXPERTS or len(counts) != EXPERTS or any(value < 0 for value in counts):
            raise NotImplementedError("unsupported GLM-5 grouped-GEMM geometry")
        if rows <= 0 or sum(counts) != rows:
            raise ValueError("expert_counts must sum to total_rows")
        return projection, rows, k, n, counts

    def make_inputs(self, case, context):
        _, _, k, n, counts = self._validate(case.symbols)
        torch = importlib.import_module("torch")
        align = importlib.import_module("deep_gemm.utils.layout").get_mn_major_tma_aligned_tensor
        device = next(iter(context.cuda), "cuda:0")
        generator = context.cuda.get(device)
        if generator is None:
            raise RuntimeError("glm5_moe_grouped_gemm requires a CUDA generator")
        aligned = _aligned_counts(counts)
        physical_rows = sum(aligned)
        n_ceil = (n + 127) // 128 * 128
        weight = torch.randn((EXPERTS, n, k), device=device, dtype=torch.bfloat16,
                             generator=generator).clamp_(-2, 2).to(torch.float8_e4m3fn)
        weight_scale = torch.ones((EXPERTS, n_ceil // 128, k // 128),
                                  device=device, dtype=torch.float32)
        activation = torch.randn((physical_rows, k), device=device, dtype=torch.bfloat16,
                                 generator=generator).clamp_(-2, 2).to(torch.float8_e4m3fn)
        activation_scale = align(torch.ones((physical_rows, k // 128),
                                            device=device, dtype=torch.float32))
        routing = torch.cat(tuple(
            torch.full((value,), expert, device=device, dtype=torch.int32)
            for expert, value in enumerate(aligned) if value
        ))
        output = torch.zeros((physical_rows, n), device=device, dtype=torch.bfloat16)
        return InputBundle(
            args=(activation, activation_scale, weight, weight_scale, output, routing),
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
        _, _, k, n, counts = self._validate(case.symbols)
        physical_rows = sum(_aligned_counts(counts))
        return {
            "flops": 2 * physical_rows * k * n,
            "estimated_bytes": physical_rows * k + EXPERTS * n * k + 2 * physical_rows * n,
            "throughput_units": physical_rows,
        }


SPEC = Glm5MoeGroupedGemmSpec()


def cost_model(case):
    return SPEC.cost_model(case)
