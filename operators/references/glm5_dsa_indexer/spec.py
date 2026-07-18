"""Reference-owned GLM-5 DSA indexer GEMM contract."""

import importlib

from benchmark_engine.correctness import FloatingComparator, InputBundle, Tolerance, clone_input_bundle
from benchmark_engine.correctness.models import OutputBundle, OutputLeaf
from benchmark_engine.models import CaseSpec


HIDDEN = 6144
Q_LORA = 2048
INDEX_HEADS = 32
INDEX_DIM = 128
TOLERANCE = Tolerance(rtol=0.02, atol=0.2, source="glm5_cublas_fp8_control")


def _sample_indices(torch, total, device, limit=256):
    count = min(int(total), int(limit))
    if count == 0:
        return torch.empty(0, device=device, dtype=torch.long)
    if count == 1:
        return torch.zeros(1, device=device, dtype=torch.long)
    positions = torch.arange(count, device=device, dtype=torch.long)
    span, intervals = int(total) - 1, count - 1
    return positions * (span // intervals) + positions * (span % intervals) // intervals
LEGACY_M_VALUES = (16, 256, 512, 1024)
LEGACY_S_VALUES = (65536, 131072, 262144)


def _shape(kind, model_input, context):
    if kind == "index_k_proj":
        return context, HIDDEN, INDEX_DIM
    if kind == "index_q_upproj":
        return model_input, Q_LORA, INDEX_HEADS * INDEX_DIM
    if kind == "index_weights_proj":
        return model_input, HIDDEN, INDEX_HEADS
    if kind == "index_score":
        return INDEX_HEADS * model_input, INDEX_DIM, context
    raise ValueError(f"unknown indexer kind {kind!r}")


def _case(case_id, kind, m, s, tags, backend="cublas_fp8", phase=None):
    gm, k, n = _shape(kind, m, s)
    return CaseSpec(
        case_id,
        {"kind": kind, "backend": backend, "m": m, "context": s, "gemm_m": gm, "k": k, "n": n,
         **({"phase": phase, "quant_profile": "glm5_fp8", "model_input": m, "raw_context": s,
             "projection_adapter_id": kind} if phase else {})},
        401,
        frozenset(tags),
        1800,
    )


def _legacy_cases():
    cases = []
    for m in LEGACY_M_VALUES:
        for s in LEGACY_S_VALUES:
            for kind in ("index_k_proj", "index_q_upproj", "index_weights_proj", "index_score"):
                cases.append(_case(
                    f"legacy__{kind}__m{m}__s{s}", kind, m, s,
                    {"legacy_full", "legacy_dsa_indexer", kind},
                ))
    return tuple(cases)


class Glm5DsaIndexerSpec:
    operator_id = "glm5_dsa_indexer"

    def cases(self):
        cases = [
            _case("smoke_index_q_upproj_m16", "index_q_upproj", 16, 65536,
                  {"smoke", "glm5_smoke", "decode"}),
            _case("regression_index_q_upproj_m1024", "index_q_upproj", 1024, 65536,
                  {"representative", "glm5_regression", "glm5_prefill", "prefill"},
                  "deepgemm_fp8", "prefill"),
        ]
        for phase, values in (("prefill", (1024, 2048, 4096)), ("decode", (1, 4, 8, 16, 32, 64))):
            for m in values:
                for kind in ("index_k_proj", "index_q_upproj", "index_weights_proj"):
                    backend = "deepgemm_bf16" if kind == "index_weights_proj" else "deepgemm_fp8"
                    cases.append(_case(f"{phase}__{kind}__m{m}__ctx65536", kind, m, 65536,
                                       {"model_projection", f"glm5_{phase}", phase}, backend, phase))
        return tuple(cases) + _legacy_cases()

    def legacy_mappings(self):
        return {
            "script": "dsa_indexer.py",
            "csv": "glm5_dsa_indexer_perf.csv",
            "m_values": LEGACY_M_VALUES,
            "context_values": LEGACY_S_VALUES,
            "operators": ("index_k_proj", "index_q_upproj", "index_weights_proj", "index_score"),
            "timer": "cuda_graph",
            "warmup": 5,
            "runs": 20,
        }

    def make_inputs(self, case, context):
        torch = importlib.import_module("torch")
        backend = str(case.symbols["backend"])
        gm, k, n = (int(case.symbols[name]) for name in ("gemm_m", "k", "n"))
        if min(gm, k, n) <= 0:
            raise ValueError("GEMM dimensions must be positive")
        device = next(iter(context.cuda), "cuda:0")
        generator = context.cuda.get(device)
        if generator is None:
            raise RuntimeError("glm5_dsa_indexer requires a CUDA generator")
        activation = torch.randn((gm, k), device=device, dtype=torch.bfloat16, generator=generator)
        weight = torch.randn((n, k), device=device, dtype=torch.bfloat16, generator=generator)
        output_dtype = torch.float32 if backend == "deepgemm_bf16" else torch.bfloat16
        output = torch.empty((gm, n), device=device, dtype=output_dtype)
        if backend == "cublas_fp8":
            maximum = torch.finfo(torch.float8_e4m3fn).max
            a_scale = (activation.abs().float().amax() / maximum).clamp(min=1e-12)
            w_scale = (weight.abs().float().amax() / maximum).clamp(min=1e-12)
            activation = (activation.float() / a_scale).to(torch.float8_e4m3fn)
            weight = (weight.float() / w_scale).to(torch.float8_e4m3fn)
            a_scale, w_scale = a_scale.reshape(1), w_scale.reshape(1)
        elif backend == "deepgemm_fp8":
            align = importlib.import_module("deep_gemm.utils.layout").get_mn_major_tma_aligned_tensor
            activation = activation.clamp_(-2, 2).to(torch.float8_e4m3fn)
            weight = weight.clamp_(-2, 2).to(torch.float8_e4m3fn)
            a_scale = align(torch.ones((gm, k // 128), device=device, dtype=torch.float32))
            w_scale = torch.ones(((n + 127) // 128, k // 128), device=device, dtype=torch.float32)
        elif backend == "deepgemm_bf16":
            a_scale = w_scale = None
        else:
            raise ValueError(f"unknown indexer backend {backend!r}")
        return InputBundle(args=(backend, activation, a_scale, weight, w_scale, output))

    def clone_inputs(self, inputs):
        return clone_input_bundle(inputs)

    def normalize_output(self, output):
        torch = importlib.import_module("torch")
        flat = output.detach().reshape(-1)
        indices = _sample_indices(torch, flat.numel(), flat.device)
        return OutputBundle((OutputLeaf("output", tuple(flat[indices].float().cpu().tolist()), str(output.dtype),
                                              tuple(output.shape), tuple(output.stride()), "strided", str(output.device)),))

    def comparator(self, case):
        _shape(str(case.symbols["kind"]), int(case.symbols["m"]), int(case.symbols["context"]))
        return FloatingComparator(operator_default=TOLERANCE, require_explicit=True)

    def cost_model(self, case):
        backend = str(case.symbols["backend"])
        gm, k, n = (int(case.symbols[name]) for name in ("gemm_m", "k", "n"))
        if backend in {"cublas_fp8", "deepgemm_fp8"}:
            estimated_bytes = gm * k + n * k + 2 * gm * n
        elif backend == "deepgemm_bf16":
            estimated_bytes = 2 * gm * k + 2 * n * k + 4 * gm * n
        else:
            raise ValueError(f"unknown indexer backend {backend!r}")
        return {"flops": 2 * gm * k * n, "estimated_bytes": estimated_bytes,
                "throughput_units": gm * n}


SPEC = Glm5DsaIndexerSpec()


def cost_model(case):
    return SPEC.cost_model(case)
