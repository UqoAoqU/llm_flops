"""Unified prefill/decode sparse FlashMLA contract for GLM-5."""

import importlib

from benchmark_engine.correctness import FloatingComparator, InputBundle, Tolerance, clone_input_bundle
from benchmark_engine.correctness.models import OutputBundle, OutputLeaf
from benchmark_engine.models import CaseSpec


HEADS = 64
KV_HEADS = 1
QK_DIM = 576
VALUE_DIM = 512
TOPK = 2048
SOFTMAX_SCALE = QK_DIM ** -0.5
TOLERANCE = Tolerance(rtol=0.02, atol=0.2, source="glm5_flashmla_unified_sparse_control")


def _sample_indices(torch, total, device, limit=256):
    count = min(int(total), int(limit))
    if count == 0:
        return torch.empty(0, device=device, dtype=torch.long)
    if count == 1:
        return torch.zeros(1, device=device, dtype=torch.long)
    positions = torch.arange(count, device=device, dtype=torch.long)
    span, intervals = int(total) - 1, count - 1
    return positions * (span // intervals) + positions * (span % intervals) // intervals


def _case(phase, model_input, tags):
    return CaseSpec(
        f"{phase}__dsa_{phase}_attn__m{model_input}__ctx65536",
        {
            "query_tokens": model_input,
            "context": 65536,
            "topk": TOPK,
            "heads": HEADS,
            "kv_heads": KV_HEADS,
            "qk_dim": QK_DIM,
            "value_dim": VALUE_DIM,
            "phase": phase,
            "quant_profile": "glm5_fp8",
            "model_input": model_input,
            "raw_context": 65536,
            "projection_adapter_id": f"dsa_{phase}_attn",
        },
        419,
        frozenset(tags),
        1800,
    )


def _projection_cases():
    cases = []
    for phase, values in (("prefill", (1024, 2048, 4096)),
                          ("decode", (1, 4, 8, 16, 32, 64))):
        for model_input in values:
            tags = {"model_projection", f"glm5_{phase}", phase}
            if (phase, model_input) == ("prefill", 1024):
                tags.add("glm5_regression")
            cases.append(_case(phase, model_input, tags))
    return tuple(cases)


class Glm5DsaUnifiedSparseAttentionSpec:
    operator_id = "glm5_dsa_unified_sparse_attention"

    def cases(self):
        return (
            CaseSpec(
                "smoke_unified_q2_kv64_topk64",
                {
                    "query_tokens": 2,
                    "context": 64,
                    "topk": 64,
                    "heads": HEADS,
                    "kv_heads": KV_HEADS,
                    "qk_dim": QK_DIM,
                    "value_dim": VALUE_DIM,
                },
                419,
                frozenset({"smoke", "glm5_smoke", "boundary", "unified"}),
                600,
            ),
        ) + _projection_cases()

    def legacy_mappings(self):
        return {
            "prefill_script": "bench_glm5_prefill.py",
            "decode_script": "bench_glm5_decode.py",
            "backend": "sgl_kernel.flash_mla.flash_mla_sparse_fwd",
            "timer": "cuda_graph",
            "graph_inner_iterations": 20,
            "warmup_replays": 5,
        }

    @staticmethod
    def _validate(symbols):
        query_tokens, context, topk, heads, kv_heads, qk_dim, value_dim = (
            int(symbols[name]) for name in
            ("query_tokens", "context", "topk", "heads", "kv_heads", "qk_dim", "value_dim")
        )
        if query_tokens <= 0 or context <= 0 or topk <= 0 or topk > context:
            raise ValueError("query/context/topk geometry is invalid")
        if (heads, kv_heads, qk_dim, value_dim) != (HEADS, KV_HEADS, QK_DIM, VALUE_DIM):
            raise NotImplementedError("unsupported GLM-5 unified sparse attention layout")
        if topk % 64:
            raise NotImplementedError("FlashMLA sparse topk must be divisible by 64")
        return query_tokens, context, topk

    def make_inputs(self, case, context):
        query_tokens, kv_tokens, topk = self._validate(case.symbols)
        torch = importlib.import_module("torch")
        device = next(iter(context.cuda), "cuda:0")
        generator = context.cuda.get(device)
        if generator is None:
            raise RuntimeError("glm5_dsa_unified_sparse_attention requires a CUDA generator")
        query = torch.randn((query_tokens, HEADS, QK_DIM), device=device,
                            dtype=torch.bfloat16, generator=generator)
        cache = torch.randn((kv_tokens, KV_HEADS, QK_DIM), device=device,
                            dtype=torch.bfloat16, generator=generator)
        indices = torch.stack(tuple(
            torch.randperm(kv_tokens, device=device, generator=generator)[:topk]
            for _ in range(query_tokens * KV_HEADS)
        )).reshape(query_tokens, KV_HEADS, topk).to(torch.int32)
        return InputBundle(
            args=(query, cache, indices, SOFTMAX_SCALE, VALUE_DIM),
            observed_state={"cache_head": cache[:1], "cache_tail": cache[-1:],
                            "indices_head": indices[:1]},
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
        query_tokens, context, topk = self._validate(case.symbols)
        flops = 2 * HEADS * query_tokens * topk * (QK_DIM + VALUE_DIM)
        estimated_bytes = 2 * (
            HEADS * query_tokens * QK_DIM
            + min(topk * query_tokens, context) * QK_DIM
            + HEADS * query_tokens * VALUE_DIM
        )
        return {"flops": flops, "estimated_bytes": estimated_bytes,
                "throughput_units": query_tokens * topk}


SPEC = Glm5DsaUnifiedSparseAttentionSpec()


def cost_model(case):
    return SPEC.cost_model(case)
