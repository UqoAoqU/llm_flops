"""Legacy ``dsa_flashmla.py`` sparse FlashMLA contract."""

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
LEGACY_Q_VALUES = (16384, 32768, 65536, 131072)
LEGACY_KV_VALUES = (16384, 32768, 65536, 131072)
TOLERANCE = Tolerance(rtol=0.02, atol=0.2, source="glm5_flashmla_sparse_control")


def _sample_indices(torch, total, device, limit=256):
    count = min(int(total), int(limit))
    if count == 0:
        return torch.empty(0, device=device, dtype=torch.long)
    if count == 1:
        return torch.zeros(1, device=device, dtype=torch.long)
    positions = torch.arange(count, device=device, dtype=torch.long)
    span, intervals = int(total) - 1, count - 1
    return positions * (span // intervals) + positions * (span % intervals) // intervals


def _case(case_id, query_tokens, context, topk, tags):
    return CaseSpec(
        case_id,
        {
            "query_tokens": query_tokens,
            "context": context,
            "topk": min(topk, context),
            "heads": HEADS,
            "kv_heads": KV_HEADS,
            "qk_dim": QK_DIM,
            "value_dim": VALUE_DIM,
        },
        419,
        frozenset(tags),
        1800,
    )


def _legacy_cases():
    return tuple(
        _case(f"legacy__q{query_tokens}__kv{context}__topk2048",
              query_tokens, context, TOPK,
              {"legacy_full", "legacy_dsa_flashmla", "prefill"})
        for context in LEGACY_KV_VALUES for query_tokens in LEGACY_Q_VALUES
    )


class Glm5DsaSparseAttentionSpec:
    operator_id = "glm5_dsa_sparse_attention"

    def cases(self):
        return (
            _case("smoke_q2_kv64_topk64", 2, 64, 64,
                  {"smoke", "glm5_smoke", "boundary", "prefill"}),
            _case("regression_q1024_kv65536_topk2048", 1024, 65536, TOPK,
                  {"glm5_regression", "representative", "prefill"}),
        ) + _legacy_cases()

    def legacy_mappings(self):
        return {
            "script": "dsa_flashmla.py",
            "csv": "glm5_sparse_prefill_perf.csv",
            "query_values": LEGACY_Q_VALUES,
            "context_values": LEGACY_KV_VALUES,
            "topk": TOPK,
            "timer": "cuda_graph",
            "graph_inner_iterations": 1,
            "warmup_replays": 5,
            "samples": 20,
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
            raise NotImplementedError("unsupported GLM-5 sparse attention layout")
        if topk % 64:
            raise NotImplementedError("FlashMLA sparse topk must be divisible by 64")
        return query_tokens, context, topk

    def make_inputs(self, case, context):
        query_tokens, kv_tokens, topk = self._validate(case.symbols)
        torch = importlib.import_module("torch")
        device = next(iter(context.cuda), "cuda:0")
        generator = context.cuda.get(device)
        if generator is None:
            raise RuntimeError("glm5_dsa_sparse_attention requires a CUDA generator")
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


SPEC = Glm5DsaSparseAttentionSpec()


def cost_model(case):
    return SPEC.cost_model(case)
