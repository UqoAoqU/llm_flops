"""Dense paged FlashMLA GLM-5 contract."""

import importlib

from benchmark_engine.correctness import FloatingComparator, InputBundle, Tolerance, clone_input_bundle
from benchmark_engine.correctness.models import OutputBundle, OutputLeaf
from benchmark_engine.models import CaseSpec


HEADS = 64
KV_HEADS = 1
QK_DIM = 576
VALUE_DIM = 512
PAGE_SIZE = 64
SOFTMAX_SCALE = QK_DIM ** -0.5
LEGACY_Q_VALUES = (16384, 32768, 65536, 131072)
LEGACY_KV_VALUES = (16384, 32768, 65536, 131072)
TOLERANCE = Tolerance(rtol=0.02, atol=0.2, source="glm5_flashmla_dense_control")


def _sample_indices(torch, total, device, limit=256):
    count = min(int(total), int(limit))
    if count == 0:
        return torch.empty(0, device=device, dtype=torch.long)
    if count == 1:
        return torch.zeros(1, device=device, dtype=torch.long)
    positions = torch.arange(count, device=device, dtype=torch.long)
    span, intervals = int(total) - 1, count - 1
    return positions * (span // intervals) + positions * (span % intervals) // intervals


def _case(case_id, q, kv, tags):
    return CaseSpec(case_id, {"query_tokens": q, "context": kv, "heads": HEADS,
                              "kv_heads": KV_HEADS, "qk_dim": QK_DIM,
                              "value_dim": VALUE_DIM, "page_size": PAGE_SIZE},
                    421, frozenset(set(tags) | {"requires_sm90a"}), 1800)


def _legacy_cases():
    return tuple(
        _case(f"legacy__q{q}__kv{kv}", q, kv,
              {"legacy_full", "legacy_mla_flashmla", "prefill"})
        for kv in LEGACY_KV_VALUES for q in LEGACY_Q_VALUES
    )


class Glm5DensePrefillAttentionSpec:
    operator_id = "glm5_dense_prefill_attention"

    def cases(self):
        return (
            _case("smoke_q2_kv64", 2, 64, {"smoke", "glm5_smoke", "boundary", "prefill"}),
            _case("regression_q128_kv1024", 128, 1024,
                  {"representative", "glm5_regression", "prefill"}),
        ) + _legacy_cases()

    def legacy_mappings(self):
        return {"script": "mla_flashmla.py", "csv": "glm5_dense_prefill_perf.csv",
                "query_values": LEGACY_Q_VALUES, "context_values": LEGACY_KV_VALUES,
                "timer": "cuda_graph", "warmup": 5, "runs": 20}

    @staticmethod
    def _validate(symbols):
        q, kv, heads, kv_heads, qk_dim, value_dim, page = (
            int(symbols[name]) for name in
            ("query_tokens", "context", "heads", "kv_heads", "qk_dim", "value_dim", "page_size")
        )
        if q <= 0 or kv <= 0:
            raise ValueError("query_tokens and context must be positive")
        if (heads, kv_heads, qk_dim, value_dim, page) != (HEADS, KV_HEADS, QK_DIM, VALUE_DIM, PAGE_SIZE):
            raise NotImplementedError("unsupported GLM-5 dense attention layout")
        return q, kv

    def make_inputs(self, case, context):
        q, kv = self._validate(case.symbols)
        torch = importlib.import_module("torch")
        flash = importlib.import_module("sgl_kernel.flash_mla")
        device = next(iter(context.cuda), "cuda:0")
        generator = context.cuda.get(device)
        if generator is None:
            raise RuntimeError("glm5_dense_prefill_attention requires a CUDA generator")
        capability = tuple(torch.cuda.get_device_capability(device))
        if capability != (9, 0):
            raise NotImplementedError(
                f"unsupported: installed dense FlashMLA backend requires SM90a; got capability {capability}"
            )
        blocks = (kv + PAGE_SIZE - 1) // PAGE_SIZE
        query = torch.randn((1, q, HEADS, QK_DIM), device=device, dtype=torch.bfloat16, generator=generator)
        cache = torch.randn((blocks, PAGE_SIZE, KV_HEADS, QK_DIM), device=device,
                            dtype=torch.bfloat16, generator=generator)
        block_table = torch.arange(blocks, device=device, dtype=torch.int32).reshape(1, blocks)
        lengths = torch.tensor([kv], device=device, dtype=torch.int32)
        scheduler, splits = flash.get_mla_metadata(lengths, q * HEADS // KV_HEADS, KV_HEADS)
        return InputBundle(args=(query, cache, block_table, lengths, VALUE_DIM, scheduler, splits, SOFTMAX_SCALE),
                           observed_state={"cache_head": cache[:1], "cache_tail": cache[-1:],
                                           "block_table": block_table, "lengths": lengths})

    def clone_inputs(self, inputs):
        return clone_input_bundle(inputs)

    def normalize_output(self, output):
        torch = importlib.import_module("torch")
        flat = output.detach().reshape(-1)
        indices = _sample_indices(torch, flat.numel(), flat.device)
        return OutputBundle((OutputLeaf("output", tuple(flat[indices].float().cpu().tolist()), str(output.dtype),
                                              tuple(output.shape), tuple(output.stride()), "strided", str(output.device)),))

    def comparator(self, case):
        self._validate(case.symbols)
        return FloatingComparator(operator_default=TOLERANCE, require_explicit=True)

    def cost_model(self, case):
        q, kv = self._validate(case.symbols)
        return {"flops": 2 * HEADS * q * kv * (QK_DIM + VALUE_DIM),
                "estimated_bytes": 2 * (HEADS * q * QK_DIM + kv * QK_DIM + HEADS * q * VALUE_DIM),
                "throughput_units": q * kv}

    def workspace_bytes(self, case):
        self._validate(case.symbols)
        return None


SPEC = Glm5DensePrefillAttentionSpec()


def cost_model(case):
    return SPEC.cost_model(case)
