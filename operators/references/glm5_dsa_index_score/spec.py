"""Prefill and paged-decode DeepGEMM index score contract."""

import importlib

from benchmark_engine.correctness import FloatingComparator, InputBundle, Tolerance, clone_input_bundle
from benchmark_engine.correctness.models import OutputBundle, OutputLeaf
from benchmark_engine.models import CaseSpec


HEADS = 32
DIM = 128
PAGE_SIZE = 64
CONTEXT = 65536
TOLERANCE = Tolerance(rtol=0.02, atol=0.2, source="glm5_deepgemm_mqa_control")


def _sample_indices(torch, total, device, limit=256):
    count = min(int(total), int(limit))
    if count == 0:
        return torch.empty(0, device=device, dtype=torch.long)
    if count == 1:
        return torch.zeros(1, device=device, dtype=torch.long)
    positions = torch.arange(count, device=device, dtype=torch.long)
    span, intervals = int(total) - 1, count - 1
    return positions * (span // intervals) + positions * (span % intervals) // intervals


def _case(phase, m, tags):
    return CaseSpec(f"{phase}__index_score__m{m}__ctx65536",
                    {"phase": phase, "m": m, "context": CONTEXT, "heads": HEADS,
                     "head_dim": DIM, "page_size": PAGE_SIZE, "quant_profile": "glm5_fp8", "raw_context": CONTEXT,
                     "model_input": m, "projection_adapter_id": "index_score"},
                    443, frozenset(tags), 1800)


class Glm5DsaIndexScoreSpec:
    operator_id = "glm5_dsa_index_score"

    def cases(self):
        cases = []
        for phase, values in (("prefill", (1024, 2048, 4096)), ("decode", (1, 4, 8, 16, 32, 64))):
            for m in values:
                tags = {"model_projection", f"glm5_{phase}", phase}
                if (phase, m) == ("decode", 16):
                    tags.add("glm5_regression")
                cases.append(_case(phase, m, tags))
        return (
            CaseSpec("smoke_prefill_m2_ctx128",
                     {"phase": "prefill", "m": 2, "context": 128, "heads": HEADS,
                      "head_dim": DIM, "page_size": PAGE_SIZE},
                     443, frozenset({"smoke", "glm5_smoke", "prefill"}), 600),
        ) + tuple(cases)

    def legacy_mappings(self):
        return {"prefill_script": "bench_glm5_prefill.py", "prefill_backend": "deep_gemm.fp8_mqa_logits",
                "decode_script": "bench_glm5_decode.py", "decode_backend": "deep_gemm.fp8_paged_mqa_logits",
                "context": CONTEXT, "heads": HEADS, "head_dim": DIM,
                "timer": "cuda_graph", "warmup": 5, "runs": 20}

    @staticmethod
    def _validate(symbols):
        phase = str(symbols["phase"])
        m, context, heads, dim, page = (int(symbols[name]) for name in
                                        ("m", "context", "heads", "head_dim", "page_size"))
        if phase not in {"prefill", "decode"} or min(m, context) <= 0:
            raise ValueError("invalid index-score phase/shape")
        if (heads, dim, page) != (HEADS, DIM, PAGE_SIZE):
            raise NotImplementedError("unsupported GLM-5 index-score layout")
        return phase, m, context

    def make_inputs(self, case, context):
        phase, m, sequence = self._validate(case.symbols)
        torch = importlib.import_module("torch")
        deep_gemm = importlib.import_module("deep_gemm")
        device = next(iter(context.cuda), "cuda:0")
        generator = context.cuda.get(device)
        if generator is None:
            raise RuntimeError("glm5_dsa_index_score requires a CUDA generator")
        weights = torch.randn((m, HEADS), device=device, dtype=torch.float32, generator=generator)
        if phase == "prefill":
            query = torch.randn((m, HEADS, DIM), device=device, dtype=torch.bfloat16, generator=generator).clamp_(-2, 2).to(torch.float8_e4m3fn)
            cache = torch.randn((sequence, DIM), device=device, dtype=torch.bfloat16, generator=generator).clamp_(-2, 2).to(torch.float8_e4m3fn)
            cache_scale = torch.ones((sequence,), device=device, dtype=torch.float32)
            starts = torch.zeros((m,), device=device, dtype=torch.int32)
            ends = torch.full((m,), sequence, device=device, dtype=torch.int32)
            block_tables = schedule = None
            max_context = sequence
        else:
            # SGLang owns the fused FP8 KV-cache packing helper used by the
            # currently pinned DeepGEMM paged-MQA ABI.
            utils = importlib.import_module("sglang.srt.layers.attention.dsa.utils")
            query = torch.randn((m, HEADS, DIM), device=device, dtype=torch.bfloat16, generator=generator).clamp_(-2, 2).to(torch.float8_e4m3fn)
            blocks_per_sequence = (sequence + PAGE_SIZE - 1) // PAGE_SIZE
            total_blocks = blocks_per_sequence * m
            kv_fp8 = torch.randn((total_blocks, PAGE_SIZE, DIM), device=device,
                                 dtype=torch.bfloat16, generator=generator).clamp_(-2, 2).to(torch.float8_e4m3fn)
            kv_scale = torch.ones((total_blocks, PAGE_SIZE), device=device, dtype=torch.float32)
            cache = utils.fp8_mqa_logits_make_fused_kv(kv_fp8, kv_scale, PAGE_SIZE, DIM)
            cache_scale = None
            starts = torch.full((m, 1), sequence, device=device, dtype=torch.int32)
            ends = None
            block_tables = torch.arange(total_blocks, device=device, dtype=torch.int32).reshape(m, blocks_per_sequence)
            schedule = deep_gemm.get_paged_mqa_logits_metadata(starts, PAGE_SIZE, deep_gemm.get_num_sms())
            max_context = blocks_per_sequence * PAGE_SIZE
        observed = {"cache_head": cache[:1], "cache_tail": cache[-1:], "lengths": starts}
        if block_tables is not None:
            observed["block_tables"] = block_tables
        return InputBundle(args=(phase, query, cache, cache_scale, weights, starts, ends,
                                 block_tables, schedule, max_context), observed_state=observed)

    def clone_inputs(self, inputs):
        return clone_input_bundle(inputs)

    def normalize_output(self, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        torch = importlib.import_module("torch")
        flat = output.detach().reshape(-1)
        indices = _sample_indices(torch, flat.numel(), flat.device)
        return OutputBundle((OutputLeaf("output", tuple(flat[indices].float().cpu().tolist()), str(output.dtype),
                                              tuple(output.shape), tuple(output.stride()), "strided", str(output.device)),))

    def comparator(self, case):
        self._validate(case.symbols)
        return FloatingComparator(operator_default=TOLERANCE, require_explicit=True)

    def cost_model(self, case):
        _, m, context = self._validate(case.symbols)
        return {"flops": 2 * m * HEADS * context * DIM,
                "estimated_bytes": m * HEADS * DIM + context * DIM + 4 * m * HEADS + 4 * m * context,
                "throughput_units": m * context}


SPEC = Glm5DsaIndexScoreSpec()


def cost_model(case):
    return SPEC.cost_model(case)
