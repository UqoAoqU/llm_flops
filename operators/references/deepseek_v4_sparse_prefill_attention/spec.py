"""Sparse-prefill contract with an independent selected-attention oracle."""

import importlib

from benchmark_engine.correctness import InputBundle
from benchmark_engine.correctness.models import ComparisonResult, OutputBundle, OutputLeaf
from benchmark_engine.models import CaseSpec


HEADS = 128
HEAD_DIM = 512
PAGE_SIZE = 64
TOLERANCE = 0.08
LEGACY_MAPPING = {"adapter_name": "Sparse Prefill Attention", "backend": "sgl-kernel FlashMLA sparse_fwd", "instances": 60, "raw_context": 65536, "topk": 1024}


def _torch():
    return importlib.import_module("torch")


def _sparse_oracle(torch, q, kv, indices, softmax_scale, value_dim):
    """Dense PyTorch definition of selected prefill attention."""

    if indices.shape[-1] == 0:
        return torch.zeros((*q.shape[:-1], value_dim), dtype=torch.float32, device=q.device)
    selected = kv[indices.long().squeeze(1), 0, :]
    scores = torch.einsum("qhd,qkd->qhk", q.float(), selected.float()) * softmax_scale
    probabilities = torch.softmax(scores, dim=-1)
    return torch.einsum("qhk,qkd->qhd", probabilities, selected[..., :value_dim].float())


def _sample_indices(torch, total, device, limit=256):
    total = int(total)
    count = min(total, int(limit))
    if count == 0:
        return torch.empty(0, device=device, dtype=torch.long)
    if count == 1:
        return torch.zeros(1, device=device, dtype=torch.long)
    positions = torch.arange(count, device=device, dtype=torch.long)
    span = total - 1
    intervals = count - 1
    return positions * (span // intervals) + positions * (span % intervals) // intervals


def _bounded_oracle(torch, output, *, representative):
    del representative
    flat = output.reshape(-1)
    indices = _sample_indices(torch, flat.numel(), flat.device)
    return flat[indices], indices


def _bounded_output(output, limit=256):
    torch = _torch()
    flat = output.detach().reshape(-1)
    indices = _sample_indices(torch, flat.numel(), flat.device, limit)
    values = tuple(flat[indices].float().cpu().tolist())
    return OutputBundle((OutputLeaf("output", values, str(output.dtype), tuple(output.shape), tuple(output.stride()), "strided", str(output.device)),))


def _check_available_memory(required_bytes, available_bytes):
    if required_bytes > int(available_bytes * 0.9):
        raise MemoryError(f"estimated allocation {required_bytes} exceeds available CUDA memory {available_bytes}")


class SparseAttentionComparator:
    def compare(self, reference, candidate, **_):
        failures, maxima = [], {}
        for role, bundle in (("reference", reference), ("candidate", candidate)):
            leaves = bundle.by_path()
            output, oracle, indices = leaves.get("output"), leaves.get("state.semantic_oracle"), leaves.get("state.oracle_indices")
            if output is None or oracle is None or indices is None:
                failures.append({"role": role, "error": "missing oracle leaves"})
                continue
            sampled = output.value
            maximum = max((abs(float(a) - float(b)) for a, b in zip(sampled, oracle.value)), default=0.0)
            maxima[role] = maximum
            if len(sampled) != len(oracle.value) or maximum > TOLERANCE:
                failures.append({"role": role, "max_abs_error": maximum})
        left, right = reference.by_path(), candidate.by_path()
        left_output, right_output = left.get("output"), right.get("output")
        if left_output is not None and right_output is not None:
            cross = max((abs(float(a) - float(b)) for a, b in zip(left_output.value, right_output.value)), default=0.0)
            maxima["reference_candidate"] = cross
            if len(left_output.value) != len(right_output.value) or cross > TOLERANCE:
                failures.append({"path": "output", "reference_candidate_max_abs_error": cross})
        for path in sorted(set(left) | set(right)):
            a, b = left.get(path), right.get(path)
            if a is None or b is None or a.contract() != b.contract():
                failures.append({"path": path, "error": "contract mismatch"})
            elif path.startswith("state.") and path not in {"state.semantic_oracle", "state.oracle_indices"} and a.value != b.value:
                failures.append({"path": path, "error": "observed state mutated"})
        return ComparisonResult(not failures, "sparse_attention_oracle_and_state", {"max_abs_error": maxima}, tuple(failures[:8]), None if not failures else "output")


class DeepSeekV4SparsePrefillAttentionSpec:
    operator_id = "deepseek_v4_sparse_prefill_attention"

    def cases(self):
        return (
            CaseSpec("smoke_tail_c4", {"query_tokens": 2, "raw_context": 257, "compression_ratio": 4, "topk": 64, "page_size": 64, "heads": 128, "head_dim": 512}, 109, frozenset({"smoke", "boundary", "prefill", "tail_page"}), 600),
            CaseSpec("boundary_short_index_c128", {"query_tokens": 1, "raw_context": 257, "compression_ratio": 128, "topk": 1, "page_size": 64, "heads": 128, "head_dim": 512}, 113, frozenset({"boundary", "prefill", "short_index", "unsupported"}), 600),
            CaseSpec("boundary_empty_index", {"query_tokens": 1, "raw_context": 1, "compression_ratio": 4, "topk": 0, "page_size": 64, "heads": 128, "head_dim": 512}, 127, frozenset({"boundary", "minimum", "prefill", "empty_index", "unsupported"}), 600),
            CaseSpec("representative_prefill_q1024_context65536", {"query_tokens": 1024, "raw_context": 65536, "compression_ratio": 1, "topk": 1024, "page_size": 64, "heads": 128, "head_dim": 512}, 131, frozenset({"representative", "legacy", "prefill", "performance_only"}), 1800),
        )

    @staticmethod
    def _validate(symbols):
        q, raw, ratio, topk, page_size, heads, head_dim = (int(symbols[n]) for n in ("query_tokens", "raw_context", "compression_ratio", "topk", "page_size", "heads", "head_dim"))
        if q <= 0 or raw <= 0 or topk < 0:
            raise ValueError("query_tokens/raw_context must be positive and topk non-negative")
        if ratio not in {1, 4, 128}:
            raise NotImplementedError("compression_ratio must be 1, C4, or C128")
        if page_size != PAGE_SIZE or heads != HEADS or head_dim != HEAD_DIM:
            raise NotImplementedError("unsupported page/head layout")
        context = (raw + ratio - 1) // ratio
        if topk > context:
            raise ValueError("topk cannot exceed compressed context")
        return q, raw, context, ratio, topk

    def layout_contract(self, case):
        q, raw, context, ratio, topk = self._validate(case.symbols)
        return {"q": [q, HEADS, HEAD_DIM], "kv": [context, 1, HEAD_DIM], "indices": [q, 1, topk], "block_table": None, "page_size": PAGE_SIZE, "raw_context": raw, "compressed_context": context, "compression_ratio": ratio, "tail_tokens": context % PAGE_SIZE, "cache_mutation": "forbidden", "observed_state": ["cache_head", "cache_tail", "indices_head"], "workspace_lifecycle": "backend-managed; no exposed persistent workspace"}

    def make_inputs(self, case, context):
        q_tokens, _, kv_context, _, topk = self._validate(case.symbols)
        if topk == 0 or topk % 64:
            raise NotImplementedError("FlashMLA sparse_fwd requires a non-empty topk divisible by 64")
        torch = _torch()
        device = next(iter(context.cuda), "cuda:0")
        generator = context.cuda.get(device)
        if generator is None:
            raise RuntimeError("sparse prefill attention requires a CUDA generator")
        required = (q_tokens * HEADS * HEAD_DIM + kv_context * HEAD_DIM) * 2 + q_tokens * topk * 4
        free, _ = torch.cuda.mem_get_info(device)
        _check_available_memory(required, free)
        representative = "representative" in case.tags
        if representative:
            # Preserve the exact legacy shape without materializing an enormous
            # full PyTorch golden during performance preparation.
            q = torch.zeros((q_tokens, HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device)
            kv = torch.zeros((kv_context, 1, HEAD_DIM), dtype=torch.bfloat16, device=device)
        else:
            q = torch.randn((q_tokens, HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device, generator=generator)
            kv = torch.randn((kv_context, 1, HEAD_DIM), dtype=torch.bfloat16, device=device, generator=generator)
        indices = torch.randint(0, kv_context, (q_tokens, 1, topk), dtype=torch.int32, device=device, generator=generator)
        scale = HEAD_DIM**-0.5
        if representative:
            output_elements = q_tokens * HEADS * HEAD_DIM
            count = min(output_elements, 256)
            oracle_indices = _sample_indices(torch, output_elements, device, count)
            oracle = torch.zeros((count,), device=device, dtype=torch.float32)
        else:
            oracle_full = _sparse_oracle(torch, q, kv, indices, scale, HEAD_DIM)
            oracle, oracle_indices = _bounded_oracle(torch, oracle_full, representative=False)
        observed = {"semantic_oracle": oracle, "oracle_indices": oracle_indices, "cache_head": kv[:1], "cache_tail": kv[-1:], "indices_head": indices[:1]}
        return InputBundle(args=(q, kv, indices, scale, HEAD_DIM), observed_state=observed)

    def clone_inputs(self, inputs):
        q, kv, indices, scale, value_dim = inputs.args
        q2, kv2, indices2 = q.clone(), kv.clone(), indices.clone()
        observed = {"semantic_oracle": inputs.observed_state["semantic_oracle"].clone(), "oracle_indices": inputs.observed_state["oracle_indices"].clone(), "cache_head": kv2[:1], "cache_tail": kv2[-1:], "indices_head": indices2[:1]}
        return InputBundle(args=(q2, kv2, indices2, scale, value_dim), observed_state=observed)

    def normalize_output(self, output):
        return _bounded_output(output)

    def comparator(self, case):
        del case
        return SparseAttentionComparator()

    def cost_model(self, case):
        q, _, context, _, topk = self._validate(case.symbols)
        if topk == 0 or topk % 64:
            return None
        selected = min(topk, context)
        return {"flops": 4 * q * HEADS * selected * HEAD_DIM, "estimated_bytes": 2 * (q * HEADS * HEAD_DIM + min(q * selected, context) * HEAD_DIM + q * HEADS * HEAD_DIM) + q * selected * 4, "throughput_units": q * selected}

    def workspace_bytes(self, case):
        self._validate(case.symbols)
        return None

    def legacy_mappings(self):
        return LEGACY_MAPPING


SPEC = DeepSeekV4SparsePrefillAttentionSpec()


def cost_model(case):
    return SPEC.cost_model(case)
