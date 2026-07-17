"""Dual-cache sparse-decode contract; heavy modules load only in workers."""

import importlib

from benchmark_engine.correctness import InputBundle
from benchmark_engine.correctness.models import ComparisonResult, OutputBundle, OutputLeaf
from benchmark_engine.models import CaseSpec


HEADS = 128
HEAD_DIM = 512
SWA_TOKENS = 128
CACHE_BYTES_PER_TOKEN = 584
TOLERANCE = 0.1
LEGACY_MAPPINGS = (
    {"adapter_name": "Sparse Decode Attention C4", "backend": "sgl-kernel FlashMLA dual-cache C4", "instances": 30, "compression_ratio": 4},
    {"adapter_name": "Sparse Decode Attention C128", "backend": "sgl-kernel FlashMLA dual-cache C128", "instances": 30, "compression_ratio": 128},
)


def _projection_cases():
    values = []
    for batch in (16, 32):
        for adapter_id, ratio, page_size in (("sparse_decode_attention_c4", 4, 64),
                                                ("sparse_decode_attention_c128", 128, 2)):
            values.append(CaseSpec(
                f"decode__{adapter_id}__m{batch}__ctx65536__fp8_mxfp8",
                {"batch": batch, "raw_context": 65536, "compression_ratio": ratio,
                 "page_size": page_size, "heads": 128, "head_dim": 512, "phase": "decode",
                 "quant_profile": "fp8_mxfp8", "model_input": batch,
                 "projection_adapter_id": adapter_id}, 239,
                frozenset({"model_projection", "deepseek_v4_decode", "phase_decode",
                           "quant_profile_fp8_mxfp8", f"m_{batch}", "context_65536",
                           "c4" if ratio == 4 else "c128", "performance_only"}), 1800))
    return tuple(values)


def _runtime():
    torch = importlib.import_module("torch")
    flash = importlib.import_module("sgl_kernel.flash_mla")
    return torch, flash.FlashMLASchedMeta


def _check_available_memory(required_bytes, available_bytes):
    if required_bytes > int(available_bytes * 0.9):
        raise MemoryError(f"estimated allocation {required_bytes} exceeds available CUDA memory {available_bytes}")


def _padded_cache(torch, pages, page_size, device):
    row_bytes = ((page_size * CACHE_BYTES_PER_TOKEN + 575) // 576) * 576
    raw = torch.zeros((pages, row_bytes), device=device, dtype=torch.uint8)
    return raw[:, : page_size * CACHE_BYTES_PER_TOKEN].view(pages, page_size, 1, CACHE_BYTES_PER_TOKEN)


def _clone_padded_cache(torch, cache):
    clone = _padded_cache(torch, cache.shape[0], cache.shape[1], cache.device)
    clone.copy_(cache)
    return clone


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


def _bounded_zero_oracle(torch, q, *, representative):
    del representative
    total = q.shape[0] * q.shape[1] * q.shape[2] * HEAD_DIM
    indices = _sample_indices(torch, total, q.device)
    return torch.zeros((indices.numel(),), device=q.device, dtype=torch.float32), indices


def _selected_attention_oracle(torch, q, cache, indices, lengths):
    """Independent BF16 semantic helper used by CPU contract tests."""

    outputs = []
    for batch in range(q.shape[0]):
        valid = int(lengths[batch])
        if valid == 0:
            outputs.append(torch.zeros((q.shape[2], q.shape[3]), dtype=torch.float32, device=q.device))
            continue
        selected = cache[indices[batch, 0, :valid].long()]
        scores = torch.einsum("hd,kd->hk", q[batch, 0].float(), selected.float()) * (q.shape[-1] ** -0.5)
        outputs.append(torch.einsum("hk,kd->hd", torch.softmax(scores, dim=-1), selected.float()))
    return torch.stack(outputs).unsqueeze(1)


def _bounded_output(output, limit=256):
    torch = importlib.import_module("torch")
    flat = output.detach().reshape(-1)
    indices = _sample_indices(torch, flat.numel(), flat.device, limit)
    values = tuple(flat[indices].float().cpu().tolist())
    return OutputBundle((OutputLeaf("output", values, str(output.dtype), tuple(output.shape), tuple(output.stride()), "strided", str(output.device)),))


class DecodeComparator:
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
                failures.append({"path": path, "error": "observed cache/index state mutated"})
        return ComparisonResult(not failures, "decode_zero_oracle_and_state", {"max_abs_error": maxima}, tuple(failures[:8]), None if not failures else "output")


class DeepSeekV4SparseDecodeAttentionSpec:
    operator_id = "deepseek_v4_sparse_decode_attention"

    def cases(self):
        return (
            CaseSpec("smoke_c4_tail", {"batch": 2, "raw_context": 257, "compression_ratio": 4, "page_size": 64, "heads": 128, "head_dim": 512}, 137, frozenset({"smoke", "boundary", "decode", "tail_page", "c4"}), 600),
            CaseSpec("boundary_c128_non_integral", {"batch": 2, "raw_context": 257, "compression_ratio": 128, "page_size": 2, "heads": 128, "head_dim": 512}, 139, frozenset({"boundary", "decode", "tail_page", "c128"}), 600),
            CaseSpec("boundary_min_context", {"batch": 2, "raw_context": 1, "compression_ratio": 4, "page_size": 64, "heads": 128, "head_dim": 512}, 149, frozenset({"boundary", "decode", "minimum"}), 600),
            CaseSpec("representative_decode_c4_b16_context65536", {"batch": 16, "raw_context": 65536, "compression_ratio": 4, "page_size": 64, "heads": 128, "head_dim": 512}, 151, frozenset({"representative", "legacy", "decode", "c4"}), 1800),
            CaseSpec("representative_decode_c128_b16_context65536", {"batch": 16, "raw_context": 65536, "compression_ratio": 128, "page_size": 2, "heads": 128, "head_dim": 512}, 157, frozenset({"representative", "legacy", "decode", "c128"}), 1800),
        ) + _projection_cases()

    @staticmethod
    def _validate(symbols):
        batch, raw, ratio, page_size, heads, head_dim = (int(symbols[n]) for n in ("batch", "raw_context", "compression_ratio", "page_size", "heads", "head_dim"))
        if batch <= 0 or raw <= 0:
            raise ValueError("batch and raw_context must be positive")
        expected_page = 64 if ratio == 4 else 2 if ratio == 128 else None
        if expected_page is None:
            raise NotImplementedError("dual-cache decode supports C4 and C128 only")
        if page_size != expected_page or heads != HEADS or head_dim != HEAD_DIM:
            raise NotImplementedError("unsupported page/head layout")
        compressed = (raw + ratio - 1) // ratio
        valid = min(1024, compressed) if ratio == 4 else compressed
        padded = ((valid + 63) // 64) * 64
        pages = (padded + page_size - 1) // page_size
        return batch, raw, ratio, compressed, valid, padded, pages

    def layout_contract(self, case):
        batch, raw, ratio, compressed, valid, padded, pages = self._validate(case.symbols)
        return {"q": [batch, 1, HEADS, HEAD_DIM], "swa_cache": [batch, SWA_TOKENS, 1, CACHE_BYTES_PER_TOKEN], "extra_cache": [batch * pages, int(case.symbols["page_size"]), 1, CACHE_BYTES_PER_TOKEN], "block_table": None, "physical_indices": [batch, 1, padded], "raw_context": raw, "compressed_context": compressed, "compression_ratio": ratio, "valid_extra_tokens": valid, "padding_tokens": padded - valid, "tail_tokens": compressed % int(case.symbols["page_size"]), "cache_mutation": "forbidden", "observed_state": ["swa_cache_head", "extra_cache_head", "extra_cache_tail", "extra_lengths"], "workspace_lifecycle": "FlashMLASchedMeta allocated once per isolated clone"}

    def make_inputs(self, case, context):
        torch, Scheduler = _runtime()
        batch, raw, _, _, valid, padded, pages = self._validate(case.symbols)
        device = next(iter(context.cuda), "cuda:0")
        if context.cuda.get(device) is None:
            raise RuntimeError("sparse decode attention requires a CUDA generator")
        page_size = int(case.symbols["page_size"])
        required = (batch * SWA_TOKENS + batch * pages * page_size) * CACHE_BYTES_PER_TOKEN + batch * HEADS * HEAD_DIM * 2
        free, _ = torch.cuda.mem_get_info(device)
        _check_available_memory(required, free)
        q = torch.zeros((batch, 1, HEADS, HEAD_DIM), device=device, dtype=torch.bfloat16)
        swa_cache = _padded_cache(torch, batch, SWA_TOKENS, device)
        swa_indices = torch.arange(batch, device=device, dtype=torch.int32).view(batch, 1, 1) * SWA_TOKENS + torch.arange(SWA_TOKENS, device=device, dtype=torch.int32).view(1, 1, SWA_TOKENS)
        swa_lengths = torch.full((batch,), min(raw, SWA_TOKENS), device=device, dtype=torch.int32)
        extra_cache = _padded_cache(torch, batch * pages, page_size, device)
        extra_indices = torch.arange(batch, device=device, dtype=torch.int32).view(batch, 1, 1) * pages * page_size + torch.arange(padded, device=device, dtype=torch.int32).view(1, 1, padded)
        extra_lengths = torch.full((batch,), valid, device=device, dtype=torch.int32)
        sink = torch.zeros((HEADS,), device=device, dtype=torch.float32)
        oracle, oracle_indices = _bounded_zero_oracle(torch, q, representative="representative" in case.tags)
        observed = {"semantic_oracle": oracle, "oracle_indices": oracle_indices, "swa_cache_head": swa_cache[:1, :1], "extra_cache_head": extra_cache[:1, :1], "extra_cache_tail": extra_cache[-1:, -1:], "extra_lengths": extra_lengths}
        return InputBundle(args=(q, swa_cache, swa_indices, swa_lengths, extra_cache, extra_indices, extra_lengths, sink, Scheduler()), observed_state=observed)

    def clone_inputs(self, inputs):
        torch, Scheduler = _runtime()
        q, swa_cache, swa_indices, swa_lengths, extra_cache, extra_indices, extra_lengths, sink, _ = inputs.args
        q2, si2, sl2 = q.clone(), swa_indices.clone(), swa_lengths.clone()
        swa2 = _clone_padded_cache(torch, swa_cache)
        extra2 = _clone_padded_cache(torch, extra_cache)
        ei2, el2, sink2 = extra_indices.clone(), extra_lengths.clone(), sink.clone()
        observed = {"semantic_oracle": inputs.observed_state["semantic_oracle"].clone(), "oracle_indices": inputs.observed_state["oracle_indices"].clone(), "swa_cache_head": swa2[:1, :1], "extra_cache_head": extra2[:1, :1], "extra_cache_tail": extra2[-1:, -1:], "extra_lengths": el2}
        return InputBundle(args=(q2, swa2, si2, sl2, extra2, ei2, el2, sink2, Scheduler()), observed_state=observed)

    def normalize_output(self, output):
        return _bounded_output(output)

    def comparator(self, case):
        del case
        return DecodeComparator()

    def cost_model(self, case):
        batch, raw, _, _, valid, _, _ = self._validate(case.symbols)
        attended = min(raw, SWA_TOKENS) + valid
        return {"flops": 4 * batch * HEADS * attended * HEAD_DIM, "estimated_bytes": batch * attended * CACHE_BYTES_PER_TOKEN + batch * HEADS * HEAD_DIM * 4, "throughput_units": batch * attended}

    def workspace_bytes(self, case):
        self._validate(case.symbols)
        return None

    def legacy_mappings(self):
        return LEGACY_MAPPINGS


SPEC = DeepSeekV4SparseDecodeAttentionSpec()


def cost_model(case):
    return SPEC.cost_model(case)
