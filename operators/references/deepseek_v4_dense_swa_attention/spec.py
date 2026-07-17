"""Dense SWA contract kept separate from sparse dual-cache semantics."""

import importlib

from benchmark_engine.correctness import InputBundle
from benchmark_engine.correctness.models import ComparisonResult, OutputBundle, OutputLeaf
from benchmark_engine.models import CaseSpec


HEADS = 128
HEAD_DIM = 512
WINDOW = 128
CACHE_BYTES_PER_TOKEN = 584
TOLERANCE = 0.1
LEGACY_MAPPINGS = (
    {"phase": "decode", "adapter_name": "Dense SWA Attention", "backend": "sgl-kernel FlashMLA SWA-only", "instances": 1},
)


def _runtime():
    torch = importlib.import_module("torch")
    flash = importlib.import_module("sgl_kernel.flash_mla")
    return torch, flash.FlashMLASchedMeta


def _check_available_memory(required_bytes, available_bytes):
    if required_bytes > int(available_bytes * 0.9):
        raise MemoryError(f"estimated allocation {required_bytes} exceeds available CUDA memory {available_bytes}")


def _padded_cache(torch, batch, device):
    row_bytes = ((WINDOW * CACHE_BYTES_PER_TOKEN + 575) // 576) * 576
    raw = torch.zeros((batch, row_bytes), device=device, dtype=torch.uint8)
    return raw[:, : WINDOW * CACHE_BYTES_PER_TOKEN].view(batch, WINDOW, 1, CACHE_BYTES_PER_TOKEN)


def _clone_padded_cache(torch, cache):
    clone = _padded_cache(torch, cache.shape[0], cache.device)
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


def _bounded_zero_oracle(torch, q, representative):
    del representative
    total = q.numel()
    indices = _sample_indices(torch, total, q.device)
    return torch.zeros((indices.numel(),), device=q.device, dtype=torch.float32), indices


def _bounded_output(output, limit=256):
    torch = importlib.import_module("torch")
    flat = output.detach().reshape(-1)
    indices = _sample_indices(torch, flat.numel(), flat.device, limit)
    values = tuple(flat[indices].float().cpu().tolist())
    return OutputBundle((OutputLeaf("output", values, str(output.dtype), tuple(output.shape), tuple(output.stride()), "strided", str(output.device)),))


class DenseSwaComparator:
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
        return ComparisonResult(not failures, "dense_swa_zero_oracle_and_state", {"max_abs_error": maxima}, tuple(failures[:8]), None if not failures else "output")


class DeepSeekV4DenseSwaAttentionSpec:
    operator_id = "deepseek_v4_dense_swa_attention"

    def cases(self):
        return (
            CaseSpec("smoke_tail_window", {"batch": 2, "raw_context": 65, "window": 128, "page_size": 128, "heads": 128, "head_dim": 512}, 163, frozenset({"smoke", "boundary", "decode", "swa", "tail_page"}), 600),
            CaseSpec("boundary_min_context", {"batch": 2, "raw_context": 1, "window": 128, "page_size": 128, "heads": 128, "head_dim": 512}, 167, frozenset({"boundary", "minimum", "decode", "swa"}), 600),
            CaseSpec("representative_decode_b16_context65536", {"batch": 16, "raw_context": 65536, "window": 128, "page_size": 128, "heads": 128, "head_dim": 512}, 173, frozenset({"representative", "legacy", "decode", "swa"}), 1800),
        )

    @staticmethod
    def _validate(symbols):
        batch, raw, window, page_size, heads, head_dim = (int(symbols[n]) for n in ("batch", "raw_context", "window", "page_size", "heads", "head_dim"))
        if batch <= 0 or raw <= 0:
            raise ValueError("batch and raw_context must be positive")
        if window != WINDOW or page_size != WINDOW or heads != HEADS or head_dim != HEAD_DIM:
            raise NotImplementedError("unsupported SWA layout")
        return batch, raw, min(raw, WINDOW)

    def layout_contract(self, case):
        batch, raw, valid = self._validate(case.symbols)
        return {"q": [batch, 1, HEADS, HEAD_DIM], "cache": [batch, WINDOW, 1, CACHE_BYTES_PER_TOKEN], "block_table": None, "physical_indices": [batch, 1, WINDOW], "raw_context": raw, "valid_window": valid, "padding_tokens": WINDOW - valid, "cache_mutation": "forbidden", "observed_state": ["cache_head", "cache_tail", "lengths"], "workspace_lifecycle": "FlashMLASchedMeta allocated once per isolated clone"}

    def make_inputs(self, case, context):
        torch, Scheduler = _runtime()
        batch, _, valid = self._validate(case.symbols)
        device = next(iter(context.cuda), "cuda:0")
        if context.cuda.get(device) is None:
            raise RuntimeError("dense SWA attention requires a CUDA generator")
        required = batch * WINDOW * CACHE_BYTES_PER_TOKEN + batch * HEADS * HEAD_DIM * 2
        free, _ = torch.cuda.mem_get_info(device)
        _check_available_memory(required, free)
        q = torch.zeros((batch, 1, HEADS, HEAD_DIM), device=device, dtype=torch.bfloat16)
        cache = _padded_cache(torch, batch, device)
        indices = torch.arange(batch, device=device, dtype=torch.int32).view(batch, 1, 1) * WINDOW + torch.arange(WINDOW, device=device, dtype=torch.int32).view(1, 1, WINDOW)
        lengths = torch.full((batch,), valid, device=device, dtype=torch.int32)
        sink = torch.zeros((HEADS,), device=device, dtype=torch.float32)
        oracle, oracle_indices = _bounded_zero_oracle(torch, q, "representative" in case.tags)
        observed = {"semantic_oracle": oracle, "oracle_indices": oracle_indices, "cache_head": cache[:1, :1], "cache_tail": cache[-1:, -1:], "lengths": lengths}
        return InputBundle(args=(q, cache, indices, lengths, sink, Scheduler()), observed_state=observed)

    def clone_inputs(self, inputs):
        torch, Scheduler = _runtime()
        q, cache, indices, lengths, sink, _ = inputs.args
        q2, indices2, lengths2, sink2 = q.clone(), indices.clone(), lengths.clone(), sink.clone()
        cache2 = _clone_padded_cache(torch, cache)
        observed = {"semantic_oracle": inputs.observed_state["semantic_oracle"].clone(), "oracle_indices": inputs.observed_state["oracle_indices"].clone(), "cache_head": cache2[:1, :1], "cache_tail": cache2[-1:, -1:], "lengths": lengths2}
        return InputBundle(args=(q2, cache2, indices2, lengths2, sink2, Scheduler()), observed_state=observed)

    def normalize_output(self, output):
        return _bounded_output(output)

    def comparator(self, case):
        del case
        return DenseSwaComparator()

    def cost_model(self, case):
        batch, _, valid = self._validate(case.symbols)
        return {"flops": 4 * batch * HEADS * valid * HEAD_DIM, "estimated_bytes": batch * valid * CACHE_BYTES_PER_TOKEN + batch * HEADS * HEAD_DIM * 4, "throughput_units": batch * valid}

    def workspace_bytes(self, case):
        self._validate(case.symbols)
        return None

    def legacy_mappings(self):
        return LEGACY_MAPPINGS


SPEC = DeepSeekV4DenseSwaAttentionSpec()


def cost_model(case):
    return SPEC.cost_model(case)
