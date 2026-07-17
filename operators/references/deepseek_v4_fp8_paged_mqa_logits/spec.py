"""Paged C4 index-logit contract; controller discovery imports no CUDA stack."""

import importlib

from benchmark_engine.correctness import InputBundle
from benchmark_engine.correctness.models import ComparisonResult, OutputBundle, OutputLeaf
from benchmark_engine.models import CaseSpec


HEADS = 64
HEAD_DIM = 128
PAGE_SIZE = 64
COMPRESSION_RATIO = 4
TOLERANCE = 0.1
LEGACY_MAPPING = {
    "adapter_name": "C4 FP8 Paged MQA Logits",
    "backend": "DeepGEMM fp8_paged_mqa_logits",
    "instances": 30,
    "raw_context": 65536,
    "compressed_context": 16384,
}


def _runtime():
    torch = importlib.import_module("torch")
    deep_gemm = importlib.import_module("deep_gemm")
    dsa = importlib.import_module("sglang.jit_kernel.dsa")
    utils = importlib.import_module("sglang.srt.layers.attention.dsa.utils")
    return torch, deep_gemm, dsa, utils


def _leaf(bundle, path):
    return bundle.by_path().get(path)


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


def _bounded_output(output, limit=256):
    flat = output.detach().reshape(-1)
    torch = importlib.import_module("torch")
    indices = _sample_indices(torch, flat.numel(), flat.device, limit)
    values = tuple(flat[indices].float().cpu().tolist())
    return OutputBundle((OutputLeaf("output", values, str(output.dtype), tuple(output.shape), tuple(output.stride()), "strided", str(output.device)),))


class OracleAndStateComparator:
    """Check both implementations against an oracle and immutable cache views."""

    def compare(self, reference, candidate, **_):
        failures = []
        maxima = {}
        for role, bundle in (("reference", reference), ("candidate", candidate)):
            output = _leaf(bundle, "output")
            oracle = _leaf(bundle, "state.semantic_oracle")
            indices = _leaf(bundle, "state.oracle_indices")
            if output is None or oracle is None or indices is None:
                failures.append({"role": role, "error": "missing bounded oracle leaves"})
                continue
            sampled = output.value
            maximum = max((abs(float(a) - float(b)) for a, b in zip(sampled, oracle.value)), default=0.0)
            maxima[role] = maximum
            if len(sampled) != len(oracle.value) or maximum > TOLERANCE:
                failures.append({"role": role, "max_abs_error": maximum})
        reference_leaves, candidate_leaves = reference.by_path(), candidate.by_path()
        reference_output = reference_leaves.get("output")
        candidate_output = candidate_leaves.get("output")
        if reference_output is not None and candidate_output is not None:
            cross = max((abs(float(a) - float(b)) for a, b in zip(reference_output.value, candidate_output.value)), default=0.0)
            maxima["reference_candidate"] = cross
            if len(reference_output.value) != len(candidate_output.value) or cross > TOLERANCE:
                failures.append({"path": "output", "reference_candidate_max_abs_error": cross})
        for path in sorted(set(reference_leaves) | set(candidate_leaves)):
            left, right = reference_leaves.get(path), candidate_leaves.get(path)
            if left is None or right is None or left.contract() != right.contract():
                failures.append({"path": path, "error": "reference/candidate contract mismatch"})
            elif path.startswith("state.") and path not in {"state.semantic_oracle", "state.oracle_indices"} and left.value != right.value:
                failures.append({"path": path, "error": "observed state mutated"})
        return ComparisonResult(not failures, "paged_mqa_oracle_and_state", {"max_abs_error": maxima}, tuple(failures[:8]), None if not failures else "output")


def _mqa_logits_oracle(torch, q, kv, weights, context_lens):
    """Independent dense semantic definition used only by correctness cases."""

    qf, kvf, wf = q.float(), kv.float(), weights.float()
    scores = torch.einsum("bhd,btd->bht", qf, kvf)
    scores = (scores * wf.unsqueeze(-1)).sum(dim=1)
    positions = torch.arange(kv.shape[1], device=kv.device).view(1, -1)
    return scores.masked_fill(positions >= context_lens.view(-1, 1), 0.0)


def _check_available_memory(required_bytes, available_bytes):
    if required_bytes > int(available_bytes * 0.9):
        raise MemoryError(f"estimated allocation {required_bytes} exceeds available CUDA memory {available_bytes}")


class DeepSeekV4Fp8PagedMqaLogitsSpec:
    operator_id = "deepseek_v4_fp8_paged_mqa_logits"

    def cases(self):
        return (
            CaseSpec("smoke_tail_page", {"batch": 2, "raw_context": 257, "compression_ratio": 4, "page_size": 64, "heads": 64, "head_dim": 128}, 101, frozenset({"smoke", "boundary", "tail_page"}), 600),
            CaseSpec("boundary_min_context", {"batch": 1, "raw_context": 1, "compression_ratio": 4, "page_size": 64, "heads": 64, "head_dim": 128}, 103, frozenset({"boundary", "minimum"}), 600),
            CaseSpec("representative_decode_b16_context65536", {"batch": 16, "raw_context": 65536, "compression_ratio": 4, "page_size": 64, "heads": 64, "head_dim": 128}, 107, frozenset({"representative", "legacy", "decode"}), 1800),
        )

    @staticmethod
    def _validate(symbols):
        batch, raw_context, ratio, page_size, heads, head_dim = (int(symbols[n]) for n in ("batch", "raw_context", "compression_ratio", "page_size", "heads", "head_dim"))
        if batch <= 0 or raw_context <= 0:
            raise ValueError("batch and raw_context must be positive")
        if ratio != COMPRESSION_RATIO:
            raise NotImplementedError("FP8 paged MQA logits supports the C4 cache only")
        if page_size != PAGE_SIZE or heads != HEADS or head_dim != HEAD_DIM:
            raise NotImplementedError("unsupported page/head layout")
        compressed = (raw_context + ratio - 1) // ratio
        return batch, raw_context, compressed

    def layout_contract(self, case):
        batch, raw, compressed = self._validate(case.symbols)
        pages = (compressed + PAGE_SIZE - 1) // PAGE_SIZE
        return {"q": [batch, HEADS, HEAD_DIM], "kv_cache": [batch * pages, PAGE_SIZE, HEAD_DIM + 4], "block_table": [batch, pages], "raw_context": raw, "compressed_context": compressed, "tail_tokens": compressed % PAGE_SIZE, "cache_mutation": "forbidden", "observed_state": ["cache_head", "cache_tail", "block_table"], "workspace_lifecycle": "schedule is prepared once per isolated input clone"}

    def make_inputs(self, case, context):
        torch, deep_gemm, _, utils = _runtime()
        batch, _, compressed = self._validate(case.symbols)
        device = next(iter(context.cuda), "cuda:0")
        generator = context.cuda.get(device)
        if generator is None:
            raise RuntimeError("paged MQA logits requires a CUDA generator")
        pages = (compressed + PAGE_SIZE - 1) // PAGE_SIZE
        blocks = batch * pages
        required = (
            batch * HEADS * HEAD_DIM
            + blocks * PAGE_SIZE * (HEAD_DIM + 4)
            + batch * pages * PAGE_SIZE * HEAD_DIM * 4
            + batch * compressed * 4
        )
        free, _ = torch.cuda.mem_get_info(device)
        _check_available_memory(required, free)
        page_table = torch.arange(blocks, dtype=torch.int32, device=device).view(batch, pages)
        context_lens = torch.full((batch,), compressed, dtype=torch.int32, device=device)
        context_lens_2d = context_lens.unsqueeze(-1)
        schedule = deep_gemm.get_paged_mqa_logits_metadata(context_lens_2d, PAGE_SIZE, deep_gemm.get_num_sms())
        q_fp8 = torch.ones((batch, HEADS, HEAD_DIM), dtype=torch.float8_e4m3fn, device=device)
        logical_tokens = pages * PAGE_SIZE
        token_values = ((torch.arange(logical_tokens, device=device) % 7) + 1).to(torch.float32) / 8.0
        logical_kv = token_values.view(1, logical_tokens, 1).expand(batch, logical_tokens, HEAD_DIM).clone()
        if compressed < logical_tokens:
            logical_kv[:, compressed:] = 4.0
        kv_fp8 = torch.empty((blocks, PAGE_SIZE, HEAD_DIM), dtype=torch.float8_e4m3fn, device=device)
        if batch > 1:
            page_table[1] = page_table[1].flip(0)
        for batch_index in range(batch):
            logical_pages = logical_kv[batch_index].view(pages, PAGE_SIZE, HEAD_DIM)
            kv_fp8[page_table[batch_index].long()] = logical_pages.to(torch.float8_e4m3fn)
        kv_scale = torch.ones((blocks, PAGE_SIZE), dtype=torch.float32, device=device)
        kv_fused = utils.fp8_mqa_logits_make_fused_kv(kv_fp8, kv_scale, PAGE_SIZE, HEAD_DIM)
        weights = torch.zeros((batch, HEADS), dtype=torch.float32, device=device)
        weights[:, 0] = 1.0 / HEAD_DIM
        oracle_full = _mqa_logits_oracle(torch, q_fp8, logical_kv[:, :compressed], weights, context_lens)
        count = min(oracle_full.numel(), 256 if "representative" in case.tags else oracle_full.numel())
        oracle_indices = _sample_indices(torch, oracle_full.numel(), device, count)
        oracle = oracle_full.reshape(-1)[oracle_indices]
        observed = {"semantic_oracle": oracle, "oracle_indices": oracle_indices, "cache_head": kv_fused[:1], "cache_tail": kv_fused[-1:], "block_table": page_table}
        return InputBundle(args=(q_fp8, kv_fused, weights, context_lens_2d, page_table, schedule, compressed, batch), observed_state=observed)

    def clone_inputs(self, inputs):
        torch, deep_gemm, _, _ = _runtime()
        q, cache, weights, lengths, table, _, max_context, q_offset = inputs.args
        q2, cache2, weights2, lengths2, table2 = q.clone(), cache.clone(), weights.clone(), lengths.clone(), table.clone()
        schedule2 = deep_gemm.get_paged_mqa_logits_metadata(lengths2, PAGE_SIZE, deep_gemm.get_num_sms())
        observed = {"semantic_oracle": inputs.observed_state["semantic_oracle"].clone(), "oracle_indices": inputs.observed_state["oracle_indices"].clone(), "cache_head": cache2[:1], "cache_tail": cache2[-1:], "block_table": table2}
        return InputBundle(args=(q2, cache2, weights2, lengths2, table2, schedule2, max_context, q_offset), observed_state=observed)

    def normalize_output(self, output):
        return _bounded_output(output)

    def comparator(self, case):
        del case
        return OracleAndStateComparator()

    def cost_model(self, case):
        batch, _, compressed = self._validate(case.symbols)
        return {"flops": 2 * batch * HEADS * compressed * HEAD_DIM, "estimated_bytes": batch * HEADS * HEAD_DIM + batch * compressed * (HEAD_DIM + 4) + batch * compressed * 4, "throughput_units": batch * compressed}

    def workspace_bytes(self, case):
        self._validate(case.symbols)
        return None

    def legacy_mappings(self):
        return LEGACY_MAPPING


SPEC = DeepSeekV4Fp8PagedMqaLogitsSpec()


def cost_model(case):
    return SPEC.cost_model(case)
