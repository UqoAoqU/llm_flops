"""Contract, deterministic cases and per-batch comparator for TopK transform."""

import importlib

from benchmark_engine.correctness import InputBundle
from benchmark_engine.correctness.models import ComparisonResult
from benchmark_engine.models import CaseSpec


LEGACY_TOPK = {
    "adapter_name": "C4 TopK Transform",
    "backend": "SGLang JIT topk_transform_512_v2",
    "instances": 30,
    "prefill_m": (1024, 2048, 4096),
    "decode_m": (16, 32),
    "context": 65536,
    "topk": 1024,
    "page_size": 64,
}


class BatchedUnorderedTopKComparator:
    """Compare each batch row as a set; never flatten across batch boundaries."""

    def compare(self, reference, candidate, **_):
        refs, cands = reference.by_path(), candidate.by_path()
        path = "output"
        if set(refs) != {path} or set(cands) != {path}:
            return ComparisonResult(False, "batched_unordered_topk", {"error": "output structure mismatch"}, failed_path=path)
        left, right = refs[path], cands[path]
        if left.shape != right.shape or len(left.shape) != 2 or left.dtype != right.dtype:
            return ComparisonResult(False, "batched_unordered_topk", {"error": "dtype/shape mismatch"}, failed_path=path)
        batch, width = left.shape
        failed = []
        for row in range(batch):
            a = tuple(int(v) for v in left.value[row * width : (row + 1) * width])
            b = tuple(int(v) for v in right.value[row * width : (row + 1) * width])
            if len(set(a)) != width or len(set(b)) != width or set(a) != set(b):
                failed.append({"batch": row, "reference": a[:8], "candidate": b[:8]})
        return ComparisonResult(
            not failed,
            "batched_unordered_topk",
            {"batch": batch, "k": width, "unordered": True, "failed_batches": len(failed)},
            tuple(failed[:8]),
            None if not failed else path,
        )


def _runtime():
    torch = importlib.import_module("torch")
    topk = importlib.import_module("sglang.jit_kernel.dsv4.topk")
    return torch, topk.plan_topk_v2


class DeepSeekV4TopKTransformSpec:
    operator_id = "deepseek_v4_topk_transform"

    def cases(self):
        return (
            CaseSpec("smoke_b2_l128_k8", {"batch": 2, "length": 128, "seq_len": 117, "topk": 8, "page_size": 64, "pattern": "random"}, 31, frozenset({"smoke"}), 600),
            CaseSpec("boundary_k1", {"batch": 2, "length": 68, "seq_len": 65, "topk": 1, "page_size": 64, "pattern": "random"}, 37, frozenset({"boundary"}), 600),
            CaseSpec("boundary_cutoff_tie", {"batch": 2, "length": 132, "seq_len": 129, "topk": 16, "page_size": 64, "pattern": "tie"}, 41, frozenset({"boundary", "adversarial"}), 600),
            CaseSpec("boundary_all_negative", {"batch": 2, "length": 132, "seq_len": 129, "topk": 8, "page_size": 64, "pattern": "negative"}, 43, frozenset({"boundary"}), 600),
            CaseSpec("boundary_tail_page", {"batch": 2, "length": 196, "seq_len": 191, "topk": 32, "page_size": 64, "pattern": "random"}, 47, frozenset({"boundary"}), 600),
            CaseSpec("representative_decode_b16_l65536_k1024", {"batch": 16, "length": 65536, "seq_len": 65536, "topk": 1024, "page_size": 64, "pattern": "random"}, 53, frozenset({"representative", "legacy", "decode"}), 1800),
        )

    def legacy_mappings(self):
        return LEGACY_TOPK

    @staticmethod
    def _validate(symbols):
        batch, length, seq_len, topk, page_size = (int(symbols[n]) for n in ("batch", "length", "seq_len", "topk", "page_size"))
        if batch <= 0 or length <= 0 or topk <= 0 or topk > 2048:
            raise ValueError("batch/length/topk must be positive and topk <= 2048")
        if seq_len < 0 or seq_len > length:
            raise ValueError("seq_len must satisfy 0 <= seq_len <= length")
        if seq_len < topk:
            raise ValueError("formal cases require seq_len >= topk; kernel -1 padding is tested separately")
        if page_size <= 0 or page_size & (page_size - 1):
            raise ValueError("page_size must be a positive power of two")
        return batch, length, seq_len, topk, page_size

    def make_inputs(self, case, context):
        torch, plan = _runtime()
        batch, length, seq_len, topk, page_size = self._validate(case.symbols)
        device = next(iter(context.cuda), "cuda:0")
        generator = context.cuda.get(device)
        if generator is None:
            raise RuntimeError("deepseek_v4_topk_transform requires a CUDA generator")
        scores = torch.randn((batch, length), device=device, dtype=torch.float32, generator=generator)
        pattern = case.symbols.get("pattern", "random")
        if pattern == "tie":
            scores.fill_(-4.0)
            scores[:, : topk + 3] = 1.0
        elif pattern == "negative":
            scores.copy_(-scores.abs() - 0.01)
        seq_lens = torch.full((batch,), seq_len, device=device, dtype=torch.int32)
        pages = (length + page_size - 1) // page_size
        page_tables = torch.arange(batch * pages, device=device, dtype=torch.int32).view(batch, pages)
        output = torch.empty((batch, topk), device=device, dtype=torch.int32)
        metadata = plan(seq_lens, static_threshold=0)
        return InputBundle(args=(scores, seq_lens, page_tables, output, page_size, metadata))

    def clone_inputs(self, inputs):
        scores, seq_lens, page_tables, output, page_size, metadata = inputs.args
        return InputBundle(args=(scores.clone(), seq_lens.clone(), page_tables.clone(), output.clone(), page_size, metadata.clone()))

    def normalize_output(self, output):
        return output

    def comparator(self, case):
        del case
        return BatchedUnorderedTopKComparator()

    def cost_model(self, case):
        batch, _, seq_len, topk, _ = self._validate(case.symbols)
        # One auditable score comparison unit per valid score. The sort's exact
        # implementation-dependent comparison count is intentionally not
        # presented as hardware FLOPs.
        work = batch * seq_len
        return {"flops": work, "estimated_bytes": work * 4 + batch * topk * 4, "throughput_units": work}


SPEC = DeepSeekV4TopKTransformSpec()


def cost_model(case):
    return SPEC.cost_model(case)
