"""DeepSeek-V4-Pro local SGLang 0.5.15 operator benchmark."""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

from benchmark_environment import (
    collect_environment,
    environment_fingerprint,
    load_lock,
)


C4_LAYERS = 30
C128_LAYERS = 30
DENSE_LAYERS = 1
MODEL_LAYERS = C4_LAYERS + C128_LAYERS + DENSE_LAYERS
INDEX_HEADS = 64
INDEX_HEAD_DIM = 128
INDEX_TOPK = 1024
PAGE_SIZE = 64
ATTN_HEADS = 128
E_GLOBAL = 384
E_LOCAL = 16
MOE_INTERMEDIATE = 3072
MOE_TOPK = 6
ATTN_HEAD_DIM = 512
QUANT_PROFILES = ("mxfp4", "fp8_mxfp8")


@dataclass(frozen=True)
class BenchmarkRow:
    name: str
    backend: str
    instances: int
    call_ms: float | None
    status: str
    error: str = ""
    input_shape: str = ""
    output_shape: str = ""


@dataclass(frozen=True)
class Summary:
    total_ms: float
    percent_by_name: dict[str, float]


@dataclass(frozen=True)
class Adapter:
    name: str
    backend: str
    instances: int
    shape: tuple[int, ...] | None = None
    kind: str = "missing"
    reason: str = ""
    m_override: int | None = None


def case_context(phase, m, context):
    if context is not None:
        return context
    return m if phase == "prefill" else 16384


def compressed_context(raw_context, ratio):
    return (raw_context + ratio - 1) // ratio


def validate_quant_profile(profile):
    if profile not in QUANT_PROFILES:
        raise ValueError(f"unknown quant profile: {profile}")
    return profile


def local_routed_pairs(m):
    """Expected token/expert pairs owned by one EP rank for uniform routing."""
    return m * MOE_TOPK * E_LOCAL // E_GLOBAL


def _shape_text(*dims):
    return "(" + ",".join(str(dim) for dim in dims) + ")"


def adapter_io_shapes(adapter, m, context):
    m = adapter.m_override if adapter.m_override is not None else m
    if adapter.kind in ("fp8", "bf16"):
        k, n = adapter.shape
        return (
            f"x={_shape_text(m, k)}; weight={_shape_text(n, k)}",
            f"y={_shape_text(m, n)}",
        )
    if adapter.kind == "grouped_bf16":
        groups, k, n = adapter.shape
        return (
            f"x={_shape_text(m, groups, k)}; weight={_shape_text(groups, k, n)}",
            f"y={_shape_text(m, groups, n)}",
        )
    if adapter.kind.startswith("moe_"):
        experts, hidden, intermediate = adapter.shape
        return (
            f"x={_shape_text(m, hidden)}; topk_ids/weights={_shape_text(m, MOE_TOPK)}; "
            f"local_pairs={local_routed_pairs(m)}; "
            f"w13={_shape_text(experts, 2 * intermediate, hidden)}; "
            f"w2={_shape_text(experts, hidden, intermediate)}",
            f"y={_shape_text(m, hidden)}",
        )
    if adapter.kind in ("prefill_attention", "dense_prefill_attention"):
        selected = INDEX_TOPK if adapter.kind == "prefill_attention" else min(128, context)
        return (
            f"q={_shape_text(m, ATTN_HEADS, ATTN_HEAD_DIM)}; "
            f"kv={_shape_text(context, 1, ATTN_HEAD_DIM)}; "
            f"indices={_shape_text(m, 1, selected)}",
            f"y={_shape_text(m, ATTN_HEADS, ATTN_HEAD_DIM)}",
        )
    if adapter.kind.startswith("decode_attention") or adapter.kind == "dense_decode_attention":
        return (
            f"q={_shape_text(m, 1, ATTN_HEADS, ATTN_HEAD_DIM)}; context={context}",
            f"y={_shape_text(m, 1, ATTN_HEADS, ATTN_HEAD_DIM)}",
        )
    if adapter.kind in ("fp4_quant", "fp8_quant"):
        quantized_dim = INDEX_HEAD_DIM // 2 if adapter.kind == "fp4_quant" else INDEX_HEAD_DIM
        quantized_name = "q_fp4" if adapter.kind == "fp4_quant" else "q_fp8"
        return (
            f"q={_shape_text(m, INDEX_HEADS, INDEX_HEAD_DIM)}",
            f"{quantized_name}={_shape_text(m, INDEX_HEADS, quantized_dim)}",
        )
    if adapter.kind in ("fp4_logits", "fp8_logits"):
        c4_context = compressed_context(context, 4)
        return (
            f"q={_shape_text(m, 1, INDEX_HEADS, INDEX_HEAD_DIM)}; "
            f"raw_kv={context}; c4_kv={c4_context}",
            f"logits={_shape_text(m, c4_context)}",
        )
    if adapter.kind == "topk":
        c4_context = compressed_context(context, 4)
        return (
            f"scores={_shape_text(m, c4_context)}; raw_kv={context}",
            f"indices={_shape_text(m, INDEX_TOPK)}",
        )
    return ("-", "-")


def summarize_rows(rows):
    valid = [
        row for row in rows if row.status == "executed" and row.call_ms is not None
    ]
    total = sum(row.instances * row.call_ms for row in valid)
    percentages = (
        {
            row.name: row.instances * row.call_ms / total * 100
            for row in valid
        }
        if total
        else {}
    )
    return Summary(total, percentages)


def _profile_adapters(profile):
    validate_quant_profile(profile)
    if profile == "mxfp4":
        return (
            Adapter("C4 Indexer FP4 Quant", "SGLang fused RoPE/Hadamard FP4", C4_LAYERS, kind="fp4_quant"),
            Adapter("C4 FP4 Paged MQA Logits", "DeepGEMM fp8_fp4_paged_mqa_logits", C4_LAYERS, kind="fp4_logits"),
            Adapter("Routed Expert Fused MoE", "FlashInfer TRTLLM MXFP4", MODEL_LAYERS, (E_LOCAL, 7168, MOE_INTERMEDIATE), "moe_mxfp4"),
        )
    return (
        Adapter("C4 Indexer FP8 Quant", "SGLang fused RoPE/Hadamard FP8", C4_LAYERS, kind="fp8_quant"),
        Adapter("C4 FP8 Paged MQA Logits", "DeepGEMM fp8_paged_mqa_logits", C4_LAYERS, kind="fp8_logits"),
        Adapter("Routed Expert Fused MoE", "FlashInfer TRTLLM FP8 weight + MXFP8 activation", MODEL_LAYERS, (E_LOCAL, 7168, MOE_INTERMEDIATE), "moe_fp8_mxfp8"),
    )


def prefill_adapters(profile="mxfp4"):
    indexer_quant, indexer_logits, moe = _profile_adapters(profile)
    return [
        Adapter("Fused WQ_A + WKV", "DeepGEMM fp8_gemm_nt", MODEL_LAYERS, (7168, 2048), "fp8"),
        Adapter("Q RMSNorm + WQ_B", "DeepGEMM fp8_gemm_nt", MODEL_LAYERS, (1536, 65536), "fp8"),
        Adapter("Compressor WKV-Gate GEMM (C4)", "DeepGEMM fp8_gemm_nt", C4_LAYERS, (7168, 2048), "fp8"),
        Adapter("Compressor WKV-Gate GEMM (C128)", "DeepGEMM fp8_gemm_nt", C128_LAYERS, (7168, 1024), "fp8"),
        Adapter("C4 Indexer Q Projection", "DeepGEMM fp8_gemm_nt", C4_LAYERS, (1536, 65536), "fp8"),
        Adapter("C4 Indexer Head Weight Projection", "cuBLAS BF16 GEMM", C4_LAYERS, (7168, 64), "bf16"),
        indexer_quant,
        indexer_logits,
        Adapter("C4 TopK Transform", "SGLang JIT topk_transform_512_v2", C4_LAYERS, kind="topk"),
        Adapter("Sparse Prefill Attention", "sgl-kernel FlashMLA sparse_fwd", C4_LAYERS + C128_LAYERS, kind="prefill_attention"),
        Adapter("Dense SWA Attention", "sgl-kernel FlashMLA SWA-only", DENSE_LAYERS, kind="dense_prefill_attention"),
        Adapter("WO_A Grouped Projection", "cuBLAS grouped BF16 GEMM", MODEL_LAYERS, (16, 4096, 1024), "grouped_bf16"),
        Adapter("WO_B Projection", "DeepGEMM fp8_gemm_nt", MODEL_LAYERS, (16384, 7168), "fp8"),
        moe,
        Adapter("LM Head", "DeepGEMM FP8 full vocab", 1, (7168, 129280), "fp8"),
    ]


def decode_adapters(profile="mxfp4"):
    indexer_quant, indexer_logits, moe = _profile_adapters(profile)
    return [
        Adapter("Fused WQ_A + WKV", "DeepGEMM fp8_gemm_nt", MODEL_LAYERS, (7168, 2048), "fp8"),
        Adapter("Q RMSNorm + WQ_B", "DeepGEMM fp8_gemm_nt", MODEL_LAYERS, (1536, 65536), "fp8"),
        Adapter("C4 Indexer Q Projection", "DeepGEMM fp8_gemm_nt", C4_LAYERS, (1536, 65536), "fp8"),
        Adapter("C4 Indexer Head Weight Projection", "cuBLAS BF16 GEMM", C4_LAYERS, (7168, 64), "bf16"),
        indexer_quant,
        indexer_logits,
        Adapter("C4 TopK Transform", "SGLang JIT topk_transform_512_v2", C4_LAYERS, kind="topk"),
        Adapter("Sparse Decode Attention C4", "sgl-kernel FlashMLA dual-cache C4", C4_LAYERS, kind="decode_attention_c4"),
        Adapter("Sparse Decode Attention C128", "sgl-kernel FlashMLA dual-cache C128", C128_LAYERS, kind="decode_attention_c128"),
        Adapter("Dense SWA Attention", "sgl-kernel FlashMLA SWA-only", DENSE_LAYERS, kind="dense_decode_attention"),
        Adapter("WO_A Grouped Projection", "cuBLAS grouped BF16 GEMM", MODEL_LAYERS, (16, 4096, 1024), "grouped_bf16"),
        Adapter("WO_B Projection", "DeepGEMM fp8_gemm_nt", MODEL_LAYERS, (16384, 7168), "fp8"),
        moe,
        Adapter("LM Head", "DeepGEMM FP8 full vocab", 1, (7168, 129280), "fp8"),
    ]


def graph_ms(fn, torch, warmup, runs):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(runs):
            fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)
    start.record()
    graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / runs


def _pack_fp4_cache(k_fp4, k_scale, num_blocks, torch):
    packed = torch.empty(
        (num_blocks, PAGE_SIZE * 68), dtype=torch.uint8, device="cuda"
    )
    packed[:, : PAGE_SIZE * 64].view(num_blocks, PAGE_SIZE, 64).copy_(
        k_fp4.view(torch.uint8).view(num_blocks, PAGE_SIZE, 64)
    )
    packed[:, PAGE_SIZE * 64 :].view(num_blocks, PAGE_SIZE, 4).copy_(
        k_scale.contiguous().view(torch.uint8).view(num_blocks, PAGE_SIZE, 4)
    )
    return packed.view(num_blocks, PAGE_SIZE, 1, 68)


def _pack_flashmla_cache(k, torch):
    value_dim = ATTN_HEAD_DIM
    tile_size = 128
    num_tiles = value_dim // tile_size
    num_blocks, block_size, num_heads, _ = k.shape
    assert num_heads == 1
    k = k.squeeze(2)
    packed = torch.empty(
        (num_blocks, block_size, value_dim + num_tiles * 4),
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    values = packed[..., :value_dim]
    scales = packed[..., value_dim:].view(torch.float32)
    for tile in range(num_tiles):
        part = k[..., tile * tile_size : (tile + 1) * tile_size]
        scale = part.abs().float().amax(dim=-1).clamp(1.0e-4) / 448.0
        scales[..., tile] = scale
        values[..., tile * tile_size : (tile + 1) * tile_size] = (
            part.float() / scale.unsqueeze(-1)
        ).to(torch.float8_e4m3fn)
    return packed.view(num_blocks, block_size, 1, -1)


def _fp4_quant_fn(batch, context, torch):
    from sglang.jit_kernel.dsv4 import fused_q_indexer_rope_hadamard_fp4_quant
    from sglang.srt.layers.deepseek_v4_rope import precompute_freqs_cis

    q = torch.randn(batch, INDEX_HEADS, INDEX_HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(batch, INDEX_HEADS, device="cuda", dtype=torch.bfloat16)
    positions = torch.full((batch,), context - 1, device="cuda", dtype=torch.int32)
    freqs = precompute_freqs_cis(64, context + 1, 0, 10000, 1, 32, 1).to("cuda")
    weight_scale = INDEX_HEAD_DIM**-0.5 * INDEX_HEADS**-0.5
    return lambda: fused_q_indexer_rope_hadamard_fp4_quant(
        q, weights, weight_scale, freqs, positions
    )


def _fp8_quant_fn(batch, context, torch):
    from sglang.jit_kernel.dsv4 import fused_q_indexer_rope_hadamard_quant
    from sglang.srt.layers.deepseek_v4_rope import precompute_freqs_cis

    q = torch.randn(
        batch,
        INDEX_HEADS,
        INDEX_HEAD_DIM,
        device="cuda",
        dtype=torch.bfloat16,
    )
    weights = torch.randn(
        batch, INDEX_HEADS, device="cuda", dtype=torch.bfloat16
    )
    positions = torch.full(
        (batch,), context - 1, device="cuda", dtype=torch.int32
    )
    freqs = precompute_freqs_cis(64, context + 1, 0, 10000, 1, 32, 1).to(
        "cuda"
    )
    weight_scale = INDEX_HEAD_DIM**-0.5 * INDEX_HEADS**-0.5
    return lambda: fused_q_indexer_rope_hadamard_quant(
        q, weights, weight_scale, freqs, positions
    )


def _fp4_logits_fn(batch, context, torch):
    import deep_gemm
    from deep_gemm.utils import per_token_cast_to_fp4

    blocks_per_sequence = (context + PAGE_SIZE - 1) // PAGE_SIZE
    padded_context = blocks_per_sequence * PAGE_SIZE
    num_blocks = batch * blocks_per_sequence
    num_tokens = num_blocks * PAGE_SIZE
    page_table = torch.arange(num_blocks, dtype=torch.int32, device="cuda").view(
        batch, blocks_per_sequence
    )
    context_lens = torch.full(
        (batch, 1), context, dtype=torch.int32, device="cuda"
    )
    schedule = deep_gemm.get_paged_mqa_logits_metadata(
        context_lens, PAGE_SIZE, deep_gemm.get_num_sms(), indices=None
    )
    q = torch.randn(
        batch, 1, INDEX_HEADS, INDEX_HEAD_DIM,
        device="cuda", dtype=torch.bfloat16,
    )
    k = torch.randn(num_tokens, INDEX_HEAD_DIM, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(batch, INDEX_HEADS, device="cuda", dtype=torch.float32)
    q_fp4, q_scale = per_token_cast_to_fp4(
        q.view(-1, INDEX_HEAD_DIM), use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )
    q_fp4 = q_fp4.view(batch, 1, INDEX_HEADS, INDEX_HEAD_DIM // 2)
    q_scale = q_scale.view(batch, 1, INDEX_HEADS)
    k_fp4, k_scale = per_token_cast_to_fp4(
        k, use_ue8m0=True, gran_k=32, use_packed_ue8m0=True
    )
    k_cache = _pack_fp4_cache(k_fp4, k_scale, num_blocks, torch)
    del q, k, k_fp4, k_scale

    return lambda: deep_gemm.fp8_fp4_paged_mqa_logits(
        (q_fp4, q_scale),
        k_cache,
        weights,
        context_lens,
        page_table,
        schedule,
        padded_context,
        clean_logits=False,
        logits_dtype=torch.float32,
        indices=None,
    )


def _fp8_logits_fn(batch, context, torch):
    import deep_gemm
    from sglang.jit_kernel.dsa import deepgemm_paged_mqa_logits_split
    from sglang.srt.layers.attention.dsa.utils import (
        fp8_mqa_logits_make_fused_kv,
    )

    blocks_per_sequence = (context + PAGE_SIZE - 1) // PAGE_SIZE
    num_blocks = batch * blocks_per_sequence
    page_table = torch.arange(
        num_blocks, dtype=torch.int32, device="cuda"
    ).view(batch, blocks_per_sequence)
    context_lens = torch.full(
        (batch,), context, dtype=torch.int32, device="cuda"
    )
    context_lens_2d = context_lens.unsqueeze(-1)
    schedule = deep_gemm.get_paged_mqa_logits_metadata(
        context_lens_2d, PAGE_SIZE, deep_gemm.get_num_sms()
    )
    q_fp8 = torch.ones(
        batch,
        INDEX_HEADS,
        INDEX_HEAD_DIM,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    kv_fp8 = torch.ones(
        num_blocks,
        PAGE_SIZE,
        INDEX_HEAD_DIM,
        dtype=torch.float8_e4m3fn,
        device="cuda",
    )
    kv_scale = torch.ones(
        num_blocks, PAGE_SIZE, dtype=torch.float32, device="cuda"
    )
    kv_fused = fp8_mqa_logits_make_fused_kv(
        kv_fp8, kv_scale, PAGE_SIZE, INDEX_HEAD_DIM
    )
    del kv_fp8, kv_scale
    weights = torch.randn(
        batch, INDEX_HEADS, device="cuda", dtype=torch.float32
    )

    return lambda: deepgemm_paged_mqa_logits_split(
        deep_gemm.fp8_paged_mqa_logits,
        q_fp8,
        kv_fused,
        weights,
        context_lens_2d,
        page_table,
        schedule,
        context,
        q_offset=batch,
    )


def _topk_fn(batch, context, torch):
    from sglang.jit_kernel.dsv4.topk import plan_topk_v2, topk_transform_512_v2

    scores = torch.randn(batch, context, dtype=torch.float32, device="cuda")
    seq_lens = torch.full((batch,), context, dtype=torch.int32, device="cuda")
    num_pages = (context + PAGE_SIZE - 1) // PAGE_SIZE
    page_table = torch.arange(num_pages, dtype=torch.int32, device="cuda").unsqueeze(0)
    page_table = page_table.expand(batch, -1).contiguous()
    output = torch.empty(batch, INDEX_TOPK, dtype=torch.int32, device="cuda")
    metadata = plan_topk_v2(seq_lens)
    return lambda: topk_transform_512_v2(
        scores, seq_lens, page_table, output, PAGE_SIZE, metadata
    )


def _padded_v4_cache(num_pages, page_size, torch):
    bytes_per_token = 584
    row_bytes = ((page_size * bytes_per_token + 575) // 576) * 576
    raw = torch.zeros(num_pages, row_bytes, device="cuda", dtype=torch.uint8)
    cache = raw[:, : page_size * bytes_per_token].view(
        num_pages, page_size, 1, bytes_per_token
    )
    return raw, cache


def _decode_attention_fn(batch, context, compress_ratio, torch):
    from sgl_kernel.flash_mla import FlashMLASchedMeta, flash_mla_with_kvcache

    q = torch.randn(
        batch, 1, ATTN_HEADS, ATTN_HEAD_DIM,
        device="cuda", dtype=torch.bfloat16,
    )
    _, swa_cache = _padded_v4_cache(batch, 128, torch)
    swa_indices = (
        torch.arange(batch, device="cuda", dtype=torch.int32).view(batch, 1, 1) * 128
        + torch.arange(128, device="cuda", dtype=torch.int32).view(1, 1, 128)
    )
    swa_lengths = torch.full(
        (batch,), min(context, 128), device="cuda", dtype=torch.int32
    )

    if compress_ratio == 0:
        scheduler = FlashMLASchedMeta()
        return lambda: flash_mla_with_kvcache(
            q=q,
            k_cache=swa_cache,
            block_table=None,
            cache_seqlens=None,
            head_dim_v=ATTN_HEAD_DIM,
            tile_scheduler_metadata=scheduler,
            softmax_scale=ATTN_HEAD_DIM**-0.5,
            is_fp8_kvcache=True,
            indices=swa_indices,
            attn_sink=torch.zeros(ATTN_HEADS, device="cuda", dtype=torch.float32),
            topk_length=swa_lengths,
            extra_k_cache=None,
            extra_indices_in_kvcache=None,
            extra_topk_length=None,
        )
    if compress_ratio == 4:
        extra_page_size = 64
        extra_valid = min(INDEX_TOPK, max(1, context // 4))
    elif compress_ratio == 128:
        extra_page_size = 2
        extra_valid = max(1, context // 128)
    else:
        raise ValueError(f"unsupported compress ratio: {compress_ratio}")
    extra_topk = ((extra_valid + 63) // 64) * 64
    pages_per_request = (extra_topk + extra_page_size - 1) // extra_page_size
    _, extra_cache = _padded_v4_cache(
        batch * pages_per_request, extra_page_size, torch
    )
    extra_indices = (
        torch.arange(batch, device="cuda", dtype=torch.int32).view(batch, 1, 1)
        * pages_per_request
        * extra_page_size
        + torch.arange(extra_topk, device="cuda", dtype=torch.int32).view(1, 1, -1)
    )
    extra_lengths = torch.full(
        (batch,), extra_valid, device="cuda", dtype=torch.int32
    )
    sink = torch.zeros(ATTN_HEADS, device="cuda", dtype=torch.float32)
    scheduler = FlashMLASchedMeta()

    return lambda: flash_mla_with_kvcache(
        q=q,
        k_cache=swa_cache,
        block_table=None,
        cache_seqlens=None,
        head_dim_v=ATTN_HEAD_DIM,
        tile_scheduler_metadata=scheduler,
        softmax_scale=ATTN_HEAD_DIM**-0.5,
        is_fp8_kvcache=True,
        indices=swa_indices,
        attn_sink=sink,
        topk_length=swa_lengths,
        extra_k_cache=extra_cache,
        extra_indices_in_kvcache=extra_indices,
        extra_topk_length=extra_lengths,
    )



_MOE_WEIGHT_CACHE = None
_FP8_MXFP8_MOE_WEIGHT_CACHE = None


def _get_moe_weights(torch):
    """Build the exact SGLang MXFP4 SM100 layout once; preparation is untimed."""
    global _MOE_WEIGHT_CACHE
    if _MOE_WEIGHT_CACHE is not None:
        return _MOE_WEIGHT_CACHE

    from flashinfer.fp4_quantization import block_scale_interleave
    from flashinfer.fused_moe.core import (
        _maybe_get_cached_w3_w1_permute_indices,
        get_w2_permute_indices_with_cache,
    )

    e, hidden, intermediate = E_LOCAL, 7168, MOE_INTERMEDIATE
    w13 = torch.randint(
        0, 256, (e, 2 * intermediate, hidden // 2),
        dtype=torch.uint8, device="cuda",
    )
    w2 = torch.randint(
        0, 256, (e, hidden, intermediate // 2),
        dtype=torch.uint8, device="cuda",
    )
    # MXFP4 uses one E8M0 scale per 32 logical FP4 values.
    s13 = torch.ones(
        e, 2 * intermediate, hidden // 32,
        dtype=torch.float8_e8m0fnu, device="cuda",
    )
    s2 = torch.ones(
        e, hidden, intermediate // 32,
        dtype=torch.float8_e8m0fnu, device="cuda",
    )

    cache = {}
    g1_w, g1_s, g2_w, g2_s = [], [], [], []
    for expert in range(e):
        w13_u8 = w13[expert].view(torch.uint8)
        s13_u8 = s13[expert].view(torch.uint8)
        w2_u8 = w2[expert].view(torch.uint8)
        s2_u8 = s2[expert].view(torch.uint8)
        perm = _maybe_get_cached_w3_w1_permute_indices(cache, w13_u8, 128)
        g1_w.append(w13_u8[perm.to(w13_u8.device)].contiguous())
        perm = _maybe_get_cached_w3_w1_permute_indices(
            cache, s13_u8, 128, num_elts_per_sf=16
        )
        g1_s.append(block_scale_interleave(s13_u8[perm].contiguous()))
        perm = get_w2_permute_indices_with_cache(cache, w2_u8, 128)
        g2_w.append(w2_u8[perm.to(w2_u8.device)].contiguous())
        perm = get_w2_permute_indices_with_cache(
            cache, s2_u8, 128, num_elts_per_sf=16
        )
        g2_s.append(block_scale_interleave(s2_u8[perm].contiguous()))

    _MOE_WEIGHT_CACHE = (
        torch.stack(g1_w),
        torch.stack(g1_s).view(torch.float8_e4m3fn).reshape(e, 2 * intermediate, -1),
        torch.stack(g2_w),
        torch.stack(g2_s).view(torch.float8_e4m3fn).reshape(e, hidden, -1),
    )
    return _MOE_WEIGHT_CACHE


def _moe_fn(m, torch):
    """B200 SGLang 0.5.15 fused MXFP4 routed-expert backend fixture."""
    import flashinfer
    from flashinfer.fused_moe import trtllm_fp4_block_scale_routed_moe
    from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
        PackTopkIds,
    )

    w13, s13, w2, s2 = _get_moe_weights(torch)
    hidden = 7168
    x = torch.randn(m, hidden, dtype=torch.bfloat16, device="cuda")

    total_pairs = m * MOE_TOPK
    flat_ids = E_LOCAL + torch.arange(total_pairs, device="cuda") % (
        E_GLOBAL - E_LOCAL
    )
    owned_pairs = local_routed_pairs(m)
    if owned_pairs:
        owned_positions = torch.div(
            torch.arange(owned_pairs, device="cuda") * total_pairs,
            owned_pairs,
            rounding_mode="floor",
        )
        flat_ids[owned_positions] = torch.arange(
            owned_pairs, device="cuda"
        ) % E_LOCAL
    topk_ids = flat_ids.view(m, MOE_TOPK).to(torch.int32).contiguous()
    topk_weights = torch.full(
        (m, MOE_TOPK), 1.0 / MOE_TOPK,
        dtype=torch.float32, device="cuda",
    )
    packed_topk = PackTopkIds.vanilla(topk_ids, topk_weights)
    scales = torch.ones(E_LOCAL, dtype=torch.float32, device="cuda")
    clamp = torch.full((E_LOCAL,), 10.0, dtype=torch.float32, device="cuda")
    output = torch.empty(m, hidden, dtype=torch.bfloat16, device="cuda")

    def run():
        x_quant, x_scale = flashinfer.mxfp8_quantize(
            x, False, alignment=hidden, backend="cute-dsl"
        )
        x_scale = x_scale.view(torch.float8_e4m3fn).reshape(m, -1)
        return trtllm_fp4_block_scale_routed_moe(
            topk_ids=packed_topk,
            routing_bias=None,
            hidden_states=x_quant,
            hidden_states_scale=x_scale,
            gemm1_weights=w13,
            gemm1_weights_scale=s13,
            gemm1_bias=None,
            gemm1_alpha=None,
            gemm1_beta=None,
            gemm1_clamp_limit=clamp,
            gemm2_weights=w2,
            gemm2_weights_scale=s2,
            gemm2_bias=None,
            output1_scale_scalar=scales,
            output1_scale_gate_scalar=scales,
            output2_scale_scalar=scales,
            num_experts=E_GLOBAL,
            top_k=MOE_TOPK,
            n_group=1,
            topk_group=1,
            intermediate_size=MOE_INTERMEDIATE,
            local_expert_offset=0,
            local_num_experts=E_LOCAL,
            routed_scaling_factor=1.0,
            routing_method_type=1,
            do_finalize=True,
            tune_max_num_tokens=1 << (m - 1).bit_length(),
            output=output,
        )[0]

    return run


def _get_fp8_mxfp8_moe_weights(torch):
    """Build SGLang's shuffled block-FP8 TRTLLM layout once."""
    global _FP8_MXFP8_MOE_WEIGHT_CACHE
    if _FP8_MXFP8_MOE_WEIGHT_CACHE is not None:
        return _FP8_MXFP8_MOE_WEIGHT_CACHE

    from sglang.srt.layers.moe.moe_runner.flashinfer_trtllm import (
        align_mxfp8_moe_weights_for_flashinfer_trtllm,
    )

    e, hidden, intermediate = E_LOCAL, 7168, MOE_INTERMEDIATE
    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(
        torch.zeros(
            e,
            2 * intermediate,
            hidden,
            dtype=torch.float8_e4m3fn,
            device="cuda",
        ),
        requires_grad=False,
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.zeros(
            e,
            hidden,
            intermediate,
            dtype=torch.float8_e4m3fn,
            device="cuda",
        ),
        requires_grad=False,
    )
    layer.w13_weight_scale_inv = torch.nn.Parameter(
        torch.ones(
            e,
            2 * intermediate,
            hidden // 32,
            dtype=torch.float8_e8m0fnu,
            device="cuda",
        ).view(torch.uint8),
        requires_grad=False,
    )
    layer.w2_weight_scale_inv = torch.nn.Parameter(
        torch.ones(
            e,
            hidden,
            intermediate // 32,
            dtype=torch.float8_e8m0fnu,
            device="cuda",
        ).view(torch.uint8),
        requires_grad=False,
    )
    align_mxfp8_moe_weights_for_flashinfer_trtllm(layer)
    _FP8_MXFP8_MOE_WEIGHT_CACHE = (
        layer.w13_weight,
        layer.w13_weight_scale_inv,
        layer.w2_weight,
        layer.w2_weight_scale_inv,
    )
    return _FP8_MXFP8_MOE_WEIGHT_CACHE


def _moe_fp8_mxfp8_fn(m, torch):
    """B200 FP8-weight/MXFP8-activation fused routed-expert fixture."""
    import flashinfer
    from flashinfer.fused_moe import (
        Fp8QuantizationType,
        trtllm_fp8_block_scale_routed_moe,
    )
    from sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe import (
        PackTopkIds,
    )

    w13, s13, w2, s2 = _get_fp8_mxfp8_moe_weights(torch)
    hidden = 7168
    x = torch.randn(m, hidden, dtype=torch.bfloat16, device="cuda")

    total_pairs = m * MOE_TOPK
    flat_ids = E_LOCAL + torch.arange(total_pairs, device="cuda") % (
        E_GLOBAL - E_LOCAL
    )
    owned_pairs = local_routed_pairs(m)
    if owned_pairs:
        owned_positions = torch.div(
            torch.arange(owned_pairs, device="cuda") * total_pairs,
            owned_pairs,
            rounding_mode="floor",
        )
        flat_ids[owned_positions] = torch.arange(
            owned_pairs, device="cuda"
        ) % E_LOCAL
    topk_ids = flat_ids.view(m, MOE_TOPK).to(torch.int32).contiguous()
    topk_weights = torch.full(
        (m, MOE_TOPK),
        1.0 / MOE_TOPK,
        dtype=torch.float32,
        device="cuda",
    )
    packed_topk = PackTopkIds.vanilla(topk_ids, topk_weights)
    output = torch.empty(m, hidden, dtype=torch.bfloat16, device="cuda")

    def run():
        x_quant, x_scale = flashinfer.mxfp8_quantize(
            x, False, backend="cute-dsl"
        )
        x_scale = x_scale.view(torch.uint8).reshape(m, -1)
        return trtllm_fp8_block_scale_routed_moe(
            topk_ids=packed_topk,
            routing_bias=None,
            hidden_states=x_quant,
            hidden_states_scale=x_scale,
            gemm1_weights=w13,
            gemm1_weights_scale=s13,
            gemm2_weights=w2,
            gemm2_weights_scale=s2,
            num_experts=E_GLOBAL,
            top_k=MOE_TOPK,
            n_group=None,
            topk_group=None,
            intermediate_size=MOE_INTERMEDIATE,
            local_expert_offset=0,
            local_num_experts=E_LOCAL,
            routed_scaling_factor=1.0,
            routing_method_type=1,
            use_shuffled_weight=True,
            do_finalize=True,
            output=output,
            tune_max_num_tokens=1 << (m - 1).bit_length(),
            fp8_quantization_type=Fp8QuantizationType.MxFp8,
            activation_type=3,
        )

    return run

def run_adapter(adapter, m, context, torch, warmup, runs):
    m = adapter.m_override if adapter.m_override is not None else m
    input_shape, output_shape = adapter_io_shapes(adapter, m, context)
    if adapter.kind == "missing":
        return BenchmarkRow(
            adapter.name,
            adapter.backend,
            adapter.instances,
            None,
            "unavailable",
            adapter.reason,
            input_shape,
            output_shape,
        )

    if adapter.kind in ("prefill_attention", "dense_prefill_attention"):
        from sgl_kernel.flash_mla import flash_mla_sparse_fwd

        q = torch.randn(m, ATTN_HEADS, ATTN_HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        kv = torch.randn(context, 1, ATTN_HEAD_DIM, device="cuda", dtype=torch.bfloat16)
        selected = INDEX_TOPK if adapter.kind == "prefill_attention" else min(128, context)
        indices = torch.randint(
            0, context, (m, 1, selected), device="cuda", dtype=torch.int32
        )
        fn = lambda: flash_mla_sparse_fwd(
            q, kv, indices, ATTN_HEAD_DIM**-0.5
        )
    elif adapter.kind == "dense_decode_attention":
        fn = _decode_attention_fn(m, context, 0, torch)
    elif adapter.kind == "decode_attention_c4":
        fn = _decode_attention_fn(m, context, 4, torch)
    elif adapter.kind == "decode_attention_c128":
        fn = _decode_attention_fn(m, context, 128, torch)
    elif adapter.kind == "fp4_quant":
        fn = _fp4_quant_fn(m, context, torch)
    elif adapter.kind == "fp8_quant":
        fn = _fp8_quant_fn(m, context, torch)
    elif adapter.kind == "fp4_logits":
        fn = _fp4_logits_fn(m, compressed_context(context, 4), torch)
    elif adapter.kind == "fp8_logits":
        fn = _fp8_logits_fn(m, compressed_context(context, 4), torch)
    elif adapter.kind == "topk":
        fn = _topk_fn(m, compressed_context(context, 4), torch)
    elif adapter.kind == "grouped_bf16":
        groups, k, n = adapter.shape
        x = torch.randn(groups, m, k, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(groups, k, n, device="cuda", dtype=torch.bfloat16)
        fn = lambda: torch.bmm(x, weight)
    elif adapter.kind == "moe_mxfp4":
        fn = _moe_fn(m, torch)
    elif adapter.kind == "moe_fp8_mxfp8":
        fn = _moe_fp8_mxfp8_fn(m, torch)
    elif adapter.kind == "bf16":
        k, n = adapter.shape
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
        fn = lambda: torch.mm(x, weight.T)
    else:
        import deep_gemm
        from deep_gemm.utils.layout import get_mn_major_tma_aligned_tensor

        k, n = adapter.shape
        x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        x_scale = get_mn_major_tma_aligned_tensor(
            torch.ones(m, k // 128, device="cuda")
        )
        weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16).to(torch.float8_e4m3fn)
        weight_scale = torch.ones(n // 128, k // 128, device="cuda")
        output = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
        fn = lambda: deep_gemm.fp8_gemm_nt(
            (x, x_scale), (weight, weight_scale), output
        )

    return BenchmarkRow(
        adapter.name,
        adapter.backend,
        adapter.instances,
        graph_ms(fn, torch, warmup, runs),
        "executed",
        input_shape=input_shape,
        output_shape=output_shape,
    )


def result_exit_code(rows):
    return 1 if any(row.status != "executed" for row in rows) else 0


def write_result_csv(path, phase, profile, m, context, rows, fingerprint):
    summary = summarize_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        writer.writerow(
            (
                "phase", "quant_profile", "environment_fingerprint", "m",
                "context", "operator", "backend", "instances", "call_ms",
                "model_ms", "pct", "status", "input_shape", "output_shape",
                "error",
            )
        )
        for row in rows:
            writer.writerow(
                (
                    phase,
                    profile,
                    fingerprint,
                    m,
                    context,
                    row.name,
                    row.backend,
                    row.instances,
                    "" if row.call_ms is None else f"{row.call_ms:.6f}",
                    "" if row.call_ms is None else f"{row.instances * row.call_ms:.6f}",
                    f"{summary.percent_by_name.get(row.name, 0):.4f}",
                    row.status,
                    row.input_shape,
                    row.output_shape,
                    row.error,
                )
            )


def main(phase):
    parser = argparse.ArgumentParser(
        description=f"DeepSeek-V4-Pro {phase} local backend benchmark"
    )
    parser.add_argument("--m", default="1024" if phase == "prefill" else "16")
    parser.add_argument("--context", type=int)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument(
        "--quant-profile", choices=QUANT_PROFILES, default="mxfp4"
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path(f"results/deepseek_v4_pro_{phase}_local.csv"),
    )
    args = parser.parse_args()

    import torch

    fingerprint = environment_fingerprint(collect_environment(load_lock()))
    all_rows = []

    for m in map(int, args.m.split(",")):
        context = case_context(phase, m, args.context)
        rows = []
        adapters = (
            prefill_adapters(args.quant_profile)
            if phase == "prefill"
            else decode_adapters(args.quant_profile)
        )
        for adapter in adapters:
            try:
                torch.cuda.empty_cache()
                rows.append(
                    run_adapter(
                        adapter, m, context, torch, args.warmup, args.runs
                    )
                )
            except Exception as error:
                rows.append(
                    BenchmarkRow(
                        adapter.name,
                        adapter.backend,
                        adapter.instances,
                        None,
                        "unavailable",
                        str(error),
                        *adapter_io_shapes(adapter, m, context),
                    )
                )

        summary = summarize_rows(rows)
        all_rows.extend(rows)
        print(
            f"DeepSeek-V4-Pro {phase}: profile={args.quant_profile}, M={m}, "
            f"environment={fingerprint}, measured_partial_total={summary.total_ms:.4f} ms"
        )
        for row in rows:
            latency = "-" if row.call_ms is None else f"{row.call_ms:.6f}"
            model_ms = "-" if row.call_ms is None else f"{row.instances * row.call_ms:.6f}"
            print(
                f"{row.name:<38} {latency:>10} ms/call "
                f"{model_ms:>12} ms/model "
                f"{summary.percent_by_name.get(row.name, 0):>7.2f}% "
                f"{row.status:<11} {row.backend}"
            )
            if row.error:
                print(f"  error: {row.error}")

        output_path = (
            args.csv
            if "," not in args.m
            else args.csv.with_name(f"{args.csv.stem}_m{m}{args.csv.suffix}")
        )
        write_result_csv(
            output_path, phase, args.quant_profile, m, context, rows, fingerprint
        )
    return result_exit_code(all_rows)
