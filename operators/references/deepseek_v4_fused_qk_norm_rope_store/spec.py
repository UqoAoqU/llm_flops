"""Contract for the fused DeepSeek V4 Q/K norm, RoPE, and SWA store."""

import importlib

from benchmark_engine.correctness import InputBundle
from benchmark_engine.models import CaseSpec
from benchmark_engine.workloads.deepseek_v4_flash import (
    NumericPath,
    OracleStateComparator,
    normalize_named_tensors,
    sample_tensor,
)


HEAD_DIM = 512
ROPE_DIM = 64
NOPE_DIM = 448
HEADS = 64
PAGE_SIZE = 128
VALUE_BYTES = 576
SCALE_BYTES = 8
BYTES_PER_TOKEN = VALUE_BYTES + SCALE_BYTES


def _rms_norm(torch, value, weight):
    normalized = value.float() * torch.rsqrt(
        value.float().square().mean(dim=-1, keepdim=True) + 1.0e-6
    )
    if weight is not None:
        normalized = normalized * weight.float()
    return normalized


def _rope(torch, value, cos, sin):
    paired = value.float().view(*value.shape[:-1], ROPE_DIM // 2, 2)
    rotated = torch.stack((-paired[..., 1], paired[..., 0]), dim=-1).flatten(-2)
    return value.float() * cos + rotated * sin


def _semantic_oracle(
    torch,
    q,
    kv,
    kv_weight,
    cos_cache,
    sin_cache,
    positions,
    cache,
    locations,
):
    m = q.shape[0]
    q_norm = _rms_norm(torch, q.view(m, HEADS, HEAD_DIM), None)
    cos = cos_cache[positions].repeat_interleave(2, dim=-1)
    sin = sin_cache[positions].repeat_interleave(2, dim=-1)
    q_out = q_norm.clone()
    q_out[..., NOPE_DIM:] = _rope(
        torch,
        q_norm[..., NOPE_DIM:],
        cos[:, None, :],
        sin[:, None, :],
    )
    q_out = q_out.to(torch.bfloat16)

    kv_norm = _rms_norm(torch, kv, kv_weight)
    kv_out = kv_norm.clone()
    kv_out[..., NOPE_DIM:] = _rope(
        torch,
        kv_norm[..., NOPE_DIM:],
        cos,
        sin,
    )
    kv_out = kv_out.to(torch.bfloat16)

    cache_out = cache.clone()
    fp8_max = torch.finfo(torch.float8_e4m3fnuz).max
    for token in range(m):
        location = int(locations[token].item())
        page, offset = divmod(location, PAGE_SIZE)
        value_base = offset * VALUE_BYTES
        scale_base = PAGE_SIZE * VALUE_BYTES + offset * SCALE_BYTES
        for tile in range(NOPE_DIM // 64):
            values = kv_out[token, tile * 64 : (tile + 1) * 64].float()
            maximum = values.abs().max().clamp_min(1.0e-8)
            exponent = torch.ceil(torch.log2(maximum / fp8_max))
            scale = torch.exp2(exponent)
            # The Triton kernel stores through ``tl.float8e4nv`` and bitcasts
            # the result to bytes. On HIP that storage encoding is OCP E4M3
            # even though the scale bound is the gfx942 FNUZ maximum (240).
            quantized = (values / scale).clamp(-fp8_max, fp8_max).to(
                torch.float8_e4m3fn
            )
            cache_out[
                page,
                value_base + tile * 64 : value_base + (tile + 1) * 64,
            ] = quantized.view(torch.uint8)
            cache_out[page, scale_base + tile] = (exponent.to(torch.int32) + 127).to(
                torch.uint8
            )
        rope_bytes = (
            kv_out[token, NOPE_DIM:].contiguous().view(torch.uint8).reshape(-1)
        )
        cache_out[
            page,
            value_base + NOPE_DIM : value_base + VALUE_BYTES,
        ] = rope_bytes
    return q_out, kv_out, cache_out


def _state_views(kv, cache):
    return {
        "kv_head": kv[: min(2, kv.shape[0]), :64],
        "cache_value_head": cache[:1, :2048],
        "cache_scale_head": cache[
            :1, PAGE_SIZE * VALUE_BYTES : PAGE_SIZE * VALUE_BYTES + 32
        ],
    }


class DeepSeekV4FusedQkNormRopeStoreSpec:
    operator_id = "deepseek_v4_fused_qk_norm_rope_store"

    def cases(self):
        return (
            CaseSpec(
                "prefill_smoke_m4",
                {"phase": "prefill", "tokens": 4, "heads": HEADS},
                4201,
                frozenset({"smoke", "oracle", "deepseek_v4_prefill"}),
                600,
            ),
            CaseSpec(
                "decode_smoke_m2",
                {"phase": "decode", "tokens": 2, "heads": HEADS},
                4202,
                frozenset({"smoke", "oracle", "deepseek_v4_decode"}),
                600,
            ),
            CaseSpec(
                "prefill_representative_m256",
                {"phase": "prefill", "tokens": 256, "heads": HEADS},
                4203,
                frozenset(
                    {"representative", "performance_only", "deepseek_v4_prefill"}
                ),
                1800,
            ),
            CaseSpec(
                "decode_representative_m64",
                {"phase": "decode", "tokens": 64, "heads": HEADS},
                4204,
                frozenset(
                    {"representative", "performance_only", "deepseek_v4_decode"}
                ),
                1800,
            ),
        )

    def make_inputs(self, case, context):
        torch = importlib.import_module("torch")
        tokens = int(case.symbols["tokens"])
        device = next(iter(context.cuda), "cuda:0")
        generator = context.cuda.get(device)
        if generator is None:
            raise RuntimeError("fused Q/K norm+RoPE requires a GPU generator")
        q = torch.randn(
            (tokens, HEADS * HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        kv = torch.randn(
            (tokens, HEAD_DIM),
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        )
        kv_weight = (
            1.0
            + 0.05
            * torch.randn(
                (HEAD_DIM,), dtype=torch.float32, device=device, generator=generator
            )
        ).to(torch.bfloat16)
        positions = torch.arange(tokens, dtype=torch.int32, device=device) + 17
        frequencies = torch.exp(
            -torch.arange(0, ROPE_DIM, 2, dtype=torch.float32, device=device)
            * (importlib.import_module("math").log(10000.0) / ROPE_DIM)
        )
        angles = torch.arange(
            tokens + 32, dtype=torch.float32, device=device
        )[:, None] * frequencies[None, :]
        cos_cache, sin_cache = angles.cos(), angles.sin()
        pages = max(1, (tokens + PAGE_SIZE - 1) // PAGE_SIZE)
        cache = torch.zeros(
            (pages, PAGE_SIZE * BYTES_PER_TOKEN),
            dtype=torch.uint8,
            device=device,
        )
        locations = torch.arange(tokens, dtype=torch.int32, device=device)
        q_out = torch.empty(
            (tokens, HEADS, HEAD_DIM), dtype=torch.bfloat16, device=device
        )
        observed = _state_views(kv, cache)
        if "performance_only" not in case.tags:
            oracle_q, oracle_kv, oracle_cache = _semantic_oracle(
                torch,
                q,
                kv,
                kv_weight,
                cos_cache,
                sin_cache,
                positions,
                cache,
                locations,
            )
            observed.update(
                {
                    "semantic_q": sample_tensor(oracle_q),
                    "semantic_kv_head": oracle_kv[: min(2, tokens), :64],
                    "semantic_cache_value_head": oracle_cache[:1, :2048],
                    "semantic_cache_scale_head": oracle_cache[
                        :1,
                        PAGE_SIZE * VALUE_BYTES : PAGE_SIZE * VALUE_BYTES + 32,
                    ],
                }
            )
        return InputBundle(
            args=(
                q,
                kv,
                None,
                kv_weight,
                cos_cache,
                sin_cache,
                positions,
                cache,
                locations,
                q_out,
            ),
            observed_state=observed,
        )

    def clone_inputs(self, inputs):
        (
            q,
            kv,
            q_weight,
            kv_weight,
            cos,
            sin,
            positions,
            cache,
            locations,
            q_out,
        ) = inputs.args
        q2, kv2, cache2 = q.clone(), kv.clone(), cache.clone()
        observed = _state_views(kv2, cache2)
        for key, value in inputs.observed_state.items():
            if key.startswith("semantic_"):
                observed[key] = value.clone()
        return InputBundle(
            args=(
                q2,
                kv2,
                q_weight,
                kv_weight.clone(),
                cos.clone(),
                sin.clone(),
                positions.clone(),
                cache2,
                locations.clone(),
                q_out.clone(),
            ),
            observed_state=observed,
        )

    def normalize_output(self, output):
        return normalize_named_tensors(output)

    def comparator(self, case):
        oracle = "performance_only" not in case.tags
        numeric = [
            NumericPath("output", 0.03, 0.01, "state.semantic_q"),
            NumericPath("state.kv_head", 0.03, 0.01, "state.semantic_kv_head"),
            NumericPath(
                "state.cache_value_head",
                0.0,
                0.0,
                "state.semantic_cache_value_head",
            ),
            NumericPath(
                "state.cache_scale_head",
                0.0,
                0.0,
                "state.semantic_cache_scale_head",
            ),
        ]
        return OracleStateComparator(
            numeric_paths=numeric,
            require_oracle=oracle,
            name="fused_qk_norm_rope_store_oracle",
        )

    def cost_model(self, case):
        tokens = int(case.symbols["tokens"])
        norm_flops = tokens * (HEADS + 1) * HEAD_DIM * 5
        rope_flops = tokens * (HEADS + 1) * ROPE_DIM * 4
        return {
            "flops": norm_flops + rope_flops,
            "estimated_bytes": (
                2 * tokens * HEADS * HEAD_DIM
                + 4 * tokens * HEADS * HEAD_DIM
                + 2 * tokens * HEAD_DIM
                + tokens * BYTES_PER_TOKEN
            ),
            "throughput_units": tokens * (HEADS + 1),
        }

    def layout_contract(self, case):
        tokens = int(case.symbols["tokens"])
        return {
            "q": [tokens, HEADS * HEAD_DIM],
            "q_out": [tokens, HEADS, HEAD_DIM],
            "kv_inplace": [tokens, HEAD_DIM],
            "swa_cache": ["pages", PAGE_SIZE * BYTES_PER_TOKEN],
            "cache_token_layout": "448 FP8 + 64 BF16 + 7 UE8M0 + 1 pad",
            "rope_style": "GPT-J pairwise",
            "backend": "SGLang Triton fused_qk_norm_rope_swa_store",
        }


SPEC = DeepSeekV4FusedQkNormRopeStoreSpec()


def cost_model(case):
    return SPEC.cost_model(case)
