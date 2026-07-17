"""Reference-owned DeepSeek V4 FP8/MXFP8 routed-MoE contract.

The module is registry-safe: torch, FlashInfer and SGLang are imported only by
runtime input/clone helpers inside the isolated worker.
"""

import importlib
import math

from benchmark_engine.correctness import FloatingComparator, InputBundle, Tolerance
from benchmark_engine.correctness.inputs import assert_input_isolation
from benchmark_engine.correctness.models import OutputBundle, OutputLeaf
from benchmark_engine.models import CaseSpec


HIDDEN = 7168
INTERMEDIATE = 3072
GLOBAL_EXPERTS = 384
LOCAL_EXPERTS = 16
LOCAL_EXPERT_OFFSET = 0
TOPK = 6
WEIGHT_SCALE_BLOCK = 32
WEIGHT_DTYPE = "float8_e4m3fn"
WEIGHT_SCALE_DTYPE = "float8_e8m0fnu-as-uint8"
ACTIVATION_DTYPE = "bfloat16"
ACTIVATION_QUANTIZATION = "mxfp8-cute-dsl"
OUTPUT_DTYPE = "bfloat16"
TOLERANCE = Tolerance(rtol=0.03, atol=0.2, source="flashinfer_trtllm_fp8_mxfp8")
PREFILL_M_VALUES = (1024, 2048, 4096)
DECODE_M_VALUES = (16, 32)

LEGACY_MAPPING = {
    "adapter_name": "Routed Expert Fused MoE",
    "backend": "FlashInfer TRTLLM FP8 weight + MXFP8 activation",
    "instances": 61,
    "kind": "moe_fp8_mxfp8",
    "shape": (LOCAL_EXPERTS, HIDDEN, INTERMEDIATE),
    "global_experts": GLOBAL_EXPERTS,
    "topk": TOPK,
}


def local_routed_pairs(m):
    """Legacy uniform-routing count owned by EP rank zero."""

    if isinstance(m, bool) or not isinstance(m, int) or m < 0:
        raise ValueError("m must be a non-negative integer")
    return m * TOPK * LOCAL_EXPERTS // GLOBAL_EXPERTS


def _projection_cases():
    cases = []
    for phase, values in (("prefill", PREFILL_M_VALUES), ("decode", DECODE_M_VALUES)):
        for m in values:
            cases.append(
                CaseSpec(
                    f"{phase}__routed_expert_fused_moe__m{m}__ctx65536__fp8_mxfp8",
                    {
                        "m": m,
                        "hidden": HIDDEN,
                        "intermediate": INTERMEDIATE,
                        "global_experts": GLOBAL_EXPERTS,
                        "local_experts": LOCAL_EXPERTS,
                        "local_expert_offset": LOCAL_EXPERT_OFFSET,
                        "topk": TOPK,
                        "weight_scale_block": WEIGHT_SCALE_BLOCK,
                        "weight_layout": "sglang_shuffled_trtllm_block_fp8",
                        "weight_dtype": WEIGHT_DTYPE,
                        "weight_scale_dtype": WEIGHT_SCALE_DTYPE,
                        "activation_dtype": ACTIVATION_DTYPE,
                        "activation_quantization": ACTIVATION_QUANTIZATION,
                        "output_dtype": OUTPUT_DTYPE,
                        "route_pattern": "legacy_uniform_ep0",
                        "phase": phase,
                        "quant_profile": "fp8_mxfp8",
                        "raw_context": 65536,
                        "model_input": m,
                        "projection_adapter_id": "routed_expert_fused_moe",
                    },
                    271,
                    frozenset(
                        {
                            "model_projection",
                            f"deepseek_v4_{phase}",
                            f"phase_{phase}",
                            "quant_profile_fp8_mxfp8",
                            f"m_{m}",
                            "context_65536",
                            "representative",
                            "performance_only",
                        }
                    ),
                    1800,
                )
            )
    return tuple(cases)


def _base_symbols(m, route_pattern):
    return {
        "m": m,
        "hidden": HIDDEN,
        "intermediate": INTERMEDIATE,
        "global_experts": GLOBAL_EXPERTS,
        "local_experts": LOCAL_EXPERTS,
        "local_expert_offset": LOCAL_EXPERT_OFFSET,
        "topk": TOPK,
        "weight_scale_block": WEIGHT_SCALE_BLOCK,
        "weight_layout": "sglang_shuffled_trtllm_block_fp8",
        "weight_dtype": WEIGHT_DTYPE,
        "weight_scale_dtype": WEIGHT_SCALE_DTYPE,
        "activation_dtype": ACTIVATION_DTYPE,
        "activation_quantization": ACTIVATION_QUANTIZATION,
        "output_dtype": OUTPUT_DTYPE,
        "route_pattern": route_pattern,
    }


def _validate_symbols(symbols):
    integer_fields = (
        "m",
        "hidden",
        "intermediate",
        "global_experts",
        "local_experts",
        "local_expert_offset",
        "topk",
        "weight_scale_block",
    )
    values = {}
    for name in integer_fields:
        value = symbols.get(name)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{name} must be an integer")
        values[name] = value
    if values["m"] <= 0:
        raise ValueError("m must be positive")
    expected = {
        "hidden": HIDDEN,
        "intermediate": INTERMEDIATE,
        "global_experts": GLOBAL_EXPERTS,
        "local_experts": LOCAL_EXPERTS,
        "local_expert_offset": LOCAL_EXPERT_OFFSET,
        "topk": TOPK,
        "weight_scale_block": WEIGHT_SCALE_BLOCK,
    }
    for name, target in expected.items():
        if values[name] != target:
            raise NotImplementedError(f"unsupported {name}={values[name]}; expected {target}")
    expected_strings = {
        "weight_layout": "sglang_shuffled_trtllm_block_fp8",
        "weight_dtype": WEIGHT_DTYPE,
        "weight_scale_dtype": WEIGHT_SCALE_DTYPE,
        "activation_dtype": ACTIVATION_DTYPE,
        "activation_quantization": ACTIVATION_QUANTIZATION,
        "output_dtype": OUTPUT_DTYPE,
    }
    for name, target in expected_strings.items():
        if symbols.get(name) != target:
            raise NotImplementedError(
                f"unsupported {name}={symbols.get(name)!r}; expected {target!r}"
            )
    pattern = symbols.get("route_pattern")
    if pattern not in {"no_local", "repeated_local", "legacy_uniform_ep0"}:
        raise ValueError(f"unknown route_pattern: {pattern!r}")
    return values["m"], str(pattern)


def _routing_values(m, pattern):
    """Pure-Python routing fixture used by CPU tests and CUDA input creation."""

    total = m * TOPK
    if pattern == "no_local":
        flat = [LOCAL_EXPERTS + (index % (GLOBAL_EXPERTS - LOCAL_EXPERTS)) for index in range(total)]
    elif pattern == "repeated_local":
        flat = [LOCAL_EXPERT_OFFSET] * total
    elif pattern == "legacy_uniform_ep0":
        flat = [LOCAL_EXPERTS + (index % (GLOBAL_EXPERTS - LOCAL_EXPERTS)) for index in range(total)]
        owned = local_routed_pairs(m)
        for index in range(owned):
            position = index * total // owned
            flat[position] = LOCAL_EXPERT_OFFSET + index % LOCAL_EXPERTS
    else:
        raise ValueError(f"unknown route_pattern: {pattern!r}")
    ids = tuple(tuple(flat[row * TOPK : (row + 1) * TOPK]) for row in range(m))
    weights = tuple(tuple(1.0 / TOPK for _ in range(TOPK)) for _ in range(m))
    return ids, weights


def _validate_routing_values(ids, weights, *, global_experts=GLOBAL_EXPERTS, topk=TOPK):
    if len(ids) != len(weights) or not ids:
        raise ValueError("routing ids/weights must have the same non-empty row count")
    for row, (id_row, weight_row) in enumerate(zip(ids, weights)):
        if len(id_row) != topk or len(weight_row) != topk:
            raise ValueError(f"routing row {row} must have topk={topk} entries")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in id_row):
            raise ValueError(f"routing ids in row {row} must be integers")
        if any(value < 0 or value >= global_experts for value in id_row):
            raise ValueError(f"routing id in row {row} is outside [0, {global_experts})")
        if any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) < 0.0
            for value in weight_row
        ):
            raise ValueError(f"routing weights in row {row} must be finite and non-negative")
        if not math.isclose(sum(float(value) for value in weight_row), 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(f"routing weights in row {row} must sum to one")


def _routed_mlp_oracle(torch, hidden_states, ids, weights, w13, w2, *, local_offset=0):
    """Small, auditable local-EP routed SwiGLU MLP oracle for CPU tests."""

    if hidden_states.ndim != 2 or ids.ndim != 2 or weights.shape != ids.shape:
        raise ValueError("oracle expects x[M,H] and matching ids/weights[M,K]")
    local_experts, twice_intermediate, hidden = w13.shape
    if twice_intermediate % 2 or w2.shape != (local_experts, hidden, twice_intermediate // 2):
        raise ValueError("oracle weight layout must be w13[E,2I,H], w2[E,H,I]")
    if hidden_states.shape[0] != ids.shape[0] or hidden_states.shape[1] != hidden:
        raise ValueError("oracle activation shape does not match routing/weights")
    result = torch.zeros_like(hidden_states, dtype=torch.float32)
    intermediate = twice_intermediate // 2
    for token in range(hidden_states.shape[0]):
        x = hidden_states[token].float()
        for slot in range(ids.shape[1]):
            global_expert = int(ids[token, slot])
            local_expert = global_expert - local_offset
            if not 0 <= local_expert < local_experts:
                continue
            gate = torch.mv(w13[local_expert, :intermediate].float(), x)
            up = torch.mv(w13[local_expert, intermediate:].float(), x)
            activated = torch.nn.functional.silu(gate) * up
            result[token] += float(weights[token, slot]) * torch.mv(
                w2[local_expert].float(), activated
            )
    return result


def _sample_indices(torch, total, device, limit=256):
    total = int(total)
    count = min(total, int(limit))
    if count == 0:
        return torch.empty(0, device=device, dtype=torch.long)
    if count == 1:
        return torch.zeros(1, device=device, dtype=torch.long)
    positions = torch.arange(count, device=device, dtype=torch.long)
    span, intervals = total - 1, count - 1
    return positions * (span // intervals) + positions * (span % intervals) // intervals


def _bounded_output(output):
    torch = importlib.import_module("torch")
    flat = output.detach().reshape(-1)
    indices = _sample_indices(torch, flat.numel(), flat.device)
    values = tuple(flat[indices].float().cpu().tolist())
    return OutputBundle(
        (
            OutputLeaf(
                "output",
                values,
                str(output.dtype),
                tuple(output.shape),
                tuple(output.stride()),
                "strided",
                str(output.device),
            ),
        )
    )


def _check_available_memory(required_bytes, available_bytes):
    if required_bytes > int(available_bytes * 0.9):
        raise MemoryError(
            f"estimated routed-MoE allocation {required_bytes} exceeds available CUDA memory {available_bytes}"
        )


def _validate_logical_weight_layout(w13_shape, w13_scale_shape, w2_shape, w2_scale_shape):
    expected = (
        (LOCAL_EXPERTS, 2 * INTERMEDIATE, HIDDEN),
        (LOCAL_EXPERTS, 2 * INTERMEDIATE, HIDDEN // WEIGHT_SCALE_BLOCK),
        (LOCAL_EXPERTS, HIDDEN, INTERMEDIATE),
        (LOCAL_EXPERTS, HIDDEN, INTERMEDIATE // WEIGHT_SCALE_BLOCK),
    )
    actual = tuple(
        tuple(int(value) for value in shape)
        for shape in (w13_shape, w13_scale_shape, w2_shape, w2_scale_shape)
    )
    if actual != expected:
        raise ValueError(
            f"invalid FP8/UE8M0 logical weight layout: {actual}; expected {expected}"
        )


def _runtime_modules():
    torch = importlib.import_module("torch")
    try:
        # SGLang's quantization package imports the MoE runner while it is
        # initialized.  Resolve PackTopkIds first so the later runner import
        # observes a fully initialized quantization module rather than
        # circularly re-entering a partially initialized flashinfer_trtllm.
        pack_module = importlib.import_module(
            "sglang.srt.layers.quantization.mxfp4_flashinfer_trtllm_moe"
        )
        align_module = importlib.import_module(
            "sglang.srt.layers.moe.moe_runner.flashinfer_trtllm"
        )
    except (ImportError, OSError) as error:
        raise RuntimeError(f"SGLang FlashInfer MoE import/build error: {error}") from error
    align = getattr(align_module, "align_mxfp8_moe_weights_for_flashinfer_trtllm", None)
    pack_class = getattr(pack_module, "PackTopkIds", None)
    if align is None or pack_class is None or not hasattr(pack_class, "vanilla"):
        raise NotImplementedError(
            "unsupported: missing SGLang FlashInfer MoE alignment or PackTopkIds symbol"
        )
    return torch, align, pack_class.vanilla


def _logical_weight_bytes():
    weight_elements = LOCAL_EXPERTS * (2 * INTERMEDIATE * HIDDEN + HIDDEN * INTERMEDIATE)
    scale_elements = LOCAL_EXPERTS * (
        2 * INTERMEDIATE * (HIDDEN // WEIGHT_SCALE_BLOCK)
        + HIDDEN * (INTERMEDIATE // WEIGHT_SCALE_BLOCK)
    )
    return weight_elements + scale_elements


def _make_aligned_weights(torch, align, device):
    layer = torch.nn.Module()
    layer.w13_weight = torch.nn.Parameter(
        torch.zeros(
            LOCAL_EXPERTS,
            2 * INTERMEDIATE,
            HIDDEN,
            dtype=torch.float8_e4m3fn,
            device=device,
        ),
        requires_grad=False,
    )
    layer.w2_weight = torch.nn.Parameter(
        torch.zeros(
            LOCAL_EXPERTS,
            HIDDEN,
            INTERMEDIATE,
            dtype=torch.float8_e4m3fn,
            device=device,
        ),
        requires_grad=False,
    )
    # A sparse deterministic diagonal makes route/expert changes observable
    # without an expensive random initialization of the full 1 GiB fixture.
    diagonal = min(32, HIDDEN, INTERMEDIATE)
    positions = torch.arange(diagonal, device=device)
    for expert in range(LOCAL_EXPERTS):
        value = 0.25 * (1 + expert % 4)
        layer.w13_weight.data[expert, positions, positions] = value
        layer.w13_weight.data[expert, INTERMEDIATE + positions, positions] = value
        layer.w2_weight.data[expert, positions, positions] = value
    layer.w13_weight_scale_inv = torch.nn.Parameter(
        torch.ones(
            LOCAL_EXPERTS,
            2 * INTERMEDIATE,
            HIDDEN // WEIGHT_SCALE_BLOCK,
            dtype=torch.float8_e8m0fnu,
            device=device,
        ).view(torch.uint8),
        requires_grad=False,
    )
    layer.w2_weight_scale_inv = torch.nn.Parameter(
        torch.ones(
            LOCAL_EXPERTS,
            HIDDEN,
            INTERMEDIATE // WEIGHT_SCALE_BLOCK,
            dtype=torch.float8_e8m0fnu,
            device=device,
        ).view(torch.uint8),
        requires_grad=False,
    )
    _validate_logical_weight_layout(
        layer.w13_weight.shape,
        layer.w13_weight_scale_inv.shape,
        layer.w2_weight.shape,
        layer.w2_weight_scale_inv.shape,
    )
    align(layer)
    return (
        layer.w13_weight,
        layer.w13_weight_scale_inv,
        layer.w2_weight,
        layer.w2_weight_scale_inv,
    )


def _observed_state(ids, weights, w13, w13_scale, w2, w2_scale):
    return {
        "routing_ids_head": ids[:8],
        "routing_ids_tail": ids[-8:],
        "routing_weights_head": weights[:8],
        "routing_weights_tail": weights[-8:],
        "w13_head": w13[:2, :1, :16],
        "w13_scale_head": w13_scale[:2, :1, :1],
        "w2_head": w2[:2, :16, :1],
        "w2_scale_head": w2_scale[:2, :1, :1],
    }


def _clone_with_packer(inputs, packer):
    hidden_states, _, ids, weights, w13, w13_scale, w2, w2_scale, output, tune = inputs.args
    ids = ids.clone()
    weights = weights.clone()
    w13_clone = w13.clone()
    w13_scale_clone = w13_scale.clone()
    w2_clone = w2.clone()
    w2_scale_clone = w2_scale.clone()
    return InputBundle(
        args=(
            hidden_states.clone(),
            packer(ids, weights),
            ids,
            weights,
            w13_clone,
            w13_scale_clone,
            w2_clone,
            w2_scale_clone,
            output.clone(),
            tune,
        ),
        observed_state=_observed_state(
            ids, weights, w13_clone, w13_scale_clone, w2_clone, w2_scale_clone
        ),
    )


class DeepSeekV4TrtllmFp8Mxfp8MoeSpec:
    operator_id = "deepseek_v4_trtllm_fp8_mxfp8_moe"

    def cases(self):
        return (
            CaseSpec(
                "smoke_m1_no_local",
                _base_symbols(1, "no_local"),
                277,
                frozenset({"smoke", "boundary", "no_local", "decode"}),
                1800,
            ),
            CaseSpec(
                "boundary_m1_repeated_local",
                _base_symbols(1, "repeated_local"),
                281,
                frozenset({"boundary", "repeated_expert", "decode"}),
                1800,
            ),
            CaseSpec(
                "boundary_m4_uniform_ep0",
                _base_symbols(4, "legacy_uniform_ep0"),
                283,
                frozenset({"boundary", "local_pair", "decode"}),
                1800,
            ),
        ) + _projection_cases()

    def legacy_mappings(self):
        return LEGACY_MAPPING

    def layout_contract(self, case):
        m, pattern = _validate_symbols(case.symbols)
        return {
            "hidden_states": [m, HIDDEN],
            "routing_ids": [m, TOPK],
            "routing_weights": [m, TOPK],
            "routing_weights_normalized": True,
            "route_pattern": pattern,
            "local_routed_pairs": local_routed_pairs(m) if pattern == "legacy_uniform_ep0" else None,
            "w13_logical": [LOCAL_EXPERTS, 2 * INTERMEDIATE, HIDDEN],
            "w2_logical": [LOCAL_EXPERTS, HIDDEN, INTERMEDIATE],
            "weight_layout": "sglang_shuffled_trtllm_block_fp8",
            "weight_dtype": WEIGHT_DTYPE,
            "weight_scale_block": WEIGHT_SCALE_BLOCK,
            "weight_scale_dtype": WEIGHT_SCALE_DTYPE,
            "activation_quantization": ACTIVATION_QUANTIZATION,
            "output": [m, HIDDEN],
            "output_dtype": OUTPUT_DTYPE,
            "observed_state": [
                "routing_ids_head",
                "routing_ids_tail",
                "routing_weights_head",
                "routing_weights_tail",
                "w13_head",
                "w13_scale_head",
                "w2_head",
                "w2_scale_head",
            ],
        }

    def make_inputs(self, case, context):
        m, pattern = _validate_symbols(case.symbols)
        ids_values, weight_values = _routing_values(m, pattern)
        _validate_routing_values(ids_values, weight_values)
        torch, align, packer = _runtime_modules()
        device = next(iter(context.cuda), "cuda:0")
        generator = context.cuda.get(device)
        if generator is None:
            raise RuntimeError("deepseek_v4_trtllm_fp8_mxfp8_moe requires a CUDA generator")
        free, _ = torch.cuda.mem_get_info(device)
        # Canonical + two correctness clones + two performance clones remain
        # comfortably bounded on B200, while smaller devices fail explicitly.
        required = 5 * (_logical_weight_bytes() + 2 * m * HIDDEN) + m * TOPK * 8
        _check_available_memory(required, free)
        hidden_states = torch.randn(
            (m, HIDDEN), dtype=torch.bfloat16, device=device, generator=generator
        ).clamp_(-2.0, 2.0)
        ids = torch.tensor(ids_values, dtype=torch.int32, device=device).contiguous()
        weights = torch.tensor(weight_values, dtype=torch.float32, device=device).contiguous()
        if ids.shape != (m, TOPK) or weights.shape != (m, TOPK):
            raise ValueError("routing ids/weights do not match [M, topk]")
        if int(ids.min()) < 0 or int(ids.max()) >= GLOBAL_EXPERTS:
            raise ValueError("routing ids are outside the global expert range")
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("routing weights must be finite and non-negative")
        if not torch.allclose(weights.sum(dim=1), torch.ones(m, device=device), atol=1e-6, rtol=0.0):
            raise ValueError("routing weights must sum to one")
        w13, w13_scale, w2, w2_scale = _make_aligned_weights(torch, align, device)
        output = torch.empty((m, HIDDEN), dtype=torch.bfloat16, device=device)
        tune = 1 << (m - 1).bit_length()
        return InputBundle(
            args=(hidden_states, packer(ids, weights), ids, weights, w13, w13_scale, w2, w2_scale, output, tune),
            observed_state=_observed_state(ids, weights, w13, w13_scale, w2, w2_scale),
        )

    def clone_inputs(self, inputs):
        _, _, packer = _runtime_modules()
        clone = _clone_with_packer(inputs, packer)
        # Keep the public contract honest even if a backend packer changes its
        # wrapper type in a future SGLang release.
        assert_input_isolation(inputs, clone)
        return clone

    def normalize_output(self, output):
        if isinstance(output, (tuple, list)):
            if not output:
                raise RuntimeError("FlashInfer routed MoE returned an empty output sequence")
            output = output[0]
        if not hasattr(output, "detach") or not hasattr(output, "shape"):
            raise TypeError("FlashInfer routed MoE output must be a tensor")
        if tuple(output.shape)[-1:] != (HIDDEN,):
            raise ValueError(f"FlashInfer routed MoE output has invalid shape {tuple(output.shape)}")
        if str(output.dtype) not in {"torch.bfloat16", "bfloat16"}:
            raise ValueError(f"FlashInfer routed MoE output has invalid dtype {output.dtype}")
        return _bounded_output(output)

    def comparator(self, case):
        _validate_symbols(case.symbols)
        return FloatingComparator(operator_default=TOLERANCE, require_explicit=True)

    def cost_model(self, case):
        m, pattern = _validate_symbols(case.symbols)
        if pattern == "no_local":
            pairs = 0
        elif pattern == "repeated_local":
            pairs = m * TOPK
        else:
            pairs = local_routed_pairs(m)
        if pairs == 0:
            return None
        flops = pairs * (6 * HIDDEN * INTERMEDIATE + 2 * HIDDEN)
        estimated_bytes = (
            m * HIDDEN * 2
            + _logical_weight_bytes()
            + m * TOPK * (4 + 4)
            + m * HIDDEN * 2
        )
        return {
            "flops": flops,
            "estimated_bytes": estimated_bytes,
            "throughput_units": pairs,
        }

    def workspace_bytes(self, case):
        _validate_symbols(case.symbols)
        return None


SPEC = DeepSeekV4TrtllmFp8Mxfp8MoeSpec()


def cost_model(case):
    return SPEC.cost_model(case)
