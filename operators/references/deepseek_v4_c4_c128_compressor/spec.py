"""Stateful C4/C128 compressor contract for DeepSeek V4 Flash."""

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


def _last_dim(ratio):
    return HEAD_DIM * (4 if ratio == 4 else 2)


def _state_location(batch_index, position, ratio):
    if ratio == 4:
        page = batch_index * 2 + (position // 4) % 2
    else:
        page = batch_index
    return page, position % ratio


def _compress_one(torch, rows, ape, position, ratio):
    if ratio == 4:
        if position < 7:
            kv_overlap = torch.zeros(
                (4, HEAD_DIM), dtype=torch.float64, device=rows.device
            )
            score_overlap = torch.full(
                (4, HEAD_DIM),
                float("-inf"),
                dtype=torch.float64,
                device=rows.device,
            )
        else:
            kv_overlap = rows[
                position - 7 : position - 3, :HEAD_DIM
            ].double()
            score_overlap = rows[
                position - 7 : position - 3, 2 * HEAD_DIM : 3 * HEAD_DIM
            ].double()
        kv_fresh = rows[
            position - 3 : position + 1, HEAD_DIM : 2 * HEAD_DIM
        ].double()
        score_fresh = rows[
            position - 3 : position + 1, 3 * HEAD_DIM :
        ].double()
        values = torch.cat((kv_overlap, kv_fresh), dim=0)
        scores = torch.cat((score_overlap, score_fresh), dim=0)
    else:
        begin = position - ratio + 1
        values = rows[begin : position + 1, :HEAD_DIM].double()
        scores = rows[begin : position + 1, HEAD_DIM:].double()
    return (values * (scores + ape.double()).softmax(dim=0)).sum(dim=0).float()


def _semantic_prefill(torch, rows, ape, batch, seq_len, ratio, pool):
    expected_pool = pool.clone()
    outputs = []
    for batch_index in range(batch):
        batch_rows = rows[
            batch_index * seq_len : (batch_index + 1) * seq_len
        ]
        for position in range(seq_len):
            # Legacy C4 prefill retains its overlap/current rows in the ring.
            # A complete C128 prefill is consumed directly by plan_c and its
            # legacy planner emits an empty plan_w, so no persistent write is
            # part of that primitive invocation.
            if ratio == 4:
                page, slot = _state_location(batch_index, position, ratio)
                expected_pool[page, slot] = batch_rows[position]
            if (position + 1) % ratio == 0:
                outputs.append(
                    _compress_one(torch, batch_rows, ape, position, ratio)
                )
    return torch.stack(outputs), expected_pool


def _semantic_decode(torch, full_rows, ape, batch, ratio, pool):
    expected_pool = pool.clone()
    outputs = []
    for batch_index in range(batch):
        rows = full_rows[batch_index]
        position = ratio - 1
        page, slot = _state_location(batch_index, position, ratio)
        expected_pool[page, slot] = rows[position]
        outputs.append(_compress_one(torch, rows, ape, position, ratio))
    return torch.stack(outputs), expected_pool


def _pool_views(pool):
    width = min(128, pool.shape[-1])
    return {
        "pool_head": pool[:1, :1, :width],
        "pool_tail": pool[-1:, -1:, :width],
    }


class DeepSeekV4CompressorSpec:
    operator_id = "deepseek_v4_c4_c128_compressor"

    def cases(self):
        cases = []
        seed = 4300
        for phase in ("prefill", "decode"):
            for ratio in (4, 128):
                seed += 1
                symbols = {
                    "phase": phase,
                    "compression_ratio": ratio,
                    "batch": 1,
                    "tokens": ratio,
                    "head_dim": HEAD_DIM,
                }
                cases.append(
                    CaseSpec(
                        f"{phase}_smoke_c{ratio}",
                        symbols,
                        seed,
                        frozenset(
                            {"smoke", "oracle", f"deepseek_v4_{phase}", f"c{ratio}"}
                        ),
                        900,
                    )
                )
        cases.extend(
            (
                CaseSpec(
                    "prefill_representative_c4_t4096",
                    {
                        "phase": "prefill",
                        "compression_ratio": 4,
                        "batch": 1,
                        "tokens": 4096,
                        "head_dim": HEAD_DIM,
                    },
                    4310,
                    frozenset(
                        {
                            "representative",
                            "performance_only",
                            "deepseek_v4_prefill",
                            "c4",
                        }
                    ),
                    1800,
                ),
                CaseSpec(
                    "prefill_representative_c128_t4096",
                    {
                        "phase": "prefill",
                        "compression_ratio": 128,
                        "batch": 1,
                        "tokens": 4096,
                        "head_dim": HEAD_DIM,
                    },
                    4311,
                    frozenset(
                        {
                            "representative",
                            "performance_only",
                            "deepseek_v4_prefill",
                            "c128",
                        }
                    ),
                    1800,
                ),
                CaseSpec(
                    "decode_representative_c4_b128",
                    {
                        "phase": "decode",
                        "compression_ratio": 4,
                        "batch": 128,
                        "tokens": 1,
                        "head_dim": HEAD_DIM,
                    },
                    4312,
                    frozenset(
                        {
                            "representative",
                            "performance_only",
                            "deepseek_v4_decode",
                            "c4",
                        }
                    ),
                    1800,
                ),
                CaseSpec(
                    "decode_representative_c128_b128",
                    {
                        "phase": "decode",
                        "compression_ratio": 128,
                        "batch": 128,
                        "tokens": 1,
                        "head_dim": HEAD_DIM,
                    },
                    4313,
                    frozenset(
                        {
                            "representative",
                            "performance_only",
                            "deepseek_v4_decode",
                            "c128",
                        }
                    ),
                    1800,
                ),
            )
        )
        return tuple(cases)

    def make_inputs(self, case, context):
        torch = importlib.import_module("torch")
        dsv4 = importlib.import_module("sglang.jit_kernel.dsv4")
        ratio = int(case.symbols["compression_ratio"])
        batch = int(case.symbols["batch"])
        phase = str(case.symbols["phase"])
        device = next(iter(context.cuda), "cuda:0")
        generator = context.cuda.get(device)
        if generator is None:
            raise RuntimeError("C4/C128 compressor requires a GPU generator")
        pages = batch * (2 if ratio == 4 else 1)
        pool = torch.zeros(
            (pages, ratio, _last_dim(ratio)),
            dtype=torch.float32,
            device=device,
        )
        ape = torch.randn(
            (8 if ratio == 4 else 128, HEAD_DIM),
            dtype=torch.float32,
            device=device,
            generator=generator,
        )
        req_ids = torch.arange(batch, dtype=torch.int64, device=device)
        observed = _pool_views(pool)
        if phase == "prefill":
            seq_len = int(case.symbols["tokens"])
            rows = torch.randn(
                (batch * seq_len, _last_dim(ratio)),
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            seq_lens = torch.full((batch,), seq_len, dtype=torch.int64)
            extend_lens = seq_lens.clone()
            plan = dsv4.CompressorPrefillPlan.generate_legacy(
                ratio,
                req_ids,
                seq_lens,
                extend_lens,
                batch * seq_len,
                torch.device(device),
            )
            semantic_args = (rows, ape, batch, seq_len, ratio, pool)
            semantic_fn = _semantic_prefill
            kv_input = rows
        else:
            full_rows = torch.randn(
                (batch, ratio, _last_dim(ratio)),
                dtype=torch.float32,
                device=device,
                generator=generator,
            )
            for batch_index in range(batch):
                for position in range(ratio - 1):
                    page, slot = _state_location(batch_index, position, ratio)
                    pool[page, slot] = full_rows[batch_index, position]
            seq_lens = torch.full(
                (batch,), ratio, dtype=torch.int64, device=device
            )
            plan = dsv4.CompressorDecodePlan.generate_legacy(
                ratio, req_ids, seq_lens
            )
            semantic_args = (full_rows, ape, batch, ratio, pool)
            semantic_fn = _semantic_decode
            kv_input = full_rows[:, -1].contiguous()
        if "performance_only" not in case.tags:
            oracle, expected_pool = semantic_fn(torch, *semantic_args)
            observed.update(
                {
                    "semantic_oracle": sample_tensor(oracle),
                    "semantic_pool_head": expected_pool[
                        :1, :1, : min(128, expected_pool.shape[-1])
                    ],
                    "semantic_pool_tail": expected_pool[
                        -1:, -1:, : min(128, expected_pool.shape[-1])
                    ],
                }
            )
        return InputBundle(
            args=(pool, kv_input, ape, plan, ratio, HEAD_DIM),
            observed_state=observed,
        )

    def clone_inputs(self, inputs):
        dsv4 = importlib.import_module("sglang.jit_kernel.dsv4")
        pool, kv_input, ape, plan, ratio, head_dim = inputs.args
        pool2 = pool.clone()
        if plan.is_decode:
            plan2 = dsv4.CompressorDecodePlan(ratio, plan.plan_d.clone())
        else:
            plan2 = dsv4.CompressorPrefillPlan(
                ratio,
                plan.plan_c.clone(),
                plan.plan_w.clone(),
                None if plan.pin_buffer is None else plan.pin_buffer.clone(),
            )
        observed = _pool_views(pool2)
        for key, value in inputs.observed_state.items():
            if key.startswith("semantic_"):
                observed[key] = value.clone()
        return InputBundle(
            args=(
                pool2,
                kv_input.clone(),
                ape.clone(),
                plan2,
                ratio,
                head_dim,
            ),
            observed_state=observed,
        )

    def normalize_output(self, output):
        return normalize_named_tensors(output)

    def comparator(self, case):
        return OracleStateComparator(
            numeric_paths=(
                NumericPath(
                    "output", 0.005, 0.005, "state.semantic_oracle"
                ),
                NumericPath(
                    "state.pool_head",
                    0.005,
                    0.005,
                    "state.semantic_pool_head",
                ),
                NumericPath(
                    "state.pool_tail",
                    0.005,
                    0.005,
                    "state.semantic_pool_tail",
                ),
            ),
            require_oracle="performance_only" not in case.tags,
            name="c4_c128_compressor_oracle",
        )

    def cost_model(self, case):
        ratio = int(case.symbols["compression_ratio"])
        batch = int(case.symbols["batch"])
        phase = str(case.symbols["phase"])
        tokens = int(case.symbols["tokens"])
        events = batch if phase == "decode" else batch * tokens // ratio
        window = 8 if ratio == 4 else 128
        return {
            "flops": events * window * HEAD_DIM * 6,
            "estimated_bytes": (
                batch * tokens * _last_dim(ratio) * 4
                + events * HEAD_DIM * 4
                + window * HEAD_DIM * 4
            ),
            "throughput_units": events * HEAD_DIM,
        }

    def layout_contract(self, case):
        ratio = int(case.symbols["compression_ratio"])
        return {
            "compression_ratio": ratio,
            "state_row": _last_dim(ratio),
            "c4_row_layout": "kv_overlap|kv|score_overlap|score",
            "c128_row_layout": "kv|score",
            "state_mutation": "writes the per-request ring before pooling",
            "backend": "SGLang jit_kernel.dsv4.compress_forward",
        }


SPEC = DeepSeekV4CompressorSpec()


def cost_model(case):
    return SPEC.cost_model(case)
