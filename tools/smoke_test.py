#!/usr/bin/env python3
"""Run one minimal GPU case for each individual-operator family."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Callable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@dataclass(frozen=True)
class SmokeCase:
    name: str
    run: Callable[[], object]


def _dsa_indexer():
    import torch
    import dsa_indexer as benchmark

    benchmark.NUM_WARMUP = 1
    benchmark.NUM_RUNS = 1
    return benchmark.bench_cublas_fp8(16, 128, 128, torch.device("cuda:0"))


def _dsa_flashmla():
    import torch
    import dsa_flashmla as benchmark

    benchmark.NUM_WARMUP = 1
    benchmark.NUM_RUNS = 1
    return benchmark.bench_one(16, 512, 64, torch.device("cuda:0"))


def _dsa_projection():
    import torch
    import dsa_projection as benchmark

    benchmark.NUM_WARMUP = 1
    benchmark.NUM_RUNS = 1
    return benchmark.bench_gemm(16, 128, 128, torch.device("cuda:0"))


def _mla_flashmla():
    import torch
    from deepseek_v4_benchmark import _decode_attention_fn, graph_ms

    run = _decode_attention_fn(16, 512, 0, torch)
    return graph_ms(run, torch, warmup=1, runs=1)


def _moe_deepgemm():
    import torch
    import moe_deepgemm as benchmark

    benchmark.NUM_WARMUP = 1
    benchmark.NUM_RUNS = 1
    distribution = [1] + [0] * (benchmark.N_EXPERT - 1)
    return benchmark.bench_grouped_gemm(
        distribution, 128, 128, torch.device("cuda:0")
    )


def build_smoke_plan() -> tuple[SmokeCase, ...]:
    return (
        SmokeCase("dsa_indexer", _dsa_indexer),
        SmokeCase("dsa_flashmla", _dsa_flashmla),
        SmokeCase("dsa_projection", _dsa_projection),
        SmokeCase("mla_flashmla", _mla_flashmla),
        SmokeCase("moe_deepgemm", _moe_deepgemm),
    )


def main() -> int:
    failures = []
    for case in build_smoke_plan():
        print(f"[smoke] {case.name}: running", flush=True)
        try:
            case.run()
            print(f"[smoke] {case.name}: PASS", flush=True)
        except Exception as error:
            failures.append((case.name, error))
            print(f"[smoke] {case.name}: FAIL: {error}", flush=True)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
