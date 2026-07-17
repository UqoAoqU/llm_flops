"""Theoretical operator cost estimates and latency-derived rates."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass


def _positive(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{field} must be finite and positive")
    return number


@dataclass(frozen=True)
class TheoreticalCost:
    flops: float
    estimated_bytes: float
    throughput_units: float | None = None

    def __post_init__(self) -> None:
        _positive(self.flops, "flops")
        _positive(self.estimated_bytes, "estimated_bytes")
        if self.throughput_units is not None:
            _positive(self.throughput_units, "throughput_units")


@dataclass(frozen=True)
class CostMetrics:
    available: bool
    reason: str | None
    theoretical: bool
    flops: float | None = None
    estimated_bytes: float | None = None
    tflops: float | None = None
    effective_bandwidth_gbps: float | None = None
    arithmetic_intensity: float | None = None
    throughput: float | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "reason": self.reason,
            "theoretical": self.theoretical,
            "flops": self.flops,
            "estimated_bytes": self.estimated_bytes,
            "tflops": self.tflops,
            "effective_bandwidth_gbps": self.effective_bandwidth_gbps,
            "arithmetic_intensity": self.arithmetic_intensity,
            "throughput": self.throughput,
        }


def _normalise_cost(value: object) -> TheoreticalCost | None:
    if value is None:
        return None
    if isinstance(value, TheoreticalCost):
        return value
    if isinstance(value, Mapping):
        unknown = set(value).difference({"flops", "bytes", "estimated_bytes", "throughput_units"})
        if unknown:
            raise ValueError("cost model has unknown fields: " + ", ".join(sorted(map(str, unknown))))
        if "flops" not in value:
            raise ValueError("cost model must provide flops")
        if ("bytes" in value) == ("estimated_bytes" in value):
            raise ValueError("cost model must provide exactly one of bytes or estimated_bytes")
        return TheoreticalCost(
            flops=_positive(value["flops"], "flops"),
            estimated_bytes=_positive(value.get("estimated_bytes", value.get("bytes")), "estimated_bytes"),
            throughput_units=(
                None
                if value.get("throughput_units") is None
                else _positive(value["throughput_units"], "throughput_units")
            ),
        )
    raise TypeError("cost model must return None, TheoreticalCost, or a mapping")


def evaluate_cost_model(value: object, latency_ms: float) -> CostMetrics:
    cost = _normalise_cost(value)
    if cost is None:
        # Absence of a cost model is authoritative.  In particular, a fake or
        # zero-duration timer must not turn an otherwise valid "unavailable"
        # result into a cost-model failure.
        return CostMetrics(False, "operator cost model returned unavailable", True)
    latency = _positive(latency_ms, "latency_ms")
    return CostMetrics(
        available=True,
        reason=None,
        theoretical=True,
        flops=float(cost.flops),
        estimated_bytes=float(cost.estimated_bytes),
        tflops=float(cost.flops) / (latency * 1.0e9),
        effective_bandwidth_gbps=float(cost.estimated_bytes) / (latency * 1.0e6),
        arithmetic_intensity=float(cost.flops) / float(cost.estimated_bytes),
        throughput=(
            None
            if cost.throughput_units is None
            else float(cost.throughput_units) * 1000.0 / latency
        ),
    )


__all__ = ["CostMetrics", "TheoreticalCost", "evaluate_cost_model"]
