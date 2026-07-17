"""Trusted Phase-9 performance eligibility and threshold gate."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class PerformanceGateConfig:
    max_slowdown_pct: float | None = 5.0
    min_speedup: float | None = None
    max_candidate_median_ms: float | None = None
    max_cv: float | None = 0.1
    max_memory_bytes: int | None = None
    unsupported_policy: str = "fail"

    def __post_init__(self) -> None:
        for name in ("max_slowdown_pct", "min_speedup", "max_candidate_median_ms", "max_cv"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(float(value)) or float(value) < 0):
                raise ValueError(f"{name} must be finite and non-negative or None")
        if self.max_memory_bytes is not None and (
            isinstance(self.max_memory_bytes, bool) or not isinstance(self.max_memory_bytes, int)
            or self.max_memory_bytes < 0
        ):
            raise ValueError("max_memory_bytes must be a non-negative integer or None")
        if self.unsupported_policy not in {"fail", "allow"}:
            raise ValueError("unsupported_policy must be fail or allow")


@dataclass(frozen=True)
class GateResult:
    status: str
    formal: bool
    ranking_eligible: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return {"status": self.status, "formal": self.formal,
                "ranking_eligible": self.ranking_eligible, "reasons": list(self.reasons)}


def _finite(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def evaluate_performance_gate(performance: Mapping[str, object], config: PerformanceGateConfig,
                              *, correctness_pass: bool,
                              perf_on_correctness_fail: bool = False) -> GateResult:
    status = performance.get("status")
    if status == "skipped":
        return GateResult("skipped", False, False,
                          (str(performance.get("reason") or "skipped"),))
    if not correctness_pass:
        return GateResult("failed", False, False, ("correctness_failed",))
    if status == "unsupported":
        allowed = config.unsupported_policy == "allow"
        return GateResult("passed" if allowed else "failed", False, False,
                          ("unsupported_allowed" if allowed else "unsupported",))
    if status not in {"pass", "unstable"}:
        return GateResult("failed", False, False,
                          (f"performance_{status or 'invalid'}",))
    measurements = performance.get("measurements")
    measurements = measurements if isinstance(measurements, Mapping) else {}
    ref = measurements.get("reference")
    cand = measurements.get("candidate")
    ref = ref if isinstance(ref, Mapping) else {}
    cand = cand if isinstance(cand, Mapping) else {}
    ref_sel = ref.get("selection") if isinstance(ref.get("selection"), Mapping) else {}
    cand_sel = cand.get("selection") if isinstance(cand.get("selection"), Mapping) else {}
    reasons: list[str] = []
    if status == "unstable":
        reasons.append("unstable")
    if ref_sel.get("effective_timer") != cand_sel.get("effective_timer"):
        reasons.append("effective_timer_mismatch")
    formal = bool(performance.get("formal", False)) and not perf_on_correctness_fail
    if not formal:
        reason = performance.get("non_formal_reason")
        reasons.append(str(reason) if isinstance(reason, str) and reason else "non_formal")
    speedup = _finite(performance.get("speedup"))
    slowdown = _finite(performance.get("slowdown_pct"))
    stats = cand.get("statistics") if isinstance(cand.get("statistics"), Mapping) else {}
    median = _finite(stats.get("median_ms"))
    cv = _finite(stats.get("cv"))
    memory = performance.get("peak_memory_allocated_bytes")
    if config.max_slowdown_pct is not None and (slowdown is None or slowdown > config.max_slowdown_pct):
        reasons.append("max_slowdown_exceeded")
    if config.min_speedup is not None and (speedup is None or speedup < config.min_speedup):
        reasons.append("min_speedup_not_met")
    if config.max_candidate_median_ms is not None and (median is None or median > config.max_candidate_median_ms):
        reasons.append("max_candidate_median_exceeded")
    if config.max_cv is not None and (cv is None or cv > config.max_cv):
        reasons.append("max_cv_exceeded")
    if config.max_memory_bytes is not None and (
        isinstance(memory, bool) or not isinstance(memory, int) or memory > config.max_memory_bytes
    ):
        reasons.append("max_memory_exceeded_or_unavailable")
    reasons = list(dict.fromkeys(reasons))
    failed = bool(reasons) or not formal
    return GateResult("failed" if failed else "passed", formal,
                      formal and not failed, tuple(reasons))


__all__ = ["GateResult", "PerformanceGateConfig", "evaluate_performance_gate"]
