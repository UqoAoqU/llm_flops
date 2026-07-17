"""Deterministic statistics computed from retained raw samples."""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass
from typing import Iterable


def _finite_samples(values: Iterable[float]) -> tuple[float, ...]:
    result: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError("samples must contain numbers")
        number = float(value)
        if not math.isfinite(number) or number < 0:
            raise ValueError("samples must be finite and non-negative")
        result.append(number)
    if not result:
        raise ValueError("at least one sample is required")
    return tuple(result)


def percentile(values: Iterable[float], quantile: float) -> float:
    """Return a linearly interpolated percentile (Hyndman-Fan type 7)."""

    samples = sorted(_finite_samples(values))
    if isinstance(quantile, bool) or not isinstance(quantile, (int, float)):
        raise TypeError("quantile must be a number")
    q = float(quantile)
    if not math.isfinite(q) or q < 0 or q > 1:
        raise ValueError("quantile must be between zero and one")
    if len(samples) == 1:
        return samples[0]
    position = (len(samples) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return samples[lower]
    fraction = position - lower
    return samples[lower] + (samples[upper] - samples[lower]) * fraction


@dataclass(frozen=True)
class SampleStatistics:
    count: int
    mean_ms: float
    median_ms: float
    min_ms: float
    max_ms: float
    stddev_ms: float
    cv: float
    p50_ms: float
    p90_ms: float
    p95_ms: float
    p99_ms: float
    unstable: bool
    instability_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "count": self.count,
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "stddev_ms": self.stddev_ms,
            "cv": self.cv,
            "p50_ms": self.p50_ms,
            "p90_ms": self.p90_ms,
            "p95_ms": self.p95_ms,
            "p99_ms": self.p99_ms,
            "unstable": self.unstable,
            "instability_reason": self.instability_reason,
            "stddev_kind": "population",
            "percentile_method": "linear_type7",
        }


def compute_statistics(
    values: Iterable[float],
    *,
    minimum_stable_samples: int = 5,
    maximum_cv: float = 0.1,
) -> SampleStatistics:
    samples = _finite_samples(values)
    if isinstance(minimum_stable_samples, bool) or not isinstance(minimum_stable_samples, int):
        raise TypeError("minimum_stable_samples must be an integer")
    if minimum_stable_samples <= 0:
        raise ValueError("minimum_stable_samples must be positive")
    if isinstance(maximum_cv, bool) or not isinstance(maximum_cv, (int, float)):
        raise TypeError("maximum_cv must be a number")
    threshold = float(maximum_cv)
    if not math.isfinite(threshold) or threshold < 0:
        raise ValueError("maximum_cv must be finite and non-negative")
    mean = statistics.fmean(samples)
    stddev = statistics.pstdev(samples)
    cv = 0.0 if mean == 0.0 and stddev == 0.0 else (math.inf if mean == 0.0 else stddev / mean)
    reasons: list[str] = []
    if len(samples) < minimum_stable_samples:
        reasons.append(f"sample_count<{minimum_stable_samples}")
    if cv > threshold:
        reasons.append(f"cv>{threshold:g}")
    return SampleStatistics(
        count=len(samples),
        mean_ms=mean,
        median_ms=statistics.median(samples),
        min_ms=min(samples),
        max_ms=max(samples),
        stddev_ms=stddev,
        cv=cv,
        p50_ms=percentile(samples, 0.50),
        p90_ms=percentile(samples, 0.90),
        p95_ms=percentile(samples, 0.95),
        p99_ms=percentile(samples, 0.99),
        unstable=bool(reasons),
        instability_reason=", ".join(reasons) if reasons else None,
    )


__all__ = ["SampleStatistics", "compute_statistics", "percentile"]
