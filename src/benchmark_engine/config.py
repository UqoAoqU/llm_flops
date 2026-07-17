"""Immutable, fully resolved worker configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import math


DEFAULT_OUTPUT_ROOT = Path("results")
DEFAULT_MODE = "all"


@dataclass(frozen=True)
class ResolvedEvaluationConfig:
    """All policy needed by a worker, with no unresolved global state."""

    mode: str = DEFAULT_MODE
    correctness_seeds: tuple[int, ...] = (0,)
    correctness_default_comparator: str = "floating"
    correctness_rtol: float = 0.0
    correctness_atol: float = 0.0
    correctness_equal_nan: bool = False
    correctness_determinism_repeats: int = 1
    performance_timer: str = "auto"
    performance_graph_mode: str = "auto"
    performance_warmup: int = 5
    performance_samples: int = 30
    performance_inner_iterations: int = 20
    performance_timeout_s: int = 600
    performance_regression_threshold_pct: float = 5.0
    perf_on_correctness_fail: bool = False
    performance_min_speedup: float | None = None
    performance_max_candidate_median_ms: float | None = None
    performance_max_cv: float | None = 0.1
    performance_max_memory_bytes: int | None = None
    performance_unsupported_policy: str = "fail"
    gpu_lock_timeout_s: float = 600.0

    def __post_init__(self) -> None:
        if self.mode not in {"all", "correctness", "performance"}:
            raise ValueError("mode must be all, correctness, or performance")
        if not isinstance(self.correctness_seeds, tuple) or not self.correctness_seeds or any(
            isinstance(seed, bool) or not isinstance(seed, int)
            for seed in self.correctness_seeds
        ):
            raise TypeError("correctness_seeds must be a non-empty tuple of integers")
        for name in (
            "correctness_determinism_repeats",
            "performance_samples",
            "performance_inner_iterations",
            "performance_timeout_s",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.performance_warmup < 0:
            raise ValueError("performance_warmup must be non-negative")
        if self.performance_timer not in {
            "auto",
            "cuda_event",
            "cuda_graph",
            "wall_clock",
        }:
            raise ValueError(
                "performance_timer must be auto, cuda_event, cuda_graph, or wall_clock"
            )
        if self.performance_graph_mode not in {"auto", "enabled", "disabled"}:
            raise ValueError("performance_graph_mode must be auto, enabled, or disabled")
        for name in (
            "correctness_rtol",
            "correctness_atol",
            "performance_regression_threshold_pct",
        ):
            value = getattr(self, name)
            if not math.isfinite(float(value)) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        for name in ("performance_min_speedup", "performance_max_candidate_median_ms", "performance_max_cv"):
            value = getattr(self, name)
            if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                      or not math.isfinite(float(value)) or value < 0):
                raise ValueError(f"{name} must be finite and non-negative or None")
        if self.performance_max_memory_bytes is not None and (
            isinstance(self.performance_max_memory_bytes, bool)
            or not isinstance(self.performance_max_memory_bytes, int)
            or self.performance_max_memory_bytes < 0
        ):
            raise ValueError("performance_max_memory_bytes must be non-negative or None")
        if self.performance_unsupported_policy not in {"fail", "allow"}:
            raise ValueError("performance_unsupported_policy must be fail or allow")
        if not math.isfinite(float(self.gpu_lock_timeout_s)) or self.gpu_lock_timeout_s < 0:
            raise ValueError("gpu_lock_timeout_s must be finite and non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "correctness": {
                "seeds": list(self.correctness_seeds),
                "default_comparator": self.correctness_default_comparator,
                "rtol": self.correctness_rtol,
                "atol": self.correctness_atol,
                "equal_nan": self.correctness_equal_nan,
                "determinism_repeats": self.correctness_determinism_repeats,
            },
            "performance": {
                "timer": self.performance_timer,
                "graph_mode": self.performance_graph_mode,
                "warmup": self.performance_warmup,
                "samples": self.performance_samples,
                "inner_iterations": self.performance_inner_iterations,
                "timeout_s": self.performance_timeout_s,
                "regression_threshold_pct": self.performance_regression_threshold_pct,
                "perf_on_correctness_fail": self.perf_on_correctness_fail,
                "min_speedup": self.performance_min_speedup,
                "max_candidate_median_ms": self.performance_max_candidate_median_ms,
                "max_cv": self.performance_max_cv,
                "max_memory_bytes": self.performance_max_memory_bytes,
                "unsupported_policy": self.performance_unsupported_policy,
                "gpu_lock_timeout_s": self.gpu_lock_timeout_s,
            },
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ResolvedEvaluationConfig":
        if not isinstance(value, dict):
            raise TypeError("ResolvedEvaluationConfig must be an object")
        if set(value) != {"mode", "correctness", "performance"}:
            raise ValueError("ResolvedEvaluationConfig has missing or unknown fields")
        correctness = value["correctness"]
        performance = value["performance"]
        if not isinstance(correctness, dict) or not isinstance(performance, dict):
            raise TypeError("resolved correctness/performance must be objects")
        expected_correctness = {
            "seeds", "default_comparator", "rtol", "atol", "equal_nan",
            "determinism_repeats",
        }
        expected_performance = {
            "timer", "graph_mode", "warmup", "samples", "inner_iterations",
            "timeout_s", "regression_threshold_pct",
            "perf_on_correctness_fail", "min_speedup", "max_candidate_median_ms",
            "max_cv", "max_memory_bytes", "unsupported_policy", "gpu_lock_timeout_s",
        }
        legacy_performance = {
            "timer", "graph_mode", "warmup", "samples", "inner_iterations",
            "timeout_s", "regression_threshold_pct",
        }
        if set(correctness) != expected_correctness or frozenset(performance) not in {frozenset(expected_performance), frozenset(legacy_performance)}:
            raise ValueError("resolved correctness/performance has missing or unknown fields")
        performance = {**ResolvedEvaluationConfig().to_dict()["performance"], **performance}
        seeds = correctness["seeds"]
        if not isinstance(seeds, list):
            raise TypeError("correctness.seeds must be an array")
        return cls(
            mode=_typed(value["mode"], str, "mode"),
            correctness_seeds=tuple(_integer(seed, "correctness.seeds[]") for seed in seeds),
            correctness_default_comparator=_typed(correctness["default_comparator"], str, "correctness.default_comparator"),
            correctness_rtol=_number(correctness["rtol"], "correctness.rtol"),
            correctness_atol=_number(correctness["atol"], "correctness.atol"),
            correctness_equal_nan=_typed(correctness["equal_nan"], bool, "correctness.equal_nan"),
            correctness_determinism_repeats=_integer(correctness["determinism_repeats"], "correctness.determinism_repeats"),
            performance_timer=_typed(performance["timer"], str, "performance.timer"),
            performance_graph_mode=_typed(performance["graph_mode"], str, "performance.graph_mode"),
            performance_warmup=_integer(performance["warmup"], "performance.warmup"),
            performance_samples=_integer(performance["samples"], "performance.samples"),
            performance_inner_iterations=_integer(performance["inner_iterations"], "performance.inner_iterations"),
            performance_timeout_s=_integer(performance["timeout_s"], "performance.timeout_s"),
            performance_regression_threshold_pct=_number(performance["regression_threshold_pct"], "performance.regression_threshold_pct"),
            perf_on_correctness_fail=_typed(performance["perf_on_correctness_fail"], bool, "performance.perf_on_correctness_fail"),
            performance_min_speedup=_optional_number(performance["min_speedup"], "performance.min_speedup"),
            performance_max_candidate_median_ms=_optional_number(performance["max_candidate_median_ms"], "performance.max_candidate_median_ms"),
            performance_max_cv=_optional_number(performance["max_cv"], "performance.max_cv"),
            performance_max_memory_bytes=_optional_integer(performance["max_memory_bytes"], "performance.max_memory_bytes"),
            performance_unsupported_policy=_typed(performance["unsupported_policy"], str, "performance.unsupported_policy"),
            gpu_lock_timeout_s=_number(performance["gpu_lock_timeout_s"], "performance.gpu_lock_timeout_s"),
        )


def _typed(value: object, expected: type, field: str) -> Any:
    if not isinstance(value, expected):
        raise TypeError(f"{field} has invalid type")
    return value


def _integer(value: object, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field} must be an integer")
    return value


def _number(value: object, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be a number")
    return float(value)


def _optional_number(value: object, field: str) -> float | None:
    return None if value is None else _number(value, field)


def _optional_integer(value: object, field: str) -> int | None:
    return None if value is None else _integer(value, field)


def resolve_evaluation_config(
    operator_manifest: object,
    suite: object,
    *,
    mode: str | None = None,
    seeds: tuple[int, ...] = (),
    performance_timer: str | None = None,
    performance_warmup: int | None = None,
    performance_samples: int | None = None,
    performance_inner_iterations: int | None = None,
    performance_max_slowdown_pct: float | None = None,
    perf_on_correctness_fail: bool | None = None,
    performance_min_speedup: float | None = None,
    performance_max_candidate_median_ms: float | None = None,
    performance_max_cv: float | None = None,
    performance_max_memory_bytes: int | None = None,
    performance_unsupported_policy: str | None = None,
    gpu_lock_timeout_s: float | None = None,
) -> ResolvedEvaluationConfig:
    """Resolve CLI > suite > operator manifest > engine defaults."""

    defaults = ResolvedEvaluationConfig()
    correctness = getattr(operator_manifest, "correctness", None)
    performance = getattr(operator_manifest, "performance", None)
    suite_seeds = tuple(getattr(suite, "correctness_seeds", ()))
    return ResolvedEvaluationConfig(
        mode=mode or getattr(suite, "mode", None) or defaults.mode,
        correctness_seeds=seeds or suite_seeds or defaults.correctness_seeds,
        correctness_default_comparator=getattr(correctness, "default_comparator", defaults.correctness_default_comparator),
        correctness_rtol=getattr(correctness, "rtol", defaults.correctness_rtol),
        correctness_atol=getattr(correctness, "atol", defaults.correctness_atol),
        correctness_equal_nan=getattr(correctness, "equal_nan", defaults.correctness_equal_nan),
        correctness_determinism_repeats=getattr(correctness, "determinism_repeats", defaults.correctness_determinism_repeats),
        performance_timer=(
            performance_timer
            or getattr(suite, "performance_timer", None)
            or getattr(performance, "timer", defaults.performance_timer)
        ),
        performance_graph_mode=getattr(performance, "graph_mode", defaults.performance_graph_mode),
        performance_warmup=(
            performance_warmup
            if performance_warmup is not None
            else getattr(suite, "performance_warmup", None)
            if getattr(suite, "performance_warmup", None) is not None
            else getattr(performance, "warmup", defaults.performance_warmup)
        ),
        performance_samples=(
            performance_samples
            if performance_samples is not None
            else getattr(suite, "performance_samples", None)
            or getattr(performance, "samples", defaults.performance_samples)
        ),
        performance_inner_iterations=(
            performance_inner_iterations
            if performance_inner_iterations is not None
            else getattr(suite, "performance_inner_iterations", None)
            or getattr(performance, "inner_iterations", defaults.performance_inner_iterations)
        ),
        performance_timeout_s=getattr(performance, "timeout_s", defaults.performance_timeout_s),
        performance_regression_threshold_pct=(performance_max_slowdown_pct if performance_max_slowdown_pct is not None else getattr(suite, "performance_max_slowdown_pct", None) if getattr(suite, "performance_max_slowdown_pct", None) is not None else getattr(performance, "regression_threshold_pct", defaults.performance_regression_threshold_pct)),
        perf_on_correctness_fail=(perf_on_correctness_fail if perf_on_correctness_fail is not None else getattr(suite, "perf_on_correctness_fail", None) if getattr(suite, "perf_on_correctness_fail", None) is not None else getattr(performance, "perf_on_correctness_fail", defaults.perf_on_correctness_fail)),
        performance_min_speedup=(performance_min_speedup if performance_min_speedup is not None else getattr(suite, "performance_min_speedup", None) if getattr(suite, "performance_min_speedup", None) is not None else getattr(performance, "min_speedup", defaults.performance_min_speedup)),
        performance_max_candidate_median_ms=(performance_max_candidate_median_ms if performance_max_candidate_median_ms is not None else getattr(suite, "performance_max_candidate_median_ms", None) if getattr(suite, "performance_max_candidate_median_ms", None) is not None else getattr(performance, "max_candidate_median_ms", defaults.performance_max_candidate_median_ms)),
        performance_max_cv=(performance_max_cv if performance_max_cv is not None else getattr(suite, "performance_max_cv", None) if getattr(suite, "performance_max_cv", None) is not None else getattr(performance, "max_cv", defaults.performance_max_cv)),
        performance_max_memory_bytes=(performance_max_memory_bytes if performance_max_memory_bytes is not None else getattr(suite, "performance_max_memory_bytes", None) if getattr(suite, "performance_max_memory_bytes", None) is not None else getattr(performance, "max_memory_bytes", defaults.performance_max_memory_bytes)),
        performance_unsupported_policy=(performance_unsupported_policy or getattr(suite, "performance_unsupported_policy", None) or getattr(performance, "unsupported_policy", defaults.performance_unsupported_policy)),
        gpu_lock_timeout_s=(gpu_lock_timeout_s if gpu_lock_timeout_s is not None else getattr(suite, "gpu_lock_timeout_s", None) if getattr(suite, "gpu_lock_timeout_s", None) is not None else getattr(performance, "gpu_lock_timeout_s", defaults.gpu_lock_timeout_s)),
    )


__all__ = [
    "DEFAULT_MODE",
    "DEFAULT_OUTPUT_ROOT",
    "ResolvedEvaluationConfig",
    "resolve_evaluation_config",
]
