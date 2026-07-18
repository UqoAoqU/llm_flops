"""Strict, safe schema-v1 suite parsing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


class SuiteValidationError(ValueError):
    """A stable suite schema error."""

    def __init__(self, code: str, path: Path, field: str, message: str) -> None:
        self.code = code
        self.path = Path(path)
        self.field = field
        self.message = message
        super().__init__(f"suite.{code}: {self.path} [{field}]: {message}")


@dataclass(frozen=True)
class SuiteConfig:
    schema_version: int
    suite_id: str
    operator_include: tuple[str, ...]
    operator_exclude: tuple[str, ...]
    case_tags: tuple[str, ...]
    mode: str
    correctness_seeds: tuple[int, ...]
    performance_samples: int | None
    performance_inner_iterations: int | None
    candidate_include: tuple[str, ...] = ("*",)
    performance_warmup: int | None = None
    performance_timer: str | None = None
    perf_on_correctness_fail: bool | None = None
    performance_min_speedup: float | None = None
    performance_max_slowdown_pct: float | None = None
    performance_max_candidate_median_ms: float | None = None
    performance_max_cv: float | None = None
    performance_max_memory_bytes: int | None = None
    performance_unsupported_policy: str | None = None
    gpu_lock_timeout_s: float | None = None


def _fail(code: str, path: Path, field: str, message: str) -> None:
    raise SuiteValidationError(code, path, field, message)


def _mapping(value: object, path: Path, field: str) -> dict[str, object]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        _fail("type", path, field, "must be a string-keyed mapping")
    return value


def _fields(
    value: dict[str, object],
    path: Path,
    field: str,
    required: frozenset[str],
) -> None:
    missing = required.difference(value)
    unknown = set(value).difference(required)
    if missing:
        _fail("missing_field", path, field, f"missing {', '.join(sorted(missing))}")
    if unknown:
        _fail("unknown_field", path, field, f"unknown {', '.join(sorted(unknown))}")


def _string(value: object, path: Path, field: str) -> str:
    if not isinstance(value, str) or not value:
        _fail("type", path, field, "must be a non-empty string")
    return value


def _strings(value: object, path: Path, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        _fail("type", path, field, "must be a list of strings")
    result = tuple(_string(item, path, f"{field}[]") for item in value)
    if len(set(result)) != len(result):
        _fail("value", path, field, "must not contain duplicates")
    return result


def _integer(value: object, path: Path, field: str, *, positive: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        _fail("type", path, field, "must be an integer")
    if positive and value <= 0:
        _fail("value", path, field, "must be positive")
    return value


def _number(value: object, path: Path, field: str) -> float:
    import math
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or value < 0:
        _fail("value", path, field, "must be a finite non-negative number")
    return float(value)


def load_suite(path: Path) -> SuiteConfig:
    """Load a suite with ``yaml.safe_load`` and an exact schema."""

    path = Path(path)
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        _fail("yaml", path, "$", f"cannot load YAML: {error}")
    root = _mapping(value, path, "$")
    required = frozenset(
        {
            "schema_version",
            "suite_id",
            "operators",
            "cases",
            "mode",
            "correctness",
            "performance",
        }
    )
    missing = required.difference(root)
    unknown = set(root).difference(required | {"candidates"})
    if missing:
        _fail("missing_field", path, "$", f"missing {', '.join(sorted(missing))}")
    if unknown:
        _fail("unknown_field", path, "$", f"unknown {', '.join(sorted(unknown))}")
    version = _integer(root["schema_version"], path, "schema_version")
    if version != 1:
        _fail("schema_version", path, "schema_version", f"unsupported version {version}")

    operators = _mapping(root["operators"], path, "operators")
    _fields(operators, path, "operators", frozenset({"include", "exclude"}))
    cases = _mapping(root["cases"], path, "cases")
    _fields(cases, path, "cases", frozenset({"tags"}))
    correctness = _mapping(root["correctness"], path, "correctness")
    _fields(correctness, path, "correctness", frozenset({"seeds"}))
    performance = _mapping(root["performance"], path, "performance")
    performance_fields = frozenset({"samples", "inner_iterations"})
    unknown_performance = set(performance).difference(
        performance_fields | {"warmup", "timer", "perf_on_correctness_fail",
                              "min_speedup", "max_slowdown_pct", "max_candidate_median_ms", "max_cv",
                              "max_memory_bytes", "unsupported_policy", "gpu_lock_timeout_s"}
    )
    if unknown_performance:
        _fail(
            "unknown_field",
            path,
            "performance",
            f"unknown {', '.join(sorted(unknown_performance))}",
        )
    mode = _string(root["mode"], path, "mode")
    candidate_include = ("*",)
    if "candidates" in root:
        candidate_section = _mapping(root["candidates"], path, "candidates")
        _fields(candidate_section, path, "candidates", frozenset({"include"}))
        candidate_include = _strings(
            candidate_section["include"], path, "candidates.include"
        )
        if not candidate_include:
            _fail("value", path, "candidates.include", "must not be empty")
    if mode not in {"all", "correctness", "performance"}:
        _fail("value", path, "mode", "must be all, correctness, or performance")
    seeds_value = correctness["seeds"]
    if not isinstance(seeds_value, list):
        _fail("type", path, "correctness.seeds", "must be a list of integers")
    seeds = tuple(
        _integer(seed, path, "correctness.seeds[]") for seed in seeds_value
    )
    if not seeds:
        _fail("value", path, "correctness.seeds", "must not be empty")
    if len(set(seeds)) != len(seeds):
        _fail("value", path, "correctness.seeds", "must not contain duplicates")
    performance_timer = None
    if "timer" in performance:
        performance_timer = _string(performance["timer"], path, "performance.timer")
        if performance_timer not in {"auto", "cuda_event", "cuda_graph", "wall_clock"}:
            _fail(
                "value",
                path,
                "performance.timer",
                "must be auto, cuda_event, cuda_graph, or wall_clock",
            )
    performance_warmup = None
    if "warmup" in performance:
        performance_warmup = _integer(
            performance["warmup"], path, "performance.warmup"
        )
        if performance_warmup < 0:
            _fail(
                "value", path, "performance.warmup", "must be non-negative"
            )
    perf_on_fail = None
    if "perf_on_correctness_fail" in performance:
        if not isinstance(performance["perf_on_correctness_fail"], bool):
            _fail("type", path, "performance.perf_on_correctness_fail", "must be boolean")
        perf_on_fail = performance["perf_on_correctness_fail"]
    unsupported_policy = None
    if "unsupported_policy" in performance:
        unsupported_policy = _string(performance["unsupported_policy"], path, "performance.unsupported_policy")
        if unsupported_policy not in {"fail", "allow"}:
            _fail("value", path, "performance.unsupported_policy", "must be fail or allow")
    max_memory = None
    if "max_memory_bytes" in performance:
        max_memory = _integer(performance["max_memory_bytes"], path, "performance.max_memory_bytes")
        if max_memory < 0:
            _fail("value", path, "performance.max_memory_bytes", "must be non-negative")
    return SuiteConfig(
        schema_version=version,
        suite_id=_string(root["suite_id"], path, "suite_id"),
        operator_include=_strings(operators["include"], path, "operators.include"),
        operator_exclude=_strings(operators["exclude"], path, "operators.exclude"),
        case_tags=_strings(cases["tags"], path, "cases.tags"),
        mode=mode,
        correctness_seeds=seeds,
        performance_samples=(
            None if "samples" not in performance else
            _integer(performance["samples"], path, "performance.samples", positive=True)
        ),
        performance_inner_iterations=(
            None if "inner_iterations" not in performance else
            _integer(performance["inner_iterations"], path,
                     "performance.inner_iterations", positive=True)
        ),
        candidate_include=candidate_include,
        performance_warmup=performance_warmup,
        performance_timer=performance_timer,
        perf_on_correctness_fail=perf_on_fail,
        performance_min_speedup=(None if "min_speedup" not in performance else _number(performance["min_speedup"], path, "performance.min_speedup")),
        performance_max_slowdown_pct=(None if "max_slowdown_pct" not in performance else _number(performance["max_slowdown_pct"], path, "performance.max_slowdown_pct")),
        performance_max_candidate_median_ms=(None if "max_candidate_median_ms" not in performance else _number(performance["max_candidate_median_ms"], path, "performance.max_candidate_median_ms")),
        performance_max_cv=(None if "max_cv" not in performance else _number(performance["max_cv"], path, "performance.max_cv")),
        performance_max_memory_bytes=max_memory,
        performance_unsupported_policy=unsupported_policy,
        gpu_lock_timeout_s=(None if "gpu_lock_timeout_s" not in performance else _number(performance["gpu_lock_timeout_s"], path, "performance.gpu_lock_timeout_s")),
    )


__all__ = ["SuiteConfig", "SuiteValidationError", "load_suite"]
