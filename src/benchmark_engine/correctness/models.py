"""Runtime-only data models used by correctness evaluation.

Unlike :mod:`benchmark_engine.models`, these objects deliberately may contain
tensor values.  They never cross the controller/worker JSON boundary.
"""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass, field
from typing import Mapping, Literal


def _json_safe(value: object, depth: int = 0) -> object:
    """Bound arbitrary comparator metadata for strict JSON serialization."""

    if depth > 10:
        return {"truncated": True, "reason": "depth"}
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return "NaN"
        if value == math.inf:
            return "+Inf"
        if value == -math.inf:
            return "-Inf"
        return value
    if isinstance(value, str):
        return value[:4096]
    if isinstance(value, Mapping):
        result: dict[str, object] = {}
        for index, (key, item) in enumerate(value.items()):
            if index >= 128:
                result["_truncated"] = True
                break
            if not isinstance(key, str):
                raise TypeError("correctness result mappings require string keys")
            result[key[:256]] = _json_safe(item, depth + 1)
        return result
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, depth + 1) for item in value[:128]]
    return repr(value)[:512]


@dataclass
class InputBundle:
    args: tuple[object, ...] = ()
    kwargs: dict[str, object] = field(default_factory=dict)
    observed_state: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.args, tuple):
            raise TypeError("InputBundle.args must be a tuple")
        if not isinstance(self.kwargs, dict) or not isinstance(self.observed_state, dict):
            raise TypeError("InputBundle kwargs and observed_state must be dicts")


@dataclass(frozen=True)
class OutputLeaf:
    path: str
    value: tuple[object, ...]
    dtype: str
    shape: tuple[int, ...]
    stride: tuple[int, ...] | None
    layout: str
    device: str

    @property
    def size(self) -> int:
        return len(self.value)

    def contract(self) -> dict[str, object]:
        return {
            "dtype": self.dtype,
            "shape": list(self.shape),
            "stride": None if self.stride is None else list(self.stride),
            "layout": self.layout,
            "device": self.device,
        }


@dataclass(frozen=True)
class OutputBundle:
    leaves: tuple[OutputLeaf, ...]

    def __post_init__(self) -> None:
        paths = [leaf.path for leaf in self.leaves]
        if len(paths) != len(set(paths)):
            raise ValueError("OutputBundle contains duplicate leaf paths")

    def by_path(self) -> dict[str, OutputLeaf]:
        return {leaf.path: leaf for leaf in self.leaves}


@dataclass(frozen=True)
class Tolerance:
    rtol: float
    atol: float
    equal_nan: bool = False
    source: str = "explicit"

    def __post_init__(self) -> None:
        for name, value in (("rtol", self.rtol), ("atol", self.atol)):
            if (
                isinstance(value, bool)
                or not isinstance(value, numbers.Real)
                or not math.isfinite(float(value))
                or float(value) < 0
            ):
                raise ValueError(f"{name} must be a finite non-negative number")
        if not isinstance(self.equal_nan, bool):
            raise TypeError("equal_nan must be a bool")
        if not isinstance(self.source, str) or not self.source:
            raise TypeError("tolerance source must be a non-empty string")


@dataclass(frozen=True)
class ComparisonResult:
    passed: bool
    comparator: str
    metrics: Mapping[str, object]
    diagnostics: tuple[Mapping[str, object], ...] = ()
    failed_path: str | None = None


@dataclass(frozen=True)
class CorrectnessResult:
    status: Literal["pass", "fail", "error", "timeout", "oom", "unsupported", "nondeterministic"]
    case_id: str
    seed: int
    case_hash: str
    generator_version: str
    input_summary: Mapping[str, object]
    comparison: ComparisonResult | None = None
    diagnostic: Mapping[str, object] | None = None

    @property
    def passed(self) -> bool:
        return self.status == "pass"

    def to_dict(self) -> dict[str, object]:
        comparison = None
        if self.comparison is not None:
            comparison = {
                "passed": self.comparison.passed,
                "comparator": self.comparison.comparator,
                "metrics": _json_safe(self.comparison.metrics),
                "diagnostics": _json_safe(self.comparison.diagnostics),
                "failed_path": self.comparison.failed_path,
            }
        return {
            "status": self.status,
            "case_id": self.case_id,
            "seed": self.seed,
            "case_hash": self.case_hash,
            "generator_version": self.generator_version,
            "input_summary": _json_safe(self.input_summary),
            "comparison": comparison,
            "diagnostic": None if self.diagnostic is None else _json_safe(self.diagnostic),
        }


@dataclass(frozen=True)
class QuantizationParameters:
    scale: float | tuple[float, ...]
    zero_point: int | tuple[int, ...] = 0
    axis: int | None = None
    layout: str | None = None
    quant_min: int | None = None
    quant_max: int | None = None

    def __post_init__(self) -> None:
        scales = self.scale if isinstance(self.scale, tuple) else (self.scale,)
        zeros = self.zero_point if isinstance(self.zero_point, tuple) else (self.zero_point,)
        if not scales:
            raise ValueError("quantization scale must not be empty")
        if any(
            isinstance(scale, bool)
            or not isinstance(scale, numbers.Real)
            or not math.isfinite(float(scale))
            or float(scale) <= 0
            for scale in scales
        ):
            raise ValueError("quantization scale must be finite and positive")
        if not zeros or any(
            isinstance(zero, bool) or not isinstance(zero, numbers.Integral)
            for zero in zeros
        ):
            raise TypeError("zero_point must contain exact integers")
        if self.axis is not None and (
            isinstance(self.axis, bool) or not isinstance(self.axis, int)
        ):
            raise TypeError("quantization axis must be an integer or None")
        if (isinstance(self.scale, tuple) or isinstance(self.zero_point, tuple)) and self.axis is None:
            raise ValueError("vector scale/zero-point requires an axis")
        if self.layout is not None and (
            not isinstance(self.layout, str) or not self.layout
        ):
            raise TypeError("quantization layout must be a non-empty string or None")
        if (self.quant_min is None) != (self.quant_max is None):
            raise ValueError("quant_min and quant_max must be provided together")
        if self.quant_min is not None:
            if any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (self.quant_min, self.quant_max)
            ):
                raise TypeError("quantized code bounds must be exact integers")
            if self.quant_min >= self.quant_max:  # type: ignore[operator]
                raise ValueError("quant_min must be less than quant_max")
