"""Reference-owned structural and numerical output comparators."""

from __future__ import annotations

import math
import numbers
from dataclasses import dataclass
from typing import Mapping

from .metrics import floating_metrics
from .models import ComparisonResult, OutputBundle, OutputLeaf, QuantizationParameters, Tolerance

_FALLBACKS = {
    "float64": (1e-7, 1e-9),
    "torch.float64": (1e-7, 1e-9),
    "float32": (1e-4, 1e-5),
    "torch.float32": (1e-4, 1e-5),
    "float16": (1e-2, 1e-2),
    "torch.float16": (1e-2, 1e-2),
    "bfloat16": (1e-2, 1e-2),
    "torch.bfloat16": (1e-2, 1e-2),
}


def _finite_or_none(value: float) -> float | None:
    return value if math.isfinite(value) else None


def resolve_tolerance(
    dtype: str,
    *,
    case_override: Tolerance | None = None,
    output_override: Tolerance | None = None,
    operator_default: Tolerance | None = None,
    require_explicit: bool = False,
) -> Tolerance:
    for source, value in (("case", case_override), ("output", output_override), ("operator", operator_default)):
        if value is not None:
            if value.rtol < 0 or value.atol < 0 or not math.isfinite(value.rtol + value.atol):
                raise ValueError("tolerances must be finite and non-negative")
            return Tolerance(value.rtol, value.atol, value.equal_nan, source)
    if require_explicit:
        raise ValueError(f"formal output dtype {dtype} requires an explicit tolerance")
    try:
        rtol, atol = _FALLBACKS[dtype]
    except KeyError as error:
        raise ValueError(f"no dtype fallback tolerance for {dtype}") from error
    return Tolerance(rtol, atol, False, "dtype_fallback")


def _structure(
    reference: OutputBundle,
    candidate: OutputBundle,
    output_paths: tuple[str, ...] | None = None,
) -> tuple[dict[str, OutputLeaf], dict[str, OutputLeaf], ComparisonResult | None]:
    refs, cands = reference.by_path(), candidate.by_path()
    if output_paths is not None:
        missing_reference = [path for path in output_paths if path not in refs]
        missing_candidate = [path for path in output_paths if path not in cands]
        if missing_reference or missing_candidate:
            failed_path = (missing_reference or missing_candidate)[0]
            metrics = {
                "declared_paths": list(output_paths),
                "missing_reference_paths": missing_reference,
                "missing_candidate_paths": missing_candidate,
            }
            return refs, cands, ComparisonResult(
                False,
                "structure",
                metrics,
                ({"path": failed_path, "reason": "declared output path missing", **metrics},),
                failed_path,
            )
        refs = {path: refs[path] for path in output_paths}
        cands = {path: cands[path] for path in output_paths}
    if set(refs) != set(cands):
        missing = sorted(set(refs) - set(cands)); extra = sorted(set(cands) - set(refs))
        return refs, cands, ComparisonResult(False, "structure", {"missing_paths": missing, "extra_paths": extra}, failed_path=(missing or extra or [None])[0])
    for path in sorted(refs):
        ref, cand = refs[path], cands[path]
        for field in ("dtype", "shape", "stride", "layout"):
            if getattr(ref, field) != getattr(cand, field):
                return refs, cands, ComparisonResult(
                    False, "structure", {"contract_field": field, "reference": ref.contract(), "candidate": cand.contract()},
                    ({"path": path, "reason": f"{field} mismatch", "reference": ref.contract(), "candidate": cand.contract()},), path,
                )
    return refs, cands, None


@dataclass(frozen=True)
class ExactComparator:
    worst_k: int = 8

    def compare(self, reference: OutputBundle, candidate: OutputBundle, **_: object) -> ComparisonResult:
        refs, cands, failure = _structure(reference, candidate)
        if failure is not None:
            return failure
        total = 0; mismatches: list[dict[str, object]] = []
        for path in sorted(refs):
            ref, cand = refs[path], cands[path]
            total += ref.size
            for index, (left, right) in enumerate(zip(ref.value, cand.value)):
                if type(left) is not type(right) or left != right:
                    safe_left = left if not isinstance(left, float) or math.isfinite(left) else None
                    safe_right = right if not isinstance(right, float) or math.isfinite(right) else None
                    mismatches.append({"path": path, "index": index, "reference": safe_left, "candidate": safe_right})
        metrics = {"mismatch_count": len(mismatches), "mismatch_rate": len(mismatches) / total if total else 0.0, "first_mismatch": mismatches[0] if mismatches else None}
        return ComparisonResult(not mismatches, "exact", metrics, tuple(mismatches[:self.worst_k]), mismatches[0]["path"] if mismatches else None)


@dataclass(frozen=True)
class FloatingComparator:
    operator_default: Tolerance | None = None
    output_overrides: Mapping[str, Tolerance] | None = None
    require_explicit: bool = False
    worst_k: int = 8

    def compare(self, reference: OutputBundle, candidate: OutputBundle, *, case_overrides: Mapping[str, Tolerance] | None = None, **_: object) -> ComparisonResult:
        refs, cands, failure = _structure(reference, candidate)
        if failure is not None:
            return failure
        all_metrics: dict[str, object] = {}; diagnostics: list[dict[str, object]] = []; failed: str | None = None
        for path in sorted(refs):
            tolerance = resolve_tolerance(
                refs[path].dtype,
                case_override=(case_overrides or {}).get(path),
                output_override=(self.output_overrides or {}).get(path),
                operator_default=self.operator_default,
                require_explicit=self.require_explicit,
            )
            metrics, worst = floating_metrics(refs[path].value, cands[path].value, rtol=tolerance.rtol, atol=tolerance.atol, equal_nan=tolerance.equal_nan)
            all_metrics[path] = {**metrics, "rtol": tolerance.rtol, "atol": tolerance.atol, "equal_nan": tolerance.equal_nan, "tolerance_source": tolerance.source}
            if metrics["mismatch_count"]:
                failed = failed or path
                diagnostics.extend({"path": path, **item} for item in worst[:self.worst_k])
        return ComparisonResult(failed is None, "floating", all_metrics, tuple(diagnostics[:self.worst_k]), failed)


@dataclass(frozen=True)
class CalcDiffComparator:
    """DeepGEMM's normalized whole-output error contract.

    This deliberately uses the upstream strict inequality and its zero-vector
    rule rather than translating the contract into element-wise tolerances.
    """

    max_diff: float
    source: str
    output_paths: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_diff, bool)
            or not isinstance(self.max_diff, numbers.Real)
            or not math.isfinite(float(self.max_diff))
            or float(self.max_diff) < 0
        ):
            raise ValueError("calc_diff max_diff must be finite and non-negative")
        if not isinstance(self.source, str) or not self.source:
            raise TypeError("calc_diff source must be a non-empty string")
        if self.output_paths is not None:
            if not isinstance(self.output_paths, tuple):
                raise TypeError("calc_diff output_paths must be a tuple or None")
            if (
                not self.output_paths
                or any(not isinstance(path, str) or not path for path in self.output_paths)
                or len(set(self.output_paths)) != len(self.output_paths)
            ):
                raise ValueError(
                    "calc_diff output_paths must contain unique non-empty strings"
                )

    def compare(self, reference: OutputBundle, candidate: OutputBundle, **_: object) -> ComparisonResult:
        refs, cands, failure = _structure(reference, candidate, self.output_paths)
        if failure is not None:
            return failure
        all_metrics: dict[str, object] = {}
        diagnostics: list[dict[str, object]] = []
        failed: str | None = None
        for path in sorted(refs):
            denominator = 0.0
            dot = 0.0
            nonfinite: list[dict[str, object]] = []
            for index, (raw_ref, raw_cand) in enumerate(zip(refs[path].value, cands[path].value)):
                ref, cand = float(raw_ref), float(raw_cand)
                if not math.isfinite(ref) or not math.isfinite(cand):
                    if len(nonfinite) < 8:
                        nonfinite.append({"index": index, "reference": _finite_or_none(ref), "candidate": _finite_or_none(cand)})
                    continue
                denominator += ref * ref + cand * cand
                dot += ref * cand
            if nonfinite:
                calc_diff: float | None = None
                passed = False
                diagnostics.extend({"path": path, "reason": "nonfinite", **item} for item in nonfinite)
            elif denominator == 0.0:
                calc_diff = 0.0
                passed = calc_diff < self.max_diff
            else:
                calc_diff = 1.0 - 2.0 * dot / denominator
                passed = calc_diff < self.max_diff
            all_metrics[path] = {
                "calc_diff": calc_diff,
                "max_diff": float(self.max_diff),
                "threshold_source": self.source,
                "zero_denominator": denominator == 0.0 and not nonfinite,
                "nonfinite_count": len(nonfinite),
            }
            if not passed:
                failed = failed or path
                if not nonfinite:
                    diagnostics.append({"path": path, "calc_diff": calc_diff, "max_diff": float(self.max_diff)})
        return ComparisonResult(failed is None, "calc_diff", all_metrics, tuple(diagnostics[:8]), failed)


@dataclass(frozen=True)
class TopKComparator:
    indices_path: str = "output.indices"
    scores_path: str | None = "output.scores"
    ordered: bool = True
    tie_aware: bool = False
    universe_size: int | None = None
    score_tolerance: Tolerance = Tolerance(1e-5, 1e-6, False, "topk")

    def __post_init__(self) -> None:
        if not isinstance(self.indices_path, str) or not self.indices_path:
            raise ValueError("indices_path must be a non-empty string")
        if self.scores_path is not None and (
            not isinstance(self.scores_path, str) or not self.scores_path
        ):
            raise ValueError("scores_path must be a non-empty string or None")
        if self.universe_size is not None and (
            isinstance(self.universe_size, bool)
            or not isinstance(self.universe_size, int)
            or self.universe_size <= 0
        ):
            raise ValueError("universe_size must be a positive integer or None")
        if not isinstance(self.ordered, bool) or not isinstance(self.tie_aware, bool):
            raise TypeError("ordered and tie_aware must be bool values")
        if not isinstance(self.score_tolerance, Tolerance):
            raise TypeError("score_tolerance must be a Tolerance")

    def compare(self, reference: OutputBundle, candidate: OutputBundle, **_: object) -> ComparisonResult:
        refs, cands, failure = _structure(reference, candidate)
        if failure is not None:
            return failure
        if self.indices_path not in refs:
            return ComparisonResult(False, "topk", {"error": "indices path missing"}, failed_path=self.indices_path)
        invalid: list[str] = []
        def validate_indices(values: tuple[object, ...], side: str) -> tuple[int, ...] | None:
            converted: list[int] = []
            for position, value in enumerate(values):
                if isinstance(value, bool) or not isinstance(value, numbers.Integral):
                    invalid.append(f"{side} index at position {position} is not an exact integer")
                    continue
                index = int(value)
                if index < 0:
                    invalid.append(f"{side} index at position {position} is negative")
                elif self.universe_size is not None and index >= self.universe_size:
                    invalid.append(f"{side} index at position {position} is out of bounds")
                converted.append(index)
            return tuple(converted) if len(converted) == len(values) else None

        raw_ref_idx = refs[self.indices_path].value
        raw_cand_idx = cands[self.indices_path].value
        ref_idx = validate_indices(raw_ref_idx, "reference")
        cand_idx = validate_indices(raw_cand_idx, "candidate")
        if any("out of bounds" in item for item in invalid):
            invalid.append("index out of bounds")
        if ref_idx is None or cand_idx is None:
            metrics = {"k": len(raw_ref_idx), "invalid": invalid}
            return ComparisonResult(
                False,
                "topk",
                metrics,
                ({"path": self.indices_path, "errors": invalid},),
                self.indices_path,
            )
        if len(ref_idx) != len(cand_idx): invalid.append("length mismatch")
        if len(set(ref_idx)) != len(ref_idx): invalid.append("reference contains duplicate index")
        if len(set(cand_idx)) != len(cand_idx): invalid.append("candidate contains duplicate index")
        ref_set, cand_set = set(ref_idx), set(cand_idx)
        intersection = len(ref_set & cand_set); union = len(ref_set | cand_set)
        metrics: dict[str, object] = {
            "k": len(ref_idx), "ordered_exact": ref_idx == cand_idx,
            "unordered_exact": len(ref_idx) == len(cand_idx) and ref_set == cand_set and not invalid,
            "precision_at_k": intersection / len(cand_idx) if cand_idx else (1.0 if not ref_idx else 0.0),
            "recall_at_k": intersection / len(ref_idx) if ref_idx else (1.0 if not cand_idx else 0.0),
            "jaccard": intersection / union if union else 1.0,
            "invalid": invalid,
        }
        score_ok = True; tie_ok = False
        if self.scores_path is not None:
            if self.scores_path not in refs:
                invalid.append("scores path missing")
                score_ok = False
            else:
                ref_scores, cand_scores = refs[self.scores_path].value, cands[self.scores_path].value
                if len(ref_scores) != len(ref_idx) or len(cand_scores) != len(cand_idx):
                    invalid.append("score length does not match indices")
                    score_ok = False
                    score_metrics = {"mismatch_count": 1, "reason": "score length"}
                else:
                    ref_score_by_index = dict(zip(ref_idx, ref_scores))
                    cand_score_by_index = dict(zip(cand_idx, cand_scores))
                    common = sorted(ref_set & cand_set)
                    score_metrics, _ = floating_metrics(
                        tuple(ref_score_by_index[index] for index in common),
                        tuple(cand_score_by_index[index] for index in common),
                        rtol=self.score_tolerance.rtol,
                        atol=self.score_tolerance.atol,
                        equal_nan=self.score_tolerance.equal_nan,
                    )
                    score_ok = score_metrics["mismatch_count"] == 0
                metrics["score_error"] = score_metrics
                if (
                    self.tie_aware
                    and ref_set == cand_set
                    and len(ref_idx) == len(cand_idx)
                    and len(ref_scores) == len(ref_idx)
                    and len(cand_scores) == len(cand_idx)
                ):
                    score_by_index = dict(zip(ref_idx, (float(v) for v in ref_scores)))
                    tie_ok = all(
                        ref_idx[pos] == cand_idx[pos] or math.isclose(score_by_index[ref_idx[pos]], score_by_index[cand_idx[pos]], rel_tol=self.score_tolerance.rtol, abs_tol=self.score_tolerance.atol)
                        for pos in range(len(ref_idx))
                    )
        selection_ok = metrics["ordered_exact"] if self.ordered else metrics["unordered_exact"]
        if self.tie_aware:
            selection_ok = bool(selection_ok or tie_ok)
        passed = bool(selection_ok and score_ok and not invalid)
        metrics["tie_aware_valid"] = tie_ok
        return ComparisonResult(passed, "topk", metrics, (() if passed else ({"path": self.indices_path, "reference": list(ref_idx), "candidate": list(cand_idx), "errors": invalid},)), None if passed else self.indices_path)


@dataclass(frozen=True)
class QuantizedComparator:
    parameters: QuantizationParameters | None = None
    mode: str = "raw_code"
    tolerance: Tolerance | None = None

    def _code_metrics(self, candidate: OutputBundle) -> dict[str, object]:
        code_count = sum(leaf.size for leaf in candidate.leaves)
        metrics: dict[str, object] = {
            "zero_rate": sum(
                value == 0 for leaf in candidate.leaves for value in leaf.value
            ) / max(1, code_count)
        }
        if self.parameters is not None and self.parameters.quant_min is not None:
            quant_min = self.parameters.quant_min
            quant_max = self.parameters.quant_max
            metrics["saturation_rate"] = sum(
                value == quant_min or value == quant_max
                for leaf in candidate.leaves
                for value in leaf.value
            ) / max(1, code_count)
        return metrics

    def _code_range_failure(
        self, reference: OutputBundle, candidate: OutputBundle
    ) -> ComparisonResult | None:
        params = self.parameters
        if params is None or params.quant_min is None or params.quant_max is None:
            return None
        failures: list[dict[str, object]] = []
        for side, bundle in (("reference", reference), ("candidate", candidate)):
            for leaf in bundle.leaves:
                for index, value in enumerate(leaf.value):
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, numbers.Integral)
                        or int(value) < params.quant_min
                        or int(value) > params.quant_max
                    ):
                        failures.append(
                            {"side": side, "path": leaf.path, "index": index, "value": value}
                        )
        if not failures:
            return None
        return ComparisonResult(
            False,
            "quantized_code_range",
            {"quant_min": params.quant_min, "quant_max": params.quant_max, "out_of_range_count": len(failures)},
            tuple(failures[:8]),
            str(failures[0]["path"]),
        )

    def _dequantize(self, leaf: OutputLeaf) -> OutputLeaf:
        if self.parameters is None:
            raise ValueError("dequant mode requires explicit scale/zero-point parameters")
        params = self.parameters
        if params.layout is not None and leaf.layout != params.layout:
            raise ValueError(f"quantized layout mismatch at {leaf.path}")
        scales = params.scale if isinstance(params.scale, tuple) else (params.scale,)
        zeros = params.zero_point if isinstance(params.zero_point, tuple) else (params.zero_point,)
        if any(not math.isfinite(float(scale)) or float(scale) <= 0 for scale in scales):
            raise ValueError("quantization scale must be finite and positive")
        if params.axis is None:
            if len(scales) != 1 or len(zeros) != 1:
                raise ValueError("vector scale/zero-point requires an axis")
            values = tuple((float(value) - zeros[0]) * float(scales[0]) for value in leaf.value)
        else:
            axis = params.axis if params.axis >= 0 else params.axis + len(leaf.shape)
            if axis < 0 or axis >= len(leaf.shape): raise ValueError("quantization axis out of range")
            if len(scales) != leaf.shape[axis] or len(zeros) not in {1, leaf.shape[axis]}: raise ValueError("scale/zero-point length does not match quantization axis")
            inner = math.prod(leaf.shape[axis + 1:])
            values = tuple((float(value) - zeros[(index // inner) % len(zeros)]) * float(scales[(index // inner) % len(scales)]) for index, value in enumerate(leaf.value))
        return OutputLeaf(leaf.path, values, "float64", leaf.shape, leaf.stride, leaf.layout, leaf.device)

    def compare(self, reference: OutputBundle, candidate: OutputBundle, **_: object) -> ComparisonResult:
        range_failure = self._code_range_failure(reference, candidate)
        if range_failure is not None:
            return range_failure
        if self.mode == "raw_code":
            result = ExactComparator().compare(reference, candidate)
            metrics = dict(result.metrics)
            metrics.update(self._code_metrics(candidate))
            return ComparisonResult(result.passed, "quantized_raw_code", metrics, result.diagnostics, result.failed_path)
        if self.mode != "dequant":
            raise ValueError("quantized mode must be raw_code or dequant")
        if self.tolerance is None:
            raise ValueError("dequant mode requires an explicit tolerance")
        _, _, structure_failure = _structure(reference, candidate)
        if structure_failure is not None:
            return structure_failure
        ref = OutputBundle(tuple(self._dequantize(leaf) for leaf in reference.leaves))
        cand = OutputBundle(tuple(self._dequantize(leaf) for leaf in candidate.leaves))
        result = FloatingComparator(operator_default=self.tolerance, require_explicit=True).compare(ref, cand)
        metrics = dict(result.metrics)
        metrics.update(self._code_metrics(candidate))
        metrics["dequant_error"] = {
            path: {
                name: values.get(name)
                for name in ("max_abs_error", "mean_abs_error", "rmse", "mismatch_count")
            }
            for path, values in result.metrics.items()
            if isinstance(values, Mapping)
        }
        return ComparisonResult(result.passed, "quantized_dequant", metrics, result.diagnostics, result.failed_path)
