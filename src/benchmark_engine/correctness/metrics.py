"""Numerically defined, JSON-safe correctness metrics."""

from __future__ import annotations

import math


def _finite(value: float) -> float | None:
    return value if math.isfinite(value) else None


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def floating_metrics(
    reference: tuple[object, ...],
    candidate: tuple[object, ...],
    *,
    rtol: float,
    atol: float,
    equal_nan: bool,
    epsilon: float = 1e-30,
) -> tuple[dict[str, object], tuple[dict[str, object], ...]]:
    if len(reference) != len(candidate):
        raise ValueError("metric inputs must have equal lengths")
    abs_errors: list[float] = []
    rel_errors: list[float] = []
    mismatch: list[dict[str, object]] = []
    sum_sq = 0.0
    ref_sq = 0.0
    dot = 0.0
    cand_sq = 0.0
    ref_nan = cand_nan = ref_pos_inf = ref_neg_inf = cand_pos_inf = cand_neg_inf = 0
    for index, (raw_ref, raw_cand) in enumerate(zip(reference, candidate)):
        ref, cand = float(raw_ref), float(raw_cand)
        ref_nan += int(math.isnan(ref)); cand_nan += int(math.isnan(cand))
        ref_pos_inf += int(ref == math.inf); ref_neg_inf += int(ref == -math.inf)
        cand_pos_inf += int(cand == math.inf); cand_neg_inf += int(cand == -math.inf)
        if not math.isfinite(ref) or not math.isfinite(cand):
            same = (math.isnan(ref) and math.isnan(cand) and equal_nan) or ref == cand
            if not same:
                mismatch.append({"index": index, "reference": _finite(ref), "candidate": _finite(cand), "reason": "nonfinite"})
            continue
        absolute = abs(cand - ref)
        relative = absolute / max(abs(ref), epsilon)
        abs_errors.append(absolute); rel_errors.append(relative)
        sum_sq += absolute * absolute; ref_sq += ref * ref; cand_sq += cand * cand; dot += ref * cand
        threshold = atol + rtol * abs(ref)
        if absolute > threshold:
            mismatch.append({"index": index, "reference": ref, "candidate": cand, "abs_error": absolute, "threshold": threshold})
    count = len(reference)
    finite_count = len(abs_errors)
    if finite_count == 0:
        cosine: float | None = None
        rel_l2: float | None = None
    else:
        denominator = math.sqrt(ref_sq) * math.sqrt(cand_sq)
        cosine = dot / denominator if denominator else (1.0 if ref_sq == cand_sq == 0 else None)
        rel_l2 = math.sqrt(sum_sq) / math.sqrt(ref_sq) if ref_sq else (0.0 if sum_sq == 0 else None)
    metrics: dict[str, object] = {
        "max_abs_error": max(abs_errors) if abs_errors else None,
        "mean_abs_error": sum(abs_errors) / len(abs_errors) if abs_errors else None,
        "p50_abs_error": _percentile(abs_errors, .50),
        "p95_abs_error": _percentile(abs_errors, .95),
        "p99_abs_error": _percentile(abs_errors, .99),
        "max_rel_error": max(rel_errors) if rel_errors else None,
        "rmse": math.sqrt(sum_sq / len(abs_errors)) if abs_errors else None,
        "relative_l2": rel_l2,
        "cosine_similarity": _finite(cosine) if cosine is not None else None,
        "mismatch_count": len(mismatch),
        "mismatch_rate": len(mismatch) / count if count else 0.0,
        "reference_nan_count": ref_nan,
        "candidate_nan_count": cand_nan,
        "reference_pos_inf_count": ref_pos_inf,
        "reference_neg_inf_count": ref_neg_inf,
        "candidate_pos_inf_count": cand_pos_inf,
        "candidate_neg_inf_count": cand_neg_inf,
    }
    worst = sorted(mismatch, key=lambda item: float(item.get("abs_error", math.inf)), reverse=True)
    return metrics, tuple(worst)
