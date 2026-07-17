"""Staged, provenance-preserving performance measurement."""

from .cost_model import CostMetrics, TheoreticalCost, evaluate_cost_model
from .evaluator import (
    ImplementationMeasurement,
    PerformanceConfig,
    PerformanceEvaluator,
    PerformanceResult,
    PreparedPerformance,
)
from .statistics import SampleStatistics, compute_statistics, percentile
from .gate import GateResult, PerformanceGateConfig, evaluate_performance_gate
from .timers import (
    AutoTimer,
    CudaEventTimer,
    CudaGraphTimer,
    GraphCaptureError,
    RawSample,
    Timer,
    TimerConfig,
    TimerError,
    TimerSelection,
    TimerUnsupportedError,
    WallClockTimer,
    select_timer,
)

__all__ = [
    "AutoTimer",
    "CostMetrics",
    "CudaEventTimer",
    "CudaGraphTimer",
    "GraphCaptureError",
    "GateResult",
    "ImplementationMeasurement",
    "PerformanceConfig",
    "PerformanceEvaluator",
    "PerformanceResult",
    "PerformanceGateConfig",
    "PreparedPerformance",
    "RawSample",
    "SampleStatistics",
    "TheoreticalCost",
    "Timer",
    "TimerConfig",
    "TimerError",
    "TimerSelection",
    "TimerUnsupportedError",
    "WallClockTimer",
    "compute_statistics",
    "evaluate_cost_model",
    "evaluate_performance_gate",
    "percentile",
    "select_timer",
]
