"""Reference/candidate staged performance evaluation inside one worker."""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from benchmark_engine.correctness.inputs import assert_input_isolation, make_generator_context
from benchmark_engine.correctness.models import InputBundle
from benchmark_engine.correctness.evaluator import synchronize_cuda
from benchmark_engine.models import CaseSpec

from .cost_model import CostMetrics, evaluate_cost_model
from .statistics import SampleStatistics, compute_statistics
from .timers import RawSample, Timer, TimerConfig, TimerSelection, select_timer


@dataclass(frozen=True)
class PerformanceConfig:
    requested_timer: str = "auto"
    warmup: int = 5
    samples: int = 30
    inner_iterations: int = 20
    minimum_stable_samples: int = 5
    maximum_cv: float = 0.1

    def __post_init__(self) -> None:
        if self.requested_timer not in {"auto", "cuda_event", "cuda_graph", "wall_clock"}:
            raise ValueError(
                "requested_timer must be one of: auto, cuda_event, cuda_graph, wall_clock"
            )
        for name in ("warmup", "samples", "inner_iterations", "minimum_stable_samples"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if name == "warmup":
                if value < 0:
                    raise ValueError("warmup must be non-negative")
            elif value <= 0:
                raise ValueError(f"{name} must be positive")
        if isinstance(self.maximum_cv, bool) or not isinstance(self.maximum_cv, (int, float)):
            raise TypeError("maximum_cv must be a number")
        if not math.isfinite(float(self.maximum_cv)) or self.maximum_cv < 0:
            raise ValueError("maximum_cv must be finite and non-negative")


@dataclass(frozen=True)
class ImplementationMeasurement:
    role: str
    selection: TimerSelection
    first_call_ms: float
    warmup_ms: float
    graph_capture_ms: float
    steady_state_ms: float
    samples: tuple[RawSample, ...]
    statistics: SampleStatistics

    def __post_init__(self) -> None:
        if self.role not in {"reference", "candidate"}:
            raise ValueError("role must be reference or candidate")

    def to_dict(self) -> dict[str, object]:
        return {
            "role": self.role,
            "selection": self.selection.to_dict(),
            "first_call_ms": self.first_call_ms,
            "warmup_ms": self.warmup_ms,
            "graph_capture_ms": self.graph_capture_ms,
            "steady_state_ms": self.steady_state_ms,
            "samples": [sample.to_dict() for sample in self.samples],
            "statistics": self.statistics.to_dict(),
        }


@dataclass(frozen=True)
class PerformanceResult:
    status: str
    reference: ImplementationMeasurement
    candidate: ImplementationMeasurement
    cost: CostMetrics

    def __post_init__(self) -> None:
        if self.status not in {"pass", "unstable"}:
            raise ValueError("performance status must be pass or unstable")

    def to_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "measurements": {
                "reference": self.reference.to_dict(),
                "candidate": self.candidate.to_dict(),
            },
            "cost": self.cost.to_dict(),
        }


@dataclass
class _PreparedImplementation:
    role: str
    timer: Timer
    invoke: Callable[[], object]
    sampling: TimerConfig
    first_call_ms: float
    warmup_ms: float
    graph_capture_ms: float


@dataclass
class PreparedPerformance:
    """Runtime-only boundary between preparation and steady sampling."""

    reference: _PreparedImplementation
    candidate: _PreparedImplementation
    spec: object
    case: CaseSpec
    config: PerformanceConfig


class PerformanceEvaluator:
    """Measure both roles sequentially without Phase-9 ranking/fair ordering."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.perf_counter,
        synchronizer: Callable[[object, InputBundle], None] = synchronize_cuda,
        timer_factory: Callable[[str, Callable[[], None]], Timer] | None = None,
    ) -> None:
        self._clock = clock
        self._synchronizer = synchronizer
        self._timer_factory = timer_factory or (
            lambda requested, sync: select_timer(
                requested, clock=self._clock, synchronizer=sync
            )
        )

    def _prepare_implementation(
        self,
        role: str,
        fn: Callable[..., object],
        inputs: InputBundle,
        config: PerformanceConfig,
    ) -> _PreparedImplementation:
        latest: list[object] = [None]

        def invoke() -> object:
            latest[0] = fn(*inputs.args, **inputs.kwargs)
            return latest[0]

        def synchronize() -> None:
            self._synchronizer(latest[0], inputs)

        sampling = TimerConfig(config.samples, config.inner_iterations)
        timer = self._timer_factory(config.requested_timer, synchronize)

        synchronize()
        started = self._clock()
        invoke()
        synchronize()
        first_call_ms = (self._clock() - started) * 1000.0

        synchronize()
        started = self._clock()
        for _ in range(config.warmup):
            invoke()
        synchronize()
        warmup_ms = (self._clock() - started) * 1000.0

        graph_capture_ms = timer.prepare(invoke, sampling)
        for name, value in (
            ("first_call_ms", first_call_ms),
            ("warmup_ms", warmup_ms),
            ("graph_capture_ms", graph_capture_ms),
        ):
            if not math.isfinite(value) or value < 0:
                raise RuntimeError(f"{name} is non-finite or negative")
        return _PreparedImplementation(
            role=role,
            timer=timer,
            invoke=invoke,
            sampling=sampling,
            first_call_ms=first_call_ms,
            warmup_ms=warmup_ms,
            graph_capture_ms=graph_capture_ms,
        )

    def _sample_implementation(
        self, prepared: _PreparedImplementation, config: PerformanceConfig
    ) -> ImplementationMeasurement:
        samples = prepared.timer.sample(prepared.invoke, prepared.sampling)
        steady_state_ms = sum(sample.elapsed_ms for sample in samples)
        statistics = compute_statistics(
            (sample.per_call_ms for sample in samples),
            minimum_stable_samples=config.minimum_stable_samples,
            maximum_cv=config.maximum_cv,
        )
        if not math.isfinite(steady_state_ms) or steady_state_ms < 0:
            raise RuntimeError("steady_state_ms is non-finite or negative")
        return ImplementationMeasurement(
            role=prepared.role,
            selection=prepared.timer.selection,
            first_call_ms=prepared.first_call_ms,
            warmup_ms=prepared.warmup_ms,
            graph_capture_ms=prepared.graph_capture_ms,
            steady_state_ms=steady_state_ms,
            samples=samples,
            statistics=statistics,
        )

    def prepare(
        self,
        *,
        spec: object,
        reference: Callable[..., object],
        candidate: Callable[..., object],
        case: CaseSpec,
        config: PerformanceConfig,
        cuda_devices: tuple[str, ...] = (),
    ) -> PreparedPerformance:
        context = make_generator_context(case.seed, cuda_devices)
        canonical = spec.make_inputs(case, context)
        if not isinstance(canonical, InputBundle):
            raise TypeError("OperatorSpec.make_inputs() must return correctness.InputBundle")
        reference_inputs = spec.clone_inputs(canonical)
        candidate_inputs = spec.clone_inputs(canonical)
        if not isinstance(reference_inputs, InputBundle) or not isinstance(candidate_inputs, InputBundle):
            raise TypeError("OperatorSpec.clone_inputs() must return correctness.InputBundle")
        assert_input_isolation(canonical, reference_inputs)
        assert_input_isolation(canonical, candidate_inputs)
        assert_input_isolation(reference_inputs, candidate_inputs)

        reference_prepared = self._prepare_implementation(
            "reference", reference, reference_inputs, config
        )
        candidate_prepared = self._prepare_implementation(
            "candidate", candidate, candidate_inputs, config
        )
        return PreparedPerformance(
            reference_prepared, candidate_prepared, spec, case, config
        )

    def sample(self, prepared: PreparedPerformance) -> PerformanceResult:
        reference_measurement = self._sample_implementation(
            prepared.reference, prepared.config
        )
        candidate_measurement = self._sample_implementation(
            prepared.candidate, prepared.config
        )
        model_value = prepared.spec.cost_model(prepared.case)
        cost = evaluate_cost_model(
            model_value, candidate_measurement.statistics.median_ms
        )
        unstable = (
            reference_measurement.statistics.unstable
            or candidate_measurement.statistics.unstable
        )
        return PerformanceResult(
            "unstable" if unstable else "pass",
            reference_measurement,
            candidate_measurement,
            cost,
        )

    def evaluate(
        self,
        *,
        spec: object,
        reference: Callable[..., object],
        candidate: Callable[..., object],
        case: CaseSpec,
        config: PerformanceConfig,
        cuda_devices: tuple[str, ...] = (),
    ) -> PerformanceResult:
        prepared = self.prepare(
            spec=spec,
            reference=reference,
            candidate=candidate,
            case=case,
            config=config,
            cuda_devices=cuda_devices,
        )
        return self.sample(prepared)


__all__ = [
    "ImplementationMeasurement",
    "PerformanceConfig",
    "PerformanceEvaluator",
    "PerformanceResult",
    "PreparedPerformance",
]
