"""Timer implementations with explicit requested/effective provenance.

Timer instances are runtime-only worker objects.  Only :class:`RawSample` and
:class:`TimerSelection` dictionaries cross the JSON controller boundary.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


class TimerError(RuntimeError):
    """Base class for stable timer failures."""


class TimerUnsupportedError(TimerError):
    """The explicitly requested timer cannot be used for this callable."""


class GraphCaptureError(TimerUnsupportedError):
    """CUDA Graph capture was attempted and failed."""


@dataclass(frozen=True)
class TimerConfig:
    """Strict steady-state sampling configuration."""

    samples: int = 30
    inner_iterations: int = 20

    def __post_init__(self) -> None:
        for name in ("samples", "inner_iterations"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer")
            if value <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class RawSample:
    sample_index: int
    inner_iterations: int
    elapsed_ms: float
    per_call_ms: float

    def __post_init__(self) -> None:
        if isinstance(self.sample_index, bool) or not isinstance(self.sample_index, int):
            raise TypeError("sample_index must be an integer")
        if self.sample_index < 0:
            raise ValueError("sample_index must be non-negative")
        if isinstance(self.inner_iterations, bool) or not isinstance(self.inner_iterations, int):
            raise TypeError("inner_iterations must be an integer")
        if self.inner_iterations <= 0:
            raise ValueError("inner_iterations must be positive")
        for name in ("elapsed_ms", "per_call_ms"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a number")
            if not math.isfinite(float(value)) or float(value) < 0:
                raise ValueError(f"{name} must be finite and non-negative")

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_index": self.sample_index,
            "inner_iterations": self.inner_iterations,
            "elapsed_ms": float(self.elapsed_ms),
            "per_call_ms": float(self.per_call_ms),
        }


@dataclass(frozen=True)
class TimerSelection:
    requested_timer: str
    effective_timer: str
    fallback_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.requested_timer or not self.effective_timer:
            raise ValueError("requested/effective timer names must not be empty")
        if self.fallback_reason is not None and not self.fallback_reason:
            raise ValueError("fallback_reason must be non-empty or None")
        if self.requested_timer != "auto" and self.fallback_reason is not None:
            raise ValueError("only auto timer selection may record a fallback")

    def to_dict(self) -> dict[str, object]:
        return {
            "requested_timer": self.requested_timer,
            "effective_timer": self.effective_timer,
            "fallback_reason": self.fallback_reason,
        }


@runtime_checkable
class Timer(Protocol):
    """Runtime protocol implemented by all steady-state timers."""

    @property
    def selection(self) -> TimerSelection: ...

    def prepare(self, fn: Callable[[], object], config: TimerConfig) -> float: ...

    def sample(self, fn: Callable[[], object], config: TimerConfig) -> tuple[RawSample, ...]: ...


def _torch_cuda(torch_module: object | None = None):
    if torch_module is None:
        try:
            import torch as torch_module  # type: ignore
        except ImportError as error:
            raise TimerUnsupportedError("PyTorch is required for CUDA timers") from error
    cuda = getattr(torch_module, "cuda", None)
    if cuda is None or not callable(getattr(cuda, "is_available", None)):
        raise TimerUnsupportedError("PyTorch CUDA runtime is unavailable")
    if not cuda.is_available():
        raise TimerUnsupportedError("CUDA is unavailable")
    return torch_module, cuda


class WallClockTimer:
    """End-to-end launch plus synchronization timer."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.perf_counter,
        synchronizer: Callable[[], None] = lambda: None,
        requested_timer: str = "wall_clock",
    ) -> None:
        self._clock = clock
        self._synchronize = synchronizer
        self._selection = TimerSelection(requested_timer, "wall_clock")

    @property
    def selection(self) -> TimerSelection:
        return self._selection

    def prepare(self, fn: Callable[[], object], config: TimerConfig) -> float:
        del fn, config
        return 0.0

    def sample(self, fn: Callable[[], object], config: TimerConfig) -> tuple[RawSample, ...]:
        result: list[RawSample] = []
        for index in range(config.samples):
            self._synchronize()
            started = self._clock()
            for _ in range(config.inner_iterations):
                fn()
            self._synchronize()
            elapsed_ms = (self._clock() - started) * 1000.0
            if not math.isfinite(elapsed_ms) or elapsed_ms < 0:
                raise TimerError("wall clock produced a non-finite or negative duration")
            result.append(
                RawSample(index, config.inner_iterations, elapsed_ms, elapsed_ms / config.inner_iterations)
            )
        return tuple(result)


class CudaEventTimer:
    """GPU elapsed timer using one CUDA event pair per raw sample."""

    def __init__(self, *, torch_module: object | None = None, requested_timer: str = "cuda_event") -> None:
        self._torch = torch_module
        self._selection = TimerSelection(requested_timer, "cuda_event")

    @property
    def selection(self) -> TimerSelection:
        return self._selection

    def prepare(self, fn: Callable[[], object], config: TimerConfig) -> float:
        del fn, config
        _torch_cuda(self._torch)
        return 0.0

    def sample(self, fn: Callable[[], object], config: TimerConfig) -> tuple[RawSample, ...]:
        _, cuda = _torch_cuda(self._torch)
        result: list[RawSample] = []
        for index in range(config.samples):
            start = cuda.Event(enable_timing=True)
            end = cuda.Event(enable_timing=True)
            start.record()
            for _ in range(config.inner_iterations):
                fn()
            end.record()
            # Synchronizing the end event both closes the measurement and
            # attributes asynchronous launch failures to this sample.
            end.synchronize()
            elapsed_ms = float(start.elapsed_time(end))
            result.append(
                RawSample(index, config.inner_iterations, elapsed_ms, elapsed_ms / config.inner_iterations)
            )
        return tuple(result)


class CudaGraphTimer:
    """Capture a fixed callable once, then time graph replays with events."""

    def __init__(
        self,
        *,
        torch_module: object | None = None,
        clock: Callable[[], float] = time.perf_counter,
        requested_timer: str = "cuda_graph",
    ) -> None:
        self._torch = torch_module
        self._clock = clock
        self._selection = TimerSelection(requested_timer, "cuda_graph")
        self._graph: object | None = None
        self._callable_identity: int | None = None

    @property
    def selection(self) -> TimerSelection:
        return self._selection

    def prepare(self, fn: Callable[[], object], config: TimerConfig) -> float:
        torch_module, cuda = _torch_cuda(self._torch)
        try:
            cuda.synchronize()
            graph = cuda.CUDAGraph()
            started = self._clock()
            with cuda.graph(graph):
                # Preserve the validated legacy graph_ms semantics: one graph
                # contains the complete inner-iteration loop.  A raw sample
                # replays that graph once, then divides by inner_iterations.
                for _ in range(config.inner_iterations):
                    fn()
            cuda.synchronize()
            elapsed_ms = (self._clock() - started) * 1000.0
        except BaseException as error:
            raise GraphCaptureError(
                f"CUDA Graph capture failed: {type(error).__name__}: {error}"
            ) from error
        if not math.isfinite(elapsed_ms) or elapsed_ms < 0:
            raise GraphCaptureError("CUDA Graph capture produced an invalid duration")
        self._graph = graph
        self._callable_identity = id(fn)
        # Keep the module alive for injected/fake runtimes too.
        self._torch = torch_module
        return elapsed_ms

    def sample(self, fn: Callable[[], object], config: TimerConfig) -> tuple[RawSample, ...]:
        if self._graph is None or self._callable_identity != id(fn):
            raise TimerError("CudaGraphTimer.sample requires prepare with the same callable")
        _, cuda = _torch_cuda(self._torch)
        result: list[RawSample] = []
        for index in range(config.samples):
            start = cuda.Event(enable_timing=True)
            end = cuda.Event(enable_timing=True)
            start.record()
            self._graph.replay()
            end.record()
            end.synchronize()
            elapsed_ms = float(start.elapsed_time(end))
            result.append(
                RawSample(index, config.inner_iterations, elapsed_ms, elapsed_ms / config.inner_iterations)
            )
        return tuple(result)


class AutoTimer:
    """Attempt CUDA Graph capture and visibly fall back to CUDA events."""

    def __init__(
        self,
        *,
        graph_factory: Callable[[], Timer] = CudaGraphTimer,
        event_factory: Callable[[], Timer] = CudaEventTimer,
    ) -> None:
        self._graph_factory = graph_factory
        self._event_factory = event_factory
        self._timer: Timer = graph_factory()
        self._selection = TimerSelection("auto", "cuda_graph")

    @property
    def selection(self) -> TimerSelection:
        return self._selection

    def prepare(self, fn: Callable[[], object], config: TimerConfig) -> float:
        try:
            capture_ms = self._timer.prepare(fn, config)
        except TimerUnsupportedError as error:
            reason = f"{type(error).__name__}: {error}"[:512]
            self._timer = self._event_factory()
            self._timer.prepare(fn, config)
            self._selection = TimerSelection("auto", "cuda_event", reason)
            return 0.0
        self._selection = TimerSelection("auto", self._timer.selection.effective_timer)
        return capture_ms

    def sample(self, fn: Callable[[], object], config: TimerConfig) -> tuple[RawSample, ...]:
        return self._timer.sample(fn, config)


def select_timer(
    requested: str,
    *,
    torch_module: object | None = None,
    clock: Callable[[], float] = time.perf_counter,
    synchronizer: Callable[[], None] = lambda: None,
) -> Timer:
    """Return a timer for one canonical name; aliases are rejected."""

    if requested == "wall_clock":
        return WallClockTimer(clock=clock, synchronizer=synchronizer)
    if requested == "cuda_event":
        return CudaEventTimer(torch_module=torch_module)
    if requested == "cuda_graph":
        return CudaGraphTimer(torch_module=torch_module, clock=clock)
    if requested == "auto":
        return AutoTimer(
            graph_factory=lambda: CudaGraphTimer(torch_module=torch_module, clock=clock),
            event_factory=lambda: CudaEventTimer(torch_module=torch_module),
        )
    raise ValueError(
        "timer must be one of: auto, cuda_event, cuda_graph, wall_clock"
    )


__all__ = [
    "AutoTimer",
    "CudaEventTimer",
    "CudaGraphTimer",
    "GraphCaptureError",
    "RawSample",
    "Timer",
    "TimerConfig",
    "TimerError",
    "TimerSelection",
    "TimerUnsupportedError",
    "WallClockTimer",
    "select_timer",
]
