import contextlib
import unittest

from benchmark_engine.performance import (
    AutoTimer,
    CudaEventTimer,
    CudaGraphTimer,
    GraphCaptureError,
    RawSample,
    TimerConfig,
    TimerSelection,
    TimerError,
    TimerUnsupportedError,
)


class StubTimer:
    def __init__(self, effective, *, error=None):
        self._selection = TimerSelection(effective, effective)
        self.error = error
        self.prepared = 0

    @property
    def selection(self):
        return self._selection

    def prepare(self, fn, config):
        self.prepared += 1
        if self.error:
            raise self.error
        return 7.0

    def sample(self, fn, config):
        return (RawSample(0, config.inner_iterations, 2.0, 1.0),)


class FakeEvent:
    def __init__(self, cuda, *, synchronize_error=None):
        self.cuda = cuda
        self.value = 0.0
        self.synchronize_error = synchronize_error

    def record(self):
        self.value = self.cuda.elapsed

    def synchronize(self):
        if self.synchronize_error is not None:
            raise self.synchronize_error

    def elapsed_time(self, other):
        return other.value - self.value


class FakeGraph:
    def __init__(self, cuda):
        self.cuda = cuda
        self.delta = 0.0
        self.replay_count = 0

    def replay(self):
        self.replay_count += 1
        self.cuda.elapsed += self.delta


class FakeCuda:
    def __init__(self, *, fail_capture=False, fail_sync=False):
        self.elapsed = 0.0
        self.fail_capture = fail_capture
        self.fail_sync = fail_sync
        self.last_graph = None

    def is_available(self):
        return True

    def synchronize(self):
        return None

    def CUDAGraph(self):
        self.last_graph = FakeGraph(self)
        return self.last_graph

    @contextlib.contextmanager
    def graph(self, graph):
        if self.fail_capture:
            raise RuntimeError("capture not supported")
        before = self.elapsed
        yield
        graph.delta = self.elapsed - before

    def Event(self, enable_timing=True):
        return FakeEvent(
            self,
            synchronize_error=(RuntimeError("async kernel failure") if self.fail_sync else None),
        )


class FakeTorch:
    def __init__(self, cuda):
        self.cuda = cuda


class TimerSelectionTests(unittest.TestCase):
    def test_auto_graph_success_records_requested_and_effective(self):
        graph = StubTimer("cuda_graph")
        timer = AutoTimer(graph_factory=lambda: graph, event_factory=lambda: StubTimer("cuda_event"))
        self.assertEqual(timer.prepare(lambda: None, TimerConfig(1, 1)), 7.0)
        self.assertEqual(timer.selection, TimerSelection("auto", "cuda_graph"))

    def test_auto_fallback_is_explicit_and_prepares_event(self):
        event = StubTimer("cuda_event")
        timer = AutoTimer(
            graph_factory=lambda: StubTimer(
                "cuda_graph", error=GraphCaptureError("dynamic shape")
            ),
            event_factory=lambda: event,
        )
        self.assertEqual(timer.prepare(lambda: None, TimerConfig(1, 1)), 0.0)
        self.assertEqual(timer.selection.requested_timer, "auto")
        self.assertEqual(timer.selection.effective_timer, "cuda_event")
        self.assertIn("dynamic shape", timer.selection.fallback_reason)
        self.assertEqual(event.prepared, 1)

    def test_explicit_graph_capture_failure_does_not_fallback(self):
        cuda = FakeCuda(fail_capture=True)
        timer = CudaGraphTimer(torch_module=FakeTorch(cuda))
        with self.assertRaises(GraphCaptureError):
            timer.prepare(lambda: None, TimerConfig(1, 2))

    def test_graph_captures_inner_loop_and_replays_once_per_sample(self):
        cuda = FakeCuda()
        calls = 0

        def kernel():
            nonlocal calls
            calls += 1
            cuda.elapsed += 2.0

        timer = CudaGraphTimer(
            torch_module=FakeTorch(cuda), clock=iter([0.0, 0.001]).__next__
        )
        self.assertEqual(timer.prepare(kernel, TimerConfig(2, 3)), 1.0)
        self.assertEqual(calls, 3)
        samples = timer.sample(kernel, TimerConfig(2, 3))
        self.assertEqual(cuda.last_graph.replay_count, 2)
        self.assertEqual([item.elapsed_ms for item in samples], [6.0, 6.0])
        self.assertEqual([item.per_call_ms for item in samples], [2.0, 2.0])

    def test_graph_accepts_repeated_access_to_same_bound_method(self):
        cuda = FakeCuda()

        class Kernel:
            def run(self):
                cuda.elapsed += 1.0

        kernel = Kernel()
        timer = CudaGraphTimer(
            torch_module=FakeTorch(cuda), clock=iter([0.0, 0.001]).__next__
        )
        timer.prepare(kernel.run, TimerConfig(1, 1))
        self.assertEqual(len(timer.sample(kernel.run, TimerConfig(1, 1))), 1)

    def test_graph_rejects_different_instance_and_different_bound_method(self):
        cuda = FakeCuda()

        class Kernel:
            def first(self):
                cuda.elapsed += 1.0

            def second(self):
                cuda.elapsed += 1.0

        prepared = Kernel()
        timer = CudaGraphTimer(
            torch_module=FakeTorch(cuda), clock=iter([0.0, 0.001]).__next__
        )
        timer.prepare(prepared.first, TimerConfig(1, 1))
        for different in (Kernel().first, prepared.second):
            with self.subTest(callable=different), self.assertRaisesRegex(
                TimerError, "same callable"
            ):
                timer.sample(different, TimerConfig(1, 1))

    def test_graph_rejects_a_distinct_ordinary_function(self):
        cuda = FakeCuda()

        def make_kernel():
            def kernel():
                cuda.elapsed += 1.0

            return kernel

        prepared = make_kernel()
        different = make_kernel()
        timer = CudaGraphTimer(
            torch_module=FakeTorch(cuda), clock=iter([0.0, 0.001]).__next__
        )
        timer.prepare(prepared, TimerConfig(1, 1))
        with self.assertRaisesRegex(TimerError, "same callable"):
            timer.sample(different, TimerConfig(1, 1))

    def test_cuda_event_sync_attributes_async_failure(self):
        cuda = FakeCuda(fail_sync=True)
        timer = CudaEventTimer(torch_module=FakeTorch(cuda))
        with self.assertRaisesRegex(RuntimeError, "async kernel failure"):
            timer.sample(lambda: None, TimerConfig(1, 1))


if __name__ == "__main__":
    unittest.main()
