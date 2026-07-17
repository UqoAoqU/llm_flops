import unittest

from benchmark_engine.performance import (
    CudaEventTimer,
    CudaGraphTimer,
    GraphCaptureError,
    TimerConfig,
)


try:
    import torch
except ImportError:  # pragma: no cover - environment dependent
    torch = None


@unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA required")
class GpuTimerSmokeTests(unittest.TestCase):
    def setUp(self):
        self.left = torch.randn(64, device="cuda")
        self.right = torch.randn(64, device="cuda")
        self.output = torch.empty_like(self.left)

    def kernel(self):
        torch.add(self.left, self.right, out=self.output)

    def test_cuda_event_retains_every_raw_sample(self):
        timer = CudaEventTimer()
        timer.prepare(self.kernel, TimerConfig(3, 2))
        samples = timer.sample(self.kernel, TimerConfig(3, 2))
        self.assertEqual(len(samples), 3)
        self.assertTrue(all(sample.elapsed_ms >= 0 for sample in samples))

    def test_cuda_graph_replays_captured_inner_loop(self):
        for _ in range(3):
            self.kernel()
        torch.cuda.synchronize()
        timer = CudaGraphTimer()
        capture_ms = timer.prepare(self.kernel, TimerConfig(3, 2))
        samples = timer.sample(self.kernel, TimerConfig(3, 2))
        self.assertGreaterEqual(capture_ms, 0)
        self.assertEqual(len(samples), 3)
        torch.testing.assert_close(self.output, self.left + self.right)

    def test_explicit_graph_capture_failure_is_not_silent(self):
        timer = CudaGraphTimer()

        def fails():
            raise RuntimeError("deliberate capture failure")

        with self.assertRaises(GraphCaptureError):
            timer.prepare(fails, TimerConfig(1, 1))


if __name__ == "__main__":
    unittest.main()
