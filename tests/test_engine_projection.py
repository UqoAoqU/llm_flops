from __future__ import annotations

import unittest

from benchmark_engine.engine import (
    _environment_package_version,
    _manifest_cuda_devices,
    _manifest_device,
)


class EngineProjectionTests(unittest.TestCase):
    def test_device_projection_comes_from_manifest_not_wall_clock_timer(self):
        self.assertEqual(_manifest_device(("cuda",)), "cuda")
        self.assertEqual(_manifest_cuda_devices(("cuda",)), ("cuda:0",))
        self.assertEqual(_manifest_device(("cpu",)), "cpu")
        self.assertEqual(_manifest_cuda_devices(("cpu",)), ())
        self.assertEqual(_manifest_device(("cuda", "cpu", "cuda")), "cpu+cuda")

    def test_torch_version_uses_nested_collector_shape_and_missing_is_empty(self):
        observed = {
            "packages": {
                "torch": {"version": "2.11.0", "module": "torch"},
            }
        }
        self.assertEqual(_environment_package_version(observed, "torch"), "2.11.0")
        for malformed in ({}, {"packages": None}, {"packages": {"torch": None}}):
            with self.subTest(observed=malformed):
                self.assertIsNone(_environment_package_version(malformed, "torch"))


if __name__ == "__main__":
    unittest.main()
