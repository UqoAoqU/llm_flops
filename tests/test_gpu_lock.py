import json
import multiprocessing
import os
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmark_engine.execution.gpu_lock import (
    GpuIdentity,
    GpuLock,
    GpuLockError,
    GpuLockTimeout,
    collect_gpu_metadata,
    resolve_gpu_identity,
)


def _lock_worker(root, queue, start):
    identity = GpuIdentity("cuda:0", "0", "AMD-KFD-process", "test")
    start.wait()
    with GpuLock(Path(root), identity, run_id=str(os.getpid()), timeout_s=5):
        queue.put(("enter", os.getpid(), time.monotonic()))
        time.sleep(0.1)
        queue.put(("exit", os.getpid(), time.monotonic()))


def _write_kfd_device(
    root: Path,
    node: int,
    *,
    gpu_id: int,
    location_id: int,
    render_minor: int,
) -> None:
    path = root / str(node)
    path.mkdir(parents=True)
    (path / "gpu_id").write_text(str(gpu_id), encoding="utf-8")
    (path / "properties").write_text(
        "\n".join(
            (
                "vendor_id 4098",
                "device_id 29857",
                f"location_id {location_id}",
                f"drm_render_minor {render_minor}",
                "gfx_target_version 90402",
            )
        )
        + "\n",
        encoding="utf-8",
    )


class GpuLockTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.identity = GpuIdentity("cuda:0", "0", "AMD-KFD-test", "test")

    def tearDown(self):
        self.temp.cleanup()

    def test_identity_uses_kfd_without_importing_torch(self):
        topology = self.root / "nodes"
        _write_kfd_device(
            topology, 2, gpu_id=111, location_id=200, render_minor=136
        )
        _write_kfd_device(
            topology, 3, gpu_id=222, location_id=100, render_minor=128
        )
        original_import = __import__

        def reject_torch_import(name, *args, **kwargs):
            if name == "torch":
                raise AssertionError("successful KFD resolution imported torch")
            return original_import(name, *args, **kwargs)

        with patch.dict(
            os.environ,
            {
                "ROCR_VISIBLE_DEVICES": "1",
                "HIP_VISIBLE_DEVICES": "0",
                "CUDA_VISIBLE_DEVICES": "0",
            },
            clear=True,
        ), patch("builtins.__import__", side_effect=reject_torch_import):
            identity = resolve_gpu_identity(0, kfd_root=topology)
        self.assertEqual(identity.uuid, "AMD-KFD-111-200-render136")
        self.assertEqual(identity.visible_device, "1")
        self.assertEqual(identity.resolution_backend, "kfd")
        self.assertEqual(identity.arch, "gfx942")
        self.assertFalse(identity.cuda_context_may_be_initialized)

    def test_torch_identity_fallback_is_opt_in_and_visible(self):
        properties = SimpleNamespace(
            uuid="AMD-fallback",
            name="AMD Radeon Graphics",
            gcnArchName="gfx942:sramecc+:xnack-",
        )
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                get_device_properties=lambda _index: properties,
            ),
            version=SimpleNamespace(hip="7.0.51831", cuda=None),
        )
        missing = self.root / "missing"
        with patch.dict(os.environ, {"HIP_VISIBLE_DEVICES": "0"}, clear=True), patch.dict(
            "sys.modules", {"torch": fake_torch}
        ):
            with self.assertRaises(GpuLockError):
                resolve_gpu_identity(0, kfd_root=missing)
            identity = resolve_gpu_identity(
                0, allow_torch_fallback=True, kfd_root=missing
            )
            metadata = collect_gpu_metadata(identity)
        self.assertEqual(identity.resolution_backend, "torch")
        self.assertTrue(identity.cuda_context_may_be_initialized)
        self.assertEqual(metadata["accelerator_backend"], "rocm")
        self.assertEqual(metadata["gpu_identity_resolution"], "torch")
        self.assertIn(
            "gpu_identity_torch_fallback_may_initialize_hip",
            metadata["telemetry_error"],
        )

    def test_atomic_metadata_and_owner_release(self):
        lock = GpuLock(self.root, self.identity, run_id="run-1", timeout_s=0)
        lock.acquire()
        metadata = json.loads(lock.path.read_text())
        self.assertEqual(metadata["pid"], os.getpid())
        self.assertEqual(metadata["gpu_uuid"], "AMD-KFD-test")
        self.assertEqual(metadata["accelerator_backend"], "rocm")
        self.assertEqual(metadata["resolution_backend"], "kfd")
        lock.release()
        self.assertFalse(lock.path.exists())

    def test_live_and_malformed_are_never_cleared(self):
        path = self.root / "AMD-KFD-test.lock"
        self.root.mkdir(exist_ok=True)
        path.write_text(json.dumps({"pid": os.getpid()}))
        with self.assertRaises(GpuLockTimeout):
            GpuLock(self.root, self.identity, run_id="other", timeout_s=0).acquire()
        path.write_text("broken")
        with self.assertRaises(GpuLockTimeout):
            GpuLock(self.root, self.identity, run_id="other", timeout_s=0).acquire()

    def test_only_explicitly_dead_owner_is_reclaimed(self):
        self.root.mkdir(exist_ok=True)
        path = self.root / "AMD-KFD-test.lock"
        path.write_text(json.dumps({"pid": 99999999, "owner_token": "old"}))
        with patch(
            "benchmark_engine.execution.gpu_lock._pid_alive", return_value=False
        ):
            lock = GpuLock(
                self.root, self.identity, run_id="new", timeout_s=0.1
            ).acquire()
        self.assertEqual(json.loads(path.read_text())["run_id"], "new")
        lock.release()

    def test_non_owner_cannot_release(self):
        lock = GpuLock(
            self.root, self.identity, run_id="run", timeout_s=0
        ).acquire()
        value = json.loads(lock.path.read_text())
        value["owner_token"] = "replaced"
        lock.path.write_text(json.dumps(value))
        with self.assertRaises(GpuLockError):
            lock.release()

    def test_two_processes_never_overlap(self):
        context = multiprocessing.get_context("spawn")
        queue, start = context.Queue(), context.Event()
        processes = [
            context.Process(
                target=_lock_worker, args=(str(self.root), queue, start)
            )
            for _ in range(2)
        ]
        for process in processes:
            process.start()
        start.set()
        records = [queue.get(timeout=10) for _ in range(4)]
        for process in processes:
            process.join(10)
            self.assertEqual(process.exitcode, 0)
        intervals = {}
        for kind, pid, timestamp in records:
            intervals.setdefault(pid, {})[kind] = timestamp
        values = sorted(intervals.values(), key=lambda item: item["enter"])
        self.assertLessEqual(values[0]["exit"], values[1]["enter"])


if __name__ == "__main__":
    unittest.main()
