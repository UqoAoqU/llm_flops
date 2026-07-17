import json
import os
import tempfile
import unittest
import multiprocessing
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from benchmark_engine.execution.gpu_lock import (
    GpuIdentity, GpuLock, GpuLockError, GpuLockTimeout, collect_gpu_metadata,
    resolve_gpu_identity,
)


def _lock_worker(root, queue, start):
    identity = GpuIdentity("cuda:0", "0", "GPU-process", "test")
    start.wait()
    with GpuLock(Path(root), identity, run_id=str(os.getpid()), timeout_s=5):
        queue.put(("enter", os.getpid(), time.monotonic()))
        time.sleep(.1)
        queue.put(("exit", os.getpid(), time.monotonic()))


class GpuLockTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.identity = GpuIdentity("cuda:0", "0", "GPU-test", "test")

    def tearDown(self): self.temp.cleanup()

    def test_identity_prefers_nvidia_smi_without_importing_torch(self):
        query = SimpleNamespace(
            returncode=0,
            stdout="0, GPU-zero, first\n1, GPU-one, second\n",
            stderr="",
        )
        original_import = __import__

        def reject_torch_import(name, *args, **kwargs):
            if name == "torch":
                raise AssertionError("successful nvidia-smi resolution imported torch")
            return original_import(name, *args, **kwargs)

        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}), patch(
            "benchmark_engine.execution.gpu_lock.subprocess.run", return_value=query
        ), patch("builtins.__import__", side_effect=reject_torch_import):
            identity = resolve_gpu_identity(0)
        self.assertEqual(identity.uuid, "GPU-one")
        self.assertEqual(identity.visible_device, "1")
        self.assertEqual(identity.resolution_backend, "nvidia-smi")
        self.assertFalse(identity.cuda_context_may_be_initialized)

    def test_torch_identity_fallback_is_opt_in_and_visible_in_telemetry(self):
        props = SimpleNamespace(uuid="GPU-fallback", name="fallback")
        fake_torch = SimpleNamespace(
            cuda=SimpleNamespace(
                is_available=lambda: True,
                get_device_properties=lambda _index: props,
            ),
            version=SimpleNamespace(cuda="13.0"),
        )
        missing_smi = FileNotFoundError("nvidia-smi")
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}), patch.dict(
            "sys.modules", {"torch": fake_torch}
        ), patch(
            "benchmark_engine.execution.gpu_lock.subprocess.run",
            side_effect=missing_smi,
        ):
            with self.assertRaises(GpuLockError):
                resolve_gpu_identity(0)
            identity = resolve_gpu_identity(0, allow_torch_fallback=True)
            metadata = collect_gpu_metadata(identity)
        self.assertEqual(identity.resolution_backend, "torch")
        self.assertTrue(identity.cuda_context_may_be_initialized)
        self.assertIn(
            "gpu_identity_torch_fallback_may_initialize_cuda",
            metadata["telemetry_error"],
        )

    def test_atomic_metadata_and_owner_release(self):
        lock = GpuLock(self.root, self.identity, run_id="run-1", timeout_s=0)
        lock.acquire()
        metadata = json.loads(lock.path.read_text())
        self.assertEqual(metadata["pid"], os.getpid())
        self.assertEqual(metadata["gpu_uuid"], "GPU-test")
        self.assertEqual(metadata["logical_device"], "cuda:0")
        lock.release()
        self.assertFalse(lock.path.exists())

    def test_live_and_malformed_are_never_cleared(self):
        path = self.root / "GPU-test.lock"
        self.root.mkdir(exist_ok=True)
        path.write_text(json.dumps({"pid": os.getpid()}))
        with self.assertRaises(GpuLockTimeout):
            GpuLock(self.root, self.identity, run_id="other", timeout_s=0).acquire()
        path.write_text("broken")
        with self.assertRaises(GpuLockTimeout):
            GpuLock(self.root, self.identity, run_id="other", timeout_s=0).acquire()

    def test_only_explicitly_dead_owner_is_reclaimed(self):
        self.root.mkdir(exist_ok=True)
        path = self.root / "GPU-test.lock"
        path.write_text(json.dumps({"pid": 99999999, "owner_token": "old"}))
        with patch("benchmark_engine.execution.gpu_lock._pid_alive", return_value=False):
            lock = GpuLock(self.root, self.identity, run_id="new", timeout_s=.1).acquire()
        self.assertEqual(json.loads(path.read_text())["run_id"], "new")
        lock.release()

    def test_non_owner_cannot_release(self):
        lock = GpuLock(self.root, self.identity, run_id="run", timeout_s=0).acquire()
        value = json.loads(lock.path.read_text())
        value["owner_token"] = "replaced"
        lock.path.write_text(json.dumps(value))
        with self.assertRaises(GpuLockError): lock.release()

    def test_two_processes_never_overlap(self):
        context = multiprocessing.get_context("spawn")
        queue, start = context.Queue(), context.Event()
        processes = [context.Process(target=_lock_worker, args=(str(self.root), queue, start)) for _ in range(2)]
        for process in processes: process.start()
        start.set()
        records = [queue.get(timeout=10) for _ in range(4)]
        for process in processes:
            process.join(10)
            self.assertEqual(process.exitcode, 0)
        intervals = {}
        for kind, pid, timestamp in records: intervals.setdefault(pid, {})[kind] = timestamp
        values = sorted(intervals.values(), key=lambda item: item["enter"])
        self.assertLessEqual(values[0]["exit"], values[1]["enter"])


if __name__ == "__main__": unittest.main()
