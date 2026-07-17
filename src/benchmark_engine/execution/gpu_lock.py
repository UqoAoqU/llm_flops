"""Conservative cross-process lock for one physical GPU."""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


class GpuLockError(RuntimeError):
    pass


class GpuLockTimeout(GpuLockError):
    pass


@dataclass(frozen=True)
class GpuIdentity:
    logical_device: str
    visible_device: str | None
    uuid: str
    name: str | None = None
    resolution_backend: str = "nvidia-smi"
    cuda_context_may_be_initialized: bool = False


def resolve_gpu_identity(
    logical_index: int = 0, *, allow_torch_fallback: bool = False
) -> GpuIdentity:
    """Resolve logical cuda:N to a physical UUID; never lock merely by index.

    ``nvidia-smi`` is deliberately queried before importing torch so the
    controller can acquire the GPU lock without creating a CUDA context.
    The torch fallback is opt-in because querying torch device properties may
    initialize a context; callers that enable it can inspect that risk on the
    returned identity and in :func:`collect_gpu_metadata`.
    """
    if isinstance(logical_index, bool) or not isinstance(logical_index, int) or logical_index < 0:
        raise ValueError("logical_index must be non-negative")
    if not isinstance(allow_torch_fallback, bool):
        raise TypeError("allow_torch_fallback must be a boolean")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    token: str | None = None
    if visible is not None:
        values = [part.strip() for part in visible.split(",") if part.strip()]
        if logical_index >= len(values):
            raise GpuLockError(f"cuda:{logical_index} is outside CUDA_VISIBLE_DEVICES")
        token = values[logical_index]
    nvidia_smi_error = "unknown failure"
    try:
        query = subprocess.run(
            ("nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader,nounits"),
            capture_output=True, text=True, check=False, timeout=10,
        )
        if query.returncode != 0:
            nvidia_smi_error = query.stderr.strip() or f"exit status {query.returncode}"
        else:
            rows = [
                parts
                for line in query.stdout.splitlines()
                if line.strip()
                for parts in [tuple(part.strip() for part in line.split(",", 2))]
                if len(parts) == 3
            ]
            if token and token.upper().startswith("GPU-"):
                selected = next((row for row in rows if row[1] == token), None)
            else:
                physical_index = token if token is not None else str(logical_index)
                selected = next((row for row in rows if row[0] == physical_index), None)
            if selected is not None:
                return GpuIdentity(
                    f"cuda:{logical_index}", token, selected[1], selected[2] or None
                )
            nvidia_smi_error = f"cannot map cuda:{logical_index} to a GPU UUID"
    except (OSError, subprocess.SubprocessError) as error:
        nvidia_smi_error = f"{type(error).__name__}: {error}"

    if not allow_torch_fallback:
        raise GpuLockError(f"cannot resolve GPU UUID with nvidia-smi: {nvidia_smi_error}")
    try:
        import torch

        if not torch.cuda.is_available():
            raise GpuLockError("CUDA is unavailable")
        props = torch.cuda.get_device_properties(logical_index)
        physical_uuid = getattr(props, "uuid", None)
        if not physical_uuid:
            raise GpuLockError("torch device properties do not expose a UUID")
        return GpuIdentity(
            f"cuda:{logical_index}",
            token,
            str(physical_uuid),
            str(getattr(props, "name", "")) or None,
            resolution_backend="torch",
            cuda_context_may_be_initialized=True,
        )
    except Exception as error:
        raise GpuLockError(
            "cannot resolve GPU UUID with nvidia-smi "
            f"({nvidia_smi_error}) or torch ({type(error).__name__}: {error})"
        ) from error


def _pid_alive(pid: int) -> bool | None:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return None
    except OSError:
        return None
    return True


class GpuLock:
    def __init__(self, root: Path, identity: GpuIdentity, *, run_id: str,
                 timeout_s: float = 600.0, poll_interval_s: float = 0.05,
                 clock=time.monotonic, sleeper=time.sleep) -> None:
        if timeout_s < 0 or poll_interval_s <= 0:
            raise ValueError("lock timeout must be non-negative and poll interval positive")
        self.root = Path(root)
        self.identity = identity
        self.run_id = run_id
        self.timeout_s = float(timeout_s)
        self.poll_interval_s = float(poll_interval_s)
        self.clock = clock
        self.sleeper = sleeper
        safe_uuid = "".join(ch if ch.isalnum() or ch in "-." else "_" for ch in identity.uuid)
        self.path = self.root / f"{safe_uuid}.lock"
        self.owner_token = uuid.uuid4().hex
        self._held = False

    def _metadata(self) -> dict[str, object]:
        return {"schema_version": 1, "pid": os.getpid(), "run_id": self.run_id,
                "started_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
                "logical_device": self.identity.logical_device,
                "visible_device": self.identity.visible_device,
                "gpu_uuid": self.identity.uuid, "owner_token": self.owner_token}

    def _try_clear_dead(self) -> bool:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return False
        if not isinstance(value, dict) or not isinstance(value.get("pid"), int):
            return False
        alive = _pid_alive(value["pid"])
        if alive is not False:
            return False
        try:
            current = json.loads(self.path.read_text(encoding="utf-8"))
            if current != value:
                return False
            self.path.unlink()
            return True
        except (OSError, UnicodeError, json.JSONDecodeError):
            return False

    def acquire(self) -> "GpuLock":
        self.root.mkdir(parents=True, exist_ok=True)
        deadline = self.clock() + self.timeout_s
        metadata = (json.dumps(self._metadata(), sort_keys=True) + "\n").encode()
        while True:
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                with os.fdopen(fd, "wb") as stream:
                    stream.write(metadata)
                    stream.flush()
                    os.fsync(stream.fileno())
                self._held = True
                return self
            except FileExistsError:
                self._try_clear_dead()
                if self.clock() >= deadline:
                    detail = "unreadable"
                    try: detail = self.path.read_text(encoding="utf-8")[:512].strip()
                    except OSError: pass
                    raise GpuLockTimeout(f"timed out waiting for {self.identity.uuid}; owner={detail}")
                self.sleeper(self.poll_interval_s)

    def release(self) -> None:
        if not self._held:
            return
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if value.get("owner_token") != self.owner_token or value.get("pid") != os.getpid():
                raise GpuLockError("refusing to release a GPU lock owned by another process")
            self.path.unlink()
            self._held = False
        except FileNotFoundError:
            raise GpuLockError("owned GPU lock disappeared")

    def __enter__(self) -> "GpuLock": return self.acquire()
    def __exit__(self, *_: object) -> None: self.release()


def collect_gpu_metadata(identity: GpuIdentity, *, ignore_pids: tuple[int, ...] = ()) -> dict[str, object]:
    """Best-effort telemetry. Unknown values are None, never fabricated zeroes."""
    result: dict[str, object] = {
        "logical_device": identity.logical_device,
        "visible_device": identity.visible_device,
        "gpu_uuid": identity.uuid,
        "gpu_name": identity.name,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "driver_version": None,
        "cuda_version": None,
        "other_compute_processes_detected": None,
        "telemetry_error": None,
    }
    errors: list[str] = []
    if identity.cuda_context_may_be_initialized:
        errors.append("gpu_identity_torch_fallback_may_initialize_cuda")
    try:
        import torch
        result["cuda_version"] = getattr(getattr(torch, "version", None), "cuda", None)
    except ImportError:
        errors.append("torch_unavailable")
    try:
        driver = subprocess.run(("nvidia-smi", "--query-gpu=uuid,driver_version",
                                 "--format=csv,noheader,nounits"), capture_output=True,
                                text=True, check=False, timeout=10)
        if driver.returncode == 0:
            for line in driver.stdout.splitlines():
                parts = [part.strip() for part in line.split(",", 1)]
                if len(parts) == 2 and parts[0] == identity.uuid:
                    result["driver_version"] = parts[1] or None
                    break
        else:
            errors.append("driver_query_failed")
        apps = subprocess.run(("nvidia-smi", "--query-compute-apps=gpu_uuid,pid",
                               "--format=csv,noheader,nounits"), capture_output=True,
                              text=True, check=False, timeout=10)
        if apps.returncode == 0:
            pids: list[int] = []
            for line in apps.stdout.splitlines():
                parts = [part.strip() for part in line.split(",", 1)]
                if len(parts) == 2 and parts[0] == identity.uuid:
                    try: pids.append(int(parts[1]))
                    except ValueError: errors.append("compute_pid_malformed")
            ignored = {os.getpid(), *ignore_pids}
            result["other_compute_processes_detected"] = any(pid not in ignored for pid in pids)
        else:
            errors.append("compute_process_query_failed")
    except (OSError, subprocess.SubprocessError) as error:
        errors.append(f"nvidia_smi:{type(error).__name__}")
    if errors:
        result["telemetry_error"] = ";".join(errors)
    return result


__all__ = ["GpuIdentity", "GpuLock", "GpuLockError", "GpuLockTimeout",
           "collect_gpu_metadata", "resolve_gpu_identity"]
