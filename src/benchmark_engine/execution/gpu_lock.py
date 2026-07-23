"""Conservative cross-process lock for one physical MI300X GPU."""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


KFD_TOPOLOGY_ROOT = Path("/sys/class/kfd/kfd/topology/nodes")
AMDGPU_VERSION_PATH = Path("/sys/module/amdgpu/version")
VISIBLE_DEVICE_VARIABLES = (
    "ROCR_VISIBLE_DEVICES",
    "HIP_VISIBLE_DEVICES",
    "CUDA_VISIBLE_DEVICES",
)


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
    resolution_backend: str = "kfd"
    cuda_context_may_be_initialized: bool = False
    accelerator_backend: str = "rocm"
    arch: str | None = None
    render_minor: int | None = None
    location_id: int | None = None


@dataclass(frozen=True)
class _KfdDevice:
    gpu_id: int
    location_id: int
    render_minor: int
    gfx_target_version: int

    @property
    def arch(self) -> str:
        major = self.gfx_target_version // 10000
        minor = (self.gfx_target_version // 100) % 100
        stepping = self.gfx_target_version % 100
        return f"gfx{major}{minor:x}{stepping:x}"

    @property
    def lock_id(self) -> str:
        return (
            f"AMD-KFD-{self.gpu_id}-{self.location_id}"
            f"-render{self.render_minor}"
        )


def _properties(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return values
    for line in lines:
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            values[parts[0]] = int(parts[1], 0)
        except ValueError:
            continue
    return values


def _kfd_devices(root: Path) -> tuple[_KfdDevice, ...]:
    devices: list[_KfdDevice] = []
    try:
        nodes = tuple(root.iterdir())
    except OSError:
        return ()
    for node in nodes:
        try:
            gpu_id = int((node / "gpu_id").read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        properties = _properties(node / "properties")
        if gpu_id <= 0 or properties.get("vendor_id") != 0x1002:
            continue
        required = ("location_id", "drm_render_minor", "gfx_target_version")
        if any(name not in properties for name in required):
            continue
        devices.append(
            _KfdDevice(
                gpu_id=gpu_id,
                location_id=properties["location_id"],
                render_minor=properties["drm_render_minor"],
                gfx_target_version=properties["gfx_target_version"],
            )
        )
    return tuple(sorted(devices, key=lambda item: item.render_minor))


def _visible_devices() -> tuple[str | None, tuple[str, ...] | None]:
    for name in VISIBLE_DEVICE_VARIABLES:
        value = os.environ.get(name)
        if value is not None:
            tokens = tuple(part.strip() for part in value.split(",") if part.strip())
            return name, tokens
    return None, None


def _resolve_kfd(logical_index: int, root: Path) -> GpuIdentity | None:
    devices = _kfd_devices(root)
    if not devices:
        return None
    _, visible = _visible_devices()
    token: str | None = None
    if visible is not None:
        if logical_index >= len(visible):
            raise GpuLockError(
                f"cuda:{logical_index} is outside the configured visible devices"
            )
        token = visible[logical_index]
        try:
            physical_index = int(token)
        except ValueError:
            selected = next(
                (item for item in devices if str(item.gpu_id) == token), None
            )
            if selected is None:
                return None
        else:
            if physical_index < 0 or physical_index >= len(devices):
                raise GpuLockError(
                    f"visible GPU index {physical_index} is outside KFD topology"
                )
            selected = devices[physical_index]
    else:
        if logical_index >= len(devices):
            raise GpuLockError(f"cuda:{logical_index} is outside KFD topology")
        selected = devices[logical_index]
    return GpuIdentity(
        logical_device=f"cuda:{logical_index}",
        visible_device=token,
        uuid=selected.lock_id,
        name=f"AMD {selected.arch}",
        resolution_backend="kfd",
        accelerator_backend="rocm",
        arch=selected.arch,
        render_minor=selected.render_minor,
        location_id=selected.location_id,
    )


def resolve_gpu_identity(
    logical_index: int = 0,
    *,
    allow_torch_fallback: bool = False,
    kfd_root: Path = KFD_TOPOLOGY_ROOT,
) -> GpuIdentity:
    """Resolve logical ``cuda:N`` to an MI300X KFD identity.

    PyTorch deliberately retains the ``cuda`` device namespace on ROCm.  KFD
    topology is preferred because it does not initialize a HIP context in the
    controller.  The explicit torch fallback is only used when KFD cannot map
    the configured visibility token.
    """

    if (
        isinstance(logical_index, bool)
        or not isinstance(logical_index, int)
        or logical_index < 0
    ):
        raise ValueError("logical_index must be non-negative")
    if not isinstance(allow_torch_fallback, bool):
        raise TypeError("allow_torch_fallback must be a boolean")
    identity = _resolve_kfd(logical_index, Path(kfd_root))
    if identity is not None:
        return identity
    if not allow_torch_fallback:
        raise GpuLockError("cannot resolve GPU identity from KFD topology")
    try:
        import torch

        if not torch.cuda.is_available():
            raise GpuLockError("ROCm accelerator is unavailable")
        properties = torch.cuda.get_device_properties(logical_index)
        physical_uuid = getattr(properties, "uuid", None)
        if not physical_uuid:
            raise GpuLockError("torch device properties do not expose a UUID")
        _, visible = _visible_devices()
        token = None if visible is None else visible[logical_index]
        return GpuIdentity(
            f"cuda:{logical_index}",
            token,
            str(physical_uuid),
            str(getattr(properties, "name", "")) or None,
            resolution_backend="torch",
            cuda_context_may_be_initialized=True,
            accelerator_backend=(
                "rocm"
                if getattr(getattr(torch, "version", None), "hip", None)
                else "cuda"
            ),
            arch=getattr(properties, "gcnArchName", None),
        )
    except Exception as error:
        raise GpuLockError(
            "cannot resolve GPU identity from KFD or torch "
            f"({type(error).__name__}: {error})"
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
    def __init__(
        self,
        root: Path,
        identity: GpuIdentity,
        *,
        run_id: str,
        timeout_s: float = 600.0,
        poll_interval_s: float = 0.05,
        clock=time.monotonic,
        sleeper=time.sleep,
    ) -> None:
        if timeout_s < 0 or poll_interval_s <= 0:
            raise ValueError(
                "lock timeout must be non-negative and poll interval positive"
            )
        self.root = Path(root)
        self.identity = identity
        self.run_id = run_id
        self.timeout_s = float(timeout_s)
        self.poll_interval_s = float(poll_interval_s)
        self.clock = clock
        self.sleeper = sleeper
        safe_uuid = "".join(
            ch if ch.isalnum() or ch in "-." else "_" for ch in identity.uuid
        )
        self.path = self.root / f"{safe_uuid}.lock"
        self.owner_token = uuid.uuid4().hex
        self._held = False

    def _metadata(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "pid": os.getpid(),
            "run_id": self.run_id,
            "started_at_utc": datetime.now(timezone.utc)
            .isoformat()
            .replace("+00:00", "Z"),
            "logical_device": self.identity.logical_device,
            "visible_device": self.identity.visible_device,
            "gpu_uuid": self.identity.uuid,
            "accelerator_backend": self.identity.accelerator_backend,
            "resolution_backend": self.identity.resolution_backend,
            "owner_token": self.owner_token,
        }

    def _try_clear_dead(self) -> bool:
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return False
        if not isinstance(value, dict) or not isinstance(value.get("pid"), int):
            return False
        if _pid_alive(value["pid"]) is not False:
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
                fd = os.open(
                    self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
                )
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
                    try:
                        detail = self.path.read_text(encoding="utf-8")[:512].strip()
                    except OSError:
                        pass
                    raise GpuLockTimeout(
                        f"timed out waiting for {self.identity.uuid}; owner={detail}"
                    )
                self.sleeper(self.poll_interval_s)

    def release(self) -> None:
        if not self._held:
            return
        try:
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if (
                value.get("owner_token") != self.owner_token
                or value.get("pid") != os.getpid()
            ):
                raise GpuLockError(
                    "refusing to release a GPU lock owned by another process"
                )
            self.path.unlink()
            self._held = False
        except FileNotFoundError:
            raise GpuLockError("owned GPU lock disappeared")

    def __enter__(self) -> "GpuLock":
        return self.acquire()

    def __exit__(self, *_: object) -> None:
        self.release()


def _driver_version(path: Path = AMDGPU_VERSION_PATH) -> str | None:
    try:
        value = path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    return value or None


def collect_gpu_metadata(
    identity: GpuIdentity, *, ignore_pids: tuple[int, ...] = ()
) -> dict[str, object]:
    """Collect ROCm provenance without fabricating unavailable process data."""

    del ignore_pids
    visible = {
        name: os.environ.get(name)
        for name in VISIBLE_DEVICE_VARIABLES
        if os.environ.get(name) is not None
    }
    result: dict[str, object] = {
        "logical_device": identity.logical_device,
        "visible_device": identity.visible_device,
        "gpu_uuid": identity.uuid,
        "gpu_name": identity.name,
        "gpu_arch": identity.arch,
        "gpu_identity_resolution": identity.resolution_backend,
        "accelerator_backend": identity.accelerator_backend,
        "accelerator_runtime_version": None,
        "visible_devices": visible,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "driver_version": _driver_version(),
        "cuda_version": None,
        "other_compute_processes_detected": None,
        "telemetry_error": None,
    }
    errors = ["rocm_process_query_unavailable"]
    if identity.cuda_context_may_be_initialized:
        errors.append("gpu_identity_torch_fallback_may_initialize_hip")
    try:
        import torch

        version = getattr(torch, "version", None)
        result["accelerator_runtime_version"] = getattr(version, "hip", None)
        result["cuda_version"] = getattr(version, "cuda", None)
        if torch.cuda.is_available():
            properties = torch.cuda.get_device_properties(0)
            result["gpu_name"] = (
                str(getattr(properties, "name", "")) or result["gpu_name"]
            )
            result["gpu_arch"] = (
                getattr(properties, "gcnArchName", None) or result["gpu_arch"]
            )
    except ImportError:
        errors.append("torch_unavailable")
    result["telemetry_error"] = ";".join(errors)
    return result


__all__ = [
    "GpuIdentity",
    "GpuLock",
    "GpuLockError",
    "GpuLockTimeout",
    "collect_gpu_metadata",
    "resolve_gpu_identity",
]
