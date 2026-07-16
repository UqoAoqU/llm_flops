"""Process-tree and bounded-output primitives used by managed workers.

This is failure isolation, not a security sandbox.  Candidate code still runs
with the worker user's filesystem, network and device permissions.
"""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from typing import BinaryIO


DEFAULT_LOG_LIMIT_BYTES = 8 * 1024 * 1024
DEFAULT_SUMMARY_LIMIT_BYTES = 16 * 1024
TRUNCATED_MARKER = b"\n[benchmark-engine: log truncated at configured byte limit]\n"


class IsolationError(RuntimeError):
    pass


def validated_root(path: Path, field: str) -> Path:
    raw = Path(path)
    if not raw.is_absolute():
        raise IsolationError(f"{field} must be absolute")
    if raw.is_symlink():
        raise IsolationError(f"{field} must not be a symlink")
    resolved = raw.resolve(strict=True)
    if not resolved.is_dir():
        raise IsolationError(f"{field} must be a directory")
    return resolved


def validated_artifact_root(path: Path) -> Path:
    raw = Path(path)
    if not raw.is_absolute():
        raise IsolationError("artifact_root must be absolute")
    raw.mkdir(parents=True, exist_ok=True)
    if raw.is_symlink():
        raise IsolationError("artifact_root must not be a symlink")
    return raw.resolve(strict=True)


def ensure_beneath(path: Path, root: Path, field: str) -> Path:
    resolved_root = validated_root(root, f"{field}.root")
    resolved = Path(path).resolve(strict=True)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise IsolationError(f"{field} escapes declared root") from error
    if resolved.is_symlink():
        raise IsolationError(f"{field} must not be a symlink")
    return resolved


def linux_descendant_pids(pid: int) -> tuple[int, ...]:
    """Return a best-effort recursive snapshot from Linux procfs."""

    if os.name != "posix" or not Path("/proc").is_dir():
        return ()
    found: set[int] = set()
    pending = [pid]
    while pending:
        parent = pending.pop()
        path = Path(f"/proc/{parent}/task/{parent}/children")
        try:
            children = tuple(int(item) for item in path.read_text().split())
        except (OSError, ValueError):
            continue
        for child in children:
            if child not in found:
                found.add(child)
                pending.append(child)
    return tuple(sorted(found))


def process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def terminate_process_group(
    process: subprocess.Popen[bytes], *, grace_s: float = 0.5
) -> int:
    """TERM then KILL the worker session and synchronously reap its leader."""

    if os.name != "posix":
        if process.poll() is None:
            process.terminate()
            try:
                return process.wait(timeout=grace_s)
            except subprocess.TimeoutExpired:
                process.kill()
        return process.wait()
    pgid = process.pid
    try:
        os.killpg(pgid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        time.sleep(0.01)
    # The leader may have exited while descendants remain in its process group.
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        return process.wait(timeout=max(grace_s, 0.1))
    except subprocess.TimeoutExpired:
        process.kill()
        return process.wait()


def cleanup_exited_process_group(pgid: int) -> None:
    """Kill descendants that survived a normally exiting worker leader."""

    if os.name != "posix":
        return
    try:
        os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass


class BoundedLogDrain:
    """Continuously drain a pipe into a capped file and capped memory tail."""

    def __init__(
        self,
        source: BinaryIO,
        destination: Path,
        *,
        file_limit_bytes: int = DEFAULT_LOG_LIMIT_BYTES,
        summary_limit_bytes: int = DEFAULT_SUMMARY_LIMIT_BYTES,
    ) -> None:
        if file_limit_bytes <= 0 or summary_limit_bytes <= 0:
            raise ValueError("log limits must be positive")
        if file_limit_bytes <= len(TRUNCATED_MARKER):
            raise ValueError("file log limit must leave room for truncation marker")
        self.source = source
        self.destination = Path(destination)
        self.file_limit_bytes = file_limit_bytes
        self.summary_limit_bytes = summary_limit_bytes
        self._tail = bytearray()
        self._tail_lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._error: BaseException | None = None
        self.truncated = False

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout: float | None = None) -> None:
        self._thread.join(timeout)
        if self._thread.is_alive():
            raise IsolationError("log drain thread did not finish")
        if self._error is not None:
            raise IsolationError(f"log drain failed: {self._error}") from self._error

    def tail(self) -> str:
        with self._tail_lock:
            return bytes(self._tail).decode("utf-8", errors="replace")

    def _remember(self, chunk: bytes) -> None:
        with self._tail_lock:
            self._tail.extend(chunk)
            if len(self._tail) > self.summary_limit_bytes:
                del self._tail[: len(self._tail) - self.summary_limit_bytes]

    def _run(self) -> None:
        self.destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.destination.touch(exist_ok=True)
            existing = self.destination.read_bytes()
            marker_written = TRUNCATED_MARKER in existing
            self.truncated = marker_written
            content_limit = self.file_limit_bytes - len(TRUNCATED_MARKER)
            written = min(len(existing), content_limit)
            with self.destination.open("ab") as output:
                while True:
                    chunk = self.source.read(64 * 1024)
                    if not chunk:
                        break
                    self._remember(chunk)
                    remaining = content_limit - written
                    if remaining > 0 and not marker_written:
                        part = chunk[:remaining]
                        output.write(part)
                        written += len(part)
                    if len(chunk) > max(remaining, 0):
                        self.truncated = True
                    if self.truncated and not marker_written:
                        output.write(TRUNCATED_MARKER)
                        marker_written = True
                    output.flush()
                os.fsync(output.fileno())
        except BaseException as error:
            self._error = error
        finally:
            try:
                self.source.close()
            except OSError:
                pass


def bounded_append_text(path: Path, text: str, *, limit_bytes: int) -> bool:
    """Append one complete UTF-8 record if it fits the evaluation-level cap."""

    data = text.encode("utf-8")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    current = path.stat().st_size if path.exists() else 0
    if current + len(data) > limit_bytes:
        return False
    with path.open("ab") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())
    return True


__all__ = [
    "DEFAULT_LOG_LIMIT_BYTES",
    "DEFAULT_SUMMARY_LIMIT_BYTES",
    "TRUNCATED_MARKER",
    "BoundedLogDrain",
    "IsolationError",
    "bounded_append_text",
    "cleanup_exited_process_group",
    "ensure_beneath",
    "linux_descendant_pids",
    "process_exists",
    "terminate_process_group",
    "validated_artifact_root",
    "validated_root",
]
