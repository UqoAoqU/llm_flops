"""Append-only structured worker event logs."""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path

from .isolation import linux_descendant_pids
from .protocol import (
    EventName,
    ProtocolError,
    WorkerEvent,
    WorkerRequest,
    WorkerStage,
    split_event_attempts,
    utc_timestamp,
)


class EventLog:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    def append(self, event: WorkerEvent) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = event.to_json().encode("utf-8")
        descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
        try:
            remaining = memoryview(data)
            while remaining:
                written = os.write(descriptor, remaining)
                if written <= 0:
                    raise OSError("short event-log write made no progress")
                remaining = remaining[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def read(self, *, tolerate_partial_last_line: bool = False) -> tuple[WorkerEvent, ...]:
        try:
            data = self.path.read_bytes()
        except FileNotFoundError:
            return ()
        lines = data.splitlines(keepends=True)
        events: list[WorkerEvent] = []
        for index, line in enumerate(lines):
            if not line.endswith((b"\n", b"\r")) and tolerate_partial_last_line and index == len(lines) - 1:
                break
            try:
                events.append(WorkerEvent.from_json(line.decode("utf-8")))
            except (UnicodeError, ProtocolError) as error:
                raise ProtocolError(f"invalid event log line {index + 1}: {error}") from error
        return tuple(events)

    def for_result(self, request: WorkerRequest) -> tuple[WorkerEvent, ...]:
        attempts = split_event_attempts(
            self.read(tolerate_partial_last_line=True)
        )
        matching = tuple(
            attempt
            for attempt in attempts
            if attempt[0].identity == request.identity
            and attempt[0].result_id == request.result_id
        )
        return () if not matching else matching[-1]


class EventEmitter:
    """Serialize lifecycle and heartbeat events from multiple worker threads."""

    def __init__(
        self,
        log: EventLog,
        request: WorkerRequest,
        *,
        initial_sequence: int = 1,
        started: float | None = None,
    ) -> None:
        self.log = log
        self.request = request
        self.started = time.monotonic() if started is None else started
        self._sequence = initial_sequence
        self._lock = threading.Lock()
        self._active_stage: WorkerStage | None = None
        self._stage_started = self.started

    @property
    def active_stage(self) -> WorkerStage | None:
        with self._lock:
            return self._active_stage

    def emit(
        self,
        event: EventName,
        stage: WorkerStage,
        status: str,
        *,
        message: str | None = None,
        child_pids: tuple[int, ...] = (),
        log_tail: str = "",
    ) -> WorkerEvent:
        with self._lock:
            if event.value.endswith("_STARTED"):
                self._active_stage = stage
                self._stage_started = time.monotonic()
            elapsed = max(0.0, time.monotonic() - self.started)
            value = WorkerEvent(
                sequence=self._sequence,
                event=event,
                timestamp_utc=utc_timestamp(),
                pid=os.getpid(),
                identity=self.request.identity,
                result_id=self.request.result_id,
                stage=stage,
                status=status,
                elapsed_s=elapsed,
                child_pids=child_pids,
                log_tail=log_tail,
                message=message,
            )
            self.log.append(value)
            self._sequence += 1
            if event.value.endswith("_FINISHED"):
                self._active_stage = None
            return value

    def heartbeat(self, log_tail: str) -> None:
        with self._lock:
            stage = self._active_stage
        if stage is None:
            return
        encoded = log_tail.encode("utf-8", errors="replace")[-16 * 1024 :]
        self.emit(
            EventName.HEARTBEAT,
            stage,
            "heartbeat",
            child_pids=linux_descendant_pids(os.getpid()),
            log_tail=encoded.decode("utf-8", errors="ignore"),
        )


__all__ = ["EventEmitter", "EventLog"]
