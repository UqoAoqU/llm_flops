"""Managed subprocess execution and strict JSON protocol contracts."""

from .controller import ControllerError, WorkerController, build_worker_request
from .event_log import EventEmitter, EventLog
from .isolation import IsolationError
from .gpu_lock import GpuIdentity, GpuLock, GpuLockError, GpuLockTimeout, collect_gpu_metadata, resolve_gpu_identity
from .protocol import (
    PROTOCOL_SCHEMA_VERSION,
    WORKER_REQUEST_SCHEMA_VERSION,
    EventName,
    ProtocolError,
    StageTimeouts,
    WorkerEvent,
    WorkerOutcome,
    WorkerRequest,
    WorkerResponse,
    WorkerStage,
    validate_event_sequence,
)

__all__ = [
    "PROTOCOL_SCHEMA_VERSION",
    "WORKER_REQUEST_SCHEMA_VERSION",
    "ControllerError",
    "EventEmitter",
    "EventLog",
    "EventName",
    "IsolationError",
    "GpuIdentity",
    "GpuLock",
    "GpuLockError",
    "GpuLockTimeout",
    "collect_gpu_metadata",
    "ProtocolError",
    "StageTimeouts",
    "WorkerController",
    "WorkerEvent",
    "WorkerOutcome",
    "WorkerRequest",
    "WorkerResponse",
    "WorkerStage",
    "build_worker_request",
    "validate_event_sequence",
    "resolve_gpu_identity",
]
