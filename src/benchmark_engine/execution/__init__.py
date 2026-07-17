"""Managed subprocess execution and strict JSON protocol contracts."""

from .controller import ControllerError, WorkerController, build_worker_request
from .event_log import EventEmitter, EventLog
from .isolation import IsolationError
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
]
