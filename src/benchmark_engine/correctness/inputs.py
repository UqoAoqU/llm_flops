"""Deterministic input generation, cloning, and summaries."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import random
from dataclasses import dataclass
from typing import Mapping

from benchmark_engine.models import CaseSpec

from .models import InputBundle

GENERATOR_VERSION = "benchmark-engine-rng-v1"


@dataclass(frozen=True)
class GeneratorContext:
    seed: int
    cpu: object
    cuda: Mapping[str, object]
    version: str = GENERATOR_VERSION


def make_generator_context(seed: int, cuda_devices: tuple[str, ...] = ()) -> GeneratorContext:
    """Create explicit CPU/CUDA generators without requiring torch on CPU hosts."""

    cpu: object = random.Random(seed)
    cuda: dict[str, object] = {}
    try:
        import torch  # type: ignore
    except ImportError:
        if cuda_devices:
            raise RuntimeError("CUDA generators require torch")
        return GeneratorContext(seed=seed, cpu=cpu, cuda=cuda)
    torch_cpu = torch.Generator(device="cpu")
    torch_cpu.manual_seed(seed)
    cpu = torch_cpu
    for index, device in enumerate(cuda_devices):
        if not torch.cuda.is_available():
            raise RuntimeError(f"CUDA generator requested for unavailable device {device}")
        generator = torch.Generator(device=device)
        generator.manual_seed(seed + index)
        cuda[device] = generator
    return GeneratorContext(seed=seed, cpu=cpu, cuda=cuda)


def _clone(value: object, memo: dict[int, object]) -> object:
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return value
    identity = id(value)
    if identity in memo:
        return memo[identity]
    clone = getattr(value, "clone", None)
    if callable(clone) and hasattr(value, "shape"):
        result = clone()
        memo[identity] = result
        return result
    if isinstance(value, tuple):
        result = tuple(_clone(item, memo) for item in value)
        memo[identity] = result
        return result
    if isinstance(value, list):
        result_list: list[object] = []
        memo[identity] = result_list
        result_list.extend(_clone(item, memo) for item in value)
        return result_list
    if isinstance(value, dict):
        result_dict: dict[object, object] = {}
        memo[identity] = result_dict
        for key, item in value.items():
            result_dict[_clone(key, memo)] = _clone(item, memo)
        return result_dict
    return copy.deepcopy(value, memo)


def clone_input_bundle(inputs: InputBundle) -> InputBundle:
    memo: dict[int, object] = {}
    return InputBundle(
        args=_clone(inputs.args, memo),  # type: ignore[arg-type]
        kwargs=_clone(inputs.kwargs, memo),  # type: ignore[arg-type]
        observed_state=_clone(inputs.observed_state, memo),  # type: ignore[arg-type]
    )


def _mutable_nodes(value: object, path: str, seen: set[int]) -> list[tuple[str, object]]:
    if value is None or isinstance(value, (bool, int, float, str, bytes)):
        return []
    identity = id(value)
    if identity in seen:
        return []
    seen.add(identity)
    nodes: list[tuple[str, object]] = []
    if hasattr(value, "shape") and (
        callable(getattr(value, "clone", None))
        or hasattr(value, "__array_interface__")
    ):
        return [(path, value)]
    if isinstance(value, dict):
        nodes.append((path, value))
        for key, item in value.items():
            nodes.extend(_mutable_nodes(item, f"{path}.{key}", seen))
    elif isinstance(value, list):
        nodes.append((path, value))
        for index, item in enumerate(value):
            nodes.extend(_mutable_nodes(item, f"{path}[{index}]", seen))
    elif isinstance(value, tuple):
        for index, item in enumerate(value):
            nodes.extend(_mutable_nodes(item, f"{path}[{index}]", seen))
    return nodes


def _storage_identity(value: object) -> object | None:
    """Return a root-storage identity for a non-empty torch-like tensor."""

    storage_method = getattr(value, "untyped_storage", None)
    if callable(storage_method):
        try:
            storage = storage_method()
            pointer = int(storage.data_ptr())
            size = int(storage.nbytes())
            if pointer == 0 or size == 0:
                return None
            return ("torch", str(getattr(value, "device", "")), pointer, size)
        except Exception:
            return None
    data_ptr = getattr(value, "data_ptr", None)
    if callable(data_ptr):
        try:
            pointer = int(data_ptr())
            size = int(getattr(value, "numel")()) if callable(getattr(value, "numel", None)) else 1
            if pointer == 0 or size == 0:
                return None
            return ("tensor", str(getattr(value, "device", "")), pointer)
        except Exception:
            return None
    return None


def _array_interval(value: object) -> tuple[int, int] | None:
    interface = getattr(value, "__array_interface__", None)
    if not isinstance(interface, dict):
        return None
    try:
        pointer = int(interface["data"][0])
        shape = tuple(int(item) for item in interface.get("shape", ()))
        if pointer == 0 or any(size == 0 for size in shape):
            return None
        itemsize = int(getattr(getattr(value, "dtype", None), "itemsize", 1))
        strides = interface.get("strides")
        if strides is None:
            byte_count = max(1, math.prod(shape)) * itemsize
            return pointer, pointer + byte_count
        lower = upper = 0
        for size, stride in zip(shape, strides):
            extent = (size - 1) * int(stride)
            lower += min(0, extent)
            upper += max(0, extent)
        return pointer + lower, pointer + upper + itemsize
    except (KeyError, TypeError, ValueError, OverflowError):
        return None


def _shares_storage(left: object, right: object) -> bool:
    left_storage = _storage_identity(left)
    right_storage = _storage_identity(right)
    if left_storage is not None and left_storage == right_storage:
        return True
    if hasattr(left, "__array_interface__") and hasattr(right, "__array_interface__"):
        try:
            import numpy  # type: ignore

            return bool(numpy.shares_memory(left, right))
        except ImportError:
            pass
        except (TypeError, ValueError, OverflowError):
            pass
        left_interval = _array_interval(left)
        right_interval = _array_interval(right)
        if left_interval is not None and right_interval is not None:
            return max(left_interval[0], right_interval[0]) < min(left_interval[1], right_interval[1])
    return False


def assert_input_isolation(left: InputBundle, right: InputBundle) -> None:
    """Reject shared mutable containers or tensor storage between two clones."""

    left_nodes = _mutable_nodes((left.args, left.kwargs, left.observed_state), "inputs", set())
    right_nodes = _mutable_nodes((right.args, right.kwargs, right.observed_state), "inputs", set())
    right_ids = {id(value): path for path, value in right_nodes}
    for path, value in left_nodes:
        if id(value) in right_ids:
            raise ValueError(f"input clones share mutable object at {path}")
        for right_path, right_value in right_nodes:
            if _shares_storage(value, right_value):
                raise ValueError(
                    f"input clones share tensor storage or array buffer at {path} and {right_path}"
                )


_MAX_SUMMARY_DEPTH = 8
_MAX_SUMMARY_ITEMS = 64
_MAX_SUMMARY_TEXT = 256


def _bounded_text(value: str) -> dict[str, object]:
    return {
        "value": value[:_MAX_SUMMARY_TEXT],
        "size": len(value),
        "truncated": len(value) > _MAX_SUMMARY_TEXT,
    }


def _float_token(value: float) -> object:
    if math.isnan(value):
        return "NaN"
    if value == math.inf:
        return "+Inf"
    if value == -math.inf:
        return "-Inf"
    return value


def _mapping_key_token(value: object, depth: int = 0) -> str:
    """Encode type-tagged mapping keys without ``str(key)`` collisions."""

    if depth > _MAX_SUMMARY_DEPTH:
        raise TypeError("mapping key nesting exceeds the summary depth limit")
    if value is None:
        encoded: object = ["none", None]
    elif isinstance(value, bool):
        encoded = ["bool", value]
    elif isinstance(value, int):
        encoded = ["int", str(value)]
    elif isinstance(value, float):
        encoded = ["float", _float_token(value)]
    elif isinstance(value, str):
        encoded = [
            "str",
            {
                "prefix": value[:_MAX_SUMMARY_TEXT],
                "size": len(value),
                "sha256": hashlib.sha256(value.encode("utf-8")).hexdigest(),
            },
        ]
    elif isinstance(value, bytes):
        encoded = [
            "bytes",
            {
                "prefix_hex": value[:_MAX_SUMMARY_TEXT].hex(),
                "size": len(value),
                "sha256": hashlib.sha256(value).hexdigest(),
            },
        ]
    elif isinstance(value, tuple):
        tokens = [_mapping_key_token(item, depth + 1) for item in value]
        digest = hashlib.sha256("\0".join(tokens).encode("utf-8")).hexdigest()
        encoded = [
            "tuple",
            {"items": tokens[:_MAX_SUMMARY_ITEMS], "size": len(tokens), "sha256": digest},
        ]
    else:
        raise TypeError(f"mapping key type {type(value).__name__} is not stable")
    return json.dumps(encoded, ensure_ascii=True, separators=(",", ":"), allow_nan=False)


def _summary(value: object, active: set[int], depth: int = 0) -> object:
    if isinstance(value, float):
        return {"type": "float", "value": _float_token(value)}
    if value is None or isinstance(value, (bool, int)):
        return {"type": type(value).__name__, "value": value}
    if isinstance(value, str):
        return {"type": "str", **_bounded_text(value)}
    if isinstance(value, bytes):
        prefix = value[:_MAX_SUMMARY_TEXT]
        return {
            "type": "bytes",
            "value_hex": prefix.hex(),
            "size": len(value),
            "truncated": len(value) > _MAX_SUMMARY_TEXT,
        }
    if depth >= _MAX_SUMMARY_DEPTH:
        try:
            size = len(value) if hasattr(value, "__len__") else None
        except Exception:
            size = None
        return {"type": type(value).__name__, "size": size, "truncated": True, "reason": "depth"}
    identity = id(value)
    if identity in active:
        return {"type": type(value).__name__, "cycle": True}
    active.add(identity)
    try:
        if hasattr(value, "shape") and hasattr(value, "dtype"):
            shape = [int(item) for item in getattr(value, "shape")]
            return {
                "type": type(value).__name__,
                "dtype": str(getattr(value, "dtype"))[:_MAX_SUMMARY_TEXT],
                "shape": shape[:_MAX_SUMMARY_ITEMS],
                "shape_rank": len(shape),
                "shape_truncated": len(shape) > _MAX_SUMMARY_ITEMS,
                "device": str(getattr(value, "device", "cpu"))[:_MAX_SUMMARY_TEXT],
            }
        if isinstance(value, Mapping):
            encoded_items = [(_mapping_key_token(key), item) for key, item in value.items()]
            encoded_items.sort(key=lambda pair: pair[0])
            tokens = [token for token, _ in encoded_items]
            if len(tokens) != len(set(tokens)):
                raise ValueError("mapping keys collide under stable summary encoding")
            retained = encoded_items[:_MAX_SUMMARY_ITEMS]
            return {
                "type": type(value).__name__,
                "size": len(encoded_items),
                "truncated": len(encoded_items) > len(retained),
                "entries": [
                    {"key": token, "value": _summary(item, active, depth + 1)}
                    for token, item in retained
                ],
            }
        if isinstance(value, (list, tuple)):
            retained = value[:_MAX_SUMMARY_ITEMS]
            return {
                "type": type(value).__name__,
                "size": len(value),
                "truncated": len(value) > len(retained),
                "items": [_summary(item, active, depth + 1) for item in retained],
            }
        return {"type": type(value).__name__}
    finally:
        active.remove(identity)


def input_summary(inputs: InputBundle) -> dict[str, object]:
    return {
        "args": _summary(inputs.args, set()),
        "kwargs": _summary(inputs.kwargs, set()),
        "observed_state": _summary(inputs.observed_state, set()),
    }


def case_fingerprint(case: CaseSpec, summary: Mapping[str, object]) -> str:
    payload = {"case": case.to_dict(), "generator_version": GENERATOR_VERSION, "input_summary": summary}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
