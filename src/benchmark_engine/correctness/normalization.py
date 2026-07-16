"""Deterministic conversion of nested runtime outputs to path-addressed leaves."""

from __future__ import annotations

import dataclasses
import math
import re
from collections.abc import Mapping, Sequence

from .models import OutputBundle, OutputLeaf

_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class OutputNormalizationError(ValueError):
    pass


def _scalar_leaf(value: object, path: str) -> OutputLeaf:
    if isinstance(value, bool):
        dtype = "bool"
    elif isinstance(value, int):
        dtype = "int64"
    elif isinstance(value, float):
        dtype = "float64"
    else:
        raise OutputNormalizationError(f"unsupported output type {type(value).__name__} at {path}")
    return OutputLeaf(path, (value,), dtype, (), (), "scalar", "cpu")


def _flatten_values(value: object, path: str) -> tuple[object, ...]:
    try:
        detached = value.detach() if callable(getattr(value, "detach", None)) else value
        cpu = detached.cpu() if callable(getattr(detached, "cpu", None)) else detached
        flattened = cpu.reshape(-1) if callable(getattr(cpu, "reshape", None)) else cpu
        raw = flattened.tolist() if callable(getattr(flattened, "tolist", None)) else flattened
    except Exception as error:
        raise OutputNormalizationError(f"cannot read tensor/array values at {path}: {error}") from error
    if not isinstance(raw, list):
        raw = [raw]
    result: list[object] = []
    stack = list(reversed(raw))
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(reversed(item))
        elif isinstance(item, (bool, int, float)):
            result.append(item)
        else:
            raise OutputNormalizationError(f"unsupported array element {type(item).__name__} at {path}")
    return tuple(result)


def _array_leaf(value: object, path: str) -> OutputLeaf:
    try:
        shape = tuple(int(item) for item in getattr(value, "shape"))
    except Exception as error:
        raise OutputNormalizationError(f"invalid shape at {path}: {error}") from error
    dtype = str(getattr(value, "dtype", type(value).__name__))
    device = str(getattr(value, "device", "cpu"))
    stride_attr = getattr(value, "stride", None)
    stride: tuple[int, ...] | None = None
    if callable(stride_attr):
        try:
            stride = tuple(int(item) for item in stride_attr())
        except Exception as error:
            raise OutputNormalizationError(f"invalid stride at {path}: {error}") from error
    elif hasattr(value, "strides"):
        raw = getattr(value, "strides")
        itemsize = int(getattr(value, "itemsize", 1)) or 1
        stride = tuple(int(item) // itemsize for item in raw)
    layout_attr = getattr(value, "layout", None)
    if layout_attr is not None:
        layout = str(layout_attr)
    else:
        flags = getattr(value, "flags", None)
        if flags is not None and bool(getattr(flags, "c_contiguous", False)):
            layout = "contiguous"
        elif flags is not None and bool(getattr(flags, "f_contiguous", False)):
            layout = "fortran"
        else:
            layout = "strided"
    values = _flatten_values(value, path)
    expected = math.prod(shape) if shape else 1
    if len(values) != expected:
        raise OutputNormalizationError(f"value count does not match shape at {path}")
    return OutputLeaf(path, values, dtype, shape, stride, layout, device)


def _is_array(value: object) -> bool:
    return hasattr(value, "shape") and hasattr(value, "dtype") and (
        callable(getattr(value, "tolist", None)) or callable(getattr(value, "detach", None))
    )


def _visit(value: object, path: str, active: set[int], leaves: list[OutputLeaf]) -> None:
    if isinstance(value, (bool, int, float)):
        leaves.append(_scalar_leaf(value, path))
        return
    if value is None:
        raise OutputNormalizationError(f"unsupported output type NoneType at {path}")
    if _is_array(value):
        leaves.append(_array_leaf(value, path))
        return

    identity = id(value)
    if identity in active:
        raise OutputNormalizationError(f"cyclic output detected at {path}")
    active.add(identity)
    try:
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            for field in dataclasses.fields(value):
                if not _NAME.fullmatch(field.name):
                    raise OutputNormalizationError(f"unstable dataclass field at {path}")
                _visit(getattr(value, field.name), f"{path}.{field.name}", active, leaves)
            return
        if isinstance(value, tuple) and hasattr(value, "_fields"):
            fields = tuple(getattr(value, "_fields"))
            if len(fields) != len(set(fields)) or not all(isinstance(name, str) and _NAME.fullmatch(name) for name in fields):
                raise OutputNormalizationError(f"duplicate or unstable named fields at {path}")
            for name in fields:
                _visit(getattr(value, name), f"{path}.{name}", active, leaves)
            return
        if isinstance(value, Mapping):
            items = list(value.items())
            keys = [item[0] for item in items]
            if not all(isinstance(key, str) and _NAME.fullmatch(key) for key in keys):
                raise OutputNormalizationError(f"mapping keys must be stable identifiers at {path}")
            if len(keys) != len(set(keys)):
                raise OutputNormalizationError(f"duplicate mapping key at {path}")
            for key, item in sorted(items, key=lambda pair: pair[0]):
                _visit(item, f"{path}.{key}", active, leaves)
            return
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            for index, item in enumerate(value):
                _visit(item, f"{path}[{index}]", active, leaves)
            return
        raise OutputNormalizationError(f"unsupported output type {type(value).__name__} at {path}")
    finally:
        active.remove(identity)


def normalize_output(output: object, observed_state: Mapping[str, object] | None = None) -> OutputBundle:
    leaves: list[OutputLeaf] = []
    _visit(output, "output", set(), leaves)
    if observed_state is not None:
        if not isinstance(observed_state, Mapping):
            raise OutputNormalizationError("observed_state must be a mapping")
        for name, value in sorted(observed_state.items()):
            if not isinstance(name, str) or not _NAME.fullmatch(name):
                raise OutputNormalizationError("observed_state keys must be stable identifiers")
            _visit(value, f"state.{name}", set(), leaves)
    return OutputBundle(tuple(leaves))


def attach_observed_state(output: OutputBundle, observed_state: Mapping[str, object]) -> OutputBundle:
    state_bundle = normalize_output(0, observed_state)
    return OutputBundle(output.leaves + tuple(leaf for leaf in state_bundle.leaves if leaf.path.startswith("state.")))
