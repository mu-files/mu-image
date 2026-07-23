# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files
"""Graph IR: emit ops, flush barriers, and engine-agnostic compute orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from ..tensor import NUMPY_FROM_DTYPE, Tensor, TensorMeta
from .base import get_default_engine

OutMetaFn = Callable[[Tensor, Dict[str, Any]], Any]


@dataclass(frozen=True)
class OpMeta:
    """Static catalog facts for an engine op (not dependent on a Tensor)."""

    name: str
    # Optional scheduler hint for a future executor; not part of the graph IR.
    granularity: str = "full_image"  # "span" | "tile" | "full_image"
    halo: int = 0


@dataclass(frozen=True)
class EngineOp:
    """Callable engine op + metadata. Public names live in ``engines.ops``."""

    meta: OpMeta
    _out_dtype: OutMetaFn
    _out_channels: OutMetaFn
    _in_channels: Optional[int]  # None = any
    _attr_specs: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)

    def __call__(self, x: Tensor, /, **attrs: Any) -> Tensor:
        return emit(self, x, **attrs)

    def infer_out_meta(self, x: Tensor, attrs: Dict[str, Any]) -> TensorMeta:
        return TensorMeta(
            dtype=self._out_dtype(x, attrs),
            height=x.meta.height,
            width=x.meta.width,
            channels=self._out_channels(x, attrs),
        )

    def __repr__(self) -> str:
        return f"EngineOp({self.meta.name!r})"


def _out_dtype_same(x: Tensor, attrs: Dict[str, Any]) -> str:
    return x.meta.dtype


def _out_dtype_const(dtype: str) -> OutMetaFn:
    if dtype not in NUMPY_FROM_DTYPE:
        raise ValueError(f"unsupported output dtype {dtype!r}")

    def _fn(x: Tensor, attrs: Dict[str, Any]) -> str:
        return dtype

    return _fn


def _out_dtype_from_attr(key: str) -> OutMetaFn:
    def _fn(x: Tensor, attrs: Dict[str, Any]) -> str:
        val = attrs.get(key)
        if not isinstance(val, str) or val not in NUMPY_FROM_DTYPE:
            raise ValueError(
                f"attr {key!r} must be a dtype string "
                f"(one of {sorted(NUMPY_FROM_DTYPE)}), got {val!r}"
            )
        return val

    return _fn


def _out_channels_same(x: Tensor, attrs: Dict[str, Any]) -> int:
    return x.meta.channels


def _out_channels_const(n: int) -> OutMetaFn:
    def _fn(x: Tensor, attrs: Dict[str, Any]) -> int:
        return n

    return _fn


@dataclass
class OpNode:
    op: str
    inputs: Tuple[Tensor, ...]
    attrs: Dict[str, Any]
    out_meta: TensorMeta


def _as_f32_array(value: Any, *, name: str, size: Optional[int] = None) -> np.ndarray:
    arr = np.ascontiguousarray(value, dtype=np.float32).reshape(-1)
    if size is not None and arr.size != size:
        raise ValueError(f"{name} must have {size} elements, got {arr.size}")
    if arr.size < 1:
        raise ValueError(f"{name} must be non-empty")
    return arr


def _as_f64_array(value: Any, *, name: str, size: Optional[int] = None) -> np.ndarray:
    arr = np.ascontiguousarray(value, dtype=np.float64).reshape(-1)
    if size is not None and arr.size != size:
        raise ValueError(f"{name} must have {size} elements, got {arr.size}")
    if arr.size < 1:
        raise ValueError(f"{name} must be non-empty")
    return arr


def _as_i32_array(value: Any, *, name: str, size: Optional[int] = None) -> np.ndarray:
    arr = np.ascontiguousarray(value, dtype=np.int32).reshape(-1)
    if size is not None and arr.size != size:
        raise ValueError(f"{name} must have {size} elements, got {arr.size}")
    if arr.size < 1:
        raise ValueError(f"{name} must be non-empty")
    return arr


def _attr_optional(spec: Dict[str, Any]) -> bool:
    return bool(spec.get("optional"))


def _coerce_attr(spec: Dict[str, Any], value: Any) -> Any:
    """Coerce a Python attr value to the catalog wire form."""
    key = spec["key"]
    typ = spec["type"]
    count = spec.get("count", 1)
    if typ == "f32":
        if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
            raise TypeError(f"attr {key!r} must be a float")
        return np.float32(value)
    if typ == "f64":
        if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
            raise TypeError(f"attr {key!r} must be a float")
        return np.float64(value)
    if typ == "i32":
        if isinstance(value, bool) or not isinstance(value, (int, float, np.integer)):
            raise TypeError(f"attr {key!r} must be an int")
        return int(value)
    if typ == "bool":
        if not isinstance(value, (bool, np.bool_, int)):
            raise TypeError(f"attr {key!r} must be a bool")
        return int(bool(value))
    if typ == "string":
        if not isinstance(value, str):
            raise TypeError(f"attr {key!r} must be a str")
        return value
    if typ == "f32_array":
        size = count if count else None
        return _as_f32_array(value, name=key, size=size)
    if typ == "f64_array":
        size = count if count else None
        return _as_f64_array(value, name=key, size=size)
    if typ == "i32_array":
        size = count if count else None
        return _as_i32_array(value, name=key, size=size)
    raise ValueError(f"attr {key!r}: unsupported catalog type {typ!r}")


def _validate_attrs(
    name: str, specs: Tuple[Dict[str, Any], ...], attrs: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate/coerce attrs against the op's attr specs; reject unknown keys."""
    by_key = {s["key"]: s for s in specs}
    unknown = set(attrs) - set(by_key)
    if unknown:
        raise ValueError(f"op {name!r}: unknown attrs {sorted(unknown)}")
    required = {k for k, s in by_key.items() if not _attr_optional(s)}
    missing = required - set(attrs)
    if missing:
        raise ValueError(f"op {name!r}: missing attrs {sorted(missing)}")
    out: Dict[str, Any] = {}
    for k, v in attrs.items():
        if v is None and _attr_optional(by_key[k]):
            continue
        out[k] = _coerce_attr(by_key[k], v)
    return out


def emit(engine_op: EngineOp, x: Tensor, /, **attrs: Any) -> Tensor:
    """Validate attrs, ask the op for output meta, and build a lazy node."""
    name = engine_op.meta.name
    if engine_op._in_channels is not None and x.meta.channels != engine_op._in_channels:
        raise ValueError(
            f"op {name!r} input[0]: expected {engine_op._in_channels} channel(s), "
            f"got {x.meta.channels}"
        )
    coerced = _validate_attrs(name, engine_op._attr_specs, attrs)
    out_meta = engine_op.infer_out_meta(x, coerced)
    node = OpNode(
        op=name,
        inputs=(x,),
        attrs=coerced,
        out_meta=out_meta,
    )
    return Tensor(_meta=out_meta, _node=node)


def op(name: str, x: Tensor, /, **attrs: Any) -> Tensor:
    """Emit a named engine op (thin alias over ``engines.ops.OPS_BY_NAME``)."""
    from .ops import OPS_BY_NAME

    engine_op = OPS_BY_NAME.get(name)
    if engine_op is None:
        raise ValueError(f"unknown engine op {name!r}")
    return emit(engine_op, x, **attrs)


def flush(x: Tensor) -> Tensor:
    """Materialize a lazy engine graph into a concrete source Tensor.

    Use before non-engine (Python/ndarray) work, then continue with ``op`` /
    ``Tensor`` as needed.
    """
    return Tensor(x.compute())


def _reachable_tensors(root: Tensor) -> List[Tensor]:
    """Post-order DFS → topological order for a DAG."""
    ordered: List[Tensor] = []
    visiting: set[int] = set()
    done: set[int] = set()

    def visit(t: Tensor) -> None:
        tid = id(t)
        if tid in done:
            return
        if tid in visiting:
            raise ValueError("cycle detected in compute graph")
        visiting.add(tid)
        if t._node is not None:
            for inp in t._node.inputs:
                visit(inp)
        visiting.remove(tid)
        done.add(tid)
        ordered.append(t)

    visit(root)
    return ordered


def _segment_boundary_outputs(
    nodes: List[Tensor],
    all_op_tensors: List[Tensor],
    root: Tensor,
) -> List[Tensor]:
    """Tensors produced in this segment that escape to later consumers or root."""
    node_set = {id(t) for t in nodes}
    outs: List[Tensor] = []
    seen: set[int] = set()
    for t in nodes:
        tid = id(t)
        if tid in seen:
            continue
        needed = t is root
        if not needed:
            for u in all_op_tensors:
                if id(u) in node_set or u._node is None:
                    continue
                if any(inp is t for inp in u._node.inputs):
                    needed = True
                    break
        if needed:
            outs.append(t)
            seen.add(tid)
    return outs


def compute(root: Tensor) -> np.ndarray:
    """Topo-sort reachable nodes and execute via the default Engine."""
    tensors = _reachable_tensors(root)
    values: Dict[int, np.ndarray] = {}

    for t in tensors:
        if t._node is None:
            if t._data is None:
                raise ValueError("source Tensor has no data")
            values[id(t)] = np.ascontiguousarray(t._data)

    op_tensors = [t for t in tensors if t._node is not None]
    if not op_tensors:
        return values[id(root)]

    outs = _segment_boundary_outputs(op_tensors, op_tensors, root)
    if not outs:
        outs = [op_tensors[-1]]
    get_default_engine().execute_segment(op_tensors, values, outs)

    if id(root) not in values:
        raise RuntimeError("compute finished without materializing root")
    return values[id(root)]
