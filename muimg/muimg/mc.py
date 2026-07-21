# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files
"""
muimg.mc — lazy compute-graph API (Phase A.5–A.6 + A′).

Graph/engine glue only: build a lazy DAG of **engine** catalog ops, serialize
to the A.4 dict IR, and call muimg._compute_engine.

Non-engine work does not live in the graph: call ``flush(x)`` (or ``x.compute()``)
then run ordinary ndarray functions, and wrap results with ``Tensor(...)``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

_DTYPE_FROM_NUMPY = {
    np.dtype(np.float32): "float32",
    np.dtype(np.float16): "float16",
    np.dtype(np.uint8): "uint8",
    np.dtype(np.uint16): "uint16",
}

_NUMPY_FROM_DTYPE = {
    "float32": np.float32,
    "float16": np.float16,
    "uint8": np.uint8,
    "uint16": np.uint16,
}


@dataclass(frozen=True)
class TensorMeta:
    dtype: str
    height: int
    width: int
    channels: int

    @property
    def shape(self) -> Tuple[int, ...]:
        if self.channels == 1:
            return (self.height, self.width)
        return (self.height, self.width, self.channels)


@dataclass
class _OpNode:
    op: str
    inputs: Tuple["Tensor", ...]
    attrs: Dict[str, Any]
    out_meta: TensorMeta


def _meta_from_array(arr: np.ndarray) -> TensorMeta:
    if arr.ndim == 2:
        h, w = arr.shape
        channels = 1
    elif arr.ndim == 3:
        h, w, channels = arr.shape
        if channels not in (1, 3, 4):
            raise ValueError(f"unsupported channel count: {channels}")
    else:
        raise ValueError("array must be (H,W) or (H,W,C)")
    dtype = _DTYPE_FROM_NUMPY.get(arr.dtype)
    if dtype is None:
        raise TypeError(f"unsupported ndarray dtype: {arr.dtype}")
    return TensorMeta(dtype=dtype, height=h, width=w, channels=channels)


def _require_scalar(value: Any, op: str) -> float:
    if isinstance(value, Tensor):
        raise TypeError(f"{op}: tensor–tensor arithmetic not supported in Phase A")
    try:
        return float(value)
    except (TypeError, ValueError) as e:
        raise TypeError(f"{op}: RHS must be a scalar") from e


def _as_f32_array(value: Any, *, name: str, size: Optional[int] = None) -> np.ndarray:
    arr = np.ascontiguousarray(value, dtype=np.float32).reshape(-1)
    if size is not None and arr.size != size:
        raise ValueError(f"{name} must have {size} elements, got {arr.size}")
    if arr.size < 1:
        raise ValueError(f"{name} must be non-empty")
    return arr


class Tensor:
    """Lazy tensor handle: either a concrete source buffer or an engine op result."""

    __slots__ = ("_meta", "_data", "_node")

    def __init__(
        self,
        data: Optional[np.ndarray] = None,
        *,
        _meta: Optional[TensorMeta] = None,
        _node: Optional[_OpNode] = None,
    ):
        if data is not None:
            if _node is not None:
                raise ValueError("source Tensor cannot also have an op node")
            arr = np.asarray(data)
            self._meta = _meta_from_array(arr)
            self._data = arr
            self._node = None
        elif _meta is not None and _node is not None:
            self._meta = _meta
            self._data = None
            self._node = _node
        else:
            raise ValueError("Tensor requires an ndarray source or an op node")

    @property
    def dtype(self) -> str:
        return self._meta.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._meta.shape

    @property
    def meta(self) -> TensorMeta:
        return self._meta

    def __sub__(self, other: Any) -> "Tensor":
        value = _require_scalar(other, "sub_scalar")
        return op("sub_scalar", self, value=value)

    def __mul__(self, other: Any) -> "Tensor":
        value = _require_scalar(other, "mul_scalar")
        return op("mul_scalar", self, value=value)

    def compute(self) -> np.ndarray:
        """Materialize this tensor (engine graph only)."""
        return _compute(self)

    def __array__(self, dtype: Any = None) -> np.ndarray:
        """NumPy array protocol: getting the array materializes the graph."""
        arr = self.compute()
        if dtype is None:
            return arr
        return np.asarray(arr, dtype=dtype)


def _coerce_attr(spec: Dict[str, Any], value: Any) -> Any:
    """Coerce a Python attr value to the catalog wire form."""
    key = spec["key"]
    typ = spec["type"]
    count = spec.get("count", 1)
    if typ == "f32":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"attr {key!r} must be a float")
        return float(value)
    if typ == "i32":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"attr {key!r} must be an int")
        return int(value)
    if typ == "string":
        if not isinstance(value, str):
            raise TypeError(f"attr {key!r} must be a str")
        return value
    if typ == "f32_array":
        size = count if count else None
        arr = _as_f32_array(value, name=key, size=size)
        if count == 0 and arr.size < 1:
            raise ValueError(f"attr {key!r} must be a non-empty array")
        return arr
    if typ == "i32_array":
        arr = np.ascontiguousarray(value, dtype=np.int32).reshape(-1)
        if count and arr.size != count:
            raise ValueError(f"attr {key!r} must have {count} elements")
        if count == 0 and arr.size < 1:
            raise ValueError(f"attr {key!r} must be a non-empty array")
        return arr
    raise ValueError(f"attr {key!r}: unsupported catalog type {typ!r}")


def _validate_attrs(
    name: str, schema: Dict[str, Any], attrs: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate/coerce attrs against the catalog; reject unknown keys."""
    specs = schema.get("attrs") or []
    by_key = {s["key"]: s for s in specs}
    unknown = set(attrs) - set(by_key)
    if unknown:
        raise ValueError(f"op {name!r}: unknown attrs {sorted(unknown)}")
    missing = set(by_key) - set(attrs)
    if missing:
        raise ValueError(f"op {name!r}: missing attrs {sorted(missing)}")
    return {k: _coerce_attr(by_key[k], attrs[k]) for k in by_key}


def _out_meta_from_catalog(inp: Tensor, schema: Dict[str, Any]) -> TensorMeta:
    outs = schema.get("outputs") or [{"channels": "same"}]
    if len(outs) != 1:
        raise ValueError("Phase A′ only supports single-output ops")
    ch = outs[0].get("channels", "same")
    if ch == "same":
        channels = inp.meta.channels
    else:
        channels = int(ch)
    return TensorMeta(
        dtype=inp.meta.dtype,
        height=inp.meta.height,
        width=inp.meta.width,
        channels=channels,
    )


def _check_inputs(name: str, schema: Dict[str, Any], inputs: Sequence[Tensor]) -> None:
    specs = schema.get("inputs") or [{"channels": "any"}]
    if len(inputs) != len(specs):
        raise ValueError(
            f"op {name!r}: expected {len(specs)} input(s), got {len(inputs)}"
        )
    for i, (tens, spec) in enumerate(zip(inputs, specs)):
        ch = spec.get("channels", "any")
        if ch != "any" and tens.meta.channels != int(ch):
            raise ValueError(
                f"op {name!r} input[{i}]: expected {ch} channel(s), "
                f"got {tens.meta.channels}"
            )


def op(name: str, x: Tensor, /, **attrs: Any) -> Tensor:
    """Emit an engine op from the shared catalog (YAML SSOT)."""
    from .engine_ops import OPS_CATALOG

    schema = OPS_CATALOG["ops"].get(name)
    if schema is None:
        raise ValueError(f"unknown engine op {name!r} (not in catalog)")
    _check_inputs(name, schema, (x,))
    coerced = _validate_attrs(name, schema, attrs)
    out_meta = _out_meta_from_catalog(x, schema)
    node = _OpNode(
        op=name,
        inputs=(x,),
        attrs=coerced,
        out_meta=out_meta,
    )
    return Tensor(_meta=out_meta, _node=node)


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


def _alloc_like(meta: TensorMeta) -> np.ndarray:
    return np.zeros(meta.shape, dtype=_NUMPY_FROM_DTYPE[meta.dtype])


def _run_engine_segment(
    nodes: List[Tensor],
    values: Dict[int, np.ndarray],
    outputs: List[Tensor],
) -> None:
    from muimg import _compute_engine

    produced = {id(t) for t in nodes}
    input_tensors: List[Tensor] = []
    seen_in: set[int] = set()
    for t in nodes:
        assert t._node is not None
        for inp in t._node.inputs:
            iid = id(inp)
            if iid not in produced and iid not in seen_in:
                input_tensors.append(inp)
                seen_in.add(iid)

    all_tensors = input_tensors + nodes
    id_of = {id(t): i for i, t in enumerate(all_tensors)}

    tensor_descs = []
    for t in all_tensors:
        m = t.meta
        tensor_descs.append(
            {
                "id": id_of[id(t)],
                "dtype": m.dtype,
                "height": m.height,
                "width": m.width,
                "channels": m.channels,
            }
        )

    graph_nodes = []
    for t in nodes:
        assert t._node is not None
        graph_nodes.append(
            {
                "id": len(graph_nodes),
                "op": t._node.op,
                "inputs": [id_of[id(inp)] for inp in t._node.inputs],
                "outputs": [id_of[id(t)]],
                "attrs": dict(t._node.attrs),
            }
        )

    in_binds: Dict[int, np.ndarray] = {}
    for t in input_tensors:
        tid = id(t)
        if tid not in values:
            raise ValueError(f"missing materialized input for tensor {tid}")
        in_binds[id_of[tid]] = np.ascontiguousarray(values[tid])

    out_binds: Dict[int, np.ndarray] = {}
    for t in outputs:
        arr = _alloc_like(t.meta)
        values[id(t)] = arr
        out_binds[id_of[id(t)]] = arr

    graph = {
        "tensor_descs": tensor_descs,
        "inputs": [id_of[id(t)] for t in input_tensors],
        "outputs": [id_of[id(t)] for t in outputs],
        "nodes": graph_nodes,
    }
    _compute_engine.execute_graph(graph, in_binds, out_binds)


def _compute(root: Tensor) -> np.ndarray:
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
    _run_engine_segment(op_tensors, values, outs)

    if id(root) not in values:
        raise RuntimeError("compute finished without materializing root")
    return values[id(root)]
