# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files
"""
muimg.mc — lazy compute-graph API (Phase A.5–A.6).

Graph/engine glue only: build a lazy DAG, partition by affinity, serialize
engine segments to the A.4 dict IR and call muimg._compute_engine; python
nodes invoke callables defined elsewhere (e.g. raw_render.demosaic).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

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
    affinity: str
    out_meta: TensorMeta
    # Optional python callable: (input_arrays..., attrs) -> ndarray
    py_fn: Optional[Callable[..., np.ndarray]] = None


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
    """Lazy tensor handle: either a concrete source buffer or an op result."""

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
        return _unary_op(
            self,
            op="sub_scalar",
            attrs={"value": value},
            out_meta=self._meta,
        )

    def __mul__(self, other: Any) -> "Tensor":
        value = _require_scalar(other, "mul_scalar")
        return _unary_op(
            self,
            op="mul_scalar",
            attrs={"value": value},
            out_meta=self._meta,
        )

    def compute(self) -> np.ndarray:
        """Materialize this tensor via affinity-partitioned segments."""
        return _compute(self)


def _python_demosaic_fn(
    *arrays: np.ndarray, attrs: Dict[str, Any]
) -> np.ndarray:
    from .raw_render import DemosaicAlgorithm, demosaic as rr_demosaic

    if len(arrays) != 1:
        raise ValueError("python demosaic expects one input")
    alg = DemosaicAlgorithm.lookup(attrs["alg"])
    return_dtype = _NUMPY_FROM_DTYPE[attrs["return_dtype"]]
    return rr_demosaic(
        arrays[0],
        attrs["cfa_pattern"],
        algorithm=alg,
        return_dtype=return_dtype,
    )


@dataclass(frozen=True)
class _OpSpec:
    """Phase A hard-wired op metadata. Phase A′: from shared YAML catalog."""

    affinity: str  # "engine" | "python"
    py_fn: Optional[Callable[..., np.ndarray]] = None


# Affinity (and python callables) live on the op, not on demosaic alg strings.
_OPS: Dict[str, _OpSpec] = {
    "sub_scalar": _OpSpec("engine"),
    "mul_scalar": _OpSpec("engine"),
    "bilinear_demosaic": _OpSpec("engine"),
    "matrix": _OpSpec("engine"),
    "lut": _OpSpec("engine"),
    "python_demosaic": _OpSpec("python", py_fn=_python_demosaic_fn),
}

# Sugar only: which concrete op mc.demosaic(alg=...) emits.
_DEMOSAIC_ALG_TO_OP = {
    "bilinear": "bilinear_demosaic",
    "DNGSDK_BILINEAR": "bilinear_demosaic",
}


def _unary_op(
    inp: Tensor,
    *,
    op: str,
    attrs: Dict[str, Any],
    out_meta: TensorMeta,
) -> Tensor:
    spec = _OPS.get(op)
    if spec is None:
        raise ValueError(f"unknown op {op!r} (not in Phase A hard-wired catalog)")
    if spec.affinity == "python" and spec.py_fn is None:
        raise ValueError(f"python op {op!r} has no callable")
    node = _OpNode(
        op=op,
        inputs=(inp,),
        attrs=attrs,
        affinity=spec.affinity,
        out_meta=out_meta,
        py_fn=spec.py_fn,
    )
    return Tensor(_meta=out_meta, _node=node)


# Phase A: hard-wired helpers matching engine / python op names+attrs.
# Phase A′: shared YAML catalog generates (or validates) these stubs.
def demosaic(x: Tensor, alg: str, cfa_pattern: str) -> Tensor:
    """Demosaic sugar: map alg → op; affinity comes from that op's spec."""
    from .tiff_metadata import CFA_PATTERN_TO_CODES

    if x.meta.channels != 1:
        raise ValueError("demosaic input must be mono / CFA (1 channel)")
    pattern = cfa_pattern.upper()
    codes = CFA_PATTERN_TO_CODES.get(pattern)
    if codes is None:
        raise ValueError(
            f"Invalid CFA pattern: {cfa_pattern!r}. "
            f"Supported patterns are: {list(CFA_PATTERN_TO_CODES)}"
        )
    out_meta = TensorMeta(
        dtype=x.meta.dtype,
        height=x.meta.height,
        width=x.meta.width,
        channels=3,
    )
    op = _DEMOSAIC_ALG_TO_OP.get(alg, "python_demosaic")
    if op == "bilinear_demosaic":
        attrs: Dict[str, Any] = {
            "cfa_pattern": np.ascontiguousarray(codes, dtype=np.int32),
        }
    else:
        attrs = {
            "alg": alg,
            "cfa_pattern": pattern,
            "return_dtype": x.meta.dtype,
        }
    return _unary_op(x, op=op, attrs=attrs, out_meta=out_meta)


def matrix(x: Tensor, m: Union[Sequence[float], np.ndarray]) -> Tensor:
    if x.meta.channels != 3:
        raise ValueError("matrix input must be RGB (3 channels)")
    mat = _as_f32_array(m, name="matrix", size=9)
    return _unary_op(
        x,
        op="matrix",
        attrs={"matrix": mat},
        out_meta=x.meta,
    )


def lut(x: Tensor, samples: Union[Sequence[float], np.ndarray]) -> Tensor:
    table = _as_f32_array(samples, name="lut")
    return _unary_op(
        x,
        op="lut",
        attrs={"lut": table},
        out_meta=x.meta,
    )


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


def _partition_segments(
    op_tensors: List[Tensor],
) -> List[Tuple[str, List[Tensor]]]:
    """Maximal contiguous runs of the same affinity (topo order)."""
    segments: List[Tuple[str, List[Tensor]]] = []
    for t in op_tensors:
        assert t._node is not None
        aff = t._node.affinity
        if aff not in ("engine", "python"):
            raise ValueError(f"unknown affinity: {aff!r}")
        if not segments or segments[-1][0] != aff:
            segments.append((aff, [t]))
        else:
            segments[-1][1].append(t)
    return segments


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
        if t._node.affinity != "engine":
            raise ValueError(
                f"engine segment contains non-engine op {t._node.op!r}"
            )
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


def _run_python_node(t: Tensor, values: Dict[int, np.ndarray]) -> None:
    assert t._node is not None
    if t._node.py_fn is None:
        raise ValueError(f"python op {t._node.op!r} has no callable")
    arrays = []
    for inp in t._node.inputs:
        if id(inp) not in values:
            raise ValueError(f"missing materialized input for op {t._node.op!r}")
        arrays.append(values[id(inp)])
    out = t._node.py_fn(*arrays, attrs=t._node.attrs)
    out = np.ascontiguousarray(out)
    # Keep declared meta; allow dtype/shape checks lightly.
    if out.shape != t.meta.shape:
        raise ValueError(
            f"python op {t._node.op!r} produced shape {out.shape}, "
            f"expected {t.meta.shape}"
        )
    values[id(t)] = out


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

    segments = _partition_segments(op_tensors)
    for aff, nodes in segments:
        if aff == "python":
            for t in nodes:
                _run_python_node(t, values)
        else:
            outs = _segment_boundary_outputs(nodes, op_tensors, root)
            if not outs:
                # Segment results unused — still run so side-effect-free
                # graphs fail clearly; treat last node as output.
                outs = [nodes[-1]]
            _run_engine_segment(nodes, values, outs)

    if id(root) not in values:
        raise RuntimeError("compute finished without materializing root")
    return values[id(root)]
