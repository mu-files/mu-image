# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files
"""muimg compute graph: OpNode / Tensor DAG, engine protocol, and orchestration.

Pipeline code (e.g. ``raw_render``) builds this portable DAG. Engines execute
engine-affinity segments of it; ``@graph_op`` kernels run in Python.
"""

from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

from ..common import PerfTimer
from ..tensor import ElementType, Tensor, TensorMeta, meta_from_array

OutMetaFn = Callable[[Tensor, Dict[str, Any]], Any]
GraphOutMetaFn = Callable[[Tensor, Dict[str, Any]], TensorMeta]

# ---------------------------------------------------------------------------
# Engine timing policy
# ---------------------------------------------------------------------------


class EngineTiming(IntEnum):
    """How much detail ``graph.compute`` / execute_segment should record."""

    OFF = 0
    SEGMENTS = 1  # one row per python op or engine execute_segment
    OPS = 2  # + per-op rows inside an engine segment


engine_timing: EngineTiming = EngineTiming.OFF


def get_engine_timing() -> EngineTiming:
    return engine_timing


def set_engine_timing(level: EngineTiming | int | str) -> None:
    """Set engine compute timing detail."""
    global engine_timing
    if isinstance(level, EngineTiming):
        engine_timing = level
    elif isinstance(level, int):
        engine_timing = EngineTiming(level)
    else:
        engine_timing = EngineTiming[str(level).strip().upper()]


# ---------------------------------------------------------------------------
# Engine protocol + default registry
# ---------------------------------------------------------------------------

_default_engine: Optional["Engine"] = None


@runtime_checkable
class Engine(Protocol):
    """Backend that executes contiguous engine-affinity graph segments."""

    @property
    def supported_ops(self) -> frozenset[str]:
        """Op names this engine can execute."""
        ...

    def execute_segment(
        self,
        nodes: List[Tensor],
        values: dict[int, np.ndarray],
        outputs: List[Tensor],
    ) -> None:
        """Run ``nodes``; write ``outputs`` into ``values`` (and any needed intermediates).

        At engine timing OPS, record per-op ``{op} (engine)`` children under the
        current ``graph_compute`` step via ``PerfTimer.current()``.
        """
        ...


def get_default_engine() -> Engine:
    global _default_engine
    if _default_engine is None:
        from .core.engine import CoreEngine

        _default_engine = CoreEngine()
    return _default_engine


def set_default_engine(engine: Engine) -> None:
    global _default_engine
    _default_engine = engine


# ---------------------------------------------------------------------------
# Op catalog types + graph_op
# ---------------------------------------------------------------------------


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
    # When set (e.g. geometry: crop), replaces dtype/channels/H×W/origin composition.
    _infer_meta: Optional[GraphOutMetaFn] = None

    def __call__(self, x: Tensor, /, **attrs: Any) -> Tensor:
        return emit(self, x, **attrs)

    def infer_out_meta(self, x: Tensor, attrs: Dict[str, Any]) -> TensorMeta:
        if self._infer_meta is not None:
            return self._infer_meta(x, attrs)
        return TensorMeta(
            dtype=self._out_dtype(x, attrs),
            height=x.meta.height,
            width=x.meta.width,
            channels=self._out_channels(x, attrs),
            origin=x.meta.origin,
        )

    def __repr__(self) -> str:
        return f"EngineOp({self.meta.name!r})"


def _out_dtype_same(x: Tensor, attrs: Dict[str, Any]) -> ElementType:
    return x.meta.dtype


def _out_dtype_const(dtype: str | ElementType) -> OutMetaFn:
    resolved = ElementType.coerce(dtype)

    def _fn(x: Tensor, attrs: Dict[str, Any]) -> ElementType:
        return resolved

    return _fn


def _out_dtype_from_attr(key: str) -> OutMetaFn:
    def _fn(x: Tensor, attrs: Dict[str, Any]) -> ElementType:
        val = attrs.get(key)
        try:
            return ElementType.coerce(val)
        except (TypeError, KeyError) as e:
            raise ValueError(
                f"attr {key!r} must be a dtype "
                f"(one of {[d.value for d in ElementType]}), got {val!r}"
            ) from e

    return _fn


def _out_channels_same(x: Tensor, attrs: Dict[str, Any]) -> int:
    return x.meta.channels


def _out_channels_const(n: int) -> OutMetaFn:
    def _fn(x: Tensor, attrs: Dict[str, Any]) -> int:
        return n

    return _fn


def _out_meta_crop(x: Tensor, attrs: Dict[str, Any]) -> TensorMeta:
    """Geometry policy ``crop``: H/W from attrs; update or reset world origin.

    Crop attrs ``x``/``y`` are column/row offsets into the current buffer.
    Default: ``origin' = origin + (y, x)`` with origin stored as ``(row, col)``.
    If ``reset_origin`` is true (ActiveArea / DefaultCrop), ``origin' = (0, 0)``.
    """
    col, row = int(attrs["x"]), int(attrs["y"])
    w, h = int(attrs["w"]), int(attrs["h"])
    if col < 0 or row < 0 or w < 1 or h < 1:
        raise ValueError(f"crop: invalid box x={col} y={row} w={w} h={h}")
    if row + h > x.meta.height or col + w > x.meta.width:
        raise ValueError(
            f"crop: box x={col} y={row} w={w} h={h} out of bounds for "
            f"{x.meta.height}x{x.meta.width}"
        )
    if attrs.get("reset_origin"):
        origin = (0, 0)
    else:
        base_row, base_col = x.meta.origin
        origin = (base_row + row, base_col + col)
    return TensorMeta(
        dtype=x.meta.dtype,
        height=h,
        width=w,
        channels=x.meta.channels,
        origin=origin,
    )


# TIFF 5–8 include a 90° (H×W swap). 1–4 keep size.
_ORIENTATION_SWAP_HW = frozenset({5, 6, 7, 8})


def _out_meta_orientation(x: Tensor, attrs: Dict[str, Any]) -> TensorMeta:
    """Geometry policy ``orientation``: swap H×W for TIFF codes 5–8."""
    code = int(attrs["orientation"])
    if code < 1 or code > 8:
        raise ValueError(f"orientation: invalid TIFF code {code} (expected 1–8)")
    swap = code in _ORIENTATION_SWAP_HW
    return TensorMeta(
        dtype=x.meta.dtype,
        height=x.meta.width if swap else x.meta.height,
        width=x.meta.height if swap else x.meta.width,
        channels=x.meta.channels,
        origin=x.meta.origin,
    )


@dataclass
class OpNode:
    """Catalog engine op (``fn is None``) or Python ``@graph_op`` kernel."""

    op: str
    inputs: Tuple[Tensor, ...]
    attrs: Dict[str, Any]
    out_meta: TensorMeta
    fn: Optional[Callable[..., np.ndarray]] = None


def graph_op(
    fn: Optional[Callable[..., np.ndarray]] = None,
    /,
    *,
    out_meta: Optional[GraphOutMetaFn] = None,
):
    """Decorator: eager ndarray body; Tensor first-arg attaches a lazy graph node.

    Usage::

        @graph_op
        def same_shape(arr, *, scale):
            return arr * scale

        @graph_op(out_meta=my_infer)
        def scale_op(arr, *, factor):
            return arr * factor
    """

    def decorate(f: Callable[..., np.ndarray]) -> Callable[..., Any]:
        sig = inspect.signature(f)
        param_names = list(sig.parameters)
        if not param_names:
            raise ValueError(f"graph_op {f.__name__!r}: need at least one parameter")
        first_name = param_names[0]

        @functools.wraps(f)
        def wrapper(image: Any, /, *args: Any, **kwargs: Any) -> Any:
            if not isinstance(image, Tensor):
                return f(image, *args, **kwargs)

            placeholder = object()
            bound = sig.bind(placeholder, *args, **kwargs)
            bound.apply_defaults()
            attrs = {
                k: v for k, v in bound.arguments.items() if k != first_name
            }

            if out_meta is None:
                resolved = image.meta
            else:
                resolved = out_meta(image, attrs)

            node = OpNode(
                op=f.__name__,
                inputs=(image,),
                attrs=attrs,
                out_meta=resolved,
                fn=f,
            )
            return Tensor(_meta=resolved, _node=node)

        wrapper.__graph_op__ = True  # type: ignore[attr-defined]
        return wrapper

    if fn is not None:
        return decorate(fn)
    return decorate


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
        allowed = spec.get("values")
        if allowed is not None and value not in allowed:
            raise ValueError(
                f"attr {key!r} must be one of {list(allowed)}, got {value!r}"
            )
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
    """Materialize a lazy graph into a concrete source Tensor.

    Prefer ``@graph_op`` helpers for reusable Python steps; ``flush`` remains
    for ad-hoc barriers.
    """
    return Tensor(x.compute())


def _is_python_node(t: Tensor) -> bool:
    return t._node is not None and t._node.fn is not None


def _run_python_node(t: Tensor, values: Dict[int, np.ndarray]) -> None:
    node = t._node
    assert node is not None and node.fn is not None
    if len(node.inputs) != 1:
        raise ValueError(f"python op {node.op!r}: expected 1 input")
    inp = values.get(id(node.inputs[0]))
    if inp is None:
        raise RuntimeError(f"python op {node.op!r}: missing input value")
    out = node.fn(inp, **node.attrs)
    if not isinstance(out, np.ndarray):
        raise TypeError(f"python op {node.op!r}: kernel must return ndarray")
    out = np.ascontiguousarray(out)
    got = meta_from_array(out)
    want = t.meta
    if (
        got.height != want.height
        or got.width != want.width
        or got.channels != want.channels
        or got.dtype != want.dtype
    ):
        raise ValueError(
            f"python op {node.op!r}: output meta {got} != inferred {want}"
        )
    values[id(t)] = out


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
    """Topo-sort reachable nodes; run engine segments and ``@graph_op`` kernels."""
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

    parent = PerfTimer.current()
    level = get_engine_timing()
    record = parent is not None and level >= EngineTiming.SEGMENTS

    engine = get_default_engine()
    i = 0
    while i < len(op_tensors):
        if _is_python_node(op_tensors[i]):
            node = op_tensors[i]._node
            assert node is not None
            if record:
                assert parent is not None
                step = parent.start_step(f"{node.op} (python)")
            else:
                step = PerfTimer.inactive
            _run_python_node(op_tensors[i], values)
            step.close()
            i += 1
            continue

        j = i + 1
        while j < len(op_tensors) and not _is_python_node(op_tensors[j]):
            j += 1
        segment = op_tensors[i:j]
        outs = _segment_boundary_outputs(segment, op_tensors, root)
        if not outs:
            outs = [segment[-1]]

        if record:
            assert parent is not None
            seg_step = parent.start_step("graph_compute")
        else:
            seg_step = PerfTimer.inactive
        engine.execute_segment(segment, values, outs)
        seg_step.close()
        i = j

    if id(root) not in values:
        raise RuntimeError("compute finished without materializing root")
    return values[id(root)]
