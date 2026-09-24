# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files
"""muraw compute graph: OpNode / Array DAG, engine protocol, and orchestration.

Pipeline code (e.g. ``raw_render``) builds this portable DAG. Engines execute
engine-affinity segments of it; ``@graph_op`` kernels run in Python.
"""

from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass, field
from enum import IntEnum
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

from ..common import PerfTimer
from ..array import (
    ElementType,
    Array,
    ArrayMeta,
    _seal_ndarray,
    meta_from_array,
)

OutMetaFn = Callable[[Array, Dict[str, Any]], Any]
GraphOutMetaFn = Callable[[Array, Dict[str, Any]], ArrayMeta]

# ---------------------------------------------------------------------------
# Engine timing policy
# ---------------------------------------------------------------------------


class EngineTiming(IntEnum):
    """How much detail ``graph.realize`` / execute_segment should record."""

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
        nodes: List[Array],
        values: dict[int, np.ndarray],
        outputs: List[Array],
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
    """Static catalog facts for an engine op (not dependent on a Array)."""

    name: str
    # Optional scheduler hint for a future executor; not part of the graph IR.
    granularity: str = "full_image"  # "span" | "tile" | "full_image"
    halo: int = 0


@dataclass(frozen=True)
class EngineOp:
    """Callable engine op + metadata. Public names live on ``muimage``."""

    meta: OpMeta
    _out_dtype: OutMetaFn
    _out_channels: OutMetaFn
    _in_channels: Optional[int]  # None = any
    # Catalog input count. 0 means ``x`` is dest meta only, not a graph input.
    _n_inputs: int
    _attr_specs: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    # When set (e.g. geometry: view), replaces dtype/channels/H×W/origin composition.
    _infer_meta: Optional[GraphOutMetaFn] = None

    def __call__(self, x: Array, /, **attrs: Any) -> Array:
        # TIFF orientation 1 is identity; skip the node.
        if self.meta.name == "orientation" and int(attrs.get("orientation", 0)) == 1:
            return x
        return emit(self, x, **attrs)

    def infer_out_meta(self, x: Array, attrs: Dict[str, Any]) -> ArrayMeta:
        if self._infer_meta is not None:
            return self._infer_meta(x, attrs)
        return x.meta.copy(
            dtype=self._out_dtype(x, attrs),
            channels=self._out_channels(x, attrs),
        )

    def __repr__(self) -> str:
        return f"EngineOp({self.meta.name!r})"


def _out_dtype_same(x: Array, attrs: Dict[str, Any]) -> ElementType:
    return x.meta.dtype


def _out_dtype_const(dtype: str | ElementType) -> OutMetaFn:
    resolved = ElementType(dtype)

    def _fn(x: Array, attrs: Dict[str, Any]) -> ElementType:
        return resolved

    return _fn


def _out_dtype_from_attr(key: str) -> OutMetaFn:
    def _fn(x: Array, attrs: Dict[str, Any]) -> ElementType:
        val = attrs.get(key)
        try:
            return ElementType(val)
        except ValueError as e:
            raise ValueError(
                f"attr {key!r} must be a dtype "
                f"(one of {[d.value for d in ElementType]}), got {val!r}"
            ) from e

    return _fn


def _out_channels_same(x: Array, attrs: Dict[str, Any]) -> int:
    return x.meta.channels


def _out_channels_const(n: int) -> OutMetaFn:
    def _fn(x: Array, attrs: Dict[str, Any]) -> int:
        return n

    return _fn


def _sample_box(origin: int, count: int, step: int) -> Tuple[int, int]:
    """Bounding box of ``count`` samples starting at ``origin`` with ``step``."""
    last = origin + (count - 1) * step
    start = min(origin, last)
    return start, abs(last - origin) + 1


def _reflected_canvas(
    canvas: Tuple[int, int, int, int],
    sample_col: int,
    sample_row: int,
    col_step: int,
    row_step: int,
    origin_col: int,
    origin_row: int,
) -> Tuple[int, int, int, int]:
    """Parent canvas in dest coordinates.

    Dest ``(0, 0)`` is the first sample. One dest step on an axis moves by
    that axis's source step, which is ±1.
    """

    def axis(
        parent_start: int,
        parent_len: int,
        sample: int,
        step: int,
        dest_origin: int,
    ) -> Tuple[int, int]:
        parent_last = parent_start + parent_len - 1
        dest_first = (parent_start - sample) * step
        dest_last = (parent_last - sample) * step
        start = min(dest_first, dest_last)
        return dest_origin + start, abs(dest_last - dest_first) + 1

    cx, cy, cw, ch = canvas
    x0, width = axis(cx, cw, sample_col, col_step, origin_col)
    y0, height = axis(cy, ch, sample_row, row_step, origin_row)
    return (x0, y0, width, height)


def _out_meta_view(x: Array, attrs: Dict[str, Any]) -> ArrayMeta:
    """Geometry policy ``view``: H/W from attrs. Canvas stays in this array's system.

    View attrs ``left``/``top`` are dest-relative. The sample bounding box
    is checked in the shared canvas system (dest ``(0, 0)`` is ``origin``).
    ``width`` and ``height`` are sample counts. ``row_step`` and ``col_step``
    default to 1. A step of -1 keeps the parent canvas, reflected so dest
    +1 follows that step. A step whose absolute value is greater than 1
    puts the result on its own canvas at origin ``(0, 0)``.
    Default: ``origin' = origin + (top, left)``, the first sample.
    ``reset_origin`` sets ``(0, 0)``. ``oob_valid`` does not change origin:
    false puts canvas on the view rect; true keeps the parent canvas (or
    remaps it into dest space when origin was reset).
    """
    left, top = int(attrs["left"]), int(attrs["top"])
    width, height = int(attrs["width"]), int(attrs["height"])
    row_step = int(attrs.get("row_step", 1))
    col_step = int(attrs.get("col_step", 1))
    if width < 1 or height < 1:
        raise ValueError(
            f"view: invalid box top={top} left={left} width={width} height={height}"
        )
    if row_step == 0 or col_step == 0:
        raise ValueError(f"view: step must be non-zero, got row={row_step} col={col_step}")
    cx, cy, cw, ch = x.meta.canvas
    base_row, base_col = x.meta.origin
    box_left, box_width = _sample_box(left, width, col_step)
    box_top, box_height = _sample_box(top, height, row_step)
    shared_left = box_left + base_col
    shared_top = box_top + base_row
    if (
        shared_left < cx
        or shared_top < cy
        or shared_left + box_width > cx + cw
        or shared_top + box_height > cy + ch
    ):
        raise ValueError(
            f"view: box top={top} left={left} width={width} height={height} "
            f"row_step={row_step} col_step={col_step} "
            f"is outside canvas {(cx, cy, cw, ch)}"
        )
    decimated = abs(row_step) > 1 or abs(col_step) > 1
    reflected = row_step != 1 or col_step != 1
    if decimated:
        origin = (0, 0)
        canvas = (0, 0, width, height)
    elif attrs.get("reset_origin"):
        origin = (0, 0)
        if not attrs.get("oob_valid", True):
            canvas = (0, 0, width, height)
        elif reflected:
            canvas = _reflected_canvas(
                (cx, cy, cw, ch),
                base_col + left,
                base_row + top,
                col_step,
                row_step,
                0,
                0,
            )
        else:
            canvas = (cx - left, cy - top, cw, ch)
    else:
        origin = (base_row + top, base_col + left)
        origin_row, origin_col = origin
        if not attrs.get("oob_valid", True):
            canvas = (origin_col, origin_row, width, height)
        elif reflected:
            canvas = _reflected_canvas(
                (cx, cy, cw, ch),
                origin_col,
                origin_row,
                col_step,
                row_step,
                origin_col,
                origin_row,
            )
        else:
            canvas = (cx, cy, cw, ch)
    channels = x.meta.channels
    if "src_channels" in attrs:
        src_channels = attrs["src_channels"]
        channels = len(src_channels)
        for value in src_channels:
            index = int(value)
            if index < 0 or index >= x.meta.channels:
                raise IndexError(
                    f"view: src_channels index {value} is out of bounds "
                    f"for {x.meta.channels} channel(s)"
                )
    return x.meta.copy(
        height=height,
        width=width,
        channels=channels,
        origin=origin,
        canvas=canvas,
    )


def _out_meta_pad(x: Array, attrs: Dict[str, Any]) -> ArrayMeta:
    """Geometry policy ``pad``: grow H/W, and channels when those attrs are set.

    Dest ``(left, top)`` is src ``(0, 0)``. Origin shifts by
    ``-(top, left)``. Canvas is the new array at that origin, same as a
    crop's view rect and canvas both sitting on the window.
    """
    top = int(attrs["top"])
    bottom = int(attrs["bottom"])
    left = int(attrs["left"])
    right = int(attrs["right"])
    channel_before = int(attrs.get("channel_before", 0))
    channel_after = int(attrs.get("channel_after", 0))
    if min(top, bottom, left, right, channel_before, channel_after) < 0:
        raise ValueError(
            "pad: top/bottom/left/right/channel_before/channel_after "
            f"{[top, bottom, left, right, channel_before, channel_after]} "
            "must be non-negative"
        )
    mode = attrs.get("mode", "constant")
    if (channel_before or channel_after) and mode != "constant":
        raise ValueError(
            f"pad: a channel pad requires mode 'constant', got {mode!r}"
        )
    dest_h = x.meta.height + top + bottom
    dest_w = x.meta.width + left + right
    base_row, base_col = x.meta.origin
    origin = (base_row - top, base_col - left)
    origin_row, origin_col = origin
    return x.meta.copy(
        height=dest_h,
        width=dest_w,
        channels=x.meta.channels + channel_before + channel_after,
        channel_axis=x.meta.channel_axis or channel_before + channel_after > 0,
        origin=origin,
        canvas=(origin_col, origin_row, dest_w, dest_h),
    )


# TIFF 5–8 include a 90° (H×W swap). 1–4 keep size.
_ORIENTATION_SWAP_HW = frozenset({5, 6, 7, 8})


def _dest_to_src(
    code: int, dx: int, dy: int, src_w: int, src_h: int
) -> Tuple[int, int]:
    if code == 1:
        return (dx, dy)
    if code == 2:
        return (src_w - 1 - dx, dy)
    if code == 3:
        return (src_w - 1 - dx, src_h - 1 - dy)
    if code == 4:
        return (dx, src_h - 1 - dy)
    if code == 5:
        return (dy, dx)
    if code == 6:
        return (dy, src_h - 1 - dx)
    if code == 7:
        return (src_w - 1 - dy, src_h - 1 - dx)
    if code == 8:
        return (src_w - 1 - dy, dx)
    raise ValueError(f"orientation: invalid TIFF code {code} (expected 1–8)")


def _invert_orientation(code: int) -> int:
    if code == 6:
        return 8
    if code == 8:
        return 6
    return code


def _map_rect_through_orientation(
    code: int, rect: Tuple[int, int, int, int], dest_w: int, dest_h: int
) -> Tuple[int, int, int, int]:
    inv = _invert_orientation(code)
    x0, y0, w, h = rect
    x1 = x0 + w - 1
    y1 = y0 + h - 1
    corners = (
        _dest_to_src(inv, x0, y0, dest_w, dest_h),
        _dest_to_src(inv, x1, y0, dest_w, dest_h),
        _dest_to_src(inv, x0, y1, dest_w, dest_h),
        _dest_to_src(inv, x1, y1, dest_w, dest_h),
    )
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    return (min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)


def _out_meta_orientation(x: Array, attrs: Dict[str, Any]) -> ArrayMeta:
    """Geometry policy ``orientation``: swap H×W for TIFF codes 5–8.

    Canvas is remapped in dest space, then stored back in the shared
    system (dest ``(0, 0)`` is ``origin``).
    """
    code = int(attrs["orientation"])
    if code < 1 or code > 8:
        raise ValueError(f"orientation: invalid TIFF code {code} (expected 1–8)")
    swap = code in _ORIENTATION_SWAP_HW
    height = x.meta.width if swap else x.meta.height
    width = x.meta.height if swap else x.meta.width
    origin_row, origin_col = x.meta.origin
    cx, cy, cw, ch = x.meta.canvas
    local = (cx - origin_col, cy - origin_row, cw, ch)
    mx, my, mw, mh = _map_rect_through_orientation(code, local, width, height)
    return x.meta.copy(
        height=height,
        width=width,
        canvas=(mx + origin_col, my + origin_row, mw, mh),
    )


@dataclass(frozen=True)
class OpNode:
    """Catalog engine op (``fn is None``) or Python ``@graph_op`` kernel."""

    op: str
    inputs: Tuple[Array, ...]
    attrs: Mapping[str, Any]
    out_meta: ArrayMeta
    fn: Optional[Callable[..., np.ndarray]] = None


def graph_op(
    fn: Optional[Callable[..., np.ndarray]] = None,
    /,
    *,
    out_meta: Optional[GraphOutMetaFn] = None,
):
    """Decorator: eager ndarray body; Array first-arg attaches a lazy graph node.

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
            if not isinstance(image, Array):
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
                attrs=MappingProxyType(dict(attrs)),
                out_meta=resolved,
                fn=f,
            )
            return Array(_meta=resolved, _node=node)

        wrapper.__graph_op__ = True  # type: ignore[attr-defined]
        return wrapper

    if fn is not None:
        return decorate(fn)
    return decorate


_SCALAR_FLOAT = {"f32": np.float32, "f64": np.float64}
_ARRAY_DTYPE = {
    "f32_array": np.float32,
    "f64_array": np.float64,
    "i32_array": np.int32,
}


def _as_typed_array(
    value: Any, dtype: Any, *, name: str, size: Optional[int] = None
) -> np.ndarray:
    arr = np.ascontiguousarray(value, dtype=dtype).reshape(-1)
    if size is not None and arr.size != size:
        raise ValueError(f"{name} must have {size} elements, got {arr.size}")
    if arr.size < 1:
        raise ValueError(f"{name} must be non-empty")
    return arr


def _coerce_attr(spec: Dict[str, Any], value: Any) -> Any:
    """Coerce a Python attr value to the catalog wire form."""
    key = spec["key"]
    typ = spec["type"]
    count = spec.get("count", 1)
    if typ in _SCALAR_FLOAT:
        if isinstance(value, bool) or not isinstance(value, (int, float, np.floating)):
            raise TypeError(f"attr {key!r} must be a float")
        return _SCALAR_FLOAT[typ](value)
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
    if typ in _ARRAY_DTYPE:
        return _as_typed_array(
            value, _ARRAY_DTYPE[typ], name=key, size=count or None
        )
    raise ValueError(f"attr {key!r}: unsupported catalog type {typ!r}")


def _validate_attrs(
    name: str, specs: Tuple[Dict[str, Any], ...], attrs: Dict[str, Any]
) -> Dict[str, Any]:
    """Validate/coerce attrs against the op's attr specs; reject unknown keys."""
    by_key = {s["key"]: s for s in specs}
    unknown = set(attrs) - set(by_key)
    if unknown:
        raise ValueError(f"op {name!r}: unknown attrs {sorted(unknown)}")
    required = {k for k, s in by_key.items() if not s.get("optional")}
    missing = required - set(attrs)
    if missing:
        raise ValueError(f"op {name!r}: missing attrs {sorted(missing)}")
    out: Dict[str, Any] = {}
    for k, v in attrs.items():
        if v is None and by_key[k].get("optional"):
            continue
        out[k] = _coerce_attr(by_key[k], v)
    return out


def emit(engine_op: EngineOp, x: Array, /, **attrs: Any) -> Array:
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
        inputs=() if engine_op._n_inputs == 0 else (x,),
        attrs=MappingProxyType(dict(coerced)),
        out_meta=out_meta,
    )
    return Array(_meta=out_meta, _node=node)


def op(name: str, x: Array, /, **attrs: Any) -> Array:
    """Emit a named engine op (thin alias over ``engines.ops.OPS_BY_NAME``)."""
    from .ops import OPS_BY_NAME

    engine_op = OPS_BY_NAME.get(name)
    if engine_op is None:
        raise ValueError(f"unknown engine op {name!r}")
    return emit(engine_op, x, **attrs)


def flush(x: Array) -> Array:
    """Materialize a lazy graph into a concrete source Array.

    Prefer ``@graph_op`` helpers for reusable Python steps; ``flush`` remains
    for ad-hoc barriers.
    """
    return Array(x.realize())


def _is_python_node(t: Array) -> bool:
    return t._node is not None and t._node.fn is not None


def _run_python_node(t: Array, values: Dict[int, np.ndarray]) -> None:
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
    got = meta_from_array(out)
    want = t.meta
    if (got.height, got.width, got.channels, got.dtype) != (
        want.height,
        want.width,
        want.channels,
        want.dtype,
    ):
        raise ValueError(
            f"python op {node.op!r}: output meta {got} != inferred {want}"
        )
    values[id(t)] = out


def _segment_boundary_outputs(
    nodes: List[Array],
    all_op_arrays: List[Array],
    root: Array,
) -> List[Array]:
    """Arrays produced in this segment that escape to later consumers or root."""
    node_set = {id(t) for t in nodes}
    outside_inputs = {
        id(inp)
        for other in all_op_arrays
        if id(other) not in node_set
        for inp in other._node.inputs
    }
    return [t for t in nodes if t is root or id(t) in outside_inputs]


def _run_op_arrays(
    op_arrays: List[Array],
    values: Dict[int, np.ndarray],
    root: Array,
) -> None:
    parent = PerfTimer.current()
    level = get_engine_timing()
    record = parent is not None and level >= EngineTiming.SEGMENTS
    timer = parent if record else None
    engine = get_default_engine()
    i = 0
    while i < len(op_arrays):
        if _is_python_node(op_arrays[i]):
            node = op_arrays[i]._node
            assert node is not None
            step = (
                timer.start_step(f"{node.op} (python)")
                if record
                else PerfTimer.inactive
            )
            _run_python_node(op_arrays[i], values)
            step.close()
            i += 1
            continue

        j = i + 1
        while j < len(op_arrays) and not _is_python_node(op_arrays[j]):
            j += 1
        segment = op_arrays[i:j]
        outs = _segment_boundary_outputs(segment, op_arrays, root)
        seg_step = (
            timer.start_step("graph_compute") if record else PerfTimer.inactive
        )
        engine.execute_segment(segment, values, outs)
        seg_step.close()
        i = j


def realize(root: Array, *, force_recompute: bool = False) -> np.ndarray:
    """Run the graph if needed and return ``root``'s pixels.

    Cached ``_data`` is handed to the engine as an extra bind. The engine
    skips a producer when that buffer covers the array's canvas.
    ``force_recompute`` omits those binds and reruns every op.
    """
    if root._data is not None and not force_recompute:
        return root._data
    if root._node is None:
        if root._data is None:
            raise ValueError("source Array has no data")
        return root._data

    values: Dict[int, np.ndarray] = {}
    op_arrays: List[Array] = []
    done: set[int] = set()
    visiting: set[int] = set()
    stack: List[Array] = [root]
    while stack:
        current = stack[-1]
        tid = id(current)
        if tid in done:
            stack.pop()
            continue
        if current._node is None:
            if current._data is None:
                raise ValueError("source Array has no data")
            values[tid] = current._data
            done.add(tid)
            stack.pop()
            continue
        if tid not in visiting:
            visiting.add(tid)
            for inp in reversed(current._node.inputs):
                inp_id = id(inp)
                if inp_id in visiting and inp_id not in done:
                    raise ValueError("cycle detected in compute graph")
                if inp_id not in done:
                    stack.append(inp)
            continue
        op_arrays.append(current)
        if current._data is not None and not force_recompute:
            values[tid] = current._data
        visiting.remove(tid)
        done.add(tid)
        stack.pop()
    if not op_arrays:
        return values[id(root)]

    _run_op_arrays(op_arrays, values, root)

    for t in op_arrays:
        arr = values.get(id(t))
        if arr is None:
            continue
        if t._data is None or force_recompute:
            t._data = _seal_ndarray(arr)
            values[id(t)] = t._data

    if id(root) not in values:
        raise RuntimeError("realize finished without materializing root")
    return values[id(root)]
