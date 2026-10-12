# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files
"""muraw compute graph: OpNode / Array DAG, engine protocol, and orchestration.

Pipeline code (e.g. ``raw_render``) builds this portable DAG, and the
default engine runs it in one call per ``realize()``. Python code that is
not an engine op runs on a realized array.
"""

from __future__ import annotations

import operator
from dataclasses import dataclass, field
from enum import IntEnum
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

from ..common import PerfTimer
from numpy.typing import DTypeLike

from ..array import (
    Array,
    ArrayMeta,
    _meta_from_shape,
    _seal_ndarray,
    reshaped_meta,
    supported_dtype,
)

OutMetaFn = Callable[[Array, Dict[str, Any]], Any]
GraphOutMetaFn = Callable[[Array, Dict[str, Any]], ArrayMeta]

# ---------------------------------------------------------------------------
# Engine timing policy
# ---------------------------------------------------------------------------


class EngineTiming(IntEnum):
    """How much detail ``graph.realize`` / execute should record."""

    OFF = 0
    SEGMENTS = 1  # one row per engine execute
    OPS = 2  # + per-op rows inside an engine segment


engine_timing: EngineTiming = EngineTiming.OFF


def get_engine_timing() -> EngineTiming:
    return engine_timing


def set_engine_timing(level: EngineTiming | int | str) -> None:
    """Set engine compute timing detail."""
    global engine_timing
    if isinstance(level, str):
        engine_timing = EngineTiming[level.strip().upper()]
    else:
        engine_timing = EngineTiming(level)


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

    def execute(
        self,
        nodes: List[Array],
        values: dict[int, np.ndarray],
        root: Array,
    ) -> None:
        """Run ``nodes``; write ``root`` into ``values`` (and any needed intermediates).

        Every array in ``values`` has its ``meta.buffer_shape``, so a 1D
        array is a one-row ``(1, N)`` buffer.

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
# Op catalog types
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OpMeta:
    """Static catalog facts for an engine op (not dependent on a Array).

    ``requires_2d``, ``is_cfa`` and ``is_rgb`` are defined in the header of
    ``engines/catalog/ops.yaml``.
    """

    name: str
    requires_2d: bool = False
    is_cfa: bool = False
    is_rgb: bool = False


@dataclass(frozen=True)
class EngineOp:
    """Callable engine op + metadata. Public names live on ``muimage``."""

    meta: OpMeta
    _out_dtype: OutMetaFn
    _out_channels: OutMetaFn
    # One channel count per catalog input entry; None = any.
    _in_channels: Tuple[Optional[int], ...]
    # Catalog input count. 0 means ``x`` is dest meta only, not a graph input.
    _n_inputs: int
    # True when the last input entry may appear one or more times on a node.
    _variable_input: bool
    _attr_specs: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    # When set (e.g. geometry: view), replaces dtype/channels/H×W/origin composition.
    _infer_meta: Optional[GraphOutMetaFn] = None

    def __call__(self, x: Array, /, *more: Array, **attrs: Any) -> Array:
        # TIFF orientation 1 is identity; skip the node.
        if self.meta.name == "orientation" and int(attrs.get("orientation", 0)) == 1:
            return x
        return emit(self, x, *more, **attrs)

    def infer_out_meta(self, x: Array, attrs: Dict[str, Any], *more: Array) -> ArrayMeta:
        """``more`` are the inputs after ``x``. Only a geometry policy of an
        op with more than one input is passed them."""
        if self._infer_meta is not None:
            return self._infer_meta(x, attrs, *more)
        return x.meta.with_size(
            channels=self._out_channels(x, attrs),
            dtype=self._out_dtype(x, attrs),
        )

    def __repr__(self) -> str:
        return f"EngineOp({self.meta.name!r})"


def _out_dtype_same(x: Array, attrs: Dict[str, Any]) -> np.dtype:
    return x.meta.dtype


def _out_dtype_const(dtype: DTypeLike) -> OutMetaFn:
    resolved = supported_dtype(dtype)

    def _fn(x: Array, attrs: Dict[str, Any]) -> np.dtype:
        return resolved

    return _fn


def _out_dtype_from_attr(key: str) -> OutMetaFn:
    def _fn(x: Array, attrs: Dict[str, Any]) -> np.dtype:
        val = attrs.get(key)
        try:
            return supported_dtype(val)
        except ValueError as e:
            raise ValueError(f"attr {key!r}: {e}") from e

    return _fn


def _out_channels_same(x: Array, attrs: Dict[str, Any]) -> int:
    return x.meta.channels


def _out_channels_const(n: int) -> OutMetaFn:
    def _fn(x: Array, attrs: Dict[str, Any]) -> int:
        return n

    return _fn


def _out_channels_from_attr(key: str) -> OutMetaFn:
    def _fn(x: Array, attrs: Dict[str, Any]) -> int:
        return int(attrs[key])

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
        start, length = _sample_box((parent_start - sample) * step, parent_len, step)
        return dest_origin + start, length

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
    first_row, first_col = base_row + top, base_col + left
    if decimated:
        origin = (0, 0)
        canvas = (0, 0, width, height)
    else:
        origin = (0, 0) if attrs.get("reset_origin") else (first_row, first_col)
        origin_row, origin_col = origin
        if not attrs.get("oob_valid", True):
            canvas = (origin_col, origin_row, width, height)
        elif reflected:
            canvas = _reflected_canvas(
                (cx, cy, cw, ch), first_col, first_row, col_step, row_step, origin_col, origin_row
            )
        else:
            canvas = (cx + origin_col - first_col, cy + origin_row - first_row, cw, ch)
    channels = x.meta.channels
    if "src_channels" in attrs:
        src_channels = attrs["src_channels"]
        channels = len(src_channels)
        if any(not 0 <= int(value) < x.meta.channels for value in src_channels):
            raise IndexError(
                f"view: src_channels {list(src_channels)} has an index out of bounds "
                f"for {x.meta.channels} channel(s)"
            )
    return x.meta.with_size(
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
    if (channel_before or channel_after) and mode != "constant" and x.meta.channel_axis != 0:
        raise ValueError(
            f"pad: a channel pad requires mode 'constant' unless the channels are "
            f"on axis 0, got {mode!r}"
        )
    dest_h = x.meta.height + top + bottom
    dest_w = x.meta.width + left + right
    base_row, base_col = x.meta.origin
    origin = (base_row - top, base_col - left)
    origin_row, origin_col = origin
    if channel_before or channel_after:
        ndim = 3
    elif x.meta.is_1d and (top or bottom):
        ndim = 2
    else:
        ndim = x.meta.ndim
    return x.meta.with_size(
        height=dest_h,
        width=dest_w,
        channels=x.meta.channels + channel_before + channel_after,
        ndim=ndim,
        origin=origin,
        canvas=(origin_col, origin_row, dest_w, dest_h),
    )


def _out_meta_tile(x: Array, attrs: Dict[str, Any]) -> ArrayMeta:
    """Geometry policy ``tile``: each axis grows by its repeat count.

    The first copy keeps the source origin. Canvas grows to the right
    and down from that origin.
    """
    row_reps = int(attrs["row_reps"])
    col_reps = int(attrs["col_reps"])
    channel_reps = int(attrs["channel_reps"])
    if min(row_reps, col_reps, channel_reps) < 1:
        raise ValueError(
            "tile: row_reps/col_reps/channel_reps "
            f"{[row_reps, col_reps, channel_reps]} must be >= 1"
        )
    dest_h = x.meta.height * row_reps
    dest_w = x.meta.width * col_reps
    origin_row, origin_col = x.meta.origin
    return x.meta.with_size(
        height=dest_h,
        width=dest_w,
        channels=x.meta.channels * channel_reps,
        ndim=2 if x.meta.is_1d and row_reps > 1 else x.meta.ndim,
        canvas=(origin_col, origin_row, dest_w, dest_h),
    )


# TIFF orientation code → (swap H and W, flip x, flip y) of the map from
# dest to source.
_ORIENTATION = {
    1: (False, False, False),
    2: (False, True, False),
    3: (False, True, True),
    4: (False, False, True),
    5: (True, False, False),
    6: (True, False, True),
    7: (True, True, True),
    8: (True, True, False),
}


def _orient_rect(
    code: int, rect: Tuple[int, int, int, int], src_w: int, src_h: int
) -> Tuple[int, int, int, int]:
    """Map ``rect`` in source coordinates to dest coordinates."""
    swap, flip_x, flip_y = _ORIENTATION[code]
    x0, y0, w, h = rect
    if flip_x:
        x0 = src_w - x0 - w
    if flip_y:
        y0 = src_h - y0 - h
    return (y0, x0, h, w) if swap else (x0, y0, w, h)


def _out_meta_orientation(x: Array, attrs: Dict[str, Any]) -> ArrayMeta:
    """Geometry policy ``orientation``: swap H×W for TIFF codes 5–8.

    Canvas is remapped in dest space, then stored back in the shared
    system (dest ``(0, 0)`` is ``origin``).
    """
    code = int(attrs["orientation"])
    if code not in _ORIENTATION:
        raise ValueError(f"orientation: invalid TIFF code {code} (expected 1–8)")
    swap = _ORIENTATION[code][0]
    height = x.meta.width if swap else x.meta.height
    width = x.meta.height if swap else x.meta.width
    origin_row, origin_col = x.meta.origin
    cx, cy, cw, ch = x.meta.canvas
    local = (cx - origin_col, cy - origin_row, cw, ch)
    mx, my, mw, mh = _orient_rect(code, local, x.meta.width, x.meta.height)
    return x.meta.with_size(
        height=height,
        width=width,
        ndim=2 if x.meta.is_1d and swap else x.meta.ndim,
        canvas=(mx + origin_col, my + origin_row, mw, mh),
    )


def _out_meta_transpose(x: Array, attrs: Dict[str, Any]) -> ArrayMeta:
    """Geometry policy ``transpose``: axis ``i`` of the output is axis
    ``axes[i]`` of the 3-axis input, and the channel axis moves with it.

    Origin and canvas describe the rows and the columns. When the rows and
    the columns trade places, they are remapped as by ``orientation`` 5.
    """
    axes = tuple(int(axis) for axis in attrs["axes"])
    meta = x.meta
    if meta.ndim != 3 or sorted(axes) != [0, 1, 2]:
        raise ValueError(
            f"transpose: axes {list(axes)} is not a permutation of the axes of shape {x.shape}"
        )
    rows_axis, cols_axis = meta.spatial_axes
    canvas = meta.canvas
    if axes.index(rows_axis) > axes.index(cols_axis):
        origin_row, origin_col = meta.origin
        cx, cy, cw, ch = canvas
        canvas = (cy - origin_row + origin_col, cx - origin_col + origin_row, ch, cw)
    return meta.copy(
        shape=tuple(meta.shape[axis] for axis in axes),
        channel_axis=axes.index(meta.channel_axis),
        canvas=canvas,
    )


def _out_meta_reshape(x: Array, attrs: Dict[str, Any]) -> ArrayMeta:
    """Geometry policy ``reshape``: the same samples as ``shape``, as
    ``np.reshape``. A 3-axis result has its channels on the last axis;
    ``Array.reshape`` is the way to choose another channel axis.
    """
    shape = tuple(int(axis) for axis in attrs["shape"])
    return reshaped_meta(x.meta, shape)


def _out_meta_broadcast(x: Array, attrs: Dict[str, Any], *more: Array) -> ArrayMeta:
    """Geometry policy ``broadcast``: NumPy's broadcast shape of every input.

    The inputs have one dtype and the same number of axes. The result is
    the meta of the first input that already has the broadcast shape, or a
    whole canvas with ``x``'s channel axis when none does.
    """
    inputs = (x, *more)
    shape = np.broadcast_shapes(*(inp.shape for inp in inputs))
    for inp in inputs:
        if inp.shape == shape:
            return inp.meta
    return _meta_from_shape(shape, x.dtype, x.meta.channel_axis)


@dataclass(frozen=True)
class OpNode:
    """A catalog engine op applied to ``inputs``."""

    op: str
    inputs: Tuple[Array, ...]
    attrs: Mapping[str, Any]
    out_meta: ArrayMeta


def check_engine_layout(meta: ArrayMeta) -> None:
    """Raise when the engine cannot read an array of ``meta``.

    The engine reads the channels of a 3-axis array last, ``(H, W, C)``, or
    first, ``(C, H, W)``, never on axis 1.
    """
    if meta.ndim == 3 and meta.channel_axis == 1:
        raise NotImplementedError(
            f"shape {meta.shape} with channel_axis 1 is not supported; put the "
            "channels last, (H, W, C), or first, (C, H, W), with mi.moveaxis"
        )


def _check_input(
    op_meta: OpMeta, in_channels: Optional[int], x: Array, index: int = 0
) -> None:
    """Raise when ``x`` is not input ``index`` of the op ``op_meta``.

    An ``rgb_`` op mixes the channels of each pixel, so it needs them on the
    last axis. A ``cfa_`` op reads a mono mosaic, ``(H, W)`` or ``(H, W, 1)``.
    ``in_channels`` is the channel count the op needs, or ``None`` for any.
    """
    name = op_meta.name
    check_engine_layout(x.meta)
    channel_axis = x.meta.channel_axis
    if op_meta.requires_2d and x.meta.is_1d:
        raise ValueError(f"op {name!r} requires a 2D input, got shape {x.shape}")
    if op_meta.is_rgb and channel_axis not in (None, 2):
        raise ValueError(
            f"op {name!r} needs the channels on the last axis, as (H, W, 3); got "
            f"shape {x.shape} with channel_axis {channel_axis}. Move them with "
            f"mi.moveaxis(x, {channel_axis}, -1)"
        )
    if op_meta.is_cfa and channel_axis not in (None, 2):
        raise ValueError(
            f"op {name!r} needs a mono (H, W) or (H, W, 1) array; got shape {x.shape} "
            f"with channel_axis {channel_axis}"
        )
    if in_channels is not None and x.meta.channels != in_channels:
        raise ValueError(
            f"op {name!r} input[{index}]: expected {in_channels} channel(s), "
            f"got {x.meta.channels}"
        )


_SCALAR_FLOAT = {"f32": np.float32, "f64": np.float64}
_ARRAY_DTYPE = {
    "f32_array": np.float32,
    "f64_array": np.float64,
    "i32_array": np.int32,
}


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
        if isinstance(value, bool):
            raise TypeError(f"attr {key!r} must be an int")
        try:
            return operator.index(value)
        except TypeError as e:
            raise TypeError(f"attr {key!r} must be an int") from e
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
        arr = np.ascontiguousarray(value, dtype=_ARRAY_DTYPE[typ]).reshape(-1)
        if count and arr.size != count:
            raise ValueError(f"{key} must have {count} elements, got {arr.size}")
        if arr.size < 1:
            raise ValueError(f"{key} must be non-empty")
        return arr
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


def _check_input_count(engine_op: EngineOp, count: int) -> None:
    """Raise when the op ``engine_op`` does not take ``count`` inputs."""
    expected = engine_op._n_inputs
    if engine_op._variable_input:
        if count < expected:
            raise ValueError(
                f"op {engine_op.meta.name!r}: expected at least {expected} "
                f"input(s), got {count}"
            )
    elif count != expected:
        raise ValueError(
            f"op {engine_op.meta.name!r}: expected {expected} input(s), got {count}"
        )


def emit(engine_op: EngineOp, x: Array, /, *more: Array, **attrs: Any) -> Array:
    """Validate inputs and attrs, ask the op for output meta, and build a lazy node.

    ``x`` is input 0 and ``more`` are the inputs after it. An op with no
    inputs takes only ``x``, as the meta of its output. When the op's last
    input is variable, input ``i`` past the catalog entries is checked
    against the last entry.
    """
    name = engine_op.meta.name
    if engine_op._n_inputs == 0:
        if more:
            raise ValueError(f"op {name!r}: takes no inputs, got {1 + len(more)}")
        inputs: Tuple[Array, ...] = ()
    else:
        inputs = (x, *more)
        _check_input_count(engine_op, len(inputs))
    for index, inp in enumerate(inputs):
        entry = min(index, len(engine_op._in_channels) - 1)
        _check_input(engine_op.meta, engine_op._in_channels[entry], inp, index)
    coerced = _validate_attrs(name, engine_op._attr_specs, attrs)
    out_meta = engine_op.infer_out_meta(x, coerced, *more)
    check_engine_layout(out_meta)
    node = OpNode(
        op=name,
        inputs=inputs,
        attrs=MappingProxyType(coerced),
        out_meta=out_meta,
    )
    return Array(_meta=out_meta, _node=node)


def op(name: str, x: Array, /, *more: Array, **attrs: Any) -> Array:
    """Emit a named engine op (thin alias over ``engines.ops.OPS_BY_NAME``)."""
    from .ops import OPS_BY_NAME

    engine_op = OPS_BY_NAME.get(name)
    if engine_op is None:
        raise ValueError(f"unknown engine op {name!r}")
    return emit(engine_op, x, *more, **attrs)


def flush(x: Array) -> Array:
    """Materialize a lazy graph into a concrete source Array."""
    return Array(x.realize())


def _run_op_arrays(
    op_arrays: List[Array],
    values: Dict[int, np.ndarray],
    root: Array,
) -> None:
    """Run ``op_arrays`` as one engine call that writes ``root``."""
    parent = PerfTimer.current()
    timing = PerfTimer.inactive
    if parent is not None and get_engine_timing() >= EngineTiming.SEGMENTS:
        timing = parent.start_step("graph_compute")
    try:
        get_default_engine().execute(op_arrays, values, root)
    finally:
        timing.close()


def realize(root: Array, *, force_recompute: bool = False) -> np.ndarray:
    """Run the graph if needed and return ``root``'s pixels.

    Only ``root`` keeps its pixels. The engine's buffers for the other
    arrays are dropped on return.

    An array the caller realized earlier has ``_data``, which is handed to
    the engine as an extra bind. The engine skips a producer when that
    buffer covers the array's canvas. ``force_recompute`` omits those binds
    and reruns every op.

    A reshape of pixels is ``np.reshape`` of those pixels, a view. When
    ``root`` is only reshapes of some array, that array is realized and
    the result is its pixels reshaped. The engine would otherwise copy:
    a ``reshape`` that is the last node has the caller's buffer to fill
    and its producer's buffer to fill it from.

    An array that several row-by-row ops read is not stored as a whole
    image: the engine computes its rows as the readers need them. Where
    the reads of one row are far apart, as with a flip of the array or a
    shifted read, the row is computed again for the later read. To compute
    such an array once, realize it first; later graphs read its pixels.
    An array read by a tiled or whole-image op (``rot90``, ``transpose``)
    is still stored as a whole image.
    """
    if root._data is not None and not force_recompute:
        return _as_numpy_shape(root)

    source = root
    while source._node is not None and source._node.op == "reshape":
        if source._data is not None and not force_recompute:
            break
        source = source._node.inputs[0]
    if source is not root:
        realize(source, force_recompute=force_recompute)
        root._data = _seal_ndarray(np.reshape(source._data, root.meta.buffer_shape))
        return _as_numpy_shape(root)

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
        return _as_numpy_shape(root)

    _run_op_arrays(op_arrays, values, root)

    root_pixels = values.get(id(root))
    if root_pixels is None:
        raise RuntimeError("realize finished without materializing root")
    root._data = _seal_ndarray(root_pixels)
    return _as_numpy_shape(root)


def _as_numpy_shape(array: Array) -> np.ndarray:
    """``array``'s cached buffer in its NumPy shape: ``(N,)`` for a 1D array."""
    assert array._data is not None
    if array.meta.is_1d:
        return array._data.reshape(array.meta.shape)
    return array._data
