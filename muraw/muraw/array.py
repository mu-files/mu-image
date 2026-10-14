# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files
"""Array handle: concrete ndarray buffer and/or lazy engine graph node."""

from __future__ import annotations

import operator
from dataclasses import dataclass, replace
from math import ceil
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence, Tuple

import numpy as np
from numpy.lib.array_utils import normalize_axis_index, normalize_axis_tuple
from numpy.typing import DTypeLike


def _base_chain(arr: np.ndarray) -> list[np.ndarray]:
    """``arr`` then every ndarray along ``.base``."""
    chain: list[np.ndarray] = []
    seen: set[int] = set()
    current: Any = arr
    while isinstance(current, np.ndarray) and id(current) not in seen:
        seen.add(id(current))
        chain.append(current)
        current = current.base
    return chain


def _seal_ndarray(arr: np.ndarray) -> np.ndarray:
    """Mark ``arr`` and every ndarray along ``.base`` read-only. Do not copy."""
    arr = np.asarray(arr)
    for current in _base_chain(arr):
        if current.flags.writeable:
            current.setflags(write=False)
    return arr


if TYPE_CHECKING:
    from .engines.graph import OpNode


# The dtypes an Array can hold. Their names match the native ``MuImgDType`` names.
SUPPORTED_DTYPES = tuple(np.dtype(t) for t in (np.float32, np.float16, np.uint8, np.uint16))


def supported_dtype(value: DTypeLike) -> np.dtype:
    """``value`` (a dtype, scalar type, or string such as ``"f4"``) as a
    native-byte-order ``np.dtype`` that an Array can hold.

    An unsupported dtype raises ``ValueError`` with a message that says how
    to convert the input. No input is converted automatically.
    """
    supported = ", ".join(dtype.name for dtype in SUPPORTED_DTYPES)
    dtype = None
    if value is not None:
        try:
            dtype = np.dtype(value)
        except TypeError:
            pass
    if dtype is None:
        raise ValueError(f"{value!r} is not a dtype. Supported dtypes: {supported}.")
    for supported_type in SUPPORTED_DTYPES:
        if dtype == supported_type:
            return supported_type
    if dtype.kind == "f" and dtype.itemsize == 8:
        raise ValueError(
            "float64 is not supported; muimage computes in float32. "
            "Convert with arr.astype(np.float32)."
        )
    if not dtype.isnative and dtype.newbyteorder("=") in SUPPORTED_DTYPES:
        raise ValueError(
            f"big-endian {dtype.str} is not supported. Convert to native byte order "
            'with arr.astype(arr.dtype.newbyteorder("=")).'
        )
    raise ValueError(f"dtype {dtype} is not supported. Supported dtypes: {supported}.")


# An array argument: an ``Array``, or an ndarray that is ingested with ``Array(...)``.
type ArrayLike = Array | np.ndarray

# An operand of +, -, *, /: an array, a number, or a (nested) list of numbers.
# A float annotation also accepts int and bool. Complex numbers are rejected.
type ArithmeticOperand = (
    ArrayLike | float | np.integer | np.floating | np.bool_ | Sequence[Any]
)


@dataclass(frozen=True)
class ArrayMeta:
    # One of ``SUPPORTED_DTYPES``. Any value ``supported_dtype`` accepts is
    # converted on construction.
    dtype: np.dtype
    # The NumPy shape: ``(N,)``, ``(H, W)``, three axes, or a stack of
    # frames on axis 0 and three axes per frame, ``(N, H, W, C)``. A mono
    # array can be ``(H, W)`` or ``(H, W, 1)``.
    shape: Tuple[int, ...]
    # Buffer top-left in the shared canvas coordinate system, as (row, col).
    origin: Tuple[int, int] = (0, 0)
    # Rect in that same system: (x0, y0, width, height). A later view or
    # crop uses it. When this array is the whole canvas, (x0, y0) is
    # (origin col, origin row).
    canvas: Tuple[int, int, int, int] = (0, 0, 0, 0)
    # The axis of a 3- or 4-axis shape that holds the channels: 2 for
    # ``(H, W, C)``, 1 for ``(H, C, W)``, 0 for ``(C, H, W)``, and 3 for
    # ``(N, H, W, C)``. A 4-axis shape has its frames on axis 0, so its
    # channels are on axis 1, 2, or 3. The other two axes are the rows and
    # the columns, in that order. ``None`` means the last axis, and a
    # negative axis counts from the end. A shape with fewer axes has no
    # channel axis, so ``channel_axis`` is ``None``.
    channel_axis: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dtype", supported_dtype(self.dtype))
        shape = tuple(int(v) for v in self.shape)
        if not 1 <= len(shape) <= 4:
            raise ValueError("array must be (N,), (H,W), 3 axes, or 4 axes")
        object.__setattr__(self, "shape", shape)
        if len(shape) >= 3:
            ndim = len(shape)
            axis = ndim - 1 if self.channel_axis is None else operator.index(self.channel_axis)
            axis = normalize_axis_index(axis, ndim, "channel_axis")
            if ndim == 4 and axis == 0:
                raise ValueError(
                    f"shape {shape}: axis 0 of a 4-axis array holds its frames, "
                    "not its channels"
                )
            object.__setattr__(self, "channel_axis", axis)
            if shape[axis] < 1:
                raise ValueError(f"unsupported channel count: {shape[axis]}")
        elif self.channel_axis is not None:
            raise ValueError(
                f"shape {shape} has no channel axis, got channel_axis={self.channel_axis}"
            )
        if min(shape) < 1:
            raise ValueError(f"shape dimensions must be at least 1, got {shape}")

    @property
    def spatial_axes(self) -> Tuple[int, int]:
        """The axes of ``shape`` that hold the rows and the columns.

        A 1D array's one axis is its columns, and it has one row.
        """
        if self.channel_axis is None:
            return (0, 1)
        first_axis = 1 if self.ndim == 4 else 0
        rows_axis, cols_axis = (
            axis for axis in range(first_axis, self.ndim) if axis != self.channel_axis
        )
        return (rows_axis, cols_axis)

    @property
    def frames(self) -> int:
        """Frames in a 4-axis stack, its axis 0. Any other array is one frame."""
        return self.shape[0] if self.ndim == 4 else 1

    @property
    def height(self) -> int:
        """Rows in the buffer. A 1D array is one row."""
        return 1 if self.is_1d else self.shape[self.spatial_axes[0]]

    @property
    def width(self) -> int:
        """Columns in the buffer. A 1D array's length is its width."""
        return self.shape[0] if self.is_1d else self.shape[self.spatial_axes[1]]

    @property
    def channels(self) -> int:
        return 1 if self.channel_axis is None else self.shape[self.channel_axis]

    @property
    def is_1d(self) -> bool:
        return len(self.shape) == 1

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def buffer_shape(self) -> Tuple[int, ...]:
        """The shape of the buffer the engine reads and writes.

        It equals ``shape`` except for a 1D array, whose buffer is ``(1, N)``.
        """
        return (1, self.shape[0]) if self.is_1d else self.shape

    def copy(self, **changes: Any) -> "ArrayMeta":
        return replace(self, **changes)

    def with_size(
        self,
        *,
        height: Optional[int] = None,
        width: Optional[int] = None,
        channels: Optional[int] = None,
        frames: Optional[int] = None,
        ndim: Optional[int] = None,
        **changes: Any,
    ) -> "ArrayMeta":
        """A copy with a new height, width, channel count, frame count, or
        rank.

        Each size that is not given keeps this array's value. ``ndim``
        defaults to this array's rank, except that more than one channel
        always needs a channel axis. A 3- or 4-axis result of this array's
        rank keeps its channel axis; otherwise the channels are last.
        ``frames`` is the size of axis 0 of a 4-axis result. A 1D result
        must be one row and one channel. ``changes`` sets other fields, as
        in ``copy``.
        """
        height = self.height if height is None else height
        width = self.width if width is None else width
        channels = self.channels if channels is None else channels
        frames = self.frames if frames is None else frames
        ndim = self.ndim if ndim is None else ndim
        if channels > 1 and ndim == 2:
            ndim = 3
        channel_axis = None
        if ndim == 1:
            if height != 1 or channels != 1:
                raise ValueError(
                    "a 1D array has one row and one channel; got "
                    f"height={height}, channels={channels}"
                )
            shape: Tuple[int, ...] = (width,)
        elif ndim == 2:
            shape = (height, width)
        else:
            channel_axis = self.channel_axis if self.ndim == ndim else ndim - 1
            first_axis = ndim - 3
            sizes = [height, width]
            sizes.insert(channel_axis - first_axis, channels)
            shape = tuple([frames] * first_axis + sizes)
        return replace(self, shape=shape, channel_axis=channel_axis, **changes)


def meta_from_array(arr: np.ndarray, channel_axis: Optional[int] = -1) -> ArrayMeta:
    """Whole-canvas meta for ``arr``. ``channel_axis`` applies to 3 or 4 axes only."""
    return _meta_from_shape(
        arr.shape, supported_dtype(arr.dtype), channel_axis if arr.ndim >= 3 else None
    )


def spatial_reading(meta: ArrayMeta) -> Tuple[bool, Tuple[int, int, int]]:
    """Which axes of an array of ``meta`` are its rows, columns, and
    channels: whether its channels come first, and its (rows, columns,
    channels).

    An array has an origin and a canvas, so it has to say which of its
    axes are spatial. A 3-axis array whose channel axis is 0 is CHW:
    channels, then rows, then columns. Every other array is packed: rows,
    then columns, then channels if there is a channel axis, after the
    frames of a 4-axis stack.
    """
    return meta.channel_axis == 0, (meta.height, meta.width, meta.channels)


def _arithmetic_function(
    apply: Callable[[Any, Any], "Array"], x1: ArithmeticOperand, x2: ArithmeticOperand
) -> "Array":
    """``apply(x1, x2)`` with at least one Array: ``x1`` is ingested when
    neither is."""
    if not isinstance(x1, Array) and not isinstance(x2, Array):
        x1 = Array(np.asarray(x1))
    return apply(x1, x2)


def add(x1: ArithmeticOperand, x2: ArithmeticOperand) -> "Array":
    """NumPy's ``add``: ``x1 + x2``, broadcast. Integer results saturate."""
    return _arithmetic_function(operator.add, x1, x2)


def subtract(x1: ArithmeticOperand, x2: ArithmeticOperand) -> "Array":
    """NumPy's ``subtract``: ``x1 - x2``, broadcast. Integer results saturate."""
    return _arithmetic_function(operator.sub, x1, x2)


def multiply(x1: ArithmeticOperand, x2: ArithmeticOperand) -> "Array":
    """NumPy's ``multiply``: ``x1 * x2``, broadcast. Integer results saturate."""
    return _arithmetic_function(operator.mul, x1, x2)


def divide(x1: ArithmeticOperand, x2: ArithmeticOperand) -> "Array":
    """NumPy's ``divide``: ``x1 / x2``, broadcast. Integer inputs give float32."""
    return _arithmetic_function(operator.truediv, x1, x2)


# The most input channels, and the most output channels, of a matmul.
_MATMUL_MAX_CHANNELS = 32

# NumPy's signature for matmul, quoted in its error messages.
_MATMUL_SIGNATURE = "(n?,k),(k,m?)->(n?,m?)"


def matmul(x1: ArrayLike | Sequence[Any], x2: ArrayLike | Sequence[Any]) -> "Array":
    """NumPy's ``matmul``: ``x1 @ x2``, for a per-pixel channel transform.

    One operand is an array whose last axis is its channels, and the other
    is a constant matrix of at most 32 by 32: ``rgb @ M.T`` applies ``M``
    to every pixel, ``rgba @ B`` with a ``(4, 3)`` ``B`` gives RGB, and
    ``rgb @ [0.299, 0.587, 0.114]`` gives an ``(H, W)`` luma image.
    ``B @ v`` works for a 1D ``v``. Other uses raise
    ``NotImplementedError``; ``np.matmul(x.realize(), B)`` runs them.

    An integer matrix keeps an integer array's dtype and the result
    saturates; a float matrix makes an integer array float32.
    """
    if _matmul_array_is_second(x1, x2):
        return _matmul_matrix_first(_matmul_constant(x1, 0), x2)
    array = x1 if isinstance(x1, Array) else Array(np.asarray(x1))
    return _matmul_array_first(array, _matmul_constant(x2, 1))


def _is_lazy(operand: Any) -> bool:
    return isinstance(operand, Array) and operand._node is not None


def _matmul_array_is_second(x1: Any, x2: Any) -> bool:
    """Whether ``x2`` is the array and ``x1`` the matrix in ``x1 @ x2``.
    A computed ``x2`` is the array unless ``x1`` is computed too. A concrete
    ``x2`` is the array when ``x1`` is not an ``mi.Array`` and has at most
    2 axes, so ``M @ mi.Array(v)`` and ``rgb @ mi.Array(M)`` both work."""
    if not isinstance(x2, Array):
        return False
    if _is_lazy(x2):
        return not _is_lazy(x1)
    return not isinstance(x1, Array) and np.ndim(x1) <= 2


def _matmul_constant(operand: Any, index: int) -> np.ndarray:
    """``operand`` as the constant matrix of a matmul, where ``index`` is
    its position in ``x1 @ x2``."""
    if _is_lazy(operand):
        raise NotImplementedError(
            "matmul with a computed mi.Array as the matrix is not supported yet; "
            "realize() the matrix first"
        )
    matrix = np.asarray(operand)
    if matrix.dtype.kind not in "biuf":
        raise TypeError(f"matmul: expected an array of numbers, got {operand!r}")
    if matrix.ndim == 0:
        raise ValueError(
            f"matmul: Input operand {index} does not have enough dimensions "
            f"(has 0, gufunc core with signature {_MATMUL_SIGNATURE} requires 1)"
        )
    if matrix.ndim > 2:
        raise NotImplementedError(
            f"matmul with a matrix of shape {matrix.shape} is not supported; "
            "use np.matmul(x.realize(), matrix)"
        )
    return matrix


def _matmul_array_first(array: "Array", matrix: np.ndarray) -> "Array":
    """``array @ matrix``: each pixel's channels times ``matrix``."""
    from .engines.graph import op

    meta = array.meta
    input_channels = array.shape[-1]
    if matrix.shape[0] != input_channels:
        raise ValueError(
            "matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc "
            f"signature {_MATMUL_SIGNATURE} (size {matrix.shape[0]} is different from "
            f"{input_channels})"
        )
    if meta.ndim >= 3 and meta.channel_axis != meta.ndim - 1:
        raise NotImplementedError(
            f"matmul of shape {array.shape} with channel_axis {meta.channel_axis} is not "
            "supported: NumPy sums over the last axis, which is not the channel axis. "
            "Move the channels last with mi.moveaxis, or use np.matmul(x.realize(), matrix)"
        )
    if meta.ndim == 1 and matrix.ndim == 1:
        raise NotImplementedError(
            "matmul of two 1D arrays is one number, which mi.Array does not hold; "
            "use np.dot(x.realize(), matrix)"
        )
    columns = matrix.shape[1] if matrix.ndim == 2 else 1
    if max(input_channels, columns) > _MATMUL_MAX_CHANNELS:
        raise NotImplementedError(
            f"matmul with a matrix of shape {matrix.shape} is not supported: at most "
            f"{_MATMUL_MAX_CHANNELS} input and output channels. "
            "Use np.matmul(x.realize(), matrix)"
        )
    dtype = array.dtype
    if _is_integer(dtype) and matrix.dtype.kind == "f":
        dtype = np.dtype(np.float32)
    if meta.ndim >= 3:
        pixels_shape = array.shape
    elif meta.ndim == 2:
        pixels_shape = (array.shape[0], 1, input_channels)
    else:
        pixels_shape = (1, 1, input_channels)
    pixels = _reshape_layout(array, pixels_shape)
    out = op(
        "matmul",
        pixels.astype(dtype),
        matrix=[float(value) for value in matrix.reshape(-1)],
        columns=columns,
    )
    out_shape = array.shape[:-1] + ((columns,) if matrix.ndim == 2 else ())
    # A stack whose channels are summed away is (N, H, W): mono frames.
    return _reshape_layout(out, out_shape, 0 if len(out_shape) == 3 and meta.ndim == 4 else None)


def _matmul_matrix_first(matrix: np.ndarray, array: "Array") -> "Array":
    """``matrix @ array``. NumPy sums over ``array``'s second-to-last axis,
    which is a spatial axis unless ``array`` is 1D, so only a 1D ``array``
    is supported: ``matrix @ v`` is ``v @ matrix.T``."""
    if array.meta.ndim != 1:
        raise NotImplementedError(
            f"matmul of a matrix and an array of shape {array.shape} is not supported: "
            "NumPy sums over the array's second-to-last axis, which is not its channels. "
            "Use array @ matrix.T, or np.matmul(matrix, x.realize())"
        )
    if matrix.ndim == 1:
        raise NotImplementedError(
            "matmul of two 1D arrays is one number, which mi.Array does not hold; "
            "use np.dot(matrix, x.realize())"
        )
    if matrix.shape[1] != array.shape[0]:
        raise ValueError(
            "matmul: Input operand 1 has a mismatch in its core dimension 0, with gufunc "
            f"signature {_MATMUL_SIGNATURE} (size {array.shape[0]} is different from "
            f"{matrix.shape[1]})"
        )
    return _matmul_array_first(array, matrix.T)


# The scalar op and the op between two arrays for each operator.
_ELEMENTWISE_OPS = {
    "add_scalar": "add",
    "sub_scalar": "subtract",
    "mul_scalar": "multiply",
    "div_scalar": "divide",
}


def _scalar_values(
    meta: ArrayMeta, const: np.ndarray
) -> Optional[Tuple[list[float], bool]]:
    """``const`` as the ``value`` attr of a scalar op on an array of
    ``meta``, and whether it holds one value per frame: one value, one per
    channel, or one per frame of a 4-axis stack. ``None`` when ``const``
    would change the array's shape or varies along any other axes, so it
    needs the op between two arrays.
    """
    shape = meta.shape
    try:
        if np.broadcast_shapes(const.shape, shape) != shape:
            return None
    except ValueError:
        return None
    padded_shape = (1,) * (len(shape) - const.ndim) + const.shape
    varying = [axis for axis, size in enumerate(padded_shape) if size > 1]
    values = [float(v) for v in const.reshape(-1)]
    if not varying or varying == [meta.channel_axis]:
        return values, False
    if meta.ndim == 4 and varying == [0]:
        return values, True
    return None


def _is_integer(dtype: np.dtype) -> bool:
    return dtype.kind == "u"


def _elementwise(name: str, first: "Array", second: "Array") -> "Array":
    """The op ``name`` between two arrays, with NumPy's broadcasting and
    dtype promotion. Integer division gives float32.

    Both arrays get the broadcast rank: a missing leading axis becomes
    size 1. Both must be CHW, or both packed.
    """
    from .engines.graph import op

    shape = np.broadcast_shapes(first.shape, second.shape)
    channel_axis = None
    if len(shape) == 3:
        channel_axis = next(a.meta.channel_axis for a in (first, second) if a.meta.ndim == 3)
    operands = []
    for operand in (first, second):
        padded = (1,) * (len(shape) - operand.meta.ndim) + operand.shape
        operands.append(_reshape_layout(operand, padded, channel_axis))
    first, second = operands
    if spatial_reading(first.meta)[0] != spatial_reading(second.meta)[0]:
        raise NotImplementedError(
            f"{name} of shape {first.shape} with channel_axis {first.meta.channel_axis} "
            f"and shape {second.shape} with channel_axis {second.meta.channel_axis} "
            "is not supported yet; give both arrays the same channel_axis"
        )
    dtype = supported_dtype(np.result_type(first.dtype, second.dtype))
    if name == "divide" and _is_integer(dtype):
        dtype = np.dtype(np.float32)
    return op(name, first.astype(dtype), second.astype(dtype))


def _expand_pad(
    value: Any, name: str, axes: int, *, nonneg: bool = False
) -> list[int] | list[float]:
    """NumPy ``pad_width`` / ``constant_values`` broadcast to ``axes`` pairs.

    Returns ``[before, after]`` for each axis, flattened. An int is every
    side. A pair is ``(before, after)`` on every axis.
    """
    try:
        pairs = np.broadcast_to(np.asarray(value), (axes, 2))
    except ValueError:
        if axes >= 3:
            detail = "one (before, after) pair per axis"
        elif axes == 1:
            detail = "((before, after),)"
        else:
            detail = "((top, bottom), (left, right))"
        raise ValueError(
            f"{name}: expected an int, a pair, or {detail}; got {value!r}"
        ) from None
    sides = [pair.item() for pair in pairs.flat]
    if nonneg and any(side < 0 for side in sides):
        raise ValueError(f"{name}: values must be non-negative; got {sides}")
    return sides


def _slice_span(slc: slice, length: int, name: str) -> Tuple[int, int, int]:
    """Return ``(start, size, step)`` for a 1-d slice on an axis of ``length``.

    ``start`` is the first sample. ``size`` is the sample count. ``step``
    is the source step, including -1.
    """
    start, stop, step = slc.indices(length)
    n = len(range(start, stop, step))
    if n == 0:
        raise ValueError(f"{name}: slice results in an empty dimension")
    return start, n, step


def _as_axis_int(value: Any, name: str) -> int:
    """Require a real integer for a view keyword (bool is not an int)."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return int(value)


def _wrap_index(value: Any, size: int, axis: int, what: str) -> int:
    """NumPy wrap of one index on axis ``axis``, of length ``size``, into
    ``0 .. size-1``. ``what`` names the axis's entries in errors:
    ``"channel"`` or ``"frame"``."""
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{what} index must be an int")
    raw = int(value)
    index = raw + size if raw < 0 else raw
    if index < 0 or index >= size:
        raise IndexError(f"index {value} is out of bounds for axis {axis} with size {size}")
    return index


def _selection_key(
    key: Any, size: int, axis: int, what: str
) -> Tuple[Optional[list[int]], bool]:
    """The source indices that ``key``, the index on axis ``axis`` of
    length ``size``, reads, and whether it drops that axis. ``what`` names
    the axis's entries in errors: ``"channel"`` or ``"frame"``.

    The list is ``None`` when it is identity ``0..size-1``. An integer
    index drops the axis, as in NumPy; a slice or a sequence keeps it,
    even for one entry.
    """
    rank_drop = isinstance(key, (int, np.integer)) and not isinstance(
        key, (bool, np.bool_)
    )
    if isinstance(key, slice):
        start, stop, step = key.indices(size)
        indices = list(range(start, stop, step))
        if not indices:
            raise ValueError(f"{what}s: slice results in an empty dimension")
    else:
        if isinstance(key, np.ndarray):
            if key.dtype == np.bool_ or key.ndim != 1:
                raise TypeError(f"{what} index must be a 1-d sequence of ints")
            values = key.tolist()
        elif isinstance(key, (list, tuple)):
            values = list(key)
        elif rank_drop:
            values = [int(key)]
        else:
            raise TypeError(f"{what} index must be an int, a slice, or a sequence of ints")
        if not values:
            raise ValueError(f"{what}s: empty index")
        indices = [_wrap_index(value, size, axis, what) for value in values]
    if indices == list(range(size)):
        return None, rank_drop
    return indices, rank_drop


def _expand_index_key(
    key: Any, meta: ArrayMeta
) -> Tuple[Tuple[Any, ...], Tuple[int, ...]]:
    """Split an index key into one entry per axis of ``meta.shape`` and the
    new axes.

    At most one Ellipsis is replaced with ``slice(None)``,
    and a short key such as ``t[:]`` or ``t[10:90]`` is padded on the right
    with ``slice(None)``, as in NumPy.

    Each ``None`` (``np.newaxis``) adds an axis of length 1. The second
    value is the position of each new axis in the result's shape, so
    ``(H, W)[:, None, :]`` gives ``(1,)`` and ``(N,)[None, :, None]`` gives
    ``(0, 2)``. The result may have at most 4 axes.

    The rows and the columns must be slices. The channel axis, and the
    frame axis of a 4-axis stack, may be an int (which drops that axis), a
    slice, or a sequence of ints.
    """
    ndim = meta.ndim
    items: Tuple[Any, ...] = key if isinstance(key, tuple) else (key,)

    n_ellipsis = sum(item is Ellipsis for item in items)
    if n_ellipsis > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    n_explicit = sum(item is not Ellipsis and item is not None for item in items)
    if n_explicit > ndim:
        raise IndexError(
            f"too many indices for array: array is {ndim}-dimensional, "
            f"but {n_explicit} were indexed"
        )
    fill = (slice(None),) * (ndim - n_explicit)
    if n_ellipsis == 1:
        ellipsis_at = items.index(Ellipsis)
        items = items[:ellipsis_at] + fill + items[ellipsis_at + 1 :]
    else:
        items = items + fill

    axes = []
    new_axes = []
    rank = 0
    for item in items:
        if item is None:
            new_axes.append(rank)
            rank += 1
            continue
        axes.append(item)
        if not isinstance(item, (int, np.integer)):
            rank += 1
    if rank > 4:
        raise IndexError(
            f"newaxis would give {rank} axes; an array has at most 4 (N, H, W, C)"
        )
    spatial = axes if ndim < 3 else [axes[axis] for axis in meta.spatial_axes]
    if any(not isinstance(item, slice) for item in spatial):
        raise TypeError(
            "region must be slice objects; integer axes and masks are not supported"
        )
    return tuple(axes), tuple(new_axes)


@dataclass(frozen=True)
class _Window:
    """Spatial box plus optional channel and frame gathers.

    ``left`` and ``top`` are the first sample. ``row_step`` and ``col_step``
    are the source steps, including -1. ``src_channels`` is ``None`` when
    the channel axis is identity, and ``src_frames`` when the frame axis of
    a 4-axis stack is. ``drops_channel_axis`` and ``drops_frame_axis`` are
    true when an integer index removes that axis. ``left`` and ``top`` may
    be negative so a later view can reach back into the parent canvas.
    """

    left: int
    top: int
    width: int
    height: int
    row_step: int = 1
    col_step: int = 1
    src_channels: Optional[list[int]] = None
    drops_channel_axis: bool = False
    src_frames: Optional[list[int]] = None
    drops_frame_axis: bool = False

    def is_full(self, meta: ArrayMeta) -> bool:
        return (
            self.left == 0
            and self.top == 0
            and self.width == meta.width
            and self.height == meta.height
            and self.row_step == 1
            and self.col_step == 1
            and self.src_channels is None
            and self.src_frames is None
        )


def _window_from_slices(
    meta: ArrayMeta, rows: Any, cols: Any, channels: Any = None, frames: Any = None
) -> _Window:
    src_channels = None
    drops_channel_axis = False
    if channels is not None:
        if len(meta.shape) < 3:
            raise IndexError("too many indices for a (H, W) array")
        src_channels, drops_channel_axis = _selection_key(
            channels, meta.channels, meta.channel_axis, "channel"
        )
    src_frames = None
    drops_frame_axis = False
    if frames is not None:
        src_frames, drops_frame_axis = _selection_key(frames, meta.frames, 0, "frame")
    top, height, row_step = _slice_span(rows, meta.height, "rows")
    left, width, col_step = _slice_span(cols, meta.width, "cols")
    return _Window(
        left=left,
        top=top,
        width=width,
        height=height,
        row_step=row_step,
        col_step=col_step,
        src_channels=src_channels,
        drops_channel_axis=drops_channel_axis,
        src_frames=src_frames,
        drops_frame_axis=drops_frame_axis,
    )


def _window_from_rect(
    left: int | None,
    top: int | None,
    width: int | None,
    height: int | None,
) -> _Window:
    """The box given as ``left``, ``top``, ``width``, ``height`` keywords.

    ``width`` and ``height`` must be at least 1.
    """
    rect = (left, top, width, height)
    if any(v is None for v in rect):
        raise TypeError("view requires a slice region or left, top, width, and height")
    left_i = _as_axis_int(left, "left")
    top_i = _as_axis_int(top, "top")
    width_i = _as_axis_int(width, "width")
    height_i = _as_axis_int(height, "height")
    if width_i < 1 or height_i < 1:
        raise ValueError(
            f"view: invalid box top={top_i} left={left_i} "
            f"width={width_i} height={height_i}"
        )
    return _Window(left=left_i, top=top_i, width=width_i, height=height_i)


def _without_dropped_axes(array: "Array", window: _Window) -> "Array":
    """``array``, the view of ``window``, without the axes that integer
    indices in ``window`` drop. A stack that loses its frame axis is one
    ``(H, W, C)`` image; one that loses its channel axis is ``(N, H, W)``,
    a frame per index of axis 0 as in a CHW array."""
    meta = array.meta
    dropped = set()
    if window.drops_frame_axis:
        dropped.add(0)
    if window.drops_channel_axis:
        dropped.add(meta.channel_axis)
    if not dropped:
        return array
    shape = tuple(size for axis, size in enumerate(meta.shape) if axis not in dropped)
    channel_axis = None
    if len(shape) == 3:
        channel_axis = 0 if window.drops_channel_axis else 2
    return _reshape_layout(array, shape, channel_axis)


def rot90(m: ArrayLike, k: int = 1, axes: Tuple[int, int] = (0, 1)) -> "Array":
    """Rotate by 90° ``k`` times, from the first of ``axes`` toward the
    second. Same arguments as ``numpy.rot90``.

    A half turn is a canvas-keeping flip. A quarter turn in the plane of the
    rows and the columns is ``orientation``, per channel on a CHW array. Any
    other quarter turn is a flip and a permutation.
    """
    m = Array(m)
    axes = tuple(axes)
    ndim = m.meta.ndim
    if len(axes) != 2:
        raise ValueError("len(axes) must be 2.")
    if axes[0] == axes[1] or abs(axes[0] - axes[1]) == ndim:
        raise ValueError("Axes must be different.")
    if not all(-ndim <= axis < ndim for axis in axes):
        raise ValueError(f"Axes={axes} out of range for array of ndim={ndim}.")
    first, second = (axis % ndim for axis in axes)
    turns = int(k) % 4
    if turns == 0:
        return m
    if turns == 2:
        return flip(m, (first, second))
    rows_axis, cols_axis = m.meta.spatial_axes
    if {first, second} == {rows_axis, cols_axis}:
        import muimage as mi

        # A quarter turn from the rows toward the columns is 90° CCW, TIFF
        # orientation 8. The other direction is 90° CW, orientation 6.
        counterclockwise = (turns == 1) == (first == rows_axis)
        return mi.orientation(m, orientation=8 if counterclockwise else 6)
    order = list(range(ndim))
    order[first], order[second] = second, first
    if turns == 1:
        return _permute(flip(m, second), tuple(order))
    return flip(_permute(m, tuple(order)), second)


def flip(m: ArrayLike, axis: Any = None) -> "Array":
    """Reverse the samples along ``axis``. Same arguments as ``numpy.flip``.

    ``axis=None`` flips every axis. A canvas-keeping view, so a later crop
    can still reach the parent.
    """
    m = Array(m)
    ndim = m.meta.ndim
    flipped = range(ndim) if axis is None else normalize_axis_tuple(axis, ndim)
    key = tuple(
        slice(None, None, -1) if position in flipped else slice(None)
        for position in range(ndim)
    )
    return m.view(key, oob_valid=True)


def fliplr(m: ArrayLike) -> "Array":
    """Flip axis 1. Same as ``numpy.fliplr``."""
    m = Array(m)
    if m.meta.is_1d:
        raise ValueError("Input must be >= 2-d.")
    return flip(m, 1)


def flipud(m: ArrayLike) -> "Array":
    """Flip axis 0. Same as ``numpy.flipud``: on a 1D array, reverse it."""
    return flip(m, 0)


def _node_permutation(array: "Array") -> Optional[Tuple[int, ...]]:
    """The axis order of the permutation that produced ``array``, or ``None``.

    A ``transpose`` node and an ``orientation`` 5 node, which swaps the rows
    and the columns, are permutations. ``array`` must present its node's
    output unchanged: an ingested array with an ``origin`` is not one.
    """
    node = array._node
    if node is None or array.meta != node.out_meta:
        return None
    if node.op == "transpose":
        return tuple(int(axis) for axis in node.attrs["axes"])
    if node.op != "orientation" or int(node.attrs["orientation"]) != 5:
        return None
    input_meta = node.inputs[0].meta
    rows_axis, cols_axis = input_meta.spatial_axes
    order = list(range(input_meta.ndim))
    order[rows_axis], order[cols_axis] = cols_axis, rows_axis
    return tuple(order)


def _permute(array: "Array", order: Tuple[int, ...]) -> "Array":
    """``array`` with its axes in ``order``: axis ``i`` of the result is axis
    ``order[i]`` of ``array``, as in ``numpy.transpose``.

    A permutation of a permutation is one permutation of the first one's
    input, so a permutation and its inverse give back that input. The
    identity returns ``array``. A swap of the rows and the columns that
    keeps the channel axis, and the frame axis of a stack, in place is
    ``orientation`` 5. Every other permutation is one ``transpose``.
    """
    inner_order = _node_permutation(array)
    if inner_order is not None:
        combined = tuple(inner_order[axis] for axis in order)
        return _permute(array._node.inputs[0], combined)
    meta = array.meta
    if order == tuple(range(meta.ndim)):
        return array
    from .engines import ops as engine_ops

    if meta.ndim == 2 or (
        order[meta.channel_axis] == meta.channel_axis and (meta.ndim < 4 or order[0] == 0)
    ):
        return engine_ops.orientation(array, orientation=5)
    return engine_ops.transpose(array, axes=list(order))


def transpose(a: ArrayLike, axes: Any = None) -> "Array":
    """Permute the axes. Same arguments as ``numpy.transpose``.

    ``axes=None`` reverses the axes, so ``(H, W, C)`` becomes ``(C, W, H)``.
    """
    a = Array(a)
    if axes is None:
        return _permute(a, tuple(reversed(range(a.meta.ndim))))
    axes = tuple(axes)
    if len(axes) != a.meta.ndim:
        raise ValueError("axes don't match array")
    order = normalize_axis_tuple(axes, a.meta.ndim, allow_duplicate=True)
    if len(set(order)) != len(order):
        raise ValueError("repeated axis in transpose")
    return _permute(a, order)


def permute_dims(a: ArrayLike, axes: Any = None) -> "Array":
    """NumPy 2's array API name for ``transpose``."""
    return transpose(a, axes)


def swapaxes(a: ArrayLike, axis1: int, axis2: int) -> "Array":
    """Swap two axes. Same arguments as ``numpy.swapaxes``."""
    a = Array(a)
    first = normalize_axis_index(int(axis1), a.meta.ndim, "axis1")
    second = normalize_axis_index(int(axis2), a.meta.ndim, "axis2")
    order = list(range(a.meta.ndim))
    order[first], order[second] = order[second], order[first]
    return _permute(a, tuple(order))


def moveaxis(a: ArrayLike, source: Any, destination: Any) -> "Array":
    """Move axes to new positions. Same arguments as ``numpy.moveaxis``."""
    a = Array(a)
    sources = normalize_axis_tuple(source, a.meta.ndim, "source")
    destinations = normalize_axis_tuple(destination, a.meta.ndim, "destination")
    if len(sources) != len(destinations):
        raise ValueError(
            "`source` and `destination` arguments must have the same number of elements"
        )
    order = [axis for axis in range(a.meta.ndim) if axis not in sources]
    for dest, src in sorted(zip(destinations, sources)):
        order.insert(dest, src)
    return _permute(a, tuple(order))


def expand_dims(a: ArrayLike, axis: Any) -> "Array":
    """Insert axes of length 1. Same arguments as ``numpy.expand_dims``.

    Adding a length-1 axis keeps the samples in the same order, so the
    result reads the same buffer. A result with more than 4 axes raises.
    The axis added to a 2D array is the channel axis.
    """
    a = Array(a)
    axes = tuple(axis) if isinstance(axis, (tuple, list)) else (axis,)
    out_ndim = len(axes) + a.meta.ndim
    new_axes = normalize_axis_tuple(axes, out_ndim)
    sizes = iter(a.shape)
    shape = tuple(1 if position in new_axes else next(sizes) for position in range(out_ndim))
    if len(shape) > 4:
        raise ValueError(
            f"expand_dims of shape {a.shape} gives {len(shape)} axes; an Array has at most 4"
        )
    channel_axis = new_axes[0] if a.meta.ndim == 2 and out_ndim == 3 else None
    return _reshape_layout(a, shape, channel_axis)


def squeeze(a: ArrayLike, axis: Any = None) -> "Array":
    """Remove axes of length 1. Same arguments as ``numpy.squeeze``.

    Removing a length-1 axis keeps the samples in the same order, so the
    result reads the same buffer. Squeezing every axis away raises, because
    an ``Array`` has at least one axis.
    """
    a = Array(a)
    if axis is None:
        removed = {position for position, size in enumerate(a.shape) if size == 1}
    else:
        if not isinstance(axis, (tuple, list)):
            axis = operator.index(axis)
        axes = normalize_axis_tuple(axis, a.meta.ndim, allow_duplicate=True)
        if len(set(axes)) != len(axes):
            raise ValueError("duplicate value in 'axis'")
        if any(a.shape[position] != 1 for position in axes):
            raise ValueError(
                "cannot select an axis to squeeze out which has size not equal to one"
            )
        removed = set(axes)
    shape = tuple(size for position, size in enumerate(a.shape) if position not in removed)
    if not shape:
        raise ValueError(f"squeeze of shape {a.shape} would leave no axes")
    return _reshape_layout(a, shape)


def _as_shape(shape: Any, size: Optional[int] = None) -> Tuple[int, ...]:
    """A ``shape`` argument as a tuple of ints. One int is a 1-tuple.

    With ``size``, one dimension may be ``-1``, as in ``np.reshape``: it
    becomes the size that gives the shape ``size`` samples. ``ArrayMeta``
    checks the rank and sizes, so every constructor rejects the same shapes
    with the same messages.
    """
    if isinstance(shape, (str, bytes, bool, np.bool_)):
        raise TypeError(f"shape must be an int or a sequence of ints, got {shape!r}")
    try:
        if isinstance(shape, (int, np.integer)):
            dims = (operator.index(shape),)
        else:
            dims = tuple(operator.index(dim) for dim in shape)
    except TypeError:
        raise TypeError(f"shape must be an int or a sequence of ints, got {shape!r}") from None
    if size is not None and -1 in dims:
        unknown = dims.index(-1)
        others = dims[:unknown] + dims[unknown + 1 :]
        known = int(np.prod(others))
        if -1 in others or known <= 0 or size % known:
            raise ValueError(f"cannot reshape array of size {size} into shape {shape}")
        dims = dims[:unknown] + (size // known,) + dims[unknown + 1 :]
    ArrayMeta(dtype=np.dtype(np.float32), shape=dims)
    return dims


def _meta_from_shape(
    shape: Tuple[int, ...], dtype: DTypeLike, channel_axis: Optional[int] = None
) -> ArrayMeta:
    """Build whole-canvas meta for ``(N,)``, ``(H, W)``, or 3 axes.

    ``channel_axis`` is as in ``ArrayMeta``: ``None`` on 3 axes means 2.
    """
    meta = ArrayMeta(dtype=dtype, shape=tuple(shape), channel_axis=channel_axis)
    return meta.copy(canvas=(0, 0, meta.width, meta.height))


def _with_meta(array: "Array", meta: ArrayMeta) -> "Array":
    """The same buffer or node as ``array``, presented with ``meta``.

    ``meta`` must describe the same buffer (same ``buffer_shape`` and dtype).
    """
    out = object.__new__(Array)
    out._take_fields(array, meta)
    return out


def _dtype_from_fill(fill_value: Any) -> np.dtype:
    """Element type of ``fill_value`` when ``full(..., dtype=None)``."""
    if isinstance(fill_value, bool):
        raise TypeError("unsupported fill_value dtype: bool")
    if isinstance(fill_value, np.generic) or (
        isinstance(fill_value, np.ndarray) and fill_value.ndim == 0
    ):
        return supported_dtype(np.asarray(fill_value).dtype)
    if isinstance(fill_value, np.ndarray):
        try:
            return supported_dtype(fill_value.dtype)
        except ValueError:
            pass
    if isinstance(fill_value, (int, float, list, tuple)):
        return np.dtype(np.float32)
    raise TypeError(f"unsupported fill_value type: {type(fill_value).__name__}")


def _fill_samples(fill_value: Any, channels: int) -> list[float]:
    """Scalar or length-``channels`` vector as a flat f32 list."""
    arr = np.asarray(fill_value)
    if arr.ndim == 0:
        return [float(arr.item())]
    if arr.ndim != 1:
        raise ValueError("fill_value must be a scalar or a length-C vector")
    if arr.size == 1:
        return [float(arr.reshape(-1)[0])]
    if arr.size == channels:
        return [float(v) for v in arr]
    raise ValueError(
        f"fill_value length {arr.size} does not match channel count {channels}"
    )


def _emit_fill(meta: ArrayMeta, fill_value: Any) -> "Array":
    import muimage as mi

    return mi.fill(
        Array(_meta=meta),
        value=_fill_samples(fill_value, meta.channels),
    )


def zeros(shape: Any, dtype: DTypeLike = np.float32) -> "Array":
    """Lazy array filled with 0. Same arguments as ``numpy.zeros`` for image rank."""
    return _emit_fill(_meta_from_shape(_as_shape(shape), supported_dtype(dtype)), 0)


def ones(shape: Any, dtype: DTypeLike = np.float32) -> "Array":
    """Lazy array filled with 1. Same arguments as ``numpy.ones`` for image rank."""
    return _emit_fill(_meta_from_shape(_as_shape(shape), supported_dtype(dtype)), 1)


def full(shape: Any, fill_value: Any, dtype: DTypeLike | None = None) -> "Array":
    """Lazy array filled with a constant. Same arguments as ``numpy.full`` for image rank."""
    et = supported_dtype(dtype) if dtype is not None else _dtype_from_fill(fill_value)
    return _emit_fill(_meta_from_shape(_as_shape(shape), et), fill_value)


def zeros_like(array: "Array", dtype: DTypeLike | None = None) -> "Array":
    """Lazy zeros with ``array``'s shape (and dtype unless ``dtype`` is set)."""
    et = array.dtype if dtype is None else supported_dtype(dtype)
    return zeros(array.shape, dtype=et)


def ones_like(array: "Array", dtype: DTypeLike | None = None) -> "Array":
    """Lazy ones with ``array``'s shape (and dtype unless ``dtype`` is set)."""
    et = array.dtype if dtype is None else supported_dtype(dtype)
    return ones(array.shape, dtype=et)


def _emit_ramp(
    meta: ArrayMeta,
    start: list[float],
    step: list[float],
    stop: list[float] | None,
) -> "Array":
    """A 1D ``meta`` ramps along its columns; a 2D one down its rows."""
    from .engines.graph import op

    along = "columns" if len(meta.shape) == 1 else "rows"
    attrs: dict[str, Any] = {"along": along, "start": start, "step": step}
    if stop is not None:
        attrs["stop"] = stop
    return op("ramp", Array(_meta=meta), **attrs)


def arange(
    start: Any,
    stop: Any = None,
    step: Any = 1,
    *,
    dtype: DTypeLike = np.float32,
) -> "Array":
    """Samples ``start + i * step`` for ``i`` below ``ceil((stop - start) / step)``.

    One argument is the stop, with start 0 and step 1, as in ``numpy.arange``.
    The samples are computed in float32, so they can differ from NumPy's
    float64 by a few float32 rounding steps. ``uint8`` and ``uint16``
    saturate where NumPy wraps. A step of 0 raises. An empty range raises:
    there are no empty images.
    """

    def require_number(value: Any, name: str) -> float:
        """A real Python or NumPy number. A sequence raises."""
        if isinstance(value, (bool, np.bool_, str, bytes)):
            raise TypeError(f"arange {name} must be a number")
        arr = np.asarray(value)
        if arr.ndim != 0 or arr.dtype.kind not in "iuf":
            raise TypeError(f"arange {name} must be a number")
        return float(arr)

    if stop is None:
        stop = start
        start = 0
    start_v = require_number(start, "start")
    stop_v = require_number(stop, "stop")
    step_v = require_number(step, "step")
    if step_v == 0.0:
        raise ZeroDivisionError("arange step must not be 0")
    count = ceil((stop_v - start_v) / step_v)
    if count < 1:
        raise ValueError(f"arange({start}, {stop}, {step}) is empty")
    numpy_dtype = supported_dtype(dtype)
    if np.issubdtype(numpy_dtype, np.integer):
        # NumPy stores start and start + step as the integer dtype, then
        # steps by their difference, so the step is an integer too.
        first, second = np.array([start_v, start_v + step_v]).astype(numpy_dtype)
        start_v = float(first)
        step_v = float(second) - start_v
    meta = _meta_from_shape((count,), numpy_dtype)
    return _emit_ramp(meta, [start_v], [step_v], None)


def _linspace_components(start: Any, stop: Any) -> Tuple[np.ndarray, np.ndarray]:
    """Broadcast ``start`` and ``stop`` to one float64 vector each.

    Two scalars come back with shape ``(1,)`` and the caller still reports
    ``(N,)``. A length-1 sequence stays length 1 so the result is ``(N, 1)``.
    """
    if isinstance(start, (bool, np.bool_)) or isinstance(stop, (bool, np.bool_)):
        raise TypeError("linspace start and stop must be numbers")
    start_arr = np.asarray(start)
    stop_arr = np.asarray(stop)
    if start_arr.dtype.kind not in "iuf" or stop_arr.dtype.kind not in "iuf":
        raise TypeError("linspace start and stop must be numbers")
    if start_arr.ndim > 1 or stop_arr.ndim > 1:
        raise ValueError("linspace start and stop must be scalars or 1D")
    try:
        start_b, stop_b = np.broadcast_arrays(start_arr, stop_arr)
    except ValueError as exc:
        raise ValueError("linspace start and stop must be the same length") from exc
    if start_b.ndim == 0:
        start_b = start_b.reshape(1)
        stop_b = stop_b.reshape(1)
    return (
        np.ascontiguousarray(start_b, dtype=np.float64),
        np.ascontiguousarray(stop_b, dtype=np.float64),
    )


def linspace(
    start: Any,
    stop: Any,
    num: Any = 50,
    *,
    endpoint: bool = True,
    dtype: DTypeLike = np.float32,
    axis: int = 0,
) -> "Array":
    """``num`` samples from ``start`` to ``stop``, like ``numpy.linspace``.

    A scalar ``start`` and ``stop`` return ``(num,)``. A length-``C`` pair
    returns ``(num, C)``: row ``y``, column ``c`` runs from ``start[c]`` to
    ``stop[c]``. ``endpoint`` true writes ``stop`` into the last sample.
    ``num`` of 1 is ``start``. ``axis`` other than 0 raises. The samples are
    computed in float32, so they can differ from NumPy by a few float32
    rounding steps.
    """
    if isinstance(num, bool) or isinstance(axis, bool):
        raise TypeError("linspace num and axis must be ints")
    try:
        count = operator.index(num)
        axis_i = operator.index(axis)
    except TypeError as exc:
        raise TypeError("linspace num and axis must be ints") from exc
    if count < 1:
        raise ValueError(f"linspace num must be >= 1, got {count}")
    if axis_i != 0:
        raise ValueError(f"linspace axis must be 0, got {axis_i}")
    if not isinstance(endpoint, (bool, np.bool_)):
        raise TypeError("linspace endpoint must be a bool")
    start_v, stop_v = _linspace_components(start, stop)
    scalar = np.ndim(start) == 0 and np.ndim(stop) == 0
    if count == 1:
        step_v = np.zeros_like(start_v)
        stop_arg = None
    elif endpoint:
        step_v = (stop_v - start_v) / (count - 1)
        stop_arg = [float(value) for value in stop_v]
    else:
        step_v = (stop_v - start_v) / count
        stop_arg = None
    if scalar:
        meta = _meta_from_shape((count,), supported_dtype(dtype))
    else:
        meta = _meta_from_shape((count, int(start_v.shape[0])), supported_dtype(dtype))
    return _emit_ramp(
        meta,
        [float(value) for value in start_v],
        [float(value) for value in step_v],
        stop_arg,
    )


def full_like(array: "Array", fill_value: Any, dtype: DTypeLike | None = None) -> "Array":
    """Lazy constant with ``array``'s shape. ``dtype`` defaults to the reference."""
    et = array.dtype if dtype is None else supported_dtype(dtype)
    return full(array.shape, fill_value, dtype=et)


def reshaped_meta(
    old: ArrayMeta, shape: Tuple[int, ...], channel_axis: Optional[int] = None
) -> ArrayMeta:
    """The meta of an array of ``old`` reshaped to ``shape`` with
    ``channel_axis`` (as in ``ArrayMeta``: ``None`` on 3 axes means 2).

    The sample count must not change. The origin stays. The canvas stays
    when the rows and the columns do (``(H, W)`` to ``(H, W, 1)``);
    otherwise the old canvas means nothing on the new shape, and the canvas
    is the new array at the old origin.
    """
    new = _meta_from_shape(shape, old.dtype, channel_axis)
    if int(np.prod(old.shape)) != int(np.prod(new.shape)):
        raise ValueError(
            f"cannot reshape {old.shape} to {new.shape}: the sample count differs"
        )
    row, col = old.origin
    canvas = old.canvas
    if spatial_reading(new)[1][:2] != spatial_reading(old)[1][:2]:
        canvas = (col, row, new.width, new.height)
    return new.copy(origin=old.origin, canvas=canvas)


def _reshape_layout(
    array: "Array", shape: Tuple[int, ...], channel_axis: Optional[int] = None
) -> "Array":
    """The same samples as ``array``, as ``shape`` with ``channel_axis``.

    The same shape is ``array`` itself. An array that has its pixels, a
    source or a realized one, is reshaped on the host, as ``np.reshape``.
    A lazy array gets a ``reshape`` node; the engine reads the producer's
    buffer as the new shape without a copy.
    """
    from types import MappingProxyType

    from .engines.graph import OpNode

    if tuple(shape) == array.shape:
        return array
    meta = reshaped_meta(array.meta, shape, channel_axis)
    if array._data is not None:
        out = object.__new__(Array)
        out._meta = meta
        out._data = np.reshape(array._data, meta.buffer_shape)
        out._node = None
        return out
    shape_attr = np.asarray(meta.buffer_shape, dtype=np.int32)
    return Array(
        _meta=meta,
        _node=OpNode(
            op="reshape",
            inputs=(array,),
            attrs=MappingProxyType({"shape": shape_attr}),
            out_meta=meta,
        ),
    )


def reshape(a: ArrayLike, /, shape: Any, order: str = "C") -> "Array":
    """The same samples as ``shape``, as ``np.reshape(a, shape)``.

    ``shape`` is an int or a sequence of ints, and one size may be ``-1``.
    A 3- or 4-axis result has its channels on the last axis. Only
    ``order="C"`` is supported.
    """
    a = Array(a)
    if order != "C":
        raise NotImplementedError(f"reshape order {order!r} is not supported; use order='C'")
    return _reshape_layout(a, _as_shape(shape, int(np.prod(a.shape))))


def _repeated_axes(arr: np.ndarray) -> Optional[Tuple[np.ndarray, Tuple[int, ...]]]:
    """Strip axes that repeat one sample, or ``None`` when nothing does.

    An axis longer than 1 with stride 0 is a NumPy broadcast. Index 0 of
    that axis is the sample, and the axis stays so the rank does not change.
    The returned counts are the repeat of each axis, 1 where the stride
    was not 0.
    """
    if arr.ndim == 0 or not any(
        int(length) > 1 and int(stride) == 0
        for length, stride in zip(arr.shape, arr.strides)
    ):
        return None
    slices = []
    reps = []
    for length, stride in zip(arr.shape, arr.strides):
        length = int(length)
        if length > 1 and int(stride) == 0:
            slices.append(slice(0, 1))
            reps.append(length)
        else:
            slices.append(slice(None))
            reps.append(1)
    return arr[tuple(slices)], tuple(reps)


def _tile_reps(meta: ArrayMeta, reps: Any) -> tuple[int, int, int, int]:
    """Right-align ``reps`` onto ``meta.shape`` and return ``(frame, row,
    col, channel)``.

    An int is the last axis. Missing leading counts are 1. A longer tuple
    would add an axis. A rank-2 array has no channel axis, so that count
    stays 1. A 1D array has only a column count. Only a 4-axis stack has a
    frame count other than 1.
    """
    shape = meta.shape
    if np.iterable(reps):
        counts = tuple(operator.index(value) for value in reps)
    else:
        counts = (operator.index(reps),)
    if len(counts) > len(shape):
        raise ValueError(f"tile reps {reps!r} would add an axis on shape {shape}")
    if any(count < 1 for count in counts):
        raise ValueError(f"tile reps must be >= 1, got {reps!r}")
    if len(shape) == 1:
        return 1, 1, counts[0], 1
    counts = (1,) * (len(shape) - len(counts)) + counts
    if meta.channel_axis is None:
        return 1, counts[0], counts[1], 1
    rows_axis, cols_axis = meta.spatial_axes
    frame_reps = counts[0] if meta.ndim == 4 else 1
    return frame_reps, counts[rows_axis], counts[cols_axis], counts[meta.channel_axis]


def tile(array: ArrayLike, reps: Any) -> "Array":
    """Repeat ``array`` like ``numpy.tile``.

    A count of 0 raises ``ValueError``. ``reps`` longer than the rank
    raises too, except 3 counts on a 1D array: NumPy promotes ``(N,)``
    to ``(1, 1, N)``, so the length becomes the channel count.
    """
    from .engines.graph import op

    array = Array(array)
    is_1d = array.meta.is_1d
    if isinstance(reps, (str, bytes)):
        raise TypeError(f"tile reps must be an int or a sequence of ints, got {reps!r}")
    if np.iterable(reps):
        reps = tuple(reps)
    reps_count = len(reps) if isinstance(reps, tuple) else 1
    if is_1d and reps_count == 3:
        # NumPy promotes (N,) to (1, 1, N), so the length becomes channels.
        array = _reshape_layout(array, (1, 1, array.shape[0]))
    elif is_1d and reps_count == 2:
        # NumPy promotes (N,) to (1, N), which is this array's buffer.
        array = _with_meta(array, array.meta.with_size(ndim=2))
    frame_reps, row_reps, col_reps, channel_reps = _tile_reps(array.meta, reps)
    return op(
        "tile",
        array,
        row_reps=row_reps,
        col_reps=col_reps,
        channel_reps=channel_reps,
        frame_reps=frame_reps if frame_reps != 1 else None,
    )


def broadcast_to(array: ArrayLike, shape: Any) -> "Array":
    """Stretch size-1 axes to ``shape``, like ``numpy.broadcast_to``.

    Shapes are right-aligned. A missing leading axis counts as length 1.
    Each source axis must be 1 or already the destination length. The
    stretch is ``tile`` with those repeat counts. A source whose axes land
    on a different height, width, or channel count is reshaped first, as a
    view of the same samples. ``(8, 8, 3)`` to ``(1080, 1920, 3)`` raises;
    repeating a patch is ``tile``.
    """
    array = Array(array)
    dest = _as_shape(shape)
    src = array.shape
    if len(dest) < len(src):
        raise ValueError(
            f"cannot broadcast shape {src} to {dest}: the source has more axes"
        )
    padded = (1,) * (len(dest) - len(src)) + src
    reps = []
    for src_len, dest_len in zip(padded, dest):
        if src_len == dest_len:
            reps.append(1)
        elif src_len == 1:
            reps.append(dest_len)
        else:
            raise ValueError(f"cannot broadcast shape {src} to {dest}")
    if padded != src:
        array = _reshape_layout(array, padded)
    if all(count == 1 for count in reps):
        return array
    return tile(array, tuple(reps))


def meshgrid(
    *xi: ArrayLike, sparse: bool = False, indexing: str = "xy"
) -> Tuple["Array", "Array"]:
    """Coordinate grids from two 1D vectors, like ``numpy.meshgrid``.

    Under ``xy`` indexing, ``x`` is the column samples and ``y`` the row
    samples, and both results are ``(len(y), len(x))``. Under ``ij`` both
    are ``(len(x), len(y))``. ``sparse=True`` returns the size-1 views
    instead, which broadcast to the full grid. Exactly two 1D inputs are
    accepted: NumPy's other counts do not give two images, and a rank-2
    input is an image, not a coordinate vector.
    """
    if indexing not in ("xy", "ij"):
        raise ValueError(
            f"meshgrid indexing must be 'xy' or 'ij', got {indexing!r}"
        )
    if len(xi) != 2:
        raise ValueError(f"meshgrid takes two 1D arrays, got {len(xi)}")
    x, y = (Array(vector) for vector in xi)
    for name, vector in (("x", x), ("y", y)):
        if len(vector.shape) != 1:
            raise ValueError(
                f"meshgrid {name} must be 1D, got shape {vector.shape}"
            )
    if indexing == "xy":
        first, second = x[None, :], y[:, None]
        shape = (y.shape[0], x.shape[0])
    else:
        first, second = x[:, None], y[None, :]
        shape = (x.shape[0], y.shape[0])
    if sparse:
        return first, second
    return broadcast_to(first, shape), broadcast_to(second, shape)


def composite(base: ArrayLike, *placements: tuple) -> "Array":
    """Write each source onto ``base``. ``base`` is unchanged.

    A placement is ``(origin, source)`` or ``(origin, source, channels)``.
    ``origin`` is ``(row, col)`` in ``base``, in NumPy index order, not
    ``(x, y)`` as in PIL or OpenCV; it is ``(col,)`` when ``base`` is 1D.
    ``channels`` is the dest channel of each of that source's channels.
    Without it, source channel ``i`` is written to dest channel ``i``. A
    later placement overwrites an earlier one where they overlap.

    The same placements work on a packed base ``(h, w, c)`` and a CHW
    base ``(c, h, w)``. For a source of height ``h`` and width ``w``, a
    placement is the NumPy assignment::

        ((r, c), src)          packed: out[r:r+h, c:c+w, :] = src
                               CHW:    out[:, r:r+h, c:c+w] = src
        ((r, c), src, chans)   packed: out[r:r+h, c:c+w, chans] = src
                               CHW:    out[chans, r:r+h, c:c+w] = src

    A source has the base's layout. A 2D source is one channel and fits
    either layout. A 3D source of the other layout raises ``ValueError``;
    transpose it first.
    """
    from .engines.ops import composite as composite_op

    base = Array(base)
    base_chw = base.meta.channel_axis == 0
    sources: list[Array] = []
    tops: list[int] = []
    lefts: list[int] = []
    dest_channels: list[int] = []
    scattered = False
    for placement in placements:
        if (
            not isinstance(placement, tuple)
            or not 2 <= len(placement) <= 3
            or not isinstance(placement[0], (tuple, list))
        ):
            raise TypeError(
                "a composite placement is (origin, source) or "
                "(origin, source, channels), as in "
                f"composite(base, ((row, col), source)), got {placement!r}"
            )
        origin, source, *channels = placement
        source = Array(source)
        if 4 in (base.meta.ndim, source.meta.ndim):
            raise NotImplementedError(
                f"composite of a 4-axis stack is not supported (base {base.shape}, "
                f"source {source.shape})"
            )
        if source.meta.ndim == 3 and (source.meta.channel_axis == 0) != base_chw:
            if base_chw:
                source_layout, axes = "packed", "(2, 0, 1)"
            else:
                source_layout, axes = "CHW", "(1, 2, 0)"
            raise ValueError(
                f"composite source of shape {source.shape} is {source_layout} and the base "
                f"of shape {base.shape} is not; give it the base's layout with "
                f"source.transpose{axes}"
            )
        if len(origin) == 2:
            row, col = origin
        elif len(origin) == 1 and base.meta.is_1d:
            row, col = 0, origin[0]
        else:
            expected = "(col,)" if base.meta.is_1d else "(row, col)"
            raise ValueError(
                f"composite origin for shape {base.shape} must be {expected}, got {origin!r}"
            )
        sources.append(source)
        tops.append(operator.index(row))
        lefts.append(operator.index(col))
        if channels:
            scattered = True
            dest_channels.extend(operator.index(channel) for channel in channels[0])
        else:
            dest_channels.extend(range(source.meta.channels))
    attrs: dict[str, Any] = {"tops": tops, "lefts": lefts}
    if scattered:
        attrs["dest_channels"] = dest_channels
    return composite_op(base, *sources, **attrs)


def _image_sequence(name: str, arrays: Sequence[ArrayLike]) -> list["Array"]:
    """The arrays in ``arrays``. An ndarray is the sequence of its first-axis
    slices, as in NumPy. An ``Array`` is lazy and is not iterated."""
    if isinstance(arrays, (Array, str, bytes)):
        raise TypeError(f"{name} takes a sequence of arrays, got {type(arrays).__name__}")
    items = [Array(item) for item in arrays]
    if not items:
        raise ValueError(f"need at least one array to {name}")
    return items


def _join_channels(name: str, images: Sequence["Array"], channel_axis: int) -> "Array":
    """One ``fill`` of the 3-axis result, with each image written at
    ``(0, 0)`` into the channels after the images before it.

    ``channel_axis`` is the result's channel axis. A ``(N,)`` or ``(H, W)``
    image is one channel; a 3-axis image has the result's layout. The
    result is that fill: origin ``(0, 0)`` and a canvas the size of the
    result. The dtype is NumPy's promotion, and an image of another dtype
    is cast with ``astype``.
    """
    height, width = images[0].meta.height, images[0].meta.width
    for image in images[1:]:
        if image.meta.height != height or image.meta.width != width:
            shapes = [tuple(image.shape) for image in images]
            raise ValueError(
                f"{name} inputs must have the same height and width, got shapes {shapes}"
            )
    dtype = supported_dtype(np.result_type(*(image.dtype for image in images)))
    placed = [image.astype(dtype) for image in images]
    shape = [height, width]
    shape.insert(channel_axis, sum(image.meta.channels for image in placed))
    base = _emit_fill(_meta_from_shape(tuple(shape), dtype, channel_axis), 0)
    start = 0
    placements = []
    for image in placed:
        count = image.meta.channels
        placements.append(((0, 0), image, list(range(start, start + count))))
        start += count
    return composite(base, *placements)


def _join_rows_or_columns(name: str, images: Sequence["Array"], axis: int) -> "Array":
    """One ``fill`` of the result, with each image written after the images
    before it along ``axis``, the rows or the columns axis of every image.

    The images have one layout and match on every other axis, as in NumPy.
    A 1D image's one axis is its columns. Dtype, origin, and canvas are as
    in ``_join_channels``.
    """
    first = images[0].meta
    for image in images[1:]:
        if any(
            size != first_size
            for index, (size, first_size) in enumerate(zip(image.shape, first.shape))
            if index != axis
        ):
            shapes = [tuple(image.shape) for image in images]
            raise ValueError(
                f"{name} inputs must match on every axis but axis {axis}, got shapes {shapes}"
            )
    along_rows = not first.is_1d and axis == first.spatial_axes[0]
    dtype = supported_dtype(np.result_type(*(image.dtype for image in images)))
    placed = [image.astype(dtype) for image in images]
    shape = list(first.shape)
    shape[axis] = sum(image.shape[axis] for image in placed)
    base = _emit_fill(_meta_from_shape(tuple(shape), dtype, first.channel_axis), 0)
    offset = 0
    placements = []
    for image in placed:
        if first.is_1d:
            origin: Tuple[int, ...] = (offset,)
        elif along_rows:
            origin = (offset, 0)
        else:
            origin = (0, offset)
        placements.append((origin, image))
        offset += image.shape[axis]
    return composite(base, *placements)


def concatenate(arrays: Sequence[ArrayLike], axis: int = 0) -> "Array":
    """Join arrays along an existing axis, as ``np.concatenate``.

    The inputs have the same number of axes and match on every axis but
    ``axis``, which may be the channel axis, the rows, or the columns. A
    3-axis input has the same channel axis as the others. The result is
    its own array at origin ``(0, 0)``, with a canvas the size of the
    result.

    The dtype is NumPy's promotion. uint8 ``255`` joined with float32 is
    ``255.0``.
    """
    images = _image_sequence("concatenate", arrays)
    shapes = [tuple(image.shape) for image in images]
    if len({image.ndim for image in images}) != 1:
        raise ValueError(
            f"concatenate inputs must have the same number of axes, got shapes {shapes}"
        )
    ndim = images[0].ndim
    if ndim == 4:
        raise NotImplementedError(
            f"concatenate of a 4-axis stack is not supported (shape {images[0].shape})"
        )
    if len({image.meta.channel_axis for image in images}) != 1:
        raise NotImplementedError(
            f"concatenate of arrays with their channels on different axes is not "
            f"supported, got shapes {shapes}; give them one layout with mi.moveaxis"
        )
    resolved = normalize_axis_index(axis, ndim, "concatenate axis")
    if resolved == images[0].meta.channel_axis:
        return _join_channels("concatenate", images, resolved)
    return _join_rows_or_columns("concatenate", images, resolved)


def stack(arrays: Sequence[ArrayLike], axis: int = 0) -> "Array":
    """Stack 1D or ``(H, W)`` arrays along a new axis, as ``np.stack``.

    ``K`` arrays of shape ``(N,)`` give ``(K, N)`` for ``axis`` ``0``,
    NumPy's default, and ``(N, K)`` for ``axis`` ``1``. ``K`` arrays of
    shape ``(H, W)`` give ``(K, H, W)`` for ``axis`` ``0`` and
    ``(H, W, K)`` for ``axis`` ``-1``. The inputs share a shape. Dtype,
    origin, and canvas match ``concatenate``.
    """
    images = _image_sequence("stack", arrays)
    if len({tuple(image.shape) for image in images}) != 1:
        shapes = [tuple(image.shape) for image in images]
        raise ValueError(f"stack inputs must have the same shape, got shapes {shapes}")
    ndim = images[0].ndim
    if ndim == 1:
        resolved = normalize_axis_index(axis, 2, "stack axis")
        new_shape = (1, -1) if resolved == 0 else (-1, 1)
        return _join_rows_or_columns(
            "stack", [image.reshape(new_shape) for image in images], resolved
        )
    if ndim != 2:
        raise NotImplementedError(
            f"stack of shape {images[0].shape} is not supported; only 1D and (H, W) "
            "arrays can be stacked"
        )
    resolved = normalize_axis_index(axis, 3, "stack axis")
    return _join_channels("stack", images, resolved)


def vstack(arrays: Sequence[ArrayLike]) -> "Array":
    """Join arrays along axis 0, as ``np.vstack``. A ``(N,)`` input is the
    row ``(1, N)``. Dtype, origin, and canvas match ``concatenate``."""
    images = _image_sequence("vstack", arrays)
    rows = [image.reshape(1, -1) if image.ndim == 1 else image for image in images]
    return concatenate(rows, axis=0)


def hstack(arrays: Sequence[ArrayLike]) -> "Array":
    """Join arrays along axis 1, as ``np.hstack``, or along axis 0 when the
    first input is 1D. Dtype, origin, and canvas match ``concatenate``."""
    images = _image_sequence("hstack", arrays)
    return concatenate(images, axis=0 if images[0].ndim == 1 else 1)


def dstack(arrays: Sequence[ArrayLike]) -> "Array":
    """Join arrays along axis 2, as ``np.dstack``.

    A ``(N,)`` input is one channel of a one-row image and a ``(H, W)``
    input is one channel. When every input has 3 axes, this is
    ``concatenate`` along axis 2: the channels of ``(H, W, C)`` inputs,
    the columns of ``(C, H, W)`` inputs. Dtype, origin, and canvas match
    ``concatenate``.
    """
    images = _image_sequence("dstack", arrays)
    if all(image.ndim == 3 for image in images):
        return concatenate(images, axis=2)
    for image in images:
        if image.ndim >= 3 and image.meta.channel_axis != 2:
            raise NotImplementedError(
                f"dstack of shape {image.shape} with channel_axis "
                f"{image.meta.channel_axis} and a 1D or 2D input is not supported; "
                "a 1D or 2D input is joined as one channel, so put the channels "
                "last with mi.moveaxis"
            )
    return _join_channels("dstack", images, 2)


class Array:
    """Lazy array handle: a source buffer and/or an engine op result.

    Do not mutate an array after wrapping it. Ingest seals the array and its
    ndarray ``.base`` root. ``realize()`` caches pixels on this handle.
    """

    __slots__ = ("_meta", "_data", "_node")

    # Above ndarray's, so ``ndarray + Array`` calls ``Array.__radd__`` and
    # stays lazy instead of realizing this array.
    __array_priority__ = 1000

    def __new__(
        cls,
        data: Optional[np.ndarray | Array] = None,
        *,
        origin: Optional[Tuple[int, int]] = None,
        channel_axis: Optional[int] = -1,
        _meta: Optional[ArrayMeta] = None,
        _node: Optional["OpNode"] = None,
    ):
        if isinstance(data, Array):
            if origin is not None:
                raise ValueError("origin= is only valid when ingesting a source buffer")
            if channel_axis != -1:
                raise ValueError("channel_axis= is only valid when ingesting a source buffer")
            if _meta is not None or _node is not None:
                raise ValueError("Array(Array) cannot take _meta= or _node=")
            return data
        return super().__new__(cls)

    def __init__(
        self,
        data: Optional[np.ndarray | Array] = None,
        *,
        origin: Optional[Tuple[int, int]] = None,
        channel_axis: Optional[int] = -1,
        _meta: Optional[ArrayMeta] = None,
        _node: Optional["OpNode"] = None,
    ):
        """``channel_axis`` is the axis of a 3- or 4-axis ``data`` that
        holds the channels. Axis 0 of a 4-axis ``data`` holds its frames;
        the other two axes are the rows and the columns, in that order. It
        is ignored for fewer axes.
        """
        if isinstance(data, Array):
            return
        if data is not None:
            if _node is not None:
                raise ValueError("source Array cannot also have an op node")
            arr = np.asarray(data)
            meta = meta_from_array(arr, channel_axis)
            repeated = _repeated_axes(arr)
            if repeated is not None:
                core, reps = repeated
                stretched = tile(
                    Array(core, origin=origin, channel_axis=meta.channel_axis), reps
                )
                self._take_fields(stretched, stretched._meta)
                return
            if origin is not None:
                row, col = int(origin[0]), int(origin[1])
                meta = meta.copy(
                    origin=(row, col),
                    canvas=(col, row, meta.width, meta.height),
                )
            # A 1D (N,) ndarray is bound as one row. Adding a leading axis of
            # length 1 is always a view, so this never copies. Other arrays
            # are not reshaped, because sealing must mark the caller's array.
            if meta.is_1d:
                arr = arr.reshape(meta.buffer_shape)
            from .engines.core.binding import bound_source, is_bindable

            if not is_bindable(arr, meta.channel_axis):
                source = bound_source(arr, meta.channel_axis)
                if source is not None:
                    _seal_ndarray(arr)
                    self._take_fields(source, meta)
                    return
                # "A" copies a misaligned array too. A one-row array can be
                # C-contiguous and still start on a byte that is not a
                # multiple of its item size.
                arr = np.require(arr, requirements=["C", "A"])
            self._meta = meta
            self._data = _seal_ndarray(arr)
            self._node = None
        elif _meta is not None:
            if origin is not None:
                raise ValueError("origin= is only valid for source Arrays")
            self._meta = _meta
            self._data = None
            self._node = _node
        else:
            raise ValueError("Array requires an ndarray source or _meta=")

    def _take_fields(self, other: "Array", meta: ArrayMeta) -> None:
        """Make this handle ``other``'s buffer and node, presented with ``meta``.

        Every place that builds one handle from another goes through here,
        so a new slot is copied in one place.
        """
        self._meta = meta
        self._data = other._data
        self._node = other._node

    @property
    def dtype(self) -> np.dtype:
        return self._meta.dtype

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._meta.shape

    @property
    def ndim(self) -> int:
        return len(self._meta.shape)

    @property
    def meta(self) -> ArrayMeta:
        return self._meta

    def _scalar_op(self, name: str, other: ArithmeticOperand, reverse: bool = False) -> "Array":
        """``self`` and ``other`` with the scalar op ``name``, or the op
        between two arrays when ``other`` is an Array or a constant that is
        not one value or one per channel. ``reverse`` puts ``other`` first.

        A constant keeps an integer array's dtype when its values are
        integers. A float constant, and ``/``, make an integer array
        float32. Integer results saturate instead of wrapping.
        """
        from .engines.graph import op

        if not isinstance(other, Array):
            const = np.asarray(other)
            if const.dtype.kind not in "biuf":
                raise TypeError(
                    f"{name}: expected a number, an array of numbers, or an Array, "
                    f"got {other!r}"
                )
            dtype = self.dtype
            if _is_integer(dtype) and (const.dtype.kind == "f" or name == "div_scalar"):
                dtype = np.dtype(np.float32)
            scalar = _scalar_values(self._meta, const)
            if scalar is not None:
                values, per_frame = scalar
                attrs: dict[str, Any] = {"value": values}
                if per_frame:
                    attrs["per_frame"] = True
                if reverse:
                    attrs["reverse"] = True
                return op(name, self.astype(dtype), **attrs)
            channel_axis = self.meta.channel_axis if const.ndim == self.meta.ndim else None
            other = Array(
                np.ascontiguousarray(const, dtype=dtype),
                channel_axis=channel_axis,
            )
        first, second = (other, self) if reverse else (self, other)
        return _elementwise(_ELEMENTWISE_OPS[name], first, second)

    def __add__(self, other: ArithmeticOperand) -> "Array":
        return self._scalar_op("add_scalar", other)

    def __radd__(self, other: ArithmeticOperand) -> "Array":
        return self._scalar_op("add_scalar", other)

    def __sub__(self, other: ArithmeticOperand) -> "Array":
        return self._scalar_op("sub_scalar", other)

    def __rsub__(self, other: ArithmeticOperand) -> "Array":
        return self._scalar_op("sub_scalar", other, reverse=True)

    def __mul__(self, other: ArithmeticOperand) -> "Array":
        return self._scalar_op("mul_scalar", other)

    def __rmul__(self, other: ArithmeticOperand) -> "Array":
        return self._scalar_op("mul_scalar", other)

    def __truediv__(self, other: ArithmeticOperand) -> "Array":
        return self._scalar_op("div_scalar", other)

    def __rtruediv__(self, other: ArithmeticOperand) -> "Array":
        return self._scalar_op("div_scalar", other, reverse=True)

    def __matmul__(self, other: ArrayLike | Sequence[Any]) -> "Array":
        return matmul(self, other)

    def __rmatmul__(self, other: ArrayLike | Sequence[Any]) -> "Array":
        return matmul(other, self)

    def view(
        self,
        region: slice | tuple[Any, ...] | None = None,
        *,
        left: int | None = None,
        top: int | None = None,
        width: int | None = None,
        height: int | None = None,
        oob_valid: bool = True,
        reset_origin: bool = False,
    ) -> "Array":
        """Window into this array.

        ``oob_valid`` true keeps the parent canvas in this coordinate system.
        A ``None`` in ``region`` adds an axis of length 1, as in NumPy.
        """
        if region is None:
            if self._meta.is_1d:
                raise TypeError(
                    "a 1D array is viewed with a slice, not left, top, width, height"
                )
            window = _window_from_rect(left, top, width, height)
            return self._view_window(
                window, oob_valid=oob_valid, reset_origin=reset_origin
            )
        if any(v is not None for v in (left, top, width, height)):
            raise TypeError("cannot mix a slice region with left, top, width, height")
        axes, new_axes = _expand_index_key(region, self._meta)
        if self._meta.is_1d:
            row = _reshape_layout(self, (1, self._meta.width))
            window = _window_from_slices(row._meta, slice(None), axes[0])
            viewed = row._view_window(
                window, oob_valid=oob_valid, reset_origin=reset_origin
            )
            shape = [viewed.meta.width]
        else:
            if self._meta.channel_axis is None:
                window = _window_from_slices(self._meta, *axes)
            else:
                rows_axis, cols_axis = self._meta.spatial_axes
                window = _window_from_slices(
                    self._meta,
                    axes[rows_axis],
                    axes[cols_axis],
                    axes[self._meta.channel_axis],
                    axes[0] if self._meta.ndim == 4 else None,
                )
            viewed = self._view_window(
                window, oob_valid=oob_valid, reset_origin=reset_origin
            )
            shape = list(viewed.shape)
        for position in new_axes:
            shape.insert(position, 1)
        return _reshape_layout(viewed, tuple(shape))

    def _view_window(
        self, window: _Window, *, oob_valid: bool, reset_origin: bool
    ) -> "Array":
        """The ``view`` op for ``window``, or this array when it is all of it."""
        import muimage as mi

        if window.is_full(self._meta):
            return _without_dropped_axes(self, window)

        attrs: dict[str, Any] = {
            "left": window.left,
            "top": window.top,
            "width": window.width,
            "height": window.height,
            "oob_valid": oob_valid,
            "reset_origin": reset_origin,
        }
        if window.src_channels is not None:
            attrs["src_channels"] = window.src_channels
        if window.src_frames is not None:
            attrs["src_frames"] = window.src_frames
        if window.row_step != 1:
            attrs["row_step"] = window.row_step
        if window.col_step != 1:
            attrs["col_step"] = window.col_step

        return _without_dropped_axes(mi.view(self, **attrs), window)

    def crop(
        self,
        region: slice | tuple[Any, ...] | None = None,
        *,
        left: int | None = None,
        top: int | None = None,
        width: int | None = None,
        height: int | None = None,
        reset_origin: bool = False,
    ) -> "Array":
        """Hard window: same as ``view(..., oob_valid=False)``."""
        return self.view(
            region,
            left=left,
            top=top,
            width=width,
            height=height,
            oob_valid=False,
            reset_origin=reset_origin,
        )

    def pad(
        self,
        pad_width: Any,
        mode: str = "constant",
        constant_values: Any = 0,
    ) -> "Array":
        """Grow the array. ``pad_width`` and ``constant_values`` follow ``numpy.pad``.

        As in NumPy, an int or a single ``(before, after)`` pair pads every
        axis, channels included; ``((0, 0), (1, 1), (1, 1))`` pads only the
        rows and columns of a CHW array. The pair on the channel axis adds
        channels. On a CHW array that pad works in any ``mode``; with
        the channels on another axis it requires ``mode="constant"``. The
        pair on axis 0 of a 4-axis stack adds frames, in any ``mode``.
        """
        import muimage as mi

        if self._meta.is_1d:
            left, right = (
                int(v) for v in _expand_pad(pad_width, "pad_width", 1, nonneg=True)
            )
            consts = [float(v) for v in _expand_pad(constant_values, "constant_values", 1)]
            return mi.pad(
                self,
                top=0,
                bottom=0,
                left=left,
                right=right,
                mode=mode,
                constant_values=[0.0, 0.0] + consts,
            )
        # The engine takes the sides as rows, columns, then channels.
        if self._meta.channel_axis is None:
            axis_order: Tuple[int, ...] = (0, 1)
        else:
            rows_axis, cols_axis = self._meta.spatial_axes
            axis_order = (rows_axis, cols_axis, self._meta.channel_axis)

        def by_engine_side(values: list[Any]) -> list[Any]:
            return [values[2 * axis + side] for axis in axis_order for side in (0, 1)]

        ndim = self._meta.ndim
        all_sides = _expand_pad(pad_width, "pad_width", ndim, nonneg=True)
        all_consts = _expand_pad(constant_values, "constant_values", ndim)
        sides = [int(v) for v in by_engine_side(all_sides)]
        consts = [float(v) for v in by_engine_side(all_consts)]
        attrs: dict[str, Any] = dict(zip(("top", "bottom", "left", "right"), sides))
        if ndim >= 3:
            attrs["channel_before"], attrs["channel_after"] = sides[4:]
        if ndim == 4 and any(all_sides[:2]):
            attrs["frame_before"], attrs["frame_after"] = (int(v) for v in all_sides[:2])
            consts += [float(v) for v in all_consts[:2]]
        return mi.pad(self, mode=mode, constant_values=consts, **attrs)

    def composite(self, *placements: tuple) -> "Array":
        """``composite(self, *placements)``."""
        return composite(self, *placements)

    def __getitem__(self, key: Any) -> "Array":
        """NumPy spatial slice: a hard crop of this array."""
        if key is None:
            # crop reads region=None as "no region given".
            key = (None,)
        return self.crop(key)

    def transpose(self, *axes: Any) -> "Array":
        """Permute the axes, as ``ndarray.transpose``: ``a.transpose(1, 0, 2)``
        or ``a.transpose((1, 0, 2))``.

        With no axes, every axis is reversed, so ``(H, W, C)`` becomes
        ``(C, W, H)``. A 1D array is returned unchanged, as in NumPy.
        """
        if len(axes) == 1 and (axes[0] is None or not isinstance(axes[0], (int, np.integer))):
            axes = () if axes[0] is None else tuple(axes[0])
        return transpose(self, axes or None)

    def reshape(self, *shape: Any, order: str = "C") -> "Array":
        """The same samples as a new shape, as ``ndarray.reshape``:
        ``a.reshape(2, 4, 3)`` or ``a.reshape((2, 4, 3))``."""
        return reshape(self, shape[0] if len(shape) == 1 else shape, order=order)

    def swapaxes(self, axis1: int, axis2: int) -> "Array":
        """Swap two axes, as ``ndarray.swapaxes``."""
        return swapaxes(self, axis1, axis2)

    def squeeze(self, axis: Any = None) -> "Array":
        """Remove axes of length 1, as ``ndarray.squeeze``."""
        return squeeze(self, axis)

    @property
    def T(self) -> "Array":
        """Every axis reversed. Same as ``transpose()``."""
        return self.transpose()

    def astype(self, dtype: DTypeLike) -> "Array":
        """Numeric cast. ``255`` uint8 becomes ``255.0`` float32."""
        import muimage as mi

        dest = supported_dtype(dtype)
        if dest is self.dtype:
            return self
        return mi.cast_dtype(self, dest_dtype=dest.name)

    def convert_type(
        self,
        dtype: DTypeLike,
        src_bits: int | None = None,
        dst_bits: int | None = None,
        clip_max: float | None = None,
    ) -> "Array":
        """Pixel-range convert. ``255`` uint8, ``65535`` uint16, and ``1.0`` float16 become ``1.0`` float32.

        If the values are already float but not yet in that 0..1 range, pass
        ``src_bits`` / ``dst_bits`` to name the integer range they still use.
        """
        import muimage as mi

        dest = supported_dtype(dtype)
        if (
            dest is self.dtype
            and src_bits is None
            and dst_bits is None
            and clip_max is None
        ):
            return self
        attrs: dict[str, Any] = {"dest_dtype": dest.name}
        if src_bits is not None:
            attrs["src_bits"] = int(src_bits)
        if dst_bits is not None:
            attrs["dst_bits"] = int(dst_bits)
        if clip_max is not None:
            attrs["clip_max"] = float(clip_max)
        return mi.convert_dtype(self, **attrs)

    def realize(
        self,
        *,
        force_recompute: bool = False,
        out: np.ndarray | None = None,
    ) -> np.ndarray:
        """Run the graph if needed and return this array's pixels.

        Without ``out``, the returned array is read-only and kept on this
        handle. With ``out``, the pixels are written into that array and
        ``out`` is returned. ``out`` must have this array's shape and be
        writeable, and it cannot share memory with a buffer the graph
        reads. A different dtype is ``self.astype(out.dtype)``, which must
        be a ``same_kind`` cast. The engine writes ``out`` when it can bind
        it, or the buffer under it when ``out`` flips or permutes that
        buffer's axes. Any other ``out``, such as ``[:, ::-2]``, is filled
        by a copy.

        An array that several row-by-row ops read is computed row by row as
        they need it, and computed again where its reads are far apart (a
        flip of it, a shifted read). Realizing it first keeps one computed
        copy.
        """
        from .engines.graph import realize

        return realize(self, force_recompute=force_recompute, out=out)

    def __array__(self, dtype: Any = None) -> np.ndarray:
        """NumPy array protocol: getting the array materializes the graph."""
        arr = self.realize()
        if dtype is None:
            return arr
        return np.asarray(arr, dtype=dtype)
