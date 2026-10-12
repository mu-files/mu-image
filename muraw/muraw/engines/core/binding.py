# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files
"""How the core engine takes a NumPy array as a source without a copy.

``is_bindable`` says whether the engine can bind an ndarray's memory as it
is. ``bound_source`` reads an array the engine cannot bind as an op over a
buffer it can: a view of a packed parent, an orientation of a row/column
swap, or a transpose of a permutation of the axes.
"""

from __future__ import annotations

import itertools
from typing import Any, Optional, Tuple

import numpy as np
from numpy.lib.array_utils import byte_bounds

from ...array import Array, _base_chain, _permute


def is_bindable(arr: np.ndarray, channel_axis: Optional[int] = None) -> bool:
    """True when the engine can bind ``arr`` itself as a source buffer.

    Pixels in a row are adjacent, the data pointer is element-aligned, and
    the row pitch is a whole number of elements that holds one packed row.
    It need not be a whole number of pixels: a padded RGB888 camera buffer
    can have a pitch that is not a multiple of 3 bytes. A negative pitch is a
    flipped view, not a source buffer. The stride of an axis of length 1 is
    ignored, as NumPy and the engine binding both do: ``v[None, :]`` has a
    row stride of 0.

    ``channel_axis`` is as in ``ArrayMeta``. The engine reads a 3-axis
    ``arr`` whose channel axis is 0 as frames, rows, and columns, and a
    4-axis ``arr`` as frames, rows, columns, and channels. Each frame must
    then be a source buffer on its own, and the frame pitch a whole number
    of elements that holds one frame, so frames do not overlap. Every other
    ``arr`` is read packed in axis order.
    """
    itemsize = int(arr.dtype.itemsize)
    if int(arr.ctypes.data) % itemsize != 0:
        return False

    def is_packed_axis(axis: int, step_bytes: int) -> bool:
        return int(arr.shape[axis]) == 1 or int(arr.strides[axis]) == step_bytes

    def pitch_holds(axis: int, span_bytes: int) -> bool:
        stride = int(arr.strides[axis])
        return int(arr.shape[axis]) == 1 or (stride >= span_bytes and stride % itemsize == 0)

    if arr.ndim == 4 or (arr.ndim == 3 and channel_axis == 0):
        pixel = itemsize * (int(arr.shape[3]) if arr.ndim == 4 else 1)
        if arr.ndim == 4 and not is_packed_axis(3, itemsize):
            return False
        if not is_packed_axis(2, pixel):
            return False
        row_bytes = int(arr.shape[2]) * pixel
        if not pitch_holds(1, row_bytes):
            return False
        row_pitch = int(arr.strides[1]) if int(arr.shape[1]) > 1 else row_bytes
        return pitch_holds(0, (int(arr.shape[1]) - 1) * row_pitch + row_bytes)
    if arr.ndim == 2:
        pixel = itemsize
        if not is_packed_axis(1, itemsize):
            return False
    elif arr.ndim == 3:
        channels = int(arr.shape[2])
        pixel = itemsize * channels
        if not (is_packed_axis(2, itemsize) and is_packed_axis(1, pixel)):
            return False
    else:
        return False
    if int(arr.shape[0]) == 1:
        return True
    row_pitch = int(arr.strides[0])
    packed = int(arr.shape[1]) * pixel
    return row_pitch >= packed and row_pitch % itemsize == 0


def bound_source(arr: np.ndarray, channel_axis: Optional[int]) -> Optional[Array]:
    """An Array that reads ``arr``'s memory, or ``None`` to copy ``arr``.

    ``arr`` is 2 or 3 axes, with its channels on ``channel_axis`` as in
    ``ArrayMeta``, and ``is_bindable`` is false for it. An array whose
    channels are last is tried as a view of a packed parent, then as a
    row/column swap; a 3-axis array is then tried as a permutation of its
    axes.
    """
    source = None
    if channel_axis in (None, 2):
        source = _view_over_packed(arr)
        if source is None:
            source = _transposed_source(arr)
    if source is None and arr.ndim == 3:
        source = _permuted_source(arr, channel_axis)
    return source


class _Memory:
    """Presents ``interface`` to NumPy as an array and keeps ``base`` alive.

    ``interface`` is a NumPy array interface that points into memory
    ``base`` reads.
    """

    def __init__(self, base: np.ndarray, interface: dict[str, Any]):
        self.base = base
        self.__array_interface__ = interface


def _view_over_packed(arr: np.ndarray) -> Optional[Array]:
    """An Array that reads ``arr`` as a view of a packed parent, or ``None`` to copy.

    ``arr`` is ``(H, W)`` or ``(H, W, C)``. The parent is built over the same
    memory from ``arr``'s strides, so it works whatever layout ``arr`` was
    sliced from. Each parent pixel is ``pixel_size`` consecutive elements
    and holds every channel ``arr`` reads at one position. Parent rows are
    one ``arr`` row stride apart. The view reads every parent row and every
    ``col_step``-th parent pixel, reverses each axis whose stride is
    negative, and picks ``arr``'s channels out of each parent pixel.

    ``None`` when a stride is not a whole number of elements, an axis longer
    than 1 has stride 0, rows overlap, or the parent would reach outside the
    memory ``arr`` belongs to.

    The result is the slice: origin ``(0, 0)`` and a canvas the size of
    ``arr``. Later crops do not reach samples outside that slice.
    """
    itemsize = int(arr.dtype.itemsize)
    if any(int(stride) % itemsize for stride in arr.strides):
        return None
    height, width = int(arr.shape[0]), int(arr.shape[1])
    channels = int(arr.shape[2]) if arr.ndim == 3 else 1
    # An axis of length 1 is never stepped along, so its stride is ignored.
    # 0 below means "any step".
    row_elements = abs(int(arr.strides[0])) // itemsize if height > 1 else 0
    col_elements = abs(int(arr.strides[1])) // itemsize if width > 1 else 0
    channel_elements = abs(int(arr.strides[2])) // itemsize if channels > 1 else 1
    if (
        (height > 1 and row_elements == 0)
        or (width > 1 and col_elements == 0)
        or channel_elements == 0
    ):
        return None

    # The pixel size must divide the column step so the view reads whole
    # parent pixels. The row pitch only has to be a whole number of elements.
    channel_span = (channels - 1) * channel_elements + 1
    if col_elements == 0:
        pixel_size = channel_span
    else:
        pixel_size = next(
            (
                size
                for size in range(channel_span, col_elements + 1)
                if col_elements % size == 0
            ),
            0,
        )
        if pixel_size == 0:
            return None
    col_step = col_elements // pixel_size if col_elements else 1
    parent_width = (width - 1) * col_step + 1
    row_pitch = row_elements if row_elements else parent_width * pixel_size
    if row_pitch < parent_width * pixel_size:
        return None

    # ``lowest`` reads the same samples with every stride non-negative, so its
    # data pointer is the lowest address ``arr`` reads.
    flipped = [int(stride) < 0 for stride in arr.strides]
    lowest = arr[tuple(slice(None, None, -1) if flip else slice(None) for flip in flipped)]
    owner_low, owner_high = byte_bounds(_base_chain(arr)[-1])
    extent = ((height - 1) * row_pitch + parent_width * pixel_size) * itemsize
    # ``first_channel`` is where ``arr``'s first channel sits in a parent
    # pixel. Any position works if the parent stays inside the owner.
    for first_channel in range(pixel_size - channel_span + 1):
        parent_start = int(lowest.ctypes.data) - first_channel * itemsize
        if owner_low <= parent_start and parent_start + extent <= owner_high:
            break
    else:
        return None

    if pixel_size == 1:
        shape: Tuple[int, ...] = (height, parent_width)
        strides: Tuple[int, ...] = (row_pitch * itemsize, itemsize)
    else:
        shape = (height, parent_width, pixel_size)
        strides = (row_pitch * itemsize, pixel_size * itemsize, itemsize)
    parent = np.asarray(
        _Memory(
            arr,
            {
                "version": 3,
                "shape": shape,
                "typestr": arr.dtype.str,
                "data": (parent_start, True),
                "strides": strides,
            },
        )
    )
    if not is_bindable(parent):
        return None

    rows = slice(None, None, -1) if flipped[0] else slice(None)
    cols = slice(None, None, -col_step) if flipped[1] else slice(None, None, col_step)
    if pixel_size == 1:
        key: tuple[Any, ...] = (rows, cols) if arr.ndim == 2 else (rows, cols, None)
    else:
        src_channels = [first_channel + i * channel_elements for i in range(channels)]
        if flipped[2]:
            src_channels.reverse()
        if src_channels == list(range(pixel_size)):
            key = (rows, cols)
        else:
            key = (rows, cols, src_channels)
    return Array(parent).view(key, oob_valid=False, reset_origin=True)


# Orientation code that transposes a source whose (rows, columns) were
# reversed: 5 is a plain transpose, 6 is np.rot90(m, 3), 7 is the
# transverse, and 8 is np.rot90(m).
_TRANSPOSE_ORIENTATION = {
    (False, False): 5,
    (True, False): 6,
    (True, True): 7,
    (False, True): 8,
}


def _transposed_source(arr: np.ndarray) -> Optional[Array]:
    """``arr`` read as an orientation of its row/column swap, or ``None`` to copy.

    ``img.T``, ``img.transpose(1, 0, 2)``, ``np.rot90(img)`` and Fortran-order
    arrays swap the row and column strides. ``arr.swapaxes(0, 1)`` with its
    row and column flips undone is read as a direct source or a view, and
    one orientation op transposes it back and applies those flips, so
    ``Array(np.rot90(m))`` builds the same graph as ``mi.rot90(Array(m))``.
    An array with one row or one column is never transposed.
    """
    if int(arr.shape[0]) == 1 or int(arr.shape[1]) == 1:
        return None
    swapped = arr.swapaxes(0, 1)
    flipped = (int(swapped.strides[0]) < 0, int(swapped.strides[1]) < 0)
    unflipped = swapped[
        slice(None, None, -1) if flipped[0] else slice(None),
        slice(None, None, -1) if flipped[1] else slice(None),
    ]
    if is_bindable(unflipped):
        inner: Optional[Array] = Array(unflipped)
    else:
        inner = _view_over_packed(unflipped)
    if inner is None:
        return None

    import muimage as mi
    return mi.orientation(inner, orientation=_TRANSPOSE_ORIENTATION[flipped])


def _permuted_source(arr: np.ndarray, channel_axis: int) -> Optional[Array]:
    """``arr`` read as a permutation of a source buffer, or ``None`` to copy.

    ``np.moveaxis(chw, 0, -1)`` and a Fortran-order ``(H, W, C)`` array
    store their axes in a different order from their shape. The first
    permutation of ``arr``'s 3 axes that the engine can bind is bound, and
    one ``transpose`` puts the axes back in ``arr``'s order. The bound
    array's channel axis is ``arr``'s axis ``channel_axis``. The engine
    reads no channels on axis 1, so a permutation that puts them there is
    not bound.
    """
    for order in itertools.permutations(range(3)):
        bound = arr.transpose(order)
        bound_channel_axis = order.index(channel_axis)
        if bound_channel_axis == 1:
            continue
        if order != (0, 1, 2) and is_bindable(bound, bound_channel_axis):
            inverse = tuple(order.index(axis) for axis in range(3))
            return _permute(Array(bound, channel_axis=bound_channel_axis), inverse)
    return None
