"""take: ``np.take`` and index-array indexing on every axis.

The rows and the columns are the ``take`` op; the channels and the
frames of a stack are a ``view`` that reads each index.

Value checks call the same function with ``mi`` and with ``np``.
"""

from __future__ import annotations

import numpy as np
import pytest

import muimage as mi
from muraw.array import Array


def _image(shape, dtype):
    rng = np.random.default_rng(5)
    if np.dtype(dtype).kind == "f":
        return rng.random(shape).astype(dtype)
    return rng.integers(0, 256, shape).astype(dtype)


def _take(lib, x, indices, axis, mode="raise"):
    """``x`` at ``indices`` along ``axis``."""
    return lib.take(x, indices, axis=axis, mode=mode)


def _take_method_after_scale(lib, x):
    """``x`` times 2, then columns 3, 0, 0 and rows 1, 1 with the method."""
    return (x * np.float32(2.0)).take([3, 0, 0], axis=1).take([1, 1], axis=0)


def _index(lib, x, key):
    """``x[key]``."""
    return x[key]


def _rows_then_columns(lib, x, rows, cols):
    """The rows ``rows`` of ``x``, then the columns ``cols`` of those."""
    return x[rows][:, cols]


def _nearest_resize(lib, x, scale):
    """Nearest-neighbour resize of ``(H, W, ...)`` ``x`` by ``scale``."""
    rows = ((np.arange(int(x.shape[0] * scale)) + 0.5) / scale).astype(np.intp)
    cols = ((np.arange(int(x.shape[1] * scale)) + 0.5) / scale).astype(np.intp)
    return x[rows][:, cols]


FORWARD = [0, 1, 1, 3]
REVERSED = [3, 2, 1, 0]
SHUFFLED = [2, 0, 3, 1, 1]
NEGATIVE = [-1, 0, -3]
# np.take truncates float indices.
FLOAT = [2.9, 0.5]


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize("indices", [FORWARD, REVERSED, SHUFFLED, NEGATIVE, FLOAT])
@pytest.mark.parametrize(
    "shape, axis",
    [
        ((5, 7), 0),
        ((5, 7), 1),
        ((4, 6, 4), 0),
        ((4, 6, 4), 1),
        ((4, 6, 4), 2),
        ((4, 4, 6, 3), 0),
        ((4, 4, 6, 3), 1),
        ((4, 4, 6, 3), 2),
        ((9,), 0),
    ],
)
def test_take_matches_numpy(shape, axis, indices, dtype):
    x = _image(shape, dtype)
    got = _take(mi, mi.Array(x), indices, axis)
    np.testing.assert_array_equal(got.realize(), _take(np, x, indices, axis))


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_take_on_chw_matches_numpy(axis):
    x = _image((4, 5, 6), np.float32)
    got = _take(mi, Array(x, channel_axis=0), SHUFFLED, axis)
    np.testing.assert_array_equal(got.realize(), _take(np, x, SHUFFLED, axis))


@pytest.mark.parametrize("mode", ["clip", "wrap"])
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_take_out_of_range_modes_match_numpy(axis, mode):
    x = _image((3, 4, 3), np.float32)
    indices = [-5, 7, 1, -1, 3]
    got = _take(mi, mi.Array(x), indices, axis, mode)
    np.testing.assert_array_equal(got.realize(), _take(np, x, indices, axis, mode))


def test_take_with_no_axis_flattens():
    x = _image((3, 4, 2), np.uint8)
    got = _take(mi, mi.Array(x), [23, 0, 5, 5], None)
    np.testing.assert_array_equal(got.realize(), _take(np, x, [23, 0, 5, 5], None))


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_take_with_one_int_drops_the_axis(axis):
    x = _image((3, 4, 3), np.float32)
    got = _take(mi, mi.Array(x), 2, axis)
    np.testing.assert_array_equal(got.realize(), _take(np, x, 2, axis))


def test_take_method_on_a_lazy_input():
    x = _image((4, 5, 3), np.float32)
    got = _take_method_after_scale(mi, mi.Array(x))
    np.testing.assert_allclose(got.realize(), _take_method_after_scale(np, x), rtol=1e-6)


@pytest.mark.parametrize(
    "indices, axis, mode, error",
    [
        ([0, 3], 0, "raise", IndexError),
        ([-4], 1, "raise", IndexError),
        ([0], 0, "bounce", ValueError),
        ([0], 3, "raise", np.exceptions.AxisError),
    ],
)
def test_take_rejects_what_numpy_rejects(indices, axis, mode, error):
    x = _image((3, 3, 3), np.float32)
    with pytest.raises(error):
        _take(np, x, indices, axis, mode)
    with pytest.raises(error):
        _take(mi, mi.Array(x), indices, axis, mode)


def test_take_with_an_index_array_of_two_axes_raises():
    with pytest.raises(NotImplementedError, match="shape"):
        _take(mi, mi.Array(_image((3, 3), np.float32)), [[0, 1], [1, 2]], 0)


def test_take_of_no_indices_raises():
    with pytest.raises(ValueError, match="empty"):
        _take(mi, mi.Array(_image((3, 3), np.float32)), [], 0)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize(
    "shape, key",
    [
        ((5, 7), [4, 0, 0]),
        ((5, 7), (slice(None), [6, -1, 2])),
        ((5, 7), np.array([1, 3])),
        ((5, 7), np.array([True, False, True, True, False])),
        ((5, 7), (slice(1, 4), np.array([0, 1, 0, 1, 1, 0, 1], dtype=bool))),
        ((4, 6, 3), [3, 1]),
        ((4, 6, 3), (slice(None), [5, 5, 0], slice(0, 2))),
        ((4, 6, 3), (Ellipsis, [2, 0], slice(None))),
        ((2, 4, 6, 3), (slice(None), [3, 0])),
        ((9,), [8, 0, 4]),
        ((9,), np.arange(9) % 2 == 0),
        ((5, 7), 3),
        ((5, 7), (slice(None), -2)),
        ((4, 6, 3), (1, slice(1, 4))),
        ((4, 6, 3), (2, 3)),
        ((4, 6, 3), (slice(None), 4, 0)),
        ((2, 4, 6, 3), (1, 2)),
        ((5, 7), (None, [1, 2])),
        ((4, 6, 3), np.array(2)),
        ((4, 6, 3), (Ellipsis, np.array(1))),
        ((4, 6, 3), (Ellipsis, np.array([True, False, True]))),
        ((2, 4, 6, 3), np.array([False, True])),
    ],
)
def test_index_with_an_index_array_or_int_on_rows_or_columns_matches_numpy(shape, key, dtype):
    x = _image(shape, dtype)
    got = _index(mi, mi.Array(x), key)
    np.testing.assert_array_equal(got.realize(), _index(np, x, key))


@pytest.mark.parametrize("key", [[2, 0], (slice(None), [3, 3, 1]), (slice(None), slice(None), [4, 0])])
def test_index_with_an_index_array_on_chw_matches_numpy(key):
    x = _image((3, 4, 5), np.float32)
    got = _index(mi, Array(x, channel_axis=0), key)
    np.testing.assert_array_equal(got.realize(), _index(np, x, key))


def test_index_rows_then_columns_matches_numpy():
    x = _image((6, 8, 3), np.float32)
    got = _rows_then_columns(mi, mi.Array(x), [5, 0, 2], [7, 7, 1, 0])
    np.testing.assert_array_equal(got.realize(), _rows_then_columns(np, x, [5, 0, 2], [7, 7, 1, 0]))


@pytest.mark.parametrize("shape", [(10, 14), (10, 14, 3)])
def test_nearest_resize_by_one_and_a_half_matches_numpy(shape):
    x = _image(shape, np.uint8)
    got = _nearest_resize(mi, mi.Array(x), 1.5)
    np.testing.assert_array_equal(got.realize(), _nearest_resize(np, x, 1.5))


@pytest.mark.parametrize(
    "key",
    [
        [0, 5],
        (slice(None), [-8]),
        (5, slice(None)),
        np.array([True, False]),
    ],
)
def test_index_rejects_what_numpy_rejects(key):
    x = _image((5, 7), np.float32)
    with pytest.raises(IndexError):
        _index(np, x, key)
    with pytest.raises(IndexError):
        _index(mi, mi.Array(x), key)


@pytest.mark.parametrize(
    "shape, key",
    [
        ((4, 5), ([0, 1], [2, 3])),
        ((4, 5), ([0, 1], 2)),
        ((4, 5, 3), (slice(None), [0, 1], 2)),
        ((4, 5), np.array([[0, 1], [1, 2]])),
    ],
)
def test_index_arrays_numpy_pairs_or_moves_raise(shape, key):
    with pytest.raises(NotImplementedError):
        _index(mi, mi.Array(_image(shape, np.float32)), key)


def test_index_of_every_axis_with_an_int_raises():
    with pytest.raises(NotImplementedError, match="scalar"):
        _index(mi, mi.Array(_image((4, 5), np.float32)), (1, 2))
