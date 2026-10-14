"""repeat: ``np.repeat`` on every axis.

One count on the rows or the columns is the ``tile`` op with
``row_repeats`` or ``col_repeats``, and a count per row or column is the
``take`` op. The channels and the frames of a stack are a ``view`` that
reads each index its count of times.

Value checks call the same function with ``mi`` and with ``np``.
"""

from __future__ import annotations

import numpy as np
import pytest

import muimage as mi
from muraw.array import Array


def _image(shape, dtype):
    rng = np.random.default_rng(11)
    if np.dtype(dtype).kind == "f":
        return rng.random(shape).astype(dtype)
    return rng.integers(0, 256, shape).astype(dtype)


def _repeat(lib, x, repeats, axis):
    """Each element of ``x`` repeated ``repeats`` times along ``axis``."""
    return lib.repeat(x, repeats, axis=axis)


def _repeat_flattened(lib, x):
    """Every sample of ``x``, flattened, twice."""
    return lib.repeat(x, 2)


def _doubled_and_repeated(lib, x):
    """``x`` times 2, each column three times, then each row twice."""
    return (x * np.float32(2.0)).repeat(3, axis=1).repeat(2, axis=0)


def _yuv444(lib, luma, chroma):
    """Full-resolution YUV from ``(H, W)`` luma and ``(H / 2, W / 2, 2)``
    chroma: nearest 4:2:0 upsampling, each chroma sample covering 2 x 2
    luma samples."""
    full = lib.repeat(lib.repeat(chroma, 2, axis=0), 2, axis=1)
    return lib.concatenate([luma[..., None], full], axis=-1)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize(
    "shape, axis, repeats",
    [
        ((5, 7), 0, 2),
        ((5, 7), 1, 3),
        ((5, 7), -1, 1),
        ((4, 6, 3), 0, 3),
        ((4, 6, 3), 1, 2),
        ((4, 6, 3), 2, 2),
        ((4, 6, 3), 2, [1, 0, 3]),
        ((2, 4, 6, 3), 0, 2),
        ((2, 4, 6, 3), 0, [0, 3]),
        ((2, 4, 6, 3), 1, 2),
        ((2, 4, 6, 3), 2, 3),
        ((2, 4, 6, 3), 3, [2, 1, 1]),
        ((9,), 0, 4),
        ((3, 4), 0, [2, 2, 2]),
        ((3, 4), 1, 2.5),
    ],
)
def test_repeat_matches_numpy(shape, axis, repeats, dtype):
    x = _image(shape, dtype)
    got = _repeat(mi, mi.Array(x), repeats, axis)
    np.testing.assert_array_equal(got.realize(), _repeat(np, x, repeats, axis))


@pytest.mark.parametrize("axis, repeats", [(0, 2), (0, [2, 0, 1]), (1, 3), (2, 2)])
def test_repeat_on_chw_matches_numpy(axis, repeats):
    x = _image((3, 4, 5), np.float32)
    got = _repeat(mi, Array(x, channel_axis=0), repeats, axis)
    assert got.meta.channel_axis == 0
    np.testing.assert_array_equal(got.realize(), _repeat(np, x, repeats, axis))


def test_repeat_with_no_axis_flattens():
    x = _image((3, 4, 2), np.uint8)
    got = _repeat_flattened(mi, mi.Array(x))
    np.testing.assert_array_equal(got.realize(), _repeat_flattened(np, x))


def test_repeat_method_on_a_lazy_input():
    x = _image((40, 300, 3), np.float32)
    got = _doubled_and_repeated(mi, mi.Array(x))
    np.testing.assert_allclose(got.realize(), _doubled_and_repeated(np, x), rtol=1e-6)


def test_repeat_upsamples_420_chroma_to_full_resolution():
    luma = _image((8, 12), np.uint8)
    chroma = _image((4, 6, 2), np.uint8)
    got = _yuv444(mi, mi.Array(luma), mi.Array(chroma))
    np.testing.assert_array_equal(got.realize(), _yuv444(np, luma, chroma))


@pytest.mark.parametrize(
    "repeats, axis, error",
    [
        (-1, 0, ValueError),
        ([1, 2], 2, ValueError),
        (1, 3, np.exceptions.AxisError),
    ],
)
def test_repeat_rejects_what_numpy_rejects(repeats, axis, error):
    x = _image((2, 2, 3), np.float32)
    with pytest.raises(error):
        _repeat(np, x, repeats, axis)
    with pytest.raises(error):
        _repeat(mi, mi.Array(x), repeats, axis)


@pytest.mark.parametrize("repeats, axis", [(0, 0), ([0, 0, 0], 2)])
def test_repeat_that_leaves_an_axis_empty_raises(repeats, axis):
    with pytest.raises(NotImplementedError, match="empty"):
        _repeat(mi, mi.Array(_image((2, 2, 3), np.float32)), repeats, axis)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize(
    "shape, axis, repeats",
    [
        ((3, 4), 0, [1, 0, 3]),
        ((3, 4), 1, [2, 1, 0, 1]),
        ((3, 4, 3), 1, [0, 2, 2, 1]),
        ((2, 3, 4, 3), 1, [3, 1, 1]),
        ((5,), 0, [1, 2, 0, 1, 3]),
    ],
)
def test_repeat_with_a_count_per_row_or_column_matches_numpy(shape, axis, repeats, dtype):
    x = _image(shape, dtype)
    got = _repeat(mi, mi.Array(x), repeats, axis)
    np.testing.assert_array_equal(got.realize(), _repeat(np, x, repeats, axis))


def test_repeat_with_a_count_per_column_on_chw_matches_numpy():
    x = _image((3, 4, 5), np.float32)
    got = _repeat(mi, Array(x, channel_axis=0), [1, 2, 0, 1, 3], 2)
    np.testing.assert_array_equal(got.realize(), _repeat(np, x, [1, 2, 0, 1, 3], 2))
