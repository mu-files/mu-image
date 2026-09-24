"""tile repeats an array the way numpy.tile does for image rank."""

from __future__ import annotations

import numpy as np
import pytest

import muimage as mi
from muraw.array import Array
from muraw.engines.graph import op


def test_tile_rows_and_columns():
    src = np.arange(6, dtype=np.float32).reshape(2, 3)
    out = mi.tile(Array(src), (2, 2)).realize()
    np.testing.assert_array_equal(out, np.tile(src, (2, 2)))
    assert out.shape == (4, 6)


def test_tile_bare_int_repeats_last_axis():
    src = np.arange(6, dtype=np.float32).reshape(2, 3)
    out = mi.tile(Array(src), 2).realize()
    np.testing.assert_array_equal(out, np.tile(src, 2))
    assert out.shape == (2, 6)


def test_tile_rgb_short_reps_aligns_right():
    src = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    out = mi.tile(Array(src), (4, 1)).realize()
    np.testing.assert_array_equal(out, np.tile(src, (4, 1)))
    assert out.shape == (2, 8, 3)


def test_tile_rgb_repeats_channels_only():
    src = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    out = mi.tile(Array(src), 2).realize()
    np.testing.assert_array_equal(out, np.tile(src, 2))
    assert out.shape == (2, 2, 6)


def test_tile_spatial_rgb():
    src = np.arange(12, dtype=np.float32).reshape(2, 2, 3)
    out = mi.tile(Array(src), (2, 3, 1)).realize()
    np.testing.assert_array_equal(out, np.tile(src, (2, 3, 1)))
    assert out.shape == (4, 6, 3)


def test_tile_uint8():
    src = np.arange(6, dtype=np.uint8).reshape(2, 3)
    out = mi.tile(Array(src), (2, 1)).realize()
    np.testing.assert_array_equal(out, np.tile(src, (2, 1)))


def test_tile_keeps_origin():
    src = Array(np.zeros((2, 3), dtype=np.float32), origin=(4, 5))
    tiled = mi.tile(src, (2, 2))
    assert tiled.meta.origin == (4, 5)
    assert tiled.meta.shape == (4, 6)
    assert tiled.meta.canvas == (5, 4, 6, 4)


def test_tile_rejects_extra_axis():
    src = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="axis"):
        mi.tile(Array(src), (2, 1, 1))


@pytest.mark.parametrize("dtype", [np.uint16, np.float16])
def test_tile_two_byte_dtypes(dtype):
    src = np.arange(24).astype(dtype).reshape(2, 4, 3)
    out = mi.tile(Array(src), (2, 3, 2)).realize()
    np.testing.assert_array_equal(out, np.tile(src, (2, 3, 2)))


@pytest.mark.parametrize("reps", [[2, 1], np.array([2, 3]), (np.int64(3),)])
def test_tile_reps_sequences(reps):
    src = np.arange(6, dtype=np.float32).reshape(2, 3)
    np.testing.assert_array_equal(mi.tile(Array(src), reps).realize(), np.tile(src, reps))


def test_tile_accepts_ndarray():
    src = np.arange(6, dtype=np.float32).reshape(2, 3)
    np.testing.assert_array_equal(mi.tile(src, 2).realize(), np.tile(src, 2))


def test_tile_in_lazy_chain():
    src = np.random.default_rng(0).random((40, 600), dtype=np.float32)
    out = (mi.tile(Array(src) * 2, (2, 3)) * 0.5).realize()
    np.testing.assert_allclose(out, np.tile(src, (2, 3)), rtol=1e-6)


def test_tile_between_span_op_and_orientation():
    """The chain runs in dest rects smaller than the 600-px tiled row."""
    src = np.random.default_rng(0).random((40, 600), dtype=np.float32)
    tiled = mi.tile(Array(src) * 2, (2, 2))
    out = op("orientation", tiled, orientation=1).realize()
    np.testing.assert_allclose(out, np.tile(src * 2, (2, 2)), rtol=1e-6)


@pytest.mark.parametrize("reps", [2.5, "2"])
def test_tile_rejects_non_integer_reps(reps):
    src = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(TypeError):
        mi.tile(Array(src), reps)


def test_tile_rejects_zero_reps():
    src = np.zeros((2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match=">= 1"):
        mi.tile(Array(src), (0, 2))
