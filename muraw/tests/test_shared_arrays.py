"""Arrays read by several ops of one graph, checked against NumPy."""

from __future__ import annotations

import time

import numpy as np
import pytest

import muimage as mi


def _random(shape, seed):
    return np.random.default_rng(seed).random(shape, dtype=np.float32)


def _row_index_and_scale(lib, rows, row):
    """The row index, scaled, and ``row`` scaled.

    ``row`` is an ``Array`` when ``lib`` is muimage, and the source ndarray
    when ``lib`` is NumPy.
    """
    _, y = lib.meshgrid(
        lib.arange(row.shape[1], dtype=np.float32),
        lib.arange(rows, dtype=np.float32),
    )
    return y * np.float32(1.5), row * np.float32(0.5)


def _square_of_difference(lib, rows, row):
    """``(y - c) * (y - c)`` from ``_row_index_and_scale``."""
    y, c = _row_index_and_scale(lib, rows, row)
    return (y - c) * (y - c)


def test_square_of_a_difference_matches_numpy():
    """`y` is each pixel's row index, computed in the graph."""
    row = _random((1, 61), 1)
    got = _square_of_difference(mi, 37, mi.Array(row))
    expected = _square_of_difference(np, 37, row)
    assert got.dtype == np.float32
    assert expected.dtype == np.float32
    np.testing.assert_array_equal(got.realize(), expected)


def _best_time(run, repeats=5):
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        best = min(best, time.perf_counter() - start)
    return best


def test_square_of_a_difference_is_faster_than_numpy():
    """Both time computing `y` from the row index as well as the
    arithmetic. The graph runs as one engine stage, so `y` and `c` stay in
    row buffers; NumPy writes `y` and each intermediate as a whole image."""
    rows = 3000
    row = _random((1, 4000), 3)
    np.testing.assert_array_equal(
        _square_of_difference(mi, rows, mi.Array(row)).realize(),
        _square_of_difference(np, rows, row),
    )
    mi_time = _best_time(
        lambda: _square_of_difference(mi, rows, mi.Array(row)).realize()
    )
    np_time = _best_time(lambda: _square_of_difference(np, rows, row))
    print(f"muimage {mi_time * 1e3:.1f} ms, numpy {np_time * 1e3:.1f} ms")
    assert mi_time < np_time, (
        f"muimage {mi_time * 1e3:.1f} ms, numpy {np_time * 1e3:.1f} ms"
    )


def test_array_read_twice_by_one_op_matches_numpy():
    image = _random((37, 61), 4)
    y = mi.Array(image) - 0.25
    expected = (image - 0.25) + (image - 0.25)
    np.testing.assert_array_equal((y + y).realize(), expected)


def _flip_and_source(lib, image):
    """``flipud(image * 2) + image * 2``."""
    scaled = image * np.float32(2.0)
    return lib.flipud(scaled) + scaled


def test_flip_and_its_source_match_numpy():
    image = _random((37, 61), 5)
    got = _flip_and_source(mi, mi.Array(image))
    expected = _flip_and_source(np, image)
    np.testing.assert_array_equal(got.realize(), expected)


def test_realizing_a_shared_array_first_gives_the_same_pixels():
    image = _random((37, 61), 6)
    computed = mi.Array(image) * 2.0
    out = (mi.flipud(computed) + computed).realize()
    realized = mi.Array(image) * 2.0
    realized.realize()
    np.testing.assert_array_equal((mi.flipud(realized) + realized).realize(), out)


_MATRIX = np.array(
    [[0.9, 0.2, -0.1], [0.05, 1.1, -0.15], [-0.2, 0.1, 1.1]], dtype=np.float32
)


@pytest.mark.parametrize("algorithm", ["bilinear", "ea", "ea_fast"])
def test_demosaic_output_read_by_two_ops_matches_realizing_it_first(algorithm):
    """The demosaic reads its input through an edge pad the engine inserts,
    and its output is read by two ops in the same graph."""
    cfa = _random((38, 54), 7)
    demosaiced = mi.cfa_demosaic(mi.Array(cfa), cfa_pattern="RGGB", algorithm=algorithm)
    got = (mi.rgb_matrix_3x3(demosaiced, matrix=_MATRIX) + demosaiced * 0.1).realize()
    realized = mi.Array(
        mi.cfa_demosaic(
            mi.Array(cfa), cfa_pattern="RGGB", algorithm=algorithm
        ).realize()
    )
    expected = (mi.rgb_matrix_3x3(realized, matrix=_MATRIX) + realized * 0.1).realize()
    np.testing.assert_array_equal(got, expected)


def _rotation_and_source(lib, image):
    """``rot90(image * 2) + image * 2``."""
    scaled = image * np.float32(2.0)
    return lib.rot90(scaled) + scaled


def test_array_read_by_a_rotation_and_an_add_matches_numpy():
    """A rotation is a tiled op, so the array it shares with the add is
    written to a buffer between the two."""
    image = _random((41, 41), 8)
    got = _rotation_and_source(mi, mi.Array(image))
    expected = _rotation_and_source(np, image)
    np.testing.assert_array_equal(got.realize(), expected)


def test_chw_array_read_twice_matches_numpy():
    chw = _random((3, 23, 37), 9)
    y = mi.Array(chw, channel_axis=0) * 2.0 - 0.5
    expected_y = chw * 2.0 - 0.5
    np.testing.assert_array_equal(
        (y * y + y).realize(), expected_y * expected_y + expected_y
    )


def _saturate(values, dtype):
    """Round half up and clamp to ``dtype``'s range, as the integer kernels do."""
    top = np.iinfo(dtype).max
    return np.clip(np.floor(values + np.float32(0.5)), 0, top).astype(dtype)


def _integers(dtype, shape, seed):
    top = np.iinfo(dtype).max
    return np.random.default_rng(seed).integers(0, top + 1, shape).astype(dtype)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_integer_square_of_a_difference_matches_numpy(dtype):
    """Each step saturates, as the integer kernels do."""
    image = _integers(dtype, (37, 61), 10)
    row = _integers(dtype, (1, 61), 11)
    y = mi.Array(image) + 3
    difference = y - mi.Array(row)
    got = (difference * difference).realize()
    expected_y = _saturate(image.astype(np.float32) + 3, dtype)
    expected_difference = _saturate(
        expected_y.astype(np.float32) - row.astype(np.float32), dtype
    )
    wide = expected_difference.astype(np.float32)
    expected = _saturate(wide * wide, dtype)
    assert got.dtype == dtype
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_integer_flip_and_its_source_match_numpy(dtype):
    image = _integers(dtype, (37, 61), 12)
    y = mi.Array(image) + 3
    got = (mi.flipud(y) + y).realize()
    expected_y = _saturate(image.astype(np.float32) + 3, dtype).astype(np.float32)
    expected = _saturate(np.flipud(expected_y) + expected_y, dtype)
    assert got.dtype == dtype
    np.testing.assert_array_equal(got, expected)


def _two_chains(first, second):
    """``(first * 2 - 1) * 3 + (second - 0.5) * 4``, with float32 constants."""
    scaled = (first * np.float32(2.0) - np.float32(1.0)) * np.float32(3.0)
    return scaled + (second - np.float32(0.5)) * np.float32(4.0)


def test_two_chains_joined_match_numpy():
    """Two chains on two inputs, then one add: the engine runs them as one
    stage."""
    first = _random((37, 61), 13)
    second = _random((37, 61), 14)
    got = _two_chains(mi.Array(first), mi.Array(second))
    expected = _two_chains(first, second)
    np.testing.assert_array_equal(got.realize(), expected)
