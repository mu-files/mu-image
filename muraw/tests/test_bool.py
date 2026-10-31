"""bool arrays, cast, arithmetic with a number, and the structural ops."""

import numpy as np
import pytest

import muimage as mi


def _cast(lib, src, dtype):
    return src.astype(dtype)


def _times(lib, src, factor):
    return src * factor


def _add(lib, src, other):
    return src + other


def test_uint8_to_bool_matches_numpy():
    src = np.array([0, 1, 2, 255], np.uint8)
    got = _cast(mi, mi.Array(src), np.bool_)
    exp = _cast(np, src, np.bool_)
    np.testing.assert_array_equal(got.realize(), exp)


def test_float_edges_to_bool_match_numpy():
    for dtype in (np.float16, np.float32):
        src = np.array([0.0, -0.0, 1.0, np.nan, np.inf, -np.inf], dtype)
        got = _cast(mi, mi.Array(src), np.bool_)
        exp = _cast(np, src, np.bool_)
        np.testing.assert_array_equal(got.realize(), exp)


def test_bool_to_numeric_matches_numpy():
    src = np.array([False, True])
    for dtype in (np.uint16, np.float16, np.float32):
        got = _cast(mi, mi.Array(src), dtype)
        exp = _cast(np, src, dtype)
        np.testing.assert_array_equal(got.realize(), exp)


def test_bool_astype_uint8_shares_bytes():
    src = np.array([False, True, False])
    mask = mi.Array(src)
    got = mask.astype(np.uint8)
    assert np.shares_memory(mask.realize(), got.realize())
    np.testing.assert_array_equal(got.realize(), src.view(np.uint8))


def test_lazy_bool_astype_uint8_stays_in_the_graph():
    src = np.array([0, 1, 2, 255], np.uint8)
    mask = mi.Array(src).astype(np.bool_)
    got = mask.astype(np.uint8)
    assert mask._data is None
    assert got._data is None
    assert got._node is not None and got._node.op == "cast_dtype"
    np.testing.assert_array_equal(got.realize(), np.array([0, 1, 1, 1], np.uint8))


def test_bool_times_uint8_matches_numpy():
    src = np.array([False, True, False, True])
    factor = np.uint8(255)
    got = _times(mi, mi.Array(src), factor)
    exp = _times(np, src, factor)
    np.testing.assert_array_equal(got.realize(), exp)
    assert got.dtype == np.dtype(np.uint8)
    np.testing.assert_array_equal((factor * mi.Array(src)).realize(), exp)


def test_bool_times_bool_matches_numpy():
    src = np.array([False, True, True, False])
    other = np.array([True, False, True, False])
    got = _times(mi, mi.Array(src), mi.Array(other))
    exp = _times(np, src, other)
    np.testing.assert_array_equal(got.realize(), exp)
    assert got.dtype == np.dtype(np.bool_)


def test_bool_times_float32_matches_numpy():
    src = np.array([False, True])
    factor = np.float32(2.5)
    got = _times(mi, mi.Array(src), factor)
    exp = _times(np, src, factor)
    np.testing.assert_array_equal(got.realize(), exp)


def test_bool_times_uint8_array_matches_numpy():
    src = np.array([False, True, True])
    factor = np.array([10, 20, 30], np.uint8)
    got = _times(mi, mi.Array(src), mi.Array(factor))
    exp = _times(np, src, factor)
    np.testing.assert_array_equal(got.realize(), exp)


def test_bool_times_python_int_is_not_implemented():
    src = np.array([False, True])
    with pytest.raises(NotImplementedError):
        _times(mi, mi.Array(src), 255)


def test_bool_plus_uint8_matches_numpy():
    src = np.array([False, True])
    other = np.uint8(1)
    got = _add(mi, mi.Array(src), other)
    exp = _add(np, src, other)
    np.testing.assert_array_equal(got.realize(), exp)


def _div(lib, src, other):
    return src / other


def test_bool_div_float32_matches_numpy():
    src = np.array([False, True])
    other = np.float32(2)
    got = _div(mi, mi.Array(src), other)
    exp = _div(np, src, other)
    np.testing.assert_array_equal(got.realize(), exp)


def test_bool_minus_uint8_saturates():
    """uint8 subtract saturates. NumPy wraps ``False - 1`` to 255."""
    src = np.array([False, True])
    got = (mi.Array(src) - np.uint8(1)).realize()
    np.testing.assert_array_equal(got, np.array([0, 0], np.uint8))


def _sub(lib, src, other):
    return src - other


def test_bool_subtract_bool_is_type_error():
    src = np.array([False, True])
    with pytest.raises(TypeError):
        _sub(mi, mi.Array(src), mi.Array(src))
    with pytest.raises(TypeError):
        _sub(np, src, src)


def _neg(lib, src):
    return -src


def test_bool_negative_is_type_error():
    src = np.array([False, True])
    with pytest.raises(TypeError):
        _neg(mi, mi.Array(src))
    with pytest.raises(TypeError):
        _neg(np, src)


def test_bool_plus_bool_matches_numpy():
    src = np.array([False, True, True, False])
    other = np.array([True, False, True, False])
    got = _add(mi, mi.Array(src), mi.Array(other))
    exp = _add(np, src, other)
    np.testing.assert_array_equal(got.realize(), exp)
    assert got.dtype == np.dtype(np.bool_)


def _mask():
    """A 3 by 4 bool mask."""
    return np.array(
        [
            [False, True, False, True],
            [True, False, True, False],
            [False, False, True, True],
        ]
    )


def _planes():
    """A bool ``(H, W, C)`` mask."""
    return np.arange(3 * 4 * 3).reshape(3, 4, 3).astype(bool)


def _window(lib, src):
    """Rows from 1 and columns 1 through 3."""
    return src[1:, 1:4]


def _stepped(lib, src):
    """Every other column."""
    return src[:, ::2]


def _reshaped(lib, src):
    """``src`` as two rows of six."""
    return lib.reshape(src, (2, 6))


def _transposed(lib, src):
    """Channels moved in front of the rows."""
    return lib.transpose(src, (2, 0, 1))


def _swapped(lib, src):
    """Rows and columns exchanged."""
    return lib.transpose(src, (1, 0))


def _turned(lib, src):
    """``src`` turned a quarter turn counterclockwise."""
    return lib.rot90(src)


def _packed(lib, src):
    """Frames on axis 0 packed into channels."""
    return lib.transpose(src, (1, 2, 0))


def _taken(lib, src):
    """Columns 2, 0, and 2."""
    return lib.take(src, [2, 0, 2], axis=1)


def _tiled(lib, src):
    """``src`` repeated twice down and three times across."""
    return lib.tile(src, (2, 3))


def _padded(lib, src, value):
    """``src`` with a one-sample constant margin."""
    return lib.pad(src, 1, constant_values=value)


def _filled(lib, value):
    """A 2 by 3 bool array filled with ``value``."""
    return lib.full((2, 3), value, dtype=np.bool_)


def _filled_as_value(lib, value):
    """A 2 by 3 array filled with ``value``, dtype taken from it."""
    return lib.full((2, 3), value)


def test_bool_slice_matches_numpy():
    src = _mask()
    got = _window(mi, mi.Array(src))
    np.testing.assert_array_equal(got.realize(), _window(np, src))


def test_bool_stepped_slice_matches_numpy():
    src = _mask()
    got = _stepped(mi, mi.Array(src))
    np.testing.assert_array_equal(got.realize(), _stepped(np, src))


def test_bool_reshape_matches_numpy():
    src = _mask()
    got = _reshaped(mi, mi.Array(src))
    np.testing.assert_array_equal(got.realize(), _reshaped(np, src))


def test_bool_transpose_matches_numpy():
    src = _planes()
    got = _transposed(mi, mi.Array(src))
    np.testing.assert_array_equal(got.realize(), _transposed(np, src))


def test_bool_swap_matches_numpy():
    src = _mask()
    got = _swapped(mi, mi.Array(src))
    np.testing.assert_array_equal(got.realize(), _swapped(np, src))


def test_bool_rot90_matches_numpy():
    src = _mask()
    got = _turned(mi, mi.Array(src))
    np.testing.assert_array_equal(got.realize(), _turned(np, src))


def test_bool_chw_transpose_matches_numpy():
    src = _planes()
    got = _packed(mi, mi.Array(src, channel_axis=0))
    np.testing.assert_array_equal(got.realize(), _packed(np, src))


def test_bool_take_matches_numpy():
    src = _mask()
    got = _taken(mi, mi.Array(src))
    np.testing.assert_array_equal(got.realize(), _taken(np, src))


def test_bool_tile_matches_numpy():
    src = _mask()
    got = _tiled(mi, mi.Array(src))
    np.testing.assert_array_equal(got.realize(), _tiled(np, src))


@pytest.mark.parametrize("value", [0, 1])
def test_bool_pad_constant_matches_numpy(value):
    src = _mask()
    got = _padded(mi, mi.Array(src), value)
    np.testing.assert_array_equal(got.realize(), _padded(np, src, value))


@pytest.mark.parametrize("value", [0, 1, 7])
def test_bool_fill_matches_numpy(value):
    got = _filled(mi, value)
    np.testing.assert_array_equal(got.realize(), _filled(np, value))


@pytest.mark.parametrize("value", [False, True])
def test_full_of_a_python_bool_matches_numpy(value):
    got = _filled_as_value(mi, value)
    want = _filled_as_value(np, value)
    assert got.dtype == want.dtype
    np.testing.assert_array_equal(got.realize(), want)
