"""bool arrays, cast_dtype, and multiply of a bool by a numeric value."""

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


def test_bool_times_bool_is_not_multiply():
    src = np.array([False, True])
    with pytest.raises(NotImplementedError):
        _times(mi, mi.Array(src), mi.Array(src))


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


def test_bool_plus_bool_is_not_implemented():
    src = np.array([False, True])
    with pytest.raises(NotImplementedError):
        _add(mi, mi.Array(src), mi.Array(src))
