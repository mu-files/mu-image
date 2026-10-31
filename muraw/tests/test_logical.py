"""Logical ops against NumPy."""

import numpy as np
import pytest

import muimage as mi

_FLOAT_DTYPES = (np.float16, np.float32)


def _specials(dtype):
    """0, 1, -0, NaN, inf, and -inf, stored as ``dtype``."""
    return np.array([0.0, 1.0, -0.0, np.nan, np.inf, -np.inf], dtype)


def _logical_and(lib, first, second):
    return lib.logical_and(first, second)


def _logical_or(lib, first, second):
    return lib.logical_or(first, second)


def _logical_xor(lib, first, second):
    return lib.logical_xor(first, second)


def _logical_not(lib, src):
    return lib.logical_not(src)


def _added(lib, first, second):
    return first + second


def _times(lib, first, second):
    return first * second


def _check(got, exp):
    np.testing.assert_array_equal(got.realize(), exp)
    assert got.dtype == exp.dtype


@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_logical_not_of_specials_matches_numpy(dtype):
    src = _specials(dtype)
    got = _logical_not(mi, mi.Array(src))
    exp = _logical_not(np, src)
    _check(got, exp)


@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_logical_and_of_specials_matches_numpy(dtype):
    first = _specials(dtype)
    second = np.array([1.0, 0.0, np.nan, -0.0, 0.0, np.inf], dtype)
    got = _logical_and(mi, mi.Array(first), mi.Array(second))
    exp = _logical_and(np, first, second)
    _check(got, exp)


@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_logical_or_of_specials_matches_numpy(dtype):
    first = _specials(dtype)
    second = np.array([0.0, -0.0, 0.0, 1.0, np.nan, 0.0], dtype)
    got = _logical_or(mi, mi.Array(first), mi.Array(second))
    exp = _logical_or(np, first, second)
    _check(got, exp)


@pytest.mark.parametrize("dtype", _FLOAT_DTYPES)
def test_logical_xor_of_specials_matches_numpy(dtype):
    first = _specials(dtype)
    second = np.array([0.0, 1.0, -0.0, np.inf, 0.0, np.nan], dtype)
    got = _logical_xor(mi, mi.Array(first), mi.Array(second))
    exp = _logical_xor(np, first, second)
    _check(got, exp)


def test_logical_ops_of_bool_match_numpy():
    first = np.array([False, True, False, True])
    second = np.array([False, False, True, True])
    got = _logical_and(mi, mi.Array(first), mi.Array(second))
    exp = _logical_and(np, first, second)
    _check(got, exp)
    got = _logical_or(mi, mi.Array(first), mi.Array(second))
    exp = _logical_or(np, first, second)
    _check(got, exp)
    got = _logical_xor(mi, mi.Array(first), mi.Array(second))
    exp = _logical_xor(np, first, second)
    _check(got, exp)
    got = _logical_not(mi, mi.Array(first))
    exp = _logical_not(np, first)
    _check(got, exp)


def test_bool_add_and_multiply_match_numpy():
    first = np.array([False, True, False, True])
    second = np.array([False, False, True, True])
    got = _added(mi, mi.Array(first), mi.Array(second))
    exp = _added(np, first, second)
    _check(got, exp)
    got = _times(mi, mi.Array(first), mi.Array(second))
    exp = _times(np, first, second)
    _check(got, exp)
    got = _added(mi, True, mi.Array(first))
    exp = _added(np, True, first)
    _check(got, exp)


def test_logical_and_broadcasts_and_mixes_dtypes():
    base = np.array([[0, 1, 2], [0, 0, 5]], np.uint8)
    column = np.array([[1.0], [np.nan]], np.float32)
    got = _logical_and(mi, mi.Array(base), mi.Array(column))
    exp = _logical_and(np, base, column)
    _check(got, exp)


def test_logical_not_of_uint8_matches_numpy():
    src = np.array([0, 1, 2, 255], np.uint8)
    got = _logical_not(mi, mi.Array(src))
    exp = _logical_not(np, src)
    _check(got, exp)


def _bitwise_and(lib, first, second):
    return lib.bitwise_and(first, second)


def _bitwise_or(lib, first, second):
    return lib.bitwise_or(first, second)


def _bitwise_xor(lib, first, second):
    return lib.bitwise_xor(first, second)


def _bitwise_not(lib, src):
    return lib.bitwise_not(src)


def _invert(lib, src):
    return lib.invert(src)


def _left_shift(lib, first, second):
    return lib.left_shift(first, second)


def _right_shift(lib, first, second):
    return lib.right_shift(first, second)


def _bitwise_count(lib, src):
    return lib.bitwise_count(src)


def _bitand(lib, first, second):
    return first & second


def _bitor(lib, first, second):
    return first | second


def _bitxor(lib, first, second):
    return first ^ second


def _bitnot(lib, src):
    return ~src


def _lshift(lib, first, second):
    return first << second


def _rshift(lib, first, second):
    return first >> second


def _wide(dtype, values):
    return np.resize(np.array(values, dtype), (2, 40))


def test_bitwise_of_uint8_matches_numpy():
    first = _wide(np.uint8, [0, 1, 2, 127, 128, 255])
    second = _wide(np.uint8, [1, 3, 7, 15, 128, 255])
    for fn in (_bitwise_and, _bitwise_or, _bitwise_xor, _bitand, _bitor, _bitxor):
        got = fn(mi, mi.Array(first), mi.Array(second))
        exp = fn(np, first, second)
        _check(got, exp)
    got = _bitwise_not(mi, mi.Array(first))
    exp = _bitwise_not(np, first)
    _check(got, exp)
    got = _bitnot(mi, mi.Array(first))
    exp = _bitnot(np, first)
    _check(got, exp)
    got = _invert(mi, mi.Array(first))
    exp = _invert(np, first)
    _check(got, exp)


def test_bitwise_of_uint16_matches_numpy():
    first = _wide(np.uint16, [0, 1, 255, 256, 32768, 65535])
    second = _wide(np.uint16, [1, 255, 256, 32768, 65535, 3])
    got = _bitwise_xor(mi, mi.Array(first), mi.Array(second))
    exp = _bitwise_xor(np, first, second)
    _check(got, exp)
    got = _bitnot(mi, mi.Array(first))
    exp = _bitnot(np, first)
    _check(got, exp)


def test_bitwise_of_bool_matches_numpy():
    first = np.array([False, True, False, True])
    second = np.array([False, False, True, True])
    got = _bitand(mi, mi.Array(first), mi.Array(second))
    exp = _bitand(np, first, second)
    _check(got, exp)
    got = _bitor(mi, mi.Array(first), True)
    exp = _bitor(np, first, True)
    _check(got, exp)
    got = _bitxor(mi, mi.Array(first), mi.Array(second))
    exp = _bitxor(np, first, second)
    _check(got, exp)
    got = _bitnot(mi, mi.Array(first))
    exp = _bitnot(np, first)
    _check(got, exp)


def test_bitwise_and_broadcasts_a_column():
    base = np.array([[1, 2, 255, 0], [7, 8, 9, 16]], np.uint8)
    column = np.array([[3], [1]], np.uint8)
    got = _bitwise_and(mi, mi.Array(base), mi.Array(column))
    exp = _bitwise_and(np, base, column)
    _check(got, exp)


def test_shifts_match_numpy():
    values = _wide(np.uint8, [0, 1, 128, 255])
    counts = _wide(np.uint8, [0, 1, 7, 8, 9, 255])
    got = _lshift(mi, mi.Array(values), mi.Array(counts))
    exp = _lshift(np, values, counts)
    _check(got, exp)
    got = _rshift(mi, mi.Array(values), mi.Array(counts))
    exp = _rshift(np, values, counts)
    _check(got, exp)
    wide = _wide(np.uint16, [0, 1, 32768, 65535])
    wide_counts = _wide(np.uint16, [0, 1, 15, 16, 17, 65535])
    got = _left_shift(mi, mi.Array(wide), mi.Array(wide_counts))
    exp = _left_shift(np, wide, wide_counts)
    _check(got, exp)
    got = _right_shift(mi, mi.Array(wide), mi.Array(wide_counts))
    exp = _right_shift(np, wide, wide_counts)
    _check(got, exp)


def test_uint8_shifted_by_uint16_matches_numpy():
    value = np.array([1, 1], np.uint8)
    count = np.array([1, 8], np.uint16)
    got = _left_shift(mi, mi.Array(value), mi.Array(count))
    exp = _left_shift(np, value, count)
    _check(got, exp)


def test_bool_shifted_by_uint8_matches_numpy():
    value = np.array([False, True])
    count = np.array([1, 1], np.uint8)
    got = _lshift(mi, mi.Array(value), mi.Array(count))
    exp = _lshift(np, value, count)
    _check(got, exp)


def test_uint8_shifted_by_true_matches_numpy():
    value = np.array([1, 2, 128], np.uint8)
    got = _lshift(mi, mi.Array(value), True)
    exp = _lshift(np, value, True)
    _check(got, exp)


def test_bitwise_count_matches_numpy():
    for src in (
        np.array([0, 1, 255], np.uint8),
        np.array([0, 1, 65535], np.uint16),
        np.array([False, True]),
        _wide(np.uint8, [0, 1, 3, 255]),
        _wide(np.uint16, [0, 1, 65535]),
    ):
        got = _bitwise_count(mi, mi.Array(src))
        exp = _bitwise_count(np, src)
        _check(got, exp)


def test_bitwise_of_float_is_type_error():
    src = np.array([1.0, 2.0], np.float32)
    with pytest.raises(TypeError):
        _bitwise_and(mi, mi.Array(src), mi.Array(src))
    with pytest.raises(TypeError):
        _bitwise_and(np, src, src)
    with pytest.raises(TypeError):
        _left_shift(mi, mi.Array(src), mi.Array(src))
    with pytest.raises(TypeError):
        _left_shift(np, src, src)
    with pytest.raises(TypeError):
        _bitwise_count(mi, mi.Array(src))
    with pytest.raises(TypeError):
        _bitwise_count(np, src)
    with pytest.raises(TypeError):
        _invert(mi, mi.Array(src))
    with pytest.raises(TypeError):
        _invert(np, src)


def test_shift_of_an_int_past_the_dtype_is_overflow():
    src = np.array([1, 2], np.uint8)
    with pytest.raises(OverflowError):
        _lshift(mi, mi.Array(src), 256)
    with pytest.raises(OverflowError):
        _lshift(np, src, 256)
    with pytest.raises(OverflowError):
        _lshift(mi, mi.Array(src), -1)
    with pytest.raises(OverflowError):
        _lshift(np, src, -1)
    with pytest.raises(OverflowError):
        _lshift(mi, 256, mi.Array(src))
    with pytest.raises(OverflowError):
        _lshift(np, 256, src)


def test_bool_and_an_int_is_not_implemented():
    src = np.array([False, True])
    with pytest.raises(NotImplementedError):
        _bitand(mi, mi.Array(src), 1)


def test_bool_shifted_by_an_int_is_not_implemented():
    src = np.array([False, True])
    with pytest.raises(NotImplementedError):
        _lshift(mi, mi.Array(src), 1)


def test_uint8_and_int64_is_not_implemented():
    src = np.array([1, 2], np.uint8)
    with pytest.raises(NotImplementedError):
        _bitwise_and(mi, mi.Array(src), np.int64(1))
