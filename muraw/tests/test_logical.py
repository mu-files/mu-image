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
