"""sqrt, hypot, square, and power against NumPy."""

from __future__ import annotations

import numpy as np
import pytest

import muimage as mi


def _sqrt_line(lib, src):
    """Square root of a two-row line.

    A negative is NaN. NumPy warns on that square root; the value is still NaN.
    """
    with np.errstate(invalid="ignore"):
        return lib.sqrt(src)


def _sqrt_count(lib, dtype):
    """Square root of ``0 .. 39`` in two rows."""
    return lib.sqrt(lib.arange(40, dtype=dtype).reshape(2, 20))


def _hypot_lines(lib, dtype):
    """Hypot of two 40-sample lines, in two rows."""
    first = lib.linspace(0.0, 8.0, 40, dtype=dtype).reshape(2, 20)
    second = lib.linspace(1.0, 5.0, 40, dtype=dtype).reshape(2, 20)
    return lib.hypot(first, second)


def _hypot_broadcast(lib):
    """A column of 4 against a row of 20."""
    column = lib.arange(4, dtype=np.float32).reshape(4, 1)
    row = lib.linspace(0.0, 8.0, 20, dtype=np.float32).reshape(1, 20)
    return lib.hypot(column, row)


def _hypot_counts(lib, dtype):
    """Hypot of ``0 .. 11`` and ``1 .. 12``."""
    first = lib.arange(12, dtype=dtype).reshape(3, 4)
    second = lib.arange(1, 13, dtype=dtype).reshape(3, 4)
    return lib.hypot(first, second)


def _hypot_large(lib):
    """Hypot of a pair of ``1e20`` samples."""
    src = lib.full((2, 8), np.float32(1.0e20), np.float32)
    return lib.hypot(src, src)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_sqrt_matches_numpy(dtype):
    src = np.linspace(-1.0, 9.0, 40, dtype=dtype).reshape(2, 20)
    got = _sqrt_line(mi, mi.Array(src))
    want = _sqrt_line(np, src)
    assert got.dtype == want.dtype
    # float16 is one rounding of that dtype. float32 of these samples matches.
    rtol = 1e-3 if dtype == "float16" else 1e-6
    np.testing.assert_allclose(got.realize(), want, rtol=rtol, atol=0, equal_nan=True)


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_sqrt_of_integers_matches_numpy(dtype):
    """uint8 gives float16, as NumPy does. uint16 gives float32."""
    got = _sqrt_count(mi, dtype)
    want = _sqrt_count(np, dtype)
    assert got.dtype == want.dtype
    rtol = 1e-3 if dtype == "uint8" else 1e-6
    np.testing.assert_allclose(got.realize(), want, rtol=rtol, atol=0)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_hypot_matches_numpy(dtype):
    got = _hypot_lines(mi, dtype)
    want = _hypot_lines(np, dtype)
    assert got.dtype == want.dtype
    rtol = 1e-3 if dtype == "float16" else 1e-6
    np.testing.assert_allclose(got.realize(), want, rtol=rtol, atol=0)


def test_hypot_broadcast_matches_numpy():
    got = _hypot_broadcast(mi)
    want = _hypot_broadcast(np)
    np.testing.assert_allclose(got.realize(), want, rtol=1e-6, atol=0)


def test_hypot_of_integers_writes_float32():
    """Integer inputs give float32. NumPy's hypot of uint8 is float16."""
    got = _hypot_counts(mi, np.uint8)
    assert got.dtype == np.float32
    np.testing.assert_allclose(
        got.realize(), _hypot_counts(np, np.float32), rtol=1e-6, atol=0
    )


def test_hypot_of_large_values_stays_finite():
    got = _hypot_large(mi)
    want = _hypot_large(np)
    np.testing.assert_allclose(got.realize(), want, rtol=1e-5, atol=0)
    assert np.isfinite(got.realize()).all()


def _square(lib, src):
    """``src`` squared."""
    return lib.square(src)


def _square_count(lib, dtype):
    """Square of ``0 .. 15``."""
    return lib.square(lib.arange(16, dtype=dtype).reshape(4, 4))


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_square_matches_numpy(dtype):
    src = np.linspace(-3.0, 4.0, 40, dtype=dtype).reshape(2, 20)
    got = _square(mi, mi.Array(src))
    want = _square(np, src)
    assert got.dtype == want.dtype
    rtol = 1e-3 if dtype == "float16" else 1e-6
    np.testing.assert_allclose(got.realize(), want, rtol=rtol, atol=0)


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_square_of_integers_matches_numpy_when_it_fits(dtype):
    """``0 .. 15`` squared fits both integer dtypes."""
    got = _square_count(mi, dtype)
    want = _square_count(np, dtype)
    assert got.dtype == want.dtype
    np.testing.assert_array_equal(got.realize(), want)


@pytest.mark.parametrize("dtype,value", [("uint8", 16), ("uint16", 256)])
def test_square_of_integers_saturates(dtype, value):
    """A square past the dtype's range saturates. NumPy wraps."""
    src = np.array([value, 1], dtype=dtype)
    got = mi.square(mi.Array(src)).realize()
    top = np.iinfo(dtype).max
    np.testing.assert_array_equal(got, np.array([top, 1], dtype=dtype))


def _power(lib, base, exponent):
    """``base`` raised to ``exponent``.

    NumPy warns on a zero to a negative power and on a negative to a
    non-integer. The value is still the one ``power`` returns.
    """
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return lib.power(base, exponent)


def _powered(lib, base, exponent):
    """``base ** exponent``."""
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        return base**exponent


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_power_matches_numpy(dtype):
    base = np.linspace(0.05, 4.0, 40, dtype=dtype).reshape(2, 20)
    exponent = np.float32(2.2) if dtype == "float32" else np.float16(2.2)
    got = _power(mi, mi.Array(base), exponent)
    want = _power(np, base, exponent)
    assert got.dtype == want.dtype
    rtol = 1e-3 if dtype == "float16" else 1e-6
    np.testing.assert_allclose(got.realize(), want, rtol=rtol, atol=0, equal_nan=True)


def test_power_specials_match_numpy():
    """The default is the real power. NumPy has no ``fast`` argument."""
    base = np.array(
        [np.nan, 1.0, -1.0, -2.0, -2.0, 0.0, np.inf, -0.0], dtype=np.float32
    )
    exponent = np.array(
        [0.0, np.nan, np.inf, 3.0, 2.5, -1.0, 0.0, -1.0], dtype=np.float32
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        got = mi.power(mi.Array(base), mi.Array(exponent))
        want = np.power(base, exponent)
    np.testing.assert_allclose(got.realize(), want, rtol=1e-6, atol=0, equal_nan=True)


def test_power_of_integers_wraps():
    """``255 ** 2`` in uint8 is 1. ``2 ** 16`` in uint16 is 0. ``0 ** 0`` is 1."""
    base = np.array([0, 1, 2, 255, 3], dtype=np.uint8)
    exponent = np.array([0, 0, 8, 2, 1], dtype=np.uint8)
    got = _power(mi, mi.Array(base), mi.Array(exponent))
    want = _power(np, base, exponent)
    assert got.dtype == np.uint8
    np.testing.assert_array_equal(got.realize(), want)
    accurate = mi.power(mi.Array(base), mi.Array(exponent), fast=False)
    np.testing.assert_array_equal(accurate.realize(), want)
    wide = np.array([2, 3], dtype=np.uint16)
    got_wide = _power(mi, mi.Array(wide), np.uint16(16))
    want_wide = _power(np, wide, np.uint16(16))
    np.testing.assert_array_equal(got_wide.realize(), want_wide)


def test_power_broadcast_matches_numpy():
    column = np.linspace(0.5, 2.0, 4, dtype=np.float32).reshape(4, 1)
    row = np.linspace(0.25, 3.0, 7, dtype=np.float32).reshape(1, 7)
    got = _power(mi, mi.Array(column), mi.Array(row))
    want = _power(np, column, row)
    np.testing.assert_allclose(got.realize(), want, rtol=1e-6, atol=0)


def test_fast_power_drops_the_real_power_specials():
    """``fast=True`` is faster and does not support a negative base."""
    base = np.array([0.0, -2.0, -0.0, 1.0], dtype=np.float32)
    exponent = np.array([0.0, 3.0, -1.0, np.inf], dtype=np.float32)
    got = mi.power(mi.Array(base), mi.Array(exponent), fast=True).realize()
    assert np.isnan(got[0])
    assert np.isnan(got[1])
    assert np.isposinf(got[2])
    assert np.isnan(got[3])
    accurate = mi.power(mi.Array(base), mi.Array(exponent)).realize()
    assert accurate[0] == np.float32(1.0)
    assert accurate[1] == np.float32(-8.0)


def test_power_operator_and_reflected_power():
    src = np.arange(8, dtype=np.float32).reshape(2, 4) + np.float32(0.5)
    got = _powered(mi, mi.Array(src), 2)
    want = _powered(np, src, 2)
    assert got.dtype == np.float32
    np.testing.assert_allclose(got.realize(), want, rtol=1e-6, atol=0)
    got_reflected = _powered(mi, 2, mi.Array(src))
    want_reflected = _powered(np, 2, src)
    assert got_reflected.dtype == np.float32
    np.testing.assert_allclose(
        got_reflected.realize(), want_reflected, rtol=1e-6, atol=0
    )


def test_pow_is_power():
    src = np.arange(4, dtype=np.float32) + np.float32(1.0)
    got = mi.pow(mi.Array(src), np.float32(3.0)).realize()
    want = mi.power(mi.Array(src), np.float32(3.0)).realize()
    np.testing.assert_array_equal(got, want)


def test_python_float_beside_uint8_is_float32():
    """``uint8 ** 2.0`` is float64 in NumPy and float32 here."""
    src = np.arange(4, dtype=np.uint8)
    got = _power(mi, mi.Array(src), 2.0)
    assert got.dtype == np.float32
    want = _power(np, src.astype(np.float32), np.float32(2.0))
    np.testing.assert_allclose(got.realize(), want, rtol=1e-6, atol=0)


def test_negative_exponent_overflows():
    src = np.arange(4, dtype=np.uint8)

    def _negative(lib, base):
        return lib.power(base, -1)

    with pytest.raises(OverflowError):
        _negative(mi, mi.Array(src))
    with pytest.raises(OverflowError):
        _negative(np, src)


def test_bool_power_is_not_implemented():
    src = np.array([True, False])
    with pytest.raises(NotImplementedError):
        mi.power(mi.Array(src), 2)


def test_uint8_power_int64_is_not_implemented():
    src = np.arange(4, dtype=np.uint8)
    with pytest.raises(NotImplementedError):
        mi.power(mi.Array(src), np.int64(2))


def test_power_with_a_modulus_is_not_implemented():
    src = mi.Array(np.arange(4, dtype=np.uint8))
    with pytest.raises(NotImplementedError):
        pow(src, 2, 3)
