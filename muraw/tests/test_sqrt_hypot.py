"""sqrt, hypot, and square against NumPy."""

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
