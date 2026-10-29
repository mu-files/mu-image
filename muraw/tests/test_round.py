"""round, rint, floor, and ceil against NumPy."""

from __future__ import annotations

import numpy as np
import pytest

import muimage as mi


def _samples(dtype):
    """Halves, a value off a half, and a NaN, in two rows."""
    return np.array(
        [-2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5, 1.1, -1.1, np.nan],
        dtype=dtype,
    ).reshape(2, 5)


def _rint(lib, src):
    """``src`` rounded to the nearest integer."""
    return lib.rint(src)


def _round(lib, src):
    """``src`` rounded with the default decimal count."""
    return lib.round(src)


def _round_at(lib, src, decimals):
    """``src`` rounded at ``decimals``."""
    return lib.round(src, decimals=decimals)


def _floor(lib, src):
    """``src`` rounded toward negative infinity."""
    return lib.floor(src)


def _ceil(lib, src):
    """``src`` rounded toward positive infinity."""
    return lib.ceil(src)


def _directed_count(lib, dtype, toward):
    """``toward`` of ``0 .. 15``."""
    values = lib.arange(16, dtype=dtype).reshape(4, 4)
    if toward == "floor":
        return lib.floor(values)
    return lib.ceil(values)


def _rint_count(lib, dtype):
    """``rint`` of ``0 .. 15``."""
    return lib.rint(lib.arange(16, dtype=dtype).reshape(4, 4))


def _round_count(lib, dtype):
    """``round`` of ``0 .. 15``."""
    return lib.round(lib.arange(16, dtype=dtype).reshape(4, 4))


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_rint_matches_numpy(dtype):
    src = _samples(dtype)
    got = _rint(mi, mi.Array(src))
    want = _rint(np, src)
    assert got.dtype == want.dtype
    realized = got.realize()
    np.testing.assert_allclose(realized, want, rtol=0, atol=0, equal_nan=True)
    finite = np.isfinite(want)
    np.testing.assert_array_equal(
        np.signbit(realized[finite]), np.signbit(want[finite])
    )


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_rint_of_integers_matches_numpy(dtype):
    """uint8 gives float16, as NumPy does. uint16 gives float32."""
    got = _rint_count(mi, dtype)
    want = _rint_count(np, dtype)
    assert got.dtype == want.dtype
    np.testing.assert_array_equal(got.realize(), want)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_round_of_floats_matches_numpy(dtype):
    src = _samples(dtype)
    got = _round(mi, mi.Array(src))
    want = _round(np, src)
    assert got.dtype == want.dtype
    np.testing.assert_allclose(got.realize(), want, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_round_of_integers_keeps_the_dtype(dtype):
    """An integer is unchanged. NumPy's ``rint`` of the same values is a float."""
    got = _round_count(mi, dtype)
    want = _round_count(np, dtype)
    assert got.dtype == want.dtype
    np.testing.assert_array_equal(got.realize(), want)


def test_round_accepts_a_zero_decimal_count():
    src = _samples(np.float32)
    got = _round_at(mi, mi.Array(src), 0)
    want = _round_at(np, src, 0)
    np.testing.assert_allclose(got.realize(), want, rtol=0, atol=0, equal_nan=True)


def test_round_rejects_a_float_decimal_count():
    src = np.zeros(4, dtype=np.float32)
    with pytest.raises(TypeError):
        _round_at(mi, mi.Array(src), 1.5)
    with pytest.raises(TypeError):
        _round_at(np, src, 1.5)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("decimals", [1, 2, 3, -1, -2, 5, -4, 8, 40])
def test_round_at_decimals_matches_numpy(dtype, decimals):
    src = np.concatenate(
        [
            np.linspace(-40, 80, 48, dtype=dtype),
            np.array([1.15, 1.25, 2.25, 2.5, 77.0, -0.5], dtype=dtype),
        ]
    )
    got = _round_at(mi, mi.Array(src), decimals)
    want = _round_at(np, src, decimals)
    assert got.dtype == want.dtype
    np.testing.assert_allclose(got.realize(), want, rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("decimals", [-1, -2, -4, -20, 0, 1])
def test_round_uint8_at_decimals_matches_numpy(decimals):
    src = np.arange(256, dtype=np.uint8)
    got = _round_at(mi, mi.Array(src), decimals)
    want = _round_at(np, src, decimals)
    assert got.dtype == want.dtype
    np.testing.assert_array_equal(got.realize(), want)


@pytest.mark.parametrize("decimals", [-1, -2, -4, -20, 1])
def test_round_uint16_at_decimals_matches_numpy(decimals):
    src = np.array([0, 5, 25, 50, 255, 1000, 40000, 65535], dtype=np.uint16)
    got = _round_at(mi, mi.Array(src), decimals)
    want = _round_at(np, src, decimals)
    assert got.dtype == want.dtype
    np.testing.assert_array_equal(got.realize(), want)


def _edges(dtype):
    """Fractions, signed zero, infinities, and a NaN."""
    return np.array(
        [-2.5, -0.5, -0.0, 0.0, 0.5, 1.1, -1.1, np.nan, np.inf, -np.inf],
        dtype=dtype,
    ).reshape(2, 5)


def _sign_bits(got, want):
    """The sign of every sample that is not NaN."""
    known = ~np.isnan(want)
    np.testing.assert_array_equal(np.signbit(got[known]), np.signbit(want[known]))


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_floor_matches_numpy(dtype):
    src = _edges(dtype)
    got = _floor(mi, mi.Array(src))
    want = _floor(np, src)
    assert got.dtype == want.dtype
    realized = got.realize()
    np.testing.assert_allclose(realized, want, rtol=0, atol=0, equal_nan=True)
    _sign_bits(realized, want)


@pytest.mark.parametrize("dtype", ["float16", "float32"])
def test_ceil_matches_numpy(dtype):
    src = _edges(dtype)
    got = _ceil(mi, mi.Array(src))
    want = _ceil(np, src)
    assert got.dtype == want.dtype
    realized = got.realize()
    np.testing.assert_allclose(realized, want, rtol=0, atol=0, equal_nan=True)
    _sign_bits(realized, want)


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
@pytest.mark.parametrize("toward", ["floor", "ceil"])
def test_floor_and_ceil_of_integers_keep_the_dtype(dtype, toward):
    got = _directed_count(mi, dtype, toward)
    want = _directed_count(np, dtype, toward)
    assert got.dtype == want.dtype
    np.testing.assert_array_equal(got.realize(), want)
