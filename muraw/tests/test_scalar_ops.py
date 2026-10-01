"""+, -, *, / between an Array and a number or a per-channel constant."""

from __future__ import annotations

import operator

import numpy as np
import pytest

import muimage as mi

_OPS = {
    "add": operator.add,
    "sub": operator.sub,
    "mul": operator.mul,
    "div": operator.truediv,
}


def _saturate(values: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Round half up and clamp to ``dtype``'s range, as the integer kernels do."""
    top = np.iinfo(dtype).max
    return np.clip(np.floor(values + np.float32(0.5)), 0, top).astype(dtype)


def _source(dtype: str) -> np.ndarray:
    rng = np.random.default_rng(7)
    if dtype == "uint8":
        return rng.integers(0, 256, (5, 9)).astype(np.uint8)
    if dtype == "uint16":
        return rng.integers(0, 65536, (5, 9)).astype(np.uint16)
    return rng.uniform(-2.0, 2.0, (5, 9)).astype(dtype)


def _check(got: mi.Array, want: np.ndarray, max_ulp: int = 0) -> None:
    """``max_ulp`` is 1 for division: the kernels build with fast math, which
    lets a division become a multiply by the reciprocal."""
    assert got.dtype == mi.ElementType(want.dtype)
    if max_ulp:
        np.testing.assert_array_max_ulp(np.asarray(got), want, maxulp=max_ulp)
    else:
        np.testing.assert_array_equal(np.asarray(got), want)


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "float16", "float32"])
@pytest.mark.parametrize("name", list(_OPS))
@pytest.mark.parametrize("reflected", [False, True])
def test_a_float_constant_matches_numpy(dtype, name, reflected):
    src = _source(dtype)
    fn = _OPS[name]
    const = 0.75
    if reflected:
        got = fn(const, mi.Array(src))
        want = fn(const, src)
    else:
        got = fn(mi.Array(src), const)
        want = fn(src, const)
    if dtype in ("uint8", "uint16"):
        f32 = src.astype(np.float32)
        want = fn(np.float32(const), f32) if reflected else fn(f32, np.float32(const))
    _check(got, want, max_ulp=1 if name == "div" else 0)


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
@pytest.mark.parametrize("name", ["add", "sub", "mul"])
@pytest.mark.parametrize("reflected", [False, True])
def test_an_int_constant_keeps_the_integer_dtype_and_saturates(dtype, name, reflected):
    src = _source(dtype)
    fn = _OPS[name]
    const = 3
    got = fn(const, mi.Array(src)) if reflected else fn(mi.Array(src), const)
    f32 = src.astype(np.float32)
    exact = fn(np.float32(const), f32) if reflected else fn(f32, np.float32(const))
    _check(got, _saturate(exact, np.dtype(dtype)))


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_integer_division_gives_float32(dtype):
    src = _source(dtype) + 1
    got = mi.Array(src) / 3
    _check(got, src.astype(np.float32) / np.float32(3), max_ulp=1)
    got = 3 / mi.Array(src)
    _check(got, np.float32(3) / src.astype(np.float32), max_ulp=1)


def test_uint8_results_saturate_at_both_ends():
    src = np.array([[10, 200]], dtype=np.uint8)
    _check(mi.Array(src) - 5, np.array([[5, 195]], dtype=np.uint8))
    _check(mi.Array(src) - 20, np.array([[0, 180]], dtype=np.uint8))
    _check(mi.Array(src) * 2, np.array([[20, 255]], dtype=np.uint8))
    _check(100 - mi.Array(src), np.array([[90, 0]], dtype=np.uint8))


def test_a_value_per_channel_scales_each_channel_of_a_packed_array():
    src = np.random.default_rng(1).uniform(0, 1, (4, 6, 3)).astype(np.float32)
    gains = [1.0, 1.5, 2.5]
    _check(mi.Array(src) * gains, src * np.float32(gains))
    _check(mi.Array(src) - np.array(gains), src - np.float32(gains))


def test_an_int_list_on_a_uint8_array_keeps_uint8():
    src = np.full((2, 2, 3), 250, dtype=np.uint8)
    got = mi.Array(src) + [1, 2, 10]
    want = np.broadcast_to(np.array([251, 252, 255], dtype=np.uint8), (2, 2, 3))
    _check(got, want)


@pytest.mark.parametrize("dtype", ["uint8", "float32"])
def test_a_value_per_plane_scales_each_plane_of_a_planar_array(dtype):
    chw = (np.arange(3 * 4 * 5) % 50).reshape(3, 4, 5).astype(dtype)
    gains = np.array([1, 2, 3]).reshape(3, 1, 1)
    got = mi.Array(chw, channel_axis=0) * gains
    assert got.meta.channel_axis == 0
    if dtype == "uint8":
        want = _saturate(chw.astype(np.float32) * gains, np.dtype(dtype))
    else:
        want = chw * gains.astype(np.float32)
    _check(got, want)


def test_a_lazy_chain_of_operators_matches_numpy():
    src = np.random.default_rng(2).uniform(0, 1, (6, 8, 3)).astype(np.float32)
    got = (2.0 - mi.Array(src) * [0.5, 1.0, 2.0]) / 4.0 + 1.0
    want = (np.float32(2.0) - src * np.float32([0.5, 1.0, 2.0])) / np.float32(4.0)
    _check(got, want + np.float32(1.0))


def test_a_constant_of_strings_is_rejected():
    arr = mi.Array(np.zeros((4, 6), dtype=np.float32))
    with pytest.raises(TypeError, match="expected a number"):
        arr + "a"
