"""+, -, *, / between two arrays, with NumPy broadcasting and dtype promotion."""

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
_DTYPES = ["uint8", "uint16", "float16", "float32"]


def _source(dtype: str, shape: tuple[int, ...], seed: int) -> np.ndarray:
    """Values away from zero, so a division stays finite."""
    rng = np.random.default_rng(seed)
    if dtype == "uint8":
        return rng.integers(1, 256, shape).astype(np.uint8)
    if dtype == "uint16":
        return rng.integers(1, 65536, shape).astype(np.uint16)
    return rng.uniform(0.25, 2.0, shape).astype(dtype)


def _expected(fn, first: np.ndarray, second: np.ndarray, divide: bool) -> np.ndarray:
    """NumPy's result, except that integer results saturate and integer
    division is float32."""
    dtype = np.result_type(first, second)
    if dtype.kind == "f":
        return fn(first.astype(dtype), second.astype(dtype))
    exact = fn(first.astype(np.float32), second.astype(np.float32))
    if divide:
        return exact
    top = np.iinfo(dtype).max
    return np.clip(np.floor(exact + np.float32(0.5)), 0, top).astype(dtype)


def _check(got: mi.Array, want: np.ndarray, divide: bool = False) -> None:
    """A division may be 1 ulp off: the kernels build with fast math, which
    lets a division become a multiply by the reciprocal."""
    assert got.dtype == mi.ElementType(want.dtype)
    assert got.shape == want.shape
    if divide:
        np.testing.assert_array_max_ulp(np.asarray(got), want, maxulp=1)
    else:
        np.testing.assert_array_equal(np.asarray(got), want)


@pytest.mark.parametrize("second_dtype", _DTYPES)
@pytest.mark.parametrize("first_dtype", _DTYPES)
@pytest.mark.parametrize("name", list(_OPS))
def test_each_operator_matches_numpy_for_every_dtype_pair(name, first_dtype, second_dtype):
    first = _source(first_dtype, (4, 9, 3), 1)
    second = _source(second_dtype, (4, 9, 3), 2)
    fn = _OPS[name]
    got = fn(mi.Array(first), mi.Array(second))
    _check(got, _expected(fn, first, second, name == "div"), divide=name == "div")


_BROADCAST_SHAPES = [
    ((5, 7), (1, 7)),
    ((5, 7), (5, 1)),
    ((5, 7), (1, 1)),
    ((5, 7), (7,)),
    ((1, 7), (5, 1)),
    ((5, 7, 3), (5, 7, 1)),
    ((5, 7, 3), (7, 3)),
    ((5, 1, 3), (1, 7, 1)),
]


@pytest.mark.parametrize("first_shape,second_shape", _BROADCAST_SHAPES)
@pytest.mark.parametrize("name", list(_OPS))
def test_broadcast_shapes_match_numpy(name, first_shape, second_shape):
    fn = _OPS[name]
    first = _source("float32", first_shape, 3)
    second = _source("float32", second_shape, 4)
    _check(fn(mi.Array(first), mi.Array(second)), fn(first, second), divide=name == "div")
    _check(fn(mi.Array(second), mi.Array(first)), fn(second, first), divide=name == "div")


@pytest.mark.parametrize("name", list(_OPS))
def test_an_ndarray_on_either_side_stays_lazy(name):
    fn = _OPS[name]
    arr = _source("float32", (5, 7), 5)
    other = _source("float32", (5, 7), 6)
    for got, want in [
        (fn(mi.Array(arr), other), fn(arr, other)),
        (fn(other, mi.Array(arr)), fn(other, arr)),
    ]:
        assert isinstance(got, mi.Array)
        _check(got, want, divide=name == "div")


def test_an_integer_ndarray_keeps_a_uint8_array_uint8():
    arr = np.full((2, 3), 250, dtype=np.uint8)
    got = mi.Array(arr) + np.array([[1, 2, 10], [0, 5, 6]])
    _check(got, np.array([[251, 252, 255], [250, 255, 255]], dtype=np.uint8))


def test_a_float_ndarray_makes_a_uint8_array_float32():
    arr = np.full((2, 3), 10, dtype=np.uint8)
    other = np.arange(6, dtype=np.float64).reshape(2, 3) / 4
    _check(mi.Array(arr) * other, arr.astype(np.float32) * other.astype(np.float32))


def test_per_channel_constants_use_the_scalar_op_and_others_the_two_input_op():
    arr = mi.Array(np.ones((4, 5, 3), dtype=np.float32))
    assert (arr * [1.0, 1.5, 2.5])._node.op == "mul_scalar"
    assert (arr * np.ones((4, 5, 1)))._node.op == "multiply"


def test_planar_arrays_broadcast_per_plane():
    chw = _source("float32", (3, 4, 5), 7)
    rows = _source("float32", (3, 4, 1), 8)
    got = mi.Array(chw, channel_axis=0) * mi.Array(rows, channel_axis=0)
    assert got.meta.channel_axis == 0
    _check(got, chw * rows)
    one_plane = _source("float32", (4, 5), 9)
    _check(mi.Array(chw, channel_axis=0) - mi.Array(one_plane), chw - one_plane)


def test_arrays_read_differently_by_the_engine_are_not_supported_yet():
    chw = _source("float32", (3, 4, 5), 7)
    with pytest.raises(NotImplementedError, match="channel_axis"):
        mi.Array(chw, channel_axis=0) + mi.Array(chw)


def test_sparse_meshgrid_sum_builds_no_full_grid():
    x, y = mi.meshgrid(mi.linspace(-1.0, 1.0, 7), mi.linspace(0.0, 2.0, 4), sparse=True)
    got = x + y
    assert got._node.op == "add"
    assert [inp.shape for inp in got._node.inputs] == [(1, 7), (4, 1)]
    _check(got, np.asarray(x) + np.asarray(y))


def test_vignette_matches_numpy():
    height, width = 6, 9
    cx, cy = (width - 1) / 2, (height - 1) / 2
    x, y = mi.meshgrid(mi.arange(width), mi.arange(height))
    got = (x - cx) * (x - cx) + (y - cy) * (y - cy)
    xx, yy = np.meshgrid(np.arange(width, dtype=np.float32), np.arange(height, dtype=np.float32))
    want = (xx - np.float32(cx)) * (xx - np.float32(cx)) + (yy - np.float32(cy)) * (
        yy - np.float32(cy)
    )
    _check(got, want)


def test_shapes_that_do_not_broadcast_raise():
    with pytest.raises(ValueError):
        mi.Array(np.zeros((4, 5), np.float32)) + mi.Array(np.zeros((3, 5), np.float32))


@pytest.mark.parametrize(
    "fn,op",
    [(mi.add, operator.add), (mi.subtract, operator.sub), (mi.multiply, operator.mul)],
)
def test_numpy_function_names(fn, op):
    first = _source("float32", (3, 4), 10)
    second = _source("float32", (1, 4), 11)
    _check(fn(mi.Array(first), second), op(first, second))
    _check(fn(first, mi.Array(second)), op(first, second))
    _check(fn(mi.Array(first), 2.0), op(first, np.float32(2.0)))


def test_divide_function():
    first = _source("uint8", (3, 4), 12)
    second = _source("uint8", (3, 4), 13)
    got = mi.divide(mi.Array(first), mi.Array(second))
    _check(got, first.astype(np.float32) / second.astype(np.float32), divide=True)
