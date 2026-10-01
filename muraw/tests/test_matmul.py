"""``x @ B`` between an Array and a small constant matrix."""

from __future__ import annotations

import numpy as np
import pytest

import muimage as mi


def _image(dtype: str, channels: int, shape: tuple[int, ...] = (5, 9)) -> np.ndarray:
    rng = np.random.default_rng(3)
    full_shape = shape + (channels,)
    if dtype == "uint8":
        return rng.integers(0, 256, full_shape).astype(np.uint8)
    if dtype == "uint16":
        return rng.integers(0, 65536, full_shape).astype(np.uint16)
    return rng.uniform(-2.0, 2.0, full_shape).astype(dtype)


def _matrix(input_channels: int, output_channels: int) -> np.ndarray:
    rng = np.random.default_rng(input_channels * 100 + output_channels)
    return rng.uniform(-1.0, 1.0, (input_channels, output_channels)).astype(np.float32)


def _check_close(got: mi.Array, want: np.ndarray) -> None:
    assert got.shape == want.shape
    assert got.dtype == mi.ElementType(want.dtype)
    tolerance = 2e-3 if want.dtype == np.float16 else 1e-5
    np.testing.assert_allclose(
        np.asarray(got).astype(np.float32),
        want.astype(np.float32),
        rtol=tolerance,
        atol=tolerance,
    )


@pytest.mark.parametrize("dtype", ["float16", "float32"])
@pytest.mark.parametrize("shape", [(3, 3), (4, 4), (4, 3), (3, 1), (1, 3), (32, 32)])
def test_float_image_matches_numpy(dtype: str, shape: tuple[int, int]) -> None:
    image = _image(dtype, shape[0])
    matrix = _matrix(*shape).astype(dtype)
    _check_close(mi.Array(image) @ matrix, image @ matrix)


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_integer_matrix_keeps_the_dtype_and_saturates(dtype: str) -> None:
    image = _image(dtype, 4)
    matrix = np.array([[1, 0, 0], [0, 2, 0], [0, 0, 1], [1, -1, 0]])
    want = image.astype(np.float32) @ matrix.astype(np.float32)
    want = np.clip(np.round(want), 0, np.iinfo(dtype).max).astype(dtype)
    got = mi.Array(image) @ matrix
    assert got.dtype == mi.ElementType(dtype)
    np.testing.assert_array_equal(np.asarray(got), want)


@pytest.mark.parametrize("dtype", ["uint8", "uint16"])
def test_float_matrix_makes_an_integer_image_float32(dtype: str) -> None:
    image = _image(dtype, 3)
    matrix = _matrix(3, 3)
    _check_close(mi.Array(image) @ matrix, image.astype(np.float32) @ matrix)


def test_a_1d_matrix_drops_the_channel_axis() -> None:
    image = _image("float32", 3)
    weights = [0.299, 0.587, 0.114]
    got = mi.Array(image) @ weights
    assert got.shape == (5, 9)
    _check_close(got, image @ np.float32(weights))


def test_a_list_matrix_and_mi_matmul_match_the_operator() -> None:
    image = _image("float32", 3)
    matrix = _matrix(3, 3)
    want = image @ matrix
    _check_close(mi.matmul(mi.Array(image), matrix.tolist()), want)
    _check_close(mi.matmul(image, matrix), want)


def test_a_concrete_mi_array_can_be_the_matrix() -> None:
    image = _image("float32", 3)
    matrix = _matrix(3, 3)
    _check_close(mi.Array(image) @ mi.Array(matrix), image @ matrix)
    _check_close(image @ mi.Array(matrix), image @ matrix)


def test_a_lazy_image_is_transformed() -> None:
    image = _image("float32", 4)
    matrix = _matrix(4, 3)
    got = (mi.Array(image) * 2.0) @ matrix
    _check_close(got, (image * np.float32(2.0)) @ matrix)


def test_2d_and_1d_arrays_are_rows_of_channels() -> None:
    rows = _image("float32", 4, shape=(6,))
    matrix = _matrix(4, 3)
    _check_close(mi.Array(rows) @ matrix, rows @ matrix)
    vector = rows[0]
    _check_close(mi.Array(vector) @ matrix, vector @ matrix)


def test_matrix_times_a_1d_array() -> None:
    vector = _image("float32", 4, shape=())
    matrix = _matrix(3, 4)
    _check_close(matrix @ mi.Array(vector), matrix @ vector)


def test_a_mismatched_matrix_raises_numpy_value_error() -> None:
    image = _image("float32", 3)
    with pytest.raises(ValueError, match="mismatch in its core dimension 0"):
        mi.Array(image) @ _matrix(4, 3)


def test_a_scalar_operand_raises_numpy_value_error() -> None:
    with pytest.raises(ValueError, match="does not have enough dimensions"):
        mi.Array(_image("float32", 3)) @ 2.0


@pytest.mark.parametrize("shape", [(33, 3), (3, 33)])
def test_more_than_32_channels_is_not_implemented(shape: tuple[int, int]) -> None:
    image = _image("float32", shape[0])
    with pytest.raises(NotImplementedError, match="np.matmul"):
        mi.Array(image) @ _matrix(*shape)


@pytest.mark.parametrize("channel_axis", [0, 1])
def test_channels_not_on_the_last_axis_is_not_implemented(channel_axis: int) -> None:
    planes = np.moveaxis(_image("float32", 3), -1, channel_axis)
    image = mi.Array(planes, channel_axis=channel_axis)
    with pytest.raises(NotImplementedError, match="channel_axis"):
        image @ _matrix(image.shape[-1], 3)


def test_a_lazy_matrix_is_not_implemented() -> None:
    matrix = mi.Array(_matrix(3, 3)) * 2.0
    with pytest.raises(NotImplementedError, match="realize"):
        mi.Array(_image("float32", 3)) @ matrix


def test_matrix_times_an_image_is_not_implemented() -> None:
    image = mi.Array(_image("float32", 3))
    with pytest.raises(NotImplementedError, match="second-to-last axis"):
        _matrix(9, 5) @ image


def test_two_1d_arrays_are_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="np.dot"):
        mi.Array(np.ones(3, np.float32)) @ [1.0, 2.0, 3.0]
