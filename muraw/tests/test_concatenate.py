"""Joins: concatenate, stack, vstack, hstack, and dstack are one composite
over a fill, along the channels, the rows, or the columns.

Value checks call the same function with ``mi`` and with ``np``.
"""

from __future__ import annotations

import numpy as np
import pytest

import muimage as mi
from muraw.array import Array


def _image(shape, dtype):
    rng = np.random.default_rng(7)
    if np.dtype(dtype).kind == "f":
        return rng.random(shape).astype(dtype)
    return rng.integers(0, 256, shape).astype(dtype)


def _concat_channels(lib, arrays):
    """Join ``arrays`` along their last axis."""
    return lib.concatenate(arrays, axis=-1)


def _concat_mixed(lib):
    """uint8 ``255`` and float32 ``1.5`` joined on the last axis."""
    rgb = lib.full((2, 3, 3), 255, np.uint8)
    mono = lib.full((2, 3, 1), 1.5, np.float32)
    return lib.concatenate([rgb, mono], axis=-1)


def _concat_axis0(lib, first, second):
    """Join ``first`` and ``second`` along axis 0."""
    return lib.concatenate([first, second], axis=0)


def _stack_axis0(lib, planes):
    """Stack ``(H, W)`` planes on a new axis 0."""
    return lib.stack(planes, axis=0)


def _stack(lib, planes):
    """``lib.stack`` with its default axis."""
    return lib.stack(planes)


def _stack_bgr(lib):
    """Restack an RGB image into BGR."""
    rgb = lib.arange(24, dtype=np.float32).reshape(2, 4, 3)
    return lib.stack([rgb[..., 2], rgb[..., 1], rgb[..., 0]], axis=-1)


def _dstack(lib, arrays):
    """``lib.dstack`` of ``arrays``."""
    return lib.dstack(arrays)


def _dstack_rows(lib):
    """Two copies of one ``arange`` row, the second shifted by 1."""
    row = lib.arange(4, dtype=np.float32)
    return lib.dstack([row, row + 1])


def _dstack_mixed(lib):
    """A float32 row and a uint8 row of shape ``(1, 4)``."""
    row = lib.arange(4, dtype=np.float32)
    one_row = lib.expand_dims(lib.arange(4, dtype=np.uint8), 0)
    return lib.dstack([row, one_row])


def _mismatched_channels(lib):
    """Two packed images whose heights differ."""
    images = [lib.zeros((2, 3, 1), np.float32), lib.zeros((4, 3, 1), np.float32)]
    return lib.concatenate(images, axis=-1)


def _mismatched_rows(lib):
    """Two rows whose lengths differ."""
    return lib.dstack([lib.zeros(3, np.float32), lib.zeros(4, np.float32)])


def _different_ranks(lib):
    """A ``(H, W)`` image and a ``(H, W, C)`` image."""
    images = [lib.zeros((2, 3), np.float32), lib.zeros((2, 3, 3), np.float32)]
    return lib.concatenate(images)


def test_concatenate_is_a_composite_over_a_fill():
    mono = mi.zeros((2, 3, 1), np.float32) * 2
    rgb = mi.full((2, 3, 3), 1, np.float32)
    assert mono._data is None
    got = _concat_channels(mi, [mono, rgb])
    assert mono._data is None
    assert got.shape == (2, 3, 4)
    assert got.dtype == np.float32
    assert got.meta.origin == (0, 0)
    assert got.meta.canvas == (0, 0, 3, 2)
    assert got._node.op == "composite"
    base, first, second = got._node.inputs
    assert base._node.op == "fill"
    assert base._node.inputs == ()
    assert first is mono
    assert second is rgb
    np.testing.assert_array_equal(got._node.attrs["dest_channels"], [0, 1, 2, 3])
    np.testing.assert_array_equal(got._node.attrs["dest_channels_source"], [0, 1])
    np.testing.assert_array_equal(got._node.attrs["by_position"], [0, 0, 0, 0])
    np.testing.assert_array_equal(got._node.attrs["by_position_source"], [0, 1])


def test_a_mixed_dtype_concatenate_casts_the_inputs_that_differ():
    got = _concat_mixed(mi)
    assert got.dtype == np.float32
    base, cast, same = got._node.inputs
    assert base._node.op == "fill"
    assert cast._node.op == "cast_dtype"
    assert cast._node.inputs[0]._node.op == "fill"
    assert cast._node.inputs[0].dtype == np.uint8
    assert same._node.op == "fill"
    assert same.dtype == np.float32
    np.testing.assert_array_equal(got.realize(), _concat_mixed(np))


def test_concatenate_resets_origin_and_canvas():
    src = Array(mi.zeros((2, 3, 1), np.float32).realize(), origin=(4, 5))
    assert src.meta.origin == (4, 5)
    got = _concat_channels(mi, [src, src])
    assert got.meta.origin == (0, 0)
    assert got.meta.canvas == (0, 0, 3, 2)


def test_concatenate_mono_and_rgb_match_numpy():
    for dtype in (np.uint8, np.float32):
        mono = _image((4, 5, 1), dtype)
        rgb = _image((4, 5, 3), dtype)
        arrays = [mono, rgb, mono]
        np.testing.assert_array_equal(
            _concat_channels(mi, arrays).realize(), _concat_channels(np, arrays)
        )


def test_concatenate_chw_along_axis_0_matches_numpy():
    first = _image((2, 4, 5), np.uint8)
    second = _image((3, 4, 5), np.uint8)
    got = _concat_axis0(mi, Array(first, channel_axis=0), Array(second, channel_axis=0))
    assert got.meta.channel_axis == 0
    np.testing.assert_array_equal(got.realize(), _concat_axis0(np, first, second))


def test_a_height_or_width_mismatch_raises():
    with pytest.raises(ValueError):
        _mismatched_channels(np)
    with pytest.raises(ValueError, match="height and width"):
        _mismatched_channels(mi)
    with pytest.raises(ValueError):
        _mismatched_rows(np)
    with pytest.raises(ValueError, match="height and width"):
        _mismatched_rows(mi)


def test_concatenate_of_a_different_number_of_axes_raises():
    with pytest.raises(ValueError):
        _different_ranks(np)
    with pytest.raises(ValueError, match="number of axes"):
        _different_ranks(mi)


def _concat(lib, arrays, axis):
    """``lib.concatenate`` of ``arrays`` along ``axis``."""
    return lib.concatenate(arrays, axis=axis)


def _chw(array):
    """``array`` as an ``mi`` input with its channels on axis 0."""
    return Array(array, channel_axis=0)


def _i420(lib, y, u, v):
    """Planar 4:2:0: the Y plane, then the U and V planes, each read as
    rows of Y's width."""
    height, width = y.shape
    rows = height // 4
    return lib.concatenate(
        [y, lib.reshape(u, (rows, width)), lib.reshape(v, (rows, width))]
    )


def _nv12(lib, y, u, v):
    """The Y plane, then one half-height plane of interleaved U and V."""
    height, width = y.shape
    return lib.concatenate([y, lib.reshape(lib.dstack([u, v]), (height // 2, width))])


def test_a_row_join_is_a_composite_at_the_running_heights():
    top = mi.zeros((2, 4, 3), np.float32) * 2
    bottom = mi.full((3, 4, 3), 1, np.float32)
    got = _concat(mi, [top, bottom], 0)
    assert got.shape == (5, 4, 3)
    assert got.meta.origin == (0, 0)
    assert got.meta.canvas == (0, 0, 4, 5)
    assert got._node.op == "composite"
    base, first, second = got._node.inputs
    assert base._node.op == "fill"
    assert first is top
    assert second is bottom
    assert "dest_channels" not in got._node.attrs
    np.testing.assert_array_equal(got._node.attrs["by_position"], [0, 0, 2, 0])


def test_a_column_join_is_a_composite_at_the_running_widths():
    got = _concat(mi, [mi.zeros((2, 3), np.uint8), mi.zeros((2, 5), np.uint8)], 1)
    assert got.shape == (2, 8)
    np.testing.assert_array_equal(got._node.attrs["by_position"], [0, 0, 0, 3])


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
@pytest.mark.parametrize(
    "shapes, axis",
    [
        ([(2, 5), (3, 5)], 0),
        ([(3, 2), (3, 4)], 1),
        ([(3, 2), (3, 4)], -1),
        ([(2, 4, 3), (3, 4, 3)], 0),
        ([(2, 4, 3), (2, 1, 3)], 1),
        ([(2, 4, 1), (2, 2, 1), (2, 3, 1)], 1),
    ],
)
def test_rows_and_columns_match_numpy(shapes, axis, dtype):
    arrays = [_image(shape, dtype) for shape in shapes]
    np.testing.assert_array_equal(
        _concat(mi, arrays, axis).realize(), _concat(np, arrays, axis)
    )


@pytest.mark.parametrize(
    "shapes, axis", [([(3, 2, 4), (3, 5, 4)], 1), ([(3, 2, 4), (3, 2, 1)], -1)]
)
def test_rows_and_columns_of_chw_match_numpy(shapes, axis):
    arrays = [_image(shape, np.uint8) for shape in shapes]
    got = _concat(mi, [_chw(array) for array in arrays], axis)
    assert got.meta.channel_axis == 0
    np.testing.assert_array_equal(got.realize(), _concat(np, arrays, axis))


def test_a_1d_concatenate_matches_numpy():
    arrays = [_image((4,), np.uint8), _image((3,), np.uint8)]
    np.testing.assert_array_equal(
        _concat(mi, arrays, 0).realize(), _concat(np, arrays, 0)
    )


def test_a_mixed_dtype_row_join_matches_numpy():
    arrays = [_image((2, 5), np.float32), _image((1, 5), np.uint8)]
    got = _concat(mi, arrays, 0)
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got.realize(), _concat(np, arrays, 0))


def test_a_lazy_row_join_matches_numpy():
    image = _image((2, 3), np.float32)
    lazy = mi.Array(image) + 1
    np.testing.assert_array_equal(
        _concat(mi, [lazy, lazy], 0).realize(), _concat(np, [image + 1, image + 1], 0)
    )


def test_a_mismatch_on_another_axis_raises():
    arrays = [_image((2, 3), np.uint8), _image((2, 4), np.uint8)]
    with pytest.raises(ValueError):
        _concat(np, arrays, 0)
    with pytest.raises(ValueError, match="every axis but axis 0"):
        _concat(mi, arrays, 0)


def test_inputs_with_their_channels_on_different_axes_raise():
    image = _image((3, 2, 4), np.uint8)
    with pytest.raises(NotImplementedError, match="moveaxis"):
        mi.concatenate([_chw(image), image], axis=1)


def test_i420_and_nv12_assemble_in_one_graph():
    y = _image((8, 12), np.uint8)
    u = _image((4, 6), np.uint8)
    v = _image((4, 6), np.uint8)
    np.testing.assert_array_equal(_i420(mi, y, u, v).realize(), _i420(np, y, u, v))
    np.testing.assert_array_equal(_nv12(mi, y, u, v).realize(), _nv12(np, y, u, v))


def test_stack_along_axis_0_matches_numpy():
    planes = [_image((3, 4), np.float32) for _ in range(3)]
    got = _stack_axis0(mi, planes)
    assert got.meta.channel_axis == 0
    np.testing.assert_array_equal(got.realize(), _stack_axis0(np, planes))


def test_stack_along_axis_1_names_moveaxis():
    images = [mi.zeros((2, 3), np.float32)] * 2
    with pytest.raises(NotImplementedError, match="moveaxis"):
        mi.stack(images, axis=1)


def test_an_ndarray_is_a_sequence_of_its_first_axis():
    planes = _image((3, 4, 5), np.uint8)
    np.testing.assert_array_equal(_stack(mi, planes).realize(), _stack(np, planes))


def test_stack_restacks_rgb_as_bgr():
    np.testing.assert_array_equal(_stack_bgr(mi).realize(), _stack_bgr(np))


@pytest.mark.parametrize("axis", [0, 1, -1])
def test_a_stack_of_1d_arrays_matches_numpy(axis):
    rows = [_image((4,), np.uint8) for _ in range(3)]
    np.testing.assert_array_equal(
        mi.stack(rows, axis=axis).realize(), np.stack(rows, axis=axis)
    )


def test_a_stack_of_different_shapes_raises():
    rows = [_image((3,), np.uint8), _image((4,), np.uint8)]
    with pytest.raises(ValueError):
        np.stack(rows)
    with pytest.raises(ValueError, match="same shape"):
        mi.stack(rows)


def test_vstack_and_hstack_match_numpy():
    row = _image((4,), np.uint8)
    rows = _image((2, 4), np.uint8)
    np.testing.assert_array_equal(
        mi.vstack([row, rows]).realize(), np.vstack([row, rows])
    )
    np.testing.assert_array_equal(
        mi.hstack([row, row]).realize(), np.hstack([row, row])
    )
    left, right = _image((2, 3), np.uint8), _image((2, 4), np.uint8)
    np.testing.assert_array_equal(
        mi.hstack([left, right]).realize(), np.hstack([left, right])
    )
    rgb = [_image((2, 3, 3), np.uint8), _image((1, 3, 3), np.uint8)]
    np.testing.assert_array_equal(mi.vstack(rgb).realize(), np.vstack(rgb))


def test_stack_rejects_an_rgb_input():
    with pytest.raises(NotImplementedError, match=r"\(H, W\)"):
        mi.stack([mi.zeros((2, 3, 3), np.float32)])


def test_dstack_promotes_a_row_and_joins_packed_channels():
    np.testing.assert_array_equal(_dstack_rows(mi).realize(), _dstack_rows(np))
    np.testing.assert_array_equal(_dstack_mixed(mi).realize(), _dstack_mixed(np))

    packed = [_image((3, 4, 3), np.uint8), _image((3, 4), np.uint8)]
    np.testing.assert_array_equal(_dstack(mi, packed).realize(), _dstack(np, packed))


def test_dstack_of_chw_inputs_joins_their_columns():
    arrays = [_image((3, 2, 4), np.uint8), _image((3, 2, 1), np.uint8)]
    got = mi.dstack([_chw(array) for array in arrays])
    assert got.meta.channel_axis == 0
    np.testing.assert_array_equal(got.realize(), np.dstack(arrays))


def test_dstack_of_a_chw_input_and_a_2d_input_names_moveaxis():
    chw = _chw(_image((3, 2, 4), np.uint8))
    with pytest.raises(NotImplementedError, match="moveaxis"):
        mi.dstack([chw, _image((3, 2), np.uint8)])


def test_an_empty_sequence_raises():
    for name in ("concatenate", "stack", "vstack", "hstack", "dstack"):
        for lib in (np, mi):
            with pytest.raises(ValueError, match="at least one"):
                getattr(lib, name)([])
