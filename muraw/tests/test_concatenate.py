"""Channel joins: concatenate, stack, and dstack are one composite over a fill.

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
    np.testing.assert_array_equal(got._node.attrs["tops"], [0, 0])
    np.testing.assert_array_equal(got._node.attrs["lefts"], [0, 0])


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


def test_joining_rows_or_columns_raises():
    images = [mi.zeros((2, 3), np.float32), mi.zeros((2, 3), np.float32)]
    for axis in (0, -1):
        with pytest.raises(NotImplementedError, match="mi.stack"):
            mi.concatenate(images, axis=axis)
    rgb = [mi.zeros((2, 3, 3), np.float32)] * 2
    with pytest.raises(NotImplementedError, match="rows or columns"):
        mi.concatenate(rgb, axis=0)


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


def test_stack_rejects_an_rgb_input():
    with pytest.raises(NotImplementedError, match=r"\(H, W\)"):
        mi.stack([mi.zeros((2, 3, 3), np.float32)])


def test_dstack_promotes_a_row_and_joins_packed_channels():
    np.testing.assert_array_equal(_dstack_rows(mi).realize(), _dstack_rows(np))
    np.testing.assert_array_equal(_dstack_mixed(mi).realize(), _dstack_mixed(np))

    packed = [_image((3, 4, 3), np.uint8), _image((3, 4), np.uint8)]
    np.testing.assert_array_equal(_dstack(mi, packed).realize(), _dstack(np, packed))


def test_chw_inputs_along_the_last_axis_raise():
    chw = Array(mi.zeros((3, 2, 4), np.float32).realize(), channel_axis=0)
    with pytest.raises(NotImplementedError, match="rows or columns"):
        mi.concatenate([chw, chw], axis=-1)
    with pytest.raises(NotImplementedError, match="rows or columns"):
        mi.dstack([chw])


def test_an_empty_sequence_raises():
    for name in ("concatenate", "stack", "dstack"):
        for lib in (np, mi):
            with pytest.raises(ValueError, match="at least one"):
                getattr(lib, name)([])
