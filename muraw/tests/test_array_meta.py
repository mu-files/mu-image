# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files
"""ArrayMeta stores the NumPy shape; the buffer sizes are read from it."""

import numpy as np
import pytest

from muraw.array import Array, ArrayMeta, ElementType


def _meta(shape, channel_axis=None):
    return ArrayMeta(dtype=ElementType.FLOAT32, shape=shape, channel_axis=channel_axis)


@pytest.mark.parametrize(
    "shape, channel_axis, height, width, channels, is_1d, buffer_shape",
    [
        ((5,), None, 1, 5, 1, True, (1, 5)),
        ((2, 5), None, 2, 5, 1, False, (2, 5)),
        ((2, 5, 1), 2, 2, 5, 1, False, (2, 5, 1)),
        ((2, 5, 3), 2, 2, 5, 3, False, (2, 5, 3)),
        ((3, 2, 5), 0, 2, 5, 3, False, (3, 2, 5)),
        ((2, 3, 5), 1, 2, 5, 3, False, (2, 3, 5)),
    ],
)
def test_properties_follow_the_shape(
    shape, channel_axis, height, width, channels, is_1d, buffer_shape
):
    meta = _meta(shape, channel_axis)
    assert meta.shape == shape
    assert meta.ndim == len(shape)
    assert (meta.height, meta.width, meta.channels) == (height, width, channels)
    assert meta.channel_axis == channel_axis
    assert meta.is_1d is is_1d
    assert meta.buffer_shape == buffer_shape


def test_channel_axis_defaults_to_last_and_counts_negative_from_the_end():
    assert _meta((2, 5, 3)).channel_axis == 2
    assert _meta((2, 5, 3), -1).channel_axis == 2
    assert _meta((3, 2, 5), -3).channel_axis == 0


@pytest.mark.parametrize("channel_axis", [3, -4])
def test_rejects_a_channel_axis_out_of_range(channel_axis):
    with pytest.raises(np.exceptions.AxisError, match="channel_axis"):
        _meta((3, 2, 5), channel_axis)


@pytest.mark.parametrize("shape", [(5,), (2, 5)])
def test_rejects_a_channel_axis_below_three_axes(shape):
    with pytest.raises(ValueError, match="has no channel axis"):
        _meta(shape, 0)


@pytest.mark.parametrize("shape", [(0,), (0, 4), (3, 0), (2, 2, 0)])
def test_rejects_an_axis_below_one(shape):
    with pytest.raises(ValueError, match="at least 1|channel count"):
        _meta(shape)


@pytest.mark.parametrize("shape", [(), (2, 2, 2, 2)])
def test_rejects_a_rank_outside_one_to_three(shape):
    with pytest.raises(ValueError, match=r"\(N,\), \(H,W\) or 3 axes"):
        _meta(shape)


def test_with_size_keeps_the_rank():
    assert _meta((2, 5)).with_size(height=4).shape == (4, 5)
    assert _meta((2, 5, 1)).with_size(width=3).shape == (2, 3, 1)
    assert _meta((2, 5, 3)).with_size(channels=1).shape == (2, 5, 1)
    assert _meta((5,)).with_size(width=7).shape == (7,)


def test_with_size_keeps_the_channel_axis():
    planar = _meta((3, 2, 5), 0).with_size(height=4, width=7)
    assert (planar.shape, planar.channel_axis) == ((3, 4, 7), 0)
    middle = _meta((2, 3, 5), 1).with_size(channels=4)
    assert (middle.shape, middle.channel_axis) == ((2, 4, 5), 1)


def test_with_size_adds_a_channel_axis_for_more_than_one_channel():
    meta = _meta((2, 5)).with_size(channels=3)
    assert (meta.shape, meta.channel_axis) == ((2, 5, 3), 2)


def test_with_size_changes_the_rank():
    assert _meta((5,)).with_size(ndim=2).shape == (1, 5)
    assert _meta((1, 5)).with_size(ndim=1).shape == (5,)
    assert _meta((2, 5)).with_size(ndim=3).shape == (2, 5, 1)
    assert _meta((2, 5, 1)).with_size(ndim=2).shape == (2, 5)
    dropped = _meta((1, 2, 5), 0).with_size(ndim=2)
    assert (dropped.shape, dropped.channel_axis) == ((2, 5), None)


@pytest.mark.parametrize("changes", [{"height": 2}, {"channels": 3}])
def test_with_size_rejects_a_1d_array_with_more_than_one_row_or_channel(changes):
    with pytest.raises(ValueError, match="one row and one channel"):
        _meta((5,)).with_size(**changes)


@pytest.mark.parametrize(
    "arr",
    [
        np.arange(24, dtype=np.float32).reshape(4, 6)[:, ::2],
        np.arange(72, dtype=np.float32).reshape(4, 6, 3)[..., 1],
        np.arange(72, dtype=np.float32).reshape(4, 6, 3)[..., 1:2],
        np.arange(72, dtype=np.float32).reshape(4, 6, 3)[:, ::-1],
        np.arange(12, dtype=np.float32)[::-2],
    ],
    ids=["column_step", "channel_pick", "channel_slice", "column_flip", "flip_1d"],
)
def test_view_node_buffer_matches_the_ingest_meta(arr):
    array = Array(arr)
    assert array._node is not None
    assert array.meta.shape == arr.shape
    node_meta = array._node.out_meta
    assert (node_meta.height, node_meta.width, node_meta.channels) == (
        array.meta.height,
        array.meta.width,
        array.meta.channels,
    )
    assert node_meta.dtype == array.meta.dtype
    np.testing.assert_array_equal(array.realize(), arr)


def test_with_size_sets_other_fields():
    meta = _meta((2, 5)).with_size(height=3, dtype=ElementType.UINT16, origin=(1, 2))
    assert (meta.shape, meta.dtype, meta.origin) == ((3, 5), ElementType.UINT16, (1, 2))
