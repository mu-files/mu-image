"""4-axis (N, H, W, C) stacks: ingest, slicing, broadcasting, and ops,
each checked against NumPy."""

from __future__ import annotations

import numpy as np
import pytest

import muimage as mi
from muraw.array import Array


def _stack(frames: int = 3, height: int = 6, width: int = 8, channels: int = 3) -> np.ndarray:
    shape = (frames, height, width, channels)
    return np.arange(np.prod(shape), dtype=np.float32).reshape(shape)


def _ops(array: Array) -> list[tuple[str, dict]]:
    found = []
    current = array
    while current._node is not None:
        found.append((current._node.op, dict(current._node.attrs)))
        current = current._node.inputs[0]
    return found


def test_a_contiguous_stack_is_bound_without_a_copy():
    stack = _stack()
    array = Array(stack)
    assert array.shape == stack.shape
    assert array.meta.channel_axis == 3
    assert array.meta.frames == 3
    assert np.shares_memory(array._data, stack)
    np.testing.assert_array_equal(array.realize(), stack)
    np.testing.assert_array_equal((array + 1).realize(), stack + 1)


def test_a_stack_with_padded_rows_and_frames_is_bound():
    parent = np.zeros((3, 7, 10, 3), dtype=np.float32)
    parent[:] = _stack(3, 7, 10, 3)
    stack = parent[:, :6, :8]
    array = Array(stack)
    assert np.shares_memory(array._data, parent)
    np.testing.assert_array_equal((array * 2).realize(), stack * 2)


def test_a_stack_with_strided_pixels_is_copied():
    stack = _stack(3, 6, 16)[:, :, ::2]
    array = Array(stack)
    assert not np.shares_memory(array._data, stack)
    np.testing.assert_array_equal((array - 1).realize(), stack - 1)


def test_channels_on_axis_0_of_4_axes_raise():
    with pytest.raises(ValueError, match="axis 0 of a 4-axis array holds its frames"):
        Array(_stack(), channel_axis=0)


@pytest.mark.parametrize(
    "key",
    [
        1,
        -1,
        (slice(None), slice(1, 4)),
        (slice(None), slice(None), slice(2, 7, 2)),
        (slice(0, 3, 2),),
        ([2, 0],),
        (slice(None, None, -1),),
        (Ellipsis, 0),
        (Ellipsis, [2, 1]),
        (1, Ellipsis, 2),
        (slice(1, 3), slice(0, 4), slice(1, 5), [0, 2]),
        (None, 1),
    ],
    ids=[
        "frame",
        "last_frame",
        "rows",
        "stepped_columns",
        "stepped_frames",
        "frame_list",
        "reversed_frames",
        "drop_channel",
        "channel_list",
        "frame_and_channel",
        "every_axis",
        "new_axis",
    ],
)
def test_slicing_a_stack_matches_numpy(key):
    stack = _stack()
    expected = stack[key]
    concrete = Array(stack)[key]
    lazy = (Array(stack) + 0)[key]
    assert concrete.shape == expected.shape
    assert lazy.shape == expected.shape
    np.testing.assert_array_equal(concrete.realize(), expected)
    np.testing.assert_array_equal(lazy.realize(), expected)


def test_a_stack_without_its_channel_axis_is_chw_frames():
    stack = _stack()
    mono = (Array(stack) * 1)[..., 1]
    assert mono.shape == (3, 6, 8)
    assert mono.meta.channel_axis == 0
    np.testing.assert_array_equal((mono + 1).realize(), stack[..., 1] + 1)


def test_one_frame_of_a_stack_is_an_image():
    stack = _stack() / 1000
    frame = (Array(stack) * 1)[2]
    assert frame.meta.channel_axis == 2
    np.testing.assert_array_equal(
        mi.rgb_matrix_3x3(frame, matrix=np.eye(3, dtype=np.float32)).realize(), stack[2]
    )


def test_a_frame_index_out_of_range_raises():
    with pytest.raises(IndexError):
        Array(_stack())[[0, 3]]


@pytest.mark.parametrize(
    "other",
    [
        np.arange(6 * 8 * 3, dtype=np.float32).reshape(6, 8, 3),
        np.array([1.0, 2.0, 3.0], dtype=np.float32).reshape(1, 1, 1, 3),
        np.array([1.0, 2.0, 3.0], dtype=np.float32).reshape(3, 1, 1, 1),
        np.arange(3 * 6 * 8, dtype=np.float32).reshape(3, 6, 8, 1),
    ],
    ids=["image", "per_channel", "per_frame", "per_pixel_mono"],
)
def test_binary_ops_broadcast_like_numpy(other):
    stack = _stack()
    np.testing.assert_array_equal(
        (Array(stack) + Array(other)).realize(), stack + other
    )
    np.testing.assert_array_equal(
        (Array(other) * Array(stack)).realize(), other * stack
    )


def test_per_frame_scalars_send_one_value_per_frame():
    stack = _stack()
    scale = np.array([1.0, 2.0, 3.0], dtype=np.float32).reshape(3, 1, 1, 1)
    out = Array(stack) * scale
    op, attrs = _ops(out)[0]
    assert op == "mul_scalar"
    assert attrs.get("per_frame")
    np.testing.assert_array_equal(out.realize(), stack * scale)


def test_per_channel_scalars_on_a_stack():
    stack = _stack()
    offset = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    out = Array(stack) + offset
    op, attrs = _ops(out)[0]
    assert op == "add_scalar"
    assert "per_frame" not in attrs
    np.testing.assert_array_equal(out.realize(), stack + offset)


def test_rgb_matrix_and_matmul_on_a_stack():
    stack = _stack() / 1000
    matrix = np.array(
        [[0.5, 0.25, 0.25], [0.0, 1.0, 0.0], [0.1, 0.2, 0.7]], dtype=np.float32
    )
    np.testing.assert_allclose(
        mi.rgb_matrix_3x3(Array(stack), matrix=matrix).realize(),
        stack @ matrix.T,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        (Array(stack) @ matrix.T).realize(), stack @ matrix.T, rtol=1e-6
    )


@pytest.mark.parametrize(
    "reps",
    [(2, 1, 1, 1), (1, 2, 3, 1), (2, 1, 1, 2), 2],
    ids=["frames", "spatial", "frames_and_channels", "last_axis"],
)
def test_tile_of_a_stack_matches_numpy(reps):
    stack = _stack(2, 3, 4, 3)
    np.testing.assert_array_equal(mi.tile(stack, reps).realize(), np.tile(stack, reps))


def test_a_broadcast_stack_is_tiled_on_ingest():
    frame = _stack(1, 3, 4, 3)
    stretched = np.broadcast_to(frame, (5, 3, 4, 3))
    array = Array(stretched)
    assert [op for op, _ in _ops(array)] == ["tile"]
    np.testing.assert_array_equal(array.realize(), stretched)
    np.testing.assert_array_equal(
        mi.broadcast_to(frame, (5, 3, 4, 3)).realize(), stretched
    )


@pytest.mark.parametrize(
    ("pad_width", "mode", "constant_values"),
    [
        (((0, 0), (1, 2), (3, 0), (0, 0)), "constant", 7.0),
        (((1, 2), (0, 0), (0, 0), (0, 0)), "constant", ((5.0, 6.0), (0, 0), (0, 0), (0, 0))),
        (((1, 1), (1, 0), (0, 1), (0, 0)), "edge", 0),
        (((0, 0), (0, 0), (0, 0), (1, 1)), "constant", 9.0),
        (((1, 0), (0, 0), (0, 0), (1, 0)), "constant", 4.0),
        (
            ((1, 2), (1, 0), (0, 1), (1, 1)),
            "constant",
            ((1.0, 2.0), (3.0, 4.0), (5.0, 6.0), (7.0, 8.0)),
        ),
    ],
    ids=[
        "spatial",
        "frames_constant",
        "frames_edge",
        "channels",
        "frames_and_channels",
        "every_axis_own_constants",
    ],
)
def test_pad_of_a_stack_matches_numpy(pad_width, mode, constant_values):
    stack = _stack(2, 3, 4, 3)
    kwargs = {"constant_values": constant_values} if mode == "constant" else {}
    expected = np.pad(stack, pad_width, mode=mode, **kwargs)
    out = Array(stack).pad(pad_width, mode=mode, **kwargs)
    assert out.shape == expected.shape
    np.testing.assert_array_equal(out.realize(), expected)


def test_expand_dims_of_an_image_is_a_stack():
    image = _stack(1, 3, 4, 3)[0]
    out = mi.expand_dims(Array(image) * 1, 0)
    assert out.shape == (1, 3, 4, 3)
    np.testing.assert_array_equal((out + Array(_stack(2, 3, 4, 3))).realize(), image + _stack(2, 3, 4, 3))


def test_swapping_the_rows_and_columns_of_a_stack_matches_numpy():
    stack = _stack()
    out = mi.swapaxes(Array(stack) * 1, 1, 2)
    assert out.shape == (3, 8, 6, 3)
    np.testing.assert_array_equal(out.realize(), np.swapaxes(stack, 1, 2))


def test_unsupported_stack_ops_raise():
    stack = Array(_stack())
    with pytest.raises(NotImplementedError, match="4-axis stack"):
        mi.transpose(stack, (2, 1, 0, 3))
    with pytest.raises(NotImplementedError, match="composite of a 4-axis stack"):
        mi.composite(stack, ((0, 0), Array(_stack(1, 2, 2, 3))))
    with pytest.raises(NotImplementedError, match="4-axis stack"):
        Array(_stack(channels=1), channel_axis=1) + 1
