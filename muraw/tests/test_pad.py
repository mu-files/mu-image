"""Pad op: emit-meta and execute vs ``numpy.pad``."""

from __future__ import annotations

import numpy as np
import pytest

from muraw.array import Array


def test_pad_emit_meta_starts_its_own_canvas():
    src = Array(np.zeros((4, 6), dtype=np.float32), origin=(2, 3))
    t = src.pad(1)
    assert t.meta.height == 6 and t.meta.width == 8
    assert t.meta.origin == (0, 0)
    assert t.meta.canvas == (0, 0, 8, 6)
    assert t._node is not None and t._node.op == "pad"


def test_pad_width_shapes():
    src = Array(np.zeros((3, 5), dtype=np.float32))
    assert src.pad(2).meta.shape == (7, 9)
    assert src.pad((1, 3)).meta.shape == (7, 9)
    assert src.pad(((1, 0), (2, 4))).meta.shape == (4, 11)


@pytest.mark.parametrize(
    "mode,pad_width",
    [
        ("constant", 1),
        ("edge", 1),
        ("reflect", 1),
        ("symmetric", 1),
        ("constant", ((1, 2), (0, 3))),
    ],
)
def test_pad_matches_numpy(mode, pad_width):
    rng = np.random.default_rng(0)
    src = rng.random((5, 7), dtype=np.float32)
    t = Array(src).pad(pad_width, mode=mode)
    np_width = pad_width if isinstance(pad_width, tuple) else pad_width
    expect = np.pad(src, np_width, mode=mode)
    np.testing.assert_array_equal(t, expect)


def test_pad_rgb_spatial_only_with_a_zero_channel_pair():
    rng = np.random.default_rng(1)
    src = rng.random((4, 6, 3), dtype=np.float32)
    widths = ((1, 1), (1, 1), (0, 0))
    out = Array(src).pad(widths, mode="edge").realize()
    np.testing.assert_array_equal(out, np.pad(src, widths, mode="edge"))


@pytest.mark.parametrize("pad_width", [1, (1, 2)], ids=["int", "pair"])
def test_pad_rgb_shorthand_pads_every_axis_like_numpy(pad_width):
    rng = np.random.default_rng(2)
    src = rng.random((4, 6, 3), dtype=np.float32)
    out = Array(src).pad(pad_width, constant_values=0.5).realize()
    np.testing.assert_array_equal(out, np.pad(src, pad_width, constant_values=0.5))


def test_pad_rgb_shorthand_in_edge_mode_rejects_the_channel_pad():
    src = np.zeros((4, 6, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="channel pad requires mode 'constant'"):
        Array(src).pad(1, mode="edge")


def test_pad_constant_values():
    src = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    out = Array(src).pad(1, mode="constant", constant_values=9).realize()
    expect = np.pad(src, 1, mode="constant", constant_values=9)
    np.testing.assert_array_equal(out, expect)
    per_edge = (
        Array(src).pad(1, mode="constant", constant_values=((9, 8), (7, 6))).realize()
    )
    expect_edge = np.pad(src, 1, mode="constant", constant_values=((9, 8), (7, 6)))
    np.testing.assert_array_equal(per_edge, expect_edge)


def test_pad_appends_channel():
    rng = np.random.default_rng(2)
    src = rng.random((4, 5, 3), dtype=np.float32)
    width = ((0, 0), (0, 0), (0, 1))
    out = Array(src).pad(width, constant_values=255).realize()
    expect = np.pad(src, width, constant_values=255)
    np.testing.assert_array_equal(out, expect)
    assert out.shape == (4, 5, 4)


def test_pad_prepends_and_appends_channels():
    rng = np.random.default_rng(3)
    src = rng.random((3, 4, 3), dtype=np.float32)
    width = ((0, 0), (0, 0), (1, 2))
    constants = ((0, 0), (0, 0), (4, 9))
    out = Array(src).pad(width, constant_values=constants).realize()
    expect = np.pad(src, width, constant_values=constants)
    np.testing.assert_array_equal(out, expect)
    assert out.shape == (3, 4, 6)


def test_pad_channel_and_spatial_margin():
    rng = np.random.default_rng(4)
    src = rng.random((3, 4, 3), dtype=np.float32)
    width = ((1, 1), (2, 0), (0, 1))
    constants = ((0, 8), (7, 0), (0, 255))
    out = Array(src).pad(width, constant_values=constants).realize()
    expect = np.pad(src, width, constant_values=constants)
    np.testing.assert_array_equal(out, expect)


def test_pad_channel_emit_meta():
    src = Array(np.zeros((4, 6, 3), dtype=np.float32), origin=(2, 3))
    t = src.pad(((1, 0), (0, 2), (0, 1)))
    assert t.meta.height == 5 and t.meta.width == 8
    assert t.meta.channels == 4
    assert t.meta.shape == (5, 8, 4)
    assert t.meta.origin == (0, 0)
    assert t.meta.canvas == (0, 0, 8, 5)
    assert t.meta.channel_axis


def test_pad_channel_on_hwc1():
    src = np.arange(6, dtype=np.float32).reshape(2, 3, 1)
    width = ((0, 0), (0, 0), (0, 1))
    out = Array(src).pad(width, constant_values=1).realize()
    expect = np.pad(src, width, constant_values=1)
    np.testing.assert_array_equal(out, expect)
    assert out.shape == (2, 3, 2)


def test_pad_uint8_appends_channel():
    src = np.zeros((2, 2, 3), dtype=np.uint8)
    src[..., 0] = 10
    width = ((0, 0), (0, 0), (0, 1))
    out = Array(src).pad(width, constant_values=255).realize()
    expect = np.pad(src, width, constant_values=255)
    np.testing.assert_array_equal(out, expect)


def test_pad_channel_requires_constant_mode():
    src = np.zeros((2, 2, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="constant"):
        Array(src).pad(((0, 0), (0, 0), (0, 1)), mode="edge")


def test_pad_channel_width_rejected_on_mono():
    src = np.zeros((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match=r"pad_width: expected an int, a pair"):
        Array(src).pad(((0, 0), (0, 0), (0, 1)))


def _pad_each_channel(src, pad_width, channel_axis, values):
    """``src`` padded with ``values[c]`` around channel ``c``; NumPy's
    ``pad_width`` with ``(0, 0)`` on the channel axis."""
    widths = [w for axis, w in enumerate(pad_width) if axis != channel_axis]
    channels = [
        np.pad(src.take(c, axis=channel_axis), widths, constant_values=value)
        for c, value in enumerate(values)
    ]
    return np.stack(channels, axis=channel_axis)


@pytest.mark.parametrize(
    "shape,channel_axis,pad_width,values,dtype",
    [
        ((4, 5, 3), 2, ((1, 2), (3, 0), (0, 0)), (0.25, -1.0, 2.5), np.float32),
        ((4, 5, 3), 2, ((1, 1), (1, 1), (0, 0)), (255, 0, 7), np.uint8),
        ((4, 5, 4), 2, ((0, 1), (2, 2), (0, 0)), (1.0, 2.0, 3.0, 4.0), np.float32),
        ((3, 4, 5), 0, ((0, 0), (1, 1), (2, 0)), (9, 8, 7), np.uint16),
        (
            (2, 3, 4, 3),
            3,
            ((1, 0), (1, 1), (0, 2), (0, 0)),
            (5.0, 6.0, 7.0),
            np.float32,
        ),
        ((0, 4, 3), 2, ((1, 1), (0, 0), (0, 0)), (1.0, 2.0, 3.0), np.float32),
    ],
    ids=["rgb", "rgb_uint8", "rgba", "chw", "stack_with_frames", "empty"],
)
def test_pad_one_constant_per_channel_fills_each_channel(
    shape, channel_axis, pad_width, values, dtype
):
    rng = np.random.default_rng(5)
    src = (rng.random(shape) * 100).astype(dtype)
    out = Array(src, channel_axis=channel_axis).pad(pad_width, constant_values=values)
    expect = _pad_each_channel(src, pad_width, channel_axis, values)
    np.testing.assert_array_equal(np.asarray(out), expect)


@pytest.mark.parametrize(
    "shape,pad_width,constant_values",
    [
        ((4, 5, 2), ((1, 1), (2, 0), (0, 0)), (3.0, 4.0)),
        ((4, 5, 3), ((1, 1), (2, 0), (0, 0)), (3.0, 4.0)),
        ((4, 5, 3), 1, (3.0,)),
    ],
    ids=["two_channels_pair", "rgb_pair", "rgb_one_value"],
)
def test_pad_one_or_two_constants_keep_numpys_meaning(
    shape, pad_width, constant_values
):
    src = np.random.default_rng(6).random(shape, dtype=np.float32)
    out = Array(src).pad(pad_width, constant_values=constant_values)
    expect = np.pad(src, pad_width, constant_values=constant_values)
    np.testing.assert_array_equal(out.realize(), expect)


@pytest.mark.parametrize(
    "shape,pad_width",
    [
        ((4, 5, 3), ((1, 1), (0, 0), (0, 1))),
        ((4, 3), 1),
        ((3,), 1),
    ],
    ids=["channel_pad", "no_channel_axis", "one_axis"],
)
def test_pad_one_constant_per_channel_is_rejected_where_numpy_rejects_it(
    shape, pad_width
):
    src = np.zeros(shape, dtype=np.float32)
    with pytest.raises(ValueError):
        Array(src).pad(pad_width, constant_values=(1.0, 2.0, 3.0))
    with pytest.raises(ValueError):
        np.pad(src, pad_width, constant_values=(1.0, 2.0, 3.0))


def test_pad_then_interior_view():
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    t = Array(src, origin=(5, 7)).pad(((1, 1), (2, 2)))
    interior = t.view(left=2, top=1, width=4, height=3, mode="crop_oob_invalid")
    assert interior.meta.origin == (1, 2)
    np.testing.assert_array_equal(interior.realize(), src)


def test_pad_starts_a_canvas_of_its_own():
    """A pad is a new image: origin (0, 0), and the padded array is its
    canvas, whatever the input's origin and canvas."""
    src = np.zeros((40, 60), dtype=np.float32)
    sliced = Array(src, origin=(5, 7))[3:10, 4:20]
    padded = sliced.pad(((1, 2), (3, 4)))
    assert padded.meta.origin == (0, 0)
    assert padded.meta.canvas == (0, 0, 23, 10)
    assert padded.meta.oob_valid
