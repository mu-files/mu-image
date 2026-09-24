"""Pad op: emit-meta and execute vs ``numpy.pad``."""

from __future__ import annotations

import numpy as np
import pytest

from muraw.array import Array


def test_pad_emit_meta_grows_canvas_at_origin():
    src = Array(np.zeros((4, 6), dtype=np.float32), origin=(2, 3))
    t = src.pad(1)
    assert t.meta.height == 6 and t.meta.width == 8
    assert t.meta.origin == (1, 2)
    assert t.meta.canvas == (2, 1, 8, 6)
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


def test_pad_rgb_does_not_pad_channels():
    rng = np.random.default_rng(1)
    src = rng.random((4, 6, 3), dtype=np.float32)
    out = Array(src).pad(1, mode="edge").realize()
    expect = np.pad(src, ((1, 1), (1, 1), (0, 0)), mode="edge")
    np.testing.assert_array_equal(out, expect)


def test_pad_constant_values():
    src = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    out = Array(src).pad(1, mode="constant", constant_values=9).realize()
    expect = np.pad(src, 1, mode="constant", constant_values=9)
    np.testing.assert_array_equal(out, expect)
    per_edge = Array(src).pad(1, mode="constant", constant_values=((9, 8), (7, 6))).realize()
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
    assert t.meta.origin == (1, 3)
    assert t.meta.canvas == (3, 1, 8, 5)
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
    with pytest.raises(ValueError, match="3 axes"):
        Array(src).pad(((0, 0), (0, 0), (0, 1)))


def test_pad_then_interior_view():
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    t = Array(src, origin=(5, 7)).pad(((1, 1), (2, 2)))
    interior = t.view(left=2, top=1, width=4, height=3, oob_valid=False)
    assert interior.meta.origin == (5, 7)
    np.testing.assert_array_equal(interior.realize(), src)
