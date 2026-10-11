"""``composite(base, (origin, source), ...)`` against NumPy slice assignment.

``origin`` is ``(row, col)``, or ``(col,)`` on a 1D base. A placement may
end with the dest channel of each of that source's channels.
"""

from __future__ import annotations

import numpy as np
import pytest

import muimage as mi
from muraw.array import Array

_RNG = np.random.default_rng(4)


def _image(shape, dtype):
    if np.dtype(dtype).kind == "f":
        return _RNG.random(shape).astype(dtype)
    return _RNG.integers(0, 256, shape).astype(dtype)


def test_mono_and_rgb_match_slice_assignment():
    for dtype in (np.uint8, np.float32):
        mono_base = _image((6, 8), dtype)
        mono_src = _image((3, 4), dtype)
        mono = mono_base.copy()
        mono[1:4, 2:6] = mono_src
        got = mi.composite(mono_base, ((1, 2), mono_src)).realize()
        np.testing.assert_array_equal(got, mono)

        rgb_base = _image((6, 8, 3), dtype)
        rgb_src = _image((3, 4, 3), dtype)
        rgb = rgb_base.copy()
        rgb[1:4, 2:6] = rgb_src
        got = Array(rgb_base).composite(((1, 2), rgb_src)).realize()
        np.testing.assert_array_equal(got, rgb)


def test_a_later_source_overwrites_the_overlap():
    base = np.zeros((6, 8), np.float32)
    expected = base.copy()
    expected[1:5, 1:5] = 1
    expected[2:5, 3:6] = 2
    got = Array(base).composite(
        ((1, 1), np.full((4, 4), 1, np.float32)),
        ((2, 3), np.full((3, 3), 2, np.float32)),
    ).realize()
    np.testing.assert_array_equal(got, expected)


def test_a_source_covering_the_base_replaces_it():
    base = _image((4, 5, 3), np.float32)
    src = _image((4, 5, 3), np.float32)
    got = Array(base).composite(((0, 0), src)).realize()
    np.testing.assert_array_equal(got, src)


def test_a_view_and_a_lazy_source_match_their_pixels():
    other = _image((8, 10, 4), np.uint8)
    base = np.zeros((4, 5, 3), np.uint8)
    viewed = other[:4, :5, :3]
    expected = base.copy()
    expected[:] = viewed
    np.testing.assert_array_equal(Array(base).composite(((0, 0), viewed)).realize(), expected)

    strided = other[::2, ::-1, :3]
    canvas = np.zeros(strided.shape, np.uint8)
    canvas[:] = strided
    got = Array(np.zeros_like(canvas)).composite(((0, 0), strided)).realize()
    np.testing.assert_array_equal(got, canvas)

    patch = np.arange(12, dtype=np.float32).reshape(3, 4)
    lazy_base = np.ones((5, 6), np.float32)
    lazy = lazy_base.copy()
    lazy[1:4, 1:5] = patch * np.float32(2)
    got = Array(lazy_base).composite(((1, 1), Array(patch) * np.float32(2))).realize()
    np.testing.assert_array_equal(got, lazy)


def test_uint8_into_float32_uses_the_pixel_scale():
    src = np.array([[0, 51], [255, 102]], np.uint8)
    got = Array(np.zeros((2, 2), np.float32)).composite(((0, 0), src)).realize()
    np.testing.assert_allclose(got, src.astype(np.float32) / np.float32(255), atol=1e-6)


def test_float32_into_uint8_raises():
    base = Array(np.zeros((2, 2), np.uint8))
    with pytest.raises(RuntimeError, match="UNSUPPORTED_DTYPE"):
        base.composite(((0, 0), np.zeros((2, 2), np.float32))).realize()


def test_rgb_into_rgba_leaves_alpha():
    base = np.zeros((4, 5, 4), np.float32)
    base[..., 3] = 0.5
    rgb = _image((3, 4, 3), np.float32)
    expected = base.copy()
    expected[1:4, 1:5, :3] = rgb
    got = Array(base).composite(((1, 1), rgb)).realize()
    np.testing.assert_array_equal(got, expected)


def test_three_mono_sources_match_stack():
    planes = [_image((3, 4), np.float32) for _ in range(3)]
    base = mi.zeros((3, 4, 3), dtype=np.float32)
    got = base.composite(
        ((0, 0), planes[0], [0]),
        ((0, 0), planes[1], [1]),
        ((0, 0), planes[2], [2]),
    ).realize()
    np.testing.assert_array_equal(got, np.stack(planes, axis=-1))


def test_scatter_channels_write_the_named_dest_channels():
    base = np.zeros((2, 3, 4), np.float32)
    two = _image((2, 3, 2), np.float32)
    expected = base.copy()
    expected[:, :, [0, 2]] = two
    got = Array(base).composite(((0, 0), two, [0, 2])).realize()
    np.testing.assert_array_equal(got, expected)


def test_full_and_broadcast_to_are_sources():
    base = mi.zeros((5, 6, 3), dtype=np.float32)
    color = mi.full((2, 3, 3), (0.2, 0.7, 0.4))
    mono = Array(np.full((2, 3, 1), 0.5, np.float32))
    stretched = mi.broadcast_to(mono, (2, 3, 3))
    got = base.composite(((1, 1), color), ((3, 2), stretched)).realize()
    expected = np.zeros((5, 6, 3), np.float32)
    expected[1:3, 1:4] = (0.2, 0.7, 0.4)
    expected[3:5, 2:5] = 0.5
    np.testing.assert_array_equal(got, expected)


def test_a_one_dimensional_origin_is_the_column():
    base = np.arange(8, dtype=np.float32)
    src = np.array([9, 8, 7], np.float32)
    expected = base.copy()
    expected[2:5] = src
    np.testing.assert_array_equal(Array(base).composite(((2,), src)).realize(), expected)


def test_the_result_keeps_the_base_origin_and_leaves_the_base_unchanged():
    raw = np.zeros((4, 5), np.float32)
    base = Array(raw, origin=(3, 7))
    out = base.composite(((1, 2), np.ones((2, 2), np.float32)))
    assert out.shape == base.shape
    assert out.dtype == base.dtype
    assert out.meta.origin == (3, 7)
    assert out.meta.canvas == base.meta.canvas
    out.realize()
    np.testing.assert_array_equal(raw, np.zeros((4, 5), np.float32))


def test_a_source_past_the_edge_raises():
    base = Array(np.zeros((4, 5, 3), np.float32))
    patch = np.zeros((2, 2, 3), np.float32)
    for origin in ((3, 0), (0, 4), (-1, 0)):
        with pytest.raises(RuntimeError, match="INVALID_ARGUMENT"):
            base.composite((origin, patch)).realize()
    with pytest.raises(RuntimeError, match="INVALID_ARGUMENT"):
        base.composite(((0, 0), patch, [1, 2, 3])).realize()


def test_a_planar_array_raises():
    base = Array(np.zeros((3, 4, 5), np.float32), channel_axis=0)
    with pytest.raises(RuntimeError, match="GRAPH_INVALID"):
        base.composite(((0, 0), np.zeros((4, 5), np.float32))).realize()
    packed = Array(np.zeros((4, 5, 3), np.float32))
    planar = Array(np.zeros((3, 2, 2), np.float32), channel_axis=0)
    with pytest.raises(RuntimeError, match="GRAPH_INVALID"):
        packed.composite(((0, 0), planar)).realize()


def test_a_call_without_a_source_raises():
    base = Array(np.zeros((2, 2), np.float32))
    with pytest.raises(ValueError, match="at least"):
        base.composite()
