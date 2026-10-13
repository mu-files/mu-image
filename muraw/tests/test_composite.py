"""``composite(base, (origin, source), ...)`` against NumPy slice assignment.

``origin`` is ``(row, col)``, or ``(col,)`` on a 1D base. A placement may
end with the dest channel of each of that source's channels, on a packed
or a CHW base.
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
    channels = [_image((3, 4), np.float32) for _ in range(3)]
    base = mi.zeros((3, 4, 3), dtype=np.float32)
    got = base.composite(
        ((0, 0), channels[0], [0]),
        ((0, 0), channels[1], [1]),
        ((0, 0), channels[2], [2]),
    ).realize()
    np.testing.assert_array_equal(got, np.stack(channels, axis=-1))


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


def _chw(samples):
    return Array(samples, channel_axis=0)


def test_a_chw_base_matches_slice_assignment():
    for dtype in (np.uint8, np.float32):
        base = _image((3, 6, 8), dtype)
        rgb = _image((3, 2, 3), dtype)
        mono = _image((2, 3), dtype)
        expected = base.copy()
        expected[:, 1:3, 2:5] = rgb
        expected[2, 3:5, 4:7] = mono
        got = _chw(base).composite(((1, 2), _chw(rgb)), ((3, 4), mono, [2])).realize()
        np.testing.assert_array_equal(got, expected)


def test_the_same_placements_on_a_packed_and_a_chw_base():
    base = _image((6, 8, 3), np.float32)
    rgb = _image((2, 3, 3), np.float32)
    mono = _image((2, 3), np.float32)
    placements = (((1, 2), rgb), ((3, 4), mono, [2]))
    packed = Array(base).composite(*placements).realize()
    chw_placements = (((1, 2), _chw(rgb.transpose(2, 0, 1).copy())), placements[1])
    chw = _chw(base.transpose(2, 0, 1).copy()).composite(*chw_placements).realize()
    np.testing.assert_array_equal(chw, packed.transpose(2, 0, 1))


def test_three_mono_sources_stack_into_chw_channels():
    channels = [_image((3, 4), np.float32) for _ in range(3)]
    base = _chw(np.zeros((3, 3, 4), np.float32))
    got = base.composite(*(((0, 0), mono, [c]) for c, mono in enumerate(channels))).realize()
    np.testing.assert_array_equal(got, np.stack(channels, axis=0))


def test_chw_scatter_channels_write_the_named_channels():
    base = np.zeros((4, 2, 3), np.float32)
    two = _image((2, 2, 3), np.float32)
    expected = base.copy()
    expected[[0, 2]] = two
    got = _chw(base).composite(((0, 0), _chw(two), [0, 2])).realize()
    np.testing.assert_array_equal(got, expected)


def test_a_packed_base_takes_planes_from_a_chw_array():
    """Channels 0 and 2 of a packed base are the first and third planes of a CHW array."""
    base = _image((6, 8, 3), np.float32)
    planes = _image((3, 6, 8), np.float32)
    expected = base.copy()
    expected[:, :, 0] = planes[0]
    expected[:, :, 2] = planes[2]
    got = Array(base).composite(
        ((0, 0), planes[0], [0]),
        ((0, 0), planes[2], [2]),
    ).realize()
    np.testing.assert_array_equal(got, expected)


def test_a_chw_base_takes_a_region_from_packed_channels():
    """A window of channels 0 and 2 on a CHW base comes from a packed array."""
    base = _image((3, 6, 8), np.float32)
    packed = _image((4, 5, 3), np.float32)
    expected = base.copy()
    expected[0, 1:5, 2:7] = packed[:, :, 0]
    expected[2, 1:5, 2:7] = packed[:, :, 2]
    got = _chw(base).composite(
        ((1, 2), packed[:, :, 0], [0]),
        ((1, 2), packed[:, :, 2], [2]),
    ).realize()
    np.testing.assert_array_equal(got, expected)


def test_a_source_of_the_other_layout_raises_naming_the_transpose():
    packed_src = _image((2, 2, 3), np.float32)
    chw_base = _chw(np.zeros((3, 4, 5), np.float32))
    with pytest.raises(ValueError, match=r"transpose\(2, 0, 1\)"):
        chw_base.composite(((0, 0), packed_src))
    got = chw_base.composite(((0, 0), Array(packed_src).transpose(2, 0, 1))).realize()
    np.testing.assert_array_equal(got[:, :2, :2], packed_src.transpose(2, 0, 1))

    packed_base = Array(np.zeros((4, 5, 3), np.float32))
    with pytest.raises(ValueError, match=r"transpose\(1, 2, 0\)"):
        packed_base.composite(((0, 0), _chw(np.zeros((3, 2, 2), np.float32))))


def test_a_placement_without_its_own_tuple_raises():
    base = Array(np.zeros((4, 5), np.float32))
    with pytest.raises(TypeError, match=r"composite\(base, \(\(row, col\), source\)\)"):
        base.composite((1, 2), np.ones((2, 2), np.float32))


def test_a_call_without_a_source_raises():
    base = Array(np.zeros((2, 2), np.float32))
    with pytest.raises(ValueError, match="at least"):
        base.composite()
