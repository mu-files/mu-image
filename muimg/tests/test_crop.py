"""View / crop: dest coverage through a second window, rotate, and slice keys."""

from __future__ import annotations

import numpy as np
import pytest

import muimg.engines.ops as engine_ops
from muimg.tensor import Tensor
from muimg.tiff_metadata import Orientation


def test_two_windows_first_view_maps_canvas():
    """First view is inside the source. Second crop reaches past the first.

    A view maps the source canvas so the second window can read source pixels.
    """
    src = np.arange(5 * 7, dtype=np.float32).reshape(5, 7) + 1.0
    left0, top0, width0, height0 = 1, 1, 3, 3
    left1, top1, width1, height1 = -1, -1, 5, 5

    t = Tensor(src).view(
        left=left0,
        top=top0,
        width=width0,
        height=height0,
    )
    t = t.crop(left=left1, top=top1, width=width1, height=height1)
    from_source = t.compute()

    left = left0 + left1
    top = top0 + top1
    np.testing.assert_array_equal(
        from_source, src[top : top + height1, left : left + width1]
    )


def test_two_crops_first_hard_crop_rejects_second():
    """A crop resets canvas. A second crop past it fails."""
    src = np.arange(5 * 7, dtype=np.float32).reshape(5, 7) + 1.0
    t = Tensor(src).crop(left=1, top=1, width=3, height=3)
    with pytest.raises(ValueError, match="outside canvas"):
        t.crop(left=-1, top=-1, width=5, height=5)


def test_view_restore_full_source():
    """6×6 → view(1,1,4,4) → crop(-1,-1,6,6) is the full input."""
    src = np.arange(6 * 6, dtype=np.float32).reshape(6, 6) + 1.0
    t = Tensor(src).view(left=1, top=1, width=4, height=4)
    assert t.meta.canvas == (-1, -1, 6, 6)
    t = t.crop(left=-1, top=-1, width=6, height=6)
    np.testing.assert_array_equal(t.compute(), src)


def _rot90_cw(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(a, -1))


def test_view_rotate_crop_maps_canvas():
    """``view → orientation(ROTATE_90_CW) → crop``.

    Second crop is ``left=-6``, ``width=2w`` (wider than the rotated square).

    The view maps the source canvas. Orientation remaps that canvas. The
    second crop is a strip of the rotated source.
    """
    w = 8
    h = 2 * w
    src = np.arange(h * w, dtype=np.float32).reshape(h, w) + 1.0
    side = w // 2
    left0 = (w - side) // 2
    top0 = (h - side) // 2

    t = Tensor(src).view(
        left=left0,
        top=top0,
        width=side,
        height=side,
    )
    t = engine_ops.orientation(t, orientation=Orientation.ROTATE_90_CW)
    t = t.crop(left=-6, top=0, width=2 * w, height=side)
    from_source = t.compute()

    rotated = _rot90_cw(src)
    middle = rotated[w // 4 : w // 4 + side, :]
    np.testing.assert_array_equal(from_source, middle)


def test_crop_rotate_crop_rejects_second():
    """A crop then rotate: a second crop past the reset canvas fails."""
    w = 8
    h = 2 * w
    src = np.arange(h * w, dtype=np.float32).reshape(h, w) + 1.0
    side = w // 2
    left0 = (w - side) // 2
    top0 = (h - side) // 2

    t = Tensor(src).crop(
        left=left0,
        top=top0,
        width=side,
        height=side,
    )
    t = engine_ops.orientation(t, orientation=Orientation.ROTATE_90_CW)
    with pytest.raises(ValueError, match="outside canvas"):
        t.crop(left=-6, top=0, width=2 * w, height=side)


def test_slice_form_matches_rect_pixels():
    src = np.arange(5 * 7, dtype=np.float32).reshape(5, 7) + 1.0
    t = Tensor(src)
    rect = t.view(left=1, top=2, width=3, height=2)
    via_s = t.view(np.s_[2:4, 1:4])
    via_tuple = t.view((slice(2, 4), slice(1, 4)))
    via_two = t.view(slice(2, 4), slice(1, 4))
    want = src[2:4, 1:4]
    for got in (rect, via_s, via_tuple, via_two):
        np.testing.assert_array_equal(got.compute(), want)
    assert via_s.meta.canvas == rect.meta.canvas


def test_crop_slice_resets_canvas_view_keeps_it():
    src = np.arange(5 * 7, dtype=np.float32).reshape(5, 7) + 1.0
    t = Tensor(src)
    viewed = t.view(np.s_[2:4, 1:4])
    cropped = t.crop(np.s_[2:4, 1:4])
    np.testing.assert_array_equal(viewed.compute(), cropped.compute())
    assert viewed.meta.canvas == (-1, -2, 7, 5)
    assert cropped.meta.canvas == (0, 0, 3, 2)


def test_getitem_is_hard_crop():
    src = np.arange(5 * 7, dtype=np.float32).reshape(5, 7) + 1.0
    t = Tensor(src)[2:4, 1:4]
    np.testing.assert_array_equal(t.compute(), src[2:4, 1:4])
    assert t.meta.canvas == (0, 0, 3, 2)
    with pytest.raises(ValueError, match="outside canvas"):
        t.crop(left=-1, top=-1, width=5, height=4)


def test_slice_negative_indices_are_numpy():
    src = np.arange(5 * 7, dtype=np.float32).reshape(5, 7) + 1.0
    t = Tensor(src).crop(np.s_[-2:, -3:])
    np.testing.assert_array_equal(t.compute(), src[-2:, -3:])
    assert t.meta.height == 2 and t.meta.width == 3


def test_slice_rejects_step_and_mixed_args():
    t = Tensor(np.zeros((4, 6), dtype=np.float32))
    with pytest.raises(ValueError, match="step"):
        t.view(np.s_[::2, :])
    with pytest.raises(TypeError, match="mix"):
        t.view(slice(1, 3), 0, 2, 2)
    with pytest.raises(TypeError, match="slice objects"):
        t.view(((2, 4), (1, 5)))
    rgb = Tensor(np.zeros((4, 6, 3), dtype=np.float32))
    np.testing.assert_array_equal(
        rgb.view(np.s_[1:3, 2:5, :]).compute(),
        np.zeros((2, 3, 3), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="channel"):
        rgb.view(np.s_[1:3, 2:5, 0:2])
