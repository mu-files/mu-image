"""View / crop: dest coverage through a second window, rotate, and slice keys."""

from __future__ import annotations

import numpy as np
import pytest

import mucompute as mc
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
    assert t.meta.canvas == (0, 0, 6, 6)
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
    t = mc.orientation(t, orientation=Orientation.ROTATE_90_CW)
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
    t = mc.orientation(t, orientation=Orientation.ROTATE_90_CW)
    with pytest.raises(ValueError, match="outside canvas"):
        t.crop(left=-6, top=0, width=2 * w, height=side)


def test_slice_form_matches_rect_pixels():
    src = np.arange(5 * 7, dtype=np.float32).reshape(5, 7) + 1.0
    t = Tensor(src)
    rect = t.view(left=1, top=2, width=3, height=2)
    via_s = t.view(np.s_[2:4, 1:4])
    via_tuple = t.view((slice(2, 4), slice(1, 4)))
    want = src[2:4, 1:4]
    for got in (rect, via_s, via_tuple):
        np.testing.assert_array_equal(got.compute(), want)
    assert via_s.meta.canvas == rect.meta.canvas


def test_crop_slice_resets_canvas_view_keeps_it():
    src = np.arange(5 * 7, dtype=np.float32).reshape(5, 7) + 1.0
    t = Tensor(src)
    viewed = t.view(np.s_[2:4, 1:4])
    cropped = t.crop(np.s_[2:4, 1:4])
    np.testing.assert_array_equal(viewed.compute(), cropped.compute())
    assert viewed.meta.canvas == (0, 0, 7, 5)
    assert cropped.meta.canvas == (1, 2, 3, 2)


def test_getitem_is_hard_crop():
    src = np.arange(5 * 7, dtype=np.float32).reshape(5, 7) + 1.0
    t = Tensor(src)[2:4, 1:4]
    np.testing.assert_array_equal(t.compute(), src[2:4, 1:4])
    assert t.meta.canvas == (1, 2, 3, 2)
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
    with pytest.raises(ValueError, match="step"):
        t.view(np.s_[::-2, :])
    with pytest.raises(TypeError, match="slice indices must be integers"):
        t.view(slice(None, None, 1.0))
    with pytest.raises(TypeError, match="slice indices must be integers"):
        t.view(slice(None, None, 2.0))
    with pytest.raises(TypeError, match="slice indices must be integers"):
        t.view(slice(1.5, 3, None))
    with pytest.raises(TypeError, match="mix"):
        t.view(np.s_[1:3, :], left=0)
    with pytest.raises(TypeError, match="slice region or left"):
        t.view(left=1, top=1)
    with pytest.raises(TypeError, match="slice objects"):
        t.view(((2, 4), (1, 5)))
    rgb = Tensor(np.zeros((4, 6, 3), dtype=np.float32))
    np.testing.assert_array_equal(
        rgb.view(np.s_[1:3, 2:5, :]).compute(),
        np.zeros((2, 3, 3), dtype=np.float32),
    )
    with pytest.raises(ValueError, match="channel"):
        rgb.view(np.s_[1:3, 2:5, 0:2])
    with pytest.raises(TypeError, match="slice indices must be integers"):
        rgb.view((slice(None), slice(None), slice(None, None, 1.0)))
    with pytest.raises(IndexError, match="too many indices"):
        t.view(np.s_[:, :, :])
    with pytest.raises(IndexError, match="ellipsis"):
        t.view((..., ...))
    with pytest.raises(TypeError, match="must be an int"):
        t.view(left=1.5, top=0, width=2, height=2)
    with pytest.raises(TypeError, match="must be an int"):
        t.view(left=True, top=0, width=2, height=2)
    with pytest.raises(ValueError, match="invalid box"):
        t.view(left=0, top=0, width=0, height=2)


def _mono() -> np.ndarray:
    return np.arange(5 * 7, dtype=np.float32).reshape(5, 7) + 1.0


def _rgb() -> np.ndarray:
    plane = np.arange(5 * 7, dtype=np.float32).reshape(5, 7)
    return np.stack([plane, plane + 100.0, plane + 200.0], axis=-1)


@pytest.mark.parametrize("src", [_mono(), _rgb()], ids=["mono", "rgb"])
@pytest.mark.parametrize(
    "key",
    [np.s_[::-1, :], np.s_[:, ::-1], np.s_[::-1, ::-1], np.s_[4:1:-1, 5:1:-1]],
    ids=["flipud", "fliplr", "rot180", "partial"],
)
@pytest.mark.parametrize("method", ["view", "crop", "getitem"])
def test_reverse_slice_matches_numpy(src, key, method):
    t = Tensor(src)
    if method == "view":
        got = t.view(key)
    elif method == "crop":
        got = t.crop(key)
    else:
        got = t[key]
    np.testing.assert_array_equal(got.compute(), src[key])


def test_reverse_slice_rgb_keeps_all_channels():
    src = _rgb()
    np.testing.assert_array_equal(
        Tensor(src).view(np.s_[::-1, :, :]).compute(), src[::-1, :, :]
    )


def test_empty_slice_is_rejected():
    t = Tensor(_mono())
    with pytest.raises(ValueError, match="empty dimension"):
        t.view(np.s_[1:4:-1, :])
    with pytest.raises(ValueError, match="empty dimension"):
        t.view(np.s_[20:10, :])
    with pytest.raises(ValueError, match="empty dimension"):
        t.view(np.s_[10:20:-1, :])
    with pytest.raises(ValueError, match="empty dimension"):
        t.view(np.s_[2:2, :])


def test_ellipsis_fills_remaining_axes():
    src = _mono()
    t = Tensor(src)
    np.testing.assert_array_equal(t.view(...).compute(), src)
    np.testing.assert_array_equal(t.view(np.s_[1:4, ...]).compute(), src[1:4])
    np.testing.assert_array_equal(t.view(np.s_[1:4, 2:6, ...]).compute(), src[1:4, 2:6])
    rgb = _rgb()
    tr = Tensor(rgb)
    np.testing.assert_array_equal(tr.view(...).compute(), rgb)
    np.testing.assert_array_equal(tr[:].compute(), rgb[:])
    np.testing.assert_array_equal(tr[1:4].compute(), rgb[1:4])
    np.testing.assert_array_equal(
        tr.view(np.s_[1:4, 2:6, ...]).compute(), rgb[1:4, 2:6]
    )
    np.testing.assert_array_equal(
        tr.view(np.s_[1:4, 2:6, ..., :]).compute(), rgb[1:4, 2:6]
    )


def test_reverse_crop_resets_canvas_view_keeps_it():
    """A reversed crop sits on its box. A reversed view remaps the parent canvas."""
    src = _mono()
    t = Tensor(src)
    key = np.s_[4:1:-1, 1:4]
    viewed = t.view(key)
    cropped = t.crop(key)
    want = src[key]
    np.testing.assert_array_equal(viewed.compute(), want)
    np.testing.assert_array_equal(cropped.compute(), want)
    assert cropped.meta.canvas == (1, 2, 3, 3)
    with pytest.raises(ValueError, match="outside canvas"):
        cropped.crop(left=-1, top=0, width=5, height=3)
    extra = viewed.crop(left=-1, top=0, width=5, height=3)
    np.testing.assert_array_equal(extra.compute(), src[4:1:-1, 0:5])
