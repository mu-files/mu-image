"""Crop: dest coverage through a second crop, and through rotate."""

from __future__ import annotations

import numpy as np
import pytest

import muimg.engines.ops as engine_ops
from muimg.tensor import Tensor
from muimg.tiff_metadata import Orientation


def test_two_crops_first_oob_valid():
    """First crop is inside the source. Second crop reaches past the first.

    ``oob_valid`` on the first crop maps the source canvas so the second
    crop can read source pixels.
    """
    src = np.arange(5 * 7, dtype=np.float32).reshape(5, 7) + 1.0
    left0, top0, width0, height0 = 1, 1, 3, 3
    left1, top1, width1, height1 = -1, -1, 5, 5

    t = Tensor(src).crop(
        left=left0,
        top=top0,
        width=width0,
        height=height0,
        oob_valid=True,
    )
    t = t.crop(left=left1, top=top1, width=width1, height=height1)
    from_source = t.compute()

    left = left0 + left1
    top = top0 + top1
    np.testing.assert_array_equal(
        from_source, src[top : top + height1, left : left + width1]
    )


def test_two_crops_first_oob_invalid_rejects_second():
    """``oob_valid=false`` resets canvas. A second crop past it fails."""
    src = np.arange(5 * 7, dtype=np.float32).reshape(5, 7) + 1.0
    t = Tensor(src).crop(left=1, top=1, width=3, height=3, oob_valid=False)
    with pytest.raises(ValueError, match="outside canvas"):
        t.crop(left=-1, top=-1, width=5, height=5)


def test_crop_restore_full_source():
    """6×6 → crop(1,1,4,4) → crop(-1,-1,6,6) is the full input."""
    src = np.arange(6 * 6, dtype=np.float32).reshape(6, 6) + 1.0
    t = Tensor(src).crop(left=1, top=1, width=4, height=4, oob_valid=True)
    assert t.meta.canvas == (-1, -1, 6, 6)
    t = t.crop(left=-1, top=-1, width=6, height=6)
    np.testing.assert_array_equal(t.compute(), src)


def _rot90_cw(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(a, -1))


def test_crop_rotate_crop_oob_valid():
    """``crop → orientation(ROTATE_90_CW) → crop``.

    Second crop is ``left=-6``, ``width=2w`` (wider than the rotated square).

    ``oob_valid=true``: first crop maps the source canvas. Orientation
    remaps that canvas. The second crop is a strip of the rotated source.
    """
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
        oob_valid=True,
    )
    t = engine_ops.orientation(t, orientation=Orientation.ROTATE_90_CW)
    t = t.crop(left=-6, top=0, width=2 * w, height=side)
    from_source = t.compute()

    rotated = _rot90_cw(src)
    middle = rotated[w // 4 : w // 4 + side, :]
    np.testing.assert_array_equal(from_source, middle)


def test_crop_rotate_crop_oob_invalid_rejects_second():
    """``oob_valid=false`` then rotate: a second crop past the reset canvas fails."""
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
        oob_valid=False,
    )
    t = engine_ops.orientation(t, orientation=Orientation.ROTATE_90_CW)
    with pytest.raises(ValueError, match="outside canvas"):
        t.crop(left=-6, top=0, width=2 * w, height=side)
