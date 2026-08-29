"""Crop: dest coverage through a second crop, and through rotate."""

from __future__ import annotations

import numpy as np

import muimg.engines.ops as engine_ops
from muimg.tensor import Tensor
from muimg.tiff_metadata import Orientation


def test_two_crops_first_oob_valid():
    """First crop is inside the source. Second crop reaches past the first.

    ``oob_valid`` on the first crop: source pixels vs the first crop's ``pad``
    on dest outside ``valid``.
    """
    src = np.arange(5 * 7, dtype=np.float32).reshape(5, 7) + 1.0
    left0, top0, width0, height0 = 1, 1, 3, 3
    left1, top1, width1, height1 = -1, -1, 5, 5

    def run(oob_valid: bool) -> np.ndarray:
        t = engine_ops.crop(
            Tensor(src),
            left=left0,
            top=top0,
            width=width0,
            height=height0,
            oob_valid=oob_valid,
            pad="zero",
        )
        t = engine_ops.crop(
            t,
            left=left1,
            top=top1,
            width=width1,
            height=height1,
            pad="zero",
        )
        return t.compute()

    from_source = run(True)
    from_pad = run(False)
    assert not np.array_equal(from_source, from_pad)

    left = left0 + left1
    top = top0 + top1
    np.testing.assert_array_equal(
        from_source, src[top : top + height1, left : left + width1]
    )

    first = src[top0 : top0 + height0, left0 : left0 + width0]
    want_pad = np.zeros((height1, width1), dtype=np.float32)
    want_pad[1:4, 1:4] = first
    np.testing.assert_array_equal(from_pad, want_pad)


def _rot90_cw(a: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(a, -1))


def test_crop_rotate_crop_oob_valid():
    """``crop → orientation(ROTATE_90_CW) → crop``. Second crop is ``(0, w/4, 2w, w/2)``.

    ``oob_valid`` on the first crop: middle of ``ROTATE_90_CW`` on the
    source vs that same middle with only the first crop's square, ``pad``
    ``zero`` on the sides.
    """
    w = 8
    h = 2 * w
    src = np.arange(h * w, dtype=np.float32).reshape(h, w) + 1.0
    side = w // 2
    left0 = (w - side) // 2
    top0 = (h - side) // 2

    def run(oob_valid: bool) -> np.ndarray:
        t = engine_ops.crop(
            Tensor(src),
            left=left0,
            top=top0,
            width=side,
            height=side,
            oob_valid=oob_valid,
            pad="zero",
        )
        t = engine_ops.orientation(
            t, orientation=Orientation.ROTATE_90_CW, pad="zero"
        )
        t = engine_ops.crop(
            t, left=-6, top=0, width=2 * w, height=side, pad="zero"
        )
        return t.compute()

    rotated = _rot90_cw(src)
    middle = rotated[w // 4 : w // 4 + side, :]

    first = src[top0 : top0 + side, left0 : left0 + side]
    want_pad = np.zeros((side, 2 * w), dtype=np.float32)
    want_pad[:, 6 : 6 + side] = _rot90_cw(first)

    from_source = run(True)
    from_pad = run(False)
    np.testing.assert_array_equal(from_source, middle)
    np.testing.assert_array_equal(from_pad, want_pad)
    assert not np.array_equal(from_source, from_pad)
