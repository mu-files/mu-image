"""Composite four rotations of one image into the quadrants of a larger one.

The source is a 420x420 8-bit RGB square. Red brightens toward the right,
green toward the bottom, and blue is an L in the top-left corner, so a
rotation does not look like the original. ``rot90`` by 0, 1, 2, and 3
turns lands in the top-left, top-right, bottom-left, and bottom-right
quadrant of an 840x840 image.
"""

from __future__ import annotations

import numpy as np

import muimage as mi
from muraw.common import PerfTimer
from muraw.engines.graph import EngineTiming, set_engine_timing

_SIDE = 420
# (block row, block column, rot90 turns). Turns go counterclockwise.
_PLACEMENTS = (
    (0, 0, 0),
    (0, 1, 1),
    (1, 0, 2),
    (1, 1, 3),
)


def _marked_square(side=_SIDE):
    """Packed uint8 RGB whose gradients and corner L show a rotation."""
    columns = (np.arange(side) * 255 // (side - 1)).astype(np.uint8)
    rows = (np.arange(side) * 255 // (side - 1)).astype(np.uint8)
    red = np.broadcast_to(columns, (side, side)).copy()
    green = np.broadcast_to(rows[:, None], (side, side)).copy()
    blue = np.zeros((side, side), np.uint8)
    blue[: side // 5, :] = 255
    blue[:, : side // 10] = 255
    return np.stack([red, green, blue], axis=-1)


def _numpy(src):
    side = src.shape[0]
    image = np.zeros((side * 2, side * 2, 3), np.uint8)
    for block_row, block_col, turns in _PLACEMENTS:
        top, left = block_row * side, block_col * side
        image[top : top + side, left : left + side] = np.rot90(src, turns)
    return image


def _muimage(src):
    side = src.shape[0]
    image = mi.Array(src)
    base = mi.zeros((side * 2, side * 2, 3), dtype=np.uint8)
    placements = tuple(
        ((block_row * side, block_col * side), mi.rot90(image, turns))
        for block_row, block_col, turns in _PLACEMENTS
    )
    return base.composite(*placements).realize()


def test_four_rotations_composite_into_quadrants():
    """The four quadrants match NumPy. The engine timing is printed."""
    src = _marked_square()
    expected = _numpy(src)
    np.testing.assert_array_equal(_muimage(src), expected)

    _muimage(src)
    _numpy(src)
    set_engine_timing(EngineTiming.OPS)
    try:
        with PerfTimer("numpy") as numpy_timer:
            with numpy_timer.start_step("quadrants"):
                _numpy(src)
        with PerfTimer("muimage") as muimage_timer:
            _muimage(src)
    finally:
        set_engine_timing(EngineTiming.OFF)
    print(numpy_timer.get_report())
    print(muimage_timer.get_report())
