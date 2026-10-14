"""Time a four-image mosaic, packed and CHW, against NumPy slice assignment.

Four quarter-frame images are written into the four corners of one frame.
NumPy assigns each into an empty array. muimage writes them with one
``composite`` over a zero buffer, which the four sources cover, so the
base is never read. The base is a buffer, not ``mi.zeros``: a CHW fill is
``mi.zeros(...).transpose(2, 0, 1)``, and the engine computes the fill in
full before a transpose.
"""

from __future__ import annotations

import time

import numpy as np

import muimage as mi

_ROWS = 3000
_COLS = 4000


def _quarters(chw):
    """Four ``(rows / 2, cols / 2, 3)`` images, or ``(3, ...)`` when ``chw``."""
    rng = np.random.default_rng(5)
    shape = (3, _ROWS // 2, _COLS // 2) if chw else (_ROWS // 2, _COLS // 2, 3)
    return [rng.random(shape, dtype=np.float32) for _ in range(4)]


def _origins():
    return [(top, left) for top in (0, _ROWS // 2) for left in (0, _COLS // 2)]


def _numpy(quarters, chw):
    out = np.empty((3, _ROWS, _COLS) if chw else (_ROWS, _COLS, 3), np.float32)
    height, width = _ROWS // 2, _COLS // 2
    for (top, left), quarter in zip(_origins(), quarters):
        rows, cols = slice(top, top + height), slice(left, left + width)
        out[(slice(None), rows, cols) if chw else (rows, cols)] = quarter
    return out


def _muimage(base, quarters):
    return base.composite(*zip(_origins(), quarters)).realize()


def _median_time(run, repeats=5):
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        samples.append(time.perf_counter() - start)
    samples.sort()
    return samples[len(samples) // 2]


def test_a_four_image_mosaic_against_numpy_assignment():
    for chw in (False, True):
        quarters = _quarters(chw)
        channel_axis = 0 if chw else -1
        base = mi.Array(
            np.zeros((3, _ROWS, _COLS) if chw else (_ROWS, _COLS, 3), np.float32),
            channel_axis=channel_axis,
        )
        sources = [mi.Array(quarter, channel_axis=channel_axis) for quarter in quarters]
        np.testing.assert_array_equal(_muimage(base, sources), _numpy(quarters, chw))
        mi_time = _median_time(lambda: _muimage(base, sources))
        np_time = _median_time(lambda: _numpy(quarters, chw))
        layout = "CHW" if chw else "packed"
        message = f"{layout}: muimage {mi_time * 1e3:.1f} ms, numpy {np_time * 1e3:.1f} ms"
        print(message)
        # Measured: packed 1.06 times NumPy's time, CHW 0.74 times.
        assert mi_time < 2 * np_time, message
