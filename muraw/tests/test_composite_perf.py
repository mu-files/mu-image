"""Time composite against other ways to compute the same image.

A packed image with a solid rectangle written over its middle: NumPy
computes the whole image and then assigns the rectangle. muimage writes
it with ``composite``, so the work under it is not run. The rectangle
grows from a few pixels to just inside the frame.

An image turned 180 degrees by placing it with reversed row and column
lists, against the same turn by ``orientation``.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

import muimage as mi

_ROWS = 3000
_COLS = 4000
_MATRIX = np.array(
    [[0.9, 0.2, -0.1], [0.05, 1.1, -0.15], [-0.2, 0.1, 1.1]], dtype=np.float32
)
_COLOR = (np.float32(0.2), np.float32(0.7), np.float32(0.4))


def _source(rows, cols):
    """A packed ``(rows, cols, 3)`` image."""
    return np.random.default_rng(3).random((rows, cols, 3), dtype=np.float32)


def _occluder_sizes(rows, cols):
    """Centered ``(height, width)`` from 4 pixels on a side to just inside the frame."""
    sizes = [(4, 4)]
    for fraction in (1 / 32, 1 / 8, 1 / 4, 1 / 2, 3 / 4, 0.95):
        height = min(rows - 2, max(4, round(rows * fraction)))
        width = min(cols - 2, max(4, round(cols * fraction)))
        sizes.append((height, width))
    sizes.append((rows - 2, cols - 2))
    return sizes


def _origin(rows, cols, height, width):
    """Top and left that center ``height`` by ``width`` on the image."""
    return (rows - height) // 2, (cols - width) // 2


def _rendered(lib, src):
    """``src`` plus the row coordinate, squared, then a 3x3 matrix.

    ``src`` is packed ``(rows, cols, 3)``. The row coordinate is one
    channel and broadcasts across the three.
    """
    rows, cols, _ = src.shape
    _, y = lib.meshgrid(
        lib.arange(cols, dtype=np.float32), lib.arange(rows, dtype=np.float32)
    )
    shifted = src + y[..., None] * np.float32(1.5)
    return (shifted * shifted) @ _MATRIX


def _numpy(src, height, width):
    image = _rendered(np, src)
    top, left = _origin(src.shape[0], src.shape[1], height, width)
    image[top : top + height, left : left + width] = _COLOR
    return image


def _muimage(src, height, width):
    image = _rendered(mi, mi.Array(src))
    top, left = _origin(src.shape[0], src.shape[1], height, width)
    patch = mi.full((height, width, 3), _COLOR)
    return image.composite(((top, left), patch)).realize()


def _median_time(run, repeats=5):
    """The median of ``repeats`` runs. One fast or slow sample does not
    move the ratio the regression check uses."""
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        run()
        samples.append(time.perf_counter() - start)
    samples.sort()
    return samples[len(samples) // 2]


def test_a_larger_occluder_skips_more_work_than_numpy():
    """Both sides start from the same packed image, add the row
    coordinate, square, and apply a 3x3 matrix. A centered solid
    rectangle covers the middle. NumPy writes it after the image
    exists. muimage places it with composite, so those pixels are not
    computed.

    The regression check is that muimage stays faster than NumPy at
    every size. NumPy always renders the whole frame. A ratio of
    muimage times against the 4x4 rectangle does not: realize cost
    and runner noise can make a large rectangle slower than 4x4 even
    when it is still several times faster than NumPy."""
    src = _source(_ROWS, _COLS)
    rows, cols, _ = src.shape
    for height, width in _occluder_sizes(rows, cols):
        got = _muimage(src, height, width)
        expected = _numpy(src, height, width)
        np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
        mi_time = _median_time(lambda: _muimage(src, height, width))
        np_time = _median_time(lambda: _numpy(src, height, width))
        message = (
            f"{height}x{width}: muimage {mi_time * 1e3:.1f} ms, "
            f"numpy {np_time * 1e3:.1f} ms"
        )
        print(message)
        assert mi_time < np_time, message


# Composite's time for a 180 degree turn divided by orientation's. Measured
# ratios were about 2.1 packed and 2.6 channels-first; before composite
# read its column runs from prepare they were about 5 and 11.
_TURN_RATIO_CEILING = 4.0


@pytest.mark.parametrize("channel_axis", [-1, 0], ids=["packed", "channels_first"])
def test_a_reversed_composite_stays_near_orientation(channel_axis):
    """A 768x1024 float32 image turned 180 degrees two ways: placed on a
    base with reversed row and column lists, and by ``orientation`` 3.
    Both match NumPy's flip. Composite takes at most
    ``_TURN_RATIO_CEILING`` times orientation's time, a ratio that does
    not depend on how fast the machine is."""
    rows, cols = 768, 1024
    shape = (3, rows, cols) if channel_axis == 0 else (rows, cols, 3)
    spatial_axes = (1, 2) if channel_axis == 0 else (0, 1)
    src = np.random.default_rng(5).random(shape, dtype=np.float32)
    image = mi.Array(src, channel_axis=channel_axis)
    base = mi.Array(np.zeros_like(src), channel_axis=channel_axis)
    row_list = list(range(rows - 1, -1, -1))
    col_list = list(range(cols - 1, -1, -1))

    def composite():
        return base.composite(((row_list, col_list), image)).realize()

    def orientation():
        return mi.orientation(image, orientation=3).realize()

    expected = np.flip(src, axis=spatial_axes)
    np.testing.assert_array_equal(composite(), expected)
    np.testing.assert_array_equal(orientation(), expected)
    composite_time = _median_time(composite, repeats=21)
    orientation_time = _median_time(orientation, repeats=21)
    ratio = composite_time / orientation_time
    message = (
        f"composite {composite_time * 1e3:.2f} ms, orientation "
        f"{orientation_time * 1e3:.2f} ms, ratio {ratio:.2f}"
    )
    print(message)
    assert ratio <= _TURN_RATIO_CEILING, message
