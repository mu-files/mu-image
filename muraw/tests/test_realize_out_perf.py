"""Time and Python memory of a 4K realize into a moveaxis view.

The result is packed ``(H, W, 3)``. ``out`` is ``np.moveaxis`` of a CHW
buffer, so its memory is already planar. muimage writes that buffer.
NumPy multiplies the same way and then ``ascontiguousarray`` of the
moveaxis view, which allocates another frame. The peaks below are arrays
Python allocates during the call. The engine's own workspace is not part
of that number.
"""

from __future__ import annotations

import time
import tracemalloc

import numpy as np

from muraw.array import Array

_ROWS = 2160
_COLS = 3840


def _scaled(image):
    return image * image.dtype.type(2)


def _measure(run):
    """Elapsed seconds and the tracemalloc peak of one call."""
    tracemalloc.reset_peak()
    start = time.perf_counter()
    run()
    elapsed = time.perf_counter() - start
    _, peak = tracemalloc.get_traced_memory()
    return elapsed, peak


def test_a_4k_moveaxis_out_does_not_allocate_a_second_frame():
    """Writing the moveaxis view stays under one frame; the copy does not."""
    src = np.random.default_rng(0).random((_ROWS, _COLS, 3), np.float32)
    chw = np.empty((3, _ROWS, _COLS), np.float32)
    out = np.moveaxis(chw, 0, -1)
    frame = src.nbytes

    def muimage():
        _scaled(Array(src)).realize(out=out)

    def numpy_copy():
        return np.ascontiguousarray(np.moveaxis(_scaled(src), -1, 0))

    muimage()
    copied = numpy_copy()
    np.testing.assert_array_equal(chw, copied)

    tracemalloc.start()
    try:
        mu_time, mu_peak = _measure(muimage)
        np_time, np_peak = _measure(numpy_copy)
    finally:
        tracemalloc.stop()

    message = (
        f"muimage {mu_time * 1e3:.1f} ms, Python peak {mu_peak / 1e6:.1f} MB; "
        f"numpy {np_time * 1e3:.1f} ms, Python peak {np_peak / 1e6:.1f} MB; "
        f"frame {frame / 1e6:.1f} MB"
    )
    print(message)
    assert mu_peak < frame, message
    assert np_peak > frame, message
