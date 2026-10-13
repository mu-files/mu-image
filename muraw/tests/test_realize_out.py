"""``realize(out=)`` writes the caller's buffer, or copies into it.

The engine writes ``out`` itself when that buffer is one it can bind.
Anything else, including a negative stride, is the engine's own buffer
copied in with ``np.copyto``. ``out`` is neither sealed nor kept on the
handle.
"""

from __future__ import annotations

import numpy as np
import pytest

import muimage as mi
from muraw.array import Array


def _scaled(image):
    """The same samples, times two. One value that stays inside uint8."""
    return image * image.dtype.type(2)


def _capture_engine_output(monkeypatch):
    """The ndarray each ``execute_graph`` call is given as its output."""
    from muraw.engines.core import _engine_load

    written: list[np.ndarray] = []
    real = _engine_load.execute_graph

    def wrap(graph, in_binds, output, record_ops=False):
        written.append(output)
        return real(graph, in_binds, output, record_ops)

    monkeypatch.setattr(_engine_load, "execute_graph", wrap)
    return written


def _assert_engine_writes(image, out, expected, monkeypatch):
    """``image.realize(out=out)`` is ``out``, and the engine's output is ``out``."""
    written = _capture_engine_output(monkeypatch)
    got = image.realize(out=out)
    assert got is out
    assert written == [out]
    assert out.flags.writeable
    assert image._data is None
    np.testing.assert_array_equal(out, expected)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_a_contiguous_out_is_the_buffer_the_engine_writes(dtype, monkeypatch):
    """A span op writes a C-contiguous ``out`` of the result's dtype."""
    src = np.arange(4 * 5, dtype=dtype).reshape(4, 5)
    out = np.empty(src.shape, dtype=dtype)
    _assert_engine_writes(_scaled(Array(src)), out, _scaled(src), monkeypatch)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_a_contiguous_chw_out_is_the_buffer_the_engine_writes(dtype, monkeypatch):
    """A span op writes a contiguous ``(C, H, W)`` ``out``."""
    src = np.arange(3 * 4 * 5, dtype=dtype).reshape(3, 4, 5)
    out = np.empty(src.shape, dtype=dtype)
    image = _scaled(Array(src, channel_axis=0))
    assert image.meta.channel_axis == 0
    _assert_engine_writes(image, out, _scaled(src), monkeypatch)


def test_a_padded_out_is_the_buffer_the_engine_writes(monkeypatch):
    """A row-padded ``(H, W, C)`` view is the engine's output, and the pad stays."""
    height, width, channels, pad = 5, 7, 3, 2
    parent = np.full((height, width + pad, channels), -1, np.float32)
    out = parent[:, :width, :]
    src = np.arange(height * width * channels, dtype=np.float32).reshape(
        height, width, channels
    )
    _assert_engine_writes(_scaled(Array(src)), out, _scaled(src), monkeypatch)
    np.testing.assert_array_equal(parent[:, width:, :], -1)
    assert parent.flags.writeable


def test_one_slot_of_a_batch_is_the_buffer_the_engine_writes(monkeypatch):
    """``out=batch[i]`` fills slot ``i`` and leaves the other slots alone."""
    batch = np.zeros((4, 3, 5, 6), np.float32)
    slot = batch[2]
    slot[...] = 1
    src = np.arange(3 * 5 * 6, dtype=np.float32).reshape(3, 5, 6)
    _assert_engine_writes(
        _scaled(Array(src, channel_axis=0)), slot, _scaled(src), monkeypatch
    )
    assert np.shares_memory(slot, batch)
    for index in (0, 1, 3):
        np.testing.assert_array_equal(batch[index], 0)
    assert batch.flags.writeable


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_a_span_then_orientation_writes_out(dtype, monkeypatch):
    """The engine writes ``out`` when the last op is an orientation of a span op."""
    src = np.arange(4 * 6, dtype=dtype).reshape(4, 6)
    image = mi.orientation(_scaled(Array(src)), orientation=6)
    out = np.empty((6, 4), dtype=dtype)
    _assert_engine_writes(image, out, np.rot90(_scaled(src), -1), monkeypatch)


def test_a_same_kind_cast_writes_out(monkeypatch):
    """uint8 pixels land in a float32 ``out`` through ``astype``, not a range scale."""
    src = np.arange(12, dtype=np.uint8).reshape(3, 4)
    out = np.empty(src.shape, np.float32)
    image = _scaled(Array(src))
    _assert_engine_writes(image, out, _scaled(src).astype(np.float32), monkeypatch)
    assert image.dtype == np.dtype(np.uint8)


def test_a_rejected_cast_raises_before_the_engine_runs(monkeypatch):
    """float32 into uint8 is not a ``same_kind`` cast."""
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = np.empty(src.shape, np.uint8)
    written = _capture_engine_output(monkeypatch)
    with pytest.raises(TypeError, match="same_kind"):
        _scaled(Array(src)).realize(out=out)
    assert written == []


def test_an_unsupported_out_dtype_raises():
    """A dtype an Array cannot hold is rejected before the graph runs."""
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = np.empty(src.shape, np.float64)
    with pytest.raises(ValueError, match="float64"):
        _scaled(Array(src)).realize(out=out)


def test_a_wrong_shape_raises():
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = np.empty((3, 5), np.float32)
    with pytest.raises(ValueError, match="does not match"):
        _scaled(Array(src)).realize(out=out)


def test_a_read_only_out_raises():
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    out = np.empty(src.shape, np.float32)
    out.setflags(write=False)
    with pytest.raises(ValueError, match="read-only"):
        _scaled(Array(src)).realize(out=out)


def test_out_must_be_an_ndarray():
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    image = _scaled(Array(src))
    with pytest.raises(TypeError, match="ndarray"):
        image.realize(out=[0])
    with pytest.raises(TypeError, match="ndarray"):
        image.realize(out=Array(src))


def test_an_input_cannot_be_out():
    """A writeable view of an input is rejected, not written."""
    parent = np.arange(3 * 8, dtype=np.float32).reshape(3, 8).copy()
    src = parent[:, :4]
    image = _scaled(Array(src))
    parent.setflags(write=True)
    overlap = parent[:, 2:6]
    overlap.setflags(write=True)
    with pytest.raises(ValueError, match="input of the graph"):
        image.realize(out=overlap)


def test_a_realized_input_cannot_be_out():
    """Pixels an earlier realize cached are an input of the graph that reads them."""
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    earlier = _scaled(Array(src))
    cached = earlier.realize()
    later = earlier + np.float32(1)
    cached.setflags(write=True)
    with pytest.raises(ValueError, match="input of the graph"):
        later.realize(out=cached)


def test_a_negative_stride_is_copied_into(monkeypatch):
    """A flipped ``out`` is not the engine's buffer; ``copyto`` fills the view."""
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    buf = np.empty_like(src)
    out = buf[::-1]
    written = _capture_engine_output(monkeypatch)
    image = _scaled(Array(src))
    got = image.realize(out=out)
    assert got is out
    assert len(written) == 1 and written[0] is not out
    np.testing.assert_array_equal(out, _scaled(src))
    assert image._data is not None and image._data is not out
    assert not image._data.flags.writeable
    assert out.flags.writeable
    again = image.realize()
    assert len(written) == 1
    assert again is image._data


def test_an_already_realized_array_copies_into_out(monkeypatch):
    """The sealed cache is copied into ``out`` and stays the cache."""
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    image = _scaled(Array(src))
    cached = image.realize()
    out = np.empty_like(cached)
    written = _capture_engine_output(monkeypatch)
    got = image.realize(out=out)
    assert written == []
    assert got is out
    np.testing.assert_array_equal(out, cached)
    assert image._data is cached
    assert not cached.flags.writeable
    assert out.flags.writeable


def test_an_already_realized_array_casts_into_out(monkeypatch):
    """A cached uint8 result is cast into a float32 ``out`` by one engine run
    of ``astype`` that reads the cache and writes ``out``."""
    src = np.arange(12, dtype=np.uint8).reshape(3, 4)
    image = _scaled(Array(src))
    cached = image.realize()
    out = np.empty(cached.shape, np.float32)
    written = _capture_engine_output(monkeypatch)
    got = image.realize(out=out)
    assert written == [out]
    assert got is out
    np.testing.assert_array_equal(out, cached.astype(np.float32))
    assert image._data is cached


def test_a_cast_into_out_does_not_depend_on_the_cache():
    """An overflowing ``same_kind`` cast gives ``astype``'s result whether or
    not the array was realized before."""
    src = np.array([[0, 255, 256, 1000, 65535]], np.uint16)
    image = Array(src) * np.uint16(1)
    expected = image.astype(np.uint8).realize()
    uncached = np.empty(src.shape, np.uint8)
    image.realize(out=uncached)
    image.realize()
    cached = np.empty(src.shape, np.uint8)
    image.realize(out=cached)
    np.testing.assert_array_equal(uncached, expected)
    np.testing.assert_array_equal(cached, expected)


def test_force_recompute_writes_out_and_drops_the_stale_cache():
    """``force_recompute`` reruns into ``out`` and the next realize sees that."""
    src = np.arange(12, dtype=np.float32).reshape(3, 4).copy()
    image = _scaled(Array(src))
    cached = image.realize()
    src.setflags(write=True)
    src[...] = 1
    out = np.empty_like(src)
    got = image.realize(out=out, force_recompute=True)
    assert got is out
    np.testing.assert_array_equal(out, np.full(src.shape, 2, np.float32))
    assert image._data is None
    again = image.realize()
    np.testing.assert_array_equal(again, out)
    assert not np.array_equal(again, cached)


def test_a_reshape_copies_into_out(monkeypatch):
    """A reshape-only array is ``np.reshape`` of its source, then copied into ``out``."""
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    image = Array(src).reshape(2, 6)
    out = np.empty((2, 6), np.float32)
    written = _capture_engine_output(monkeypatch)
    got = image.realize(out=out)
    assert written == []
    assert got is out
    np.testing.assert_array_equal(out, src.reshape(2, 6))
    assert image._data is not out
    assert not image._data.flags.writeable
    assert image.realize() is image._data


def test_realize_without_out_is_unchanged():
    """``realize()`` still returns the sealed cache and reuses it."""
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    image = _scaled(Array(src))
    got = image.realize()
    np.testing.assert_array_equal(got, _scaled(src))
    assert not got.flags.writeable
    assert image.realize() is got
