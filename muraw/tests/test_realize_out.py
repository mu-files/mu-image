"""``realize(out=)`` writes the caller's buffer, or copies into it.

The engine writes ``out`` itself when that buffer is one it can bind,
including a layout ``Array`` reads without a copy. Anything else,
including a stride that skips elements, is the engine's own buffer
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


def _flipped_scaled(lib, image):
    """``image`` with its rows reversed, times two, plus one. The first
    output row reads the last input row, so a result written straight
    into the input's memory would overwrite rows later output rows read."""
    return image[::-1] * np.float32(2) + np.float32(1)


def test_an_input_can_be_out():
    """``out`` may be the input itself, as with NumPy's ``out=``."""
    src = np.arange(5 * 6, dtype=np.float32).reshape(5, 6).copy()
    expected = _flipped_scaled(np, src.copy())
    image = _flipped_scaled(mi, Array(src))
    src.setflags(write=True)
    assert image.realize(out=src) is src
    np.testing.assert_array_equal(src, expected)


def test_a_view_that_overlaps_an_input_can_be_out():
    """``out`` may share some of an input's memory, here half its columns."""
    parent = np.arange(5 * 8, dtype=np.float32).reshape(5, 8).copy()
    src = parent[:, :4]
    expected = _flipped_scaled(np, src.copy())
    image = _flipped_scaled(mi, Array(src))
    parent.setflags(write=True)
    out = parent[:, 2:6]
    assert image.realize(out=out) is out
    np.testing.assert_array_equal(out, expected)


def test_an_input_with_channels_first_can_be_out():
    """A ``(C, H, W)`` input, held one frame per channel, can be ``out``.
    Reversing axis 0 reverses the frames, so the first output frame reads
    the last input frame."""
    src = np.arange(3 * 5 * 6, dtype=np.float32).reshape(3, 5, 6).copy()
    expected = _flipped_scaled(np, src.copy())
    image = _flipped_scaled(mi, Array(src, channel_axis=0))
    src.setflags(write=True)
    assert image.realize(out=src) is src
    np.testing.assert_array_equal(src, expected)


def test_a_realized_input_can_be_out():
    """Pixels an earlier realize cached are an input of the graph that
    reads them, and can be its ``out``."""
    src = np.arange(5 * 6, dtype=np.float32).reshape(5, 6)
    earlier = _scaled(Array(src))
    cached = earlier.realize()
    expected = _flipped_scaled(np, cached.copy())
    later = _flipped_scaled(mi, earlier)
    cached.setflags(write=True)
    assert later.realize(out=cached) is cached
    np.testing.assert_array_equal(cached, expected)


def _assert_layout_write(image, out, expected, monkeypatch):
    """The engine's output shares memory with ``out``, and ``out`` holds ``expected``."""
    written = _capture_engine_output(monkeypatch)
    got = image.realize(out=out)
    assert got is out
    assert len(written) == 1 and np.shares_memory(written[0], out)
    assert out.flags.writeable
    assert image._data is None
    np.testing.assert_array_equal(out, expected)
    return written[0]


def _flip(parent, kind):
    if kind == "rows":
        return parent[::-1]
    if kind == "cols":
        return parent[:, ::-1]
    if kind == "both":
        return parent[::-1, ::-1]
    if kind == "channels":
        return parent[..., ::-1]
    raise ValueError(kind)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize("kind", ["rows", "cols", "both"])
def test_a_whole_axis_flip_is_the_buffer_the_engine_writes(dtype, kind, monkeypatch):
    """A span op writes a row flip, a column flip, or both, with no copy."""
    src = np.arange(4 * 6, dtype=dtype).reshape(4, 6)
    parent = np.empty(src.shape, dtype=dtype)
    out = _flip(parent, kind)
    _assert_layout_write(_scaled(Array(src)), out, _scaled(src), monkeypatch)
    assert parent.flags.writeable


@pytest.mark.parametrize("dtype", [np.uint8, np.float32])
def test_reversed_channels_are_the_buffer_the_engine_writes(dtype, monkeypatch):
    """A channel reversal is a flip of the last axis, written into the parent."""
    src = np.arange(4 * 5 * 3, dtype=dtype).reshape(4, 5, 3)
    parent = np.empty(src.shape, dtype=dtype)
    out = _flip(parent, "channels")
    _assert_layout_write(_scaled(Array(src)), out, _scaled(src), monkeypatch)
    np.testing.assert_array_equal(parent, _scaled(src)[..., ::-1])


def test_an_rgb_result_is_written_into_bgr(monkeypatch):
    """A packed RGB result is written into ``bgr`` through ``bgr[..., ::-1]``."""
    rgb = np.arange(2 * 3 * 3, dtype=np.uint8).reshape(2, 3, 3)
    bgr = np.empty(rgb.shape, np.uint8)
    out = bgr[..., ::-1]
    _assert_layout_write(_scaled(Array(rgb)), out, _scaled(rgb), monkeypatch)
    np.testing.assert_array_equal(bgr, _scaled(rgb)[..., ::-1])
    assert bgr.flags.writeable


def _layout_case(layout):
    """``(source, channel_axis, buffer, out)`` for one ``out`` layout.

    ``buffer`` is the caller's memory, filled with ``-1``, and ``out`` the
    part of it the result goes to: the whole buffer, or slot 1 of a batch.
    """
    height, width, channels = 4, 5, 3
    packed = np.arange(height * width * channels, dtype=np.float32).reshape(
        height, width, channels
    )
    chw = np.ascontiguousarray(np.moveaxis(packed, -1, 0))
    if layout == "HWC":
        buffer = np.full(packed.shape, -1, np.float32)
        return packed, 2, buffer, buffer
    if layout == "CHW":
        buffer = np.full(chw.shape, -1, np.float32)
        return chw, 0, buffer, buffer
    if layout == "NHWC slot":
        buffer = np.full((2,) + packed.shape, -1, np.float32)
        return packed, 2, buffer, buffer[1]
    if layout == "NCHW slot":
        buffer = np.full((2,) + chw.shape, -1, np.float32)
        return chw, 0, buffer, buffer[1]
    if layout == "HWC into an NCHW slot":
        buffer = np.full((2,) + chw.shape, -1, np.float32)
        return packed, 2, buffer, np.moveaxis(buffer[1], 0, -1)
    raise ValueError(layout)


def _flipped(out, channel_axis, kind):
    """``out`` with its rows or its columns reversed, or ``out`` itself."""
    if kind == "whole":
        return out
    rows_axis, cols_axis = (axis for axis in range(3) if axis != channel_axis)
    axis = rows_axis if kind == "rows" else cols_axis
    return np.flip(out, axis)


@pytest.mark.parametrize("kind", ["whole", "rows", "cols"])
@pytest.mark.parametrize(
    "layout", ["HWC", "CHW", "NHWC slot", "NCHW slot", "HWC into an NCHW slot"]
)
def test_each_layout_and_its_flips_are_the_buffer_the_engine_writes(
    layout, kind, monkeypatch
):
    """The engine writes ``out``'s memory for each layout, flipped or not, and
    a batch slot leaves the other slot alone."""
    src, channel_axis, buffer, slot = _layout_case(layout)
    out = _flipped(slot, channel_axis, kind)
    _assert_layout_write(
        _scaled(Array(src, channel_axis=channel_axis)), out, _scaled(src), monkeypatch
    )
    if buffer.ndim == 4:
        np.testing.assert_array_equal(buffer[0], -1)


@pytest.mark.parametrize("kind", ["whole", "frames"])
def test_a_4_axis_out_is_the_buffer_the_engine_writes(kind, monkeypatch):
    """An ``(N, H, W, C)`` result is written into an ``(N, H, W, C)`` ``out``,
    and into one whose frames are reversed."""
    src = np.arange(2 * 4 * 5 * 3, dtype=np.float32).reshape(2, 4, 5, 3)
    buffer = np.empty(src.shape, np.float32)
    out = buffer if kind == "whole" else buffer[::-1]
    _assert_layout_write(_scaled(Array(src)), out, _scaled(src), monkeypatch)


def test_a_flip_after_orientation_is_the_buffer_the_engine_writes(monkeypatch):
    """The engine writes ``out`` when a row flip follows an orientation."""
    src = np.arange(4 * 6, dtype=np.float32).reshape(4, 6)
    image = mi.orientation(_scaled(Array(src)), orientation=6)
    parent = np.empty(image.shape, np.float32)
    out = parent[::-1]
    _assert_layout_write(image, out, np.rot90(_scaled(src), -1), monkeypatch)


def test_a_moveaxis_out_is_the_buffer_the_engine_writes(monkeypatch):
    """``moveaxis`` of a CHW buffer is written as that buffer, with a transpose."""
    src = np.arange(4 * 6 * 3, dtype=np.float32).reshape(4, 6, 3)
    chw = np.empty((3, 4, 6), np.float32)
    out = np.moveaxis(chw, 0, -1)
    written = _assert_layout_write(_scaled(Array(src)), out, _scaled(src), monkeypatch)
    assert written.shape == chw.shape
    np.testing.assert_array_equal(chw, np.moveaxis(_scaled(src), -1, 0))
    assert chw.flags.writeable


def test_a_rot90_out_is_the_buffer_the_engine_writes(monkeypatch):
    """``np.rot90`` is a flip and a permutation, and the engine writes the parent."""
    src = np.arange(6 * 4 * 3, dtype=np.float32).reshape(6, 4, 3)
    parent = np.empty((4, 6, 3), np.float32)
    out = np.rot90(parent)
    assert out.shape == src.shape
    _assert_layout_write(_scaled(Array(src)), out, _scaled(src), monkeypatch)
    np.testing.assert_array_equal(parent, np.rot90(_scaled(src), -1))


def test_a_transpose_out_is_the_buffer_the_engine_writes(monkeypatch):
    """A 2D ``out`` that is ``parent.T`` is written as ``parent``."""
    src = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)
    parent = np.empty((5, 4), np.float32)
    out = parent.T
    _assert_layout_write(_scaled(Array(src)), out, _scaled(src), monkeypatch)
    np.testing.assert_array_equal(parent, _scaled(src).T)


def test_a_negative_stride_is_copied_into(monkeypatch):
    """A flipped ``out`` that skips elements is not the engine's buffer."""
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    buf = np.zeros((3, 8), np.float32)
    out = buf[:, ::-2]
    written = _capture_engine_output(monkeypatch)
    image = _scaled(Array(src))
    got = image.realize(out=out)
    assert got is out
    assert len(written) == 1 and not np.shares_memory(written[0], out)
    np.testing.assert_array_equal(out, _scaled(src))
    np.testing.assert_array_equal(buf[:, ::2], 0)
    assert image._data is not None and image._data is not out
    assert not image._data.flags.writeable
    assert out.flags.writeable
    again = image.realize()
    assert len(written) == 1
    assert again is image._data


def test_an_already_realized_flip_is_copied(monkeypatch):
    """A cache hit copies into a flipped ``out`` and does not run the engine."""
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    image = _scaled(Array(src))
    cached = image.realize()
    parent = np.empty_like(cached)
    out = parent[::-1]
    written = _capture_engine_output(monkeypatch)
    got = image.realize(out=out)
    assert written == []
    assert got is out
    np.testing.assert_array_equal(out, cached)
    assert image._data is cached
    assert parent.flags.writeable


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
