"""Array accepts any C >= 1. Listed ops run; others reject a bad C."""

from __future__ import annotations

import numpy as np
import pytest

import muimage as mi
from muraw.array import Array
from muraw.engines.core import _engine_load


def _hwc(channels: int) -> np.ndarray:
    return np.arange(4 * 6 * channels, dtype=np.float32).reshape(4, 6, channels) + 1.0


def _realize_graph(monkeypatch, array: Array) -> tuple[np.ndarray, dict]:
    calls: list[dict] = []
    real = _engine_load.execute_graph

    def wrap(graph, in_binds, out_binds, record_ops=False):
        calls.append(graph)
        return real(graph, in_binds, out_binds, record_ops)

    monkeypatch.setattr(_engine_load, "execute_graph", wrap)
    out = array.realize()
    assert len(calls) == 1
    return out, calls[0]


def _assert_graph_channels(graph: dict, channels: int, op: str) -> None:
    assert [node["op"] for node in graph["nodes"]] == [op]
    assert {desc["channels"] for desc in graph["tensor_descs"]} == {channels}


@pytest.mark.parametrize("channels", [2, 5])
def test_view_pad_orientation_any_channels(channels, monkeypatch):
    src = _hwc(channels)
    t = Array(src)

    viewed, view_graph = _realize_graph(
        monkeypatch, t.view(left=1, top=1, width=3, height=2)
    )
    _assert_graph_channels(view_graph, channels, "view")
    np.testing.assert_array_equal(viewed, src[1:3, 1:4])

    padded, pad_graph = _realize_graph(
        monkeypatch, t.pad(1, mode="constant", constant_values=9)
    )
    _assert_graph_channels(pad_graph, channels, "pad")
    expect_pad = np.pad(
        src, ((1, 1), (1, 1), (0, 0)), mode="constant", constant_values=9
    )
    np.testing.assert_array_equal(padded, expect_pad)

    oriented, orient_graph = _realize_graph(
        monkeypatch, mi.orientation(t, orientation=3)
    )
    _assert_graph_channels(orient_graph, channels, "orientation")
    np.testing.assert_array_equal(oriented, np.ascontiguousarray(src[::-1, ::-1]))


def _assert_gather_channels(graph: dict, src_channels: int, dest_channels: int) -> None:
    assert [node["op"] for node in graph["nodes"]] == ["view"]
    node = graph["nodes"][0]
    by_id = {desc["id"]: desc["channels"] for desc in graph["tensor_descs"]}
    assert by_id[node["inputs"][0]] == src_channels
    assert by_id[node["outputs"][0]] == dest_channels


def test_view_n_to_1(monkeypatch):
    src = _hwc(5)
    got, graph = _realize_graph(monkeypatch, Array(src)[1:3, 1:4, 2])
    _assert_gather_channels(graph, 5, 1)
    assert got.shape == (2, 3)
    np.testing.assert_array_equal(got, src[1:3, 1:4, 2])
    last = Array(src)[..., -1].realize()
    np.testing.assert_array_equal(last, src[..., -1])


def test_view_channel_cut_and_list(monkeypatch):
    src = _hwc(5)
    t = Array(src)
    via_slice, slice_graph = _realize_graph(monkeypatch, t[:, :, 1:4])
    _assert_gather_channels(slice_graph, 5, 3)
    np.testing.assert_array_equal(via_slice, src[:, :, 1:4])
    via_list, list_graph = _realize_graph(monkeypatch, t[:, :, [1, 2, 3]])
    _assert_gather_channels(list_graph, 5, 3)
    np.testing.assert_array_equal(via_list, src[:, :, [1, 2, 3]])


def test_view_rgb_to_bgr(monkeypatch):
    src = np.arange(2 * 3 * 3, dtype=np.float32).reshape(2, 3, 3) + 1.0
    t = Array(src)
    via_list, list_graph = _realize_graph(monkeypatch, t[:, :, [2, 1, 0]])
    _assert_gather_channels(list_graph, 3, 3)
    np.testing.assert_array_equal(via_list, src[:, :, ::-1])
    via_rev, rev_graph = _realize_graph(monkeypatch, t[:, :, ::-1])
    _assert_gather_channels(rev_graph, 3, 3)
    np.testing.assert_array_equal(via_rev, src[:, :, ::-1])


def test_view_rgba_to_bgra():
    src = np.arange(2 * 2 * 4, dtype=np.float32).reshape(2, 2, 4) + 1.0
    got = Array(src)[:, :, [2, 1, 0, 3]].realize()
    np.testing.assert_array_equal(got, src[:, :, [2, 1, 0, 3]])
    assert not np.array_equal(got, src[:, :, ::-1])


def test_view_rejects_spatial_fancy_and_mono_channel():
    src = _hwc(5)
    t = Array(src)
    with pytest.raises(TypeError, match="slice objects"):
        t.view(([0, 1], slice(None), slice(None)))
    with pytest.raises(TypeError, match="slice objects"):
        Array(_hwc(3))[[0, 1], :, :]
    mono = Array(np.zeros((4, 6), dtype=np.float32))
    with pytest.raises(IndexError, match="too many indices"):
        mono[:, :, 0]
    with pytest.raises(ValueError, match="empty"):
        t[:, :, 1:4:-1]
    with pytest.raises(ValueError, match="empty"):
        t[:, :, []]
    with pytest.raises(IndexError, match="out of bounds"):
        mi.view(
            t,
            left=0,
            top=0,
            width=t.meta.width,
            height=t.meta.height,
            src_channels=[7],
        )


def test_scalar_two_channels():
    src = _hwc(2)
    got = ((Array(src) - 1.0) * 2.0).realize()
    np.testing.assert_array_equal(got, (src - 1.0) * 2.0)


def test_convert_and_cast_two_channels():
    src = np.arange(3 * 4 * 2, dtype=np.uint8).reshape(3, 4, 2)
    t = Array(src)
    converted = t.convert_type("float32").realize()
    np.testing.assert_allclose(converted, src.astype(np.float32) / 255.0)
    cast = t.astype("float32").realize()
    np.testing.assert_array_equal(cast, src.astype(np.float32))


def test_matrix_3x3_rejects_two_channels():
    x = Array(np.zeros((2, 2, 2), dtype=np.float32))
    with pytest.raises(ValueError, match="expected 3 channel"):
        mi.matrix_3x3(x, matrix=np.eye(3, dtype=np.float32))
