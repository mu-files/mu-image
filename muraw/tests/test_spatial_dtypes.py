"""view, pad, orientation, and fill run natively on every supported dtype."""

from __future__ import annotations

import numpy as np
import pytest

import muimage as mi
from muraw.array import Array, SUPPORTED_DTYPES
from muraw.engines.core import _engine_load


def _src(dtype: np.dtype) -> np.ndarray:
    return np.arange(5 * 7, dtype=np.uint8).reshape(5, 7).astype(dtype)


def _realize_ops(monkeypatch, array: Array) -> tuple[np.ndarray, list[str]]:
    calls: list[dict] = []
    real = _engine_load.execute_graph

    def wrap(graph, in_binds, out_binds, record_ops=False):
        calls.append(graph)
        return real(graph, in_binds, out_binds, record_ops)

    monkeypatch.setattr(_engine_load, "execute_graph", wrap)
    out = array.realize()
    assert len(calls) == 1
    return out, [node["op"] for node in calls[0]["nodes"]]


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES, ids=lambda dt: dt.name)
def test_view_all_dtypes(dtype, monkeypatch):
    src = _src(dtype)
    got, ops = _realize_ops(monkeypatch, Array(src).view(left=1, top=1, width=3, height=2))
    assert ops == ["view"]
    np.testing.assert_array_equal(got, src[1:3, 1:4])


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES, ids=lambda dt: dt.name)
def test_pad_all_dtypes(dtype, monkeypatch):
    src = _src(dtype)
    got, ops = _realize_ops(
        monkeypatch, Array(src).pad(1, mode="constant", constant_values=9)
    )
    assert ops == ["pad"]
    np.testing.assert_array_equal(
        got, np.pad(src, 1, mode="constant", constant_values=9)
    )


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES, ids=lambda dt: dt.name)
def test_orientation_all_dtypes(dtype, monkeypatch):
    src = _src(dtype)
    got, ops = _realize_ops(
        monkeypatch, mi.orientation(Array(src), orientation=6)
    )
    assert ops == ["orientation"]
    np.testing.assert_array_equal(got, np.ascontiguousarray(np.rot90(src, -1)))


@pytest.mark.parametrize("dtype", SUPPORTED_DTYPES, ids=lambda dt: dt.name)
def test_fill_all_dtypes(dtype, monkeypatch):
    got, ops = _realize_ops(monkeypatch, mi.full((3, 4), 7, dtype=dtype))
    assert ops == ["fill"]
    np.testing.assert_array_equal(got, np.full((3, 4), 7, dtype=dtype))
