"""broadcast_to stretches size-1 axes by calling tile."""

from __future__ import annotations

import numpy as np
import pytest

import muimage as mi
from muraw.array import Array


def _ops(array: Array) -> list[tuple[str, dict]]:
    found = []
    current = array
    while current._node is not None:
        found.append((current._node.op, dict(current._node.attrs)))
        current = current._node.inputs[0]
    return found


@pytest.mark.parametrize(
    ("src", "shape"),
    [
        (np.arange(6, dtype=np.float32).reshape(1, 6), (4, 6)),
        (np.arange(6, dtype=np.float32), (4, 6)),
        (np.arange(4, dtype=np.float32).reshape(4, 1), (4, 6)),
        (np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32), (4, 6, 3)),
        (np.ones((1, 1), dtype=np.float32), (2, 4, 3)),
        (np.arange(24, dtype=np.float32).reshape(4, 6, 1), (4, 6, 3)),
        (np.array([10.0, 20.0, 30.0], dtype=np.float32), (2, 4, 3)),
        (np.array([[10.0, 20.0, 30.0]], dtype=np.float32), (2, 4, 3)),
        (np.arange(3, dtype=np.float32).reshape(3, 1), (2, 3, 4)),
    ],
    ids=[
        "row",
        "vector_as_row",
        "column",
        "color",
        "gray_pixel",
        "channel",
        "vector_as_channels",
        "row_as_channels",
        "column_as_width",
    ],
)
def test_broadcast_to_matches_numpy(src, shape):
    expected = np.broadcast_to(src, shape)
    concrete = mi.broadcast_to(src, shape)
    lazy = mi.broadcast_to(mi.Array(src) * 2, shape)
    assert concrete.shape == expected.shape
    assert lazy.shape == expected.shape
    np.testing.assert_array_equal(concrete.realize(), expected)
    np.testing.assert_array_equal(lazy.realize(), expected * 2)
    ingested = Array(np.broadcast_to(src, shape))
    assert _ops(ingested) == _ops(concrete)


def test_tile_of_vector_matches_numpy():
    src = np.arange(8, dtype=np.float32)
    out = mi.tile(src, (1, 1, 2))
    expected = np.tile(src, (1, 1, 2))
    assert out.shape == expected.shape
    np.testing.assert_array_equal(out.realize(), expected)


def test_broadcast_to_rejects_mismatched_axis():
    src = np.ones((8, 8, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="cannot broadcast"):
        mi.broadcast_to(src, (10, 10, 3))
    with pytest.raises(ValueError, match="cannot broadcast"):
        mi.broadcast_to(np.ones((4, 6), dtype=np.float32), (4, 6, 3))


def test_broadcast_to_rejects_fewer_destination_axes():
    with pytest.raises(ValueError, match="more axes"):
        mi.broadcast_to(np.ones((1, 5), dtype=np.float32), (5,))


def test_broadcast_ingest_shares_memory():
    row = np.arange(8, dtype=np.float32)
    column = np.arange(4, dtype=np.float32).reshape(4, 1)
    pixel = np.array([[[1.0, 2.0, 3.0]]], dtype=np.float32)
    plane = np.arange(8, dtype=np.float32).reshape(2, 4, 1)
    cases = [
        (row, (4, 8)),
        (column, (4, 6)),
        (pixel, (3, 5, 3)),
        (plane, (2, 4, 3)),
    ]
    for src, shape in cases:
        broadcast = np.broadcast_to(src, shape)
        array = Array(broadcast)
        assert array._node is not None and array._node.op == "tile"
        assert np.shares_memory(array._node.inputs[0]._data, src)
        np.testing.assert_array_equal(array.realize(), broadcast)


def test_sliced_broadcast_matches_numpy():
    stretched = np.broadcast_to(np.arange(6, dtype=np.float32), (4, 6))[::2, 1:]
    array = Array(stretched)
    assert array._node is not None and array._node.op == "tile"
    np.testing.assert_array_equal(array.realize(), stretched)


def test_downward_row_repeat_runs_no_tile_kernel():
    from muraw.common import PerfTimer
    from muraw.engines.graph import EngineTiming, engine_timing, set_engine_timing

    row = np.arange(32, dtype=np.float32)
    column = np.arange(64, dtype=np.float32).reshape(64, 1)
    prev = engine_timing
    try:
        set_engine_timing(EngineTiming.OPS)
        with PerfTimer("down") as down_timer:
            down = mi.broadcast_to(row, (64, 32)).realize()
        with PerfTimer("across") as across_timer:
            across = mi.broadcast_to(column, (64, 256)).realize()
    finally:
        set_engine_timing(prev)

    np.testing.assert_array_equal(down, np.broadcast_to(row, (64, 32)))
    np.testing.assert_array_equal(across, np.broadcast_to(column, (64, 256)))
    assert [child.name for child in down_timer.children[0].children] == ["tile (engine)"]
    assert down_timer.children[0].children[0].get_elapsed_ms() == 0.0
    assert across_timer.children[0].children[0].get_elapsed_ms() > 0.0
