"""Fusion of ops a user chains: two ``take``, two nearest ``resize``.

The engine merges a nearest resample into the nearest resample that
writes its input when only the rows change, the producer's columns are
cheap to write, and nothing else reads the producer. A fused pair must
give exactly what the two ops give one at a time, so each resize pair is
checked against the same ops with a ``.realize()`` between them.

The engine timing rows show whether a pair fused: a fused node has one
row, named for both ops.
"""

from __future__ import annotations

import numpy as np
import pytest

import muimage as mi
from muraw.common import PerfTimer
from muraw.engines.graph import EngineTiming, engine_timing, set_engine_timing


def _image(shape, dtype=np.float32):
    rng = np.random.default_rng(3)
    if np.dtype(dtype).kind == "f":
        return rng.random(shape).astype(dtype)
    return rng.integers(0, 256, shape).astype(dtype)


def _realized_with_timing_rows(result):
    """``result`` realized, and the names of its engine timing rows."""
    prev = engine_timing
    try:
        set_engine_timing(EngineTiming.OPS)
        with PerfTimer("root") as root:
            realized = result.realize()
    finally:
        set_engine_timing(prev)
    return realized, [c.name for c in root.children[0].children]


def _ops_of(rows):
    """The op names of timing rows, without the ``(engine, ...)`` suffix."""
    return [row.split(" (engine")[0] for row in rows]


def _columns_then_rows(lib, x):
    """Columns 3, 0, 7, 7, 1 of ``x``, then rows 5, 0, 0, 2 of those."""
    return lib.take(lib.take(x, [3, 0, 7, 7, 1], axis=1), [5, 0, 0, 2], axis=0)


def _rows_at_generated_indices(lib, x):
    """The rows of ``x`` at indices that are computed as an array."""
    return lib.take(x, lib.arange(4, dtype=np.uint16) * np.uint16(2), axis=0)


def _rows_at_float_indices(lib, x):
    """``x`` at a float32 index array, which ``np.take`` rejects."""
    return lib.take(x, lib.arange(3, dtype=np.float32), axis=0)


def _nearest(size, **kwargs):
    return lambda image: mi.resize(image, size, interpolation="nearest", **kwargs)


def test_a_take_of_rows_after_a_take_of_cheap_columns_is_one_node():
    x = _image((8, 12, 3))
    got, rows = _realized_with_timing_rows(_columns_then_rows(mi, mi.Array(x)))
    np.testing.assert_array_equal(got, _columns_then_rows(np, x))
    assert rows == ["take + take (engine)"]


def test_a_take_of_rows_after_a_take_of_scattered_columns_stays_two_nodes():
    x = _image((8, 12), np.uint8)
    got, rows = _realized_with_timing_rows(_columns_then_rows(mi, mi.Array(x)))
    np.testing.assert_array_equal(got, _columns_then_rows(np, x))
    assert rows == ["take (engine)", "take (engine, passed through)"]


@pytest.mark.parametrize(
    "first, second, fused_ops",
    [
        (_nearest((4, 12)), _nearest((8, 12)), "resize (nearest) + resize (nearest)"),
        (_nearest((4, 12)), _nearest((2, 12)), "resize (nearest) + resize (nearest)"),
        (_nearest((24, 12)), _nearest((8, 12)), "resize (nearest) + resize (nearest)"),
        (_nearest((12, 12)), _nearest((24, 12)), "resize (nearest) + resize (nearest)"),
        (
            lambda image: image.take([5, 1, 1, 6], axis=0),
            _nearest((8, 12)),
            "take + resize (nearest)",
        ),
    ],
    ids=[
        "rows 2:1 then 1:2",
        "rows 2:1 then 2:1",
        "rows 1:3 then 3:1",
        "rows 1:1.5 then 1:2",
        "take rows then rows 1:2",
    ],
)
def test_a_nearest_pair_that_only_changes_rows_is_one_node(first, second, fused_ops):
    x = _image((8, 12, 3))
    got, rows = _realized_with_timing_rows(second(first(mi.Array(x))))
    one_at_a_time = second(mi.Array(first(mi.Array(x)).realize())).realize()
    np.testing.assert_array_equal(got, one_at_a_time)
    assert _ops_of(rows) == [fused_ops]


_TWO_RESIZES = ["resize (nearest)", "resize (nearest)"]


@pytest.mark.parametrize(
    "first, second, ops",
    [
        (_nearest((4, 6)), _nearest((8, 12)), _TWO_RESIZES),
        (
            _nearest((4, 12)),
            _nearest((6, 12), scale=(2.0, 1.0), mode="constant"),
            _TWO_RESIZES,
        ),
        # The first resize is a column node, which pads, and a row grow;
        # the second fuses with the row grow only.
        (
            _nearest((20, 12), scale=(2.0, 1.0), mode="constant", constant_values=0.5),
            _nearest((10, 12)),
            ["resize (nearest)", "resize (nearest) + resize (nearest)"],
        ),
    ],
    ids=[
        "both axes 2:1 then 1:2",
        "rows 2:1 then rows past its output",
        "rows past the input then 2:1",
    ],
)
def test_a_nearest_pair_stays_two_nodes_when_it_changes_columns_or_reads_a_pad(
    first, second, ops
):
    x = _image((8, 12, 3))
    got, rows = _realized_with_timing_rows(second(first(mi.Array(x))))
    one_at_a_time = second(mi.Array(first(mi.Array(x)).realize())).realize()
    np.testing.assert_array_equal(got, one_at_a_time)
    assert _ops_of(rows) == ops


def test_take_at_indices_computed_by_a_graph_matches_numpy():
    x = _image((9, 5, 3))
    got = _rows_at_generated_indices(mi, mi.Array(x))
    np.testing.assert_array_equal(got.realize(), _rows_at_generated_indices(np, x))


def test_take_at_a_float_index_array_raises_like_numpy():
    x = _image((4, 5))
    with pytest.raises(TypeError):
        _rows_at_float_indices(np, x)
    with pytest.raises(TypeError):
        _rows_at_float_indices(mi, mi.Array(x))
