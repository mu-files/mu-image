"""Arrays with an axis of length 0, against NumPy.

Every call on an empty array gives NumPy's result or raises
``NotImplementedError``, and a call whose NumPy result is empty gives it.
When NumPy raises, ``mi`` raises the same exception type. An empty array
is never sent to an engine.
"""

from __future__ import annotations

import numpy as np
import pytest

import muimage as mi
from muraw.array import ArrayMeta
from muraw.engines.graph import OpNode, get_default_engine

# (shape, channel_axis) of each empty array, on every layout mi holds.
EMPTY = {
    "no_rows": ((0, 5), None),
    "no_columns": ((4, 0), None),
    "hwc_no_rows": ((0, 5, 3), 2),
    "hwc_no_channels": ((4, 5, 0), 2),
    "chw_no_rows": ((3, 0, 5), 0),
    "chw_no_channels": ((0, 4, 5), 0),
    "stack_no_frames": ((0, 4, 5, 3), 3),
    "stack_no_columns": ((2, 4, 0, 3), 3),
    "1d": ((0,), None),
}


def _pad(lib, x, width, **kwargs):
    """``np.pad(x, width)``; an ``mi.Array`` pads with its method."""
    return np.pad(x, width, **kwargs) if lib is np else x.pad(width, **kwargs)


CALLS = {
    "add": lambda lib, x: x + 1,
    "multiply": lambda lib, x: x * x,
    "subtract_reversed": lambda lib, x: 2 - x,
    "divide": lambda lib, x: x / 2,
    "astype": lambda lib, x: x.astype(np.uint8),
    "reshape_flat": lambda lib, x: x.reshape(-1),
    "transpose": lambda lib, x: x.T,
    "swapaxes": lambda lib, x: lib.swapaxes(x, 0, -1),
    "moveaxis": lambda lib, x: lib.moveaxis(x, -1, 0),
    "reverse": lambda lib, x: x[::-1],
    "every_other": lambda lib, x: x[..., ::2],
    "inserted_axis": lambda lib, x: x[None],
    "take_first": lambda lib, x: lib.take(x, [0], axis=-1),
    "take_none": lambda lib, x: lib.take(x, [], axis=0),
    "repeat": lambda lib, x: lib.repeat(x, 2, axis=0),
    "repeat_flat": lambda lib, x: lib.repeat(x, 3),
    "tile": lambda lib, x: lib.tile(x, 2),
    "tile_zero": lambda lib, x: lib.tile(x, 0),
    "concatenate_first_axis": lambda lib, x: lib.concatenate([x, x], axis=0),
    "concatenate_last_axis": lambda lib, x: lib.concatenate([x, x], axis=-1),
    "stack": lambda lib, x: lib.stack([x, x]),
    "vstack": lambda lib, x: lib.vstack([x, x]),
    "hstack": lambda lib, x: lib.hstack([x, x]),
    "dstack": lambda lib, x: lib.dstack([x, x]),
    "broadcast_to": lambda lib, x: lib.broadcast_to(x, x.shape),
    "zeros_like": lambda lib, x: lib.zeros_like(x),
    "full_like": lambda lib, x: lib.full_like(x, 3),
    "pad_constant": lambda lib, x: _pad(lib, x, 1, constant_values=4),
    "pad_edge": lambda lib, x: _pad(lib, x, 1, mode="edge"),
    "matmul": lambda lib, x: x @ np.ones((x.shape[-1], 2), np.float32),
}


def _check(call, x, source):
    """``call`` with NumPy on ``x`` and with mi on ``source``. Returns a
    description of how they differ, or ``None``."""
    try:
        expect = call(np, x)
    except Exception as numpy_error:
        try:
            call(mi, source)
        except type(numpy_error):
            return None
        except Exception as error:
            return f"NumPy raises {numpy_error!r}, mi {error!r}"
        return f"NumPy raises {numpy_error!r}, mi does not"
    try:
        got = call(mi, source).realize()
    except NotImplementedError as error:
        if expect.size == 0 and expect.ndim <= 4:
            return f"NumPy gives empty {expect.shape}, mi {error!r}"
        return None
    except Exception as error:
        return f"NumPy gives {expect.shape}, mi raises {error!r}"
    if got.shape != expect.shape or not np.array_equal(got, expect):
        return f"NumPy gives {expect.shape}, mi {got.shape}"
    return None


@pytest.mark.parametrize("layout", sorted(EMPTY))
def test_every_call_on_an_empty_array_matches_numpy(layout):
    shape, channel_axis = EMPTY[layout]
    x = np.zeros(shape, np.float32)
    source = mi.Array(x) if channel_axis is None else mi.Array(x, channel_axis=channel_axis)
    differ = {name: _check(call, x, source) for name, call in CALLS.items()}
    differ = {name: why for name, why in differ.items() if why is not None}
    assert not differ, "\n".join(f"{name}: {why}" for name, why in differ.items())


def _join(lib, pieces, axis):
    return lib.concatenate(pieces, axis=axis)


@pytest.mark.parametrize(
    "shapes, axis",
    [
        ([(3, 4), (0, 4), (2, 4)], 0),
        ([(3, 0), (3, 4)], 1),
        ([(2, 3, 0), (2, 3, 2), (2, 3, 0)], 2),
        ([(4,), (0,), (3,)], 0),
    ],
)
def test_a_join_skips_its_empty_pieces(shapes, axis):
    pieces = [np.arange(np.prod(shape), dtype=np.float32).reshape(shape) + 1 for shape in shapes]
    got = _join(mi, [mi.Array(piece) for piece in pieces], axis)
    np.testing.assert_array_equal(got.realize(), _join(np, pieces, axis))


def _accumulate(lib, rows):
    """Rows appended one at a time to an array that starts empty."""
    out = lib.zeros((0, rows.shape[1]), np.float32)
    for index in range(rows.shape[0]):
        out = lib.concatenate([out, rows[index : index + 1]], axis=0)
    return out


def test_an_accumulator_that_starts_empty_matches_numpy():
    rows = np.arange(12, dtype=np.float32).reshape(4, 3)
    np.testing.assert_array_equal(_accumulate(mi, mi.Array(rows)).realize(), _accumulate(np, rows))


@pytest.mark.parametrize(
    "shape, width, constant_values",
    [
        ((0, 3), ((1, 2), (1, 1)), ((5, 6), (7, 8))),
        ((2, 0), 2, 9),
        ((2, 3, 0), ((0, 0), (0, 0), (1, 1)), 4),
        ((0, 2, 3), ((1, 0), (1, 1), (0, 0)), ((5, 6), (7, 8), (0, 0))),
    ],
)
def test_a_constant_pad_of_an_empty_array_matches_numpy(shape, width, constant_values):
    x = np.zeros(shape, np.float32)
    channel_axis = 0 if len(shape) == 3 and shape[0] == 0 else -1
    got = _pad(mi, mi.Array(x, channel_axis=channel_axis), width, constant_values=constant_values)
    np.testing.assert_array_equal(got.realize(), _pad(np, x, width, constant_values=constant_values))


def test_a_pad_that_reads_an_empty_axis_raises_what_numpy_raises():
    x = np.zeros((0, 3), np.float32)
    for lib, array in ((np, x), (mi, mi.Array(x))):
        with pytest.raises(ValueError):
            _pad(lib, array, 1, mode="edge")


def _matmul(lib, x, matrix):
    return x @ matrix


@pytest.mark.parametrize(
    "shape, matrix_shape", [((4, 5, 0), (0, 3)), ((4, 5, 3), (3, 0)), ((0,), (0, 2))]
)
def test_matmul_over_no_channels_matches_numpy(shape, matrix_shape):
    x = np.zeros(shape, np.float32)
    matrix = np.ones(matrix_shape, np.float32)
    got = _matmul(mi, mi.Array(x), matrix)
    np.testing.assert_array_equal(got.realize(), _matmul(np, x, matrix))


def test_realize_into_an_empty_out_returns_it():
    out = np.zeros((0, 4), np.float32)
    assert (mi.zeros((5, 4))[2:2] + 1).realize(out=out) is out


def test_an_op_from_an_empty_input_to_a_full_result_raises():
    empty = mi.Array(np.zeros((0, 3), np.float32))
    with pytest.raises(NotImplementedError, match="is empty and the result"):
        mi.pad(empty, top=1, bottom=1, left=0, right=0)


def test_the_core_engine_rejects_an_empty_array():
    meta = ArrayMeta(shape=(0, 3), dtype=np.dtype(np.float32))
    empty = mi.Array(
        _meta=meta,
        _node=OpNode(op="fill", inputs=(), attrs={"value": [0.0]}, out_meta=meta),
    )
    with pytest.raises(NotImplementedError, match="empty"):
        get_default_engine().execute([empty], {}, empty)


def test_an_empty_result_has_no_node():
    x = mi.Array(np.zeros((4, 5), np.float32))
    empty = (x[1:1] * 2).astype(np.uint8)
    assert empty._node is None and empty.realize().shape == (0, 5)
