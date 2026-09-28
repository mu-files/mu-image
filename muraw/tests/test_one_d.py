"""1D (N,) arrays follow NumPy's shape rules; the buffer underneath is one row."""

from __future__ import annotations

import numpy as np
import pytest

import muimage as mi
from muraw.array import Array, fliplr, flipud, rot90
from muraw.engines.graph import graph_op
from muraw.engines.ops import OPS_BY_NAME
from muraw.engines.pyops import cfa_demosaic_op, radial_distortion_op

V = np.arange(8, dtype=np.float32)


@pytest.mark.parametrize(
    "src",
    [
        V,
        np.arange(20, dtype=np.float32)[3:17:3],
        np.arange(20, dtype=np.float32)[::-2],
        np.arange(24, dtype=np.float32).reshape(4, 6)[:, 2],
        np.arange(24, dtype=np.float32).reshape(4, 6)[1],
    ],
    ids=["contiguous", "strided", "reversed", "column_of_2d", "row_of_2d"],
)
def test_ingest_keeps_numpy_shape(src):
    arr = Array(src)
    assert arr.shape == src.shape
    assert arr.meta.ndim == 1
    out = arr.realize()
    assert out.shape == src.shape
    np.testing.assert_array_equal(out, src)


def test_contiguous_ingest_does_not_copy():
    assert np.shares_memory(Array(V).realize(), V)


def test_elementwise_ops_keep_1d():
    out = (Array(V) * 3 - 1).astype("uint16")
    assert out.shape == (8,)
    np.testing.assert_array_equal(out.realize()[1:], (V * 3 - 1).astype(np.uint16)[1:])


def test_lut_keeps_1d():
    out = mi.lut(Array(V / 8), lut=np.linspace(0.0, 1.0, 16, dtype=np.float32))
    assert out.shape == (8,)
    assert out.realize().shape == (8,)


@pytest.mark.parametrize(
    "shape", [(5,), 5], ids=["tuple", "int"]
)
def test_constructors_accept_1d(shape):
    np.testing.assert_array_equal(mi.zeros(shape).realize(), np.zeros(shape, np.float32))
    np.testing.assert_array_equal(mi.ones(shape).realize(), np.ones(shape, np.float32))
    np.testing.assert_array_equal(
        mi.full(shape, 3, dtype="uint8").realize(), np.full(shape, 3, np.uint8)
    )


def test_like_constructors_accept_1d():
    assert mi.zeros_like(Array(V)).shape == (8,)
    assert mi.ones_like(Array(V)).realize().shape == (8,)
    assert mi.full_like(Array(V), 2.0).realize().shape == (8,)


@pytest.mark.parametrize(
    "key",
    [
        np.s_[2:5],
        np.s_[::-1],
        np.s_[1::3],
        np.s_[...],
        np.s_[None, :],
        np.s_[:, None],
        np.s_[None, 2:6],
        np.s_[None, :, None],
        np.s_[None],
    ],
    ids=["slice", "reverse", "step", "ellipsis", "row", "column", "row_slice", "row_newaxis", "none"],
)
def test_indexing_matches_numpy(key):
    out = Array(V)[key]
    assert out.shape == V[key].shape
    np.testing.assert_array_equal(out.realize(), V[key])


def test_too_many_indices_raises():
    with pytest.raises(IndexError, match="1-dimensional"):
        Array(V)[1:2, 3:4]


def test_flipud_reverses():
    np.testing.assert_array_equal(flipud(Array(V)).realize(), np.flipud(V))


@pytest.mark.parametrize("flip", [fliplr, rot90], ids=["fliplr", "rot90"])
def test_2d_only_flips_raise(flip):
    with pytest.raises(ValueError):
        flip(Array(V))


def test_transpose_is_unchanged():
    arr = Array(V)
    assert arr.T is arr
    assert arr.transpose() is arr


@pytest.mark.parametrize(
    ("pad_width", "kwargs"),
    [
        (2, {}),
        ((1, 3), {"constant_values": 9.0}),
        (((2, 1),), {}),
        (1, {"mode": "edge"}),
    ],
    ids=["int", "pair", "one_pair", "edge"],
)
def test_pad_matches_numpy(pad_width, kwargs):
    out = Array(V).pad(pad_width, **kwargs)
    expected = np.pad(V, pad_width, **kwargs)
    assert out.shape == expected.shape
    np.testing.assert_array_equal(out.realize(), expected)


@pytest.mark.parametrize("reps", [2, (4, 1), (1, 2), (3, 2)])
def test_tile_matches_numpy(reps):
    out = mi.tile(Array(V), reps)
    expected = np.tile(V, reps)
    assert out.shape == expected.shape
    np.testing.assert_array_equal(out.realize(), expected)


def test_tile_three_reps_makes_channels():
    out = mi.tile(Array(V), (1, 1, 2))
    expected = np.tile(V, (1, 1, 2))
    assert out.shape == expected.shape
    np.testing.assert_array_equal(out.realize(), expected)


def test_graph_op_sees_numpy_shape_and_may_return_1d():
    seen = []

    @graph_op
    def reverse(arr):
        seen.append(arr.shape)
        return arr[::-1].copy()

    out = reverse(Array(V))
    assert out.shape == (8,)
    np.testing.assert_array_equal(out.realize(), V[::-1])
    assert seen == [(8,)]


@pytest.mark.parametrize(
    "name", sorted(name for name, engine_op in OPS_BY_NAME.items() if engine_op.meta.requires_2d)
)
def test_requires_2d_catalog_ops_raise(name):
    with pytest.raises(ValueError, match="requires a 2D input"):
        OPS_BY_NAME[name](Array(V))


def test_requires_2d_python_ops_raise():
    with pytest.raises(ValueError, match="requires a 2D input"):
        radial_distortion_op(Array(V), k1=0.0, k2=0.0, k3=0.0, focal_length_mm=50.0)
    with pytest.raises(ValueError, match="requires a 2D input"):
        cfa_demosaic_op(Array(V), "RGGB")


def test_rgb_ops_reject_1d_by_channel_count():
    with pytest.raises(ValueError, match="expected 3 channel"):
        mi.rgb_matrix_3x3(Array(V), matrix=np.eye(3, dtype=np.float32))
