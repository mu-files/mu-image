# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files
"""Ops on a CHW array read its channel axis from the metadata."""

from __future__ import annotations

import numpy as np
import pytest

import muimage as mi
from muraw.array import Array, fliplr, flipud, rot90

_MATRIX = np.eye(3, dtype=np.float32)[[1, 2, 0]]


def _chw(height: int = 6, width: int = 9) -> np.ndarray:
    rng = np.random.default_rng(height * 100 + width)
    return rng.random((3, height, width), dtype=np.float32) * 0.5


def _op_names(t: Array) -> list[str]:
    names = []
    while t._node is not None:
        names.append(t._node.op)
        t = t._node.inputs[0]
    return names


def test_rgb_op_on_chw_raises_and_runs_after_moveaxis():
    chw = _chw()
    x = Array(chw, channel_axis=0)
    with pytest.raises(ValueError, match=r"shape \(3, 6, 9\).*mi\.moveaxis\(x, 0, -1\)"):
        mi.rgb_matrix_3x3(x, matrix=_MATRIX)
    got = mi.rgb_matrix_3x3(mi.moveaxis(x, 0, -1), matrix=_MATRIX).realize()
    np.testing.assert_allclose(got, np.moveaxis(chw, 0, -1) @ _MATRIX.T, rtol=1e-6)


@pytest.mark.parametrize("shape", [(3, 6, 9), (1, 6, 9)])
def test_cfa_op_on_chw_raises(shape):
    x = Array(np.zeros(shape, np.float32), channel_axis=0)
    with pytest.raises(ValueError, match=r"mono \(H, W\) or \(H, W, 1\)"):
        mi.cfa_bilinear_demosaic(x, cfa_pattern="RGGB")


@pytest.mark.parametrize("height, width", [(6, 9), (9, 6)])
def test_row_slice_of_chw_moves_the_origin_row(height, width):
    chw = _chw(height, width)
    got = Array(chw, channel_axis=0)[:, 2:5, :]
    assert got.meta.origin == (2, 0)
    assert got.meta.channel_axis == 0
    np.testing.assert_array_equal(got.realize(), chw[:, 2:5, :])


@pytest.mark.parametrize(
    "key",
    [
        np.s_[::2],
        np.s_[1],
        np.s_[-1, ::2],
        np.s_[[2, 0]],
        np.s_[1:2, 1:4, ::-1],
        np.s_[..., 1:3],
        np.s_[:, ::-1, 3:],
    ],
    ids=["plane_step", "plane_int", "plane_int_rows", "plane_list", "all_axes", "ellipsis", "row_flip"],
)
def test_chw_index_matches_numpy(key):
    chw = _chw()
    np.testing.assert_array_equal(Array(chw, channel_axis=0)[key].realize(), chw[key])


def test_chw_flips_match_numpy():
    chw = _chw()
    x = Array(chw, channel_axis=0)
    np.testing.assert_array_equal(fliplr(x).realize(), np.fliplr(chw))
    np.testing.assert_array_equal(flipud(x).realize(), np.flipud(chw))


@pytest.mark.parametrize("mode", ["constant", "edge", "reflect", "symmetric"])
def test_chw_pad_with_plane_widths_matches_numpy(mode):
    chw = _chw()
    pad_width = ((1, 2), (1, 0), (0, 2))
    kwargs = {"constant_values": 0.5} if mode == "constant" else {}
    got = Array(chw, channel_axis=0).pad(pad_width, mode=mode, **kwargs)
    assert got.meta.channel_axis == 0
    np.testing.assert_array_equal(got.realize(), np.pad(chw, pad_width, mode=mode, **kwargs))


@pytest.mark.parametrize("mode", ["edge", "reflect"])
def test_chw_pad_shorthand_pads_planes_like_numpy(mode):
    chw = _chw()
    got = Array(chw, channel_axis=0).pad(1, mode=mode)
    np.testing.assert_array_equal(got.realize(), np.pad(chw, 1, mode=mode))


def test_chw_pad_constants_per_axis_match_numpy():
    chw = _chw()
    pad_width = ((1, 1), (2, 2), (3, 3))
    constants = ((7, 8), (1, 2), (3, 4))
    got = Array(chw, channel_axis=0).pad(pad_width, constant_values=constants)
    np.testing.assert_array_equal(got.realize(), np.pad(chw, pad_width, constant_values=constants))


@pytest.mark.parametrize("reps", [(2, 1, 1), (2, 2, 3), (1, 2), 3], ids=str)
def test_chw_tile_matches_numpy(reps):
    chw = _chw()
    np.testing.assert_array_equal(mi.tile(Array(chw, channel_axis=0), reps).realize(), np.tile(chw, reps))


def test_chw_broadcast_matches_numpy():
    chw = _chw()
    plane = Array(chw[:1], channel_axis=0)
    np.testing.assert_array_equal(
        mi.broadcast_to(plane, (3, 6, 9)).realize(), np.broadcast_to(chw[:1], (3, 6, 9))
    )
    repeated = np.broadcast_to(chw[:1], (4, 6, 9))
    np.testing.assert_array_equal(Array(repeated, channel_axis=0).realize(), repeated)


@pytest.mark.parametrize("name", ["fix_vignette", "warp_rectilinear", "map_polynomial"])
def test_legacy_full_frame_ops_rejected_on_chw_and_run_after_moveaxis(name):
    chw = _chw(40, 48)

    def run(x: Array) -> Array:
        if name == "fix_vignette":
            return mi.fix_vignette(
                x, params=np.array([0.1, 0.0, 0.0, 0.0, 0.0]), center_x=0.5, center_y=0.5
            )
        if name == "warp_rectilinear":
            return mi.warp_rectilinear(
                x,
                radial_params=np.array([1.0, 0.0, 0.0, 0.0]),
                num_planes=1,
                num_coeffs=4,
                center_x=0.5,
                center_y=0.5,
                use_bicubic=False,
            )
        return mi.map_polynomial(
            x,
            top=0,
            left=0,
            bottom=40,
            right=48,
            start_plane=0,
            num_planes=3,
            row_pitch=1,
            col_pitch=1,
            coefficients=np.array([0.0, 1.0], np.float32),
            degree=1,
        )

    x = Array(chw, channel_axis=0)
    with pytest.raises(RuntimeError, match="GRAPH_INVALID"):
        run(x).realize()
    assert run(mi.moveaxis(x, 0, -1)).realize().shape == (40, 48, 3)


def test_planar_crop_then_moveaxis_then_rgb_op_matches_numpy():
    chw = _chw(40, 48)
    x = Array(chw, channel_axis=0)
    x = x[:, 16:-16, 16:-16] * 2.0
    x = mi.moveaxis(x, 0, -1)
    x = mi.rgb_matrix_3x3(x, matrix=_MATRIX)
    assert _op_names(x).count("transpose") == 1
    want = np.moveaxis(chw[:, 16:-16, 16:-16] * 2.0, 0, -1) @ _MATRIX.T
    np.testing.assert_allclose(x.realize(), want, rtol=1e-6)


def test_rot90_of_chw_not_supported_yet():
    x = Array(_chw(), channel_axis=0)
    with pytest.raises(NotImplementedError, match=r"mi\.moveaxis\(x, 0, -1\)"):
        rot90(x)


@pytest.mark.parametrize(
    "call",
    [lambda a: a[1:3], lambda a: a.pad(1), lambda a: mi.tile(a, (2, 1, 1))],
    ids=["index", "pad", "tile"],
)
def test_spatial_ops_on_channels_in_the_middle_not_supported_yet(call):
    x = Array(np.zeros((6, 3, 9), np.float32), channel_axis=1)
    with pytest.raises(NotImplementedError, match=r"mi\.moveaxis\(x, 1, -1\)"):
        call(x)
