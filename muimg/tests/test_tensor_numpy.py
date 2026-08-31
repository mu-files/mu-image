"""Tensor rot90 / fliplr / flipud / transpose match the same NumPy calls."""

from __future__ import annotations

import numpy as np
import pytest

from muimg.tensor import Tensor, fliplr, flipud, rot90


def _mono() -> np.ndarray:
    return np.arange(5 * 7, dtype=np.float32).reshape(5, 7) + 1.0


def _rgb() -> np.ndarray:
    h, w = 5, 7
    plane = np.arange(h * w, dtype=np.float32).reshape(h, w)
    return np.stack([plane, plane + 100.0, plane + 200.0], axis=-1)


@pytest.mark.parametrize("src", [_mono(), _rgb()], ids=["mono", "rgb"])
@pytest.mark.parametrize("k", [0, 1, 2, 3, 4, -1, 5])
def test_rot90_matches_numpy(src, k):
    np.testing.assert_array_equal(rot90(Tensor(src), k), np.rot90(src, k))


def test_rot90_rejects_non_spatial_axes():
    t = Tensor(_mono())
    with pytest.raises(ValueError, match="spatial plane"):
        rot90(t, 1, axes=(1, 0))


@pytest.mark.parametrize("src", [_mono(), _rgb()], ids=["mono", "rgb"])
def test_fliplr_matches_numpy(src):
    np.testing.assert_array_equal(fliplr(Tensor(src)), np.fliplr(src))


@pytest.mark.parametrize("src", [_mono(), _rgb()], ids=["mono", "rgb"])
def test_flipud_matches_numpy(src):
    np.testing.assert_array_equal(flipud(Tensor(src)), np.flipud(src))


@pytest.mark.parametrize("src", [_mono(), _rgb()], ids=["mono", "rgb"])
def test_rot90_then_fliplr_matches_numpy(src):
    np.testing.assert_array_equal(fliplr(rot90(Tensor(src), 1)), np.fliplr(np.rot90(src, 1)))


def test_transpose_and_T_match_numpy_2d():
    src = _mono()
    t = Tensor(src)
    want = src.T
    np.testing.assert_array_equal(t.transpose(), want)
    np.testing.assert_array_equal(t.T, want)
    np.testing.assert_array_equal(t.transpose(1, 0), want)
    np.testing.assert_array_equal(t.transpose((1, 0)), want)


def test_transpose_spatial_matches_numpy_rgb():
    src = _rgb()
    want = src.transpose(1, 0, 2)
    t = Tensor(src)
    np.testing.assert_array_equal(t.transpose(), want)
    np.testing.assert_array_equal(t.T, want)
    np.testing.assert_array_equal(t.transpose(1, 0, 2), want)


def test_rot90_rejects_channel_axis():
    t = Tensor(_rgb())
    with pytest.raises(ValueError, match="spatial plane"):
        rot90(t, 1, axes=(0, 2))


def test_transpose_rejects_other_axes():
    t = Tensor(_rgb())
    with pytest.raises(ValueError, match="2D spatial"):
        t.transpose(2, 1, 0)
