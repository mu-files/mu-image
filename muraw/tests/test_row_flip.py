"""A row flip (negative row step) after a computed op, against NumPy's flip of the same result."""

from __future__ import annotations

import numpy as np
import pytest

import muimage as mi
from muraw.array import Array

HEIGHT, WIDTH = 300, 70

UPSTREAM = {
    "source": lambda a: a,
    "mul": lambda a: a * 2,
    "lut": lambda a: mi.lut(
        a.convert_type(np.float32), lut=np.linspace(0.0, 1.0, 16, dtype=np.float32) ** 2
    ),
    "astype": lambda a: a.astype(np.float32),
    "chain": lambda a: (a.astype(np.float32) * 2 - 1) * 0.5,
    "col_step_view": lambda a: (a * 2)[:, ::2],
    "demosaic": lambda a: mi.cfa_bilinear_demosaic(a.convert_type(np.float32), cfa_pattern="RGGB"),
    "pad": lambda a: (a * 2).pad(3, mode="reflect"),
    "tile": lambda a: mi.tile(a * 2, (2, 1)),
    "orientation": lambda a: (a * 2).T,
}

ROW_FLIPS = {
    "flip": np.s_[::-1],
    "step_minus_2": np.s_[::-2],
    "cropped_flip": np.s_[250:3:-1],
}


def _source(dtype) -> np.ndarray:
    rng = np.random.default_rng(20260928)
    if dtype == np.uint8:
        return rng.integers(0, 100, (HEIGHT, WIDTH), dtype=np.uint8)
    return rng.random((HEIGHT, WIDTH), dtype=np.float32)


@pytest.mark.parametrize("dtype", [np.uint8, np.float32], ids=["uint8", "float32"])
@pytest.mark.parametrize("flip", list(ROW_FLIPS), ids=list(ROW_FLIPS))
@pytest.mark.parametrize("upstream", list(UPSTREAM), ids=list(UPSTREAM))
def test_row_flip_after_op_matches_numpy(upstream, flip, dtype):
    src = _source(dtype)
    build = UPSTREAM[upstream]
    key = ROW_FLIPS[flip]
    expected = build(Array(src)).realize()[key]
    flipped = build(Array(src))[key]
    assert flipped._node.op == "view" and flipped._node.attrs["row_step"] < 0
    np.testing.assert_array_equal(flipped.realize(), expected)


@pytest.mark.parametrize("upstream", list(UPSTREAM), ids=list(UPSTREAM))
def test_ops_after_row_flip_match_numpy(upstream):
    src = _source(np.float32)
    build = UPSTREAM[upstream]
    flipped = build(Array(src)).realize()[::-1]
    expected = (Array(np.ascontiguousarray(flipped)) * 3 - 1).realize()
    got = (build(Array(src))[::-1] * 3 - 1).realize()
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("upstream", list(UPSTREAM), ids=list(UPSTREAM))
def test_two_row_flips_cancel(upstream):
    src = _source(np.float32)
    build = UPSTREAM[upstream]
    expected = build(Array(src)).realize()
    got = build(Array(src))[::-1][::-1].realize()
    np.testing.assert_array_equal(got, expected)
