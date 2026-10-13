"""meshgrid builds coordinate grids from newaxis views and broadcast_to."""

from __future__ import annotations

import numpy as np
import pytest

import muimage as mi


def _coordinate_vectors(lib, kind: str):
    """A 7-sample x and a 4-sample y, so a swapped axis changes the shape."""
    if kind == "linspace":
        return (
            lib.linspace(-1.0, 1.0, 7, dtype=np.float32),
            lib.linspace(0.0, 3.0, 4, dtype=np.float32),
        )
    return lib.arange(7, dtype=np.float32), lib.arange(10, 14, dtype=np.float32)


def _grids(lib, kind, indexing, sparse):
    """``lib.meshgrid`` of the ``kind`` coordinate vectors."""
    x, y = _coordinate_vectors(lib, kind)
    return lib.meshgrid(x, y, sparse=sparse, indexing=indexing)


def _graph_vectors(kind: str):
    """Vectors that exist only as a muimage graph or an ingested buffer."""
    if kind == "lazy":
        return mi.arange(7) * 0.5 - 1.0, mi.linspace(0.0, 1.0, 4) * 3.0
    return (
        mi.Array(np.linspace(-1.0, 1.0, 7, dtype=np.float32)),
        mi.Array(np.arange(4, dtype=np.float32)),
    )


@pytest.mark.parametrize("kind", ["linspace", "arange", "lazy", "ingested"])
@pytest.mark.parametrize("indexing", ["xy", "ij"])
@pytest.mark.parametrize("sparse", [False, True])
def test_meshgrid_matches_numpy(kind, indexing, sparse):
    if kind in ("linspace", "arange"):
        got = _grids(mi, kind, indexing, sparse)
        expect = _grids(np, kind, indexing, sparse)
    else:
        x, y = _graph_vectors(kind)
        got = mi.meshgrid(x, y, sparse=sparse, indexing=indexing)
        expect = np.meshgrid(np.asarray(x), np.asarray(y), sparse=sparse, indexing=indexing)
    assert isinstance(got, tuple) and len(got) == 2
    for grid, want in zip(got, expect):
        assert isinstance(grid, mi.Array)
        assert grid.shape == want.shape
        assert grid.dtype == np.float32
        pixels = np.asarray(grid)
        if kind == "linspace":
            # linspace is computed in float32, so a sample can differ from
            # NumPy's float64 result cast to float32 by a few rounding steps.
            scale = max(1.0, float(np.max(np.abs(want))))
            atol = 4 * float(np.finfo(np.float32).eps) * scale
            np.testing.assert_allclose(pixels, want, rtol=0, atol=atol)
        else:
            np.testing.assert_array_equal(pixels, want)


def test_meshgrid_rejects_a_rank_2_input():
    image = mi.zeros((4, 7))
    with pytest.raises(ValueError, match="must be 1D"):
        mi.meshgrid(mi.arange(7), image)


@pytest.mark.parametrize("count", [1, 3])
def test_meshgrid_takes_exactly_two_inputs(count):
    with pytest.raises(ValueError, match="takes two 1D arrays"):
        mi.meshgrid(*[mi.arange(5)] * count)


def test_meshgrid_rejects_unknown_indexing():
    with pytest.raises(ValueError, match="indexing"):
        mi.meshgrid(mi.arange(3), mi.arange(4), indexing="yx")
