"""Pad op: emit-meta and execute vs ``numpy.pad``."""

from __future__ import annotations

import numpy as np
import pytest

from muimg.tensor import Tensor


def test_pad_emit_meta_grows_canvas_at_origin():
    src = Tensor(np.zeros((4, 6), dtype=np.float32), origin=(2, 3))
    t = src.pad(1)
    assert t.meta.height == 6 and t.meta.width == 8
    assert t.meta.origin == (1, 2)
    assert t.meta.canvas == (2, 1, 8, 6)
    assert t._node is not None and t._node.op == "pad"


def test_pad_width_shapes():
    src = Tensor(np.zeros((3, 5), dtype=np.float32))
    assert src.pad(2).meta.shape == (7, 9)
    assert src.pad((1, 3)).meta.shape == (7, 9)
    assert src.pad(((1, 0), (2, 4))).meta.shape == (4, 11)


@pytest.mark.parametrize(
    "mode,pad_width",
    [
        ("constant", 1),
        ("edge", 1),
        ("reflect", 1),
        ("symmetric", 1),
        ("constant", ((1, 2), (0, 3))),
    ],
)
def test_pad_matches_numpy(mode, pad_width):
    rng = np.random.default_rng(0)
    src = rng.random((5, 7), dtype=np.float32)
    t = Tensor(src).pad(pad_width, mode=mode)
    np_width = pad_width if isinstance(pad_width, tuple) else pad_width
    expect = np.pad(src, np_width, mode=mode)
    np.testing.assert_array_equal(t, expect)


def test_pad_rgb_does_not_pad_channels():
    rng = np.random.default_rng(1)
    src = rng.random((4, 6, 3), dtype=np.float32)
    out = Tensor(src).pad(1, mode="edge").compute()
    expect = np.pad(src, ((1, 1), (1, 1), (0, 0)), mode="edge")
    np.testing.assert_array_equal(out, expect)


def test_pad_constant_values():
    src = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    out = Tensor(src).pad(1, mode="constant", constant_values=9).compute()
    expect = np.pad(src, 1, mode="constant", constant_values=9)
    np.testing.assert_array_equal(out, expect)
    per_edge = Tensor(src).pad(1, mode="constant", constant_values=((9, 8), (7, 6))).compute()
    expect_edge = np.pad(src, 1, mode="constant", constant_values=((9, 8), (7, 6)))
    np.testing.assert_array_equal(per_edge, expect_edge)


def test_pad_then_interior_view():
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    t = Tensor(src, origin=(5, 7)).pad(((1, 1), (2, 2)))
    interior = t.view(left=2, top=1, width=4, height=3, oob_valid=False)
    assert interior.meta.origin == (5, 7)
    np.testing.assert_array_equal(interior.compute(), src)
