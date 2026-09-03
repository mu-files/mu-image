"""zeros / ones / full are lazy fill nodes."""

from __future__ import annotations

import numpy as np
import pytest

import mucompute as mc
from muimg.tensor import ElementType, Tensor


def test_zeros_is_lazy_then_realizes():
    z = mc.zeros((3, 4))
    assert z._data is None
    assert z._node is not None
    assert z._node.op == "fill"
    assert z._node.inputs == ()
    assert z.dtype == ElementType.FLOAT32
    assert z.shape == (3, 4)
    np.testing.assert_array_equal(z.realize(), np.zeros((3, 4), dtype=np.float32))


def test_ones_rgb():
    t = mc.ones((2, 3, 3), dtype="float32")
    assert t.shape == (2, 3, 3)
    np.testing.assert_array_equal(t.realize(), np.ones((2, 3, 3), dtype=np.float32))


def test_full_scalar_uint8():
    t = mc.full((2, 2), 128, dtype="uint8")
    assert t.dtype == ElementType.UINT8
    np.testing.assert_array_equal(t.realize(), np.full((2, 2), 128, dtype=np.uint8))


def test_full_default_dtype_python_int_is_float32():
    t = mc.full((2, 2), 128)
    assert t.dtype == ElementType.FLOAT32
    np.testing.assert_array_equal(t.realize(), np.full((2, 2), 128, dtype=np.float32))


def test_full_inherits_numpy_scalar_dtype():
    t = mc.full((2, 2), np.uint8(7))
    assert t.dtype == ElementType.UINT8
    np.testing.assert_array_equal(t.realize(), np.full((2, 2), 7, dtype=np.uint8))


def test_full_rgb_vector():
    t = mc.full((2, 2, 3), [1.0, 2.0, 3.0])
    want = np.full((2, 2, 3), [1.0, 2.0, 3.0], dtype=np.float32)
    np.testing.assert_array_equal(t.realize(), want)


def test_full_rejects_non_broadcast_vector():
    with pytest.raises(ValueError, match="channel count"):
        mc.full((2, 2), [1.0, 2.0, 3.0])


def test_zeros_like_does_not_realize_input():
    src = Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
    lazy = src - 1.0
    assert lazy._data is None
    z = mc.zeros_like(lazy)
    assert lazy._data is None
    assert z.shape == lazy.shape
    assert z.dtype == lazy.dtype
    np.testing.assert_array_equal(z.realize(), np.zeros((2, 2), dtype=np.float32))
    assert lazy._data is None


def test_ones_like_dtype_override():
    src = Tensor(np.zeros((2, 2), dtype=np.float32))
    t = mc.ones_like(src, dtype="uint8")
    assert t.dtype == ElementType.UINT8
    np.testing.assert_array_equal(t.realize(), np.ones((2, 2), dtype=np.uint8))


def test_full_like_uses_reference_dtype():
    src = Tensor(np.zeros((2, 3), dtype=np.uint8))
    t = mc.full_like(src, 9)
    assert t.dtype == ElementType.UINT8
    np.testing.assert_array_equal(t.realize(), np.full((2, 3), 9, dtype=np.uint8))


def test_zeros_minus_one_runs():
    t = mc.zeros((2, 2)) - 1.0
    np.testing.assert_array_equal(t.realize(), np.full((2, 2), -1.0, dtype=np.float32))


def test_zeros_rejects_bad_rank():
    with pytest.raises(ValueError, match=r"\(H,W\)"):
        mc.zeros((2,))


def test_zeros_rejects_bad_channels():
    with pytest.raises(ValueError, match="channel count"):
        mc.zeros((2, 2, 2))


def test_zeros_rejects_zero_size():
    with pytest.raises(ValueError, match="at least 1"):
        mc.zeros((0, 4))
    with pytest.raises(ValueError, match="at least 1"):
        mc.zeros((3, 0))


def test_tensor_rejects_zero_size_array():
    with pytest.raises(ValueError, match="at least 1"):
        Tensor(np.zeros((0, 4), dtype=np.float32))
    with pytest.raises(ValueError, match="at least 1"):
        Tensor(np.zeros((3, 0, 3), dtype=np.float32))
