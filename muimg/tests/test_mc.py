"""Phase A.5–A.6: muimg.mc graph builder, compute, auto-split."""

import numpy as np
import pytest

import muimg.mc as mc
from muimg.raw_render import DemosaicAlgorithm, demosaic as rr_demosaic


def test_mc_sub_mul_chain():
    inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    x = mc.Tensor(inp)
    x = x - 1.0
    x = x * 2.0
    out = x.compute()
    np.testing.assert_allclose(out, [[0.0, 2.0], [4.0, 6.0]])


def test_mc_matrix_identity():
    eye = np.eye(3, dtype=np.float32)
    inp = np.array([[[0.25, 0.5, 0.75]]], dtype=np.float32)
    out = mc.matrix(mc.Tensor(inp), eye).compute()
    np.testing.assert_allclose(out, inp)


def test_mc_lut_identity_rgb():
    inp = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    out = mc.lut(mc.Tensor(inp), [0.0, 1.0]).compute()
    np.testing.assert_allclose(out, inp)


def test_mc_bilinear_demosaic_rggb():
    cfa = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    out = mc.demosaic(mc.Tensor(cfa), "bilinear", "RGGB").compute()
    assert out.shape == (2, 2, 3)
    np.testing.assert_allclose(out[0, 0, 0], 0.2)


def test_mc_rejects_tensor_tensor_sub():
    a = mc.Tensor(np.zeros((2, 2), dtype=np.float32))
    b = mc.Tensor(np.ones((2, 2), dtype=np.float32))
    with pytest.raises(TypeError, match="tensor–tensor"):
        _ = a - b


def test_mc_python_demosaic_opencv_ea():
    """Python-affinity demosaic alone (single python segment)."""
    # Larger than 2x2 so OpenCV EA has room; uint16 avoids float→u16 roundtrip noise.
    rng = np.random.default_rng(0)
    cfa = rng.integers(0, 1000, size=(16, 16), dtype=np.uint16)
    out = mc.demosaic(mc.Tensor(cfa), "OPENCV_EA", "RGGB").compute()
    ref = rr_demosaic(cfa, "RGGB", algorithm=DemosaicAlgorithm.OPENCV_EA)
    assert out.shape == (16, 16, 3)
    np.testing.assert_array_equal(out, ref)


def test_mc_auto_split_engine_python_engine():
    """Normalize (engine) → OpenCV demosaic (python) → matrix+lut (engine)."""
    rng = np.random.default_rng(1)
    cfa = (
        rng.integers(100, 1000, size=(16, 16), dtype=np.uint16).astype(np.float32)
        / 1000.0
    )

    eye = np.eye(3, dtype=np.float32)
    lut = np.array([0.0, 1.0], dtype=np.float32)

    x = mc.Tensor(cfa)
    x = x - 0.0
    x = x * 1.0
    x = mc.demosaic(x, "OPENCV_EA", "RGGB")
    x = mc.matrix(x, eye)
    x = mc.lut(x, lut)
    out = x.compute()

    ref = rr_demosaic(
        cfa, "RGGB", algorithm=DemosaicAlgorithm.OPENCV_EA, return_dtype=np.float32
    )
    assert out.shape == (16, 16, 3)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)
