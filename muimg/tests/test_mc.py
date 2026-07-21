"""Phase A.5–A.6 + A′: muimg.mc engine graph + eager flush at python barriers."""

import numpy as np
import pytest

import muimg.engine_ops as engine_ops
import muimg.mc as mc
from muimg.engine_ops import OPS_CATALOG
from muimg.raw_render import DemosaicAlgorithm, demosaic


def test_catalog_engine_ops_io():
    """Phase A′: engine_ops carries OPS_CATALOG + typed wrappers."""
    assert "sub_scalar" in OPS_CATALOG["ops"]
    assert OPS_CATALOG["ops"]["bilinear_demosaic"]["inputs"][0]["channels"] == 1
    assert OPS_CATALOG["ops"]["bilinear_demosaic"]["outputs"][0]["channels"] == 3
    assert callable(engine_ops.matrix)
    assert callable(engine_ops.lut)
    assert hasattr(mc, "flush")
    assert not hasattr(mc, "apply")


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
    out = engine_ops.matrix(mc.Tensor(inp), matrix=eye).compute()
    np.testing.assert_allclose(out, inp)


def test_mc_lut_identity_rgb():
    inp = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    out = engine_ops.lut(mc.Tensor(inp), lut=[0.0, 1.0]).compute()
    np.testing.assert_allclose(out, inp)


def test_mc_bilinear_demosaic_rggb():
    cfa = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    out = engine_ops.bilinear_demosaic(mc.Tensor(cfa), cfa_pattern="RGGB").compute()
    assert out.shape == (2, 2, 3)
    np.testing.assert_allclose(out[0, 0, 0], 0.2)


def test_mc_op_rejects_bad_channels():
    rgb = mc.Tensor(np.zeros((2, 2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="expected 1 channel"):
        engine_ops.bilinear_demosaic(rgb, cfa_pattern="RGGB")


def test_mc_op_rejects_unknown_attr():
    x = mc.Tensor(np.zeros((2, 2, 3), dtype=np.float32))
    with pytest.raises(TypeError):
        engine_ops.matrix(x, matrix=np.eye(3, dtype=np.float32), extra=1)  # type: ignore[call-arg]


def test_mc_rejects_tensor_tensor_sub():
    a = mc.Tensor(np.zeros((2, 2), dtype=np.float32))
    b = mc.Tensor(np.ones((2, 2), dtype=np.float32))
    with pytest.raises(TypeError, match="tensor–tensor"):
        _ = a - b


def test_mc_demosaic_tensor_auto_flush():
    """demosaic(Tensor) computes the graph, runs eagerly, returns Tensor."""
    rng = np.random.default_rng(0)
    cfa = rng.integers(0, 1000, size=(16, 16), dtype=np.uint16)
    out_t = demosaic(mc.Tensor(cfa), "RGGB", algorithm=DemosaicAlgorithm.OPENCV_EA)
    assert out_t._node is None
    out = out_t.compute()
    ref = demosaic(cfa, "RGGB", algorithm=DemosaicAlgorithm.OPENCV_EA)
    assert out.shape == (16, 16, 3)
    np.testing.assert_array_equal(out, ref)


def test_mc_flush_then_engine_again():
    """Normalize (engine) → demosaic(Tensor) flush → matrix+lut (new engine chain)."""
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
    x = demosaic(x, "RGGB", algorithm=DemosaicAlgorithm.OPENCV_EA)
    x = engine_ops.matrix(x, matrix=eye)
    x = engine_ops.lut(x, lut=lut)
    out = x.compute()

    ref = demosaic(
        cfa, "RGGB", algorithm=DemosaicAlgorithm.OPENCV_EA, return_dtype=np.float32
    )
    assert out.shape == (16, 16, 3)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)
