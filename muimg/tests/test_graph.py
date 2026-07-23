"""Tensor / engines.graph tests + eager flush at python barriers."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pytest

import muimg.engines.ops as engine_ops
from muimg.engines import get_default_engine, set_default_engine
from muimg.engines.core import CoreEngine
from muimg.engines.graph import EngineOp, flush
from muimg.engines.ops import OPS_BY_NAME
from muimg.raw_render import DemosaicAlgorithm, demosaic
from muimg.tensor import Tensor


def test_catalog_engine_ops_io():
    """engines.ops carries EngineOp callables + OPS_BY_NAME."""
    assert "sub_scalar" in OPS_BY_NAME
    assert isinstance(engine_ops.bilinear_demosaic, EngineOp)
    assert engine_ops.bilinear_demosaic._in_channels == 1
    x = Tensor(np.zeros((2, 2), dtype=np.float32))
    assert engine_ops.bilinear_demosaic.infer_out_meta(x, {}).channels == 3
    assert callable(engine_ops.matrix)
    assert callable(engine_ops.lut)
    assert callable(flush)


def test_sub_mul_chain():
    inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    x = Tensor(inp)
    x = x - 1.0
    x = x * 2.0
    out = x.compute()
    np.testing.assert_allclose(out, [[0.0, 2.0], [4.0, 6.0]])


def test_matrix_identity():
    eye = np.eye(3, dtype=np.float32)
    inp = np.array([[[0.25, 0.5, 0.75]]], dtype=np.float32)
    out = engine_ops.matrix(Tensor(inp), matrix=eye).compute()
    np.testing.assert_allclose(out, inp)


def test_lut_identity_rgb():
    inp = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    out = engine_ops.lut(Tensor(inp), lut=[0.0, 1.0]).compute()
    np.testing.assert_allclose(out, inp)


def test_bilinear_demosaic_rggb():
    cfa = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    out = engine_ops.bilinear_demosaic(Tensor(cfa), cfa_pattern="RGGB").compute()
    assert out.shape == (2, 2, 3)
    np.testing.assert_allclose(out[0, 0, 0], 0.2)


def test_op_rejects_bad_channels():
    rgb = Tensor(np.zeros((2, 2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="expected 1 channel"):
        engine_ops.bilinear_demosaic(rgb, cfa_pattern="RGGB")


def test_op_rejects_unknown_attr():
    x = Tensor(np.zeros((2, 2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="unknown attrs"):
        engine_ops.matrix(x, matrix=np.eye(3, dtype=np.float32), extra=1)


def test_rejects_tensor_tensor_sub():
    a = Tensor(np.zeros((2, 2), dtype=np.float32))
    b = Tensor(np.ones((2, 2), dtype=np.float32))
    with pytest.raises(TypeError, match="tensor–tensor"):
        _ = a - b


def test_demosaic_tensor_auto_flush():
    """demosaic(Tensor) computes the graph, runs eagerly, returns Tensor."""
    rng = np.random.default_rng(0)
    cfa = rng.integers(0, 1000, size=(16, 16), dtype=np.uint16)
    out_t = demosaic(Tensor(cfa), "RGGB", algorithm=DemosaicAlgorithm.OPENCV_EA)
    assert out_t._node is None
    out = out_t.compute()
    ref = demosaic(cfa, "RGGB", algorithm=DemosaicAlgorithm.OPENCV_EA)
    assert out.shape == (16, 16, 3)
    np.testing.assert_array_equal(out, ref)


def test_flush_then_engine_again():
    """Normalize (engine) → demosaic(Tensor) flush → matrix+lut (new engine chain)."""
    rng = np.random.default_rng(1)
    cfa = (
        rng.integers(100, 1000, size=(16, 16), dtype=np.uint16).astype(np.float32)
        / 1000.0
    )

    eye = np.eye(3, dtype=np.float32)
    lut = np.array([0.0, 1.0], dtype=np.float32)

    x = Tensor(cfa)
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


def test_apply_opcodes_single_execute():
    """Multi-opcode RGB chain runs one execute_graph."""
    from muimg.engines.core import _compute_engine
    from muimg.raw_render import apply_opcodes

    rgb = np.full((8, 8, 3), 0.5, dtype=np.float32)
    opcodes = [
        {
            "type": "FixVignetteRadial",
            "id": 3,
            "coefficients": np.zeros(5, dtype=np.float64),
            "center_x": 0.5,
            "center_y": 0.5,
            "planes": 1,
        },
        {
            "type": "MapPolynomial",
            "id": 8,
            "coefficients": np.array([0.0, 1.0], dtype=np.float32),
            "area": {"top": 0, "left": 0, "bottom": 0, "right": 0},
            "plane": 0,
            "planes": 3,
            "row_pitch": 1,
            "col_pitch": 1,
            "degree": 1,
        },
    ]

    calls = {"n": 0}
    real = _compute_engine.execute_graph

    def counting_execute(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    _compute_engine.execute_graph = counting_execute
    try:
        out = apply_opcodes(rgb, opcodes, use_bicubic=False)
    finally:
        _compute_engine.execute_graph = real

    assert calls["n"] == 1
    assert out.shape == rgb.shape
    np.testing.assert_allclose(out, rgb, rtol=1e-5, atol=1e-5)


class _RecordingEngine:
    """Minimal Engine stub that records execute_segment calls."""

    def __init__(self) -> None:
        self.calls: List[int] = []
        self.supported_ops = frozenset({"sub_scalar", "mul_scalar"})

    def execute_segment(
        self,
        nodes: List[Tensor],
        values: Dict[int, np.ndarray],
        outputs: List[Tensor],
    ) -> None:
        self.calls.append(len(nodes))
        # Produce zeros for outputs (enough to exercise the dispatch path).
        for t in outputs:
            values[id(t)] = np.zeros(t.meta.shape, dtype=np.float32)


def test_set_default_engine_stub():
    """set_default_engine swaps the backend used by Tensor.compute()."""
    prev = get_default_engine()
    stub = _RecordingEngine()
    set_default_engine(stub)
    try:
        assert get_default_engine() is stub
        x = Tensor(np.ones((2, 2), dtype=np.float32)) - 0.0
        out = x.compute()
        assert stub.calls == [1]
        assert out.shape == (2, 2)
    finally:
        set_default_engine(prev)
        assert isinstance(get_default_engine(), CoreEngine)


def test_core_binaries_path():
    """CoreEngine package ships _binaries next to the Python package."""
    import muimg.engines.core as core_pkg
    from pathlib import Path

    binaries = Path(core_pkg.__file__).resolve().parent / "_binaries"
    assert binaries.is_dir()
    libs = list(binaries.glob("libmuimg_core.*")) + list(
        binaries.glob("muimg_core.*")
    )
    assert libs, f"no core libs under {binaries}"
