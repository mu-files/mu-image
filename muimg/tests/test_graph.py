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


def test_demosaic_tensor_lazy():
    """demosaic(Tensor) returns a lazy Tensor; compute materializes RGB."""
    rng = np.random.default_rng(0)
    cfa = rng.integers(0, 1000, size=(16, 16), dtype=np.uint16)
    out_t = demosaic(Tensor(cfa), "RGGB", algorithm=DemosaicAlgorithm.OPENCV_EA)
    assert out_t._node is not None
    out = out_t.compute()
    ref = demosaic(Tensor(cfa), "RGGB", algorithm=DemosaicAlgorithm.OPENCV_EA).compute()
    assert out.shape == (16, 16, 3)
    np.testing.assert_array_equal(out, ref)


def test_flush_then_engine_again():
    """Normalize (engine) → demosaic(Tensor) → matrix+lut (same DAG)."""
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
        Tensor(cfa), "RGGB", algorithm=DemosaicAlgorithm.OPENCV_EA, dst_dtype="float32"
    ).compute()
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
        out_t = apply_opcodes(Tensor(rgb), opcodes, use_bicubic=False)
        out = out_t.compute()
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


def test_graph_op_crop_cast():
    from muimg.engines.pyops import cast_dtype_op, crop_op

    src = np.arange(16, dtype=np.uint8).reshape(4, 4)
    np.testing.assert_array_equal(crop_op(src, 1, 1, 3, 2), src[1:3, 1:4])

    x = cast_dtype_op(Tensor(src), "uint16")
    x = crop_op(x, 1, 1, 3, 2)
    assert x._node is not None and x._node.fn is not None
    out = x.compute()
    np.testing.assert_array_equal(out, src.astype(np.uint16)[1:3, 1:4])


def test_add_completed_step_duration():
    from muimg.common import PerfTimer

    root = PerfTimer("root")
    child = root.add_completed_step("native_op (engine)", 0.025)
    assert child.end_time is not None
    assert abs(child.get_elapsed_ms() - 25.0) < 1.0
    assert root.children == [child]
    root.close()


def test_add_completed_steps_sequential_no_overlap():
    from muimg.common import PerfTimer

    root = PerfTimer("root")
    children = root.add_completed_steps(
        [
            ("op_a (engine)", 0.010),
            ("op_b (engine)", 0.020),
            ("op_c (engine)", 0.005),
        ]
    )
    assert [c.name for c in children] == [
        "op_a (engine)",
        "op_b (engine)",
        "op_c (engine)",
    ]
    assert abs(children[0].get_elapsed_ms() - 10.0) < 1.0
    assert abs(children[1].get_elapsed_ms() - 20.0) < 1.0
    assert abs(children[2].get_elapsed_ms() - 5.0) < 1.0
    # End-to-end layout: each child starts when the previous ends.
    assert children[0].end_time == children[1].start_time
    assert children[1].end_time == children[2].start_time
    root.close()


def test_engine_timing_setting():
    from muimg.engines.timing import (
        EngineTiming,
        engine_timing,
        get_engine_timing,
        set_engine_timing,
    )

    prev = engine_timing
    try:
        set_engine_timing(EngineTiming.OFF)
        assert get_engine_timing() is EngineTiming.OFF
        set_engine_timing("SEGMENTS")
        assert get_engine_timing() is EngineTiming.SEGMENTS
        set_engine_timing(EngineTiming.OPS)
        assert get_engine_timing() is EngineTiming.OPS
    finally:
        set_engine_timing(prev)


def test_perftimer_context_manager_nests():
    from muimg.common import PerfTimer

    with PerfTimer("root") as root:
        assert PerfTimer.current() is root
        with PerfTimer("child") as child:
            assert child.parent is root
            assert child in root.children
            assert PerfTimer.current() is child
            with PerfTimer("grand") as grand:
                assert grand.parent is child
                assert PerfTimer.current() is grand
            assert PerfTimer.current() is child
        assert PerfTimer.current() is root
    assert PerfTimer.current() is None
    assert [c.name for c in root.children] == ["child"]
    assert [c.name for c in root.children[0].children] == ["grand"]


def test_perftimer_step_is_fire_and_forget():
    from muimg.common import PerfTimer

    root = PerfTimer("root")
    a = PerfTimer.step("a")
    assert a is not None
    a.close()
    b = PerfTimer.step("b")
    assert b is not None
    b.close()
    assert PerfTimer.current() is root
    root.close()
    assert [c.name for c in root.children] == ["a", "b"]
    assert all(c.end_time is not None for c in root.children)


def test_perftimer_step_nests_under_current_not_root():
    from muimg.common import PerfTimer

    root = PerfTimer("root")
    bucket = root.start_step("bucket")
    setup = PerfTimer.step("render_setup")
    assert setup is not None
    setup.close()
    assert bucket.end_time is None
    assert [c.name for c in bucket.children] == ["render_setup"]
    bucket.close()
    root.close()
    assert [c.name for c in root.children] == ["bucket"]


def test_perftimer_broken_stack_report():
    from muimg.common import PerfTimer

    root = PerfTimer("root")
    child = root.start_step("a")
    # Force an out-of-order pop of the root while child is still deeper on the stack.
    PerfTimer._pop(root)
    root._on_stack = False
    root._broken = True
    root.end_time = root.start_time
    child.close()
    assert root.get_report() == "broken stack"
    PerfTimer._stack().clear()


def test_compute_times_python_ops():
    from muimg.common import PerfTimer
    from muimg.engines.pyops import cast_dtype_op, crop_op
    from muimg.engines.timing import EngineTiming, engine_timing, set_engine_timing

    src = np.arange(16, dtype=np.uint8).reshape(4, 4)
    x = cast_dtype_op(Tensor(src), "uint16")
    x = crop_op(x, 1, 1, 3, 2)

    prev = engine_timing
    try:
        set_engine_timing(EngineTiming.SEGMENTS)
        with PerfTimer("root") as root:
            parent = root.start_step("camera_space")
            out = x.compute()
            parent.close()
    finally:
        set_engine_timing(prev)

    np.testing.assert_array_equal(out, src.astype(np.uint16)[1:3, 1:4])
    names = [c.name for c in parent.children]
    assert names == ["cast_dtype_op (python)", "crop_op (python)"]
    assert all(c.get_elapsed_ms() >= 0.0 for c in parent.children)


def test_compute_times_engine_ops():
    from muimg.common import PerfTimer
    from muimg.engines.timing import EngineTiming, engine_timing, set_engine_timing

    prev = engine_timing
    try:
        set_engine_timing(EngineTiming.OPS)
        with PerfTimer("root") as root:
            out = (Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)) - 1.0)
            out = (out * 2.0).compute()
    finally:
        set_engine_timing(prev)

    np.testing.assert_allclose(out, [[0.0, 2.0], [4.0, 6.0]])
    assert [c.name for c in root.children] == ["graph_compute"]
    ops = root.children[0].children
    names = [c.name for c in ops]
    assert names == ["sub_scalar (engine)", "mul_scalar (engine)"]
    assert all(c.get_elapsed_ms() >= 0.0 for c in ops)
    assert ops[0].end_time == ops[1].start_time


def test_compute_engine_segments_no_op_children():
    from muimg.common import PerfTimer
    from muimg.engines.timing import EngineTiming, engine_timing, set_engine_timing

    prev = engine_timing
    try:
        set_engine_timing(EngineTiming.SEGMENTS)
        with PerfTimer("root") as root:
            out = (Tensor(np.array([[1.0, 2.0]], dtype=np.float32)) * 2.0).compute()
    finally:
        set_engine_timing(prev)

    np.testing.assert_allclose(out, [[2.0, 4.0]])
    assert [c.name for c in root.children] == ["graph_compute"]
    assert root.children[0].children == []


def test_compute_engine_off_no_rows_even_with_open_timer():
    from muimg.common import PerfTimer
    from muimg.engines.timing import EngineTiming, engine_timing, set_engine_timing

    prev = engine_timing
    try:
        set_engine_timing(EngineTiming.OFF)
        with PerfTimer("root") as root:
            (Tensor(np.array([[1.0]], dtype=np.float32)) * 2.0).compute()
    finally:
        set_engine_timing(prev)

    assert root.children == []


def test_compute_nests_under_current_stack_top():
    from muimg.common import PerfTimer
    from muimg.engines.timing import EngineTiming, engine_timing, set_engine_timing

    prev = engine_timing
    try:
        set_engine_timing(EngineTiming.SEGMENTS)
        with PerfTimer("root") as root:
            fence = root.start_step("fence")
            (Tensor(np.array([[1.0]], dtype=np.float32)) * 2.0).compute()
            fence.close()
    finally:
        set_engine_timing(prev)

    assert [c.name for c in fence.children] == ["graph_compute"]


def test_compute_ops_under_graph_compute():
    from muimg.common import PerfTimer
    from muimg.engines.timing import EngineTiming, engine_timing, set_engine_timing

    prev = engine_timing
    try:
        set_engine_timing(EngineTiming.OPS)
        with PerfTimer("root") as root:
            parent = root.start_step("camera_space")
            out = (Tensor(np.array([[1.0, 2.0]], dtype=np.float32)) * 2.0).compute()
            parent.close()
    finally:
        set_engine_timing(prev)

    np.testing.assert_allclose(out, [[2.0, 4.0]])
    assert [c.name for c in parent.children] == ["graph_compute"]
    assert [c.name for c in parent.children[0].children] == ["mul_scalar (engine)"]


def test_graph_op_splits_engine_segments():
    from muimg.engines.core import _compute_engine
    from muimg.engines.pyops import crop_op

    src = np.arange(16, dtype=np.float32).reshape(4, 4)
    x = Tensor(src) - 0.0
    x = crop_op(x, 0, 0, 2, 2)
    x = x * 2.0

    calls = {"n": 0}
    real = _compute_engine.execute_graph

    def counting_execute(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    _compute_engine.execute_graph = counting_execute
    try:
        out = x.compute()
    finally:
        _compute_engine.execute_graph = real

    assert calls["n"] == 2
    np.testing.assert_allclose(out, src[:2, :2] * 2.0)


def test_demosaic_op_lazy():
    from muimg.engines.pyops import demosaic_op

    rng = np.random.default_rng(4)
    cfa = rng.integers(0, 1000, size=(16, 16), dtype=np.uint16)
    out = demosaic_op(Tensor(cfa), "RGGB", "OPENCV_EA").compute()
    ref = demosaic(Tensor(cfa), "RGGB", algorithm=DemosaicAlgorithm.OPENCV_EA).compute()
    np.testing.assert_array_equal(out, ref)
