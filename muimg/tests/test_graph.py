"""Tensor / engines.graph tests + eager flush at python barriers."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pytest

import mucompute as mc
from muimg.engines import get_default_engine, set_default_engine
from muimg.engines.core import CoreEngine
from muimg.engines.graph import EngineOp, flush
from muimg.engines.ops import OPS_BY_NAME
from muimg.raw_render import DemosaicAlgorithm, demosaic
from muimg.tensor import Tensor


def test_catalog_engine_ops_io():
    """engines.ops carries EngineOp callables + OPS_BY_NAME."""
    assert "sub_scalar" in OPS_BY_NAME
    assert "view" in OPS_BY_NAME
    assert "pad" in OPS_BY_NAME
    assert "orientation" in OPS_BY_NAME
    assert isinstance(mc.bilinear_demosaic, EngineOp)
    assert mc.bilinear_demosaic._in_channels == 1
    x = Tensor(np.zeros((2, 2), dtype=np.float32))
    assert mc.bilinear_demosaic.infer_out_meta(x, {}).channels == 3
    assert callable(mc.matrix_3x3)
    assert callable(mc.lut)
    assert callable(flush)
    assert "view" in get_default_engine().supported_ops


def test_tensor_meta_origin_default():
    x = Tensor(np.zeros((4, 6), dtype=np.float32))
    assert x.meta.origin == (0, 0)
    assert x.meta.canvas == (0, 0, 6, 4)
    y = x - 1.0
    assert y.meta.origin == (0, 0)
    assert y.meta.canvas == (0, 0, 6, 4)
    assert y.meta.height == 4 and y.meta.width == 6


def test_tensor_origin_kwarg():
    src = Tensor(np.zeros((4, 6), dtype=np.float32), origin=(-3, -5))
    assert src.meta.origin == (-3, -5)
    assert src.meta.canvas == (-5, -3, 6, 4)


def test_view_emit_meta_updates_origin_and_size():
    """Default view accumulates origin; reset_origin re-zeros world."""
    base = Tensor(np.arange(5 * 7, dtype=np.float32).reshape(5, 7))
    cat = base.view(left=1, top=2, width=3, height=2)
    assert cat.meta.height == 2 and cat.meta.width == 3
    assert cat.meta.origin == (2, 1)
    assert cat.meta.canvas == (0, 0, 7, 5)
    assert cat._node is not None and cat._node.op == "view"
    assert cat._node.fn is None

    cat2 = cat.view(left=1, top=0, width=2, height=1)
    assert cat2.meta.origin == (2, 2)

    reset = base.view(left=1, top=2, width=3, height=2, reset_origin=True)
    assert reset.meta.origin == (0, 0)

    out = cat.realize()
    assert out.shape == (2, 3)
    np.testing.assert_array_equal(out, np.asarray(base)[2:4, 1:4])


def test_orientation_emit_meta_swaps_hw():
    """TIFF 5–8 swap H×W; 1–4 keep size; origin unchanged."""
    base = Tensor(np.zeros((4, 6, 3), dtype=np.float32))
    same = mc.orientation(base, orientation=3)
    assert same.meta.height == 4 and same.meta.width == 6
    assert same.meta.origin == (0, 0)
    assert same._node is not None and same._node.op == "orientation"

    rot = mc.orientation(base, orientation=6)
    assert rot.meta.height == 6 and rot.meta.width == 4
    assert rot.meta.origin == (0, 0)

    with pytest.raises(ValueError, match="invalid TIFF code"):
        mc.orientation(base, orientation=0)


def _tiff_orientation_numpy(arr: np.ndarray, code: int) -> np.ndarray:
    """Reference for TIFF 1–8, matching ``Orientation`` enum names."""
    if code == 1:
        return arr
    if code == 2:  # MIRROR_HORIZONTAL
        return arr[:, ::-1]
    if code == 3:  # ROTATE_180
        return np.rot90(arr, 2)
    if code == 4:  # MIRROR_VERTICAL
        return arr[::-1]
    if code == 5:  # MIRROR_HORIZONTAL then ROTATE_270_CW
        return np.rot90(arr[:, ::-1], 1)
    if code == 6:  # ROTATE_90_CW
        return np.rot90(arr, -1)
    if code == 7:  # MIRROR_HORIZONTAL then ROTATE_90_CW
        return np.rot90(arr[:, ::-1], -1)
    if code == 8:  # ROTATE_270_CW
        return np.rot90(arr, 1)
    raise ValueError(code)


def test_engine_orientation_executes_all_tiff_codes(monkeypatch):
    """mc.orientation is issued natively for TIFF 1–8."""
    from muimg.engines.core import _engine_load

    calls: List[dict] = []
    real = _engine_load.execute_graph

    def wrap(graph, in_binds, out_binds, record_ops=False):
        calls.append(graph)
        return real(graph, in_binds, out_binds, record_ops)

    monkeypatch.setattr(_engine_load, "execute_graph", wrap)

    # Remainder-sized vs 256 so dest tiles are not a full-grid multiple.
    src_u8 = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
    src_f32 = np.arange(5 * 7, dtype=np.float32).reshape(5, 7)
    for src in (src_u8, src_f32):
        for code in range(1, 9):
            calls.clear()
            src_t = Tensor(src)
            t = mc.orientation(src_t, orientation=code)
            if code == 1:
                assert t is src_t
                assert t._node is None
                assert calls == []
                np.testing.assert_array_equal(t.realize(), src)
                continue
            assert t._node is not None and t._node.op == "orientation"
            out = t.realize()
            assert len(calls) == 1
            assert [n["op"] for n in calls[0]["nodes"]] == ["orientation"]
            expect = np.ascontiguousarray(_tiff_orientation_numpy(src, code))
            np.testing.assert_array_equal(out, expect)


# TIFF 1–8 inverses: 6↔8, the rest are involutions.
_ORIENTATION_INVERSE = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 8, 7: 7, 8: 6}


def test_engine_orientation_span_sandwich(monkeypatch):
    """sub → orient(code) → mul → orient(inverse) is one native segment.

    Scalar mul commutes with the pixel permute, so the pair is a spatial
    no-op and the result matches sub → mul.
    """
    from muimg.engines.core import _engine_load

    calls: List[dict] = []
    real = _engine_load.execute_graph

    def wrap(graph, in_binds, out_binds, record_ops=False):
        calls.append(graph)
        return real(graph, in_binds, out_binds, record_ops)

    monkeypatch.setattr(_engine_load, "execute_graph", wrap)

    src = np.arange(5 * 7, dtype=np.float32).reshape(5, 7)
    expect = (src - 1.0) * 2.0
    for code in range(1, 9):
        inv = _ORIENTATION_INVERSE[code]
        calls.clear()
        x = Tensor(src) - 1.0
        x = mc.orientation(x, orientation=code)
        x = x * 2.0
        x = mc.orientation(x, orientation=inv)
        out = x.realize()

        assert len(calls) == 1, f"code {code}"
        want_ops = (
            ["sub_scalar", "mul_scalar"]
            if code == 1
            else [
                "sub_scalar",
                "orientation",
                "mul_scalar",
                "orientation",
            ]
        )
        assert [n["op"] for n in calls[0]["nodes"]] == want_ops, f"code {code}"
        np.testing.assert_allclose(out, expect, err_msg=f"code {code}")
        assert out.shape == src.shape, f"code {code}"


def test_crop_emit_rejects_window_outside_canvas():
    x = Tensor(np.zeros((4, 4), dtype=np.float32))
    t = x.view(left=1, top=1, width=2, height=2)
    assert t._node is not None and t._node.op == "view"
    assert t.meta.height == 2 and t.meta.width == 2
    with pytest.raises(ValueError, match="outside canvas"):
        x.view(left=1, top=1, width=4, height=2)
    with pytest.raises(ValueError, match="outside canvas"):
        x.view(left=-2, top=-1, width=3, height=3)
    with pytest.raises(ValueError, match="outside canvas"):
        x.crop(left=4, top=0, width=2, height=2)
    with pytest.raises(ValueError, match="outside canvas"):
        x.crop(left=0, top=-3, width=2, height=3)


def test_span_crop_span_one_execute_graph(monkeypatch):
    """Native crop stays in one CoreEngine segment (C4c2)."""
    from muimg.engines.core import _engine_load

    calls: List[dict] = []
    real = _engine_load.execute_graph

    def wrap(graph, in_binds, out_binds, record_ops=False):
        calls.append(graph)
        return real(graph, in_binds, out_binds, record_ops)

    monkeypatch.setattr(_engine_load, "execute_graph", wrap)

    inp = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32
    )
    x = Tensor(inp) - 1.0
    x = x.view(left=1, top=1, width=2, height=2)
    x = x * 2.0
    out = x.realize()

    assert len(calls) == 1
    ops = [n["op"] for n in calls[0]["nodes"]]
    assert ops == ["sub_scalar", "view", "mul_scalar"]
    np.testing.assert_allclose(out, [[8.0, 10.0], [14.0, 16.0]])


def test_crop_sub_crop_sub_ramp(monkeypatch):
    """input → crop → sub → crop → sub in one segment; known ramp pixels."""
    from muimg.engines.core import _engine_load

    calls: List[dict] = []
    real = _engine_load.execute_graph

    def wrap(graph, in_binds, out_binds, record_ops=False):
        calls.append(graph)
        return real(graph, in_binds, out_binds, record_ops)

    monkeypatch.setattr(_engine_load, "execute_graph", wrap)

    # Unique per-pixel ramp: value = 10*row + col (easy hand checks).
    rows = np.arange(8, dtype=np.float32)[:, None]
    cols = np.arange(8, dtype=np.float32)[None, :]
    inp = 10.0 * rows + cols

    x = Tensor(inp)
    x = x.view(left=1, top=1, width=6, height=6)  # → inp[1:7, 1:7]
    x = x - 1.0
    x = x.view(left=1, top=1, width=4, height=4)  # → inp[2:6, 2:6] after first crop
    x = x - 2.0
    out = x.realize()

    assert len(calls) == 1
    ops = [n["op"] for n in calls[0]["nodes"]]
    assert ops == ["view", "sub_scalar", "view", "sub_scalar"]

    expected = inp[2:6, 2:6] - 3.0
    np.testing.assert_allclose(out, expected)
    # Spot-check corners against the ramp formula.
    assert out[0, 0] == pytest.approx(10.0 * 2 + 2 - 3.0)  # 19
    assert out[0, 3] == pytest.approx(10.0 * 2 + 5 - 3.0)  # 22
    assert out[3, 0] == pytest.approx(10.0 * 5 + 2 - 3.0)  # 49
    assert out[3, 3] == pytest.approx(10.0 * 5 + 5 - 3.0)  # 52


def test_sub_mul_chain():
    inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    x = Tensor(inp)
    x = x - 1.0
    x = x * 2.0
    out = x.realize()
    np.testing.assert_allclose(out, [[0.0, 2.0], [4.0, 6.0]])


def test_matrix_3x3_identity():
    eye = np.eye(3, dtype=np.float32)
    inp = np.array([[[0.25, 0.5, 0.75]]], dtype=np.float32)
    out = mc.matrix_3x3(Tensor(inp), matrix=eye).realize()
    np.testing.assert_allclose(out, inp)


def test_lut_identity_rgb():
    inp = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    out = mc.lut(Tensor(inp), lut=[0.0, 1.0]).realize()
    np.testing.assert_allclose(out, inp)


def test_bilinear_demosaic_rggb():
    cfa = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    out = mc.bilinear_demosaic(Tensor(cfa), cfa_pattern="RGGB").realize()
    assert out.shape == (2, 2, 3)
    np.testing.assert_allclose(out[0, 0, 0], 0.2)


def test_ea_demosaic_rggb():
    cfa = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    out = mc.ea_demosaic(Tensor(cfa), cfa_pattern="RGGB").realize()
    assert out.shape == (2, 2, 3)
    np.testing.assert_allclose(
        out,
        [
            [[0.2, 0.5, 0.8], [0.1, 0.4, 0.7]],
            [[0.3, 0.6, 0.9], [0.2, 0.5, 0.8]],
        ],
        atol=1e-6,
    )


def test_ea_demosaic_then_crop_matches_slice():
    """Fused EA + DefaultCrop (nonzero origin) must match a sliced full frame.

    Tile last-compute used to address the cropped dest with CFA coordinates
    and overwrite past the buffer (R5 create_dng_from_page / render path).
    """
    rng = np.random.default_rng(0)
    cfa = rng.random((17, 19), dtype=np.float32)
    full = mc.ea_demosaic(Tensor(cfa), cfa_pattern="RGGB").realize()
    fused = mc.ea_demosaic(Tensor(cfa), cfa_pattern="RGGB").view(
        left=3,
        top=2,
        width=11,
        height=13,
        reset_origin=True,
    ).realize()
    # Fused EA vs a sliced full frame can differ by 1 ULP on some
    # platforms (reduction order).
    np.testing.assert_array_max_ulp(fused, full[2:15, 3:14], maxulp=1)


def test_ea_demosaic_fast_differs_from_ha():
    cfa = np.full((5, 5), 0.5, dtype=np.float32)
    cfa[1, 2] = 0.1
    cfa[3, 2] = 0.9
    cfa[2, 1] = 0.2
    cfa[2, 3] = 0.2
    cfa[2, 0] = 0.0
    cfa[2, 4] = 0.0
    ha = mc.ea_demosaic(Tensor(cfa), cfa_pattern="RGGB").realize()
    fast = mc.ea_demosaic(Tensor(cfa), cfa_pattern="RGGB", fast=True).realize()
    wrap = demosaic(
        Tensor(cfa), "RGGB", algorithm=DemosaicAlgorithm.EA_FAST
    ).realize()
    np.testing.assert_allclose(fast[2, 2, 1], 0.2, atol=1e-6)
    np.testing.assert_allclose(ha[2, 2, 1], 0.5, atol=1e-6)
    np.testing.assert_array_equal(fast, wrap)


def test_ea_demosaic_fast_timing_label():
    from muimg.common import PerfTimer
    from muimg.engines.graph import EngineTiming, engine_timing, set_engine_timing

    cfa = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    prev = engine_timing
    try:
        set_engine_timing(EngineTiming.OPS)
        with PerfTimer("root") as ha_root:
            mc.ea_demosaic(Tensor(cfa), cfa_pattern="RGGB").realize()
        with PerfTimer("root") as fast_root:
            mc.ea_demosaic(
                Tensor(cfa), cfa_pattern="RGGB", fast=True
            ).realize()
    finally:
        set_engine_timing(prev)

    assert [c.name for c in ha_root.children[0].children] == [
        "ea_demosaic (engine)"
    ]
    assert [c.name for c in fast_root.children[0].children] == [
        "ea_fast_demosaic (engine)"
    ]


def test_op_rejects_bad_channels():
    rgb = Tensor(np.zeros((2, 2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="expected 1 channel"):
        mc.bilinear_demosaic(rgb, cfa_pattern="RGGB")


def test_op_rejects_unknown_attr():
    x = Tensor(np.zeros((2, 2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="unknown attrs"):
        mc.matrix_3x3(x, matrix=np.eye(3, dtype=np.float32), extra=1)


def test_rejects_tensor_tensor_sub():
    a = Tensor(np.zeros((2, 2), dtype=np.float32))
    b = Tensor(np.ones((2, 2), dtype=np.float32))
    with pytest.raises(TypeError, match="tensor–tensor"):
        _ = a - b


def test_demosaic_tensor_lazy():
    """demosaic(Tensor) returns a lazy Tensor; compute materializes RGB."""
    rng = np.random.default_rng(0)
    cfa = rng.integers(0, 1000, size=(16, 16), dtype=np.uint16)
    out_t = demosaic(Tensor(cfa), "RGGB", algorithm=DemosaicAlgorithm.EA)
    assert out_t._node is not None
    out = out_t.realize()
    ref = demosaic(Tensor(cfa), "RGGB", algorithm=DemosaicAlgorithm.EA).realize()
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
    x = demosaic(x, "RGGB", algorithm=DemosaicAlgorithm.EA)
    x = mc.matrix_3x3(x, matrix=eye)
    x = mc.lut(x, lut=lut)
    out = x.realize()

    ref = demosaic(
        Tensor(cfa), "RGGB", algorithm=DemosaicAlgorithm.EA, dst_dtype="float32"
    )
    ref = mc.matrix_3x3(ref, matrix=eye)
    ref = mc.lut(ref, lut=lut).realize()
    assert out.shape == (16, 16, 3)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_apply_opcodes_single_execute():
    """Multi-opcode RGB chain runs one execute_graph."""
    from muimg.engines.core import _engine_load
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
    real = _engine_load.execute_graph

    def counting_execute(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    _engine_load.execute_graph = counting_execute
    try:
        out_t = apply_opcodes(Tensor(rgb), opcodes, use_bicubic=False)
        out = out_t.realize()
    finally:
        _engine_load.execute_graph = real

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
    """set_default_engine swaps the backend used by Tensor.realize()."""
    prev = get_default_engine()
    stub = _RecordingEngine()
    set_default_engine(stub)
    try:
        assert get_default_engine() is stub
        x = Tensor(np.ones((2, 2), dtype=np.float32)) - 0.0
        out = x.realize()
        assert stub.calls == [1]
        assert out.shape == (2, 2)
    finally:
        set_default_engine(prev)
        assert isinstance(get_default_engine(), CoreEngine)


def test_core_binaries_path():
    """CoreEngine package ships platform-tagged abi3 extensions in _binaries/."""
    import muimg.engines.core as core_pkg
    from pathlib import Path

    binaries = Path(core_pkg.__file__).resolve().parent / "_binaries"
    assert binaries.is_dir()
    libs = list(binaries.glob("_core_engine.*.abi3.so")) + list(
        binaries.glob("_core_engine.*.abi3.pyd")
    )
    assert libs, f"no _core_engine abi3 binaries under {binaries}"


def test_graph_op_cast_then_native_crop():
    from muimg.engines.pyops import cast_dtype_op

    src = np.arange(16, dtype=np.uint8).reshape(4, 4)
    x = cast_dtype_op(Tensor(src), "uint16")
    x = x.view(left=1, top=1, width=3, height=2)
    assert x._node is not None and x._node.op == "view" and x._node.fn is None
    out = x.realize()
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
    from muimg.engines.graph import (
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


def test_perftimer_missed_close_then_continue_at_parent():
    """Worker nests via ``PerfTimer.step``; outer continues under L0 after a missed close.

    Outer starts L0 (keeps the handle). A worker stacks L1/L2 with ``step()`` and
    skips ``L2.close()``. Outer then ``L0.start_step("L1b")`` — L0 auto-closes the
    abandoned L1 subtree (including L2) and opens L1b as the next child of L0.
    """
    from muimg.common import PerfTimer

    try:
        root = PerfTimer("root")
        # Outer stage (caller keeps L0).
        L0 = PerfTimer.step("L0")
        assert PerfTimer.current() is L0

        # Worker: only uses the stack, no parent handle.
        L1 = PerfTimer.step("L1")
        L2 = PerfTimer.step("L2")
        assert PerfTimer.current() is L2
        # Miss L2.close() (and L1.close()).

        # Outer continues under L0 — not PerfTimer.step(), which would nest under L2.
        L1b = L0.start_step("L1b")

        assert L2.end_time is not None
        assert L1.end_time is not None
        assert PerfTimer.current() is L1b
        assert [c.name for c in L0.children] == ["L1", "L1b"]
        assert L1b.parent is L0

        stack = PerfTimer._stack()
        assert L2 not in stack and L1 not in stack
        assert stack[-1] is L1b
        assert L0 in stack and root in stack

        L1b.close()
        L0.close()
        root.close()

        assert PerfTimer.current() is None
        assert PerfTimer._stack() == []
        assert root.get_report() != "broken stack"
        assert [c.name for c in L0.children] == ["L1", "L1b"]
        assert [c.name for c in L1.children] == ["L2"]
    finally:
        PerfTimer._stack().clear()


def test_compute_times_python_ops():
    from muimg.common import PerfTimer
    from muimg.engines.pyops import cast_dtype_op
    from muimg.engines.graph import EngineTiming, engine_timing, set_engine_timing

    src = np.arange(16, dtype=np.uint8).reshape(4, 4)
    x = cast_dtype_op(Tensor(src), "uint16")
    x = cast_dtype_op(x, "float32")

    prev = engine_timing
    try:
        set_engine_timing(EngineTiming.SEGMENTS)
        with PerfTimer("root") as root:
            parent = root.start_step("camera_space")
            out = x.realize()
            parent.close()
    finally:
        set_engine_timing(prev)

    np.testing.assert_allclose(out, src.astype(np.float32))
    names = [c.name for c in parent.children]
    assert names == ["cast_dtype_op (python)", "cast_dtype_op (python)"]
    assert all(c.get_elapsed_ms() >= 0.0 for c in parent.children)


def test_compute_times_engine_ops():
    from muimg.common import PerfTimer
    from muimg.engines.graph import EngineTiming, engine_timing, set_engine_timing

    prev = engine_timing
    try:
        set_engine_timing(EngineTiming.OPS)
        with PerfTimer("root") as root:
            out = (Tensor(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)) - 1.0)
            out = (out * 2.0).realize()
    finally:
        set_engine_timing(prev)

    np.testing.assert_allclose(out, [[0.0, 2.0], [4.0, 6.0]])
    assert [c.name for c in root.children] == ["graph_compute"]
    ops = root.children[0].children
    names = [c.name for c in ops]
    assert names == ["sub_scalar (engine)", "mul_scalar (engine)"]
    assert all(c.get_elapsed_ms() >= 0.0 for c in ops)
    assert ops[0].end_time == ops[1].start_time


def test_core_engine_segments_no_op_children():
    from muimg.common import PerfTimer
    from muimg.engines.graph import EngineTiming, engine_timing, set_engine_timing

    prev = engine_timing
    try:
        set_engine_timing(EngineTiming.SEGMENTS)
        with PerfTimer("root") as root:
            out = (Tensor(np.array([[1.0, 2.0]], dtype=np.float32)) * 2.0).realize()
    finally:
        set_engine_timing(prev)

    np.testing.assert_allclose(out, [[2.0, 4.0]])
    assert [c.name for c in root.children] == ["graph_compute"]
    assert root.children[0].children == []


def test_core_engine_off_no_rows_even_with_open_timer():
    from muimg.common import PerfTimer
    from muimg.engines.graph import EngineTiming, engine_timing, set_engine_timing

    prev = engine_timing
    try:
        set_engine_timing(EngineTiming.OFF)
        with PerfTimer("root") as root:
            (Tensor(np.array([[1.0]], dtype=np.float32)) * 2.0).realize()
    finally:
        set_engine_timing(prev)

    assert root.children == []


def test_compute_nests_under_current_stack_top():
    from muimg.common import PerfTimer
    from muimg.engines.graph import EngineTiming, engine_timing, set_engine_timing

    prev = engine_timing
    try:
        set_engine_timing(EngineTiming.SEGMENTS)
        with PerfTimer("root") as root:
            fence = root.start_step("fence")
            (Tensor(np.array([[1.0]], dtype=np.float32)) * 2.0).realize()
            fence.close()
    finally:
        set_engine_timing(prev)

    assert [c.name for c in fence.children] == ["graph_compute"]


def test_compute_ops_under_graph_compute():
    from muimg.common import PerfTimer
    from muimg.engines.graph import EngineTiming, engine_timing, set_engine_timing

    prev = engine_timing
    try:
        set_engine_timing(EngineTiming.OPS)
        with PerfTimer("root") as root:
            parent = root.start_step("camera_space")
            out = (Tensor(np.array([[1.0, 2.0]], dtype=np.float32)) * 2.0).realize()
            parent.close()
    finally:
        set_engine_timing(prev)

    np.testing.assert_allclose(out, [[2.0, 4.0]])
    assert [c.name for c in parent.children] == ["graph_compute"]
    assert [c.name for c in parent.children[0].children] == ["mul_scalar (engine)"]


def test_graph_op_splits_engine_segments():
    from muimg.engines.core import _engine_load
    from muimg.engines.pyops import cast_dtype_op

    src = np.arange(16, dtype=np.float32).reshape(4, 4)
    x = Tensor(src) - 0.0
    x = cast_dtype_op(x, "float32")  # python fence between engine segments
    x = x * 2.0

    calls = {"n": 0}
    real = _engine_load.execute_graph

    def counting_execute(*args, **kwargs):
        calls["n"] += 1
        return real(*args, **kwargs)

    _engine_load.execute_graph = counting_execute
    try:
        out = x.realize()
    finally:
        _engine_load.execute_graph = real

    assert calls["n"] == 2
    np.testing.assert_allclose(out, src * 2.0)


def test_demosaic_op_lazy():
    from muimg.engines.pyops import demosaic_op

    rng = np.random.default_rng(4)
    cfa = rng.integers(0, 1000, size=(16, 16), dtype=np.uint16)
    out = demosaic_op(Tensor(cfa), "RGGB", "VNG").realize()
    ref = demosaic(Tensor(cfa), "RGGB", algorithm=DemosaicAlgorithm.VNG).realize()
    np.testing.assert_array_equal(out, ref)

    out_ea = demosaic_op(Tensor(cfa), "RGGB", "OPENCV_EA").realize()
    ref_ea = demosaic(
        Tensor(cfa), "RGGB", algorithm=DemosaicAlgorithm.OPENCV_EA
    ).realize()
    np.testing.assert_array_equal(out_ea, ref_ea)


def test_ingest_seals_view_and_base():
    parent = np.array(
        np.arange(16, dtype=np.float32).reshape(4, 4), copy=True
    )
    view = parent[1:3, 1:3]
    t = Tensor(view)
    assert view.base is parent
    assert not t._data.flags.writeable
    assert not view.flags.writeable
    assert not parent.flags.writeable
    with pytest.raises(ValueError):
        parent[0, 0] = 99.0


def test_realized_view_walks_upstream_for_canvas_crop():
    """A realized view's _data is the window; canvas pixels still need the graph."""
    src = np.arange(5 * 7, dtype=np.float32).reshape(5, 7)
    viewed = Tensor(src).view(left=1, top=2, width=3, height=2)
    viewed.realize()
    assert viewed._data is not None
    extra = viewed.crop(left=-1, top=0, width=5, height=2)
    np.testing.assert_array_equal(extra.realize(), src[2:4, 0:5])


def test_realized_crop_is_extra_bind(monkeypatch):
    """A hard crop's cache is an extra in_bind; the submitted graph still has the crop."""
    from muimg.engines.core import _engine_load

    calls: List[dict] = []
    real = _engine_load.execute_graph

    def wrap(graph, in_binds, out_binds, record_ops=False):
        calls.append({"graph": graph, "in_binds": dict(in_binds)})
        return real(graph, in_binds, out_binds, record_ops)

    monkeypatch.setattr(_engine_load, "execute_graph", wrap)

    src = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)
    cropped = Tensor(src).crop(left=1, top=1, width=2, height=2)
    cropped.realize()
    assert len(calls) == 1
    extra = cropped * 2.0
    out = extra.realize()
    assert len(calls) == 2
    graph = calls[1]["graph"]
    ops = [n["op"] for n in graph["nodes"]]
    assert ops == ["view", "mul_scalar"]
    assert len(calls[1]["in_binds"]) == 2
    np.testing.assert_array_equal(out, src[1:3, 1:3] * 2.0)


def test_realize_caches_and_force_recompute():
    prev = get_default_engine()
    stub = _RecordingEngine()
    set_default_engine(stub)
    try:
        x = Tensor(np.ones((2, 2), dtype=np.float32)) - 0.0
        first = x.realize()
        assert stub.calls == [1]
        assert x._data is first
        assert not first.flags.writeable
        second = x.realize()
        assert second is first
        assert stub.calls == [1]
        third = x.realize(force_recompute=True)
        assert stub.calls == [1, 1]
        assert third is not first
        assert x._data is third
        assert not third.flags.writeable
    finally:
        set_default_engine(prev)


def test_op_node_is_frozen():
    x = Tensor(np.ones((2, 2), dtype=np.float32)) - 1.0
    assert x._node is not None
    with pytest.raises(AttributeError):
        x._node.op = "mul_scalar"
    with pytest.raises(TypeError):
        x._node.attrs["value"] = 0.0


def test_strided_numpy_crop_ported_and_image_op():
    parent = np.arange(16, dtype=np.float32).reshape(4, 4)
    crop = parent[1:3, 1:3]
    got = (Tensor(crop) - 1.0).realize()
    np.testing.assert_array_equal(got, crop - 1.0)

    identity = mc.apply_flat_gain_map(
        Tensor(crop),
        gain_map=[1.0, 1.0, 1.0, 1.0],
        gain_h=2,
        gain_w=2,
    ).realize()
    np.testing.assert_array_equal(identity, crop)


def test_fortran_array_copied_on_ingest():
    arr = np.asfortranarray(np.arange(16, dtype=np.float32).reshape(4, 4))
    t = Tensor(arr)
    assert t._data.strides[1] == t._data.dtype.itemsize
    np.testing.assert_array_equal((t - 0.0).realize(), arr)


def test_stepped_slice_copied_on_ingest():
    parent = np.arange(16, dtype=np.float32).reshape(4, 4)
    stepped = parent[::2, ::2]
    t = Tensor(stepped)
    assert t._data.strides[1] == t._data.dtype.itemsize
    assert t._data.base is not parent
    np.testing.assert_array_equal(t._data, stepped)
