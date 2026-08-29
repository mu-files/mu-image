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
from muimg.raw_render import DemosaicAlgorithm, apply_tiff_orientation, demosaic
from muimg.tensor import Tensor


def test_catalog_engine_ops_io():
    """engines.ops carries EngineOp callables + OPS_BY_NAME."""
    assert "sub_scalar" in OPS_BY_NAME
    assert "crop" in OPS_BY_NAME
    assert "orientation" in OPS_BY_NAME
    assert isinstance(engine_ops.bilinear_demosaic, EngineOp)
    assert engine_ops.bilinear_demosaic._in_channels == 1
    x = Tensor(np.zeros((2, 2), dtype=np.float32))
    assert engine_ops.bilinear_demosaic.infer_out_meta(x, {}).channels == 3
    assert callable(engine_ops.matrix_3x3)
    assert callable(engine_ops.lut)
    assert callable(flush)
    assert "crop" in get_default_engine().supported_ops


def test_tensor_meta_origin_default():
    x = Tensor(np.zeros((4, 6), dtype=np.float32))
    assert x.meta.origin == (0, 0)
    y = x - 1.0
    assert y.meta.origin == (0, 0)
    assert y.meta.height == 4 and y.meta.width == 6


def test_tensor_origin_kwarg():
    src = Tensor(np.zeros((4, 6), dtype=np.float32), origin=(-3, -5))
    assert src.meta.origin == (-3, -5)


def test_crop_emit_meta_updates_origin_and_size():
    """Default crop accumulates origin; reset_origin re-zeros world."""
    base = Tensor(np.arange(5 * 7, dtype=np.float32).reshape(5, 7))
    cat = engine_ops.crop(base, left=1, top=2, width=3, height=2)
    assert cat.meta.height == 2 and cat.meta.width == 3
    assert cat.meta.origin == (2, 1)
    assert cat._node is not None and cat._node.op == "crop"
    assert cat._node.fn is None

    cat2 = engine_ops.crop(cat, left=1, top=0, width=2, height=1)
    assert cat2.meta.origin == (2, 2)

    reset = engine_ops.crop(base, left=1, top=2, width=3, height=2, reset_origin=True)
    assert reset.meta.origin == (0, 0)

    out = cat.compute()
    assert out.shape == (2, 3)
    np.testing.assert_array_equal(out, np.asarray(base)[2:4, 1:4])


def test_orientation_emit_meta_swaps_hw():
    """TIFF 5–8 swap H×W; 1–4 keep size; origin unchanged."""
    base = Tensor(np.zeros((4, 6, 3), dtype=np.float32))
    same = engine_ops.orientation(base, orientation=3)
    assert same.meta.height == 4 and same.meta.width == 6
    assert same.meta.origin == (0, 0)
    assert same._node is not None and same._node.op == "orientation"

    rot = engine_ops.orientation(base, orientation=6)
    assert rot.meta.height == 6 and rot.meta.width == 4
    assert rot.meta.origin == (0, 0)

    with pytest.raises(ValueError, match="invalid TIFF code"):
        engine_ops.orientation(base, orientation=0)


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
    """engine_ops.orientation is issued natively for TIFF 1–8."""
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
            out = engine_ops.orientation(Tensor(src), orientation=code).compute()
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
        x = engine_ops.orientation(x, orientation=code)
        x = x * 2.0
        x = engine_ops.orientation(x, orientation=inv)
        out = x.compute()

        assert len(calls) == 1, f"code {code}"
        assert [n["op"] for n in calls[0]["nodes"]] == [
            "sub_scalar",
            "orientation",
            "mul_scalar",
            "orientation",
        ], f"code {code}"
        np.testing.assert_allclose(out, expect, err_msg=f"code {code}")
        assert out.shape == src.shape, f"code {code}"


def test_apply_tiff_orientation_uses_engine_for_flips():
    """Render helper emits engine_ops.orientation; codes 2/4 flip (not no-ops)."""
    src = np.arange(5 * 7, dtype=np.float32).reshape(5, 7)
    t = Tensor(src)
    assert apply_tiff_orientation(t, 1) is t

    for code in (2, 4):
        out_t = apply_tiff_orientation(Tensor(src), code)
        assert out_t._node is not None and out_t._node.op == "orientation"
        expect = np.ascontiguousarray(_tiff_orientation_numpy(src, code))
        np.testing.assert_array_equal(out_t.compute(), expect)


def test_crop_emit_allows_oob_window_rejects_disjoint():
    x = Tensor(np.zeros((4, 4), dtype=np.float32))
    # An OOB window is allowed (the engine fills the out-of-bounds part
    # with the crop's pad mode) ...
    t = engine_ops.crop(x, left=1, top=1, width=4, height=2)
    assert t._node is not None and t._node.op == "crop"
    assert t.meta.height == 2 and t.meta.width == 4
    t = engine_ops.crop(x, left=-2, top=-1, width=3, height=3)
    assert t.meta.height == 3 and t.meta.width == 3
    # ... but the window must still overlap the input.
    with pytest.raises(ValueError, match="does not overlap"):
        engine_ops.crop(x, left=4, top=0, width=2, height=2)
    with pytest.raises(ValueError, match="does not overlap"):
        engine_ops.crop(x, left=0, top=-3, width=2, height=3)


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
    x = engine_ops.crop(x, left=1, top=1, width=2, height=2)
    x = x * 2.0
    out = x.compute()

    assert len(calls) == 1
    ops = [n["op"] for n in calls[0]["nodes"]]
    assert ops == ["sub_scalar", "crop", "mul_scalar"]
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
    x = engine_ops.crop(x, left=1, top=1, width=6, height=6)  # → inp[1:7, 1:7]
    x = x - 1.0
    x = engine_ops.crop(x, left=1, top=1, width=4, height=4)  # → inp[2:6, 2:6] after first crop
    x = x - 2.0
    out = x.compute()

    assert len(calls) == 1
    ops = [n["op"] for n in calls[0]["nodes"]]
    assert ops == ["crop", "sub_scalar", "crop", "sub_scalar"]

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
    out = x.compute()
    np.testing.assert_allclose(out, [[0.0, 2.0], [4.0, 6.0]])


def test_matrix_3x3_identity():
    eye = np.eye(3, dtype=np.float32)
    inp = np.array([[[0.25, 0.5, 0.75]]], dtype=np.float32)
    out = engine_ops.matrix_3x3(Tensor(inp), matrix=eye).compute()
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


def test_ea_demosaic_rggb():
    cfa = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    out = engine_ops.ea_demosaic(Tensor(cfa), cfa_pattern="RGGB").compute()
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
    full = engine_ops.ea_demosaic(Tensor(cfa), cfa_pattern="RGGB").compute()
    fused = engine_ops.crop(
        engine_ops.ea_demosaic(Tensor(cfa), cfa_pattern="RGGB"),
        left=3,
        top=2,
        width=11,
        height=13,
        reset_origin=True,
    ).compute()
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
    ha = engine_ops.ea_demosaic(Tensor(cfa), cfa_pattern="RGGB").compute()
    fast = engine_ops.ea_demosaic(Tensor(cfa), cfa_pattern="RGGB", fast=True).compute()
    wrap = demosaic(
        Tensor(cfa), "RGGB", algorithm=DemosaicAlgorithm.EA_FAST
    ).compute()
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
            engine_ops.ea_demosaic(Tensor(cfa), cfa_pattern="RGGB").compute()
        with PerfTimer("root") as fast_root:
            engine_ops.ea_demosaic(
                Tensor(cfa), cfa_pattern="RGGB", fast=True
            ).compute()
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
        engine_ops.bilinear_demosaic(rgb, cfa_pattern="RGGB")


def test_op_rejects_unknown_attr():
    x = Tensor(np.zeros((2, 2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="unknown attrs"):
        engine_ops.matrix_3x3(x, matrix=np.eye(3, dtype=np.float32), extra=1)


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
    out = out_t.compute()
    ref = demosaic(Tensor(cfa), "RGGB", algorithm=DemosaicAlgorithm.EA).compute()
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
    x = engine_ops.matrix_3x3(x, matrix=eye)
    x = engine_ops.lut(x, lut=lut)
    out = x.compute()

    ref = demosaic(
        Tensor(cfa), "RGGB", algorithm=DemosaicAlgorithm.EA, dst_dtype="float32"
    )
    ref = engine_ops.matrix_3x3(ref, matrix=eye)
    ref = engine_ops.lut(ref, lut=lut).compute()
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
        out = out_t.compute()
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
    x = engine_ops.crop(x, left=1, top=1, width=3, height=2)
    assert x._node is not None and x._node.op == "crop" and x._node.fn is None
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
            out = x.compute()
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


def test_core_engine_segments_no_op_children():
    from muimg.common import PerfTimer
    from muimg.engines.graph import EngineTiming, engine_timing, set_engine_timing

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


def test_core_engine_off_no_rows_even_with_open_timer():
    from muimg.common import PerfTimer
    from muimg.engines.graph import EngineTiming, engine_timing, set_engine_timing

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
    from muimg.engines.graph import EngineTiming, engine_timing, set_engine_timing

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
    from muimg.engines.graph import EngineTiming, engine_timing, set_engine_timing

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
        out = x.compute()
    finally:
        _engine_load.execute_graph = real

    assert calls["n"] == 2
    np.testing.assert_allclose(out, src * 2.0)


def test_demosaic_op_lazy():
    from muimg.engines.pyops import demosaic_op

    rng = np.random.default_rng(4)
    cfa = rng.integers(0, 1000, size=(16, 16), dtype=np.uint16)
    out = demosaic_op(Tensor(cfa), "RGGB", "VNG").compute()
    ref = demosaic(Tensor(cfa), "RGGB", algorithm=DemosaicAlgorithm.VNG).compute()
    np.testing.assert_array_equal(out, ref)

    out_ea = demosaic_op(Tensor(cfa), "RGGB", "OPENCV_EA").compute()
    ref_ea = demosaic(
        Tensor(cfa), "RGGB", algorithm=DemosaicAlgorithm.OPENCV_EA
    ).compute()
    np.testing.assert_array_equal(out_ea, ref_ea)
