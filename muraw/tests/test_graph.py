"""Array / engines.graph tests + eager flush at python barriers."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pytest

import muimage as mi
from muraw.engines import get_default_engine, set_default_engine
from muraw.engines.core import CoreEngine
from muraw.engines import graph
from muraw.engines.graph import EngineOp, OpMeta, flush, graph_op
from muraw.engines.ops import OPS_BY_NAME
from muraw.raw_render import DemosaicAlgorithm, demosaic
from muraw.array import Array, ArrayMeta, ElementType, rot90
from conftest import generate_rgb_ramp


def _fence_cast_out_meta(t: Array, attrs: dict) -> ArrayMeta:
    dest = ElementType(attrs["dst_dtype"])
    return t.meta.copy(dtype=dest)


@graph_op(out_meta=_fence_cast_out_meta)
def _fence_cast(arr: np.ndarray, dst_dtype: str) -> np.ndarray:
    dest = ElementType(dst_dtype)
    return arr.astype(dest.numpy_dtype, copy=False)


def test_catalog_engine_ops_io():
    """engines.ops carries EngineOp callables + OPS_BY_NAME."""
    assert "sub_scalar" in OPS_BY_NAME
    assert "view" in OPS_BY_NAME
    assert "pad" in OPS_BY_NAME
    assert "orientation" in OPS_BY_NAME
    assert "cast_dtype" in OPS_BY_NAME
    assert callable(mi.cast_dtype)
    assert callable(mi.convert_dtype)
    assert isinstance(mi.cfa_bilinear_demosaic, EngineOp)
    assert mi.cfa_bilinear_demosaic._in_channels == (1,)
    x = Array(np.zeros((2, 2), dtype=np.float32))
    assert mi.cfa_bilinear_demosaic.infer_out_meta(x, {}).channels == 3
    assert callable(mi.rgb_matrix_3x3)
    assert callable(mi.lut)
    assert callable(flush)
    assert "view" in get_default_engine().supported_ops


def _engine_op(
    name: str, in_channels: tuple, variable_input: bool = False
) -> EngineOp:
    """An engine op that is not in the catalog, for building nodes only."""
    return EngineOp(
        meta=OpMeta(name=name),
        _out_dtype=graph._out_dtype_same,
        _out_channels=graph._out_channels_same,
        _in_channels=in_channels,
        _n_inputs=len(in_channels),
        _variable_input=variable_input,
    )


def test_two_input_op_builds_a_node_with_both_inputs():
    add2 = _engine_op("add2", (None, 1))
    first = Array(np.zeros((2, 3, 3), dtype=np.float32))
    second = Array(np.zeros((2, 3), dtype=np.float32))
    out = add2(first, second)
    assert out._node.op == "add2"
    assert out._node.inputs == (first, second)
    assert out.meta == first.meta


def test_two_input_op_checks_each_input():
    add2 = _engine_op("add2", (None, 1))
    rgb = Array(np.zeros((2, 3, 3), dtype=np.float32))
    with pytest.raises(ValueError, match=r"input\[1\]: expected 1 channel"):
        add2(rgb, rgb)
    with pytest.raises(ValueError, match="expected 2 input"):
        add2(rgb)
    with pytest.raises(ValueError, match="expected 2 input"):
        add2(rgb, rgb, rgb)


def test_variable_input_op_takes_one_or_more_of_its_last_input():
    sum_n = _engine_op("sum_n", (1,), variable_input=True)
    mono = Array(np.zeros((2, 3), dtype=np.float32))
    assert sum_n(mono)._node.inputs == (mono,)
    assert sum_n(mono, mono, mono, mono)._node.inputs == (mono,) * 4
    rgb = Array(np.zeros((2, 3, 3), dtype=np.float32))
    with pytest.raises(ValueError, match=r"input\[2\]: expected 1 channel"):
        sum_n(mono, mono, rgb)


def test_op_with_no_inputs_takes_only_its_output_meta():
    fill = OPS_BY_NAME["fill"]
    like = Array(np.zeros((2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="takes no inputs"):
        fill(like, like, value=1.0)


def test_array_meta_origin_default():
    x = Array(np.zeros((4, 6), dtype=np.float32))
    assert x.meta.origin == (0, 0)
    assert x.meta.canvas == (0, 0, 6, 4)
    y = x - 1.0
    assert y.meta.origin == (0, 0)
    assert y.meta.canvas == (0, 0, 6, 4)
    assert y.meta.height == 4 and y.meta.width == 6


def test_array_origin_kwarg():
    src = Array(np.zeros((4, 6), dtype=np.float32), origin=(-3, -5))
    assert src.meta.origin == (-3, -5)
    assert src.meta.canvas == (-5, -3, 6, 4)


def test_view_emit_meta_updates_origin_and_size():
    """Default view accumulates origin; reset_origin re-zeros world."""
    base = Array(np.arange(5 * 7, dtype=np.float32).reshape(5, 7))
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

    # The parent canvas moves by the first sample in the shared system,
    # which includes the parent's own origin.
    shifted = Array(np.zeros((10, 10), dtype=np.float32), origin=(4, 5))
    reset_shifted = shifted.view(left=2, top=3, width=4, height=4, reset_origin=True)
    assert reset_shifted.meta.canvas == (-2, -3, 10, 10)

    out = cat.realize()
    assert out.shape == (2, 3)
    np.testing.assert_array_equal(out, np.asarray(base)[2:4, 1:4])


def test_orientation_emit_meta_swaps_hw():
    """TIFF 5–8 swap H×W; 1–4 keep size; origin unchanged."""
    base = Array(np.zeros((4, 6, 3), dtype=np.float32))
    same = mi.orientation(base, orientation=3)
    assert same.meta.height == 4 and same.meta.width == 6
    assert same.meta.origin == (0, 0)
    assert same._node is not None and same._node.op == "orientation"

    rot = mi.orientation(base, orientation=6)
    assert rot.meta.height == 6 and rot.meta.width == 4
    assert rot.meta.origin == (0, 0)

    with pytest.raises(ValueError, match="invalid TIFF code"):
        mi.orientation(base, orientation=0)


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
    """mi.orientation is issued natively for TIFF 1–8."""
    from muraw.engines.core import _engine_load

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
            src_t = Array(src)
            t = mi.orientation(src_t, orientation=code)
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
    from muraw.engines.core import _engine_load

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
        x = Array(src) - 1.0
        x = mi.orientation(x, orientation=code)
        x = x * 2.0
        x = mi.orientation(x, orientation=inv)
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
    x = Array(np.zeros((4, 4), dtype=np.float32))
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
    from muraw.engines.core import _engine_load

    calls: List[dict] = []
    real = _engine_load.execute_graph

    def wrap(graph, in_binds, out_binds, record_ops=False):
        calls.append(graph)
        return real(graph, in_binds, out_binds, record_ops)

    monkeypatch.setattr(_engine_load, "execute_graph", wrap)

    inp = np.array(
        [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32
    )
    x = Array(inp) - 1.0
    x = x.view(left=1, top=1, width=2, height=2)
    x = x * 2.0
    out = x.realize()

    assert len(calls) == 1
    ops = [n["op"] for n in calls[0]["nodes"]]
    assert ops == ["sub_scalar", "view", "mul_scalar"]
    np.testing.assert_allclose(out, [[8.0, 10.0], [14.0, 16.0]])


def test_crop_sub_crop_sub_ramp(monkeypatch):
    """input → crop → sub → crop → sub in one segment; known ramp pixels."""
    from muraw.engines.core import _engine_load

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

    x = Array(inp)
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
    x = Array(inp)
    x = x - 1.0
    x = x * 2.0
    out = x.realize()
    np.testing.assert_allclose(out, [[0.0, 2.0], [4.0, 6.0]])


def test_rgb_matrix_3x3_identity():
    eye = np.eye(3, dtype=np.float32)
    inp = np.array([[[0.25, 0.5, 0.75]]], dtype=np.float32)
    out = mi.rgb_matrix_3x3(Array(inp), matrix=eye).realize()
    np.testing.assert_allclose(out, inp)


def test_lut_identity_rgb():
    inp = np.array([[[0.0, 0.5, 1.0]]], dtype=np.float32)
    out = mi.lut(Array(inp), lut=[0.0, 1.0]).realize()
    np.testing.assert_allclose(out, inp)


def test_convert_type_f32_to_f16_ramp():
    """Public dest is np.float16. Crate GraphRun is Vec<u16>; this is the gap."""
    rgb = generate_rgb_ramp(1280, 720, dtype=np.float32)
    got = np.asarray(Array(rgb).convert_type(np.float16))
    expect = rgb.astype(np.float16)
    assert np.isfinite(got).all()
    assert np.array_equal(got, expect)


def test_convert_type_f32_to_f16_subnormals():
    """Values below 2^-14 store as float16 subnormals. The noisy write_dng
    fixtures produce them via np.clip(ramp + noise, 0, 1); on Windows they
    take the MSVC software float16 converter that clang/gcc never compile.
    """
    src = np.linspace(-6.2e-5, 6.2e-5, 12288, dtype=np.float32).reshape(4, 3072)
    got = np.asarray(Array(src).convert_type(np.float16))
    expect = src.astype(np.float16)
    assert np.isfinite(got).all()
    assert np.array_equal(got, expect)


def test_cfa_bilinear_demosaic_rggb():
    cfa = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    out = mi.cfa_bilinear_demosaic(Array(cfa), cfa_pattern="RGGB").realize()
    assert out.shape == (2, 2, 3)
    np.testing.assert_allclose(out[0, 0, 0], 0.2)


def test_cfa_ea_demosaic_rggb():
    cfa = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    out = mi.cfa_ea_demosaic(Array(cfa), cfa_pattern="RGGB").realize()
    assert out.shape == (2, 2, 3)
    np.testing.assert_allclose(
        out,
        [
            [[0.2, 0.5, 0.8], [0.1, 0.4, 0.7]],
            [[0.3, 0.6, 0.9], [0.2, 0.5, 0.8]],
        ],
        atol=1e-6,
    )


def test_cfa_ea_demosaic_then_crop_matches_slice():
    """Fused EA + DefaultCrop (nonzero origin) must match a sliced full frame.

    Tile last-compute used to address the cropped dest with CFA coordinates
    and overwrite past the buffer (R5 create_dng_from_page / render path).
    """
    rng = np.random.default_rng(0)
    cfa = rng.random((17, 19), dtype=np.float32)
    full = mi.cfa_ea_demosaic(Array(cfa), cfa_pattern="RGGB").realize()
    fused = mi.cfa_ea_demosaic(Array(cfa), cfa_pattern="RGGB").view(
        left=3,
        top=2,
        width=11,
        height=13,
        reset_origin=True,
    ).realize()
    # Fused EA vs a sliced full frame can differ by 1 ULP on some
    # platforms (reduction order).
    np.testing.assert_array_max_ulp(fused, full[2:15, 3:14], maxulp=1)


def test_cfa_ea_demosaic_fast_differs_from_ha():
    cfa = np.full((5, 5), 0.5, dtype=np.float32)
    cfa[1, 2] = 0.1
    cfa[3, 2] = 0.9
    cfa[2, 1] = 0.2
    cfa[2, 3] = 0.2
    cfa[2, 0] = 0.0
    cfa[2, 4] = 0.0
    ha = mi.cfa_ea_demosaic(Array(cfa), cfa_pattern="RGGB").realize()
    fast = mi.cfa_ea_demosaic(Array(cfa), cfa_pattern="RGGB", fast=True).realize()
    wrap = demosaic(
        Array(cfa), "RGGB", algorithm=DemosaicAlgorithm.EA_FAST
    ).realize()
    np.testing.assert_allclose(fast[2, 2, 1], 0.2, atol=1e-6)
    np.testing.assert_allclose(ha[2, 2, 1], 0.5, atol=1e-6)
    np.testing.assert_array_equal(fast, wrap)


def test_cfa_ea_demosaic_fast_timing_label():
    from muraw.common import PerfTimer
    from muraw.engines.graph import EngineTiming, engine_timing, set_engine_timing

    cfa = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    prev = engine_timing
    try:
        set_engine_timing(EngineTiming.OPS)
        with PerfTimer("root") as ha_root:
            mi.cfa_ea_demosaic(Array(cfa), cfa_pattern="RGGB").realize()
        with PerfTimer("root") as fast_root:
            mi.cfa_ea_demosaic(
                Array(cfa), cfa_pattern="RGGB", fast=True
            ).realize()
    finally:
        set_engine_timing(prev)

    assert [c.name for c in ha_root.children[0].children] == [
        "cfa_ea_demosaic (engine)"
    ]
    assert [c.name for c in fast_root.children[0].children] == [
        "ea_fast_demosaic (engine)"
    ]


def test_op_rejects_bad_channels():
    rgb = Array(np.zeros((2, 2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="expected 1 channel"):
        mi.cfa_bilinear_demosaic(rgb, cfa_pattern="RGGB")


def test_op_rejects_unknown_attr():
    x = Array(np.zeros((2, 2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="unknown attrs"):
        mi.rgb_matrix_3x3(x, matrix=np.eye(3, dtype=np.float32), extra=1)


def test_rejects_array_array_sub():
    a = Array(np.zeros((2, 2), dtype=np.float32))
    b = Array(np.ones((2, 2), dtype=np.float32))
    with pytest.raises(TypeError, match="array–array"):
        _ = a - b


def test_demosaic_array_lazy():
    """demosaic(Array) returns a lazy Array; compute materializes RGB."""
    rng = np.random.default_rng(0)
    cfa = rng.integers(0, 1000, size=(16, 16), dtype=np.uint16)
    out_t = demosaic(Array(cfa), "RGGB", algorithm=DemosaicAlgorithm.EA)
    assert out_t._node is not None
    out = out_t.realize()
    ref = demosaic(Array(cfa), "RGGB", algorithm=DemosaicAlgorithm.EA).realize()
    assert out.shape == (16, 16, 3)
    np.testing.assert_array_equal(out, ref)


def test_flush_then_engine_again():
    """Normalize (engine) → demosaic(Array) → matrix+lut (same DAG)."""
    rng = np.random.default_rng(1)
    cfa = (
        rng.integers(100, 1000, size=(16, 16), dtype=np.uint16).astype(np.float32)
        / 1000.0
    )

    eye = np.eye(3, dtype=np.float32)
    lut = np.array([0.0, 1.0], dtype=np.float32)

    x = Array(cfa)
    x = x - 0.0
    x = x * 1.0
    x = demosaic(x, "RGGB", algorithm=DemosaicAlgorithm.EA)
    x = mi.rgb_matrix_3x3(x, matrix=eye)
    x = mi.lut(x, lut=lut)
    out = x.realize()

    ref = demosaic(
        Array(cfa), "RGGB", algorithm=DemosaicAlgorithm.EA, dst_dtype="float32"
    )
    ref = mi.rgb_matrix_3x3(ref, matrix=eye)
    ref = mi.lut(ref, lut=lut).realize()
    assert out.shape == (16, 16, 3)
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-5)


def test_apply_opcodes_single_execute():
    """Multi-opcode RGB chain runs one execute_graph."""
    from muraw.engines.core import _engine_load
    from muraw.raw_render import apply_opcodes

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
        out_t = apply_opcodes(Array(rgb), opcodes, use_bicubic=False)
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
        nodes: List[Array],
        values: Dict[int, np.ndarray],
        outputs: List[Array],
    ) -> None:
        self.calls.append(len(nodes))
        # Produce zeros for outputs (enough to exercise the dispatch path).
        for t in outputs:
            values[id(t)] = np.zeros(t.meta.shape, dtype=np.float32)


def test_set_default_engine_stub():
    """set_default_engine swaps the backend used by Array.realize()."""
    prev = get_default_engine()
    stub = _RecordingEngine()
    set_default_engine(stub)
    try:
        assert get_default_engine() is stub
        x = Array(np.ones((2, 2), dtype=np.float32)) - 0.0
        out = x.realize()
        assert stub.calls == [1]
        assert out.shape == (2, 2)
    finally:
        set_default_engine(prev)
        assert isinstance(get_default_engine(), CoreEngine)


def test_core_binaries_path():
    """CoreEngine package ships platform-tagged abi3 extensions in _binaries/."""
    import muraw.engines.core as core_pkg
    from pathlib import Path

    binaries = Path(core_pkg.__file__).resolve().parent / "_binaries"
    assert binaries.is_dir()
    libs = list(binaries.glob("_core_engine.*.abi3.so")) + list(
        binaries.glob("_core_engine.*.abi3.pyd")
    )
    assert libs, f"no _core_engine abi3 binaries under {binaries}"


def test_graph_op_cast_then_native_crop():
    src = np.arange(16, dtype=np.uint8).reshape(4, 4)
    x = _fence_cast(Array(src), "uint16")
    x = x.view(left=1, top=1, width=3, height=2)
    assert x._node is not None and x._node.op == "view" and x._node.fn is None
    out = x.realize()
    np.testing.assert_array_equal(out, src.astype(np.uint16)[1:3, 1:4])


def test_add_completed_step_duration():
    from muraw.common import PerfTimer

    root = PerfTimer("root")
    child = root.add_completed_step("native_op (engine)", 0.025)
    assert child.end_time is not None
    assert abs(child.get_elapsed_ms() - 25.0) < 1.0
    assert root.children == [child]
    root.close()


def test_add_completed_steps_sequential_no_overlap():
    from muraw.common import PerfTimer

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
    from muraw.engines.graph import (
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
    from muraw.common import PerfTimer

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
    from muraw.common import PerfTimer

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
    from muraw.common import PerfTimer

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
    from muraw.common import PerfTimer

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
    from muraw.common import PerfTimer

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
    from muraw.common import PerfTimer
    from muraw.engines.graph import EngineTiming, engine_timing, set_engine_timing

    src = np.arange(16, dtype=np.uint8).reshape(4, 4)
    x = _fence_cast(Array(src), "uint16")
    x = _fence_cast(x, "float32")

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
    assert names == ["_fence_cast (python)", "_fence_cast (python)"]
    assert all(c.get_elapsed_ms() >= 0.0 for c in parent.children)


def test_compute_times_engine_ops():
    from muraw.common import PerfTimer
    from muraw.engines.graph import EngineTiming, engine_timing, set_engine_timing

    prev = engine_timing
    try:
        set_engine_timing(EngineTiming.OPS)
        with PerfTimer("root") as root:
            out = (Array(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)) - 1.0)
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
    from muraw.common import PerfTimer
    from muraw.engines.graph import EngineTiming, engine_timing, set_engine_timing

    prev = engine_timing
    try:
        set_engine_timing(EngineTiming.SEGMENTS)
        with PerfTimer("root") as root:
            out = (Array(np.array([[1.0, 2.0]], dtype=np.float32)) * 2.0).realize()
    finally:
        set_engine_timing(prev)

    np.testing.assert_allclose(out, [[2.0, 4.0]])
    assert [c.name for c in root.children] == ["graph_compute"]
    assert root.children[0].children == []


def test_core_engine_off_no_rows_even_with_open_timer():
    from muraw.common import PerfTimer
    from muraw.engines.graph import EngineTiming, engine_timing, set_engine_timing

    prev = engine_timing
    try:
        set_engine_timing(EngineTiming.OFF)
        with PerfTimer("root") as root:
            (Array(np.array([[1.0]], dtype=np.float32)) * 2.0).realize()
    finally:
        set_engine_timing(prev)

    assert root.children == []


def test_compute_nests_under_current_stack_top():
    from muraw.common import PerfTimer
    from muraw.engines.graph import EngineTiming, engine_timing, set_engine_timing

    prev = engine_timing
    try:
        set_engine_timing(EngineTiming.SEGMENTS)
        with PerfTimer("root") as root:
            fence = root.start_step("fence")
            (Array(np.array([[1.0]], dtype=np.float32)) * 2.0).realize()
            fence.close()
    finally:
        set_engine_timing(prev)

    assert [c.name for c in fence.children] == ["graph_compute"]


def test_compute_ops_under_graph_compute():
    from muraw.common import PerfTimer
    from muraw.engines.graph import EngineTiming, engine_timing, set_engine_timing

    prev = engine_timing
    try:
        set_engine_timing(EngineTiming.OPS)
        with PerfTimer("root") as root:
            parent = root.start_step("camera_space")
            out = (Array(np.array([[1.0, 2.0]], dtype=np.float32)) * 2.0).realize()
            parent.close()
    finally:
        set_engine_timing(prev)

    np.testing.assert_allclose(out, [[2.0, 4.0]])
    assert [c.name for c in parent.children] == ["graph_compute"]
    assert [c.name for c in parent.children[0].children] == ["mul_scalar (engine)"]


def test_graph_op_splits_engine_segments():
    from muraw.engines.core import _engine_load

    src = np.arange(16, dtype=np.float32).reshape(4, 4)
    x = Array(src) - 0.0
    x = _fence_cast(x, "float32")  # python fence between engine segments
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


def test_cfa_demosaic_op_lazy():
    from muraw.engines.pyops import cfa_demosaic_op

    rng = np.random.default_rng(4)
    cfa = rng.integers(0, 1000, size=(16, 16), dtype=np.uint16)
    out = cfa_demosaic_op(Array(cfa), "RGGB", "VNG").realize()
    ref = demosaic(Array(cfa), "RGGB", algorithm=DemosaicAlgorithm.VNG).realize()
    np.testing.assert_array_equal(out, ref)

    out_ea = cfa_demosaic_op(Array(cfa), "RGGB", "OPENCV_EA").realize()
    ref_ea = demosaic(
        Array(cfa), "RGGB", algorithm=DemosaicAlgorithm.OPENCV_EA
    ).realize()
    np.testing.assert_array_equal(out_ea, ref_ea)


def test_ingest_seals_view_and_base():
    parent = np.array(
        np.arange(16, dtype=np.float32).reshape(4, 4), copy=True
    )
    view = parent[1:3, 1:3]
    t = Array(view)
    assert view.base is parent
    assert not t._data.flags.writeable
    assert not view.flags.writeable
    assert not parent.flags.writeable
    with pytest.raises(ValueError):
        parent[0, 0] = 99.0


def test_ingest_strided_view_seals_array_and_base():
    """The object passed to Array, and its .base chain, become read-only.

    A sibling view created earlier is a separate NumPy object. Sealing
    does not walk sideways, so that sibling can stay writeable.
    """
    parent = np.array(np.arange(16, dtype=np.float32).reshape(4, 4), copy=True)
    sibling = parent[1:3, :]
    stepped = parent[::2, ::2]
    t = Array(stepped)
    assert t._node is not None and t._node.op == "view"
    assert not stepped.flags.writeable
    assert not parent.flags.writeable
    with pytest.raises(ValueError):
        stepped[0, 0] = 99.0
    with pytest.raises(ValueError):
        parent[0, 0] = 99.0
    np.testing.assert_array_equal(t.realize(), np.arange(16, dtype=np.float32).reshape(4, 4)[::2, ::2])
    assert sibling.flags.writeable


def test_realized_view_walks_upstream_for_canvas_crop():
    """A realized view's _data is the window; canvas pixels still need the graph."""
    src = np.arange(5 * 7, dtype=np.float32).reshape(5, 7)
    viewed = Array(src).view(left=1, top=2, width=3, height=2)
    viewed.realize()
    assert viewed._data is not None
    extra = viewed.crop(left=-1, top=0, width=5, height=2)
    np.testing.assert_array_equal(extra.realize(), src[2:4, 0:5])


def test_realized_crop_is_extra_bind(monkeypatch):
    """A hard crop's cache is an extra in_bind; the submitted graph still has the crop."""
    from muraw.engines.core import _engine_load

    calls: List[dict] = []
    real = _engine_load.execute_graph

    def wrap(graph, in_binds, out_binds, record_ops=False):
        calls.append({"graph": graph, "in_binds": dict(in_binds)})
        return real(graph, in_binds, out_binds, record_ops)

    monkeypatch.setattr(_engine_load, "execute_graph", wrap)

    src = np.arange(3 * 4, dtype=np.float32).reshape(3, 4)
    cropped = Array(src).crop(left=1, top=1, width=2, height=2)
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
        x = Array(np.ones((2, 2), dtype=np.float32)) - 0.0
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


def _upstream_op_arrays(root: Array) -> List[Array]:
    """Every op result ``root`` reads, directly or through other ops."""
    found: List[Array] = []
    seen: set[int] = set()
    stack = list(root._node.inputs) if root._node is not None else []
    while stack:
        array = stack.pop()
        if id(array) in seen or array._node is None:
            continue
        seen.add(id(array))
        found.append(array)
        stack.extend(array._node.inputs)
    return found


def _two_segments_through_a_python_op() -> Array:
    src = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)
    return _fence_cast(Array(src) - 1.0, "float32") * 2.0


def _two_segments_through_a_reshape() -> Array:
    src = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)
    return (Array(src) - 1.0)[None, ...] * 2.0


def _one_segment_chain() -> Array:
    src = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)
    return ((Array(src) - 1.0) * 2.0) - 3.0


def _ramp_reshaped_and_broadcast() -> Array:
    ramp = mi.linspace(np.zeros(3, np.float32), np.ones(3, np.float32), 5) * 0.5
    return mi.broadcast_to(ramp[None, ...], (4, 5, 3))


@pytest.mark.parametrize(
    "build",
    [
        _two_segments_through_a_python_op,
        _two_segments_through_a_reshape,
        _one_segment_chain,
        _ramp_reshaped_and_broadcast,
    ],
)
@pytest.mark.parametrize("force_recompute", [False, True])
def test_realize_keeps_pixels_only_on_the_realized_array(build, force_recompute):
    """realize() must not leave buffers on intermediate arrays: an
    intermediate that is kept alive would hold its full buffer forever."""
    root = build()
    upstream = _upstream_op_arrays(root)
    assert upstream
    root.realize(force_recompute=force_recompute)
    assert root._data is not None
    holding = [array._node.op for array in upstream if array._data is not None]
    assert holding == []


def test_realize_reuses_an_array_the_caller_realized_and_adds_no_others():
    src = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)
    pinned = _fence_cast(Array(src) - 1.0, "float32")
    pinned_pixels = pinned.realize()
    root = (pinned * 2.0) - 3.0
    out = root.realize()
    np.testing.assert_array_equal(out, (src - 1.0) * 2.0 - 3.0)
    assert pinned._data is pinned_pixels
    others = [
        array._node.op
        for array in _upstream_op_arrays(root)
        if array is not pinned and array._data is not None
    ]
    assert others == []


def test_op_node_is_frozen():
    x = Array(np.ones((2, 2), dtype=np.float32)) - 1.0
    assert x._node is not None
    with pytest.raises(AttributeError):
        x._node.op = "mul_scalar"
    with pytest.raises(TypeError):
        x._node.attrs["value"] = 0.0


def test_strided_numpy_crop_ported_and_image_op():
    parent = np.arange(16, dtype=np.float32).reshape(4, 4)
    crop = parent[1:3, 1:3]
    got = (Array(crop) - 1.0).realize()
    np.testing.assert_array_equal(got, crop - 1.0)

    identity = mi.apply_flat_gain_map(
        Array(crop),
        gain_map=[1.0, 1.0, 1.0, 1.0],
        gain_h=2,
        gain_w=2,
    ).realize()
    np.testing.assert_array_equal(identity, crop)


def _op_chain(t: Array) -> list[str]:
    """Op names from ``t`` back to its source, following input 0."""
    names = []
    while t._node is not None:
        names.append(t._node.op)
        t = t._node.inputs[0]
    return names


def _source_buffer(t: Array) -> np.ndarray:
    """The bound buffer at the end of ``t``'s input-0 chain."""
    while t._node is not None:
        t = t._node.inputs[0]
    return t._data


def test_fortran_array_ingested_as_transpose():
    arr = np.asfortranarray(np.arange(4 * 6, dtype=np.float32).reshape(4, 6))
    t = Array(arr)
    assert _op_chain(t) == ["orientation"]
    assert np.shares_memory(_source_buffer(t), arr)
    np.testing.assert_array_equal(t.realize(), arr)
    np.testing.assert_array_equal((t - 1.0).realize(), arr - 1.0)


_MONO_4X6 = np.arange(4 * 6, dtype=np.float32).reshape(4, 6)
_RGB_4X6 = np.arange(4 * 6 * 3, dtype=np.uint8).reshape(4, 6, 3)


@pytest.mark.parametrize(
    ("arr", "chain"),
    [
        (_MONO_4X6.T, ["orientation"]),
        (_RGB_4X6.transpose(1, 0, 2), ["orientation"]),
        (np.swapaxes(_RGB_4X6, 0, 1), ["orientation"]),
        (np.rot90(_MONO_4X6), ["orientation"]),
        (np.rot90(_MONO_4X6, 3), ["orientation"]),
        (np.rot90(_RGB_4X6), ["orientation"]),
        (_RGB_4X6[:, :, 1].T, ["orientation", "view"]),
        (_RGB_4X6[::2, 1:].transpose(1, 0, 2), ["orientation"]),
    ],
    ids=[
        "mono_T",
        "rgb_transpose",
        "swapaxes",
        "rot90",
        "rot90_k3",
        "rot90_rgb",
        "channel_T",
        "stepped_transpose",
    ],
)
def test_transposed_view_ingested_without_copy(arr, chain):
    t = Array(arr)
    assert _op_chain(t) == chain
    owner = arr.base if arr.base is not None else arr
    assert np.shares_memory(_source_buffer(t), owner)
    assert t.shape == arr.shape
    np.testing.assert_array_equal(t.realize(), arr)
    np.testing.assert_array_equal((t * 2.0).realize(), arr.astype(np.float32) * 2.0)


def test_transposed_ingest_matches_array_transpose():
    assert _op_chain(Array(_MONO_4X6.T)) == _op_chain(Array(_MONO_4X6).T)
    np.testing.assert_array_equal(Array(_MONO_4X6.T).realize(), Array(_MONO_4X6).T.realize())


@pytest.mark.parametrize(
    ("arr", "code"),
    [
        (_MONO_4X6.T, 5),
        (np.rot90(_MONO_4X6, 3), 6),
        (_MONO_4X6.T[::-1, ::-1], 7),
        (np.rot90(_MONO_4X6), 8),
        (np.rot90(_RGB_4X6[::2, 1:]), 8),
    ],
    ids=["transpose", "rot90_k3", "transverse", "rot90", "rot90_of_stepped_slice"],
)
def test_transposed_ingest_folds_flips_into_one_orientation(arr, code):
    t = Array(arr)
    assert t._node.op == "orientation"
    assert t._node.attrs["orientation"] == code
    assert t._node.inputs[0]._node is None
    np.testing.assert_array_equal(t.realize(), arr)


@pytest.mark.parametrize("k", [1, 3])
def test_rot90_ingest_builds_the_same_graph_as_array_rot90(k):
    ingested = Array(np.rot90(_RGB_4X6, k))
    rotated = rot90(Array(_RGB_4X6), k)
    assert ingested._node.op == rotated._node.op == "orientation"
    assert ingested._node.attrs["orientation"] == rotated._node.attrs["orientation"]
    assert ingested._node.inputs[0]._data is not None
    np.testing.assert_array_equal(ingested.realize(), rotated.realize())


def test_rot90_of_padded_camera_frame_not_copied():
    frame, raw = _padded_xrgb_frame(5, 7)
    for k in (1, 3):
        rotated = np.rot90(frame[..., :3], k)
        t = Array(rotated)
        assert _op_chain(t)[0] == "orientation"
        assert np.shares_memory(_source_buffer(t), raw)
        np.testing.assert_array_equal(t.realize(), rotated)


def test_one_row_and_one_column_transposes_are_not_transposed():
    row = np.arange(6, dtype=np.float32)[None, :]
    col = np.arange(6, dtype=np.float32)[:, None]
    for arr in (row.T, col.T):
        t = Array(arr)
        assert "orientation" not in _op_chain(t)
        np.testing.assert_array_equal(t.realize(), arr)


def test_stepped_slice_installed_as_view():
    parent = np.arange(16, dtype=np.float32).reshape(4, 4)
    stepped = parent[::2, ::2]
    t = Array(stepped)
    assert t._data is None
    assert t._node is not None and t._node.op == "view"
    assert np.shares_memory(t._node.inputs[0]._data, parent)
    assert t.meta.origin == (0, 0)
    assert t.meta.canvas == (0, 0, stepped.shape[1], stepped.shape[0])
    np.testing.assert_array_equal(t.realize(), stepped)
    np.testing.assert_array_equal((t - 1.0).realize(), stepped - 1.0)


def test_row_step_stays_a_source_buffer():
    parent = np.arange(16, dtype=np.float32).reshape(4, 4)
    rows = parent[::2, :]
    t = Array(rows)
    assert t._node is None
    assert np.shares_memory(t._data, parent)
    np.testing.assert_array_equal(t.realize(), rows)


def test_reversed_slice_installed_as_view():
    parent = np.arange(20, dtype=np.float32).reshape(4, 5)
    flipped = parent[::-1, ::-1]
    t = Array(flipped)
    assert t._data is None
    assert t._node is not None and t._node.op == "view"
    assert np.shares_memory(t._node.inputs[0]._data, parent)
    np.testing.assert_array_equal(t.realize(), flipped)


def test_channel_slice_and_pad_installed_as_view():
    rgb = np.arange(4 * 6 * 4, dtype=np.uint8).reshape(4, 6, 4)
    gathered = rgb[:, :, :3]
    t = Array(gathered)
    assert t._data is None
    assert t._node is not None and t._node.op == "view"
    assert np.shares_memory(t._node.inputs[0]._data, rgb)
    np.testing.assert_array_equal(t.realize(), gathered)

    reversed_channels = rgb[:, :, ::-1]
    rev = Array(reversed_channels)
    assert rev._node is not None and rev._node.op == "view"
    assert np.shares_memory(rev._node.inputs[0]._data, rgb)
    np.testing.assert_array_equal(rev.realize(), reversed_channels)

    one = Array(rgb[:, :, 1])
    assert one.shape == (4, 6)
    assert one._node is not None and one._node.op == "view"
    np.testing.assert_array_equal(one.realize(), rgb[:, :, 1])

    kept = Array(rgb[:, :, 0:1])
    assert kept.shape == (4, 6, 1)
    np.testing.assert_array_equal(kept.realize(), rgb[:, :, 0:1])

    planes = rgb[::2, 1::2, ::-1]
    mixed = Array(planes)
    assert mixed._node is not None and mixed._node.op == "view"
    assert np.shares_memory(mixed._node.inputs[0]._data, rgb)
    np.testing.assert_array_equal(mixed.realize(), planes)

    mono = np.arange(4 * 5, dtype=np.float32).reshape(4, 5)
    lifted = Array(mono[:, :, None])
    assert lifted.shape == (4, 5, 1)
    assert np.shares_memory(lifted.realize(), mono)


def test_padded_pixel_without_channel_axis_installed_as_view():
    height, width = 3, 5
    raw = np.arange(height * width * 4, dtype=np.uint8)
    view = np.ndarray(
        shape=(height, width, 3),
        dtype=np.uint8,
        buffer=raw,
        strides=(width * 4, 4, 1),
    )
    t = Array(view)
    assert t._data is None
    assert t._node is not None and t._node.op == "view"
    assert np.shares_memory(t._node.inputs[0]._data, raw)
    np.testing.assert_array_equal(t.realize(), view)


def test_broadcast_ingests_as_tile_and_odd_pad_still_copies():
    row = np.arange(8, dtype=np.float32)
    broadcast = np.broadcast_to(row, (4, 8))
    t = Array(broadcast)
    assert t._node is not None and t._node.op == "tile"
    assert np.shares_memory(t._node.inputs[0]._data, broadcast)
    np.testing.assert_array_equal(t.realize(), broadcast)

    raw = np.arange(2 * 2 * 7, dtype=np.uint8)
    odd = np.ndarray(
        shape=(2, 2, 3),
        dtype="<u2",
        buffer=raw,
        strides=(14, 7, 2),
    )
    copied = Array(odd)
    assert copied._node is None
    assert not np.shares_memory(copied._data, odd)
    np.testing.assert_array_equal(copied.realize(), odd)


def _padded_xrgb_frame(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """An XRGB8888 frame whose rows are 16 bytes longer than its pixels, like a
    camera buffer. Returns the frame and the flat buffer it reads."""
    pitch = width * 4 + 16
    raw = (np.arange(height * pitch) % 251).astype(np.uint8)
    frame = np.ndarray(
        shape=(height, width, 4), dtype=np.uint8, buffer=raw, strides=(pitch, 4, 1)
    )
    return frame, raw


@pytest.mark.parametrize(
    "key",
    [
        np.s_[..., :3],
        np.s_[..., 2::-1],
        np.s_[..., 1],
        np.s_[:, ::2],
        np.s_[::-1, :, :3],
        np.s_[1::2, ::-3, 1:3],
    ],
    ids=["rgb", "bgr", "one_channel", "column_step", "row_flip_rgb", "mixed"],
)
def test_padded_camera_frame_installed_as_view(key):
    frame, raw = _padded_xrgb_frame(5, 7)
    view = frame[key]
    t = Array(view)
    assert t._node is not None and t._node.op == "view"
    assert np.shares_memory(t._node.inputs[0]._data, raw)
    np.testing.assert_array_equal(t.realize(), view)


_CHW = np.arange(3 * 4 * 5, dtype=np.float32).reshape(3, 4, 5)


@pytest.mark.parametrize(
    "arr, axes",
    [
        (np.moveaxis(_CHW, 0, -1), [1, 2, 0]),
        (np.moveaxis(_CHW, 0, -1).transpose(1, 0, 2), [2, 1, 0]),
        (np.asfortranarray(np.moveaxis(_CHW, 0, -1)), [2, 1, 0]),
    ],
    ids=["moveaxis", "moveaxis_transposed", "fortran"],
)
def test_planar_layouts_bound_on_ingest(arr, axes):
    t = Array(arr)
    assert _op_chain(t) == ["transpose"]
    assert list(t._node.attrs["axes"]) == axes
    bound = t._node.inputs[0]
    assert bound.meta.channel_axis == 0
    assert np.shares_memory(bound._data, arr)
    out = (t * 2.0).realize()
    assert out.flags.c_contiguous
    np.testing.assert_array_equal(out, arr * 2.0)


@pytest.mark.parametrize("shape", [(4,), (1, 4), (3, 4)])
def test_misaligned_buffer_copied_on_ingest(shape):
    raw = (np.arange(64) % 200).astype(np.uint8)
    count = int(np.prod(shape))
    arr = np.frombuffer(raw, dtype="<u2", count=count, offset=1).reshape(shape)
    assert arr.ctypes.data % arr.dtype.itemsize != 0
    t = Array(arr)
    assert t._node is None
    assert t._data.ctypes.data % t._data.dtype.itemsize == 0
    assert not np.shares_memory(t._data, raw)
    np.testing.assert_array_equal(t.realize(), arr)


def _random_axis_slice(rng: np.random.Generator, length: int) -> slice:
    """A non-empty slice on an axis of ``length``."""
    for _ in range(32):
        step = int(rng.choice([-3, -2, -1, 1, 2, 3]))
        start = int(rng.integers(-length, length))
        stop = int(rng.integers(-length, length + 1))
        slc = slice(start, stop, step)
        if len(range(*slc.indices(length))) >= 1:
            return slc
    return slice(None)


def test_ingest_random_views_match_numpy():
    """Packed-parent slices ingest as a view or a bound buffer, never a wrong copy."""
    rng = np.random.default_rng(20260923)
    mono = np.arange(16 * 20, dtype=np.float32).reshape(16, 20)
    rgb = np.arange(12 * 15 * 4, dtype=np.uint8).reshape(12, 15, 4)
    for _ in range(25):
        view = mono[_random_axis_slice(rng, 16), _random_axis_slice(rng, 20)]
        if rng.random() < 0.25:
            view = view[:, :, None]
        t = Array(view)
        np.testing.assert_array_equal(t.realize(), view)
        src = t._data if t._node is None else t._node.inputs[0]._data
        assert np.shares_memory(src, mono)
    for _ in range(25):
        rows = _random_axis_slice(rng, 12)
        cols = _random_axis_slice(rng, 15)
        choice = int(rng.integers(0, 4))
        if choice == 0:
            view = rgb[rows, cols]
        elif choice == 1:
            view = rgb[rows, cols, ::-1]
        elif choice == 2:
            view = rgb[rows, cols, :3]
        else:
            view = rgb[rows, cols, int(rng.integers(0, 4))]
        t = Array(view)
        np.testing.assert_array_equal(t.realize(), view)
        src = t._data if t._node is None else t._node.inputs[0]._data
        assert np.shares_memory(src, rgb)
    frame, raw = _padded_xrgb_frame(9, 11)
    for _ in range(25):
        rows = _random_axis_slice(rng, 9)
        cols = _random_axis_slice(rng, 11)
        channels = _random_axis_slice(rng, 4)
        view = frame[rows, cols, channels]
        t = Array(view)
        np.testing.assert_array_equal(t.realize(), view)
        src = t._data if t._node is None else t._node.inputs[0]._data
        assert np.shares_memory(src, raw)


def test_ingest_random_transposed_views_are_not_copied():
    """The transpose of a packed-parent slice reads the parent, never a copy."""
    rng = np.random.default_rng(20260929)
    mono = np.arange(16 * 20, dtype=np.float32).reshape(16, 20)
    rgb = np.arange(12 * 15 * 4, dtype=np.uint8).reshape(12, 15, 4)
    frame, raw = _padded_xrgb_frame(9, 11)
    for _ in range(25):
        slices = [
            (mono[_random_axis_slice(rng, 16), _random_axis_slice(rng, 20)], mono),
            (rgb[_random_axis_slice(rng, 12), _random_axis_slice(rng, 15), :3], rgb),
            (
                frame[
                    _random_axis_slice(rng, 9),
                    _random_axis_slice(rng, 11),
                    _random_axis_slice(rng, 4),
                ],
                raw,
            ),
        ]
        for view, owner in slices:
            transposed = view.swapaxes(0, 1)
            t = Array(transposed)
            np.testing.assert_array_equal(t.realize(), transposed)
            assert np.shares_memory(_source_buffer(t), owner)


def _padded_rgb888_frame(height: int, width: int, pitch: int) -> tuple[np.ndarray, np.ndarray]:
    """A packed RGB888 frame whose rows are ``pitch`` bytes apart, like a camera
    buffer whose row alignment is not a multiple of 3. Returns the frame and
    the flat buffer it reads."""
    raw = (np.arange(height * pitch) % 251).astype(np.uint8)
    frame = np.ndarray(shape=(height, width, 3), dtype=np.uint8, buffer=raw, strides=(pitch, 3, 1))
    return frame, raw


@pytest.mark.parametrize(
    "key",
    [np.s_[:], np.s_[1:3, 2:9], np.s_[:, ::2], np.s_[::-1], np.s_[..., ::-1], np.s_[..., 1]],
    ids=["whole", "crop", "column_step", "row_flip", "bgr", "one_channel"],
)
def test_padded_rgb888_frame_is_not_copied(key):
    frame, raw = _padded_rgb888_frame(4, 1000, 3008)
    view = frame[key]
    t = Array(view)
    assert np.shares_memory(_source_buffer(t), raw)
    np.testing.assert_array_equal(t.realize(), view)
    np.testing.assert_array_equal(
        (t.astype("float32") * 0.5).realize(), view.astype(np.float32) * 0.5
    )


def test_padded_rgb888_frame_is_bound_directly():
    frame, raw = _padded_rgb888_frame(4, 1000, 3008)
    t = Array(frame)
    assert t._node is None
    assert np.shares_memory(t._data, raw)


def test_transposed_padded_rgb888_frame_is_not_copied():
    frame, raw = _padded_rgb888_frame(4, 10, 32)
    for arr in (frame.transpose(1, 0, 2), np.rot90(frame)):
        t = Array(arr)
        assert t._node.op == "orientation"
        assert np.shares_memory(_source_buffer(t), raw)
        np.testing.assert_array_equal(t.realize(), arr)


def test_xrgb_frame_with_odd_pitch_reads_rgb_as_view():
    height, width, pitch = 4, 1000, 4010
    raw = (np.arange(height * pitch) % 251).astype(np.uint8)
    frame = np.ndarray(shape=(height, width, 4), dtype=np.uint8, buffer=raw, strides=(pitch, 4, 1))
    rgb = frame[..., :3]
    t = Array(rgb)
    assert _op_chain(t) == ["view"]
    assert np.shares_memory(_source_buffer(t), raw)
    np.testing.assert_array_equal(t.realize(), rgb)


def test_padded_uint16_rgb_is_bound_directly():
    height, width, pitch_elements = 3, 10, 31
    raw = np.arange(height * pitch_elements, dtype=np.uint16)
    frame = np.ndarray(
        shape=(height, width, 3), dtype=np.uint16, buffer=raw, strides=(pitch_elements * 2, 6, 2)
    )
    t = Array(frame)
    assert t._node is None
    assert np.shares_memory(t._data, raw)
    np.testing.assert_array_equal((t * 1.0).realize(), frame.astype(np.float32))


def _random_layout(rng: np.random.Generator) -> np.ndarray:
    """A random slice of a layout that is sometimes a view and sometimes copied."""
    dtype = rng.choice([np.uint8, np.uint16, np.float32])
    height, width, channels = (int(value) for value in rng.integers(1, 7, 3))
    key = (
        _random_axis_slice(rng, height),
        _random_axis_slice(rng, width),
        _random_axis_slice(rng, channels),
    )
    samples = (np.arange(height * width * channels * 2 + 8) % 200).astype(dtype)
    kind = int(rng.integers(0, 4))
    if kind == 0:
        # Row padding that is not a whole number of pixels.
        pitch = width * channels + 1
        frame = np.ndarray(
            shape=(height, width, channels),
            dtype=dtype,
            buffer=samples,
            strides=(pitch * samples.itemsize, channels * samples.itemsize, samples.itemsize),
        )
        return frame[key]
    image = samples[: height * width * channels]
    if kind == 1:
        return np.moveaxis(image.reshape(channels, height, width), 0, -1)[key]
    if kind == 2:
        return image.reshape(width, height, channels).transpose(1, 0, 2)[key]
    row = image[: width * channels].reshape(1, width, channels)
    return np.broadcast_to(row, (height, width, channels))[key]


def test_ingest_random_layouts_match_numpy():
    """Every layout matches NumPy, and anything bound directly is aligned and packed."""
    rng = np.random.default_rng(20260928)
    for _ in range(200):
        arr = _random_layout(rng)
        t = Array(arr)
        np.testing.assert_array_equal(t.realize(), arr)
        bound = t
        while bound._node is not None:
            bound = bound._node.inputs[0]
        assert bound._data is not None
        assert bound._data.ctypes.data % bound._data.dtype.itemsize == 0


@pytest.mark.parametrize(
    "dtype, message",
    [
        (np.float64, r"float64 is not supported.*arr\.astype\(np\.float32\)"),
        (">f8", r"float64 is not supported.*arr\.astype\(np\.float32\)"),
        (">u2", r"big-endian >u2 is not supported.*newbyteorder"),
        (">f4", r"big-endian >f4 is not supported.*newbyteorder"),
        (np.int32, r"dtype int32 is not supported\. Supported dtypes: float32, float16, uint8, uint16"),
        (np.bool_, r"dtype bool is not supported\. Supported dtypes"),
    ],
    ids=["float64", "float64_big_endian", "uint16_big_endian", "float32_big_endian", "int32", "bool"],
)
def test_unsupported_dtype_ingest_names_the_fix(dtype, message):
    with pytest.raises(ValueError, match=message):
        Array(np.zeros((2, 3, 3), dtype=dtype))


def test_suggested_conversion_ingests():
    for dtype in (">u2", ">f4"):
        arr = np.arange(18, dtype=dtype).reshape(2, 3, 3)
        native = arr.astype(arr.dtype.newbyteorder("="))
        np.testing.assert_array_equal(Array(native).realize(), arr)
    arr = np.linspace(0.0, 1.0, 18).reshape(2, 3, 3)
    np.testing.assert_array_equal(Array(arr.astype(np.float32)).realize(), arr.astype(np.float32))


def test_unsupported_dtype_argument_names_the_fix():
    with pytest.raises(ValueError, match="float64 is not supported"):
        mi.zeros((2, 2), dtype="float64")
    with pytest.raises(ValueError, match="float64 is not supported"):
        Array(np.zeros((2, 2), np.float32)).astype(np.float64)
    with pytest.raises(ValueError, match="dtype int64 is not supported"):
        mi.full((2, 2), 1, dtype=int)
    with pytest.raises(ValueError, match="'rgb8' is not a dtype"):
        mi.zeros((2, 2), dtype="rgb8")


def test_dtype_strings_and_types_map_to_element_type():
    assert ElementType("f4") is ElementType.FLOAT32
    assert ElementType(np.uint16) is ElementType.UINT16
    assert ElementType(np.dtype("<f2")) is ElementType.FLOAT16
