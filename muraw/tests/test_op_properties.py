"""The requires_2d / is_cfa / is_rgb op properties and the rgb_ / cfa_ naming rule."""

from __future__ import annotations

import numpy as np
import pytest

import muimage as mi
from muraw.array import Array
from muraw.engines.graph import graph_op
from muraw.engines.ops import OPS_BY_NAME
from muraw.engines.pyops import cfa_demosaic_op, radial_distortion_op, rgb_channel_luts_op
from muraw.engines.tools.gen_ops import validate_ops


def _catalog_op(name, channels, **props):
    return {
        "name": name,
        "inputs": [{"channels": channels}],
        "outputs": [{"channels": "same"}],
        **props,
    }


@pytest.mark.parametrize(
    ("op", "message"),
    [
        (_catalog_op("cfa_thing", 1, requires_2d=True), "named cfa_"),
        (_catalog_op("thing", 1, is_cfa=True, requires_2d=True), "named cfa_"),
        (_catalog_op("rgb_thing", 3), "named rgb_"),
        (_catalog_op("thing", 3, is_rgb=True), "named rgb_"),
        (_catalog_op("cfa_thing", 1, is_cfa=True), "requires requires_2d"),
        (_catalog_op("cfa_thing", "any", is_cfa=True, requires_2d=True), "channels: 1"),
        (_catalog_op("rgb_thing", 1, is_rgb=True), "channels: 3"),
        (_catalog_op("rgb_thing", 3, is_rgb=True, is_cfa=True, requires_2d=True), "both true"),
    ],
)
def test_gen_ops_rejects_property_violations(op, message):
    with pytest.raises(ValueError, match=message):
        validate_ops([op])


def test_gen_ops_accepts_consistent_properties():
    validate_ops(
        [
            _catalog_op("cfa_thing", 1, is_cfa=True, requires_2d=True),
            _catalog_op("rgb_thing", 3, is_rgb=True),
            _catalog_op("warp_thing", "any", requires_2d=True),
        ]
    )


@pytest.mark.parametrize(
    ("name", "requires_2d", "is_cfa", "is_rgb"),
    [
        ("cfa_ea_demosaic", True, True, False),
        ("cfa_normalize_raw", True, True, False),
        ("normalize_raw", True, False, False),
        ("rgb_transform", False, False, True),
        ("warp_rectilinear", True, False, False),
        ("apply_flat_gain_map", True, False, False),
        ("lut", False, False, False),
    ],
)
def test_catalog_op_meta_reports_properties(name, requires_2d, is_cfa, is_rgb):
    meta = OPS_BY_NAME[name].meta
    assert (meta.requires_2d, meta.is_cfa, meta.is_rgb) == (requires_2d, is_cfa, is_rgb)


def test_every_catalog_op_follows_the_naming_rule():
    for name, op in OPS_BY_NAME.items():
        meta = op.meta
        assert meta.is_cfa == name.startswith("cfa_"), name
        assert meta.is_rgb == name.startswith("rgb_"), name
        assert not meta.is_cfa or meta.requires_2d, name


def test_python_ops_report_properties():
    assert (cfa_demosaic_op.meta.is_cfa, cfa_demosaic_op.meta.requires_2d) == (True, True)
    assert rgb_channel_luts_op.meta.is_rgb
    assert radial_distortion_op.meta.requires_2d
    assert not radial_distortion_op.meta.is_rgb


def test_graph_op_rgb_checks_channels_when_called():
    @graph_op(is_rgb=True)
    def rgb_identity(arr):
        return arr

    with pytest.raises(ValueError, match="expected 3 channel"):
        rgb_identity(Array(np.zeros((2, 2), dtype=np.float32)))
    out = rgb_identity(Array(np.ones((2, 2, 3), dtype=np.float32))).realize()
    np.testing.assert_array_equal(out, np.ones((2, 2, 3), dtype=np.float32))

def _normalize_attrs(channels):
    return dict(
        black_level=np.full(channels, 100.0, dtype=np.float32),
        black_repeat_rows=1,
        black_repeat_cols=1,
        white_level=np.full(channels, 1100.0, dtype=np.float32),
    )


@pytest.mark.parametrize(
    "src",
    [
        np.array([[[100, 600, 1100], [350, 850, 1600]]], dtype=np.uint16),
        np.array([[100, 600], [1100, 1600]], dtype=np.uint16),
    ],
    ids=["rgb", "mono"],
)
def test_normalize_raw_matches_numpy(src):
    channels = src.shape[2] if src.ndim == 3 else 1
    out = mi.normalize_raw(Array(src), **_normalize_attrs(channels)).realize()
    expected = np.clip((src.astype(np.float32) - 100.0) / 1000.0, 0.0, 1.0)
    np.testing.assert_allclose(np.asarray(out).reshape(src.shape), expected, atol=1e-6)


def test_cfa_normalize_raw_rejects_three_channels():
    with pytest.raises(ValueError, match="expected 1 channel"):
        mi.cfa_normalize_raw(
            Array(np.zeros((2, 2, 3), dtype=np.uint16)), **_normalize_attrs(1)
        )
