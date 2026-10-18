"""``mi.rgb_apply_gain_table_map`` against a NumPy port of its formula.

NumPy has no such function, so each case compares ``mi`` with
``_reference``. The op computes in float32 and the reference in double.
The weight's rounding is magnified by a gamma below 1 near 0 and by the
number of table points, so the tolerance is 1e-5, scaled by
table_points / 32 for tables of more than 32 points.
"""

import numpy as np
import pytest

import muimage as mi

_WEIGHTS = np.array([0.3, 0.4, 0.2, 0.05, 0.1], np.float32)


def _table(points_y, points_x, table_points):
    """Gains from 0.5 to 1.7 that change every sample."""
    count = points_y * points_x * table_points
    gains = 0.5 + (np.arange(count) * 37 % 13) / 10.0
    return gains.reshape(points_y, points_x, table_points).astype(np.float32)


def _image(shape, seed=3):
    """RGB from -0.25 to 1.25, so every clamp changes some samples."""
    rng = np.random.default_rng(seed)
    return (rng.random(shape) * 1.5 - 0.25).astype(np.float32)


def _reference(
    src,
    table,
    *,
    origin_x,
    origin_y,
    spacing_x,
    spacing_y,
    weights,
    gamma,
    exposure_weight_gain,
    clamp="none",
):
    """The gain table interpolated at each pixel's grid position and
    weight, times the pixel; ``src`` is the whole canvas."""
    rows, cols, _ = src.shape
    points_y, points_x, table_points = table.shape
    rgb = src.astype(np.float64)
    table = table.astype(np.float64)

    def split(position, points):
        index = np.clip(position, 0, points - 1)
        lower = index.astype(np.int64)
        return lower, index - lower, np.minimum(lower + 1, points - 1)

    u = (np.arange(cols) + 0.5) / cols
    v = (np.arange(rows) + 0.5) / rows
    x0, xf, x1 = split((u - origin_x) / spacing_x, points_x)
    y0, yf, y1 = split((v - origin_y) / spacing_y, points_y)
    x0, xf, x1 = x0[None, :], xf[None, :], x1[None, :]
    y0, yf, y1 = y0[:, None], yf[:, None], y1[:, None]
    terms = np.stack(
        [rgb[..., 0], rgb[..., 1], rgb[..., 2], rgb.min(axis=-1), rgb.max(axis=-1)],
        axis=-1,
    )
    weight = np.clip(
        (terms @ np.asarray(weights, np.float64)) * exposure_weight_gain, 0, 1
    )
    if gamma != 1:
        weight = weight**gamma
    scaled = weight * table_points
    w0 = np.minimum(scaled.astype(np.int64), table_points - 1)
    w1 = np.minimum(w0 + 1, table_points - 1)
    wf = scaled - w0

    def lerp(a, b, t):
        return a + (b - a) * t

    def along_w(y, x):
        return lerp(table[y, x, w0], table[y, x, w1], wf)

    top = lerp(along_w(y0, x0), along_w(y0, x1), xf)
    bottom = lerp(along_w(y1, x0), along_w(y1, x1), xf)
    out = rgb * lerp(top, bottom, yf)[..., None]
    if clamp == "max_one":
        out = np.minimum(out, 1)
    elif clamp == "zero_one":
        out = np.clip(out, 0, 1)
    return out.astype(np.float32)


def _covering(points_y, points_x):
    """Grid attrs that spread the table from fraction 0 to 1 of the canvas."""
    return {
        "origin_x": 0.0,
        "origin_y": 0.0,
        "spacing_x": 1.0 / max(points_x - 1, 1),
        "spacing_y": 1.0 / max(points_y - 1, 1),
    }


_TABLES = {
    "covering": ((5, 7, 9), _covering(5, 7)),
    "offset": (
        (4, 5, 7),
        {"origin_x": -0.05, "origin_y": 0.1, "spacing_x": 0.15, "spacing_y": 0.2},
    ),
    "one_point": ((1, 1, 1), _covering(1, 1)),
    "one_row": ((1, 4, 5), _covering(1, 4)),
    "many_table_points": ((6, 2, 257), _covering(6, 2)),
    "over_blend_limit": ((3, 4, 1100), _covering(3, 4)),
}


@pytest.mark.parametrize("clamp", ["none", "max_one", "zero_one"])
@pytest.mark.parametrize("gamma", [1.0, 0.6, 2.2])
@pytest.mark.parametrize("table_name", list(_TABLES))
def test_rgb_apply_gain_table_map_matches_reference(table_name, gamma, clamp):
    points, grid = _TABLES[table_name]
    src = _image((40, 57, 3))
    table = _table(*points)
    attrs = dict(
        grid,
        weights=_WEIGHTS,
        gamma=gamma,
        exposure_weight_gain=1.41,
        clamp=clamp,
    )
    got = mi.rgb_apply_gain_table_map(mi.Array(src), mi.Array(table), **attrs).realize()
    assert got.dtype == np.float32 and got.shape == src.shape
    tolerance = 1e-5 * max(1.0, points[2] / 32)
    np.testing.assert_allclose(
        got, _reference(src, table, **attrs), rtol=tolerance, atol=tolerance
    )


@pytest.mark.parametrize("gamma", [1.0, 0.6])
@pytest.mark.parametrize("table_name", list(_TABLES))
def test_rgb_apply_gain_table_map_neon_matches_scalar(table_name, gamma):
    """``engine_opts {"core": {"neon": false}}`` forces the scalar loop,
    which gives the same bits as the NEON loop. The image is wide enough
    that each grid cell holds several groups of four pixels."""
    points, grid = _TABLES[table_name]
    src = mi.Array(_image((12, 301, 3)))
    table = mi.Array(_table(*points))
    attrs = dict(
        grid, weights=_WEIGHTS, gamma=gamma, exposure_weight_gain=1.41, clamp="zero_one"
    )
    neon = mi.rgb_apply_gain_table_map(src, table, **attrs).realize()
    scalar = mi.rgb_apply_gain_table_map(
        src, table, **attrs, engine_opts='{"core": {"neon": false}}'
    ).realize()
    np.testing.assert_array_equal(neon, scalar)


_ATTRS = dict(
    _TABLES["offset"][1],
    weights=_WEIGHTS,
    gamma=0.6,
    exposure_weight_gain=1.2,
    clamp="zero_one",
)


def test_rgb_apply_gain_table_map_on_a_slice_matches_the_slice_of_the_whole():
    src = mi.Array(_image((40, 57, 3)))
    table = mi.Array(_table(4, 5, 7))
    whole = mi.rgb_apply_gain_table_map(src, table, **_ATTRS).realize()
    before = mi.rgb_apply_gain_table_map(src[7:31, 12:50], table, **_ATTRS)
    np.testing.assert_array_equal(before.realize(), whole[7:31, 12:50])
    after = mi.rgb_apply_gain_table_map(src, table, **_ATTRS)[7:31, 12:50]
    np.testing.assert_array_equal(after.realize(), whole[7:31, 12:50])


def test_rgb_apply_gain_table_map_takes_a_table_computed_in_the_graph():
    src = _image((40, 57, 3))
    table = _table(4, 5, 7)
    from_graph = mi.Array(table * np.float32(0.5)) * np.float32(2.0)
    got = mi.rgb_apply_gain_table_map(mi.Array(src), from_graph, **_ATTRS).realize()
    expect = mi.rgb_apply_gain_table_map(mi.Array(src), mi.Array(table), **_ATTRS)
    np.testing.assert_array_equal(got, expect.realize())
