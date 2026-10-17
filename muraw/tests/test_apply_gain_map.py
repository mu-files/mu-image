"""``apply_gain_map`` (DNG GainMap) against a NumPy port of the gain map
formula.

NumPy has no such function, so each case compares ``mi`` with
``_reference``. The reference interpolates the map in double precision
and the kernel in float32, so the two agree to a few float32 ulps.
"""

import numpy as np
import pytest

import muimage as mi
from muraw.array import Array

_HEIGHT = 40
_WIDTH = 53


def _image(shape, low=0.0, high=1.0, seed=5):
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, shape).astype(np.float32)


def _gains(points_v, points_h, map_planes=None, low=0.6, high=1.8, seed=7):
    """A float32 gain table, (points_v, points_h) when ``map_planes`` is
    None."""
    shape = (points_v, points_h) + (() if map_planes is None else (map_planes,))
    return _image(shape, low, high, seed)


def _axis(coords, origin, spacing, points):
    """For each coordinate, the two map indices on either side of it and
    the fraction of the way from the first to the second. Past the first
    or last map sample the coordinate is held at that sample."""
    position = np.clip((coords - origin) / spacing, 0.0, points - 1)
    first = np.minimum(np.floor(position).astype(np.int64), max(points - 2, 0))
    second = np.minimum(first + 1, points - 1)
    return first, second, position - first


def _changed(length, start, end, pitch):
    """Which canvas rows or columns of ``length`` an area from ``start`` to
    ``end`` changes: those inside it and a whole number of ``pitch`` from
    ``start``."""
    index = np.arange(length)
    return (index >= start) & (index < end) & ((index - start) % pitch == 0)


def _reference(
    x,
    gains,
    *,
    map_x0,
    map_y0,
    map_spacing_x,
    map_spacing_y,
    top=None,
    left=None,
    bottom=None,
    right=None,
    start_plane=0,
    num_planes=None,
    row_pitch=1,
    col_pitch=1,
    clamp="none",
    max_gain=None,
):
    """DNG GainMap on an (H, W) or (H, W, C) array whose first pixel is
    canvas (0, 0). Plane p, a channel, reads map plane min(p, map_planes -
    1). An area of none, or with bottom <= top or right <= left, is the
    whole image. ``max_gain`` changes only how the kernel computes and is
    ignored here."""
    del max_gain
    height, width = x.shape[:2]
    channels = 1 if x.ndim == 2 else x.shape[2]
    table = np.asarray(gains, np.float64)
    if table.ndim == 2:
        table = table[..., None]
    points_v, points_h, map_planes = table.shape

    rows = np.arange(height, dtype=np.float64)
    cols = np.arange(width, dtype=np.float64)
    row0, row1, fy = _axis(rows, map_y0, map_spacing_y, points_v)
    col0, col1, fx = _axis(cols, map_x0, map_spacing_x, points_h)
    fy = fy[:, None]
    fx = fx[None, :]

    if top is None or bottom <= top or right <= left:
        top, left, bottom, right = 0, 0, height, width
    rows_changed = _changed(height, top, bottom, row_pitch)
    cols_changed = _changed(width, left, right, col_pitch)
    changed = rows_changed[:, None] & cols_changed[None, :]
    plane_end = channels if num_planes is None else start_plane + num_planes

    out = np.asarray(x, np.float64).reshape(height, width, channels).copy()
    for plane in range(max(start_plane, 0), min(plane_end, channels)):
        g = table[..., min(plane, map_planes - 1)]
        upper = g[row0][:, col0] * (1 - fx) + g[row0][:, col1] * fx
        lower = g[row1][:, col0] * (1 - fx) + g[row1][:, col1] * fx
        value = out[..., plane] * (upper * (1 - fy) + lower * fy)
        if clamp == "max_one":
            value = np.minimum(value, 1.0)
        elif clamp == "zero_one":
            value = np.clip(value, 0.0, 1.0)
        out[..., plane] = np.where(changed, value, out[..., plane])
    return out.reshape(x.shape)


def _assert_close(got, expect):
    """The kernel places each cell edge at a float32 canvas column, which
    near column 500 is off by up to 3e-5 pixels; times the gain's slope,
    about 0.1 per pixel here, a gain is off by a few 1e-6, also where it
    crosses 0."""
    np.testing.assert_allclose(got, expect, rtol=2e-6, atol=4e-6)


def _attrs(
    map_x0=-0.5, map_y0=-0.5, map_spacing_x=10.3, map_spacing_y=9.7, area=None, **kwargs
):
    """Default placement: cell edges at fractional columns and rows, so
    they fall inside spans, and a map that covers the image. ``area`` is
    (top, left, bottom, right)."""
    sides = {} if area is None else dict(zip(("top", "left", "bottom", "right"), area))
    return {
        "map_x0": map_x0,
        "map_y0": map_y0,
        "map_spacing_x": map_spacing_x,
        "map_spacing_y": map_spacing_y,
        **sides,
        **kwargs,
    }


def _apply(x, gains, attrs):
    """``mi.apply_gain_map`` with a table that is an ndarray or already an
    ``mi.Array``."""
    table = gains if isinstance(gains, Array) else Array(gains)
    return mi.apply_gain_map(x, table, **attrs)


def _one_op(src, gains, before=None, after=None, **attrs):
    """A case of one ``apply_gain_map`` with ``_attrs(**attrs)``."""
    return (src, [(gains, _attrs(**attrs))], before, after)


def _four_phases():
    """The CFA layout cameras write: one opcode per Bayer phase, each at
    row and column pitch 2 from an area offset by the phase, each with its
    own one-plane map."""
    return [
        (
            _gains(5, 6, seed=11 + 2 * row + col),
            _attrs(area=(row, col, _HEIGHT, _WIDTH), row_pitch=2, col_pitch=2),
        )
        for row in (0, 1)
        for col in (0, 1)
    ]


_RGB = _image((_HEIGHT, _WIDTH, 3))
_MONO = _image((_HEIGHT, _WIDTH))
_UNBOUNDED = _image((_HEIGHT, _WIDTH, 3), low=-0.5, high=1.5)
_SIGNED_GAINS = _gains(5, 6, 3, low=-0.5, high=2.5)
_CROP_BEFORE = (slice(3, None), slice(5, None))
_CROP_AFTER = (slice(4, 29), slice(9, 37))


def _channels(count):
    return _image((_HEIGHT, _WIDTH, count))


# Each case: the source, the (gains, attrs) of each apply_gain_map in
# order, a crop of the source before the ops, and a crop of the result
# after them. A crop keeps the source's canvas, so the map stays where it
# is on the source either way.
CASES = {
    "rgb_one_plane": _one_op(_RGB, _gains(5, 6, 1)),
    "rgb_two_dimensional_table": _one_op(_RGB, _gains(5, 6)),
    "rgb_three_planes": _one_op(_RGB, _gains(5, 6, 3)),
    "rgb_planes_1_and_2_one_map_plane": _one_op(
        _RGB, _gains(5, 6, 1), start_plane=1, num_planes=2
    ),
    "rgb_planes_1_and_2_three_map_planes": _one_op(
        _RGB, _gains(5, 6, 3), start_plane=1, num_planes=2
    ),
    "rgb_plane_2_two_map_planes": _one_op(
        _RGB, _gains(5, 6, 2), start_plane=2, num_planes=1
    ),
    "mono_2d": _one_op(_MONO, _gains(5, 6)),
    "one_channel": _one_op(_channels(1), _gains(5, 6, 1)),
    "two_channels_two_map_planes": _one_op(_channels(2), _gains(5, 6, 2)),
    "four_channels_four_map_planes": _one_op(_channels(4), _gains(5, 6, 4)),
    "four_channels_one_map_plane": _one_op(_channels(4), _gains(5, 6, 1)),
    "five_channels_five_map_planes": _one_op(_channels(5), _gains(5, 6, 5)),
    "five_channels_two_map_planes": _one_op(_channels(5), _gains(5, 6, 2)),
    "partial_area": _one_op(_RGB, _gains(5, 6, 3), area=(5, 7, 31, 44)),
    "rgb_row_pitch_2_col_pitch_3": _one_op(
        _RGB, _gains(5, 6, 3), area=(3, 5, 37, 51), row_pitch=2, col_pitch=3
    ),
    "mono_col_pitch_2": _one_op(
        _MONO, _gains(5, 6), area=(0, 1, _HEIGHT, _WIDTH), col_pitch=2
    ),
    "mono_col_pitch_3": _one_op(
        _MONO, _gains(5, 6), area=(2, 2, 33, 50), row_pitch=3, col_pitch=3
    ),
    "mono_col_pitch_4": _one_op(
        _MONO, _gains(5, 6), area=(1, 3, _HEIGHT, _WIDTH), col_pitch=4
    ),
    "mono_col_pitch_5": _one_op(
        _MONO, _gains(5, 6), area=(0, 4, _HEIGHT, _WIDTH), col_pitch=5
    ),
    "four_bayer_phases": (_MONO, _four_phases(), None, None),
    # The pitch counts from the area's corner, as the DNG SDK does, also
    # when the corner is before the image.
    "area_before_top_left_with_pitch": _one_op(
        _MONO, _gains(5, 6), area=(-3, -1, 30, 40), row_pitch=2, col_pitch=2
    ),
    "area_past_bottom_right": _one_op(
        _RGB, _gains(5, 6, 3), area=(10, 20, _HEIGHT + 7, _WIDTH + 9)
    ),
    "area_of_all_zeros_is_the_whole_image": _one_op(
        _RGB, _gains(5, 6, 3), area=(0, 0, 0, 0)
    ),
    "inverted_area_is_the_whole_image": _one_op(
        _RGB, _gains(5, 6, 3), area=(20, 30, 10, 5)
    ),
    "map_inside_the_image_holds_its_edges": _one_op(
        _RGB,
        _gains(3, 4, 3),
        map_x0=12.0,
        map_y0=9.5,
        map_spacing_x=6.0,
        map_spacing_y=5.0,
    ),
    "one_point_map": _one_op(_RGB, _gains(1, 1, 3)),
    "one_row_map": _one_op(_RGB, _gains(1, 6, 1)),
    "one_column_map": _one_op(_RGB, _gains(5, 1, 1)),
    # Integer spacing that is a multiple of 4 on a whole-column origin: the
    # kernel's loop over whole cells.
    "rgb_whole_cells": _one_op(
        _RGB,
        _gains(7, 9, 1),
        map_x0=-8.0,
        map_y0=-3.0,
        map_spacing_x=8.0,
        map_spacing_y=8.0,
    ),
    "mono_whole_cells": _one_op(
        _image((_HEIGHT, 97)),
        _gains(7, 9),
        map_x0=0.0,
        map_y0=0.0,
        map_spacing_x=12.0,
        map_spacing_y=8.0,
    ),
    "clamp_max_one": _one_op(_UNBOUNDED, _SIGNED_GAINS, clamp="max_one"),
    "clamp_zero_one": _one_op(_UNBOUNDED, _SIGNED_GAINS, clamp="zero_one"),
    "clamp_none": _one_op(_UNBOUNDED, _SIGNED_GAINS, clamp="none"),
    "one_pixel": _one_op(_image((1, 1, 3)), _gains(2, 2, 3)),
    "large": _one_op(_image((300, 517, 3)), _gains(31, 52, 3)),
    "cropped_before": _one_op(
        _RGB, _gains(5, 6, 3), before=_CROP_BEFORE, area=(4, 3, 38, 50), row_pitch=2
    ),
    "cropped_after": _one_op(
        _RGB, _gains(5, 6, 3), after=(slice(7, 31), slice(11, 40))
    ),
    "cropped_before_and_after": _one_op(
        _RGB,
        _gains(5, 6, 3),
        before=_CROP_BEFORE,
        after=_CROP_AFTER,
        area=(4, 3, 38, 50),
        col_pitch=2,
    ),
    "four_bayer_phases_cropped_before_and_after": (
        _MONO,
        _four_phases(),
        _CROP_BEFORE,
        _CROP_AFTER,
    ),
}


def run_case(name, dtype=np.float32, max_gain=False, engine_opts=None):
    """The case's ``mi`` result, realized, on its source as ``dtype``. With
    ``max_gain`` each op gets the largest gain of its table, so a uint16
    source multiplies by a fixed-point gain."""
    src, ops, before, after = CASES[name]
    x = Array(_as_dtype(src, dtype))
    if before is not None:
        x = x[before]
    for gains, attrs in ops:
        if max_gain:
            attrs = {**attrs, "max_gain": float(np.max(gains))}
        if engine_opts is not None:
            attrs = {**attrs, "engine_opts": engine_opts}
        x = _apply(x, gains, attrs)
    if after is not None:
        x = x[after]
    return x.realize()


def reference_case(name, dtype=np.float32):
    """The case's reference: the ops on the whole source, then both crops."""
    src, ops, before, after = CASES[name]
    y = _as_unit(_as_dtype(src, dtype))
    for gains, attrs in ops:
        y = _reference(y, gains, **attrs)
    if before is not None:
        y = y[before]
    return y if after is None else y[after]


def _as_dtype(x, dtype):
    """``x`` as ``dtype``: a uint16 code is the value times 65535, rounded,
    and a value outside [0, 1] saturates."""
    if dtype == np.uint16:
        return np.round(np.clip(x, 0.0, 1.0) * 65535).astype(np.uint16)
    return x.astype(dtype)


def _as_unit(x):
    """The values of ``x`` in double: a uint16 code over 65535."""
    if x.dtype == np.uint16:
        return x.astype(np.float64) / 65535
    return x.astype(np.float64)


@pytest.mark.parametrize("name", sorted(CASES))
def test_apply_gain_map_matches_reference(name):
    _assert_close(run_case(name), reference_case(name))


@pytest.mark.parametrize("dtype", [np.float32, np.uint16, np.float16])
@pytest.mark.parametrize("name", sorted(CASES))
def test_apply_gain_map_neon_matches_scalar(name, dtype):
    """``engine_opts {"core": {"neon": false}}`` forces the scalar loops,
    which give the same bits as the NEON loops."""
    scalar = run_case(name, dtype, engine_opts='{"core": {"neon": false}}')
    np.testing.assert_array_equal(run_case(name, dtype), scalar)


def test_apply_gain_map_changes_only_the_area_at_its_pitch():
    """A sample outside the area, off the pitch, or on a plane not named
    is copied bit for bit."""
    src, _, _, _ = CASES["rgb_row_pitch_2_col_pitch_3"]
    got = run_case("rgb_row_pitch_2_col_pitch_3")
    changed = got != src
    assert changed[3:37:2, 5:51:3].all()
    changed[3:37:2, 5:51:3] = False
    assert not changed.any()


def test_apply_gain_map_reads_a_computed_table():
    """The table may be a node of the graph, not only a source array."""
    gains = _gains(5, 6, 3)
    computed = Array(gains) * np.float32(0.5) + np.float32(0.6)
    got = _apply(Array(_RGB), computed, _attrs()).realize()
    want = _reference(_RGB, gains.astype(np.float64) * 0.5 + 0.6, **_attrs())
    _assert_close(got, want)


@pytest.mark.parametrize(
    "plane_attrs",
    [{}, {"start_plane": 1, "num_planes": 2}],
    ids=["every_plane", "planes_1_and_2"],
)
@pytest.mark.parametrize("map_planes", [1, 3])
def test_apply_gain_map_on_chw_matches_hwc(map_planes, plane_attrs):
    """On a (C, H, W) array each frame is one DNG plane, and reads the map
    plane of its frame."""
    gains = _gains(5, 6, map_planes)
    attrs = _attrs(area=(3, 5, 37, 51), row_pitch=2, **plane_attrs)
    chw = Array(np.ascontiguousarray(np.moveaxis(_RGB, -1, 0)), channel_axis=0)
    got = mi.moveaxis(_apply(chw, gains, attrs), 0, -1).realize()
    _assert_close(got, _reference(_RGB, gains, **attrs))


def test_apply_gain_map_on_a_stack_uses_the_channel_as_the_plane():
    """On an (N, H, W, C) stack the plane is the channel, in every frame."""
    stack = _image((2, _HEIGHT, _WIDTH, 3))
    gains = _gains(5, 6, 3)
    attrs = _attrs(start_plane=1, num_planes=2)
    got = _apply(Array(stack), gains, attrs).realize()
    _assert_close(got, np.stack([_reference(frame, gains, **attrs) for frame in stack]))


_SIXTEEN_BIT_CASES = [
    "rgb_one_plane",
    "rgb_three_planes",
    "rgb_planes_1_and_2_three_map_planes",
    "mono_2d",
    "two_channels_two_map_planes",
    "four_channels_four_map_planes",
    "five_channels_two_map_planes",
    "rgb_row_pitch_2_col_pitch_3",
    "mono_col_pitch_2",
    "mono_col_pitch_3",
    "mono_col_pitch_4",
    "four_bayer_phases",
    "map_inside_the_image_holds_its_edges",
    "rgb_whole_cells",
    "mono_whole_cells",
    "clamp_max_one",
    "large",
    "four_bayer_phases_cropped_before_and_after",
]


@pytest.mark.parametrize(
    "max_gain", [False, True], ids=["float_gain", "fixed_point_gain"]
)
@pytest.mark.parametrize("name", _SIXTEEN_BIT_CASES)
def test_apply_gain_map_on_uint16_matches_reference(name, max_gain):
    """A uint16 source keeps its dtype. Without ``max_gain`` each code is
    multiplied by the float32 gain and rounded; with it, by a fixed-point
    gain with at least 14 fraction bits for these gains, so a result is
    within one code of the reference plus half a fixed-point step times
    the code. A code saturates at 65535."""
    got = run_case(name, np.uint16, max_gain)
    assert got.dtype == np.uint16
    expect = np.clip(reference_case(name, np.uint16) * 65535.0, 0.0, 65535.0)
    np.testing.assert_allclose(
        got.astype(np.float64), expect, rtol=0, atol=1.0 + 65535 * 2.0**-15
    )


@pytest.mark.parametrize("name", _SIXTEEN_BIT_CASES)
def test_apply_gain_map_on_float16_matches_reference(name):
    """A float16 source keeps its dtype and is multiplied by the gain
    rounded to float16, so a result is within about two float16
    roundings of the reference."""
    got = run_case(name, np.float16)
    assert got.dtype == np.float16
    np.testing.assert_allclose(
        got.astype(np.float32),
        reference_case(name, np.float16),
        rtol=2.0**-10,
        atol=2.0**-24,
    )
