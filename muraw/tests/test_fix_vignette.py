"""``fix_vignette`` (DNG FixVignetteRadial) against a NumPy port of the
radial gain.

NumPy has no such function, so each case compares ``mi`` with
``_reference``, at every spacing in ``_MAP_SPACINGS``. With
``map_spacing`` 1 the gain is the formula at each pixel; above 1 it is
the formula at every ``map_spacing``-th canvas column and row,
interpolated bilinearly. The reference computes in double precision and
the kernels in float32, so the two agree to a few float32 ulps.
"""

import numpy as np
import pytest

import muimage as mi
from muraw.array import Array

_HEIGHT = 40
_WIDTH = 53

_FIVE = [0.4, -0.2, 0.3, 0.1, 0.05]

# 8 is the default.
_MAP_SPACINGS = [1, 2, 8]


def _image(shape, low=0.0, high=1.0, seed=5):
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, shape).astype(np.float32)


def _reference(x, *, params, center_x, center_y, map_spacing=1):
    """FixVignetteRadial on an (H, W) or (H, W, C) float32 array whose
    first pixel is (0, 0). The distance is measured from the pixel's
    center, half a pixel right of and below its column and row, as the DNG
    SDK measures it, and divided by the distance from the optical center
    to the farthest corner, at least 1. At most five coefficients are
    used. Above 1,
    ``map_spacing`` interpolates the gain from the four multiples of it
    around each pixel."""
    height, width = x.shape[:2]
    cx = center_x * width
    cy = center_y * height
    corners = [
        np.sqrt(cx * cx + cy * cy),
        np.sqrt((width - cx) * (width - cx) + cy * cy),
        np.sqrt(cx * cx + (height - cy) * (height - cy)),
        np.sqrt((width - cx) * (width - cx) + (height - cy) * (height - cy)),
    ]
    max_dist = max(max(corners), 1.0)

    def gain_at(cols, rows):
        dx = (cols + 0.5 - cx) / max_dist
        dy = (rows + 0.5 - cy) / max_dist
        r2 = dx * dx + dy * dy
        gain = np.ones_like(r2)
        power = r2.copy()
        for value in list(params)[:5]:
            gain = gain + value * power
            power = power * r2
        return gain

    rows, cols = np.mgrid[0:height, 0:width].astype(np.float64)
    step = float(map_spacing)
    left = np.floor(cols / step) * step
    top = np.floor(rows / step) * step
    fx = (cols - left) / step
    fy = (rows - top) / step
    upper = gain_at(left, top) * (1 - fx) + gain_at(left + step, top) * fx
    lower = gain_at(left, top + step) * (1 - fx) + gain_at(left + step, top + step) * fx
    gain = upper * (1 - fy) + lower * fy
    if x.ndim == 3:
        gain = gain[..., None]
    return np.clip(x * gain, 0.0, 1.0).astype(np.float32)


def _assert_close(got, expect):
    np.testing.assert_allclose(got, expect, rtol=2e-6, atol=1e-7)


def _mi_fix_vignette(x, attrs, map_spacing):
    return mi.fix_vignette(
        x,
        params=np.asarray(attrs["params"], dtype=np.float64),
        center_x=float(attrs["center_x"]),
        center_y=float(attrs["center_y"]),
        map_spacing=map_spacing,
    )


def _attrs(params=_FIVE, center_x=0.3, center_y=0.6):
    return {"params": list(params), "center_x": center_x, "center_y": center_y}


# Each case: source array, attrs, a crop of the source before the op, and a
# crop of the result after it. A crop keeps the source's canvas, so the
# center and the farthest corner stay the source's either way.
CASES = {
    "rgb_off_center_five_terms": (_image((_HEIGHT, _WIDTH, 3)), _attrs(), None, None),
    "rgb_centered_one_term": (
        _image((_HEIGHT, _WIDTH, 3)),
        _attrs([0.25], 0.5, 0.5),
        None,
        None,
    ),
    "rgb_darkening": (
        _image((_HEIGHT, _WIDTH, 3)),
        _attrs([-0.5, 0.1, -0.05]),
        None,
        None,
    ),
    "rgb_more_than_five_terms": (
        _image((_HEIGHT, _WIDTH, 3)),
        _attrs([*_FIVE, 3.0, -7.0]),
        None,
        None,
    ),
    "rgb_center_outside_the_image": (
        _image((_HEIGHT, _WIDTH, 3)),
        _attrs(center_x=1.4, center_y=-0.2),
        None,
        None,
    ),
    "rgb_values_outside_unit_range": (
        _image((_HEIGHT, _WIDTH, 3), low=-0.5, high=1.5),
        _attrs(),
        None,
        None,
    ),
    "mono_2d": (_image((_HEIGHT, _WIDTH)), _attrs(), None, None),
    "one_channel": (_image((_HEIGHT, _WIDTH, 1)), _attrs(), None, None),
    "two_channels": (_image((_HEIGHT, _WIDTH, 2)), _attrs(), None, None),
    "four_channels": (_image((_HEIGHT, _WIDTH, 4)), _attrs(), None, None),
    "one_pixel": (_image((1, 1, 3)), _attrs([0.5], 0.5, 0.5), None, None),
    "large": (_image((300, 517, 3)), _attrs(), None, None),
    "cropped_before": (
        _image((_HEIGHT, _WIDTH, 3)),
        _attrs(),
        (slice(3, None), slice(5, None)),
        None,
    ),
    "cropped_after": (
        _image((_HEIGHT, _WIDTH, 3)),
        _attrs(),
        None,
        (slice(7, 31), slice(11, 40)),
    ),
    "cropped_before_and_after": (
        _image((_HEIGHT, _WIDTH, 3)),
        _attrs(),
        (slice(3, None), slice(5, None)),
        (slice(4, 29), slice(9, 37)),
    ),
    "mono_cropped_before_and_after": (
        _image((_HEIGHT, _WIDTH)),
        _attrs(),
        (slice(3, None), slice(5, None)),
        (slice(4, 29), slice(9, 37)),
    ),
}


def run_case(name, map_spacing):
    """The case's ``mi`` result, realized."""
    src, attrs, before, after = CASES[name]
    x = Array(src)
    if before is not None:
        x = x[before]
    y = _mi_fix_vignette(x, attrs, map_spacing)
    if after is not None:
        y = y[after]
    return y.realize()


def reference_case(name, map_spacing):
    """The case's reference: the vignette of the whole source, then both
    crops."""
    src, attrs, before, after = CASES[name]
    y = _reference(src, **attrs, map_spacing=map_spacing)
    if before is not None:
        y = y[before]
    return y if after is None else y[after]


@pytest.mark.parametrize("map_spacing", _MAP_SPACINGS)
@pytest.mark.parametrize("name", sorted(CASES))
def test_fix_vignette_matches_reference(name, map_spacing):
    _assert_close(run_case(name, map_spacing), reference_case(name, map_spacing))


@pytest.mark.parametrize("map_spacing", _MAP_SPACINGS)
def test_fix_vignette_after_reset_canvas_centers_on_the_view(map_spacing):
    """A ``crop_reset_canvas`` view starts a canvas of its own, so the
    vignette is centered in the view, as on a copy of it."""
    src, attrs, before, _ = CASES["cropped_before"]
    view = Array(src).view(before, mode="crop_reset_canvas")
    copy = Array(np.ascontiguousarray(src[before]))
    from_copy = _mi_fix_vignette(copy, attrs, map_spacing).realize()
    np.testing.assert_array_equal(
        _mi_fix_vignette(view, attrs, map_spacing).realize(), from_copy
    )
    _assert_close(from_copy, _reference(src[before], **attrs, map_spacing=map_spacing))


_SIXTEEN_BIT_CASES = [
    "rgb_off_center_five_terms",
    "rgb_values_outside_unit_range",
    "mono_2d",
    "two_channels",
    "four_channels",
    "large",
    "cropped_before_and_after",
]


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


@pytest.mark.parametrize("dtype", [np.uint16, np.float16])
@pytest.mark.parametrize("map_spacing", _MAP_SPACINGS)
@pytest.mark.parametrize("name", _SIXTEEN_BIT_CASES)
def test_fix_vignette_on_16_bit_samples_matches_reference(name, map_spacing, dtype):
    """A uint16 or float16 source keeps its dtype. The gain is float32 and
    each sample is multiplied by it at its dtype. A uint16 code is
    multiplied by a fixed-point gain with at least 14 fraction bits for
    these gains, so a result is within one code of the reference plus half
    a fixed-point step times the code. A float16 result is within about two
    float16 roundings (one of the gain, one of the product)."""
    src, attrs, before, after = CASES[name]
    source = _as_dtype(src, dtype)
    x = Array(source)
    if before is not None:
        x = x[before]
    y = _mi_fix_vignette(x, attrs, map_spacing)
    if after is not None:
        y = y[after]
    got = y.realize()
    assert got.dtype == dtype

    expect = _reference(_as_unit(source), **attrs, map_spacing=map_spacing)
    if before is not None:
        expect = expect[before]
    if after is not None:
        expect = expect[after]
    if dtype == np.uint16:
        np.testing.assert_allclose(
            got.astype(np.float64),
            expect * 65535.0,
            rtol=0,
            atol=1.0 + 65535 * 2.0**-15,
        )
    else:
        np.testing.assert_allclose(
            got.astype(np.float32), expect, rtol=2.0**-10, atol=2.0**-24
        )


def test_fix_vignette_default_map_spacing_is_8():
    src, attrs, _, _ = CASES["rgb_off_center_five_terms"]
    got = mi.fix_vignette(
        Array(src),
        params=np.asarray(attrs["params"], dtype=np.float64),
        center_x=float(attrs["center_x"]),
        center_y=float(attrs["center_y"]),
    ).realize()
    np.testing.assert_array_equal(got, run_case("rgb_off_center_five_terms", 8))
