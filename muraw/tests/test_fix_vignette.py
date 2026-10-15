"""``fix_vignette`` (DNG FixVignetteRadial) against a NumPy port of the
radial gain.

NumPy has no such function, so each case compares ``mi`` with
``_reference``. The gain is computed in double precision in the same
order as the kernel and cast to float32 before the multiply, so the two
give the same bits.
"""

import numpy as np
import pytest

import muimage as mi
from muraw.array import Array

_HEIGHT = 40
_WIDTH = 53

_FIVE = [0.4, -0.2, 0.3, 0.1, 0.05]


def _image(shape, low=0.0, high=1.0, seed=5):
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, shape).astype(np.float32)


def _reference(x, *, params, center_x, center_y):
    """FixVignetteRadial on an (H, W) or (H, W, C) float32 array whose
    first pixel is (0, 0). The distance is measured from the integer
    column and row and divided by the distance to the farthest corner,
    at least 1. At most five coefficients are used."""
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
    rows, cols = np.mgrid[0:height, 0:width].astype(np.float64)
    dx = (cols - cx) / max_dist
    dy = (rows - cy) / max_dist
    r2 = dx * dx + dy * dy
    gain = np.ones_like(r2)
    power = r2.copy()
    for value in list(params)[:5]:
        gain = gain + value * power
        power = power * r2
    gain = gain.astype(np.float32)
    if x.ndim == 3:
        gain = gain[..., None]
    return np.clip(x * gain, np.float32(0.0), np.float32(1.0))


def _mi_fix_vignette(x, attrs):
    return mi.fix_vignette(
        x,
        params=np.asarray(attrs["params"], dtype=np.float64),
        center_x=float(attrs["center_x"]),
        center_y=float(attrs["center_y"]),
    )


def _attrs(params=_FIVE, center_x=0.3, center_y=0.6):
    return {"params": list(params), "center_x": center_x, "center_y": center_y}


# Each case: source array, attrs, a crop of the source before the op, and a
# crop of the result after it.
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


def run_case(name):
    """The case's ``mi`` result, realized."""
    src, attrs, before, after = CASES[name]
    x = Array(src)
    if before is not None:
        x = x[before]
    y = _mi_fix_vignette(x, attrs)
    if after is not None:
        y = y[after]
    return y.realize()


def reference_case(name):
    src, attrs, before, after = CASES[name]
    x = src if before is None else np.ascontiguousarray(src[before])
    y = _reference(x, **attrs)
    return y if after is None else y[after]


@pytest.mark.parametrize("name", sorted(CASES))
def test_fix_vignette_matches_reference(name):
    np.testing.assert_array_equal(run_case(name), reference_case(name))
