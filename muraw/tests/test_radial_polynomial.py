"""``mi.radial_polynomial`` against a NumPy port of its formula.

NumPy has no such function, so each case compares ``mi`` with
``_reference``. The op computes in float32 and the reference in double,
so the two agree to a few float32 ulps.
"""

import numpy as np
import pytest

import muimage as mi

_ATTRS = {
    "coefficients": [0.2, 0.5, -0.25, 0.1],
    "center_x": 10.25,
    "center_y": 2.75,
    "radius": 12.5,
    "x0": -1.5,
    "y0": 0.5,
    "spacing_x": 2.0,
    "spacing_y": 3.0,
}


def _reference(
    shape, *, coefficients, center_x, center_y, radius, x0, y0, spacing_x, spacing_y
):
    """``c0 + c1 r^2 + ...`` at sample ``(row, col)``, which sits at
    ``(x0 + col * spacing_x, y0 + row * spacing_y)``."""
    rows, cols = np.mgrid[0 : shape[0], 0 : shape[1]].astype(np.float64)
    dx = (x0 + cols * spacing_x - center_x) / radius
    dy = (y0 + rows * spacing_y - center_y) / radius
    r_squared = dx * dx + dy * dy
    value = np.zeros_like(r_squared)
    for coefficient in reversed(coefficients):
        value = value * r_squared + coefficient
    return value


def _composed(
    lib,
    shape,
    *,
    coefficients,
    center_x,
    center_y,
    radius,
    x0,
    y0,
    spacing_x,
    spacing_y,
):
    """The same polynomial built from ``arange``, ``meshgrid`` and float32
    arithmetic, in Horner order."""
    rows, cols = shape
    inv_radius = np.float32(1 / radius)
    xs = lib.arange(cols, dtype=np.float32) * np.float32(spacing_x)
    xs = (xs + np.float32(x0 - center_x)) * inv_radius
    ys = lib.arange(rows, dtype=np.float32) * np.float32(spacing_y)
    ys = (ys + np.float32(y0 - center_y)) * inv_radius
    dx, dy = lib.meshgrid(xs, ys)
    r_squared = dx * dx + dy * dy
    value = lib.full(shape, np.float32(coefficients[-1]), dtype=np.float32)
    for coefficient in reversed(coefficients[:-1]):
        value = value * r_squared + np.float32(coefficient)
    return value


def test_radial_polynomial_composed_from_array_ops_matches_numpy_and_the_op():
    shape = (31, 45)
    got = _composed(mi, shape, **_ATTRS).realize()
    np.testing.assert_allclose(got, _composed(np, shape, **_ATTRS), rtol=1e-6)
    op = mi.radial_polynomial(shape, **_ATTRS).realize()
    np.testing.assert_allclose(got, op, rtol=4e-6, atol=4e-6)


def test_radial_polynomial_matches_reference():
    got = mi.radial_polynomial((7, 37), **_ATTRS).realize()
    assert got.dtype == np.float32 and got.shape == (7, 37)
    np.testing.assert_allclose(got, _reference((7, 37), **_ATTRS), rtol=4e-6, atol=4e-6)


def test_radial_polynomial_defaults_to_one_sample_per_unit_from_the_origin():
    got = mi.radial_polynomial(
        (5, 9), coefficients=[0.0, 1.0], center_x=0.0, center_y=0.0, radius=1.0
    ).realize()
    rows, cols = np.mgrid[0:5, 0:9]
    np.testing.assert_array_equal(got, (rows * rows + cols * cols).astype(np.float32))


def test_radial_polynomial_on_a_slice_matches_the_slice_of_the_whole():
    whole = mi.radial_polynomial((9, 14), **_ATTRS)
    np.testing.assert_array_equal(
        whole[2:7, 3:11].realize(), whole.realize()[2:7, 3:11]
    )


def test_radial_polynomial_takes_a_trailing_channel_axis_of_one():
    got = mi.radial_polynomial((7, 37, 1), **_ATTRS).realize()
    np.testing.assert_array_equal(
        got[..., 0], mi.radial_polynomial((7, 37), **_ATTRS).realize()
    )


@pytest.mark.parametrize(
    "shape, kwargs, error, match",
    [
        (37, {}, ValueError, "writes a 2D array"),
        ((7, 37, 3), {}, ValueError, r"writes 1 channel\(s\); got shape \(7, 37, 3\)"),
        ((7, 37), {"dtype": np.uint16}, ValueError, "writes float32; got dtype uint16"),
        (mi.zeros((7, 37)), {}, TypeError, "takes the shape of its output"),
    ],
    ids=["one_axis", "three_channels", "uint16", "an_array"],
)
def test_radial_polynomial_rejects_what_it_does_not_write(shape, kwargs, error, match):
    with pytest.raises(error, match=match):
        mi.radial_polynomial(shape, **kwargs, **_ATTRS)


def test_fill_and_ramp_take_a_shape_and_a_dtype():
    np.testing.assert_array_equal(
        mi.fill((2, 3), value=[1.5]).realize(), np.full((2, 3), 1.5, np.float32)
    )
    got = mi.ramp((4,), dtype=np.uint16, along="columns", start=[2.0], step=[3.0])
    np.testing.assert_array_equal(got.realize(), np.array([2, 5, 8, 11], np.uint16))
