"""``map_polynomial`` (DNG MapPolynomial) against a NumPy port of the DNG
SDK's ``RefBaselineMapPoly32``.

NumPy has no such function, so each case compares ``mi`` with
``_reference``. The engine kernels are built with ``-ffast-math``, which
may fuse a multiply and an add, so the comparison allows a few float32
rounding steps instead of asking for equal bits.
"""

import numpy as np
import pytest

import muimage as mi
from muraw.array import Array

_HEIGHT = 40
_WIDTH = 53

# Coefficients c0 .. c_degree for each degree the kernel has a form for.
# Degrees 0 to 4 each have their own Horner form; 5 and up share a loop.
_COEFFICIENTS = {
    0: [0.25],
    1: [0.02, 0.9],
    2: [0.01, 1.1, -0.15],
    3: [0.0, 1.2, -0.4, 0.15],
    4: [0.01, 0.8, 0.5, -0.6, 0.25],
    6: [0.0, 1.05, -0.2, 0.3, -0.25, 0.15, -0.05],
}

# Three degree-3 sets, one per plane of an RGB image.
_PER_PLANE_SETS = [
    0.0,
    1.2,
    -0.4,
    0.15,
    0.01,
    0.9,
    0.2,
    -0.1,
    -0.02,
    1.4,
    -0.6,
    0.2,
]


def _image(shape, low=0.0, high=1.0, seed=3):
    rng = np.random.default_rng(seed)
    return rng.uniform(low, high, shape).astype(np.float32)


def _poly(x, coefficients, degree):
    """``RefBaselineMapPoly32`` on the samples ``x``, before the clamp. A
    negative sample is evaluated with the signs the SDK uses there."""
    c = [np.float32(value) for value in coefficients]
    if degree == 0:
        return np.full_like(x, c[0])
    if degree == 1:
        return c[0] + x * c[1]
    negative = x < 0
    if degree <= 4:
        inner_pos = np.full_like(x, c[degree])
        inner_neg = np.full_like(x, c[degree])
        for k in range(degree - 1, 0, -1):
            inner_pos = c[k] + x * inner_pos
            inner_neg = c[k] - x * inner_neg
        return np.where(negative, c[0] + x * inner_neg, c[0] + x * inner_pos)
    magnitude = np.abs(x)
    power = magnitude.copy()
    total = np.zeros_like(x)
    for j in range(1, degree + 1):
        total = total + c[j] * power
        power = power * magnitude
    return np.where(negative, c[0] - total, c[0] + total)


def _reference(
    x,
    *,
    top,
    left,
    bottom,
    right,
    start_plane,
    num_planes,
    row_pitch,
    col_pitch,
    coefficients,
    degree,
):
    """DNG MapPolynomial on an (H, W) or (H, W, C) float32 array. The area
    is clipped to the image; a sample is changed when it is at a whole
    number of pitches from the area's top-left corner. ``coefficients``
    is one set for every plane, or one set per plane."""
    out = x.copy()
    height, width = x.shape[:2]
    planes = x.shape[2] if x.ndim == 3 else 1
    if top == 0 and bottom == 0:
        bottom = height
    if left == 0 and right == 0:
        right = width
    rows = [r for r in range(top, bottom, row_pitch) if 0 <= r < height]
    cols = [c for c in range(left, right, col_pitch) if 0 <= c < width]
    plane_list = [p for p in range(start_plane, start_plane + num_planes) if p < planes]
    if not rows or not cols or not plane_list:
        return out
    sets = np.asarray(coefficients).reshape(-1, degree + 1)
    view = out if x.ndim == 3 else out[..., None]
    for plane in plane_list:
        index = np.ix_(rows, cols, [plane])
        values = sets[0] if len(sets) == 1 else sets[plane - start_plane]
        view[index] = np.clip(_poly(view[index], values, degree), -1.0, 1.0)
    return out


def _attrs(
    degree=3,
    *,
    top=0,
    left=0,
    bottom=0,
    right=0,
    start_plane=0,
    num_planes=3,
    row_pitch=1,
    col_pitch=1,
    coefficients=None,
):
    if coefficients is None:
        coefficients = _COEFFICIENTS[degree]
    return dict(
        top=top,
        left=left,
        bottom=bottom,
        right=right,
        start_plane=start_plane,
        num_planes=num_planes,
        row_pitch=row_pitch,
        col_pitch=col_pitch,
        coefficients=np.asarray(coefficients, np.float32),
        degree=degree,
    )


def _mi_map_polynomial(x, attrs):
    return mi.map_polynomial(x, **attrs)


def _assert_close(got, want):
    assert got.shape == want.shape
    assert got.dtype == want.dtype == np.float32
    np.testing.assert_allclose(got, want, rtol=0, atol=4e-6)


# name -> (source, attrs, crop before the op, crop after the op).
CASES = {
    "rgb_degree3_full": (_image((_HEIGHT, _WIDTH, 3)), _attrs(3), None, None),
    "rgb_full_area_explicit": (
        _image((_HEIGHT, _WIDTH, 3)),
        _attrs(3, bottom=_HEIGHT, right=_WIDTH),
        None,
        None,
    ),
    **{
        f"rgb_degree{degree}_partial": (
            _image((_HEIGHT, _WIDTH, 3)),
            _attrs(degree, top=4, left=6, bottom=33, right=47),
            None,
            None,
        )
        for degree in _COEFFICIENTS
    },
    **{
        f"rgb_degree{degree}_negative": (
            _image((_HEIGHT, _WIDTH, 3), low=-1.0, high=1.0),
            _attrs(degree),
            None,
            None,
        )
        for degree in _COEFFICIENTS
    },
    "rgb_pitch2_odd_offsets": (
        _image((_HEIGHT, _WIDTH, 3)),
        _attrs(3, top=3, left=5, bottom=37, right=51, row_pitch=2, col_pitch=2),
        None,
        None,
    ),
    "rgb_pitch3_by_4": (
        _image((_HEIGHT, _WIDTH, 3)),
        _attrs(
            2, top=1, left=2, row_pitch=3, col_pitch=4, bottom=_HEIGHT, right=_WIDTH
        ),
        None,
        None,
    ),
    "rgb_planes_1_and_2": (
        _image((_HEIGHT, _WIDTH, 3)),
        _attrs(3, start_plane=1, num_planes=2),
        None,
        None,
    ),
    "rgb_plane_count_past_channels": (
        _image((_HEIGHT, _WIDTH, 3)),
        _attrs(3, start_plane=2, num_planes=4),
        None,
        None,
    ),
    "rgb_per_plane_sets": (
        _image((_HEIGHT, _WIDTH, 3)),
        _attrs(3, top=4, left=6, bottom=33, right=47, coefficients=_PER_PLANE_SETS),
        None,
        None,
    ),
    "rgb_per_plane_sets_planes_1_and_2_pitch2": (
        _image((_HEIGHT, _WIDTH, 3)),
        _attrs(
            3,
            start_plane=1,
            num_planes=2,
            row_pitch=2,
            col_pitch=2,
            coefficients=_PER_PLANE_SETS[4:],
        ),
        None,
        None,
    ),
    "rgb_clamped": (
        _image((_HEIGHT, _WIDTH, 3), low=-1.0, high=1.0),
        _attrs(2, coefficients=[0.3, 2.5, 1.5]),
        None,
        None,
    ),
    "mono_degree3_pitch2": (
        _image((_HEIGHT, _WIDTH)),
        _attrs(
            3,
            top=1,
            left=0,
            bottom=_HEIGHT,
            right=_WIDTH,
            row_pitch=2,
            col_pitch=2,
            num_planes=1,
        ),
        None,
        None,
    ),
    "mono_degree6_full": (
        _image((_HEIGHT, _WIDTH)),
        _attrs(6, num_planes=1),
        None,
        None,
    ),
    "rgb_cropped_before": (
        _image((_HEIGHT, _WIDTH, 3)),
        _attrs(3, top=2, left=3, bottom=30, right=41, row_pitch=2, col_pitch=3),
        (slice(3, None), slice(5, None)),
        None,
    ),
    "rgb_cropped_after": (
        _image((_HEIGHT, _WIDTH, 3)),
        _attrs(4, top=3, left=5, bottom=37, right=51, row_pitch=2, col_pitch=2),
        None,
        (slice(7, 31), slice(11, 40)),
    ),
    "rgb_partial_cropped_after": (
        _image((_HEIGHT, _WIDTH, 3)),
        _attrs(3, top=2, left=9, bottom=35, right=44),
        None,
        (slice(5, 30), slice(4, 50)),
    ),
    "mono_cropped_before_and_after": (
        _image((_HEIGHT, _WIDTH)),
        _attrs(
            2,
            top=1,
            left=1,
            row_pitch=2,
            col_pitch=2,
            bottom=_HEIGHT - 3,
            right=_WIDTH - 5,
            num_planes=1,
        ),
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
    y = _mi_map_polynomial(x, attrs)
    if after is not None:
        y = y[after]
    return y.realize()


def reference_case(name):
    """The case's reference. The area is on the canvas, so a crop before the
    op gives the crop of the op on the whole source."""
    src, attrs, before, after = CASES[name]
    y = _reference(src, **attrs)
    if before is not None:
        y = y[before]
    return y if after is None else y[after]


@pytest.mark.parametrize("name", sorted(CASES))
def test_map_polynomial_matches_reference(name):
    _assert_close(run_case(name), reference_case(name))


def test_map_polynomial_changes_the_samples_it_should():
    """The cases would pass for an op that changes nothing if the reference
    did too; check the reference itself moves the samples in the area."""
    src, attrs, _, _ = CASES["rgb_pitch2_odd_offsets"]
    want = reference_case("rgb_pitch2_odd_offsets")
    changed = want != src
    assert changed[3:37:2, 5:51:2].all()
    changed[3:37:2, 5:51:2] = False
    assert not changed.any()


def test_map_polynomial_on_chw_matches_hwc():
    """On a (C, H, W) array each frame is one DNG plane, with its own set."""
    hwc = _image((_HEIGHT, _WIDTH, 3))
    attrs = _attrs(
        3,
        top=3,
        left=5,
        bottom=37,
        right=51,
        start_plane=1,
        num_planes=2,
        row_pitch=2,
        col_pitch=2,
        coefficients=_PER_PLANE_SETS[4:],
    )
    chw = Array(np.ascontiguousarray(np.moveaxis(hwc, -1, 0)), channel_axis=0)
    got = mi.moveaxis(_mi_map_polynomial(chw, attrs), 0, -1).realize()
    _assert_close(got, _reference(hwc, **attrs))


def test_map_polynomial_on_a_stack_uses_the_channel_as_the_plane():
    """On an (N, H, W, C) stack the plane is the channel, in every frame."""
    stack = _image((2, _HEIGHT, _WIDTH, 3))
    attrs = _attrs(2, start_plane=2, num_planes=1)
    got = _mi_map_polynomial(Array(stack), attrs).realize()
    want = np.stack([_reference(frame, **attrs) for frame in stack])
    _assert_close(got, want)


@pytest.mark.parametrize(
    "area",
    [
        dict(top=10, left=20, bottom=_HEIGHT + 7, right=_WIDTH + 9),
        dict(top=-3, left=-4, bottom=25, right=30),
    ],
    ids=["past_bottom_right", "before_top_left"],
)
def test_map_polynomial_clips_the_area_to_the_image(area):
    """The area is clipped to the image, keeping the pitch counted from the
    area's own top-left corner."""
    src = _image((_HEIGHT, _WIDTH, 3))
    attrs = _attrs(3, row_pitch=2, col_pitch=3, **area)
    _assert_close(
        _mi_map_polynomial(Array(src), attrs).realize(), _reference(src, **attrs)
    )


def _plane_opcode(plane, coefficients, **area):
    return {
        "type": "MapPolynomial",
        "area": {"top": 0, "left": 0, "bottom": 0, "right": 0, **area},
        "plane": plane,
        "planes": 1,
        "row_pitch": 1,
        "col_pitch": 1,
        "degree": 3,
        "coefficients": np.asarray(coefficients, np.float64),
    }


def test_apply_opcodes_joins_one_opcode_per_plane_into_one_node():
    """Three MapPolynomial opcodes, one per plane with the same area, run
    as one node with a coefficient set per plane."""
    from muraw.raw_render import apply_opcodes

    src = _image((_HEIGHT, _WIDTH, 3))
    sets = np.reshape(_PER_PLANE_SETS, (3, 4))
    got = apply_opcodes(Array(src), [_plane_opcode(p, sets[p]) for p in range(3)])
    assert got._node.op == "map_polynomial"
    assert got._node.inputs[0]._node is None
    _assert_close(
        got.realize(), _reference(src, **_attrs(3, coefficients=_PER_PLANE_SETS))
    )


def test_apply_opcodes_keeps_opcodes_with_different_areas_apart():
    from muraw.raw_render import apply_opcodes

    src = _image((_HEIGHT, _WIDTH, 3))
    sets = np.reshape(_PER_PLANE_SETS, (3, 4))
    opcodes = [_plane_opcode(0, sets[0]), _plane_opcode(1, sets[1], top=5, bottom=30)]
    got = apply_opcodes(Array(src), opcodes)
    assert got._node.inputs[0]._node.op == "map_polynomial"
    want = _reference(src, **_attrs(3, num_planes=1, coefficients=sets[0]))
    want = _reference(
        want,
        **_attrs(
            3, start_plane=1, num_planes=1, top=5, bottom=30, coefficients=sets[1]
        ),
    )
    _assert_close(got.realize(), want)


def test_map_polynomial_rejects_a_coefficient_count_that_fits_no_set_layout():
    attrs = _attrs(3, coefficients=_PER_PLANE_SETS[:8])
    with pytest.raises(RuntimeError):
        _mi_map_polynomial(Array(_image((_HEIGHT, _WIDTH, 3))), attrs).realize()


def test_map_polynomial_rejects_too_few_coefficients():
    attrs = _attrs(3, coefficients=[0.0, 1.0])
    with pytest.raises(RuntimeError):
        _mi_map_polynomial(Array(_image((_HEIGHT, _WIDTH, 3))), attrs).realize()
