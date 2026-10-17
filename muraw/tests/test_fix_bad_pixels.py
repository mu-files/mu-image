"""``cfa_fix_bad_pixels_constant`` (DNG FixBadPixelsConstant) against a
NumPy port of the DNG SDK's loop.

NumPy has no such function, so each case compares ``mi`` with
``_reference``. The reference reads ``bayer_phase`` as the DNG spec
defines it (0 red, 1 green on a red row, 2 green on a blue row, 3 blue);
the SDK's ``IsGreen`` reads it the other way round. Past the image edge
it replaces a neighbour as the SDK does, by the sample of the same colour
in the two outer rows or columns.
"""

import numpy as np
import pytest

import muimage as mi
from muraw.array import Array

_GREEN_OFFSETS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
_RED_BLUE_OFFSETS = ((-2, 0), (2, 0), (0, -2), (0, 2))


def _repeat_edge(index, size):
    """``index`` past ``0..size`` moved to the index of the same colour
    among the two outer ones, then held inside ``0..size`` when that holds
    a single index."""
    if index < 0:
        index = index % 2
    elif index >= size:
        index = size - 1 - (size - 1 - index) % 2
    return min(max(index, 0), size - 1)


def _reference(x, constant, bayer_phase, origin=(0, 0)):
    """DNG FixBadPixelsConstant on the (H, W) mosaic ``x``, whose first
    sample is canvas ``origin`` and whose edge is the image edge. On uint16
    the constant is cut to 16 bits and the mean is rounded as the SDK
    rounds it; on float32 the mean is not rounded. A bad sample with no
    good neighbour keeps the constant."""
    height, width = x.shape
    if x.dtype == np.uint16:
        constant &= 0xFFFF
    bad = constant
    out = x.copy()
    for row, col in zip(*np.nonzero(x == bad)):
        green = (row + col + sum(origin) + bayer_phase + (bayer_phase >> 1)) % 2 == 1
        offsets = _GREEN_OFFSETS if green else _RED_BLUE_OFFSETS
        good = [
            x[
                _repeat_edge(row + row_offset, height),
                _repeat_edge(col + col_offset, width),
            ]
            for row_offset, col_offset in offsets
        ]
        good = [value for value in good if value != bad]
        if not good:
            continue
        if x.dtype == np.uint16:
            total = sum(int(value) for value in good)
            count = len(good)
            out[row, col] = (
                (total + 2) >> 2 if count == 4 else (total + count // 2) // count
            )
        else:
            out[row, col] = np.float32(sum(float(value) for value in good) / len(good))
    return out


def _planted(shape, constant, dtype=np.uint16, seed=3):
    """A mosaic that holds ``constant`` only at planted bad samples: about
    one in twenty at random, every corner, samples on and next to each
    edge, an adjacent pair, and a 5x5 block whose centre has no good
    neighbour of either colour."""
    height, width = shape
    rng = np.random.default_rng(seed)
    if dtype == np.uint16:
        x = rng.integers(1, 4095, shape).astype(np.uint16)
    else:
        x = rng.uniform(1.0, 4094.0, shape).astype(np.float32)
    assert not np.any(x == constant)
    bad = rng.random(shape) < 0.05
    planted = [
        (0, 0),
        (0, width - 1),
        (height - 1, 0),
        (height - 1, width - 1),
        (0, 5),
        (1, 8),
        (height - 1, 6),
        (height - 2, 9),
        (4, 0),
        (7, 1),
        (5, width - 1),
        (8, width - 2),
        (10, 10),
        (10, 11),
    ]
    for row, col in planted:
        if row < height and col < width:
            bad[row, col] = True
    bad[20:25, 20:25] = True
    x[bad] = constant
    return x


@pytest.mark.parametrize("bayer_phase", [0, 1, 2, 3])
@pytest.mark.parametrize("constant", [0, 4095])
@pytest.mark.parametrize("shape", [(40, 53), (41, 30)])
def test_uint16_matches_the_reference_exactly(shape, constant, bayer_phase):
    x = _planted(shape, constant)
    got = mi.cfa_fix_bad_pixels_constant(
        Array(x), constant=constant, bayer_phase=bayer_phase
    )
    want = _reference(x, constant, bayer_phase)
    assert not np.array_equal(want, x)
    np.testing.assert_array_equal(got.realize(), want)


@pytest.mark.parametrize("bayer_phase", [0, 1, 2, 3])
def test_float32_is_the_unrounded_mean(bayer_phase):
    x = _planted((40, 53), 0, np.float32)
    got = mi.cfa_fix_bad_pixels_constant(
        Array(x), constant=0, bayer_phase=bayer_phase
    ).realize()
    assert got.dtype == np.float32
    np.testing.assert_allclose(got, _reference(x, 0, bayer_phase), rtol=1e-6)


@pytest.mark.parametrize("dtype", [np.uint16, np.float32])
def test_wide_rows_with_sparse_bad_samples(dtype):
    """Rows of 1000 samples with a few bad ones, some in a run across
    column 64 and some at columns 127 and 999, which the kernel searches a
    block at a time."""
    rng = np.random.default_rng(1)
    x = rng.integers(1, 4095, (60, 1000)).astype(dtype)
    x[rng.random(x.shape) < 0.002] = 0
    x[:, 63:66] = 0
    x[:, 127] = 0
    x[:, 999] = 0
    for bayer_phase in range(4):
        got = mi.cfa_fix_bad_pixels_constant(
            Array(x), constant=0, bayer_phase=bayer_phase
        ).realize()
        np.testing.assert_allclose(got, _reference(x, 0, bayer_phase), rtol=1e-6)


@pytest.mark.parametrize("shape", [(1, 1), (1, 6), (2, 3), (3, 5), (5, 2)])
@pytest.mark.parametrize("bayer_phase", [0, 1, 2, 3])
def test_images_smaller_than_the_neighbourhood(shape, bayer_phase):
    rng = np.random.default_rng(11)
    x = rng.integers(1, 4095, shape).astype(np.uint16)
    x[rng.random(shape) < 0.4] = 0
    x.flat[0] = 0
    got = mi.cfa_fix_bad_pixels_constant(Array(x), constant=0, bayer_phase=bayer_phase)
    np.testing.assert_array_equal(got.realize(), _reference(x, 0, bayer_phase))


def test_the_phase_follows_the_spec():
    """At phase 0 the first sample is red: a bad red sample takes the
    samples 2 away on its row and column, and a bad green one its
    diagonal neighbours."""
    x = np.arange(1, 101, dtype=np.uint16).reshape(10, 10)
    x[4, 4] = 0
    x[4, 7] = 0
    got = mi.cfa_fix_bad_pixels_constant(Array(x), constant=0, bayer_phase=0).realize()
    assert got[4, 4] == (x[2, 4] + x[6, 4] + x[4, 2] + x[4, 6] + 2) // 4
    assert got[4, 7] == (x[3, 6] + x[3, 8] + x[5, 6] + x[5, 8] + 2) // 4


def test_a_constant_above_65535_is_cut_to_16_bits_on_uint16():
    x = _planted((40, 53), 4095)
    got = mi.cfa_fix_bad_pixels_constant(Array(x), constant=65536 + 4095, bayer_phase=0)
    np.testing.assert_array_equal(got.realize(), _reference(x, 4095, 0))


_KEYS = [np.s_[3:30, 5:40], np.s_[1:39, 0:52], np.s_[0:7, 2:9]]


@pytest.mark.parametrize("bayer_phase", [0, 1, 2, 3])
@pytest.mark.parametrize("key", _KEYS)
def test_on_a_slice_the_phase_follows_the_parent(key, bayer_phase):
    """A slice keeps its parent's canvas, so the phase counts from the
    parent's first sample. It cannot read past its own edge, so that edge
    is the image edge."""
    x = _planted((40, 53), 0)
    got = mi.cfa_fix_bad_pixels_constant(
        Array(x)[key], constant=0, bayer_phase=bayer_phase
    )
    origin = (key[0].start, key[1].start)
    np.testing.assert_array_equal(
        got.realize(), _reference(x[key], 0, bayer_phase, origin)
    )


@pytest.mark.parametrize("bayer_phase", [0, 1, 2, 3])
@pytest.mark.parametrize("key", _KEYS)
def test_on_a_view_is_the_view_of_the_whole_image(key, bayer_phase):
    """A ``view`` reads its parent's samples past its edge, so the result
    is that window of the op on the parent."""
    x = _planted((40, 53), 0)
    got = mi.cfa_fix_bad_pixels_constant(
        Array(x).view(key), constant=0, bayer_phase=bayer_phase
    )
    np.testing.assert_array_equal(got.realize(), _reference(x, 0, bayer_phase)[key])


@pytest.mark.parametrize("bayer_phase", [0, 1, 2, 3])
def test_a_slice_after_is_the_slice_of_the_result(bayer_phase):
    x = _planted((40, 53), 0)
    key = np.s_[3:30, 5:40]
    fixed = mi.cfa_fix_bad_pixels_constant(
        Array(x), constant=0, bayer_phase=bayer_phase
    )
    got = fixed[key]
    np.testing.assert_array_equal(got.realize(), _reference(x, 0, bayer_phase)[key])


@pytest.mark.parametrize("bayer_phase", [0, 1, 2, 3])
def test_a_wrapped_numpy_crop_is_its_own_image(bayer_phase):
    """A NumPy crop wrapped in ``mi.Array`` starts a canvas of its own."""
    x = _planted((40, 53), 0)
    crop = x[3:30, 5:40]
    got = mi.cfa_fix_bad_pixels_constant(
        Array(crop), constant=0, bayer_phase=bayer_phase
    )
    np.testing.assert_array_equal(got.realize(), _reference(crop, 0, bayer_phase))
