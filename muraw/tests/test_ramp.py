"""arange and linspace, and the indexes that turn (N, C) into an image."""

import numpy as np
import pytest

import muimage as mi
from muraw.array import Array

C0 = np.array([0.0, 0.0, 0.2], dtype=np.float32)
C1 = np.array([1.0, 0.5, 0.0], dtype=np.float32)


def _assert_ramp_close(got, ref):
    """The ramp computes in float32, so allow a few rounding steps of the
    largest sample. An integer dtype can floor one lower than NumPy's."""
    ref = np.asarray(ref)
    assert got.shape == ref.shape
    assert got.dtype == ref.dtype
    if np.issubdtype(got.dtype, np.integer):
        atol = 1.0
    else:
        scale = max(1.0, float(np.max(np.abs(ref.astype(np.float64)))))
        atol = 4 * float(np.finfo(got.dtype).eps) * scale
    np.testing.assert_allclose(got.astype(np.float64), ref.astype(np.float64), rtol=0, atol=atol)


def _numpy_arange(*args):
    """NumPy's float64 arange, stored as float32 like ``mi.arange``."""
    return np.arange(*args).astype(np.float32)


def test_arange_stop_only():
    got = mi.arange(5).realize()
    np.testing.assert_array_equal(got, _numpy_arange(5))
    assert got.shape == (5,)


def test_arange_start_stop_and_negative_step():
    np.testing.assert_array_equal(mi.arange(2, 8).realize(), _numpy_arange(2, 8))
    np.testing.assert_array_equal(mi.arange(5, 0, -1).realize(), _numpy_arange(5, 0, -1))


def test_arange_float_step():
    _assert_ramp_close(mi.arange(0, 1, 0.3).realize(), _numpy_arange(0, 1, 0.3))


def test_arange_length_is_numpy_ceil_rule():
    got = mi.arange(1, 1.3, 0.1).realize()
    assert got.shape == np.arange(1, 1.3, 0.1).shape == (4,)
    _assert_ramp_close(got, _numpy_arange(1, 1.3, 0.1))


def test_arange_matches_numpy_over_many_ranges():
    rng = np.random.default_rng(7)
    for _ in range(200):
        start = float(rng.uniform(-10, 10))
        step = float(rng.choice([-1, 1]) * rng.uniform(0.01, 2))
        stop = start + step * float(rng.uniform(1, 300))
        got = mi.arange(start, stop, step).realize()
        _assert_ramp_close(got, _numpy_arange(start, stop, step))


def test_arange_float16_matches_numpy():
    got = mi.arange(-3, 40, 0.37, dtype="float16").realize()
    _assert_ramp_close(got, np.arange(-3, 40, 0.37, dtype=np.float16))


def test_arange_integer_dtypes_fill_like_numpy():
    for args in ((0, 250, 10.7), (3.9, 200, 7.2), (200, 3, -9.5)):
        for dtype in (np.uint8, np.uint16):
            got = mi.arange(*args, dtype=dtype.__name__).realize()
            np.testing.assert_array_equal(got, np.arange(*args, dtype=dtype))


def test_arange_rejects_empty_and_zero_step_and_a_list_and_complex():
    with pytest.raises(ValueError, match="empty"):
        mi.arange(5, 0)
    with pytest.raises(ZeroDivisionError):
        mi.arange(0, 1, 0)
    with pytest.raises(TypeError):
        mi.arange([0, 1, 2])
    with pytest.raises(TypeError):
        mi.arange(1j)


def test_linspace_scalar_endpoint():
    got = mi.linspace(0, 1, 5).realize()
    assert got.shape == (5,)
    _assert_ramp_close(got, np.linspace(0, 1, 5, dtype=np.float32))
    assert got[-1] == np.float32(1)


def test_linspace_matches_numpy_over_many_ranges():
    rng = np.random.default_rng(11)
    for _ in range(200):
        start, stop = (float(value) for value in rng.uniform(-10, 10, 2))
        num = int(rng.integers(2, 300))
        endpoint = bool(rng.integers(0, 2))
        got = mi.linspace(start, stop, num, endpoint=endpoint).realize()
        ref = np.linspace(start, stop, num, endpoint=endpoint, dtype=np.float32)
        _assert_ramp_close(got, ref)


def test_linspace_matches_numpy_for_float32_inputs():
    rng = np.random.default_rng(13)
    for _ in range(200):
        start = rng.uniform(-10, 10, 3).astype(np.float32)
        stop = rng.uniform(-10, 10, 3).astype(np.float32)
        num = int(rng.integers(2, 300))
        endpoint = bool(rng.integers(0, 2))
        got = mi.linspace(start, stop, num, endpoint=endpoint).realize()
        _assert_ramp_close(got, np.linspace(start, stop, num, endpoint=endpoint))


def test_linspace_excludes_the_endpoint_and_num_one_is_start():
    got = mi.linspace(0, 1, 5, endpoint=False).realize()
    ref = np.linspace(0, 1, 5, endpoint=False, dtype=np.float32)
    _assert_ramp_close(got, ref)
    assert got[-1] != np.float32(1)
    one = mi.linspace(0.25, 1, 1).realize()
    assert one.shape == (1,)
    assert one[0] == np.float32(0.25)


def test_linspace_integer_dtypes_floor_like_numpy():
    for dtype in (np.uint8, np.uint16):
        got = mi.linspace(0, 200, 7, dtype=dtype.__name__).realize()
        _assert_ramp_close(got, np.linspace(0, 200, 7, dtype=dtype))


def test_linspace_colors_match_numpy_and_the_image_indexes():
    got = mi.linspace(C0, C1, 4).realize()
    ref = np.linspace(C0, C1, 4, dtype=np.float32)
    assert got.shape == (4, 3)
    _assert_ramp_close(got, ref)
    assert np.all(got[-1] == C1)

    row = mi.linspace(C0, C1, 6)[None, ...]
    assert row.shape == (1, 6, 3)
    _assert_ramp_close(row.realize(), np.linspace(C0, C1, 6, dtype=np.float32)[None, ...])
    column = mi.linspace(C0, C1, 5)[:, None, :]
    assert column.shape == (5, 1, 3)
    _assert_ramp_close(column.realize(), np.linspace(C0, C1, 5, dtype=np.float32)[:, None, :])


def test_linspace_broadcasts_a_scalar_and_keeps_a_length_one_sequence():
    got = mi.linspace(0, C1, 4).realize()
    _assert_ramp_close(got, np.linspace(0, C1, 4, dtype=np.float32))
    seq = mi.linspace([0.0], [1.0], 5).realize()
    assert seq.shape == (5, 1)
    _assert_ramp_close(seq, np.linspace([0.0], [1.0], 5, dtype=np.float32))


def test_linspace_rejects_axis_rank_and_a_short_count():
    with pytest.raises(ValueError, match="axis"):
        mi.linspace(0, 1, 4, axis=-1)
    with pytest.raises(ValueError, match="1D"):
        mi.linspace([[0, 1]], [[1, 2]], 3)
    with pytest.raises(ValueError, match="num"):
        mi.linspace(0, 1, 0)


def test_inserted_axis_on_a_concrete_array_matches_numpy():
    src = np.arange(12, dtype=np.float32).reshape(3, 4)
    bound = Array(src)
    np.testing.assert_array_equal(bound[None, ...].realize(), src[None, ...])
    np.testing.assert_array_equal(bound[:, None, :].realize(), src[:, None, :])
    np.testing.assert_array_equal(bound[None, 1:].realize(), src[None, 1:])
