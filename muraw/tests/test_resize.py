"""resize: ``mi.resize`` against a NumPy reference of its definition.

NumPy has no resize, so each check compares ``mi.resize`` with
``_reference``, which computes the definition with ``np.pad`` and
``np.take``. Where OpenCV is installed, ``nearest`` is also compared with
``cv2.INTER_NEAREST_EXACT``.
"""

from __future__ import annotations

import numpy as np
import pytest

import muimage as mi
from muraw.array import Array

DTYPES = (np.uint8, np.uint16, np.float16, np.float32)
MODES = ("reflect", "symmetric", "edge", "constant")


def _image(shape, dtype=np.float32):
    rng = np.random.default_rng(11)
    if np.dtype(dtype).kind == "f":
        return rng.random(shape).astype(dtype)
    return rng.integers(0, np.iinfo(dtype).max, shape, endpoint=True).astype(dtype)


def _source_indices(input_len, output_len, scale, mode, first, readable_len):
    """The index into the ``readable_len`` readable samples each output
    index on one axis reads, or -1 where ``constant`` fills. The input is
    ``input_len`` samples from readable sample ``first``."""
    dest = np.arange(output_len)
    if scale is None:
        index = (2 * dest + 1) * input_len // (2 * output_len)
    else:
        index = np.floor((dest + 0.5) / scale).astype(np.intp)
    index = index + first
    before = max(0, -int(index.min()))
    after = max(0, int(index.max()) - readable_len + 1)
    kwargs = {"constant_values": -1} if mode == "constant" else {}
    padded = np.pad(np.arange(readable_len), (before, after), mode, **kwargs)
    return padded[index + before]


def _reference(
    readable,
    size,
    scale=None,
    mode="reflect",
    constant_values=0,
    channel_axis=None,
    window=None,
):
    """Nearest resize of the window ``(top, left, rows, cols)`` of
    ``readable`` to ``size``, reading ``readable`` past the window and
    padding past ``readable``. Without ``window`` the window is all of
    ``readable``."""
    rows_axis, cols_axis = (1, 2) if channel_axis == 0 else (0, 1)
    if window is None:
        window = (0, 0, readable.shape[rows_axis], readable.shape[cols_axis])
    top, left, in_rows, in_cols = window
    scale_rows, scale_cols = (None, None) if scale is None else scale
    rows = _source_indices(
        in_rows, size[0], scale_rows, mode, top, readable.shape[rows_axis]
    )
    cols = _source_indices(
        in_cols, size[1], scale_cols, mode, left, readable.shape[cols_axis]
    )
    out = np.take(
        np.take(readable, np.maximum(rows, 0), rows_axis),
        np.maximum(cols, 0),
        cols_axis,
    )
    fill_mask = (rows[:, None] < 0) | (cols[None, :] < 0)
    if channel_axis == 0:
        fill = np.broadcast_to(np.ravel(constant_values), (out.shape[0],))
        out[:, fill_mask] = fill[:, None]
    else:
        out[fill_mask] = constant_values
    return out


@pytest.mark.parametrize("shape", [(8, 10, 3), (8, 10), (37, 53, 4)])
@pytest.mark.parametrize("size", [(3, 4), (13, 17), (3, 25), (1, 1), (64, 7)])
def test_nearest_with_size_matches_reference(shape, size):
    x = _image(shape)
    got = mi.resize(Array(x), size, interpolation="nearest").realize()
    np.testing.assert_array_equal(got, _reference(x, size))


@pytest.mark.parametrize(
    "scale, expected_size",
    [(0.5, (5, 3)), ((1.5, 2.25), (15, 15)), (3, (30, 21)), ((0.3, 1), (3, 7))],
)
def test_nearest_with_scale_alone_takes_the_integer_part(scale, expected_size):
    x = _image((10, 7, 3))
    got = mi.resize(Array(x), scale=scale, interpolation="nearest").realize()
    assert got.shape[:2] == expected_size
    np.testing.assert_array_equal(got, _reference(x, expected_size))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "size, scale", [((9, 14), (0.5, 1.5)), ((12, 3), (2.0, 0.25)), ((4, 6), (0.3, 0.7))]
)
def test_nearest_past_the_edge_follows_the_pad_mode(mode, size, scale):
    x = _image((5, 6, 3))
    got = mi.resize(
        Array(x),
        size,
        scale=scale,
        interpolation="nearest",
        mode=mode,
        constant_values=0.25,
    ).realize()
    np.testing.assert_array_equal(
        got, _reference(x, size, scale, mode, constant_values=0.25)
    )


def test_constant_fills_one_value_per_channel():
    x = _image((5, 6, 3))
    fill = [0.1, 0.2, 0.3]
    got = mi.resize(
        Array(x),
        (9, 14),
        scale=(0.5, 1.5),
        interpolation="nearest",
        mode="constant",
        constant_values=fill,
    ).realize()
    np.testing.assert_array_equal(
        got,
        _reference(
            x, (9, 14), (0.5, 1.5), "constant", constant_values=np.float32(fill)
        ),
    )


@pytest.mark.parametrize("mode", MODES)
def test_channels_first(mode):
    x = _image((3, 5, 6))
    fill = [0.1, 0.2, 0.3]
    got = mi.resize(
        Array(x, channel_axis=0),
        (9, 14),
        scale=(0.5, 1.5),
        interpolation="nearest",
        mode=mode,
        constant_values=fill,
    ).realize()
    expected = _reference(
        x, (9, 14), (0.5, 1.5), mode, constant_values=np.float32(fill), channel_axis=0
    )
    np.testing.assert_array_equal(got, expected)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("mode", ["reflect", "constant"])
def test_every_dtype_copies_its_samples(dtype, mode):
    x = _image((9, 11, 3), dtype)
    got = mi.resize(
        Array(x),
        (7, 20),
        scale=(0.7, 1.7),
        interpolation="nearest",
        mode=mode,
        constant_values=7,
    ).realize()
    assert got.dtype == np.dtype(dtype)
    np.testing.assert_array_equal(
        got, _reference(x, (7, 20), (0.7, 1.7), mode, constant_values=7)
    )


def test_a_view_reads_its_parent_past_its_edge_and_pads_past_the_canvas():
    parent = _image((12, 15, 3))
    window = (4, 5, 3, 4)
    top, left, rows, cols = window
    for mode in MODES:
        view = Array(parent).view(left=left, top=top, width=cols, height=rows)
        got = mi.resize(
            view, (9, 13), scale=(0.4, 0.5), interpolation="nearest", mode=mode
        ).realize()
        expected = _reference(parent, (9, 13), (0.4, 0.5), mode, window=window)
        np.testing.assert_array_equal(got, expected, err_msg=mode)


@pytest.mark.parametrize("mode", MODES)
def test_a_slice_resizes_as_a_copy_of_it(mode):
    x = _image((12, 15, 3))
    sliced = mi.resize(
        Array(x)[4:7, 5:9],
        (9, 13),
        scale=(0.4, 0.5),
        interpolation="nearest",
        mode=mode,
    ).realize()
    copied = mi.resize(
        Array(x[4:7, 5:9].copy()),
        (9, 13),
        scale=(0.4, 0.5),
        interpolation="nearest",
        mode=mode,
    ).realize()
    np.testing.assert_array_equal(sliced, copied)


def test_area_on_growing_axes_is_nearest():
    x = _image((6, 7, 3))
    area = mi.resize(Array(x), (13, 7), interpolation="area").realize()
    nearest = mi.resize(Array(x), (13, 7), interpolation="nearest").realize()
    np.testing.assert_array_equal(area, nearest)


@pytest.mark.parametrize("interpolation", ["auto", "nearest", "area", "bilinear"])
@pytest.mark.parametrize(
    "size, scale", [((6, 7), None), (None, 1), ((6, 7), 1.0), ((6, 7), (1, 1))]
)
def test_a_copy_is_the_input_itself(interpolation, size, scale):
    x = Array(_image((6, 7, 3)))
    assert mi.resize(x, size, scale=scale, interpolation=interpolation) is x


def test_a_copy_checks_its_arguments():
    x = Array(_image((6, 7, 3)))
    with pytest.raises(ValueError):
        mi.resize(x, (6, 7), interpolation="cubic")
    with pytest.raises(ValueError):
        mi.resize(x, (6, 7), mode="wrap")


def test_the_same_size_at_another_scale_is_not_a_copy():
    x = _image((6, 7, 3))
    got = mi.resize(Array(x), (6, 7), scale=0.5, interpolation="nearest").realize()
    np.testing.assert_array_equal(got, _reference(x, (6, 7), (0.5, 0.5)))
    assert not np.array_equal(got, x)


@pytest.mark.parametrize(
    "size, scale, interpolation",
    [
        ((12, 14), None, "bilinear"),
        ((6, 7), (1.0, 0.5), "bilinear"),
        ((3, 7), None, "area"),
        ((12, 7), None, "auto"),
        ((3, 7), None, "auto"),
        ((6, 7), (1.0, 0.5), "auto"),
    ],
)
def test_modes_that_need_a_resize_kernel_are_not_implemented(
    size, scale, interpolation
):
    x = Array(_image((6, 7)))
    with pytest.raises(NotImplementedError, match="interpolation='nearest'"):
        mi.resize(x, size, scale=scale, interpolation=interpolation)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"size": (0, 4)},
        {"size": (4, -1)},
        {"size": (4, 4, 4)},
        {"scale": 0},
        {"scale": (1.0, -2.0)},
        {"scale": float("inf")},
        {"scale": 0.01},
        {"size": (4, 4), "mode": "constant", "constant_values": [1, 2]},
    ],
)
def test_invalid_arguments_raise_value_error(kwargs):
    x = Array(_image((6, 7, 3)))
    with pytest.raises(ValueError):
        mi.resize(x, interpolation="nearest", **kwargs)


def test_size_or_scale_is_required():
    with pytest.raises(TypeError):
        mi.resize(Array(_image((6, 7))))


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize("size", [(3, 4), (13, 17), (5, 31), (100, 3)])
def test_nearest_matches_opencv_nearest_exact(dtype, size):
    cv2 = pytest.importorskip("cv2")
    x = _image((37, 29, 3), dtype)
    got = mi.resize(Array(x), size, interpolation="nearest").realize()
    expected = cv2.resize(x, (size[1], size[0]), interpolation=cv2.INTER_NEAREST_EXACT)
    np.testing.assert_array_equal(got, expected)
