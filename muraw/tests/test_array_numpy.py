"""Array rot90 / fliplr / flipud / transpose match the same NumPy calls."""

from __future__ import annotations

import numpy as np
import pytest

from muraw.array import Array, fliplr, flipud, rot90


def _mono() -> np.ndarray:
    return np.arange(5 * 7, dtype=np.float32).reshape(5, 7) + 1.0


def _rgb() -> np.ndarray:
    h, w = 5, 7
    plane = np.arange(h * w, dtype=np.float32).reshape(h, w)
    return np.stack([plane, plane + 100.0, plane + 200.0], axis=-1)


@pytest.mark.parametrize("src", [_mono(), _rgb()], ids=["mono", "rgb"])
@pytest.mark.parametrize("k", [0, 1, 2, 3, 4, -1, 5])
def test_rot90_matches_numpy(src, k):
    np.testing.assert_array_equal(rot90(Array(src), k), np.rot90(src, k))


def test_rot90_rejects_non_spatial_axes():
    t = Array(_mono())
    with pytest.raises(ValueError, match="spatial plane"):
        rot90(t, 1, axes=(1, 0))


@pytest.mark.parametrize("src", [_mono(), _rgb()], ids=["mono", "rgb"])
def test_fliplr_matches_numpy(src):
    np.testing.assert_array_equal(fliplr(Array(src)), np.fliplr(src))


@pytest.mark.parametrize("src", [_mono(), _rgb()], ids=["mono", "rgb"])
def test_flipud_matches_numpy(src):
    np.testing.assert_array_equal(flipud(Array(src)), np.flipud(src))


@pytest.mark.parametrize("src", [_mono(), _rgb()], ids=["mono", "rgb"])
def test_rot90_then_fliplr_matches_numpy(src):
    np.testing.assert_array_equal(fliplr(rot90(Array(src), 1)), np.fliplr(np.rot90(src, 1)))


def test_flips_and_rot180_are_views():
    src = Array(_mono())
    assert fliplr(src)._node is not None and fliplr(src)._node.op == "view"
    assert flipud(src)._node is not None and flipud(src)._node.op == "view"
    assert rot90(src, 2)._node is not None and rot90(src, 2)._node.op == "view"
    quarter = rot90(src, 1)
    assert quarter._node is not None and quarter._node.op == "orientation"
    assert rot90(src, 3)._node is not None and rot90(src, 3)._node.op == "orientation"


def test_flipud_keeps_parent_canvas_reachable():
    src = _mono()
    windowed = Array(src).view(left=1, top=1, width=4, height=3)
    flipped = flipud(windowed)
    extra = flipped.view(left=0, top=-1, width=4, height=4)
    np.testing.assert_array_equal(extra.realize(), src[4:0:-1, 1:5])


def test_fliplr_keeps_parent_canvas_reachable():
    src = _mono()
    windowed = Array(src).view(left=1, top=1, width=4, height=3)
    flipped = fliplr(windowed)
    extra = flipped.view(left=-1, top=0, width=5, height=3)
    np.testing.assert_array_equal(extra.realize(), src[1:4, 5:0:-1])


def test_flipud_fuses_as_span_not_orientation(monkeypatch):
    from muraw.engines.core import _engine_load

    calls: list[dict] = []
    real = _engine_load.execute_graph

    def wrap(graph, in_binds, out_binds, record_ops=False):
        calls.append(graph)
        return real(graph, in_binds, out_binds, record_ops)

    monkeypatch.setattr(_engine_load, "execute_graph", wrap)

    src = _mono()
    out = flipud((Array(src) - 1.0) * 2.0).realize()
    assert [n["op"] for n in calls[0]["nodes"]] == ["sub_scalar", "mul_scalar", "view"]
    np.testing.assert_array_equal(out, np.flipud((src - 1.0) * 2.0))


def test_transpose_and_T_match_numpy_2d():
    src = _mono()
    t = Array(src)
    want = src.T
    np.testing.assert_array_equal(t.transpose(), want)
    np.testing.assert_array_equal(t.T, want)
    np.testing.assert_array_equal(t.transpose(1, 0), want)
    np.testing.assert_array_equal(t.transpose((1, 0)), want)


def test_transpose_spatial_matches_numpy_rgb():
    src = _rgb()
    want = src.transpose(1, 0, 2)
    t = Array(src)
    np.testing.assert_array_equal(t.transpose(), want)
    np.testing.assert_array_equal(t.T, want)
    np.testing.assert_array_equal(t.transpose(1, 0, 2), want)


def test_rot90_rejects_channel_axis():
    t = Array(_rgb())
    with pytest.raises(ValueError, match="spatial plane"):
        rot90(t, 1, axes=(0, 2))


def test_transpose_that_moves_the_channel_axis_not_supported_yet():
    t = Array(_rgb())
    with pytest.raises(NotImplementedError, match="moving the channel axis"):
        t.transpose(2, 1, 0)


def _op_names(t: Array) -> list[str]:
    ops = []
    while t._node is not None:
        ops.append(t._node.op)
        t = t._node.inputs[0]
    return ops


_V = np.arange(6, dtype=np.float32)
_SHAPES = {"1d": _V, "mono": _mono(), "mono1": _mono()[..., None], "rgb": _rgb()}

# Each call takes the module (np or mi), the array, and the array's rank. It
# returns None when the permutation would move the channel axis at that rank.
_SPATIAL_CALLS = [
    ("transpose_2d", lambda m, a, ndim: m.transpose(a, (1, 0)) if ndim == 2 else None),
    ("transpose_3d", lambda m, a, ndim: m.transpose(a, (1, 0, 2)) if ndim == 3 else None),
    ("transpose_negative", lambda m, a, ndim: m.transpose(a, (-2, -3, -1)) if ndim == 3 else None),
    ("transpose_identity", lambda m, a, ndim: m.transpose(a, tuple(range(ndim)))),
    ("transpose_none", lambda m, a, ndim: m.transpose(a) if ndim < 3 else None),
    ("permute_dims", lambda m, a, ndim: m.permute_dims(a, (1, 0, 2)) if ndim == 3 else None),
    ("swapaxes", lambda m, a, ndim: m.swapaxes(a, 0, 1) if ndim >= 2 else None),
    ("swapaxes_negative", lambda m, a, ndim: m.swapaxes(a, -2, 0) if ndim == 2 else None),
    ("swapaxes_same", lambda m, a, ndim: m.swapaxes(a, 0, -ndim)),
    ("moveaxis", lambda m, a, ndim: m.moveaxis(a, 0, 1) if ndim >= 2 else None),
    ("moveaxis_sequence", lambda m, a, ndim: m.moveaxis(a, (0, 1), (1, 0)) if ndim >= 2 else None),
    ("moveaxis_1d", lambda m, a, ndim: m.moveaxis(a, 0, -1) if ndim == 1 else None),
]


@pytest.mark.parametrize("shape", list(_SHAPES), ids=list(_SHAPES))
@pytest.mark.parametrize("call", [c for _, c in _SPATIAL_CALLS], ids=[n for n, _ in _SPATIAL_CALLS])
def test_spatial_permutations_match_numpy(shape, call):
    import muimage as mi

    src = _SHAPES[shape]
    want = call(np, src, src.ndim)
    if want is None:
        pytest.skip("not a spatial permutation for this rank")
    got = call(mi, Array(src), src.ndim)
    assert got.shape == want.shape
    np.testing.assert_array_equal(got.realize(), want)


def test_identity_permutation_returns_the_same_array():
    import muimage as mi

    t = Array(_rgb())
    assert mi.transpose(t, (0, 1, 2)) is t
    assert mi.swapaxes(t, 2, 2) is t
    assert mi.moveaxis(t, -1, 2) is t
    v = Array(_V)
    assert mi.transpose(v) is v


def test_spatial_permutations_emit_one_orientation():
    import muimage as mi

    t = Array(_rgb())
    for got in (
        mi.transpose(t, (1, 0, 2)),
        mi.swapaxes(t, 0, 1),
        mi.moveaxis(t, 1, 0),
        t.transpose(1, 0, 2),
        t.swapaxes(1, 0),
        t.T,
    ):
        assert _op_names(got) == ["orientation"]
        assert got._node.attrs["orientation"] == 5


@pytest.mark.parametrize("src", [_mono(), _rgb()], ids=["mono", "rgb"])
def test_ops_before_and_after_swapaxes_match_numpy(src):
    import muimage as mi

    got = mi.swapaxes(Array(src) * 2.0, 0, 1)[1:4, ::2] * 0.5
    want = np.swapaxes(src * 2.0, 0, 1)[1:4, ::2] * 0.5
    np.testing.assert_array_equal(got.realize(), want)


def test_ndarray_input_is_ingested():
    import muimage as mi

    src = _mono()
    np.testing.assert_array_equal(mi.transpose(src).realize(), src.T)
    rgb = _rgb()
    np.testing.assert_array_equal(rot90(rgb).realize(), np.rot90(rgb))
    np.testing.assert_array_equal(fliplr(rgb).realize(), np.fliplr(rgb))
    np.testing.assert_array_equal(flipud(rgb).realize(), np.flipud(rgb))


@pytest.mark.parametrize(
    "call",
    [
        lambda m, a: m.transpose(a, (0, 0, 1)),
        lambda m, a: m.transpose(a, (0, 1)),
        lambda m, a: m.transpose(a, (0, 1, 3)),
        lambda m, a: m.swapaxes(a, 0, 3),
        lambda m, a: m.moveaxis(a, (0, 1), 2),
        lambda m, a: m.moveaxis(a, 0, 3),
        lambda m, a: m.moveaxis(a, (0, 0), (1, 2)),
    ],
    ids=["repeated", "too_few", "out_of_range", "swap_out_of_range", "count_mismatch", "move_out_of_range", "move_repeated"],
)
def test_invalid_axes_raise_like_numpy(call):
    import muimage as mi

    src = _rgb()
    with pytest.raises(Exception) as want:
        call(np, src)
    with pytest.raises(type(want.value)) as got:
        call(mi, Array(src))
    assert str(got.value) == str(want.value)


def test_1d_transpose_with_bad_axis_raises_like_numpy():
    with pytest.raises(np.exceptions.AxisError):
        Array(_V).transpose(1)


@pytest.mark.parametrize(
    "call",
    [
        lambda m, a: m.transpose(a),
        lambda m, a: m.transpose(a, (2, 0, 1)),
        lambda m, a: m.moveaxis(a, -1, 0),
        lambda m, a: m.swapaxes(a, 1, 2),
        lambda m, a: m.permute_dims(a, (0, 2, 1)),
    ],
    ids=["transpose_none", "transpose_chw", "moveaxis_chw", "swapaxes_hcw", "permute_dims"],
)
def test_moving_the_channel_axis_not_supported_yet(call):
    import muimage as mi

    with pytest.raises(NotImplementedError, match="moving the channel axis"):
        call(mi, Array(_rgb()))


def test_astype_is_numeric_cast():
    src = np.array([[0, 128, 255]], dtype=np.uint8)
    out = Array(src).astype("float32").realize()
    np.testing.assert_array_equal(out, src.astype(np.float32))
    assert out.dtype == np.float32


def test_convert_type_rescales_pixel_range():
    src = np.array([[0, 128, 255]], dtype=np.uint8)
    out = Array(src).convert_type("float32").realize()
    np.testing.assert_allclose(out, np.array([[0.0, 128.0 / 255.0, 1.0]], dtype=np.float32))


def test_convert_type_src_bits_scales_count_valued_float():
    src = np.array([[0.0, 65535.0]], dtype=np.float32)
    out = Array(src).convert_type("float32", src_bits=16).realize()
    np.testing.assert_allclose(out, np.array([[0.0, 1.0]], dtype=np.float32))


def test_array_constructor_aliases_existing_array():
    src = Array(_mono())
    assert Array(src) is src
    lazy = src.astype("uint8")
    assert lazy._node is not None
    assert Array(lazy) is lazy
    assert lazy._data is None


def test_astype_same_dtype_is_identity():
    t = Array(_mono())
    assert t.astype("float32") is t


def test_convert_type_same_dtype_no_attrs_is_identity():
    t = Array(_mono())
    assert t.convert_type("float32") is t
