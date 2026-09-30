"""Array rot90 / fliplr / flipud / transpose match the same NumPy calls."""

from __future__ import annotations

import itertools

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
    t = Array(src)
    assert t.T.shape == (3, 7, 5)
    np.testing.assert_array_equal(t.transpose(), src.T)
    np.testing.assert_array_equal(t.T, src.T)
    np.testing.assert_array_equal(t.transpose(1, 0, 2), src.transpose(1, 0, 2))


def test_rot90_rejects_channel_axis():
    t = Array(_rgb())
    with pytest.raises(ValueError, match="spatial plane"):
        rot90(t, 1, axes=(0, 2))


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
    ("transpose_none", lambda m, a, ndim: m.transpose(a)),
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
    ):
        assert _op_names(got) == ["orientation"]
        assert got._node.attrs["orientation"] == 5


@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
def test_each_permutation_emits_one_node_and_realizes_c_contiguous(order):
    import muimage as mi

    src = _rgb()
    t = Array(src)
    got = mi.transpose(t, order)
    if order == (0, 1, 2):
        assert got is t
    elif order == (1, 0, 2):
        assert _op_names(got) == ["orientation"]
    else:
        assert _op_names(got) == ["transpose"]
        assert list(got._node.attrs["axes"]) == list(order)
    assert got.meta.channel_axis == order.index(2)
    out = got.realize()
    assert out.flags.c_contiguous
    np.testing.assert_array_equal(out, src.transpose(order))


@pytest.mark.parametrize("order", list(itertools.permutations(range(3))))
def test_a_permutation_and_its_inverse_emit_nothing(order):
    import muimage as mi

    lazy = Array(_rgb()) * 2.0
    inverse = tuple(int(axis) for axis in np.argsort(order))
    assert mi.transpose(mi.transpose(lazy, order), inverse) is lazy


def test_permutations_compose_into_one_node():
    import muimage as mi

    t = Array(_rgb())
    got = mi.swapaxes(mi.moveaxis(t, -1, 0), 1, 2)
    assert _op_names(got) == ["transpose"]
    assert list(got._node.attrs["axes"]) == [2, 1, 0]
    np.testing.assert_array_equal(got.realize(), np.swapaxes(np.moveaxis(_rgb(), -1, 0), 1, 2))


def test_channel_axis_follows_the_channels():
    import muimage as mi

    chw = np.ascontiguousarray(np.moveaxis(_rgb(), -1, 0))
    bound = Array(chw, channel_axis=0)
    assert bound._node is None and np.shares_memory(bound._data, chw)
    assert bound.meta.channel_axis == 0
    assert (bound.meta.height, bound.meta.width, bound.meta.channels) == (5, 7, 3)
    moved = mi.moveaxis(Array(_rgb()), -1, 0)
    assert moved.meta.channel_axis == 0
    assert mi.moveaxis(moved, 1, 2).meta.channel_axis == 0
    assert mi.moveaxis(Array(_rgb()), 1, 2).meta.channel_axis == 1


def test_channel_axis_keyword_is_checked():
    chw = np.zeros((3, 5, 7), np.float32)
    with pytest.raises(np.exceptions.AxisError, match="channel_axis"):
        Array(chw, channel_axis=3)
    with pytest.raises(ValueError, match="only valid when ingesting"):
        Array(Array(chw), channel_axis=0)
    assert Array(_mono(), channel_axis=0).meta.channel_axis is None


def test_swapping_rows_and_columns_of_a_chw_array_runs_per_plane():
    import muimage as mi

    chw = np.ascontiguousarray(np.moveaxis(_rgb(), -1, 0))
    got = mi.swapaxes(Array(chw, channel_axis=0), 1, 2)
    assert _op_names(got) == ["orientation"]
    np.testing.assert_array_equal((got * 2.0).realize(), np.swapaxes(chw, 1, 2) * 2.0)



@pytest.mark.parametrize("src", [_mono(), _rgb()], ids=["mono", "rgb"])
def test_ops_before_and_after_swapaxes_match_numpy(src):
    import muimage as mi

    got = mi.swapaxes(Array(src) * 2.0, 0, 1)[1:4, ::2] * 0.5
    want = np.swapaxes(src * 2.0, 0, 1)[1:4, ::2] * 0.5
    np.testing.assert_array_equal(got.realize(), want)


_SIZE_ONE_CASES = [
    ("expand_1d_front", _V, lambda m, a: m.expand_dims(a, 0)),
    ("expand_1d_back", _V, lambda m, a: m.expand_dims(a, 1)),
    ("expand_1d_negative", _V, lambda m, a: m.expand_dims(a, -1)),
    ("expand_1d_two", _V, lambda m, a: m.expand_dims(a, (0, 1))),
    ("expand_mono_channel", _mono(), lambda m, a: m.expand_dims(a, -1)),
    ("expand_mono_front", _mono(), lambda m, a: m.expand_dims(a, 0)),
    ("expand_mono_middle", _mono(), lambda m, a: m.expand_dims(a, 1)),
    ("squeeze_row", _V[None, :], lambda m, a: m.squeeze(a)),
    ("squeeze_column", _V[:, None], lambda m, a: m.squeeze(a, axis=1)),
    ("squeeze_channel", _mono()[..., None], lambda m, a: m.squeeze(a, -1)),
    ("squeeze_pixel_rgb", _rgb()[:1, :1], lambda m, a: m.squeeze(a)),
    ("squeeze_column_rgb", _rgb()[:, :1], lambda m, a: m.squeeze(a, 1)),
    ("squeeze_nothing", _mono(), lambda m, a: m.squeeze(a)),
]


@pytest.mark.parametrize("src, call", [(s, c) for _, s, c in _SIZE_ONE_CASES], ids=[n for n, _, _ in _SIZE_ONE_CASES])
@pytest.mark.parametrize("lazy", [False, True], ids=["source", "lazy"])
def test_expand_dims_and_squeeze_match_numpy(src, call, lazy):
    import muimage as mi

    want = call(np, src * 2.0 if lazy else src)
    got = call(mi, Array(src) * 2.0 if lazy else Array(src))
    assert got.shape == want.shape
    np.testing.assert_array_equal(got.realize(), want)


def test_squeeze_method_matches_numpy():
    src = _mono()[..., None]
    np.testing.assert_array_equal(Array(src).squeeze().realize(), src.squeeze())


def test_size_one_axis_on_the_same_buffer_sizes_keeps_the_node():
    import muimage as mi

    mono = Array(_mono()) * 2.0
    assert mi.expand_dims(mono, -1)._node is mono._node
    assert mi.squeeze(mi.expand_dims(mono, -1))._node is mono._node
    line = Array(_V) * 2.0
    assert mi.expand_dims(line, 0)._node is line._node
    assert mi.squeeze(mono) is mono


@pytest.mark.parametrize("lazy", [False, True], ids=["source", "lazy"])
def test_the_axis_expand_dims_adds_to_a_2d_array_is_the_channel_axis(lazy):
    import muimage as mi

    mono = Array(_mono()) * 2.0 if lazy else Array(_mono())
    for axis in (0, 1, 2):
        expanded = mi.expand_dims(mono, axis)
        assert expanded.meta.channel_axis == axis
        assert (expanded.meta.height, expanded.meta.width) == (5, 7)
        assert mi.squeeze(expanded, axis).meta.channel_axis is None
    assert Array(_rgb())[:, :, 0].meta.channel_axis is None


def test_squeeze_of_a_source_column_shares_memory():
    import muimage as mi

    src = _V[:, None].copy()
    got = mi.squeeze(src)
    assert got._node is None
    assert np.shares_memory(got._data, src)


@pytest.mark.parametrize(
    "src, call",
    [
        (_mono(), lambda m, a: m.squeeze(a, 0)),
        (_mono(), lambda m, a: m.expand_dims(a, 3)),
        (_mono(), lambda m, a: m.expand_dims(a, (0, 0))),
        (_rgb(), lambda m, a: m.squeeze(a, 5)),
        (_rgb()[:1, :1], lambda m, a: m.squeeze(a, (0, 0))),
        (_mono()[:1], lambda m, a: m.squeeze(a, 1.0)),
    ],
    ids=[
        "squeeze_not_one",
        "expand_out_of_range",
        "expand_repeated",
        "squeeze_out_of_range",
        "squeeze_repeated",
        "squeeze_float_axis",
    ],
)
def test_expand_dims_and_squeeze_errors_match_numpy(src, call):
    import muimage as mi

    with pytest.raises(Exception) as want:
        call(np, src)
    with pytest.raises(type(want.value)) as got:
        call(mi, Array(src))
    assert str(got.value) == str(want.value)


def test_squeeze_to_no_axes_and_expand_past_three_axes_raise():
    import muimage as mi

    with pytest.raises(ValueError, match="would leave no axes"):
        mi.squeeze(np.zeros((1, 1), np.float32))
    with pytest.raises(ValueError, match="would leave no axes"):
        mi.squeeze(np.zeros((1, 1, 1), np.float32))
    with pytest.raises(ValueError, match="at most 3"):
        mi.expand_dims(_rgb(), 0)


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
def test_moving_the_channel_axis_matches_numpy(call):
    import muimage as mi

    want = call(np, _rgb())
    got = call(mi, Array(_rgb()))
    assert got.shape == want.shape
    out = got.realize()
    assert out.flags.c_contiguous
    np.testing.assert_array_equal(out, want)


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
