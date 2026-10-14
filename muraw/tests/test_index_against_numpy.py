"""Indexing with generated keys against NumPy.

Every key either gives NumPy's result or raises ``NotImplementedError``.
When NumPy raises, ``mi`` raises the same exception type. Keys are drawn
from a fixed seed: slices with steps, ints, negative and out-of-range
indices, lists, int arrays, boolean masks, ``None``, ``...`` and
``np.ix_`` meshes, on every layout ``mi`` holds.
"""

from __future__ import annotations

import numpy as np
import pytest

import muimage as mi
from muraw.array import Array

LAYOUTS = {
    "hw": ((5, 6), None),
    "hwc": ((4, 5, 3), 2),
    "chw": ((3, 4, 5), 0),
    "stack": ((2, 4, 5, 3), 3),
    "1d": ((7,), None),
}
CASES_PER_LAYOUT = 400


def _index(lib, x, key):
    """``x[key]``."""
    return x[key]


def _slice(rng, size):
    def bound():
        return None if rng.random() < 0.3 else int(rng.integers(-size - 1, size + 2))

    step = [None, 1, 2, 3, -1, -2][rng.integers(6)]
    return slice(bound(), bound(), step)


def _index_list(rng, size):
    count = int(rng.integers(1, 5))
    low = -size - (1 if rng.random() < 0.1 else 0)
    high = size + (1 if rng.random() < 0.1 else 0)
    values = rng.integers(low, high, count)
    return values.tolist() if rng.random() < 0.5 else values


def _mask(rng, size):
    length = size + (int(rng.integers(-1, 2)) if rng.random() < 0.1 else 0)
    return rng.random(max(length, 0)) < 0.6


def _item(rng, size):
    kind = rng.integers(10)
    if kind < 4:
        return _slice(rng, size)
    if kind < 6:
        return int(rng.integers(-size - (rng.random() < 0.1), size + (rng.random() < 0.1)))
    if kind < 8:
        return _index_list(rng, size)
    if kind < 9:
        return _mask(rng, size)
    return slice(None)


def _key(rng, shape):
    """One key for an array of ``shape``."""
    ndim = len(shape)
    items = [_item(rng, size) for size in shape]
    if ndim >= 2 and rng.random() < 0.2:
        count = int(rng.integers(2, ndim + 1))
        first = int(rng.integers(0, ndim - count + 1))
        lists = [
            np.asarray(_index_list(rng, shape[first + k])) for k in range(count)
        ]
        items[first : first + count] = np.ix_(*lists)
    if rng.random() < 0.3:
        items = items[: int(rng.integers(0, ndim + 1))]
    if items and rng.random() < 0.2:
        start = int(rng.integers(0, len(items)))
        stop = int(rng.integers(start, len(items) + 1))
        items[start:stop] = [Ellipsis]
    if rng.random() < 0.15:
        items.insert(int(rng.integers(0, len(items) + 1)), None)
    return tuple(items)


@pytest.mark.parametrize("layout", sorted(LAYOUTS))
def test_index_matches_numpy_or_raises_not_implemented(layout):
    shape, channel_axis = LAYOUTS[layout]
    rng = np.random.default_rng(sorted(LAYOUTS).index(layout))
    x = rng.random(shape).astype(np.float32)
    source = Array(x) if channel_axis is None else Array(x, channel_axis=channel_axis)
    mismatches = []
    for _ in range(CASES_PER_LAYOUT):
        key = _key(rng, shape)
        try:
            expect = _index(np, x, key)
        except Exception as numpy_error:
            try:
                _index(mi, source, key)
            except type(numpy_error):
                continue
            except Exception as error:
                mismatches.append(f"{key!r}: NumPy raises {numpy_error!r}, mi {error!r}")
                continue
            mismatches.append(f"{key!r}: NumPy raises {numpy_error!r}, mi does not")
            continue
        try:
            got = np.asarray(_index(mi, source, key))
        except NotImplementedError:
            continue
        except Exception as error:
            mismatches.append(f"{key!r}: NumPy gives shape {expect.shape}, mi raises {error!r}")
            continue
        if got.shape != expect.shape or not np.array_equal(got, expect):
            mismatches.append(f"{key!r}: NumPy gives shape {expect.shape}, mi {got.shape}")
    assert not mismatches, f"{len(mismatches)} keys differ:\n" + "\n".join(mismatches[:40])
