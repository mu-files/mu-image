# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files
"""muimage — ``import muimage as mi``.

Ships from the muraw wheel today. When the graph is extracted into a
standalone package, this becomes that package and the import spelling
at call sites does not change::

    import muimage as mi

    x = mi.Array(cfa)
    x = mi.cfa_ea_demosaic(x, cfa_pattern="RGGB")
    x = mi.rgb_matrix_3x3(x, matrix=M)
    out = x.realize()

Every op in ``engines/catalog/ops.yaml`` is a callable here (via the
generated ``muraw.engines.ops``). Engines are pluggable backends that
execute the graph; pipeline code calls ``mi.<op>``, not ``engines.*``.
"""

from __future__ import annotations

from muraw.engines import ops as _catalog
from muraw.engines.graph import emit, flush, op
from muraw.engines.ops import *  # noqa: F401,F403 — generated __all__ is the catalog surface
from muraw.array import (
    ElementType,
    ElementTypeLike,
    Array,
    ArrayLike,
    ArrayMeta,
    arange,
    expand_dims,
    flip,
    fliplr,
    flipud,
    full,
    full_like,
    linspace,
    meshgrid,
    moveaxis,
    ones,
    ones_like,
    permute_dims,
    broadcast_to,
    rot90,
    squeeze,
    swapaxes,
    tile,
    transpose,
    zeros,
    zeros_like,
)

__all__ = [
    "ElementType",
    "ElementTypeLike",
    "Array",
    "ArrayLike",
    "ArrayMeta",
    "emit",
    "flush",
    "arange",
    "expand_dims",
    "flip",
    "fliplr",
    "flipud",
    "full",
    "full_like",
    "linspace",
    "meshgrid",
    "moveaxis",
    "ones",
    "ones_like",
    "op",
    "permute_dims",
    "broadcast_to",
    "rot90",
    "squeeze",
    "swapaxes",
    "tile",
    "transpose",
    "zeros",
    "zeros_like",
]
__all__ += [name for name in _catalog.__all__ if name not in __all__]
