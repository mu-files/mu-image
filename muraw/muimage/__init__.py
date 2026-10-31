# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files
"""muimage — ``import muimage as mi``.

Ships from the muraw wheel today. When the graph is extracted into a
standalone package, this becomes that package and the import spelling
at call sites does not change::

    import muimage as mi

    x = mi.Array(cfa)
    x = mi.cfa_demosaic(x, cfa_pattern="RGGB")
    x = mi.rgb_matrix_3x3(x, matrix=M)
    out = x.realize()

Every op in ``engines/catalog/ops.yaml`` is a callable here (via the
generated ``muraw.engines.ops``). Engines are pluggable backends that
execute the graph; pipeline code calls ``mi.<op>``, not ``engines.*``.
"""

from __future__ import annotations

# Catalog first, then array. The star import defines every catalog op;
# the array names that follow (pad, tile, take, matmul, …) are wrappers and
# must win. isort would reverse that (I001 is ignored for this file).
from muraw.engines import ops as _catalog
from muraw.engines.graph import emit, flush, op
from muraw.engines.ops import *  # noqa: F401,F403 — generated __all__ is the catalog surface
from muraw.array import (
    Array,
    ArrayLike,
    ArrayMeta,
    add,
    arange,
    broadcast_to,
    ceil,
    composite,
    concatenate,
    divide,
    dstack,
    expand_dims,
    flip,
    fliplr,
    flipud,
    floor,
    full,
    full_like,
    hstack,
    hypot,
    linspace,
    logical_and,
    logical_not,
    logical_or,
    logical_xor,
    matmul,
    meshgrid,
    moveaxis,
    multiply,
    ones,
    ones_like,
    pad,
    permute_dims,
    repeat,
    reshape,
    resize,
    rint,
    rot90,
    round,
    squeeze,
    square,
    sqrt,
    stack,
    subtract,
    swapaxes,
    take,
    tile,
    transpose,
    vstack,
    zeros,
    zeros_like,
)

__all__ = [
    "Array",
    "ArrayLike",
    "ArrayMeta",
    "emit",
    "flush",
    "add",
    "arange",
    "divide",
    "expand_dims",
    "flip",
    "fliplr",
    "flipud",
    "floor",
    "full",
    "full_like",
    "linspace",
    "matmul",
    "meshgrid",
    "moveaxis",
    "multiply",
    "ones",
    "ones_like",
    "op",
    "pad",
    "permute_dims",
    "broadcast_to",
    "ceil",
    "composite",
    "concatenate",
    "dstack",
    "hstack",
    "hypot",
    "logical_and",
    "logical_not",
    "logical_or",
    "logical_xor",
    "repeat",
    "reshape",
    "resize",
    "rint",
    "rot90",
    "round",
    "squeeze",
    "square",
    "sqrt",
    "stack",
    "subtract",
    "swapaxes",
    "take",
    "tile",
    "transpose",
    "vstack",
    "zeros",
    "zeros_like",
]
__all__ += [name for name in _catalog.__all__ if name not in __all__]
