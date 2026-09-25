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
    ArrayMeta,
    full,
    full_like,
    ones,
    ones_like,
    tile,
    zeros,
    zeros_like,
)

__all__ = [
    "ElementType",
    "ElementTypeLike",
    "Array",
    "ArrayMeta",
    "emit",
    "flush",
    "full",
    "full_like",
    "ones",
    "ones_like",
    "op",
    "tile",
    "zeros",
    "zeros_like",
]
__all__ += _catalog.__all__
