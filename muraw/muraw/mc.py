# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files
"""Compute graph implementation. Public import is ``import muimage as mi``.

The top-level ``muimage`` package re-exports this module::

    import muimage as mi

    x = mi.Array(cfa)
    x = mi.ea_demosaic(x, cfa_pattern="RGGB")
    x = mi.matrix_3x3(x, matrix=M)
    out = x.realize()

Every op in ``engines/catalog/ops.yaml`` is a callable here (via the
generated ``muraw.engines.ops``). Engines are pluggable backends that
execute the graph; pipeline code calls ``mi.<op>``, not ``engines.*``.
"""

from __future__ import annotations

from .engines import ops as _catalog
from .engines.graph import emit, flush, op
from .engines.ops import *  # noqa: F401,F403 — generated __all__ is the catalog surface
from .array import (
    ElementType,
    Array,
    ArrayMeta,
    full,
    full_like,
    ones,
    ones_like,
    zeros,
    zeros_like,
)

__all__ = [
    "ElementType",
    "Array",
    "ArrayMeta",
    "emit",
    "flush",
    "full",
    "full_like",
    "ones",
    "ones_like",
    "op",
    "zeros",
    "zeros_like",
]
__all__ += _catalog.__all__
