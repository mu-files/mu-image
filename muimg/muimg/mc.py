# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files
"""mu compute graph (mc) — the public namespace for building compute graphs.

Usage (the top-level ``mucompute`` package re-exports this module)::

    import mucompute as mc

    x = mc.Tensor(cfa)
    x = mc.ea_demosaic(x, cfa_pattern="RGGB")
    x = mc.matrix_3x3(x, matrix=M)
    out = x.realize()

Every op in ``engines/catalog/ops.yaml`` is a callable here (via the
generated ``muimg.engines.ops``). Engines are pluggable backends that
execute the graph; pipeline code calls ``mc.<op>``, not ``engines.*``.
"""

from __future__ import annotations

from .engines import ops as _catalog
from .engines.graph import emit, flush, op
from .engines.ops import *  # noqa: F401,F403 — generated __all__ is the catalog surface
from .tensor import (
    ElementType,
    Tensor,
    TensorMeta,
    full,
    full_like,
    ones,
    ones_like,
    zeros,
    zeros_like,
)

__all__ = [
    "ElementType",
    "Tensor",
    "TensorMeta",
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
