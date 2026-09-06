# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files
"""muimage — ``import muimage as mi``.

Today this re-exports ``muraw.mc`` from the muraw wheel. When the graph
is extracted into a standalone package, this becomes that package and
the import spelling at call sites does not change.
"""

from muraw.mc import *  # noqa: F401,F403 — __all__ defined by muraw.mc
from muraw.mc import __all__  # noqa: F401
