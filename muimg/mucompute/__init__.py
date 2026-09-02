# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files
"""mu compute graph — ``import mucompute as mc``.

Today this re-exports ``muimg.mc`` from the muimg wheel. When the mc core
is extracted into a standalone package, this becomes that package and the
import spelling at call sites does not change.
"""

from muimg.mc import *  # noqa: F401,F403 — __all__ defined by muimg.mc
from muimg.mc import __all__  # noqa: F401
