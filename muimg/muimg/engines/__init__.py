# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files
"""muimg.engines — compute engines, portable op catalog, and graph execution."""

from .graph import (
    Engine,
    EngineTiming,
    get_default_engine,
    get_engine_timing,
    set_default_engine,
    set_engine_timing,
)

__all__ = [
    "Engine",
    "EngineTiming",
    "get_default_engine",
    "get_engine_timing",
    "set_default_engine",
    "set_engine_timing",
]
