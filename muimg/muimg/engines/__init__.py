# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files
"""muimg.engines — compute engines, portable op catalog, and graph execution."""

from .base import Engine, get_default_engine, set_default_engine

__all__ = [
    "Engine",
    "get_default_engine",
    "set_default_engine",
]
