# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files
"""Timing policy for graph / engine compute (owned by engines, not PerfTimer)."""

from __future__ import annotations

from enum import IntEnum


class EngineTiming(IntEnum):
    """How much detail ``graph.compute`` / execute_segment should record."""

    OFF = 0
    SEGMENTS = 1  # one row per python op or engine execute_segment
    OPS = 2  # + per-op rows inside an engine segment


engine_timing: EngineTiming = EngineTiming.OFF


def get_engine_timing() -> EngineTiming:
    return engine_timing


def set_engine_timing(level: EngineTiming | int | str) -> None:
    """Set engine compute timing detail."""
    global engine_timing
    if isinstance(level, EngineTiming):
        engine_timing = level
    elif isinstance(level, int):
        engine_timing = EngineTiming(level)
    else:
        engine_timing = EngineTiming[str(level).strip().upper()]
