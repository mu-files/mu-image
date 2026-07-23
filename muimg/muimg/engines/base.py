# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files
"""Engine protocol and default-engine registry."""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from ..tensor import Tensor

_default_engine: Optional["Engine"] = None


@runtime_checkable
class Engine(Protocol):
    """Backend that executes contiguous engine-affinity graph segments."""

    @property
    def supported_ops(self) -> frozenset[str]:
        """Op names this engine can execute."""
        ...

    def execute_segment(
        self,
        nodes: List["Tensor"],
        values: Dict[int, np.ndarray],
        outputs: List["Tensor"],
    ) -> None:
        """Run ``nodes``; write ``outputs`` into ``values`` (and any needed intermediates)."""
        ...


def get_default_engine() -> Engine:
    global _default_engine
    if _default_engine is None:
        from .core.engine import CoreEngine

        _default_engine = CoreEngine()
    return _default_engine


def set_default_engine(engine: Engine) -> None:
    global _default_engine
    _default_engine = engine
