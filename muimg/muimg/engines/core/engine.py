# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files
"""CoreEngine — Python Engine adapter that runs segments via libmuimg_core."""

from __future__ import annotations

from typing import Dict, List, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ...tensor import Tensor


class CoreEngine:
    """Python adapter over ``muimg.engines.core._core_engine``."""

    def __init__(self) -> None:
        self._supported_ops: frozenset[str] | None = None

    @property
    def supported_ops(self) -> frozenset[str]:
        if self._supported_ops is None:
            from ..ops import OPS_BY_NAME

            self._supported_ops = frozenset(OPS_BY_NAME)
        return self._supported_ops

    def execute_segment(
        self,
        nodes: List["Tensor"],
        values: Dict[int, np.ndarray],
        outputs: List["Tensor"],
    ) -> None:
        from ...common import PerfTimer
        from ..timing import EngineTiming, get_engine_timing
        from . import _core_engine

        unknown = sorted(
            {
                t._node.op
                for t in nodes
                if t._node is not None and t._node.op not in self.supported_ops
            }
        )
        if unknown:
            raise ValueError(
                f"CoreEngine does not support op(s): {unknown}"
            )

        produced = {id(t) for t in nodes}
        input_tensors: List["Tensor"] = []
        seen_in: set[int] = set()
        for t in nodes:
            assert t._node is not None
            for inp in t._node.inputs:
                iid = id(inp)
                if iid not in produced and iid not in seen_in:
                    input_tensors.append(inp)
                    seen_in.add(iid)

        all_tensors = input_tensors + nodes
        id_of = {id(t): i for i, t in enumerate(all_tensors)}

        tensor_descs = []
        for t in all_tensors:
            m = t.meta
            tensor_descs.append(
                {
                    "id": id_of[id(t)],
                    "dtype": m.dtype.value,
                    "height": m.height,
                    "width": m.width,
                    "channels": m.channels,
                }
            )

        graph_nodes = []
        for t in nodes:
            assert t._node is not None
            graph_nodes.append(
                {
                    "id": len(graph_nodes),
                    "op": t._node.op,
                    "inputs": [id_of[id(inp)] for inp in t._node.inputs],
                    "outputs": [id_of[id(t)]],
                    "attrs": dict(t._node.attrs),
                }
            )

        in_binds: Dict[int, np.ndarray] = {}
        for t in input_tensors:
            tid = id(t)
            if tid not in values:
                raise ValueError(f"missing materialized input for tensor {tid}")
            in_binds[id_of[tid]] = np.ascontiguousarray(values[tid])

        out_binds: Dict[int, np.ndarray] = {}
        for t in outputs:
            # empty, not zeros: kernels fully overwrite; zeros would first-touch
            # ~full-frame pages only to discard them (several ms on R5-sized buffers).
            arr = np.empty(t.meta.shape, dtype=t.meta.dtype.numpy)
            values[id(t)] = arr
            out_binds[id_of[id(t)]] = arr

        graph = {
            "tensor_descs": tensor_descs,
            "inputs": [id_of[id(t)] for t in input_tensors],
            "outputs": [id_of[id(t)] for t in outputs],
            "nodes": graph_nodes,
        }
        # OPS detail attaches under the open graph_compute step (stack top).
        record_ops = (
            get_engine_timing() >= EngineTiming.OPS
            and PerfTimer.current() is not None
        )
        timings = _core_engine.execute_graph(
            graph, in_binds, out_binds, record_ops
        )
        if record_ops and timings:
            timer = PerfTimer.current()
            assert timer is not None
            op_by_id = {n["id"]: n["op"] for n in graph_nodes}
            steps = [
                (
                    f"{op_by_id.get(row['node_id'], f'node_{row['node_id']}')} (engine)",
                    row["exclusive_ns"] * 1e-9,
                )
                for row in timings
            ]
            timer.add_completed_steps(steps)
