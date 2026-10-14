# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files
"""CoreEngine — Python Engine adapter over the abi3 native extension."""

from __future__ import annotations

from typing import Any, Dict, List, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ...array import Array
    from ..graph import OpNode


# The engine reads an array whose channels are on axis 0 as a stack of
# frames of one channel each, so a node whose input 0 is such an array
# names its channels by these frame attrs instead.
_FRAME_ATTRS = {
    "dest_channels": "dest_frames",
    "dest_channels_source": "dest_frames_source",
    "src_channels": "src_frames",
    "channel_reps": "frame_reps",
    "channel_before": "frame_before",
    "channel_after": "frame_after",
}
_SCALAR_OPS = frozenset({"add_scalar", "sub_scalar", "mul_scalar", "div_scalar"})


def _engine_attrs(node: "OpNode") -> Dict[str, Any]:
    """The attrs of ``node`` as the engine reads them."""
    attrs = dict(node.attrs)
    if not node.inputs or node.inputs[0].meta.channel_axis != 0:
        return attrs
    lowered = {_FRAME_ATTRS.get(key, key): value for key, value in attrs.items()}
    if "frame_reps" in lowered:
        lowered["channel_reps"] = 1
    constants = attrs.get("constant_values")
    if node.op == "pad" and constants is not None and len(constants) == 6:
        # The engine reads the frame fills after the channel fills.
        lowered["constant_values"] = np.concatenate(
            [constants[:4], np.zeros(2, constants.dtype), constants[4:]]
        )
    if node.op in _SCALAR_OPS and len(attrs["value"]) > 1:
        lowered["per_frame"] = True
    return lowered


class CoreEngine:
    """Python adapter over the abi3 ``_core_engine`` extension (via ``_engine_load``)."""

    def __init__(self) -> None:
        self._supported_ops: frozenset[str] | None = None

    @property
    def supported_ops(self) -> frozenset[str]:
        if self._supported_ops is None:
            from ..ops import OPS_BY_NAME

            self._supported_ops = frozenset(OPS_BY_NAME)
        return self._supported_ops

    def execute(
        self,
        nodes: List["Array"],
        values: Dict[int, np.ndarray],
        root: "Array",
        output: np.ndarray | None = None,
    ) -> None:
        """Run ``nodes``; write ``root`` into ``values``.

        Every array in ``values`` has its ``meta.buffer_shape``, so a 1D
        array is a one-row ``(1, N)`` buffer.

        ``output``, when given, is the buffer for ``root``. It already has
        ``root``'s buffer shape, dtype, and a layout this engine can bind.
        """
        from ...common import PerfTimer
        from ..graph import EngineTiming, check_engine_layout, get_engine_timing
        from . import _engine_load

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
        input_arrays: List["Array"] = []
        seen_in: set[int] = set()
        for t in nodes:
            assert t._node is not None
            for inp in t._node.inputs:
                iid = id(inp)
                if iid not in produced and iid not in seen_in:
                    input_arrays.append(inp)
                    seen_in.add(iid)

        all_arrays = input_arrays + nodes
        id_of = {id(t): i for i, t in enumerate(all_arrays)}

        root_id = id(root)
        in_binds: Dict[int, np.ndarray] = {}
        for t in input_arrays:
            tid = id(t)
            if tid not in values:
                raise ValueError(f"missing materialized input for array {tid}")
            in_binds[id_of[tid]] = values[tid]
        for t in nodes:
            tid = id(t)
            if tid in values and tid != root_id:
                in_binds[id_of[tid]] = values[tid]

        # Kernels write every dest pixel, so do not spend a pass zeroing arr.
        if output is None:
            out = np.empty(root.meta.buffer_shape, dtype=root.meta.dtype)
        else:
            out = output
        values[root_id] = out

        array_descs = []
        for t in all_arrays:
            m = t.meta
            check_engine_layout(m)
            if m.size == 0:
                raise NotImplementedError(
                    f"CoreEngine cannot run an array of shape {m.shape}; an empty "
                    "array has no node to run"
                )
            origin_row, origin_col = m.origin
            # The engine reads a CHW array as frames on axis 0 with one
            # channel each, a 4-axis stack as frames on axis 0, and any
            # other array with its channels last.
            desc = {
                "id": id_of[id(t)],
                "dtype": m.dtype.name,
                "shape": list(m.buffer_shape),
                "frame_axis": 0 if m.channel_axis == 0 or m.ndim == 4 else -1,
                "origin_y": int(origin_row),
                "origin_x": int(origin_col),
                "canvas_x0": int(m.canvas[0]),
                "canvas_y0": int(m.canvas[1]),
                "canvas_width": int(m.canvas[2]),
                "canvas_height": int(m.canvas[3]),
            }
            array_descs.append(desc)

        graph_nodes = []
        for t in nodes:
            assert t._node is not None
            graph_nodes.append(
                {
                    "id": len(graph_nodes),
                    "op": t._node.op,
                    "inputs": [id_of[id(inp)] for inp in t._node.inputs],
                    "outputs": [id_of[id(t)]],
                    "attrs": _engine_attrs(t._node),
                }
            )

        graph = {
            "array_descs": array_descs,
            "inputs": [id_of[id(t)] for t in input_arrays],
            "output": id_of[root_id],
            "nodes": graph_nodes,
        }
        # OPS detail attaches under the open graph_compute step (stack top).
        record_ops = (
            get_engine_timing() >= EngineTiming.OPS
            and PerfTimer.current() is not None
        )
        timings = _engine_load.execute_graph(
            graph, in_binds, out, record_ops
        )
        if record_ops and timings:
            timer = PerfTimer.current()
            assert timer is not None
            op_by_id = {n["id"]: n["op"] for n in graph_nodes}
            steps = [
                (
                    _timing_row_name(row, op_by_id),
                    row["exclusive_ns"] * 1e-9,
                )
                for row in timings
            ]
            timer.add_completed_steps(steps)


def _timing_row_name(row: Dict, op_by_id: Dict[int, str]) -> str:
    """Name for ``--timing ops``: the label the op's prepare chose, else the
    op name, then ``(engine)``. It is ``(engine, not run)`` when the engine
    never called the op's kernel, and ``(engine, not sampled)`` when the
    kernel ran but none of its calls was timed, so its time is unknown."""
    if "label" in row:
        name = row["label"]
    else:
        name = op_by_id.get(row["node_id"], f"node_{row['node_id']}")
    if row.get("calls") == 0:
        return f"{name} (engine, not run)"
    if row.get("timed_calls") == 0:
        return f"{name} (engine, not sampled)"
    return f"{name} (engine)"
