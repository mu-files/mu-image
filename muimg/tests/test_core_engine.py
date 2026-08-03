"""Python binding smoke tests for muimg.engines.core._engine_load."""

import numpy as np
import pytest


def test_execute_graph_sub_mul():
    from muimg.engines.core import _engine_load

    h = w = 2
    graph = {
        "tensor_descs": [
            {"id": 0, "dtype": "float32", "height": h, "width": w, "channels": 1},
            {"id": 1, "dtype": "float32", "height": h, "width": w, "channels": 1},
            {"id": 2, "dtype": "float32", "height": h, "width": w, "channels": 1},
        ],
        "inputs": [0],
        "outputs": [2],
        "nodes": [
            {
                "id": 0,
                "op": "sub_scalar",
                "inputs": [0],
                "outputs": [1],
                "attrs": {"value": 1.0},
            },
            {
                "id": 1,
                "op": "mul_scalar",
                "inputs": [1],
                "outputs": [2],
                "attrs": {"value": 2.0},
            },
        ],
    }
    inp = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    out = np.full((h, w), -1.0, dtype=np.float32)
    _engine_load.execute_graph(graph, {0: inp}, {2: out})
    np.testing.assert_allclose(out, [[0.0, 2.0], [4.0, 6.0]])


def test_execute_graph_unknown_op():
    from muimg.engines.core import _engine_load

    graph = {
        "tensor_descs": [
            {"id": 0, "dtype": "float32", "height": 1, "width": 1, "channels": 1},
            {"id": 1, "dtype": "float32", "height": 1, "width": 1, "channels": 1},
        ],
        "inputs": [0],
        "outputs": [1],
        "nodes": [
            {
                "id": 0,
                "op": "no_such_op",
                "inputs": [0],
                "outputs": [1],
                "attrs": {},
            }
        ],
    }
    inp = np.zeros((1, 1), dtype=np.float32)
    out = np.zeros((1, 1), dtype=np.float32)
    with pytest.raises(RuntimeError, match="UNKNOWN_OP"):
        _engine_load.execute_graph(graph, {0: inp}, {1: out})


def test_execute_graph_matrix_identity():
    from muimg.engines.core import _engine_load

    eye = np.eye(3, dtype=np.float32).reshape(-1)
    graph = {
        "tensor_descs": [
            {"id": 0, "dtype": "float32", "height": 1, "width": 1, "channels": 3},
            {"id": 1, "dtype": "float32", "height": 1, "width": 1, "channels": 3},
        ],
        "inputs": [0],
        "outputs": [1],
        "nodes": [
            {
                "id": 0,
                "op": "matrix",
                "inputs": [0],
                "outputs": [1],
                "attrs": {"matrix": eye},
            }
        ],
    }
    inp = np.array([[[0.25, 0.5, 0.75]]], dtype=np.float32)
    out = np.zeros_like(inp)
    _engine_load.execute_graph(graph, {0: inp}, {1: out})
    np.testing.assert_allclose(out, inp)
