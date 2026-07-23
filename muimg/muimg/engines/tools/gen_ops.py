#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files
"""Generate muimg.engines.ops from the portable ops.yaml catalog.

Reads:
  muimg/engines/catalog/ops.yaml

Writes:
  muimg/engines/ops.py

Run from anywhere (stdlib only, no PyYAML):
  python -m muimg.engines.tools.gen_ops

Catalog YAML is a constrained subset: top-level maps, list-of-maps,
scalar string/int values, ``#`` comments.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ENGINES_DIR = Path(__file__).resolve().parents[1]
YAML_PATH = ENGINES_DIR / "catalog" / "ops.yaml"
OPS_OUT = ENGINES_DIR / "ops.py"


def _parse_scalar(raw: str):
    raw = raw.strip()
    if not raw:
        return ""
    if (raw[0] == raw[-1]) and raw[0] in "\"'":
        return raw[1:-1]
    if raw == "true":
        return True
    if raw == "false":
        return False
    if re.fullmatch(r"-?\d+", raw):
        return int(raw)
    return raw


def load_ops_yaml(path: Path) -> dict:
    """Indentation-based loader for the catalog YAML subset."""
    lines = []
    for lineno, raw in enumerate(path.read_text().splitlines(), 1):
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        content = raw.strip()
        if "#" in content and '"' not in content and "'" not in content:
            content = content.split("#", 1)[0].rstrip()
        lines.append((lineno, indent, content))

    def parse_block(start: int, min_indent: int):
        if start >= len(lines):
            return None, start
        _, ind0, c0 = lines[start]
        if ind0 <= min_indent:
            return None, start
        if c0.startswith("- "):
            return parse_seq(start, ind0)
        return parse_map(start, ind0)

    def parse_map(start: int, map_indent: int):
        result = {}
        i = start
        while i < len(lines):
            lineno, ind, content = lines[i]
            if ind < map_indent:
                break
            if ind > map_indent:
                raise ValueError(f"line {lineno}: unexpected indent in map")
            if content.startswith("- "):
                break
            if ":" not in content:
                raise ValueError(f"line {lineno}: expected key: {content!r}")
            k, _, v = content.partition(":")
            k = k.strip()
            v = v.strip()
            i += 1
            if v == "":
                child, i = parse_block(i, map_indent)
                result[k] = child if child is not None else None
            else:
                result[k] = _parse_scalar(v)
        return result, i

    def parse_seq(start: int, seq_indent: int):
        result = []
        i = start
        while i < len(lines):
            lineno, ind, content = lines[i]
            if ind < seq_indent:
                break
            if ind > seq_indent:
                raise ValueError(f"line {lineno}: unexpected indent in seq")
            if not content.startswith("- "):
                break
            rest = content[2:].strip()
            i += 1
            if ":" not in rest:
                raise ValueError(f"line {lineno}: unsupported list item")
            k, _, v = rest.partition(":")
            k = k.strip()
            v = v.strip()
            item = {k: _parse_scalar(v) if v else None}
            while i < len(lines):
                ln, ind2, c2 = lines[i]
                if ind2 <= seq_indent:
                    break
                if c2.startswith("- "):
                    break
                if ":" not in c2:
                    raise ValueError(f"line {ln}: expected key in list item")
                ck, _, cv = c2.partition(":")
                ck = ck.strip()
                cv = cv.strip()
                i += 1
                if cv == "":
                    child, i = parse_block(i, ind2)
                    item[ck] = child
                else:
                    item[ck] = _parse_scalar(cv)
            result.append(item)
        return result, i

    doc, end = parse_map(0, 0)
    if end != len(lines):
        raise ValueError(f"unconsumed YAML starting line {lines[end][0]}")
    if "ops" not in doc or not isinstance(doc["ops"], list):
        raise ValueError("ops.yaml must contain an ops: list")
    return doc


def _out_dtype_expr(out_spec: dict) -> str:
    dtype_spec = out_spec.get("dtype", "same")
    if dtype_spec is None or dtype_spec == "same":
        return "graph._out_dtype_same"
    if isinstance(dtype_spec, dict):
        attr_key = dtype_spec.get("from_attr")
        if not attr_key:
            raise ValueError(f"invalid dtype spec: {dtype_spec!r}")
        return f"graph._out_dtype_from_attr({attr_key!r})"
    if isinstance(dtype_spec, str):
        return f"graph._out_dtype_const({dtype_spec!r})"
    raise ValueError(f"invalid dtype spec: {dtype_spec!r}")


def _out_channels_expr(out_spec: dict) -> str:
    ch = out_spec.get("channels", "same")
    if ch == "same":
        return "graph._out_channels_same"
    return f"graph._out_channels_const({int(ch)})"


def _in_channels_expr(inputs: list) -> str:
    if not inputs:
        raise ValueError("op requires at least one input")
    if len(inputs) != 1:
        raise ValueError("only single-input ops are supported in engines.ops")
    ch = inputs[0].get("channels", "any")
    if ch == "any":
        return "None"
    return str(int(ch))


def gen_ops_py(doc: dict) -> str:
    ops = doc["ops"]
    lines = [
        "# Generated by muimg.engines.tools.gen_ops — do not edit.",
        "# EngineOp callables + OPS_BY_NAME from engines/catalog/ops.yaml.",
        "from __future__ import annotations",
        "",
        "import json",
        "",
        "from . import graph",
        "from .graph import EngineOp, OpMeta",
        "",
    ]

    names: list[str] = []
    for op in ops:
        name = op["name"]
        names.append(name)
        attrs = op.get("attrs") or []
        inputs = op.get("inputs") or [{"channels": "any"}]
        outputs = op.get("outputs") or [{"channels": "same"}]
        if len(outputs) != 1:
            raise ValueError(f"op {name!r}: only single-output ops are supported")
        out_spec = outputs[0]
        attrs_json = json.dumps(attrs, indent=2, sort_keys=True)
        lines.append(f"{name} = EngineOp(")
        lines.append(f"    meta=OpMeta(name={name!r}),")
        lines.append(f"    _out_dtype={_out_dtype_expr(out_spec)},")
        lines.append(f"    _out_channels={_out_channels_expr(out_spec)},")
        lines.append(f"    _in_channels={_in_channels_expr(inputs)},")
        lines.append(
            f"    _attr_specs=tuple(json.loads(r'''{attrs_json}''')),"
        )
        lines.append(")")
        lines.append("")

    lines.append("OPS_BY_NAME = {")
    for name in names:
        lines.append(f"    {name!r}: {name},")
    lines.append("}")
    lines.append("")
    return "\n".join(lines)


def validate_ops(ops: list) -> None:
    if not ops:
        raise ValueError("No ops in catalog")
    for op in ops:
        if "name" not in op:
            raise ValueError(f"op missing name: {op}")
        if "affinity" in op:
            raise ValueError(
                f"op {op['name']!r}: affinity is not allowed "
                "(catalog is portable interface only)"
            )
        if not op.get("inputs") or not op.get("outputs"):
            raise ValueError(
                f"op {op['name']!r}: inputs and outputs are required"
            )
        if len(op["inputs"]) != 1 or len(op["outputs"]) != 1:
            raise ValueError(
                f"op {op['name']!r}: only single-input/single-output ops "
                "are supported"
            )


def main() -> int:
    if not YAML_PATH.is_file():
        print(f"Missing catalog: {YAML_PATH}", file=sys.stderr)
        return 1
    try:
        doc = load_ops_yaml(YAML_PATH)
        validate_ops(doc.get("ops") or [])
        text = gen_ops_py(doc)
    except ValueError as e:
        print(str(e), file=sys.stderr)
        return 1

    OPS_OUT.write_text(text)
    print(f"Wrote {OPS_OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
