# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files
"""Load the platform-tagged abi3 ``_core_engine`` extension from ``_binaries/``.

Built in private mu-image-engine (Rust/PyO3 + linked C++) and synced via
copy_to_public.sh. One binary per platform (abi3, Python >= 3.12).

This module is the Python loader only; the extension filename / PyInit symbol
remain ``_core_engine``.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import platform
import sys
from pathlib import Path

__all__ = ["execute_graph", "version"]


def _host_tagged_name() -> str:
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Windows":
        return "_core_engine.windows-amd64.abi3.pyd"
    if system == "Darwin":
        tag = "macos-arm64" if machine in ("arm64", "aarch64") else "macos-x86_64"
        return f"_core_engine.{tag}.abi3.so"
    if machine in ("arm64", "aarch64"):
        return "_core_engine.linux-aarch64.abi3.so"
    return "_core_engine.linux-x86_64.abi3.so"


def _load():
    name = _host_tagged_name()
    path = Path(__file__).resolve().parent / "_binaries" / name
    if not path.is_file():
        raise ImportError(
            f"Native CoreEngine extension not found: {path}\n"
            "Build in mu-image-engine and run ./copy_to_public.sh local|ci"
        )
    # Last path component must be `_core_engine` so PyInit__core_engine is used.
    mod_name = "muimg.engines.core._binaries._core_engine"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    loader = importlib.machinery.ExtensionFileLoader(mod_name, str(path))
    spec = importlib.util.spec_from_loader(mod_name, loader)
    if spec is None:
        raise ImportError(f"Failed to create import spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    loader.exec_module(mod)
    return mod


_mod = _load()
execute_graph = _mod.execute_graph
version = getattr(_mod, "version", None)
