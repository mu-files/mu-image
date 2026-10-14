# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files

import os
import platform
import sys

import numpy as np
from setuptools import Extension, setup


def _is_android():
    if sys.platform == "android" or hasattr(sys, "getandroidapilevel"):
        return True
    # Termux reports Linux. The Android dynamic linker is on the device.
    return os.path.exists("/system/bin/linker64") or os.path.exists(
        "/system/bin/linker"
    )


def host_core_binaries():
    """Ship only the host abi3 CoreEngine extension in wheels; sdist keeps all via MANIFEST.in."""
    system = platform.system()
    machine = platform.machine().lower()
    if _is_android():
        if machine not in ("arm64", "aarch64"):
            raise RuntimeError(
                f"Android {machine} is not a supported CoreEngine target"
            )
        name = "_core_engine.android-aarch64.abi3.so"
    elif system == "Windows":
        if machine in ("arm64", "aarch64"):
            name = "_core_engine.windows-arm64.abi3.pyd"
        else:
            name = "_core_engine.windows-amd64.abi3.pyd"
    elif system == "Darwin":
        if machine in ("arm64", "aarch64"):
            name = "_core_engine.macos-arm64.abi3.so"
        else:
            name = "_core_engine.macos-x86_64.abi3.so"
    else:
        if machine in ("arm64", "aarch64"):
            name = "_core_engine.linux-aarch64.abi3.so"
        else:
            name = "_core_engine.linux-x86_64.abi3.so"
    return [f"_binaries/{name}"]


if sys.platform == "win32":
    common_compile_args = [
        "/O2",
        "/fp:fast",
        "/GL",
    ]
    common_link_args = [
        "/LTCG",
    ]
else:
    is_ci = (
        os.environ.get("CI") == "true"
        or os.environ.get("GITHUB_ACTIONS") == "true"
        or os.environ.get("CIBUILDWHEEL") == "1"
    )

    common_compile_args = [
        "-O3",
        "-ffast-math",
        "-funroll-loops",
        "-flto",
        "-fomit-frame-pointer",
        "-fno-strict-aliasing",
    ]

    if not is_ci:
        common_compile_args.extend(
            [
                "-march=native",
                "-mtune=native",
            ]
        )

    common_link_args = [
        "-flto",
    ]

# VNG demosaic extension (stays in public; LGPL/CDDL)
vng_extension = Extension(
    "muraw._vng",
    sources=["c-src/demosaic/vng.c"],
    include_dirs=[np.get_include()],
    extra_compile_args=common_compile_args,
    extra_link_args=common_link_args,
)

# RCD extension - only built if user has renamed rcd.txt to rcd.c (GPL code)
rcd_source = "c-src/demosaic/rcd.c"
rcd_extension = None
if os.path.exists(rcd_source):
    rcd_extension = Extension(
        "muraw._rcd",
        sources=[rcd_source],
        include_dirs=[np.get_include()],
        extra_compile_args=common_compile_args,
        extra_link_args=common_link_args,
    )

ext_modules = [vng_extension]
if rcd_extension:
    ext_modules.append(rcd_extension)

setup(
    ext_modules=ext_modules,
    include_package_data=False,
    # All package-data lives here. A [tool.setuptools.package-data] table
    # in pyproject.toml replaces this dict, which dropped the host binary
    # from wheels and left only ops.yaml.
    package_data={
        "muraw.engines": ["catalog/ops.yaml"],
        "muraw.engines.core": host_core_binaries(),
    },
)
