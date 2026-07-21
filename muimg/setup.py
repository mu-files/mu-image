# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 mu-files

from setuptools import setup, Extension
import numpy as np
import os
import platform
import sys


def host_core_binaries():
    """Ship only the host libmuimg_core in wheels; sdist keeps all via MANIFEST.in."""
    system = platform.system()
    machine = platform.machine().lower()
    if system == "Windows":
        name = "muimg_core.windows-amd64.dll"
    elif system == "Darwin":
        if machine in ("arm64", "aarch64"):
            name = "libmuimg_core.macos-arm64.dylib"
        else:
            name = "libmuimg_core.macos-x86_64.dylib"
    else:
        if machine in ("arm64", "aarch64"):
            name = "libmuimg_core.linux-aarch64.so"
        else:
            name = "libmuimg_core.linux-x86_64.so"
    return [f"_binaries/{name}"]

if sys.platform == 'win32':
    # MSVC flags for Windows
    common_compile_args = [
        '/O2',                # Maximum optimization
        '/fp:fast',           # Fast math operations
        '/GL',                # Whole program optimization (LTO)
    ]
    common_link_args = [
        '/LTCG',              # Link-time code generation (LTO)
    ]
    cpp_extra_args = ['/std:c++17']
else:
    # GCC/Clang flags for macOS and Linux
    # Skip -march=native in CI / cibuildwheel (non-portable CPU flags break wheels)
    is_ci = (
        os.environ.get('CI') == 'true'
        or os.environ.get('GITHUB_ACTIONS') == 'true'
        or os.environ.get('CIBUILDWHEEL') == '1'
    )
    
    common_compile_args = [
        '-O3',                    # Maximum optimization
        '-ffast-math',            # Fast math operations
        '-funroll-loops',         # Loop unrolling
        '-flto',                  # Link-time optimization
        '-fomit-frame-pointer',   # Don't keep frame pointer (faster)
        '-fno-strict-aliasing',   # Allow pointer aliasing optimizations
    ]
    
    # Only use native CPU optimizations when not in CI
    if not is_ci:
        common_compile_args.extend([
            '-march=native',      # Use native CPU instructions (ARM on M1/M2/M3)
            '-mtune=native',      # Optimize for specific CPU
        ])
    
    common_link_args = [
        '-flto',                  # Link-time optimization
    ]
    cpp_extra_args = ['-std=c++17']

# Raw render extension - incrementally migrating to use libmuimg_core
raw_render_extension = Extension(
    'muimg._raw_render',
    sources=['c-src/raw_render/raw_render_ops.cpp'],
    include_dirs=[np.get_include(), 'c-src'],
    extra_compile_args=common_compile_args + cpp_extra_args,
    extra_link_args=common_link_args,
    language='c++',
)

# Compute-graph engine binding (Phase A)
compute_engine_extension = Extension(
    'muimg._compute_engine',
    sources=['c-src/compute_engine/compute_engine.cpp'],
    include_dirs=[np.get_include(), 'c-src'],
    extra_compile_args=common_compile_args + cpp_extra_args,
    extra_link_args=common_link_args,
    language='c++',
)

# VNG demosaic extension
vng_extension = Extension(
    'muimg._vng',
    sources=['c-src/demosaic/vng.c'],
    include_dirs=[np.get_include()],
    extra_compile_args=common_compile_args,
    extra_link_args=common_link_args,
)

# RCD extension - only built if user has renamed rcd.txt to rcd.c (GPL code)
rcd_source = 'c-src/demosaic/rcd.c'
rcd_extension = None
if os.path.exists(rcd_source):
    rcd_extension = Extension(
        'muimg._rcd',
        sources=[rcd_source],
        include_dirs=[np.get_include()],
        extra_compile_args=common_compile_args,
        extra_link_args=common_link_args,
    )

# Build list of extensions
ext_modules = [raw_render_extension, compute_engine_extension, vng_extension]
if rcd_extension:
    ext_modules.append(rcd_extension)

setup(
    ext_modules=ext_modules,
    include_package_data=False,
    package_data={"muimg": host_core_binaries()},
)
