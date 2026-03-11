from setuptools import setup, Extension
import numpy as np
import platform
import os

# Common optimization flags
common_compile_args = [
    '-O3',                    # Maximum optimization
    '-march=native',          # Use native CPU instructions (ARM on M1/M2/M3)
    '-mtune=native',          # Optimize for specific CPU
    '-ffast-math',            # Fast math operations
    '-funroll-loops',         # Loop unrolling
    '-flto',                  # Link-time optimization
    '-fomit-frame-pointer',   # Don't keep frame pointer (faster)
    '-fno-strict-aliasing',   # Allow pointer aliasing optimizations
]

common_link_args = [
    '-flto',                  # Link-time optimization
]

# Demosaic C extensions with ARM optimizations
vng_extension = Extension(
    'muimg._vng',
    sources=['src/demosaic/vng.c'],
    include_dirs=[np.get_include()],
    extra_compile_args=common_compile_args,
    extra_link_args=common_link_args,
)

# RCD extension - only built if user has renamed rcd.txt to rcd.c (GPL code)
rcd_source = 'src/demosaic/rcd.c'
rcd_extension = None
if os.path.exists(rcd_source):
    rcd_extension = Extension(
        'muimg._rcd',
        sources=[rcd_source],
        include_dirs=[np.get_include()],
        extra_compile_args=common_compile_args,
        extra_link_args=common_link_args,
    )

# DNG color processing C++ extension
# Standalone implementation of DNG SDK color algorithms (no SDK dependencies)
raw_render_extension = Extension(
    'muimg._raw_render',
    sources=['src/raw_render/raw_render_ops.cpp'],
    include_dirs=[np.get_include()],
    extra_compile_args=common_compile_args + ['-std=c++17'],
    extra_link_args=common_link_args,
    language='c++',
)

# Build list of extensions
ext_modules = [vng_extension, raw_render_extension]
if rcd_extension:
    ext_modules.append(rcd_extension)

setup(
    ext_modules=ext_modules,
)
