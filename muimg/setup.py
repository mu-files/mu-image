from setuptools import setup, Extension
import numpy as np
import platform

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

rcd_extension = Extension(
    'muimg._rcd',
    sources=['src/demosaic/rcd.c'],
    include_dirs=[np.get_include()],
    extra_compile_args=common_compile_args,
    extra_link_args=common_link_args,
)

# DNG color processing C++ extension
# Standalone implementation of DNG SDK color algorithms (no SDK dependencies)
dng_color_extension = Extension(
    'muimg._dng_color',
    sources=['src/dng_color/dng_color_standalone.cpp'],
    include_dirs=[np.get_include()],
    extra_compile_args=common_compile_args + ['-std=c++17'],
    extra_link_args=common_link_args,
    language='c++',
)

setup(
    ext_modules=[vng_extension, rcd_extension, dng_color_extension],
)
