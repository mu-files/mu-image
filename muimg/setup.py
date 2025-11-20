from setuptools import setup, Extension
import numpy as np

# Demosaic C extensions with ARM optimizations
vng_extension = Extension(
    'muimg._vng',
    sources=['src/demosaic/vng.c'],
    include_dirs=[np.get_include()],
    extra_compile_args=[
        '-O3',                    # Maximum optimization
        '-march=native',          # Use native CPU instructions (ARM on M1/M2/M3)
        '-mtune=native',          # Optimize for specific CPU
        '-ffast-math',            # Fast math operations
        '-funroll-loops',         # Loop unrolling
        '-flto',                  # Link-time optimization
        '-fomit-frame-pointer',   # Don't keep frame pointer (faster)
        '-fno-strict-aliasing',   # Allow pointer aliasing optimizations
    ],
    extra_link_args=[
        '-flto',                  # Link-time optimization
    ],
)

rcd_extension = Extension(
    'muimg._rcd',
    sources=['src/demosaic/rcd.c'],
    include_dirs=[np.get_include()],
    extra_compile_args=[
        '-O3',
        '-march=native',
        '-mtune=native',
        '-ffast-math',
        '-funroll-loops',
        '-flto',
        '-fomit-frame-pointer',
        '-fno-strict-aliasing',
    ],
    extra_link_args=[
        '-flto',
    ],
)

setup(
    ext_modules=[vng_extension, rcd_extension],
)
