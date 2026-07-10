# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.20260710.1135] - 2026-07-10

### Changed
- **Error Reporting**: Add `log_callback` parameter to `run_batch_convert()`, `run_batch_copy_dng()`, and `run_batch_to_video()` to pass rendering errors to GUI applications instead of only logging to stderr

## [0.1.20260709.1718] - 2026-07-09

### Added
- **XMP Crop Support**: Read and apply XMP crop metadata (`crs:HasCrop`, `CropTop/Left/Bottom/Right`) during DNG rendering
- **Batch DNG Copy/Transcode**: New `run_batch_copy_dng()` function for GUI batch DNG copy and transcode operations with progress callbacks

### Changed
- **Preview/Pyramid Copy**: Intelligently copy existing preview and pyramid pages when no transforms are applied, regenerate only when needed (scale/demosaic applied or structure mismatch)
- **Pyramid Levels**: Set default pyramid levels to 2 for better fast-load performance
- **Metadata Operations**: Improved timezone handling in DNG metadata operations
- **Time Shift**: Fix time shift to update all datetime tags (DateTime, DateTimeOriginal, DateTimeDigitized)

### Fixed
- **Uncompressed Transcode**: Fixed uncompressed transcode to use `COMPRESSION.NONE` correctly
- **XMP Rendering**: Apply XMP adjustments (except crop) to preview rendering
- **Video XMP Crop**: Fixed video XMP crop rendering and UI layout

## [0.1.20260601.0852] - 2026-06-01

### Changed
- **Dependency**: Updated minimum `tifffile` requirement to `>=2026.6.1`. This version changed `PHOTOMETRIC_SAMPLES[LINEAR_RAW]` from 3 to 1 for single-channel images allowing for monochrome or color LINEAR_RAW
- **Preview**: Monochrome (`MINISBLACK`) previews are converted to RGB (single channel replicated across all three) before JPEG compression, avoiding yet to be debugged issue `imagecodecs.jpeg8_encode` with MINISBLACK multi-threaded jpeg encode

## [0.1.20260531.1501] - 2026-05-31

### Added
- **Monochrome DNG support**: Write single-channel (grayscale) LINEAR_RAW DNGs from 2D `(H, W)` or 3D `(H, W, 1)` arrays via `write_dng_from_array`.
- **Monochrome preview/pyramid**: Preview and pyramid generation for monochrome DNGs, rendering via a dedicated `_render_camera_monochrome` pipeline (exposure, tone curve, gamma) `MINISBLACK` preview.
- **Optimized `mono_lut` C++ function**: Fast single-channel LUT application with 8-bit direct lookup optimization.

## [0.1.20260527.0727] - 2026-05-27

### Changed
- **Performance**: Release Python GIL in all 17 C++ compute kernels (`raw_render_ops.cpp`), enabling true multi-threaded parallelism. With `--num-workers 4`: 2.1x speedup (was ~14% before).
- **Performance**: Set Core Image `use_gpu=False` (software renderer benchmarks faster on Apple Silicon).
- **Refactor**: Extract reusable `run_batch_convert()` and `run_batch_to_video()` functions from CLI for shared use by GUI app.
- **Refactor**: Add `on_task_done` callback to `ProcessingPipeline` for unified progress reporting and cancellation.

## [0.1.20260522.1828] - 2026-05-22

### Added
- **Synchronous Processing Mode**: `ProcessingPipeline` supports `num_workers=0` for single-threaded operation, useful for debugging and resource-constrained environments.
- **Background Module Loader**: Lazy loading of slow dependencies to reduce startup time.

### Changed
- **Preview Generation**: Faster preview rendering in `write_dng_from_array` and `write_dng_from_page` with optimized color pipeline.
- **Pyramid Images**: Faster multi-resolution pyramid generation with configurable filter types (`CATMULL_ROM`, `LANCZOS`)
- **Performance**: Major refactor of `normalize_black_white` with NEON implementation, fused uint16→float32 conversion, and specialized fast paths for common black/white level configurations.
- **Performance**: NEON-optimized `DtypeConverter` for `convert_dtype` on AArch64.
- **Performance**: Added `__restrict` qualifiers to non-inplace C++ pixel-processing functions.
- **Performance**: Replaced `fminf`/`fmaxf` with `std::clamp`/`std::min`/`std::max` in C++ ops.
- **Performance**: `BitsPerSample` and `CFAPattern` resolved from spec fields or metadata tags with priority (spec > metadata > default), with validation against data dtype.
- **Refactor**: Extracted raw-to-camera-RGB path (`_raw_to_camera_rgb`, `_render_camera_rgb`) to allow standalone use outside `Page`.
- **Refactor**: Removed eager re-exports from `__init__.py`; tests use explicit submodule imports.

## [0.1.20260519.1642] - 2026-05-19

### Added
- **Performance Timing**: Comprehensive hierarchical timing of DNG rendering pipeline. Run `muimg dng convert -v` to get a per-step breakdown (decode, linearize, color transforms, tone curves, etc.) with milliseconds and percentage of total render time.
- **NEON Optimization**: AArch64 NEON SIMD acceleration for the 3x3 color matrix transform (`ColorMatrix3x3`), improving performance on ARM platforms (Raspberry Pi, Apple Silicon).

### Changed
- **Performance**: Fused input clipping into `clip_and_transform_color` C++ function, eliminating a separate pass over pixel data.

### Technical Details
- Added hierarchical `PerfTimer` with thread-local storage root ownership (`common.py`, `raw_render_ops.cpp`)
- `ColorMatrix3x3` uses NEON intrinsics on AArch64, scalar fallback on all other platforms
- GCC AArch64 compatibility fixes for NEON code path
- All tests passing on macOS (x86_64/ARM), Linux/AArch64 (Raspberry Pi), and Windows (x86_64)

## [0.1.20260515.1312] - 2026-05-15

### Changed
- **Performance**: Fused color transform C++ function provides 2x speedup over previous implementation
- **Refactor**: Consolidated color transforms into unified transform_color C++ function with optimized clip_and_transform_color operation
- **Refactor**: Converted all C++ functions to use RAII PyPtr helpers for better memory safety
- **Refactor**: Consolidated LUT handling and removed duplicate curve functions
- **Code Quality**: Removed unnecessary dtype conversions in rendering pipeline
- **Enhancement**: Added templated hue-preserving transforms and refined post-rendering logic
- **Architecture**: Added LUT and ColorSpaceLUT classes to splines.py for better abstraction

### Technical Details
- Major refactoring of `raw_render_ops.cpp` (+1848/-1221 lines)
- Enhanced `splines.py` with new LUT infrastructure (+289 lines)
- Streamlined `raw_render.py` color pipeline (+752/-752 lines)
- All tests passing (562 passed, 2 skipped, 46 xfailed)
