# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased] - 2026-05-19

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
