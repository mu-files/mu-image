## High-performance image tools for Python

[![License](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE)

`mu-image` is a cross-platform toolkit for software engineers who process
2D images in Python. The available set of image-processing primitives are
highly-optimized and can be chained to run sequentially in native code.
Where equivalent functions exist mu-image should be much faster and use less 
memory than numpy or opencv. But interfacing with both those frameworks is
seamless.

Included in the package is also a fully spec compliant DNG RAW decoder and
encoder that is built completely on mu-image.

The repository contains two packages:

- **[`muimg`](muimg/README.md)**: A Python library and CLI for DNG I/O,
  rendering, metadata, and multi-threaded batch processing
  (`pip install muimg`).
- **[`mu-dng-converter`](mu-dng-converter/README.md)**: A desktop GUI on
  `muimg` and [PyWebView](https://pywebview.flowrl.com/) for batch
  convert and transcode.

---

## Primary use cases

- **High performance image processing**: basic imaging primitives to develop
  your own pipelines.
- **DNG in your app**:  read, write, and render DNG.
- **DNG transcode**: Re-encode a DNG with a different codec (uncompressed,
  JPEG XL lossless/lossy) and/or update TIFF/DNG metadata, without a
  full develop.
- **Batch convert**: RAW/DNG sequences to TIFF (8- or 16-bit), JPEG, or
  JPEG XL, including multi-core folder jobs.
- **DNG to video**: Stitch a folder of DNG frames into MP4/H.264.
- **FITS to DNG**: Convert scientific `.fits` to DNG for Photoshop,
  Lightroom, or Camera Raw.

---

## Key technical features

### RAW rendering (`muimg`)

- **Demosaicing**: `BILINEAR`, `VNG`, `RCD`, `EA`, `EA_FAST`, and
  `OPENCV_EA`.
- **Opcode and color**: Linearization, DNG opcodes, color matrices, and
  tone curves.
- **macOS**: Built-in renderer or Apple Core Image.
- **XMP**: Temperature, tint, exposure, curves, and radial distortion
  when present.
- **CLI**: Render, transcode, and inspect DNG files.

### Desktop app (`mu-dng-converter`)

A batch-scale GUI: select files or a folder of DNG (or FITS). No Python
install if you use the pre-built binary.

Render develops to TIFF, JPEG, JPEG XL, or MP4. Transcode stays in DNG:
change the codec and/or rewrite metadata without a full develop.

- **Transcode**: Uncompressed or JPEG XL (lossless/lossy); set or strip
  tags, shift timestamps, timezone.
- **Render control**: White balance (presets or Kelvin/tint), exposure,
  output bit depth, resolution scale (0.125×–1.0×) for previews.
- **Video**: Codec, resolution, frame rate, and CRF for MP4 from raw
  frames.
- **Convert FITS to DNG**: Histogram auto-exposure so files open at a
  usable baseline; AVM XMP from WCS and instrument headers; JPEG
  previews and pyramids for fast browse.

---

## Supported platforms

- **Windows** (x86_64)
- **macOS** (Intel and Apple Silicon)
- **Linux** (Ubuntu, Debian, Fedora)
- **Raspberry Pi** (ARM64)

---

## Getting started

**Desktop app (macOS, Windows, Linux):** Download a pre-built binary from
the [mu-dng-converter releases page](https://github.com/mu-files/mu-image/releases/latest).
No Python install required. See
[mu-dng-converter/README.md](mu-dng-converter/README.md).

**Python library:** See [muimg/README.md](muimg/README.md) for install,
API, and CLI.

**Release history:** See [CHANGELOG.md](muimg/CHANGELOG.md).
