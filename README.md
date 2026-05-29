## RAW Image Infrastructure & Utilities

[![License](https://img.shields.io/badge/License-PolyForm%20Small%20Business-blue.svg)](muimg/LICENSE)

`mu-image` is a cross-platform collection of high-performance tools for software engineers, astrophotographers, and time-lapse editors to process, convert, and render raw image files.

The repository contains two packages:

- **[`muimg`](muimg/README.md)**: A Python library and CLI engine for Adobe DNG manipulation, rendering, metadata handling, and multi-threaded batch processing.
- **[`mu-dng-converter`](mu-dng-converter/README.md)**: A cross-platform desktop GUI application built on `muimg` and [Flet](https://flet.dev) for batch-converting raw image folders, time-lapse sequences, and astronomical data.

---

## Primary Use Cases

- **DNG support in your app**: a source-available python package to add DNG support to your project (no restrictions except for large companies)
- **DNG to video time-lapse assembly**: Stitch folders of raw DNG frames directly into MP4/H.264 video clips. Suited for allsky camera sequences and day-to-night astrophotography arrays.
- **Astronomical FITS to DNG conversion**: Convert scientific `.fits` files into compliant `.dng` raw images for editing in Adobe Photoshop, Lightroom, or Camera Raw.
- **Batch image transcoding**: Mass-convert raw sequences to TIFF (8-bit/16-bit), JPEG, and JPEG XL (JXL) with multi-core parallel processing.

---

## Key Technical Features

### Advanced RAW Rendering Pipeline (`muimg`)

- **Demosaicing**: Multiple algorithms — `DNGSDK_BILINEAR`, `VNG`, `RCD`, and `OPENCV_EA`.
- **Opcode & color correction**: Full DNG rendering pipeline including linearization, opcodes, color correction matrices, and custom tone curves.
- **Native OS acceleration**: On macOS, choose between the built-in pipeline renderer and the hardware-accelerated Apple Core Image renderer.
- **XMP metadata**: Some support for embedded XMP. Automatically applies `Temperature`, `Tint`, `Exposure`, custom curves, and radial distortion corrections as configured in your RAW editor.
- **cli**: feature-rich cli to render, transcode, and inspect DNG files

### Desktop GUI Application (`mu-dng-converter`)

#### DNG to Image & Video Transcoding

- **Full rendering control**: Override white balance (presets or custom Kelvin/tint), adjust exposure, change output bit depth, or apply a resolution scale (0.125x–1.0x) for quick previews.
- **Multi-threaded video encoding**: Dedicated video mode with direct control over codec, resolution, frame rate, and CRF for lightweight MP4 output from raw frames.

#### Astronomical FITS to DNG

- **Histogram-based auto exposure**: Analyses the image histogram to estimate a black level (1st percentile) and an exposure shift (targeting ~6% brightness), so FITS files open at a usable baseline in standard RAW editors.
- **AVM XMP metadata mapping**: Maps WCS data (`CRVAL`, `CDELT`), object names, filter parameters, and telescope details from FITS headers into Astronomy Visualization Metadata (AVM) XMP tags embedded in the output DNG.
- **Fast-load optimization**: Embeds JPEG previews and image pyramids in the exported DNG for instant browsing.

---

## Supported Platforms

- **Windows** (x86_64)
- **macOS** (Intel & Apple Silicon)
- **Linux** (Ubuntu, Debian, Fedora)
- **Raspberry Pi** (ARM64 — suitable for low-power allsky camera rigs and remote observatories)

---

## Getting Started

**Desktop app (macOS, Windows, Linux):** Download a pre-built binary from the [mu-dng-converter releases page](https://github.com/mu-files/mu-image/releases/latest). No Python installation required. See [mu-dng-converter/README.md](mu-dng-converter/README.md) for GUI usage and feature details.

**Python library:** See [muimg/README.md](muimg/README.md) for installation, API reference, and CLI usage.

**Release history:** See [CHANGELOG.md](muimg/CHANGELOG.md).
