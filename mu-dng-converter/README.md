# mu DNG Converter

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Releases](https://img.shields.io/github/v/release/mu-files/mu-image?label=Download)](https://github.com/mu-files/mu-image/releases/latest)

A cross-platform desktop application for batch image conversion built on [muimg](../muimg/README.md) and [Flet](https://flet.dev).

## DNG → Image

<img src="docs/DNG-Image.png" width="480" alt="DNG to Image tab">

Convert DNG files to TIFF or JPEG with full rendering control:
- White balance (presets or custom temperature/tint)
- Exposure adjustment
- Output bit depth (8-bit or 16-bit)
- Resolution scale (0.125–1.0)
- Multi-threaded batch processing (1–8 workers)

> **Output Mode**  
> The mode selector at the top controls the output format. In addition to TIFF, JXL, and JPEG, selecting **Video** produces an MP4 from a DNG sequence — useful for timelapse or allsky footage. Video mode has its own codec, resolution, frame rate, and CRF controls.

> **Use XMP**  
> When enabled (default), white balance, exposure, and tone curve are read from the DNG's embedded XMP metadata — exactly as set in your RAW editor. Disable this to apply your own white balance and exposure overrides instead.

> **Scale**  
> Renders output at a fraction of full resolution (e.g. `0.5` = half size). Useful for generating quick previews or producing smaller deliverables without changing the source files.

## FITS → DNG

<img src="docs/FITS-DNG.png" width="480" alt="FITS to DNG tab">

Convert astronomical FITS files to DNG for use in Photoshop and other RAW editors:
- Auto exposure (histogram-based EV shift and black level estimation)
- AVM XMP metadata mapping from FITS headers
- Color temperature selection
- Colour space and channel order configuration
- JPEG preview and fast-load pyramid embedding
- Multi-threaded batch processing (1–8 workers)

> **Auto Exposure**  
> FITS files from astronomical cameras often have no embedded exposure hint, causing RAW editors like Photoshop to render them nearly black. When enabled, Auto Exposure analyses the image histogram to estimate a black level (1st percentile) and an exposure shift (targeting ~6% brightness), so the image opens at a reasonable starting point. The `PEDESTAL` FITS header is used as the black level when present.

> **AVM XMP Metadata**  
> Astronomy Visualization Metadata (AVM) is a standard for embedding sky coordinates, instrument details, and observation data in image files. When FITS headers contain WCS coordinates (`CRVAL`, `CDELT`, etc.), object names, filter, or telescope information, these are mapped to AVM XMP tags in the output DNG — transferring these tags to downstream applications.

## Getting Started

### Desktop (macOS, Windows, Linux)

Download a pre-built binary from the [Releases](https://github.com/mu-files/mu-image/releases) page. No Python installation required.

### Raspberry Pi

Install directly from GitHub using pip (Python 3.12+ required), then launch the desktop GUI:

```bash
pip install "mu-dng-converter @ git+https://github.com/mu-files/mu-image.git#subdirectory=mu-dng-converter"
mu-dng-converter
```

### Developers

Clone the repository and install in editable mode (Python 3.12+ required):

```bash
git clone https://github.com/mu-files/mu-image.git
cd mu-image/mu-dng-converter
pip install -e .
mu-dng-converter
```
