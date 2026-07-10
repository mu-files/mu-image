# Changelog

All notable changes to mu-dng-converter will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## app-v0.1.20260601.0857 - 2026-06-01

### Changed
- Migrated from Flet to PyWebView with custom HTML/CSS/JS interface
- Added Create DNG tab for unified DNG→DNG and FITS→DNG conversion
- Added metadata operations (set/strip tags, time shift, timezone)
- Added transcode options (compression, demosaic, scale)
- Added file vs folder input selection
- Added overwrite policy handling with skip/overwrite prompt

### Fixed
- **FITS→DNG**: Fixed monochrome preview JPEG encoding failure in batch/multi-threaded mode.

## [app-v0.1.20260531.1519] - 2026-05-31

### Fixed
- **FITS→DNG**: Switch to `pyavm` for parsing FITS headers; add endianness conversion to support more FITS files.
- **FITS→DNG**: Fix monochrome FITS detection and Bayer pattern handling.
- **FITS→DNG**: Fix AVM XMP metadata extraction.

## [app-v0.1.20260529.1537] - 2026-05-29

### Added
- Initial PyWebView-based GUI with batch DNG-to-TIF/JPEG conversion
- Real-time progress bar and cancellation support
- Configurable output format, bit depth, and white balance
- Multi-worker parallel processing via `muimg` pipeline
- macOS, Windows, and Linux desktop builds via GitHub Actions
- FITS→DNG conversion tab with AVM XMP metadata mapping
- Windows installer via Inno Setup
- Persistent scrollable log pane with progress reporting and cancellation
