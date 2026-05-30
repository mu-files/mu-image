"""FITS → DNG conversion: core logic and CLI.

Converts FITS files containing CFA/Bayer data to DNG using the muimg library.
Supports single-file and batch conversion with full metadata mapping,
AVM XMP embedding, and optional batch pipeline via ImageSequencePipeline.

Can be used as a library (imported by the GUI view) or as a CLI:
    mu-dng-fits input.FIT
    mu-dng-fits input_folder/ -o output_folder/
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


# =============================================================================
# AVM (Astronomy Visualization Metadata) XMP mapping
# =============================================================================


def _parse_ra_dec(ra_str: str, dec_str: str) -> tuple[float, float] | None:
    """Parse RA/DEC strings to decimal degrees.

    Handles both decimal and sexagesimal (HH:MM:SS / DD:MM:SS) formats.
    RA is in hours (0-24) and converted to degrees (0-360).
    """
    try:
        ra_deg = float(ra_str)
        dec_deg = float(dec_str)
        return ra_deg, dec_deg
    except (ValueError, TypeError):
        pass

    # Try sexagesimal: "HH:MM:SS.s" or "HH MM SS.s"
    try:
        parts = ra_str.replace(":", " ").split()
        h, m, s = float(parts[0]), float(parts[1]), float(parts[2])
        ra_deg = (h + m / 60.0 + s / 3600.0) * 15.0  # hours to degrees

        parts = dec_str.replace(":", " ").split()
        sign = -1 if parts[0].startswith("-") else 1
        d, m, s = abs(float(parts[0])), float(parts[1]), float(parts[2])
        dec_deg = sign * (d + m / 60.0 + s / 3600.0)

        return ra_deg, dec_deg
    except (ValueError, TypeError, IndexError):
        return None


def fits_header_to_avm_xmp(header) -> dict[str, Any]:
    """Extract AVM XMP properties from a FITS header.

    Args:
        header: astropy FITS header object

    Returns:
        Dict with 'avm:'-prefixed keys suitable for XmpMetadata.from_attributes().
        Only includes fields actually present in the header.
    """
    attrs: dict[str, Any] = {}

    # Object / subject name
    obj = header.get("OBJECT")
    if obj and str(obj).strip():
        attrs["avm:Subject.Name"] = str(obj).strip()

    # Instrument (camera)
    instrume = header.get("INSTRUME")
    if instrume and str(instrume).strip():
        attrs["avm:Instrument"] = str(instrume).strip()

    # Telescope / facility
    telescop = header.get("TELESCOP")
    if telescop and str(telescop).strip():
        attrs["avm:Facility"] = str(telescop).strip()

    # Coordinate frame from equinox
    equinox = header.get("EQUINOX")
    if equinox is not None:
        eq_val = float(equinox)
        if eq_val == 2000.0:
            attrs["avm:Spatial.CoordinateFrame"] = "ICRS"
        elif eq_val == 1950.0:
            attrs["avm:Spatial.CoordinateFrame"] = "FK4"
        else:
            attrs["avm:Spatial.CoordinateFrame"] = "FK5"

    # WCS reference value (RA, Dec in degrees)
    # Try CRVAL first, fall back to RA/DEC keywords
    crval1 = header.get("CRVAL1")
    crval2 = header.get("CRVAL2")
    if crval1 is not None and crval2 is not None:
        attrs["avm:Spatial.ReferenceValue"] = [
            str(float(crval1)), str(float(crval2)),
        ]
    else:
        ra = header.get("RA")
        dec = header.get("DEC")
        if ra is not None and dec is not None:
            coords = _parse_ra_dec(str(ra), str(dec))
            if coords:
                attrs["avm:Spatial.ReferenceValue"] = [
                    str(coords[0]), str(coords[1]),
                ]

    # WCS reference pixel
    crpix1 = header.get("CRPIX1")
    crpix2 = header.get("CRPIX2")
    if crpix1 is not None and crpix2 is not None:
        attrs["avm:Spatial.ReferencePixel"] = [
            str(float(crpix1)), str(float(crpix2)),
        ]

    # Image dimensions
    naxis1 = header.get("NAXIS1")
    naxis2 = header.get("NAXIS2")
    if naxis1 is not None and naxis2 is not None:
        attrs["avm:Spatial.ReferenceDimension"] = [
            str(int(naxis1)), str(int(naxis2)),
        ]

    # Plate scale (deg/pixel)
    # Try CDELT first, then compute from FOCALLEN + pixel size
    cdelt1 = header.get("CDELT1")
    cdelt2 = header.get("CDELT2")
    if cdelt1 is not None and cdelt2 is not None:
        attrs["avm:Spatial.Scale"] = [
            str(float(cdelt1)), str(float(cdelt2)),
        ]
    else:
        focallen = header.get("FOCALLEN")
        xpixsz = header.get("XPIXSZ")
        ypixsz = header.get("YPIXSZ")
        if focallen and xpixsz and ypixsz:
            fl_mm = float(focallen)
            if fl_mm > 0:
                scale_x = float(xpixsz) / fl_mm * 206.265 / 3600.0
                scale_y = float(ypixsz) / fl_mm * 206.265 / 3600.0
                attrs["avm:Spatial.Scale"] = [
                    str(-scale_x), str(scale_y),
                ]

    # Rotation
    crota2 = header.get("CROTA2")
    if crota2 is not None:
        attrs["avm:Spatial.Rotation"] = str(float(crota2))

    # CD matrix (alternative to CDELT+CROTA)
    cd1_1 = header.get("CD1_1")
    cd1_2 = header.get("CD1_2")
    cd2_1 = header.get("CD2_1")
    cd2_2 = header.get("CD2_2")
    if all(v is not None for v in (cd1_1, cd1_2, cd2_1, cd2_2)):
        attrs["avm:Spatial.CDMatrix"] = [
            str(float(cd1_1)), str(float(cd1_2)),
            str(float(cd2_1)), str(float(cd2_2)),
        ]

    # Projection (extract from CTYPE, e.g. "RA---TAN" -> "TAN")
    ctype1 = header.get("CTYPE1")
    if ctype1 is not None:
        ctype_str = str(ctype1).strip()
        # Standard WCS convention: last 3 chars after "---"
        if "---" in ctype_str:
            proj = ctype_str.split("---")[-1]
            if proj:
                attrs["avm:Spatial.CoordsystemProjection"] = proj

    # Spectral
    filt = header.get("FILTER")
    if filt and str(filt).strip():
        attrs["avm:Spectral.Bandpass"] = str(filt).strip()

    # Temporal
    date_obs = header.get("DATE-OBS")
    if date_obs is not None:
        attrs["avm:TemporalStartTime"] = str(date_obs).strip()

    exptime = header.get("EXPTIME")
    if exptime is not None:
        attrs["avm:TemporalIntegrationTime"] = str(float(exptime))

    # Creator (try multiple keywords)
    observer = (
        header.get("OBSERVER")
        or header.get("AUTHOR")
        or header.get("CREATOR")
    )
    if observer and str(observer).strip():
        attrs["avm:Creator"] = str(observer).strip()

    # Type
    attrs["avm:Type"] = "Observation"
    attrs["avm:MetadataVersion"] = "1.1"

    return attrs


# =============================================================================
# FITS Header Inspection
# =============================================================================


def print_fits_info(fits_path: Path) -> None:
    """Print all FITS header keywords and HDU summary."""
    from astropy.io import fits as astropy_fits

    with astropy_fits.open(fits_path) as hdul:
        print(f"File: {fits_path}")
        print(f"HDU count: {len(hdul)}")
        print()

        for i, hdu in enumerate(hdul):
            hdu_type = type(hdu).__name__
            print(f"--- HDU {i} ({hdu_type}) ---")
            if hdu.data is not None:
                shape = hdu.data.shape
                dtype = hdu.data.dtype
                print(
                    f"  Data: shape={shape}  "
                    f"dtype={dtype}  "
                    f"min={hdu.data.min()}  "
                    f"max={hdu.data.max()}"
                )
            else:
                print("  Data: None")
            print()

            header = hdu.header
            print(f"  Header keywords ({len(header)}):")
            for key in header:
                if key in ("", "COMMENT", "HISTORY"):
                    continue
                val = header[key]
                comment = header.comments[key]
                if comment:
                    print(f"    {key:20s} = {val!r:>30s}  / {comment}")
                else:
                    print(f"    {key:20s} = {val!r:>30s}")
            print()


# =============================================================================
# Camera calibration tables
# =============================================================================

# ZWO ASI676MC gain table: (camera_gain, e-/ADU)
# Source: https://www.zwoastro.com/product/asi676mc/
_ASI676MC_GAIN_TABLE = np.array([
    [0, 2.55],
    [50, 1.50],
    [100, 0.85],
    [150, 0.50],
    [200, 0.30],
])

# ZWO ASI676MC blue white balance table: (wb_b, balance)
# Derived from piecewise calibration, neutral at wb_b=75 (app default)
_ASI676MC_WB_BLUE_TABLE = np.array([
    [1, 0.01],
    [50, 0.59],
    [55, 0.73],
    [65, 0.87],
    [75, 1.00],
    [100, 1.34],
])


def _gain_to_iso(gain: float, gain_table: np.ndarray) -> int:
    """Convert camera gain to ISO using e-/ADU gain table.

    Interpolates e-/ADU, then computes:
        ISO = 100 * (reference_e_per_adu / interpolated_e_per_adu)

    Args:
        gain: Camera gain value
        gain_table: Nx2 array of [[gain, e_per_adu], ...], gain ascending

    Returns:
        ISO speed rating (integer)
    """
    e = np.interp(gain, gain_table[:, 0], gain_table[:, 1])
    return int(100 * gain_table[0, 1] / e)


# =============================================================================
# FITS Metadata -> DNG Tag Mapping
# =============================================================================


def _parse_fits_datetime(date_str: str) -> datetime | None:
    """Parse FITS DATE-OBS string to datetime.

    Handles common FITS date formats:
        YYYY-MM-DDTHH:MM:SS.sss
        YYYY-MM-DDTHH:MM:SS
        YYYY-MM-DD
    """
    formats = [
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d",
    ]
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt)
        except ValueError:
            continue
    return None


def compute_image_stats(
    data, white_level: int = 65535, percentiles=(1, 50)
) -> dict:
    """Compute percentile values from raw CFA data via histogram.

    Args:
        data: 2D numpy array (CFA data)
        white_level: maximum valid pixel value
        percentiles: tuple of percentiles to compute (0-100)

    Returns:
        Dict mapping percentile -> pixel value, e.g. {1: 192, 50: 4200}.
        Also includes 'clipped': True if >25% of pixels are near white_level.
    """
    num_bins = min(white_level, 10000)
    hist, bin_edges = np.histogram(
        data.ravel(), bins=num_bins, range=(0, white_level),
    )
    cdf = np.cumsum(hist).astype(np.float64) / np.sum(hist)

    result = {}
    for p in percentiles:
        idx = np.searchsorted(cdf, p / 100.0)
        result[p] = int(bin_edges[min(idx, len(bin_edges) - 1)])

    # Clipping check
    clip_threshold = 0.975 * white_level
    clip_idx = np.searchsorted(bin_edges[:-1], clip_threshold)
    fraction_bright = 1.0 - (
        cdf[clip_idx - 1] if clip_idx > 0 else 0.0
    )
    result["clipped"] = fraction_bright > 0.25

    return result


def _build_camera_tags(tags, header) -> None:
    """Add camera-specific tags based on INSTRUME header.

    Currently supported cameras:
        - ZWO ASI676MC
    """
    instrume = str(header.get("INSTRUME", "")).strip()
    gain = header.get("GAIN")

    if instrume == "ZWO ASI676MC":
        if gain is not None:
            iso = _gain_to_iso(float(gain), _ASI676MC_GAIN_TABLE)
            tags.add_tag("ISOSpeedRatings", iso)

        # Analog balance (white balance)
        wb_r = header.get("WB_RED")
        wb_b = header.get("WB_BLUE")
        if wb_r is not None and wb_b is not None:
            wb_r_neutral, wb_b_neutral = 80, 100
            red_balance = float(wb_r) / float(wb_r_neutral)
            blue_balance = float(
                np.interp(
                    wb_b,
                    _ASI676MC_WB_BLUE_TABLE[:, 0],
                    _ASI676MC_WB_BLUE_TABLE[:, 1],
                )
            ) / float(
                np.interp(
                    wb_b_neutral,
                    _ASI676MC_WB_BLUE_TABLE[:, 0],
                    _ASI676MC_WB_BLUE_TABLE[:, 1],
                )
            )
            tags.add_tag("AnalogBalance", [red_balance, 1.0, blue_balance])
    else:
        if gain is not None:
            tags.add_tag("ISOSpeedRatings", int(gain))
        tags.add_tag("AnalogBalance", [1.0, 1.0, 1.0])


def _build_metadata_tags(
    header,
    data,
    auto_exposure: bool = True,
    ev_value=0,
    use_tone_curve: bool = True,
    wb_xy=None,
):
    """Build DNG MetadataTags from FITS header.

    Args:
        header: astropy FITS header
        data: 2D numpy array (CFA data)
        auto_exposure: compute BaselineExposure automatically
        ev_value: manual EV if not auto_exposure
        use_tone_curve: add S-curve XMP
        wb_xy: tuple (x, y) or None for default D50
    """
    from muimg.tiff_metadata import MetadataTags
    from muimg.raw_render import add_supported_xmp_from_dict

    tags = MetadataTags()

    # Bayer pattern from FITS header
    fits_pattern = header.get("BAYERPAT")
    if fits_pattern:
        pattern = str(fits_pattern).strip().upper()
        tags.add_tag("CFAPattern", pattern)

    # Camera model
    instrume = header.get("INSTRUME")
    if instrume:
        tags.add_tag("UniqueCameraModel", str(instrume))

    # Camera-specific tags
    _build_camera_tags(tags, header)

    # Software
    swcreate = header.get("SWCREATE")
    if swcreate:
        tags.add_tag("Software", str(swcreate))

    # Exposure time
    exptime = header.get("EXPTIME")
    if exptime is not None:
        tags.add_tag("ExposureTime", float(exptime))

    # Date/time
    date_obs = header.get("DATE-OBS")
    if date_obs is not None:
        dt = _parse_fits_datetime(str(date_obs))
        if dt is not None:
            tags.add_time_tags(dt, "original")

    # Black level
    blklevel = header.get("PEDESTAL")
    if blklevel is not None:
        tags.add_tag("BlackLevel", int(blklevel))

    # White level
    whtlevel = header.get("CWHITE")
    if whtlevel is not None:
        tags.add_tag("WhiteLevel", int(whtlevel))

    # Linear tone curve (bypass Adobe Camera Raw default)
    tags.add_tag("ProfileToneCurve", [0.0, 0.0, 1.0, 1.0])

    # White balance override
    if wb_xy is not None:
        tags.add_tag("AsShotWhiteXY", list(wb_xy))

    # Auto exposure + auto black level
    if auto_exposure:
        wl = int(header.get("CWHITE", 65535))
        stats = compute_image_stats(data, wl, percentiles=(1, 50))

        # Estimated black level (when PEDESTAL not in header)
        if blklevel is None and stats[1] > 0:
            tags.add_tag("BlackLevel", stats[1])

        # BaselineExposure from median
        if not stats["clipped"]:
            median_val = stats[50]
            if median_val > 0:
                target = wl * 0.06
                ev_shift = float(np.log2(target / median_val))
                tags.add_tag("BaselineExposure", ev_shift)
    else:
        if ev_value and float(ev_value) != 0:
            tags.add_tag("BaselineExposure", float(ev_value))

    # Build XMP properties (tone curve + AVM)
    xmp_props = {}

    if use_tone_curve:
        xmp_props["ToneCurvePV2012"] = [
            (0.0, 0.0),
            (64 / 255, 32 / 255),
            (128 / 255, 128 / 255),
            (192 / 255, 224 / 255),
            (1.0, 1.0),
        ]

    # AVM (Astronomy Visualization Metadata) from FITS header
    avm_props = fits_header_to_avm_xmp(header)
    xmp_props.update(avm_props)

    if xmp_props:
        add_supported_xmp_from_dict(tags, xmp_props)

    return tags


# =============================================================================
# Compression helpers
# =============================================================================


def _get_compression_map():
    from tifffile import COMPRESSION

    return {
        "uncompressed": COMPRESSION.NONE,
        "jpeg_lossless": COMPRESSION.JPEG,
        "jxl_lossless": COMPRESSION.JPEGXL_DNG,
        "jxl_lossy": COMPRESSION.JPEGXL_DNG,
    }


def _get_compression_args(compression_name: str) -> dict | None:
    if compression_name == "jpeg_lossless":
        return {"lossless": True}
    elif compression_name == "jxl_lossless":
        return {"distance": 0.0, "effort": 2}
    elif compression_name == "jxl_lossy":
        return {"distance": 0.5, "effort": 4}
    return None


# =============================================================================
# Batch conversion (ImageSequencePipeline)
# =============================================================================


def run_batch_fits_to_dng(
    fits_files,
    output_folder,
    compression_name="uncompressed",
    auto_exposure=True,
    ev_value=0,
    use_tone_curve=True,
    wb_xy=None,
    do_preview=False,
    do_fast_load=False,
    num_workers=4,
    on_task_done=None,
    log_callback=None,
):
    """Batch convert FITS files to DNG using ImageSequencePipeline.

    Reusable function called by both GUI and CLI.

    Args:
        fits_files: List of Path objects pointing to FITS files.
        output_folder: Path to output directory.
        compression_name: One of "uncompressed", "jpeg_lossless",
            "jxl_lossless", "jxl_lossy".
        auto_exposure: Whether to auto-compute BaselineExposure.
        ev_value: Manual EV if not auto_exposure.
        use_tone_curve: Whether to add default S-curve.
        wb_xy: (x, y) chromaticity tuple, or None for D50 default.
        do_preview: Whether to generate JPEG preview IFD.
        do_fast_load: Whether to generate pyramid levels.
        num_workers: Number of parallel consumer threads.
        on_task_done: Optional callback(completed, total) -> bool.
        log_callback: Optional callback(msg: str) for per-file status messages.

    Returns:
        dict with keys: "completed", "written", "skipped", "errored",
        "total", "elapsed", "queue_stats".
    """
    import io
    import threading
    import time

    import setproctitle
    setproctitle.setproctitle("mu-dng-converter: FITS -> DNG")

    from astropy.io import fits as astropy_fits
    from tifffile import COMPRESSION

    from muimg.dngio import (
        IfdDataSpec,
        PageEncoding,
        PreviewParams,
        PreviewScale,
        PyramidParams,
        write_dng_from_array,
    )
    from muimg.imgio import ImageSequencePipeline

    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)

    class _Counters:
        def __init__(self):
            self._lock = threading.Lock()
            self.written = 0
            self.skipped = 0
            self.errored = 0

        def inc(self, key: str):
            with self._lock:
                setattr(self, key, getattr(self, key) + 1)

    counts = _Counters()

    def _log(msg):
        if log_callback:
            log_callback(msg)

    # Compression setup
    comp_map = _get_compression_map()
    comp_enum = comp_map[compression_name]
    comp_args = _get_compression_args(compression_name)

    # Preview params (JPEG quality 90, quarter scale)
    preview_params = None
    if do_preview:
        preview_params = PreviewParams(
            scale=PreviewScale.QUARTER,
            compression=COMPRESSION.JPEG,
            compression_args={"level": 90},
        )

    # Pyramid params (JXL distance 1.0, 3 levels)
    pyramid_params = None
    if do_fast_load:
        pyramid_params = PyramidParams(
            levels=3,
            encoding=PageEncoding(
                compression=COMPRESSION.JPEGXL_DNG,
                compression_args={"distance": 1.0},
                tile_size=(256, 256),
            ),
        )

    def fits_consumer(task):
        index, file_path, blob = task
        try:
            with astropy_fits.open(io.BytesIO(blob)) as hdul:
                data = hdul[0].data
                header = hdul[0].header

            if data is None or data.ndim != 2:
                ndim = data.ndim if data is not None else None
                msg = (
                    f"Skipping {Path(file_path).name}"
                    f" (no 2D data, ndim={ndim})"
                )
                _log(msg)
                counts.inc("skipped")
                return (index, file_path, None)

            # Build metadata
            tags = _build_metadata_tags(
                header,
                data,
                auto_exposure=auto_exposure,
                ev_value=ev_value,
                use_tone_curve=use_tone_curve,
                wb_xy=wb_xy,
            )

            # Determine if image is CFA (color) or monochrome
            bayer_pat = header.get("BAYERPAT", "").strip().upper()
            is_cfa = bool(bayer_pat)

            # Encoding
            encoding = PageEncoding(
                compression=comp_enum,
                compression_args=comp_args,
                tile_size=(
                    (256, 256) if comp_enum != COMPRESSION.NONE else None
                ),
            )

            data_spec = IfdDataSpec(
                data=data,
                photometric="CFA" if is_cfa else "LINEAR_RAW",
                cfa_pattern=bayer_pat if is_cfa else None,
                encoding=encoding,
                extratags=tags,
            )

            # Write DNG to memory
            dng_buf = io.BytesIO()
            write_dng_from_array(
                destination_file=dng_buf,
                data_spec=data_spec,
                preview=preview_params,
                pyramid=pyramid_params,
                num_compression_workers=1,
            )
            return (index, file_path, dng_buf.getvalue())
        except Exception as e:
            msg = (
                f"Error: {Path(file_path).name}"
                f" ({type(e).__name__}): {e}"
            )
            _log(msg)
            counts.inc("errored")
            return (index, file_path, None)

    start_time = time.perf_counter()

    def counting_writer(result):
        """Writer that counts successful writes."""
        if result is None:
            return
        _, file_path, blob = result
        if blob is None:
            return
        out_file = output_path / Path(file_path).with_suffix(".dng").name
        try:
            with open(out_file, "wb") as f:
                f.write(blob)
            counts.inc("written")
        except Exception as e:
            msg = (
                f"Write failed: {Path(file_path).name}"
                f" ({type(e).__name__}): {e}"
            )
            _log(msg)
            counts.inc("errored")

    pipeline = ImageSequencePipeline(
        source_files=fits_files,
        output_folder=output_path,
        output_format="dng",
        consumer=fits_consumer,
        writer=counting_writer,
        num_workers=num_workers,
        task_name="FITS -> DNG",
        on_task_done=on_task_done,
    )
    pipeline.run()

    elapsed = time.perf_counter() - start_time

    return {
        "completed": pipeline._completed,
        "written": counts.written,
        "skipped": counts.skipped,
        "errored": counts.errored,
        "total": len(fits_files),
        "elapsed": elapsed,
        "queue_stats": pipeline.get_queue_stats(),
    }


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Convert FITS CFA data to DNG"
    )
    parser.add_argument(
        "input",
        type=Path,
        help=(
            "Input FITS file or folder. If a folder is given, all .fit and"
            " .fits files are converted."
        ),
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help=(
            "Output DNG path (single file) or output folder (batch)."
            " Default: same location as input with .dng extension."
        ),
    )
    parser.add_argument(
        "--compression",
        choices=["uncompressed", "jpeg_lossless", "jxl_lossless", "jxl_lossy"],
        default="uncompressed",
        help="Compression type (default: uncompressed)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of compression workers (default: 4)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Generate embedded JPEG preview",
    )
    parser.add_argument(
        "--fast-load",
        action="store_true",
        help="Embed fast-load pyramid levels",
    )
    parser.add_argument(
        "--no-auto-exposure",
        action="store_true",
        help="Disable automatic BaselineExposure calculation",
    )
    parser.add_argument(
        "--ev",
        type=float,
        default=0,
        help="Manual exposure shift in EV (used when --no-auto-exposure)",
    )
    parser.add_argument(
        "--no-tone-curve",
        action="store_true",
        help="Disable default S-curve tone curve",
    )
    parser.add_argument(
        "--wb-temperature",
        type=float,
        default=None,
        help="White balance color temperature in Kelvin",
    )
    parser.add_argument(
        "--wb-tint",
        type=float,
        default=0,
        help="White balance tint (default: 0)",
    )
    parser.add_argument(
        "--info",
        action="store_true",
        help="Print FITS header info and exit (no conversion)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="count",
        default=0,
        help="Verbosity: -v INFO (pipeline stats), -vv DEBUG",
    )
    args = parser.parse_args()

    if args.verbose:
        import logging
        level = logging.DEBUG if args.verbose >= 2 else logging.INFO
        logging.basicConfig(level=level, format="%(message)s")

    if not args.input.exists():
        print(f"ERROR: Not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    if args.info:
        if args.input.is_dir():
            print("ERROR: --info requires a single file", file=sys.stderr)
            sys.exit(1)
        print_fits_info(args.input)
        return

    # Resolve input file list
    if args.input.is_dir():
        fits_files = sorted(
            p
            for p in args.input.iterdir()
            if p.suffix.lower() in (".fits", ".fit")
        )
        if not fits_files:
            print(
                f"ERROR: No FITS files found in {args.input}",
                file=sys.stderr,
            )
            sys.exit(1)
        output_folder = args.output or args.input
    else:
        fits_files = [args.input]
        if args.output and args.output.is_dir():
            output_folder = args.output
        else:
            output_folder = (args.output or args.input.with_suffix(".dng")).parent
            # For single file with explicit -o pointing to a .dng, rename output
            if (
                args.output
                and not args.output.is_dir()
                and args.output.suffix.lower() == ".dng"
            ):
                output_folder = args.output.parent
                # Patch: rename the output file by injecting a rename step below

    # White balance -> xy chromaticity
    wb_xy = None
    if args.wb_temperature is not None:
        from muimg.raw_render import temp_tint_to_xy
        wb_xy = temp_tint_to_xy(args.wb_temperature, args.wb_tint)

    def _log(msg):
        print(msg)

    result = run_batch_fits_to_dng(
        fits_files=fits_files,
        output_folder=output_folder,
        compression_name=args.compression,
        auto_exposure=not args.no_auto_exposure,
        ev_value=args.ev,
        use_tone_curve=not args.no_tone_curve,
        wb_xy=wb_xy,
        do_preview=args.preview,
        do_fast_load=args.fast_load,
        num_workers=max(1, args.workers),
        log_callback=_log,
    )

    # For single-file with explicit -o <name>.dng, rename the output
    if (
        len(fits_files) == 1
        and args.output
        and not args.output.is_dir()
        and args.output.suffix.lower() == ".dng"
    ):
        default_name = fits_files[0].with_suffix(".dng").name
        produced = output_folder / default_name
        if produced.exists() and produced != args.output:
            produced.rename(args.output)

    elapsed = result["elapsed"]
    fps = result["written"] / elapsed if elapsed > 0 else 0
    parts = [f"{result['written']} written"]
    if result["skipped"]:
        parts.append(f"{result['skipped']} skipped")
    if result["errored"]:
        parts.append(f"{result['errored']} errors")
    print(
        f"Done: {', '.join(parts)} / {result['total']} total"
        f" in {elapsed:.1f}s ({fps:.1f} files/s)"
    )
    qs = result["queue_stats"]
    if "task_queue" in qs:
        q = qs["task_queue"]
        print(f"  Task queue: avg_depth={q['avg_depth']:.1f}, empty={q['empty_time']:.1f}s")
    if "writer_queue" in qs:
        q = qs["writer_queue"]
        print(f"  Writer queue: avg_depth={q['avg_depth']:.1f}, empty={q['empty_time']:.1f}s")


if __name__ == "__main__":
    main()
