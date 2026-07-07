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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from mu_dng_converter.common import parse_time_shift

# =============================================================================
# AVM (Astronomy Visualization Metadata) XMP mapping
# =============================================================================


def fits_header_to_avm_xmp(header) -> dict[str, Any]:
    """Extract AVM XMP properties from a FITS header using pyavm.

    Args:
        header: astropy FITS header object

    Returns:
        Dict with 'avm:'-prefixed keys suitable for XmpMetadata.from_attributes().
    """
    from pyavm import AVM

    attrs: dict[str, Any] = {}

    # Use pyavm to extract WCS metadata from FITS header
    try:
        avm = AVM.from_header(header)
        # Convert AVM object to dict with 'avm:' prefixes using official API
        for key, val in avm.to_keyword_dict().items():
            if val is not None:
                # Skip pyavm's internal non-standard keys
                if key == "Spatial.ReferenceDimension":
                    continue
                attrs[f'avm:{key}'] = val
    except Exception:
        pass  # pyavm may fail on incomplete headers

    # --- Independent spatial fallback checks ---
    # 1. Coordinate & Frame Check
    if "avm:Spatial.ReferenceValue" not in attrs:
        try:
            from astropy.coordinates import SkyCoord
            from astropy import units as u

            ra_val = header.get("RA") or header.get("OBJCTRA")
            dec_val = header.get("DEC") or header.get("OBJCTDEC")
            if ra_val is not None and dec_val is not None:
                coord = SkyCoord(str(ra_val), str(dec_val), unit=(u.hourangle, u.deg))
                attrs["avm:Spatial.ReferenceValue"] = [coord.ra.deg, coord.dec.deg]
                attrs["avm:Spatial.CoordinateFrame"] = "ICRS"
        except Exception:
            pass  # Failed parsing basic coordinates

    # 2. Add structural spatial keys if coordinates exist
    if "avm:Spatial.ReferenceValue" in attrs:
        attrs.setdefault("avm:Spatial.CoordinateDimension", [2, 2])
        attrs.setdefault("avm:Spatial.CoordsystemProjection", "TAN")

    # 3. Scale Check (independent)
    if "avm:Spatial.Scale" not in attrs:
        scale_val = header.get("SCALE") or header.get("CDELT2")
        if scale_val:
            try:
                deg_scale = float(scale_val) if float(scale_val) < 0.01 else float(scale_val) / 3600.0
                attrs["avm:Spatial.Scale"] = [-deg_scale, deg_scale]
            except (ValueError, TypeError):
                pass

    # 4. Rotation Check (independent)
    # Check both standard name and pyavm's internal names (RotationRef, PositionAngle)
    if not any(k in attrs for k in ("avm:Spatial.Rotation", "avm:Spatial.RotationRef", "avm:Spatial.PositionAngle")):
        rot_val = header.get("CROTA1") or header.get("CROTA2") or header.get("ROTATANG")
        if rot_val is not None:
            try:
                attrs["avm:Spatial.Rotation"] = float(rot_val)
            except (ValueError, TypeError):
                pass

    # 5. Reference Pixel Check (independent)
    if "avm:Spatial.ReferencePixel" not in attrs:
        naxis1 = header.get("NAXIS1") or header.get("IMAGEW")
        naxis2 = header.get("NAXIS2") or header.get("IMAGEH")
        if naxis1 and naxis2:
            try:
                attrs["avm:Spatial.ReferencePixel"] = [float(naxis1) / 2.0, float(naxis2) / 2.0]
            except (ValueError, TypeError):
                pass

    # Add non-WCS metadata from FITS header (only if not already set by pyavm)
    obj = header.get("OBJECT")
    if obj and str(obj).strip():
        attrs.setdefault("avm:Subject.Name", str(obj).strip())

    instrume = header.get("INSTRUME")
    if instrume and str(instrume).strip():
        attrs.setdefault("avm:Instrument", str(instrume).strip())

    telescop = header.get("TELESCOP")
    if telescop and str(telescop).strip():
        attrs.setdefault("avm:Facility", str(telescop).strip())

    filt = header.get("FILTER")
    if filt and str(filt).strip():
        attrs.setdefault("avm:Spectral.Bandpass", str(filt).strip())

    date_obs = header.get("DATE-OBS")
    if date_obs is not None:
        attrs.setdefault("avm:TemporalStartTime", str(date_obs).strip())

    exptime = header.get("EXPTIME")
    if exptime is not None:
        attrs.setdefault("avm:TemporalIntegrationTime", str(float(exptime)))

    observer = (
        header.get("OBSERVER")
        or header.get("AUTHOR")
        or header.get("CREATOR")
    )
    if observer and str(observer).strip():
        attrs.setdefault("avm:Creator", str(observer).strip())

    # Set defaults (only if not already set)
    attrs.setdefault("avm:Type", "Observation")
    attrs.setdefault("avm:MetadataVersion", "1.1")

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
# Derived from piecewise calibration, neutral at wb_b=100 (app default)
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

        # Analog balance (white balance) - only when WB data available
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

def _build_metadata_tags(
    header,
    data,
    auto_exposure: bool = True,
    ev_value=0,
    use_tone_curve: bool = True,
    wb_xy=None,
    time_offset_seconds: float = 0.0,
    time_timezone: str | None = None,
):
    """Build DNG MetadataTags from FITS header.

    Args:
        header: astropy FITS header
        data: 2D numpy array (CFA data)
        auto_exposure: compute BaselineExposure automatically
        ev_value: manual EV if not auto_exposure
        use_tone_curve: add S-curve XMP
        wb_xy: tuple (x, y) or None for default D50
        time_offset_seconds: signed seconds to add to the DATE-OBS datetime
        time_timezone: timezone offset string (e.g. "+02:00") for OffsetTime* tags
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
            if time_offset_seconds:
                dt = dt + timedelta(seconds=time_offset_seconds)
            tags.add_time_tags(dt, "original")

    # Timezone offset tags
    if time_timezone:
        tags.add_tag("OffsetTimeOriginal", time_timezone)
        tags.add_tag("OffsetTimeDigitized", time_timezone)
        tags.add_tag("OffsetTime", time_timezone)

    # Black level and white level
    blklevel = header.get("PEDESTAL")
    whtlevel = header.get("CWHITE")
    if blklevel is not None:
        tags.add_tag("BlackLevel", int(blklevel))
    if whtlevel is not None:
        tags.add_tag("WhiteLevel", int(whtlevel))

    # Linear tone curve (bypass Adobe Camera Raw default)
    # Only add when use_tone_curve=False to force linear processing
    if not use_tone_curve:
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
            tags.add_tag("BlackLevel", int(stats[1]))

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

    # Build XMP properties (AVM only - no tone curve)
    xmp_props = {}

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


def _get_compression_args(
    compression_name: str,
    jxl_distance: float | None = None,
    jxl_effort: int | None = None,
) -> dict | None:
    if compression_name == "jpeg_lossless":
        return {"lossless": True}
    elif compression_name in ("jxl_lossless", "jxl_lossy"):
        args = (
            {"distance": 0.0, "effort": 2}
            if compression_name == "jxl_lossless"
            else {"distance": 0.5, "effort": 4}
        )
        if jxl_distance is not None:
            args["distance"] = float(jxl_distance)
        if jxl_effort is not None:
            args["effort"] = int(jxl_effort)
        return args
    return None


# =============================================================================
# Batch conversion (ImageSequencePipeline)
# =============================================================================


def run_batch_fits_to_dng(
    fits_files,
    output_folder,
    compression_name="uncompressed",
    jxl_distance=None,
    jxl_effort=None,
    demosaic=False,
    demosaic_algorithm=None,
    scale=1.0,
    auto_exposure=True,
    ev_value=0,
    use_tone_curve=True,
    wb_xy=None,
    strip_tags=None,
    extra_tags=None,
    time_offset_seconds=0.0,
    time_timezone=None,
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
        jxl_distance: JXL distance (0=lossless, >0=lossy). None = default.
        jxl_effort: JXL effort (1-9). None = default.
        demosaic: If True, demosaic CFA to LINEAR_RAW.
        demosaic_algorithm: DemosaicAlgorithm enum value, or None for default.
        scale: Scale factor (0.125–1.0). If != 1.0, forces LINEAR_RAW.
        auto_exposure: Whether to auto-compute BaselineExposure.
        ev_value: Manual EV if not auto_exposure.
        use_tone_curve: Whether to add default S-curve.
        wb_xy: (x, y) chromaticity tuple, or None for D50 default.
        strip_tags: Set of tag name strings to strip from the generated
            metadata, or None.
        extra_tags: MetadataTags object with tags to add/override, or None.
        time_offset_seconds: Signed seconds to add to the DATE-OBS datetime.
        time_timezone: Timezone offset string (e.g. "+02:00") to set in all
            OffsetTime* tags, or None.
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
    from muimg.raw_render import DemosaicAlgorithm

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
    comp_args = _get_compression_args(compression_name, jxl_distance, jxl_effort)

    demosaic_algo = (
        demosaic_algorithm
        if demosaic_algorithm is not None
        else DemosaicAlgorithm.DNGSDK_BILINEAR
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

            from muimg.tiff_metadata import normalize_array_to_target_byteorder

            with astropy_fits.open(io.BytesIO(blob)) as hdul:
                data = hdul[0].data
                header = hdul[0].header

            # Fix endianness, convert unsupported types, reject invalid types
            # DNG spec: unsigned 8-32 bits/sample, or float16/float32
            # (signed integers not supported - SampleFormat must be 1 or 3)
            if data is not None:
                data = normalize_array_to_target_byteorder(data, '<')
                supported = (np.uint8, np.uint16, np.uint32, np.float16, np.float32)
                if data.dtype == np.float64:
                    _log(f"Converting float64 → float32: {Path(file_path).name}")
                    data = data.astype(np.float32)
                elif data.dtype in (np.int8, np.int16, np.int32):
                    raise ValueError(
                        f"Signed integer dtype {data.dtype} not supported by DNG. "
                        f"Convert to unsigned or float."
                    )
                elif data.dtype not in supported:
                    # Check for structured arrays (interferometry, tables, etc.)
                    if data.dtype.names is not None:
                        raise ValueError(f"Structured array (not image data): fields={list(data.dtype.names)}")
                    raise ValueError(f"Unsupported dtype: {data.dtype}")

            # DNG compression not supported for 32-bit or float data
            # Check after dtype conversion (float64 -> float32)
            no_compress_dtypes = (np.uint32, np.float16, np.float32)
            if (
                data is not None
                and data.dtype in no_compress_dtypes
                and comp_enum != COMPRESSION.NONE
            ):
                _log(
                    f"Warning: Compression disabled for {Path(file_path).name} "
                    f"(DNG does not support compression with {data.dtype})"
                )
                file_comp_enum = COMPRESSION.NONE
                file_comp_args = None
            else:
                file_comp_enum = comp_enum
                file_comp_args = comp_args

            if data is None or data.ndim != 2:
                if data is None:
                    reason = "no data in primary HDU"
                elif data.ndim == 3:
                    reason = f"3D data cube {data.shape} (needs flattening)"
                elif data.ndim == 1:
                    reason = f"1D spectrum {data.shape}"
                else:
                    reason = f"ndim={data.ndim}"
                msg = f"Skipping {Path(file_path).name} ({reason})"
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
                time_offset_seconds=time_offset_seconds,
                time_timezone=time_timezone,
            )

            # Apply user tag operations (strip first, then add/override)
            if strip_tags:
                for tag_name in strip_tags:
                    tags.remove_tag(tag_name)
            if extra_tags is not None:
                tags.extend(extra_tags)

            # Determine if image is CFA (color) or monochrome
            bayer_pat = header.get("BAYERPAT", "").strip().upper()
            is_cfa = bool(bayer_pat)

            # Encoding (use per-file compression, may be overridden for 32-bit/float)
            encoding = PageEncoding(
                compression=file_comp_enum,
                compression_args=file_comp_args,
                tile_size=(
                    (256, 256) if file_comp_enum != COMPRESSION.NONE else None
                ),
            )

            data_spec = IfdDataSpec(
                data=data,
                photometric="CFA" if is_cfa else "LINEAR_RAW",
                cfa_pattern=bayer_pat if is_cfa else None,
                encoding=encoding,
                extratags=tags,
            )

            # Build preview params per-file so scale reflects actual image dimensions
            preview_params = None
            if do_preview:
                h, w = data.shape[:2]
                scaled_h, scaled_w = int(h * scale), int(w * scale)
                min_dim = min(scaled_h, scaled_w)
                if min_dim >= 128 * 4:
                    preview_scale = PreviewScale.QUARTER
                elif min_dim >= 128 * 2:
                    preview_scale = PreviewScale.HALF
                else:
                    preview_scale = PreviewScale.FULL
                preview_params = PreviewParams(
                    scale=preview_scale,
                    compression=COMPRESSION.JPEG,
                    compression_args={"level": 90},
                )

            # Write DNG to memory
            dng_buf = io.BytesIO()
            write_dng_from_array(
                destination_file=dng_buf,
                data_spec=data_spec,
                scale=scale,
                demosaic=demosaic,
                demosaic_algorithm=demosaic_algo,
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
        "--jxl-distance",
        type=float,
        default=None,
        help="JXL distance (0=lossless, >0=lossy). Default: per-compression preset",
    )
    parser.add_argument(
        "--jxl-effort",
        type=int,
        default=None,
        help="JXL effort 1-9. Default: per-compression preset",
    )
    parser.add_argument(
        "--demosaic",
        action="store_true",
        help="Demosaic CFA to LINEAR_RAW",
    )
    parser.add_argument(
        "--demosaic-algorithm",
        default="DNGSDK_BILINEAR",
        choices=["DNGSDK_BILINEAR", "OPENCV_EA", "VNG", "RCD"],
        help="Demosaic algorithm (default: DNGSDK_BILINEAR)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Scale factor 0.125-1.0 (default: 1.0). If != 1.0, forces LINEAR_RAW",
    )
    parser.add_argument(
        "--strip-tag",
        action="append",
        default=None,
        metavar="NAME",
        help="Strip a tag from the output metadata (repeatable, comma-separated lists supported)",
    )
    parser.add_argument(
        "--tag",
        action="append",
        default=None,
        metavar="NAME=VALUE",
        help="Add/override a metadata tag (repeatable)",
    )
    parser.add_argument(
        "--time-shift",
        default=None,
        metavar="OFFSET",
        help='Shift DATE-OBS times, format "[+|-][D ]HH:MM[:SS]" (e.g. "+1:30", "-2 04:00:00")',
    )
    parser.add_argument(
        "--timezone",
        default=None,
        metavar="+HH:MM",
        help='Set OffsetTime* tags to this timezone offset (e.g. "+05:00")',
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

    # Demosaic algorithm
    from muimg.raw_render import DemosaicAlgorithm
    demosaic_algorithm = DemosaicAlgorithm.lookup(args.demosaic_algorithm)

    # Parse --strip-tag options, supporting comma-separated lists
    strip_tags = None
    if args.strip_tag:
        strip_tags = set()
        for tag_spec in args.strip_tag:
            for tag_name in tag_spec.split(","):
                tag_name = tag_name.strip()
                if tag_name:
                    strip_tags.add(tag_name)

    # Parse --tag NAME=VALUE options into MetadataTags
    extra_tags = None
    if args.tag:
        from muimg.tiff_metadata import MetadataTags
        extra_tags = MetadataTags()
        for tag_spec in args.tag:
            if "=" not in tag_spec:
                print(f"ERROR: Invalid tag format '{tag_spec}'. Use NAME=VALUE", file=sys.stderr)
                sys.exit(1)
            name, value = tag_spec.split("=", 1)
            try:
                extra_tags.add_tag(name.strip(), value.strip())
            except KeyError as e:
                print(f"ERROR: {e}", file=sys.stderr)
                sys.exit(1)

    # Parse --time-shift
    time_offset_seconds = 0.0
    if args.time_shift:
        try:
            time_offset_seconds = parse_time_shift(args.time_shift)
        except (ValueError, IndexError):
            print(f"ERROR: Could not parse time shift: {args.time_shift}", file=sys.stderr)
            sys.exit(1)

    def _log(msg):
        print(msg)

    result = run_batch_fits_to_dng(
        fits_files=fits_files,
        output_folder=output_folder,
        compression_name=args.compression,
        jxl_distance=args.jxl_distance,
        jxl_effort=args.jxl_effort,
        demosaic=args.demosaic,
        demosaic_algorithm=demosaic_algorithm,
        scale=max(0.125, min(1.0, args.scale)),
        auto_exposure=not args.no_auto_exposure,
        ev_value=args.ev,
        use_tone_curve=not args.no_tone_curve,
        wb_xy=wb_xy,
        strip_tags=strip_tags,
        extra_tags=extra_tags,
        time_offset_seconds=time_offset_seconds,
        time_timezone=args.timezone,
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
