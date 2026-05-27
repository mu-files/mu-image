"""FITS → DNG conversion view."""

import json
import threading
from pathlib import Path

import flet as ft


def _settings_path() -> Path:
    """Return path to persistent app settings file."""
    import platform
    if platform.system() == "Darwin":
        base = Path.home() / "Library" / "Application Support"
    elif platform.system() == "Windows":
        import os
        base = Path(os.environ.get("APPDATA", str(Path.home())))
    else:
        base = Path.home() / ".config"
    d = base / "mu-dng-converter"
    d.mkdir(parents=True, exist_ok=True)
    return d / "fits_settings.json"


def _load_settings() -> dict:
    try:
        return json.loads(_settings_path().read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_settings(settings: dict):
    _settings_path().write_text(json.dumps(settings, indent=2))


# =============================================================================
# FITS → DNG conversion helpers (replicated from mu-rasppi/fits2dng.py)
# =============================================================================

# ZWO ASI676MC gain table: (camera_gain, e-/ADU)
_ASI676MC_GAIN_TABLE = None  # lazy-loaded numpy array

# ZWO ASI676MC blue white balance table: (wb_b, balance)
_ASI676MC_WB_BLUE_TABLE = None  # lazy-loaded numpy array


def _get_asi676mc_tables():
    """Lazy-load ASI676MC calibration tables."""
    global _ASI676MC_GAIN_TABLE, _ASI676MC_WB_BLUE_TABLE
    if _ASI676MC_GAIN_TABLE is None:
        import numpy as np
        _ASI676MC_GAIN_TABLE = np.array([
            [0, 2.55], [50, 1.50], [100, 0.85],
            [150, 0.50], [200, 0.30],
        ])
        _ASI676MC_WB_BLUE_TABLE = np.array([
            [1, 0.01], [50, 0.59], [55, 0.73],
            [65, 0.87], [75, 1.00], [100, 1.34],
        ])
    return _ASI676MC_GAIN_TABLE, _ASI676MC_WB_BLUE_TABLE


def compute_baseline_exposure(data, white_level: int = 65535):
    """Estimate BaselineExposure (in EV) from raw CFA data."""
    import numpy as np
    num_bins = min(white_level, 10000)
    hist, bin_edges = np.histogram(
        data.ravel(), bins=num_bins, range=(0, white_level),
    )
    cdf = np.cumsum(hist).astype(np.float64) / np.sum(hist)

    clip_threshold = 0.975 * white_level
    clip_idx = np.searchsorted(bin_edges[:-1], clip_threshold)
    fraction_bright = 1.0 - (
        cdf[clip_idx - 1] if clip_idx > 0 else 0.0
    )
    if fraction_bright > 0.25:
        return None

    median_idx = np.searchsorted(cdf, 0.5)
    median_val = bin_edges[median_idx]
    if median_val <= 0:
        return None

    target = white_level * 0.06
    ev_shift = np.log2(target / median_val)
    return float(ev_shift)


def _parse_fits_datetime(date_str: str):
    """Parse FITS DATE-OBS string to datetime."""
    from datetime import datetime
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


def _build_metadata_tags(header, data, auto_exposure, ev_value,
                         use_tone_curve, wb_xy):
    """Build DNG MetadataTags from FITS header.

    Args:
        header: astropy FITS header
        data: 2D numpy array (CFA data)
        auto_exposure: bool - compute BaselineExposure automatically
        ev_value: float - manual EV if not auto
        use_tone_curve: bool - add S-curve XMP
        wb_xy: tuple (x, y) or None for default D50
    """
    import numpy as np
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

    # Exposure
    if auto_exposure:
        wl = int(header.get("CWHITE", 65535))
        ev_shift = compute_baseline_exposure(data, wl)
        if ev_shift is not None:
            tags.add_tag("BaselineExposure", ev_shift)
    else:
        if ev_value and float(ev_value) != 0:
            tags.add_tag("BaselineExposure", float(ev_value))

    # Tone curve XMP
    if use_tone_curve:
        tone_curve = [
            (0.0, 0.0),
            (64 / 255, 32 / 255),
            (128 / 255, 128 / 255),
            (192 / 255, 224 / 255),
            (1.0, 1.0),
        ]
        add_supported_xmp_from_dict(tags, {
            'ToneCurvePV2012': tone_curve,
        })

    return tags


def _build_camera_tags(tags, header):
    """Add camera-specific tags based on INSTRUME header."""
    import numpy as np

    instrume = str(header.get("INSTRUME", "")).strip()
    gain = header.get("GAIN")

    if instrume == "ZWO ASI676MC":
        gain_table, wb_blue_table = _get_asi676mc_tables()
        if gain is not None:
            e = np.interp(
                float(gain), gain_table[:, 0], gain_table[:, 1],
            )
            iso = int(100 * gain_table[0, 1] / e)
            tags.add_tag("ISOSpeedRatings", iso)

        wb_r = header.get("WB_RED")
        wb_b = header.get("WB_BLUE")
        if wb_r is not None and wb_b is not None:
            wb_r_neutral, wb_b_neutral = 80, 100
            red_balance = float(wb_r) / float(wb_r_neutral)
            blue_balance = float(
                np.interp(
                    wb_b, wb_blue_table[:, 0], wb_blue_table[:, 1],
                )
            ) / float(
                np.interp(
                    wb_b_neutral,
                    wb_blue_table[:, 0],
                    wb_blue_table[:, 1],
                )
            )
            tags.add_tag(
                "AnalogBalance", [red_balance, 1.0, blue_balance],
            )
    else:
        if gain is not None:
            tags.add_tag("ISOSpeedRatings", int(gain))
        tags.add_tag("AnalogBalance", [1.0, 1.0, 1.0])


# Compression name → tifffile COMPRESSION mapping
_COMPRESSION_MAP = None


def _get_compression_map():
    global _COMPRESSION_MAP
    if _COMPRESSION_MAP is None:
        from tifffile import COMPRESSION
        _COMPRESSION_MAP = {
            "uncompressed": COMPRESSION.NONE,
            "jpeg_lossless": COMPRESSION.JPEG,
            "jxl_lossless": COMPRESSION.JPEGXL_DNG,
            "jxl_lossy": COMPRESSION.JPEGXL_DNG,
        }
    return _COMPRESSION_MAP


def _get_compression_args(compression_name: str) -> dict | None:
    if compression_name == "jpeg_lossless":
        return {'lossless': True}
    elif compression_name == "jxl_lossless":
        return {'distance': 0.0, 'effort': 2}
    elif compression_name == "jxl_lossy":
        return {'distance': 0.5, 'effort': 4}
    return None


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
):
    """Batch convert FITS files to DNG using ImageSequencePipeline.

    Reusable function called by both GUI and potentially CLI.

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

    Returns:
        dict with keys: "completed", "total", "elapsed", "queue_stats".
    """
    import io
    import time

    import setproctitle
    setproctitle.setproctitle("mu-dng-converter: FITS → DNG")

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
            compression_args={'level': 90},
        )

    # Pyramid params (JXL distance 1.0, 3 levels)
    pyramid_params = None
    if do_fast_load:
        pyramid_params = PyramidParams(
            levels=3,
            encoding=PageEncoding(
                compression=COMPRESSION.JPEGXL_DNG,
                compression_args={'distance': 1.0},
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
                import logging
                logging.getLogger(__name__).warning(
                    f"Frame {index}: Skipping {Path(file_path).name} "
                    f"(no 2D data, ndim={data.ndim if data is not None else 'None'})")
                return (index, file_path, None)

            # Build metadata
            tags = _build_metadata_tags(
                header, data,
                auto_exposure=auto_exposure,
                ev_value=ev_value,
                use_tone_curve=use_tone_curve,
                wb_xy=wb_xy,
            )

            # Encoding
            encoding = PageEncoding(
                compression=comp_enum,
                compression_args=comp_args,
                tile_size=(
                    (256, 256)
                    if comp_enum != COMPRESSION.NONE
                    else None
                ),
            )

            data_spec = IfdDataSpec(
                data=data,
                photometric="CFA",
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
            import logging
            logging.getLogger(__name__).warning(
                f"Frame {index}: Error converting {Path(file_path).name} "
                f"({type(e).__name__}): {e}")
            return (index, file_path, None)

    start_time = time.perf_counter()

    pipeline = ImageSequencePipeline(
        source_files=fits_files,
        output_folder=output_path,
        output_format="dng",
        consumer=fits_consumer,
        num_workers=num_workers,
        task_name="FITS → DNG",
        on_task_done=on_task_done,
    )
    pipeline.run()

    elapsed = time.perf_counter() - start_time

    return {
        "completed": pipeline._completed,
        "total": len(fits_files),
        "elapsed": elapsed,
        "queue_stats": pipeline.get_queue_stats(),
    }


def build_fits_view(page: ft.Page) -> ft.Control:
    """Build the FITS → DNG conversion tab content."""

    _settings = _load_settings()
    state = {
        "running": False,
        "cancel": False,
        "last_input_dir": _settings.get("last_input_dir"),
        "last_output_dir": _settings.get("last_output_dir"),
    }

    # --- Controls ---
    input_path_text = ft.Text(
        "No folder selected", size=13,
        overflow=ft.TextOverflow.ELLIPSIS,
        no_wrap=True, expand=True,
    )
    output_path_text = ft.Text(
        "No folder selected", size=13,
        overflow=ft.TextOverflow.ELLIPSIS,
        no_wrap=True, expand=True,
    )

    input_mode = ft.Dropdown(
        value="folder",
        options=[
            ft.dropdown.Option("folder", "Folder"),
            ft.dropdown.Option("files", "Files"),
        ],
        width=110,
        text_size=12,
        content_padding=ft.Padding(left=8, top=4, right=8, bottom=4),
    )

    # White balance presets (temp K, tint)
    WB_PRESETS = {
        "d50": (5000, 0),
        "daylight": (5500, 10),
        "cloudy": (6500, 10),
        "shade": (7500, 10),
        "tungsten": (2850, 0),
        "fluorescent": (3800, 21),
        "flash": (5500, 0),
        "custom": (None, None),
    }

    wb_dropdown = ft.Dropdown(
        label="White Balance",
        value="d50",
        options=[
            ft.dropdown.Option("d50", "D50 (Neutral)"),
            ft.dropdown.Option("daylight", "Daylight"),
            ft.dropdown.Option("cloudy", "Cloudy"),
            ft.dropdown.Option("shade", "Shade"),
            ft.dropdown.Option("tungsten", "Tungsten"),
            ft.dropdown.Option("fluorescent", "Fluorescent"),
            ft.dropdown.Option("flash", "Flash"),
            ft.dropdown.Option("custom", "Custom"),
        ],
        width=180,
    )
    temperature = ft.TextField(
        label="Temperature (K)", value="5000", width=140,
        keyboard_type=ft.KeyboardType.NUMBER, disabled=True,
    )
    tint = ft.TextField(
        label="Tint", value="0", width=100,
        keyboard_type=ft.KeyboardType.NUMBER, disabled=True,
    )

    auto_exposure = ft.Checkbox(label="Auto Exposure", value=True)
    exposure = ft.TextField(
        label="Exposure (EV)", value="0", width=130,
        keyboard_type=ft.KeyboardType.NUMBER, disabled=True,
    )

    tone_curve = ft.Checkbox(label="Default Tone Curve", value=True)

    # Output options
    num_workers = ft.TextField(
        label="Workers", value="4", width=100,
        keyboard_type=ft.KeyboardType.NUMBER,
    )

    compression = ft.Dropdown(
        label="Compression",
        value="uncompressed",
        options=[
            ft.dropdown.Option("uncompressed", "Uncompressed"),
            ft.dropdown.Option("jpeg_lossless", "JPEG Lossless"),
            ft.dropdown.Option("jxl_lossless", "JXL Lossless"),
            ft.dropdown.Option("jxl_lossy", "JXL Lossy"),
        ],
        width=180,
    )

    preview = ft.Checkbox(label="JPEG Preview", value=False)
    fast_load = ft.Checkbox(label="Embed Fast Load Data", value=False)

    # Progress
    progress_bar = ft.ProgressBar(value=0, visible=False)
    progress_text = ft.Text("", size=12)
    log_text = ft.TextField(
        multiline=True, read_only=True, text_size=11, expand=True,
    )

    _btn_style = ft.ButtonStyle(
        bgcolor=ft.Colors.with_opacity(0.15, ft.Colors.WHITE),
    )
    run_button = ft.Button(
        content="Run", icon=ft.Icons.PLAY_ARROW, style=_btn_style,
    )
    cancel_button = ft.Button(
        content="Cancel", icon=ft.Icons.STOP,
        visible=False, style=_btn_style,
    )

    # --- Event handlers ---
    async def pick_input(e):
        from mu_dng_converter.dialogs import IS_MACOS, pick_directory, pick_files

        mode = input_mode.value
        initial = state["last_input_dir"]
        if mode == "folder":
            if IS_MACOS:
                result = pick_directory("Select FITS input folder", initial)
            else:
                result = await ft.FilePicker().get_directory_path(
                    dialog_title="Select FITS input folder",
                    initial_directory=initial,
                )
            if result:
                state["last_input_dir"] = result
                state["input_files"] = None
                input_path_text.value = result
                input_path_text.tooltip = result
                _save_settings({
                    "last_input_dir": state["last_input_dir"],
                    "last_output_dir": state["last_output_dir"],
                })
                page.update()
        else:
            if IS_MACOS:
                paths = pick_files(
                    "Select FITS file(s)", initial, ["fits", "fit"], True
                )
            else:
                files = await ft.FilePicker().pick_files(
                    dialog_title="Select FITS file(s)",
                    initial_directory=initial,
                    allowed_extensions=["fits", "fit", "FITS", "FIT"],
                    file_type=ft.FilePickerFileType.CUSTOM,
                    allow_multiple=True,
                )
                paths = [f.path for f in files] if files else None
            if paths:
                state["last_input_dir"] = str(Path(paths[0]).parent)
                state["input_files"] = paths
                if len(paths) == 1:
                    input_path_text.value = paths[0]
                    input_path_text.tooltip = paths[0]
                else:
                    input_path_text.value = (
                        f"{len(paths)} files in "
                        f"{Path(paths[0]).parent}"
                    )
                    input_path_text.tooltip = str(
                        Path(paths[0]).parent
                    )
                _save_settings({
                    "last_input_dir": state["last_input_dir"],
                    "last_output_dir": state["last_output_dir"],
                })
                page.update()

    async def pick_output(e):
        from mu_dng_converter.dialogs import IS_MACOS, pick_directory

        initial = state["last_output_dir"]
        if IS_MACOS:
            result = pick_directory("Select output folder", initial)
        else:
            result = await ft.FilePicker().get_directory_path(
                dialog_title="Select output folder",
                initial_directory=initial,
            )
        if result:
            state["last_output_dir"] = result
            output_path_text.value = result
            output_path_text.tooltip = result
            _save_settings({
                "last_input_dir": state["last_input_dir"],
                "last_output_dir": state["last_output_dir"],
            })
            page.update()

    def on_wb_changed(e):
        preset = wb_dropdown.value
        if preset == "custom":
            temperature.disabled = False
            tint.disabled = False
        else:
            t, ti = WB_PRESETS[preset]
            temperature.value = str(t) if t is not None else ""
            tint.value = str(ti) if ti is not None else ""
            temperature.disabled = True
            tint.disabled = True
        page.update()

    def on_auto_exposure_changed(e):
        exposure.disabled = auto_exposure.value
        page.update()

    wb_dropdown.on_select = on_wb_changed
    auto_exposure.on_change = on_auto_exposure_changed

    def on_cancel(e):
        state["cancel"] = True
        state["log"] = (
            (state.get("log") or "") + "Cancellation requested...\n"
        )
        page.update()

    def on_run(e):
        inp = input_path_text.value
        out = output_path_text.value
        if (
            not inp
            or inp == "No folder selected"
            or not out
            or out == "No folder selected"
        ):
            log_text.value = (
                "ERROR: Select input and output folders first.\n"
                + (log_text.value or "")
            )
            page.update()
            return

        state["running"] = True
        state["cancel"] = False
        state["finished"] = False
        state["progress_fraction"] = 0
        state["progress_text"] = ""
        state["log"] = ""
        run_button.visible = False
        cancel_button.visible = True
        progress_bar.visible = True
        progress_bar.value = 0
        state["_old_log"] = log_text.value or ""
        page.update()

        thread = threading.Thread(
            target=run_conversion, args=(inp, out), daemon=True,
        )
        thread.start()
        page.run_task(_poll_ui)

    run_button.on_click = on_run
    cancel_button.on_click = on_cancel

    # --- Conversion logic (runs in thread) ---
    def run_conversion(input_path, output_path):
        from muimg.raw_render import temp_tint_to_xy

        try:
            input_files = state.get("input_files")
            if input_files:
                fits_files = sorted(Path(f) for f in input_files)
            else:
                input_dir = Path(input_path)
                fits_files = sorted(
                    list(input_dir.glob("*.fits"))
                    + list(input_dir.glob("*.fit"))
                    + list(input_dir.glob("*.FITS"))
                    + list(input_dir.glob("*.FIT"))
                )
            if not fits_files:
                log(f"No FITS files found in {input_path}")
                finish()
                return

            total = len(fits_files)
            log(f"Input: {total} FITS files → DNG")
            log(f"Output: {output_path}")

            # White balance → xy chromaticity
            wb_preset = wb_dropdown.value
            if wb_preset == "d50":
                wb_xy = None
            else:
                temp_val = float(temperature.value or 5000)
                tint_val = float(tint.value or 0)
                wb_xy = temp_tint_to_xy(temp_val, tint_val)

            def on_task_done(completed, total):
                update_progress(
                    completed / total, f"{completed}/{total}",
                )
                return state["cancel"]

            result = run_batch_fits_to_dng(
                fits_files=fits_files,
                output_folder=output_path,
                compression_name=compression.value,
                auto_exposure=auto_exposure.value,
                ev_value=exposure.value,
                use_tone_curve=tone_curve.value,
                wb_xy=wb_xy,
                do_preview=preview.value,
                do_fast_load=fast_load.value,
                num_workers=int(num_workers.value),
                on_task_done=on_task_done,
            )

            fps = result['completed'] / result['elapsed'] if result['elapsed'] > 0 else 0
            log(
                f"Done: {result['completed']}/{result['total']} "
                f"files in {result['elapsed']:.1f}s "
                f"({fps:.1f} files/s, "
                f"{int(num_workers.value)} workers)"
            )
            stats = result.get('queue_stats', {})
            if 'task_queue' in stats:
                q = stats['task_queue']
                log(
                    f"  Task queue: avg_depth="
                    f"{q['avg_depth']:.1f}, "
                    f"empty={q['empty_time']:.1f}s"
                )
            if 'writer_queue' in stats:
                q = stats['writer_queue']
                log(
                    f"  Writer queue: avg_depth="
                    f"{q['avg_depth']:.1f}, "
                    f"empty={q['empty_time']:.1f}s"
                )
        except Exception as ex:
            log(f"ERROR: {ex}")
        finally:
            finish()

    # --- Helpers ---
    def update_progress(fraction, text):
        state["progress_fraction"] = fraction
        state["progress_text"] = text

    _MAX_LOG_LINES = 100

    def _build_display_log(current_log, old_log):
        parts = [
            p for p in [current_log.rstrip(), old_log.rstrip()] if p
        ]
        combined = ("\n" + "\u2500" * 40 + "\n").join(parts)
        lines = combined.split("\n")
        if len(lines) > _MAX_LOG_LINES:
            lines = lines[:_MAX_LOG_LINES]
        return "\n".join(lines)

    def log(message):
        state["log"] = (state.get("log") or "") + message + "\n"

    def finish():
        state["running"] = False
        state["finished"] = True

    async def _poll_ui():
        import asyncio
        prev_frac = -1
        prev_log = ""
        while state["running"]:
            frac = state.get("progress_fraction", 0)
            text = state.get("progress_text", "")
            log_val = state.get("log", "")
            changed = False
            if frac != prev_frac:
                progress_bar.value = frac
                progress_text.value = text
                prev_frac = frac
                changed = True
            if log_val != prev_log:
                log_text.value = _build_display_log(
                    log_val, state.get("_old_log", ""),
                )
                prev_log = log_val
                changed = True
            if changed:
                page.update()
            await asyncio.sleep(0.2)
        # Final flush
        progress_bar.value = state.get("progress_fraction", 0)
        progress_text.value = state.get("progress_text", "")
        remaining = state.get("log", "")
        if remaining:
            log_text.value = _build_display_log(
                remaining, state.get("_old_log", ""),
            )
        run_button.visible = True
        cancel_button.visible = False
        page.update()

    # --- Build layout ---
    input_btn = ft.Button(
        content=ft.Row(
            [ft.Icon(ft.Icons.FOLDER_OPEN), ft.Text("Select Input")],
            alignment=ft.MainAxisAlignment.START, spacing=8,
        ),
        on_click=pick_input, style=_btn_style, width=220,
    )
    output_btn = ft.Button(
        content=ft.Row(
            [ft.Icon(ft.Icons.FOLDER_OPEN),
             ft.Text("Select Output Folder")],
            alignment=ft.MainAxisAlignment.START, spacing=8,
        ),
        on_click=pick_output, style=_btn_style, width=220,
    )

    return ft.Column(
        controls=[
            ft.Container(
                content=ft.Row(
                    controls=[
                        run_button, cancel_button, progress_text,
                    ],
                    alignment=ft.MainAxisAlignment.START,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=12,
                ),
                padding=ft.Padding(left=0, top=8, right=0, bottom=0),
            ),
            ft.Divider(height=8),
            ft.Row(controls=[input_btn, input_mode, input_path_text]),
            ft.Row(controls=[output_btn, output_path_text]),
            ft.Divider(height=8),
            ft.Text(
                "Rendering Parameters",
                weight=ft.FontWeight.BOLD, size=13,
            ),
            ft.Row(controls=[auto_exposure, tone_curve]),
            ft.Row(
                controls=[
                    wb_dropdown, temperature, tint, exposure,
                ],
                wrap=True,
            ),
            ft.Divider(height=8),
            ft.Text(
                "Output Options",
                weight=ft.FontWeight.BOLD, size=13,
            ),
            ft.Row(
                controls=[compression, num_workers],
            ),
            ft.Row(
                controls=[preview, fast_load],
            ),
            ft.Divider(height=8),
            progress_bar,
            ft.Container(content=log_text, expand=True),
        ],
        expand=True,
    )
