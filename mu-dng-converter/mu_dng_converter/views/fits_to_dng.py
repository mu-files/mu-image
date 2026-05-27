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
        mode = input_mode.value
        initial = state["last_input_dir"]
        if mode == "folder":
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
            files = await ft.FilePicker().pick_files(
                dialog_title="Select FITS file(s)",
                initial_directory=initial,
                allowed_extensions=["fits", "fit", "FITS", "FIT"],
                file_type=ft.FilePickerFileType.CUSTOM,
                allow_multiple=True,
            )
            if files:
                paths = [f.path for f in files]
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
        initial = state["last_output_dir"]
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

            # TODO: Wire up actual conversion using convert_fits_to_dng
            log("Conversion not yet implemented.")
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
