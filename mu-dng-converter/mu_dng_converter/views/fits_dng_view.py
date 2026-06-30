"""FITS → DNG conversion view."""

import json
import threading
from pathlib import Path

import flet as ft

from mu_dng_converter.fits2dng import run_batch_fits_to_dng


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



def build_fits_view(page: ft.Page, dir_picker: ft.FilePicker | None = None,
                    file_picker: ft.FilePicker | None = None) -> ft.Control:
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
    def _clamp_workers(e):
        try:
            v = max(1, min(8, int(num_workers.value)))
        except (ValueError, TypeError):
            v = 4
        num_workers.value = str(v)
        page.update()

    num_workers = ft.TextField(
        label="Workers (1-8)", value="4", width=100,
        keyboard_type=ft.KeyboardType.NUMBER,
        on_blur=_clamp_workers,
        on_submit=_clamp_workers,
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
        width=200,
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
        padding=ft.Padding(left=10, top=6, right=10, bottom=6),
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
        from mu_dng_converter.dialogs import pick_directory_async, pick_files_async

        mode = input_mode.value
        initial = state["last_input_dir"]
        if mode == "folder":
            try:
                result = await pick_directory_async(
                    "Select FITS input folder", initial,
                    can_create_directories=False, picker=dir_picker)
            except Exception as ex:
                log(f"[error] folder picker failed: {ex}")
                page.update()
                return
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
            try:
                paths = await pick_files_async(
                    "Select FITS file(s)", initial, ["fits", "fit"],
                    allow_multiple=True, picker=file_picker)
            except Exception as ex:
                log(f"[error] file picker failed: {ex}")
                page.update()
                return
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
        from mu_dng_converter.dialogs import pick_directory_async

        initial = state["last_output_dir"]
        result = await pick_directory_async("Select output folder", initial, picker=dir_picker)
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

    # --- Conversion logic (worker function for BatchRunner) ---
    def run_conversion_worker(output_path, worker_state):
        """Worker function that runs in background thread."""
        from muimg.raw_render import temp_tint_to_xy
        from mu_dng_converter.batch_runner import make_state_logger

        _log = make_state_logger(worker_state)

        try:
            fits_files = worker_state["_fits_files"]
            total = len(fits_files)
            _log(f"Input: {total} FITS files from {Path(output_path).parent}")
            _log(f"Output: {output_path}")

            # White balance → xy chromaticity
            wb_preset = wb_dropdown.value
            if wb_preset == "d50":
                wb_xy = None
            else:
                temp_val = float(temperature.value or 5000)
                tint_val = float(tint.value or 0)
                wb_xy = temp_tint_to_xy(temp_val, tint_val)

            def on_task_done(completed, total):
                worker_state["progress_fraction"] = completed / total if total > 0 else 0
                worker_state["progress_text"] = f"{completed}/{total}"
                return worker_state["cancel"]

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
                num_workers=max(1, min(8, int(num_workers.value))),
                on_task_done=on_task_done,
                log_callback=_log,
            )

            elapsed = result['elapsed']
            fps = result['written'] / elapsed if elapsed > 0 else 0
            parts = [f"{result['written']} written"]
            if result['skipped']:
                parts.append(f"{result['skipped']} skipped")
            if result['errored']:
                parts.append(f"{result['errored']} errors")
            _log(
                f"Done: {', '.join(parts)} / {result['total']} total "
                f"in {elapsed:.1f}s ({fps:.1f} files/s)"
            )
            stats = result.get('queue_stats', {})
            if 'task_queue' in stats:
                q = stats['task_queue']
                _log(
                    f"  Task queue: avg_depth="
                    f"{q['avg_depth']:.1f}, "
                    f"empty={q['empty_time']:.1f}s"
                )
            if 'writer_queue' in stats:
                q = stats['writer_queue']
                _log(
                    f"  Writer queue: avg_depth="
                    f"{q['avg_depth']:.1f}, "
                    f"empty={q['empty_time']:.1f}s"
                )
        except Exception as ex:
            _log(f"ERROR: {ex}")
        finally:
            worker_state["running"] = False
            worker_state["finished"] = True

    # --- Setup BatchRunner for run/cancel/polling logic ---
    from mu_dng_converter.batch_runner import BatchRunner

    runner = BatchRunner(
        page=page,
        state=state,
        worker_fn=run_conversion_worker,
        run_button=run_button,
        cancel_button=cancel_button,
        progress_bar=progress_bar,
        log_text=log_text,
    )

    async def on_run_wrapper(e):
        """Custom on_run with FITS file handling."""
        from mu_dng_converter.dialogs import check_overwrite

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

        # Gather file list before starting (FITS files)
        input_files = state.get("input_files")
        if input_files:
            fits_files = sorted(Path(f) for f in input_files)
        else:
            input_dir = Path(inp)
            fits_files = sorted(
                p for p in input_dir.iterdir()
                if p.suffix.lower() in (".fits", ".fit")
            )
        if not fits_files:
            log_text.value = (
                f"ERROR: No FITS files found in {inp}\n"
                + (log_text.value or "")
            )
            page.update()
            return

        # Check for existing output files (.dng extension)
        output_dir = Path(out)
        existing = [
            f for f in fits_files
            if (output_dir / (f.stem + ".dng")).exists()
        ]
        if existing:
            action = await check_overwrite(page, len(existing), len(fits_files))
            if action == "cancel":
                return
            elif action == "skip":
                fits_files = [f for f in fits_files if f not in existing]
                if not fits_files:
                    log_text.value = (
                        "All files already exist — nothing to do.\n"
                        + (log_text.value or "")
                    )
                    page.update()
                    return

        # Store the filtered file list for the worker
        state["_fits_files"] = fits_files
        state["input_files"] = [str(f) for f in fits_files]

        # Call common runner logic
        await runner.on_run(
            input_path_text=input_path_text,
            output_path_text=output_path_text,
            persist_settings_fn=lambda: None,
        )

    run_button.on_click = on_run_wrapper
    cancel_button.on_click = runner.on_cancel

    # --- Build layout ---
    input_btn = ft.Button(
        content=ft.Row(
            [ft.Icon(ft.Icons.FOLDER_OPEN), ft.Text("Select Input")],
            alignment=ft.MainAxisAlignment.START, spacing=8,
        ),
        on_click=pick_input, style=_btn_style, width=180,
    )
    output_btn = ft.Button(
        content=ft.Row(
            [ft.Icon(ft.Icons.FOLDER_OPEN),
             ft.Text("Select Output Folder")],
            alignment=ft.MainAxisAlignment.START, spacing=8,
        ),
        on_click=pick_output, style=_btn_style, width=180,
    )

    return ft.Column(
        controls=[
            ft.Container(
                height=48,  # Fixed height for Run row
                content=ft.Row(
                    controls=[run_button, cancel_button, progress_text],
                    alignment=ft.MainAxisAlignment.START,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                    spacing=12,
                ),
                padding=ft.Padding(left=0, top=6, right=0, bottom=6),
            ),
            ft.Divider(height=1),
            ft.Container(
                height=40,  # Fixed height for Input row
                content=ft.Row(controls=[input_btn, input_mode, input_path_text]),
            ),
            ft.Container(
                height=40,  # Fixed height for Output row
                content=ft.Row(controls=[output_btn, output_path_text]),
            ),
            ft.Divider(height=1),
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
            ft.Divider(height=1),
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
            ft.Divider(height=1),
            progress_bar,
            ft.Container(content=log_text, expand=True),
        ],
        expand=True,
        spacing=4,
    )
