"""DNG → TIF/Video conversion view."""

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
    return d / "settings.json"


def _load_settings() -> dict:
    try:
        return json.loads(_settings_path().read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_settings(settings: dict):
    _settings_path().write_text(json.dumps(settings, indent=2))


def build_dng_view(page: ft.Page) -> ft.Control:
    """Build the DNG conversion tab content."""

    _settings = _load_settings()
    state = {
        "running": False,
        "cancel": False,
        "last_input_dir": _settings.get("last_input_dir"),
        "last_output_dir": _settings.get("last_output_dir"),
    }

    # --- Controls ---
    input_path_text = ft.Text(
        "No folder selected", size=13, overflow=ft.TextOverflow.ELLIPSIS,
        no_wrap=True, expand=True,
    )
    output_path_text = ft.Text(
        "No folder selected", size=13, overflow=ft.TextOverflow.ELLIPSIS,
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

    mode_dropdown = ft.Dropdown(
        label="Mode",
        value="tif",
        options=[
            ft.dropdown.Option("tif", "DNG → TIF (with metadata)"),
            ft.dropdown.Option("jpg", "DNG → JPG"),
            ft.dropdown.Option("video", "DNG → Video (MP4)"),
        ],
        width=320,
    )

    # White balance presets (temp K, tint)
    WB_PRESETS = {
        "as_shot": (None, None),
        "daylight": (5500, 10),
        "cloudy": (6500, 10),
        "shade": (7500, 10),
        "tungsten": (2850, 0),
        "fluorescent": (3800, 21),
        "flash": (5500, 0),
        "custom": (None, None),
    }

    use_xmp = ft.Checkbox(label="Use XMP metadata", value=True)
    wb_dropdown = ft.Dropdown(
        label="White Balance",
        value="as_shot",
        options=[
            ft.dropdown.Option("as_shot", "As Shot"),
            ft.dropdown.Option("daylight", "Daylight"),
            ft.dropdown.Option("cloudy", "Cloudy"),
            ft.dropdown.Option("shade", "Shade"),
            ft.dropdown.Option("tungsten", "Tungsten"),
            ft.dropdown.Option("fluorescent", "Fluorescent"),
            ft.dropdown.Option("flash", "Flash"),
            ft.dropdown.Option("custom", "Custom"),
        ],
        width=180,
        disabled=True,
    )
    temperature = ft.TextField(
        label="Temperature (K)", width=140, keyboard_type=ft.KeyboardType.NUMBER,
        disabled=True,
    )
    tint = ft.TextField(
        label="Tint", width=100, keyboard_type=ft.KeyboardType.NUMBER,
        disabled=True,
    )
    exposure = ft.TextField(
        label="Exposure (EV)", value="0", width=130,
        keyboard_type=ft.KeyboardType.NUMBER, disabled=True,
    )

    bit_depth = ft.Dropdown(
        label="Bit Depth",
        value="16",
        options=[
            ft.dropdown.Option("8", "8-bit"),
            ft.dropdown.Option("16", "16-bit"),
        ],
        width=120,
    )
    num_workers = ft.TextField(
        label="Workers", value="4", width=100,
        keyboard_type=ft.KeyboardType.NUMBER,
    )

    # Video options
    resolution = ft.TextField(label="Resolution", value="1920x1080", width=140)
    codec = ft.Dropdown(
        label="Codec", value="h264", width=120,
        options=[
            ft.dropdown.Option("h264", "H.264"),
            ft.dropdown.Option("hevc", "HEVC"),
            ft.dropdown.Option("vp9", "VP9"),
        ],
    )
    crf = ft.TextField(label="CRF", value="20", width=80, keyboard_type=ft.KeyboardType.NUMBER)
    frame_rate = ft.TextField(label="Frame Rate", value="30", width=110, keyboard_type=ft.KeyboardType.NUMBER)
    video_bit_depth = ft.Dropdown(
        label="Video Bit Depth", value="8", width=140,
        options=[ft.dropdown.Option("8", "8-bit"), ft.dropdown.Option("10", "10-bit")],
    )
    overlay_txt = ft.Checkbox(label="Filename overlay", value=False)

    video_options = ft.Column(
        controls=[
            ft.Text("Video Options", weight=ft.FontWeight.BOLD, size=13),
            ft.Row(controls=[resolution, codec, crf, frame_rate, video_bit_depth, overlay_txt], wrap=True),
        ],
        visible=False,
    )

    # Progress
    progress_bar = ft.ProgressBar(value=0, visible=False)
    progress_text = ft.Text("", size=12)
    log_text = ft.TextField(multiline=True, read_only=True, text_size=11, expand=True)

    _btn_style = ft.ButtonStyle(bgcolor=ft.Colors.with_opacity(0.15, ft.Colors.WHITE))
    run_button = ft.Button(content="Run", icon=ft.Icons.PLAY_ARROW, style=_btn_style)
    cancel_button = ft.Button(content="Cancel", icon=ft.Icons.STOP, visible=False, style=_btn_style)

    # --- Async event handlers ---
    async def pick_input(e):
        mode = input_mode.value
        initial = state["last_input_dir"]
        if mode == "folder":
            result = await ft.FilePicker().get_directory_path(
                dialog_title="Select DNG input folder",
                initial_directory=initial,
            )
            if result:
                state["last_input_dir"] = result
                state["input_files"] = None
                input_path_text.value = result
                input_path_text.tooltip = result
                _save_settings({"last_input_dir": state["last_input_dir"], "last_output_dir": state["last_output_dir"]})
                page.update()
        else:
            files = await ft.FilePicker().pick_files(
                dialog_title="Select DNG file(s)",
                initial_directory=initial,
                allowed_extensions=["dng", "DNG"],
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
                    input_path_text.value = f"{len(paths)} files in {Path(paths[0]).parent}"
                    input_path_text.tooltip = str(Path(paths[0]).parent)
                _save_settings({"last_input_dir": state["last_input_dir"], "last_output_dir": state["last_output_dir"]})
                page.update()

    async def pick_output(e):
        initial = state["last_output_dir"]
        if mode_dropdown.value == "video":
            result = await ft.FilePicker().save_file(
                dialog_title="Save video as",
                file_name="output.mp4",
                initial_directory=initial,
                allowed_extensions=["mp4"],
                file_type=ft.FilePickerFileType.CUSTOM,
            )
        else:
            result = await ft.FilePicker().get_directory_path(
                dialog_title="Select output folder",
                initial_directory=initial,
            )
        if result:
            if mode_dropdown.value == "video":
                if not result.lower().endswith(".mp4"):
                    result += ".mp4"
                state["last_output_dir"] = str(Path(result).parent)
            else:
                state["last_output_dir"] = result
            output_path_text.value = result
            output_path_text.tooltip = result
            _save_settings({"last_input_dir": state["last_input_dir"], "last_output_dir": state["last_output_dir"]})
            page.update()

    def on_mode_changed(e):
        mode = mode_dropdown.value
        is_video = mode == "video"
        is_jpg = mode == "jpg"
        video_options.visible = is_video
        bit_depth.visible = not is_video
        _output_btn_text.value = "Select Output File" if is_video else "Select Output Folder"
        output_path_text.value = "No folder selected"
        output_path_text.tooltip = None
        if is_jpg:
            bit_depth.value = "8"
            bit_depth.disabled = True
        else:
            bit_depth.disabled = False
        page.update()

    def on_xmp_changed(e):
        xmp_on = use_xmp.value
        wb_dropdown.disabled = xmp_on
        exposure.disabled = xmp_on
        if xmp_on:
            temperature.disabled = True
            tint.disabled = True
        else:
            on_wb_changed(None)
        page.update()

    def on_wb_changed(e):
        preset = wb_dropdown.value
        if preset == "custom":
            temperature.disabled = False
            tint.disabled = False
        elif preset == "as_shot":
            temperature.value = ""
            tint.value = ""
            temperature.disabled = True
            tint.disabled = True
        else:
            t, ti = WB_PRESETS[preset]
            temperature.value = str(t) if t is not None else ""
            tint.value = str(ti) if ti is not None else ""
            temperature.disabled = True
            tint.disabled = True
        page.update()

    mode_dropdown.on_select = on_mode_changed
    wb_dropdown.on_select = on_wb_changed
    use_xmp.on_change = on_xmp_changed

    def on_cancel(e):
        state["cancel"] = True
        state["log"] = (state.get("log") or "") + "Cancellation requested...\n"
        page.update()

    def on_run(e):
        inp = input_path_text.value
        out = output_path_text.value
        if not inp or inp == "No folder selected" or not out or out == "No folder selected":
            log_text.value = "ERROR: Select input and output folders first.\n" + (log_text.value or "")
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
        # Save previous log and start fresh
        state["_old_log"] = log_text.value or ""
        page.update()

        thread = threading.Thread(target=run_conversion, args=(inp, out), daemon=True)
        thread.start()
        page.run_task(_poll_ui)

    run_button.on_click = on_run
    cancel_button.on_click = on_cancel

    # --- Conversion logic (runs in thread) ---
    def run_conversion(input_path, output_path):
        import setproctitle
        setproctitle.setproctitle("mu-dng-converter: DNG → Image")

        try:
            mode = mode_dropdown.value
            # Use explicit file list if available (file picker mode)
            input_files = state.get("input_files")
            if input_files:
                dng_files = sorted(Path(f) for f in input_files)
            else:
                input_dir = Path(input_path)
                dng_files = sorted(input_dir.glob("*.dng"))
            if not dng_files:
                log(f"No .dng files found in {input_path}")
                finish()
                return

            total = len(dng_files)
            if mode == "video":
                out_desc = Path(output_path).name
            else:
                out_desc = f".{mode}"
            log(f"Input: {total} DNG files, Output: {out_desc}")

            rendering_params = build_rendering_params()

            def on_task_done(completed, total):
                update_progress(completed / total, f"{completed}/{total}")
                return state["cancel"]

            if mode == "video":
                from muimg.cli import run_batch_to_video
                w, h = resolution.value.split("x")
                result = run_batch_to_video(
                    dng_files=dng_files,
                    output_mp4=Path(output_path),
                    rendering_params=rendering_params,
                    use_xmp=use_xmp.value,
                    resolution=(int(w), int(h)),
                    codec=codec.value,
                    crf=int(crf.value),
                    bit_depth=video_bit_depth.value,
                    frame_rate=float(frame_rate.value),
                    num_workers=int(num_workers.value),
                    overlay_txt=overlay_txt.value,
                    on_task_done=on_task_done,
                )
            else:
                from muimg.cli import run_batch_convert
                result = run_batch_convert(
                    dng_files=dng_files,
                    output_folder=output_path,
                    output_format=mode,
                    bit_depth=bit_depth.value,
                    rendering_params=rendering_params,
                    use_xmp=use_xmp.value,
                    num_workers=int(num_workers.value),
                    on_task_done=on_task_done,
                )

            fps = result['completed'] / result['elapsed'] if result['elapsed'] > 0 else 0
            log(f"Done: {result['completed']}/{result['total']} files in {result['elapsed']:.1f}s ({fps:.1f} files/s, {int(num_workers.value)} workers)")
            stats = result.get('queue_stats', {})
            if 'task_queue' in stats:
                q = stats['task_queue']
                log(f"  Task queue: avg_depth={q['avg_depth']:.1f}, empty={q['empty_time']:.1f}s")
            if 'writer_queue' in stats:
                q = stats['writer_queue']
                log(f"  Writer queue: avg_depth={q['avg_depth']:.1f}, empty={q['empty_time']:.1f}s")
        except Exception as ex:
            log(f"ERROR: {ex}")
        finally:
            finish()

    # --- Helpers ---
    def build_rendering_params():
        if use_xmp.value:
            return {}
        params = {}
        if temperature.value:
            params["Temperature"] = float(temperature.value)
        if tint.value:
            params["Tint"] = float(tint.value)
        params["Exposure2012"] = float(exposure.value or 0)
        return params

    def update_progress(fraction, text):
        state["progress_fraction"] = fraction
        state["progress_text"] = text

    _MAX_LOG_LINES = 100

    def _build_display_log(current_log, old_log):
        """Newest run on top, chronological within each run."""
        parts = [p for p in [current_log.rstrip(), old_log.rstrip()] if p]
        combined = ("\n" + "─" * 40 + "\n").join(parts)
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
        """Async poller: runs on Flet event loop, reads shared state."""
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
                log_text.value = _build_display_log(log_val, state.get("_old_log", ""))
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
            log_text.value = _build_display_log(remaining, state.get("_old_log", ""))
        run_button.visible = True
        cancel_button.visible = False
        page.update()

    # --- Build layout ---
    input_btn = ft.Button(
        content=ft.Row([ft.Icon(ft.Icons.FOLDER_OPEN), ft.Text("Select Input")], alignment=ft.MainAxisAlignment.START, spacing=8),
        on_click=pick_input, style=_btn_style, width=220,
    )
    _output_btn_text = ft.Text("Select Output Folder")
    output_btn = ft.Button(
        content=ft.Row([ft.Icon(ft.Icons.FOLDER_OPEN), _output_btn_text], alignment=ft.MainAxisAlignment.START, spacing=8),
        on_click=pick_output, style=_btn_style, width=220,
    )

    return ft.Column(
        controls=[
            ft.Container(
                content=ft.Row(
                    controls=[mode_dropdown, run_button, cancel_button, progress_text],
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                ),
                padding=ft.Padding(left=0, top=8, right=0, bottom=0),
            ),
            ft.Divider(height=8),
            ft.Row(controls=[input_btn, input_mode, input_path_text]),
            ft.Row(controls=[output_btn, output_path_text]),
            ft.Divider(height=8),
            ft.Text("Rendering Parameters", weight=ft.FontWeight.BOLD, size=13),
            ft.Row(controls=[use_xmp], wrap=True),
            ft.Row(controls=[wb_dropdown, temperature, tint, exposure], wrap=True),
            ft.Divider(height=8),
            ft.Text("Output Options", weight=ft.FontWeight.BOLD, size=13),
            ft.Row(controls=[bit_depth, num_workers], wrap=True),
            video_options,
            ft.Divider(height=8),
            progress_bar,
            ft.Container(content=log_text, expand=True),
        ],
        expand=True,
    )
