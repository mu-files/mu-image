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


def build_dng_view(page: ft.Page, dir_picker: ft.FilePicker | None = None,
                   file_picker: ft.FilePicker | None = None,
                   save_picker: ft.FilePicker | None = None) -> ft.Control:
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
        value="tif",
        options=[
            ft.dropdown.Option("tif", "DNG → TIF (with metadata)"),
            ft.dropdown.Option("jpg", "DNG → JPG"),
            ft.dropdown.Option("video", "DNG → Video (MP4)"),
        ],
        width=320,
        content_padding=ft.Padding(left=8, top=4, right=8, bottom=4),
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

    def _clamp_scale(e):
        try:
            v = max(0.125, min(1.0, float(scale.value)))
        except (ValueError, TypeError):
            v = 1.0
        scale.value = str(v)
        page.update()

    scale = ft.TextField(
        label="Scale (0.125–1)", value="1.0", width=130,
        keyboard_type=ft.KeyboardType.NUMBER,
        on_blur=_clamp_scale,
        on_submit=_clamp_scale,
    )

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

    _btn_style = ft.ButtonStyle(
        bgcolor=ft.Colors.with_opacity(0.15, ft.Colors.WHITE),
        padding=ft.Padding(left=10, top=6, right=10, bottom=6),
    )
    run_button = ft.Button(content="Run", icon=ft.Icons.PLAY_ARROW, style=_btn_style)
    cancel_button = ft.Button(content="Cancel", icon=ft.Icons.STOP, visible=False, style=_btn_style)

    # --- Async event handlers ---
    async def pick_input(e):
        from mu_dng_converter.dialogs import pick_directory_async, pick_files_async

        mode = input_mode.value
        initial = state["last_input_dir"]
        if mode == "folder":
            try:
                result = await pick_directory_async(
                    "Select DNG input folder", initial,
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
                _save_settings({"last_input_dir": state["last_input_dir"], "last_output_dir": state["last_output_dir"]})
                page.update()
        else:
            try:
                paths = await pick_files_async(
                    "Select DNG file(s)", initial, ["dng", "DNG"],
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
                    input_path_text.value = f"{len(paths)} files in {Path(paths[0]).parent}"
                    input_path_text.tooltip = str(Path(paths[0]).parent)
                _save_settings({"last_input_dir": state["last_input_dir"], "last_output_dir": state["last_output_dir"]})
                page.update()

    async def pick_output(e):
        from mu_dng_converter.dialogs import pick_directory_async

        initial = state["last_output_dir"]
        if mode_dropdown.value == "video":
            _sp = save_picker if save_picker is not None else ft.FilePicker()
            result = await _sp.save_file(
                dialog_title="Save video as",
                file_name="output.mp4",
                initial_directory=initial,
                allowed_extensions=["mp4"],
                file_type=ft.FilePickerFileType.CUSTOM,
            )
        else:
            result = await pick_directory_async("Select output folder", initial, picker=dir_picker)
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
        scale.visible = not is_video
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

    # --- Helper to build rendering params from UI controls ---
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

    # --- Conversion logic (worker function for BatchRunner) ---
    def run_conversion_worker(output_path, worker_state):
        """Worker function that runs in background thread."""
        import setproctitle
        from mu_dng_converter.batch_runner import make_state_logger
        setproctitle.setproctitle("mu-dng-converter: DNG → Image")

        _log = make_state_logger(worker_state)

        try:
            mode = mode_dropdown.value
            dng_files = worker_state["_dng_files"]
            total = len(dng_files)
            if mode == "video":
                out_desc = Path(output_path).name
            else:
                out_desc = f".{mode}"
            _log(f"Input: {total} DNG files, Output: {out_desc}")

            rendering_params = build_rendering_params()

            def on_task_done(completed, total):
                worker_state["progress_fraction"] = completed / total if total > 0 else 0
                worker_state["progress_text"] = f"{completed}/{total}"
                return worker_state["cancel"]

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
                    num_workers=max(1, min(8, int(num_workers.value))),
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
                    scale=max(0.125, min(1.0, float(scale.value or 1.0))),
                    num_workers=max(1, min(8, int(num_workers.value))),
                    on_task_done=on_task_done,
                )

            fps = result['completed'] / result['elapsed'] if result['elapsed'] > 0 else 0
            nw = max(1, min(8, int(num_workers.value)))
            _log(f"Done: {result['completed']}/{result['total']} files in {result['elapsed']:.1f}s ({fps:.1f} files/s, {nw} workers)")
            stats = result.get('queue_stats', {})
            if 'task_queue' in stats:
                q = stats['task_queue']
                _log(f"  Task queue: avg_depth={q['avg_depth']:.1f}, empty={q['empty_time']:.1f}s")
            if 'writer_queue' in stats:
                q = stats['writer_queue']
                _log(f"  Writer queue: avg_depth={q['avg_depth']:.1f}, empty={q['empty_time']:.1f}s")
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
        """Custom on_run with video mode handling for existing files."""
        from mu_dng_converter.dialogs import check_overwrite

        inp = input_path_text.value
        out = output_path_text.value
        if not inp or inp == "No folder selected" or not out or out == "No folder selected":
            log_text.value = "ERROR: Select input and output folders first.\n" + (log_text.value or "")
            page.update()
            return

        # Gather file list before starting
        mode = mode_dropdown.value
        input_files = state.get("input_files")
        if input_files:
            dng_files = sorted(Path(f) for f in input_files)
        else:
            input_dir = Path(inp)
            dng_files = sorted(input_dir.glob("*.dng"))
        if not dng_files:
            log_text.value = f"ERROR: No .dng files found in {inp}\n" + (log_text.value or "")
            page.update()
            return

        # Check for existing output files (skip for video mode)
        if mode != "video":
            ext = mode  # tif, jxl, jpg
            output_dir = Path(out)
            existing = [
                f for f in dng_files
                if (output_dir / (f.stem + "." + ext)).exists()
            ]
            if existing:
                action = await check_overwrite(page, len(existing), len(dng_files))
                if action == "cancel":
                    return
                elif action == "skip":
                    dng_files = [f for f in dng_files if f not in existing]
                    if not dng_files:
                        log_text.value = "All files already exist — nothing to do.\n" + (log_text.value or "")
                        page.update()
                        return

        # Store the filtered file list for the worker
        state["_dng_files"] = dng_files
        state["input_files"] = [str(f) for f in dng_files]

        # Call common runner logic
        await runner.on_run(
            input_path_text=input_path_text,
            output_path_text=output_path_text,
            persist_settings_fn=lambda: None,  # Settings saved in custom logic above
        )

    run_button.on_click = on_run_wrapper
    cancel_button.on_click = runner.on_cancel
    
    # Set runner for webview access
    set_runner(runner)

    # --- Build layout ---
    input_btn = ft.Button(
        content=ft.Row([ft.Icon(ft.Icons.FOLDER_OPEN), ft.Text("Select Input")], alignment=ft.MainAxisAlignment.START, spacing=8),
        on_click=pick_input, style=_btn_style, width=180,
    )
    _output_btn_text = ft.Text("Select Output Folder")
    output_btn = ft.Button(
        content=ft.Row([ft.Icon(ft.Icons.FOLDER_OPEN), _output_btn_text], alignment=ft.MainAxisAlignment.START, spacing=8),
        on_click=pick_output, style=_btn_style, width=180,
    )

    return ft.Column(
        controls=[
            ft.Container(
                height=48,  # Fixed height for Run row
                content=ft.Row(
                    controls=[mode_dropdown, run_button, cancel_button, progress_text],
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
            ft.Text("Rendering Parameters", weight=ft.FontWeight.BOLD, size=13),
            ft.Row(controls=[use_xmp], wrap=True),
            ft.Row(controls=[wb_dropdown, temperature, tint, exposure], wrap=True),
            ft.Divider(height=1),
            ft.Text("Output Options", weight=ft.FontWeight.BOLD, size=13),
            ft.Row(controls=[bit_depth, scale, num_workers], wrap=True),
            video_options,
            ft.Divider(height=1),
            progress_bar,
            ft.Container(content=log_text, expand=True),
        ],
        expand=True,
        spacing=4,
    )

# Store runner reference for webview access
_runner = None

def get_runner():
    """Get the runner instance for webview bridge."""
    return _runner

def set_runner(runner_instance):
    """Set the runner instance for webview bridge."""
    global _runner
    _runner = runner_instance
