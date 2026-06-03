"""DNG → DNG copy/transcode view."""

import json
import threading
from pathlib import Path

import flet as ft


# Curated list of simple string tags supported by add_tag()
_SETTABLE_TAGS = [
    "Artist",
    "Copyright",
]


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
    return d / "dng_copy_settings.json"


def _load_settings() -> dict:
    try:
        return json.loads(_settings_path().read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _save_settings(settings: dict):
    _settings_path().write_text(json.dumps(settings, indent=2))


def build_dng_dng_view(page: ft.Page, dir_picker: ft.FilePicker | None = None,
                       file_picker: ft.FilePicker | None = None) -> ft.Control:
    """Build the DNG → DNG copy/transcode tab content."""

    _settings = _load_settings()
    state = {
        "running": False,
        "cancel": False,
        "last_input_dir": _settings.get("last_input_dir"),
        "last_output_dir": _settings.get("last_output_dir"),
        "extra_tags": _settings.get("extra_tags", []),  # list of {"name": ..., "value": ...}
    }

    _btn_style = ft.ButtonStyle(bgcolor=ft.Colors.with_opacity(0.15, ft.Colors.WHITE))

    # --- Input / Output ---
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

    # --- Transcode Options ---
    transcode_checkbox = ft.Checkbox(label="Transcode (re-encode)", value=False)

    def _clamp_float(field, lo, hi, default):
        def handler(e):
            try:
                v = max(lo, min(hi, float(field.value)))
            except (ValueError, TypeError):
                v = default
            field.value = str(v)
            page.update()
        return handler

    compression_dropdown = ft.Dropdown(
        label="Compression",
        value="uncompressed",
        options=[
            ft.dropdown.Option("uncompressed", "Uncompressed"),
            ft.dropdown.Option("jxl_lossless", "JXL Lossless"),
            ft.dropdown.Option("jxl_lossy", "JXL Lossy"),
        ],
        width=190,
        disabled=True,
    )

    jxl_distance = ft.TextField(
        label="JXL Distance", value="1.0", width=130,
        keyboard_type=ft.KeyboardType.NUMBER,
        disabled=True,
    )
    jxl_distance.on_blur = _clamp_float(jxl_distance, 0.0, 25.0, 1.0)
    jxl_distance.on_submit = jxl_distance.on_blur

    jxl_effort = ft.TextField(
        label="JXL Effort (1-9)", value="5", width=140,
        keyboard_type=ft.KeyboardType.NUMBER,
        disabled=True,
    )

    scale = ft.TextField(
        label="Scale (0.125–1)", value="1.0", width=130,
        keyboard_type=ft.KeyboardType.NUMBER,
        disabled=True,
    )
    scale.on_blur = _clamp_float(scale, 0.125, 1.0, 1.0)
    scale.on_submit = scale.on_blur

    demosaic = ft.Checkbox(
        label="Demosaic (CFA → Linear RAW)", value=False, disabled=True,
    )

    demosaic_algorithm = ft.Dropdown(
        label="Demosaic Algorithm",
        value="DNGSDK_BILINEAR",
        options=[
            ft.dropdown.Option("DNGSDK_BILINEAR", "DNG SDK Bilinear"),
            ft.dropdown.Option("OPENCV_EA", "OpenCV Edge-Aware"),
            ft.dropdown.Option("VNG", "VNG"),
        ],
        width=210,
        disabled=True,
    )

    _transcode_controls = [
        compression_dropdown, jxl_distance, jxl_effort, scale, demosaic, demosaic_algorithm,
    ]

    def on_transcode_changed(e):
        enabled = transcode_checkbox.value
        for ctrl in _transcode_controls:
            ctrl.disabled = not enabled
        page.update()

    transcode_checkbox.on_change = on_transcode_changed

    transcode_row = ft.Column(
        controls=[
            ft.Row(
                controls=[compression_dropdown, jxl_distance, jxl_effort, scale],
                spacing=8,
            ),
            ft.Row(
                controls=[demosaic, demosaic_algorithm],
                spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
        ],
        spacing=12,
    )

    # --- Output Options ---
    preview = ft.Checkbox(label="JPEG Preview", value=False)
    fast_load = ft.Checkbox(label="Embed Fast Load Data", value=False)

    def _clamp_workers(e):
        try:
            v = max(1, min(8, int(num_workers.value)))
        except (ValueError, TypeError):
            v = 4
        num_workers.value = str(v)
        page.update()

    num_workers = ft.TextField(
        label="Workers (1-8)", value="4", width=130,
        keyboard_type=ft.KeyboardType.NUMBER,
        on_blur=_clamp_workers,
        on_submit=_clamp_workers,
    )

    # --- Strip Tags ---
    strip_tags_field = ft.TextField(
        label="Strip Tags (comma-separated)",
        value=_settings.get("strip_tags", ""),
        expand=True,
    )

    # --- Set Tags (picker + list) ---
    tag_name_dropdown = ft.Dropdown(
        label="Tag",
        value=_SETTABLE_TAGS[0],
        options=[ft.dropdown.Option(t) for t in _SETTABLE_TAGS],
        width=200,
    )
    tag_value_field = ft.TextField(
        label="Value", width=280,
    )

    tag_list_column = ft.Column(spacing=2, tight=True)

    def _rebuild_tag_list():
        tag_list_column.controls.clear()
        for i, entry in enumerate(state["extra_tags"]):
            idx = i  # capture

            def make_remove(captured_idx):
                def on_remove(e):
                    state["extra_tags"].pop(captured_idx)
                    _rebuild_tag_list()
                    _persist_settings()
                    page.update()
                return on_remove

            tag_list_column.controls.append(
                ft.Row(
                    controls=[
                        ft.Text(
                            f"{entry['name']}: {entry['value']}",
                            size=12,
                            expand=True,
                            overflow=ft.TextOverflow.ELLIPSIS,
                        ),
                        ft.IconButton(
                            icon=ft.Icons.CLOSE,
                            icon_size=14,
                            on_click=make_remove(idx),
                            tooltip="Remove",
                        ),
                    ],
                    spacing=4,
                    vertical_alignment=ft.CrossAxisAlignment.CENTER,
                )
            )

    _rebuild_tag_list()

    def on_add_tag(e):
        name = tag_name_dropdown.value
        value = tag_value_field.value.strip()
        if not value:
            return
        # Replace existing entry for the same tag name
        state["extra_tags"] = [t for t in state["extra_tags"] if t["name"] != name]
        state["extra_tags"].append({"name": name, "value": value})
        tag_value_field.value = ""
        _rebuild_tag_list()
        _persist_settings()
        page.update()

    add_tag_btn = ft.Button(
        content="Add", icon=ft.Icons.ADD, style=_btn_style,
        on_click=on_add_tag,
    )

    # --- Adjust Time ---
    time_offset_enabled = ft.Checkbox(label="Adjust DateTimeOriginal", value=False)
    time_sign = ft.Dropdown(
        value="+",
        options=[ft.dropdown.Option("+"), ft.dropdown.Option("−")],
        width=80,
        disabled=True,
    )
    time_hours = ft.TextField(
        label="H", value="0", width=80,
        keyboard_type=ft.KeyboardType.NUMBER,
        disabled=True,
    )
    time_minutes = ft.TextField(
        label="M", value="0", width=80,
        keyboard_type=ft.KeyboardType.NUMBER,
        disabled=True,
    )
    time_seconds = ft.TextField(
        label="S", value="0", width=80,
        keyboard_type=ft.KeyboardType.NUMBER,
        disabled=True,
    )
    time_hint = ft.Text(
        "e.g. to fix timezone offset or clock drift",
        size=11, italic=True, visible=False,
    )

    def on_time_offset_changed(e):
        enabled = time_offset_enabled.value
        for ctrl in [time_sign, time_hours, time_minutes, time_seconds]:
            ctrl.disabled = not enabled
        time_hint.visible = enabled
        page.update()

    time_offset_enabled.on_change = on_time_offset_changed

    # --- Progress ---
    progress_bar = ft.ProgressBar(value=0, visible=False)
    progress_text = ft.Text("", size=12)
    log_text = ft.TextField(
        multiline=True, read_only=True, text_size=11,
        min_lines=3, max_lines=10, expand=True,
    )

    run_button = ft.Button(
        content="Run", icon=ft.Icons.PLAY_ARROW, style=_btn_style,
    )
    cancel_button = ft.Button(
        content="Cancel", icon=ft.Icons.STOP,
        visible=False, style=_btn_style,
    )

    # --- Settings persistence ---
    def _persist_settings():
        _save_settings({
            "last_input_dir": state.get("last_input_dir"),
            "last_output_dir": state.get("last_output_dir"),
            "strip_tags": strip_tags_field.value,
            "extra_tags": state["extra_tags"],
            "transcode": transcode_checkbox.value,
            "compression": compression_dropdown.value,
            "jxl_distance": jxl_distance.value,
            "jxl_effort": jxl_effort.value,
            "scale": scale.value,
            "demosaic": demosaic.value,
            "demosaic_algorithm": demosaic_algorithm.value,
            "preview": preview.value,
            "fast_load": fast_load.value,
            "num_workers": num_workers.value,
        })

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
                _log(f"[error] folder picker failed: {ex}")
                page.update()
                return
            if result:
                state["last_input_dir"] = result
                state["input_files"] = None
                input_path_text.value = result
                input_path_text.tooltip = result
                _persist_settings()
                page.update()
        else:
            try:
                paths = await pick_files_async(
                    "Select DNG file(s)", initial, ["dng", "DNG"],
                    allow_multiple=True, picker=file_picker)
            except Exception as ex:
                _log(f"[error] file picker failed: {ex}")
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
                _persist_settings()
                page.update()

    async def pick_output(e):
        from mu_dng_converter.dialogs import pick_directory_async

        initial = state["last_output_dir"]
        result = await pick_directory_async("Select output folder", initial, picker=dir_picker)
        if result:
            state["last_output_dir"] = result
            output_path_text.value = result
            output_path_text.tooltip = result
            _persist_settings()
            page.update()

    def on_cancel(e):
        state["cancel"] = True
        state["log"] = (state.get("log") or "") + "Cancellation requested...\n"
        page.update()

    async def on_run(e):
        from mu_dng_converter.dialogs import check_overwrite

        inp = input_path_text.value
        out = output_path_text.value
        if (
            not inp or inp == "No folder selected"
            or not out or out == "No folder selected"
        ):
            log_text.value = (
                "ERROR: Select input and output folders first.\n"
                + (log_text.value or "")
            )
            page.update()
            return

        input_files = state.get("input_files")
        if input_files:
            dng_files = sorted(Path(f) for f in input_files)
        else:
            input_dir = Path(inp)
            dng_files = sorted(input_dir.glob("*.dng"))
        if not dng_files:
            log_text.value = (
                f"ERROR: No .dng files found in {inp}\n"
                + (log_text.value or "")
            )
            page.update()
            return

        output_dir = Path(out)
        existing = [f for f in dng_files if (output_dir / f.name).exists()]
        if existing:
            action = await check_overwrite(page, len(existing), len(dng_files))
            if action == "cancel":
                return
            elif action == "skip":
                dng_files = [f for f in dng_files if f not in existing]
                if not dng_files:
                    log_text.value = (
                        "All files already exist — nothing to do.\n"
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
        state["_dng_files"] = dng_files
        _persist_settings()
        page.update()

        thread = threading.Thread(target=run_copy, args=(out,), daemon=True)
        thread.start()
        page.run_task(_poll_ui)

    run_button.on_click = on_run
    cancel_button.on_click = on_cancel

    # --- Copy/transcode logic (runs in thread) ---
    def run_copy(output_path):
        import setproctitle
        setproctitle.setproctitle("mu-dng-converter: DNG → DNG")

        try:
            from muimg.cli import run_batch_copy_dng
            from muimg.tiff_metadata import MetadataTags

            dng_files = state["_dng_files"]
            _log(f"Input: {len(dng_files)} DNG files → {output_path}")

            # Build strip tags set from comma-separated field
            strip_tags_set = None
            raw_strip = strip_tags_field.value.strip()
            if raw_strip:
                strip_tags_set = {t.strip() for t in raw_strip.split(",") if t.strip()}

            # Build extra tags MetadataTags from picker list
            extra_tags_obj = None
            if state["extra_tags"]:
                extra_tags_obj = MetadataTags()
                for entry in state["extra_tags"]:
                    extra_tags_obj.add_tag(entry["name"], entry["value"])

            # Build time offset in seconds
            time_offset = 0.0
            if time_offset_enabled.value:
                try:
                    h = int(time_hours.value or 0)
                    m = int(time_minutes.value or 0)
                    s = int(time_seconds.value or 0)
                    offset = h * 3600 + m * 60 + s
                    if time_sign.value == "−":
                        offset = -offset
                    time_offset = float(offset)
                except (ValueError, TypeError):
                    pass

            def on_task_done(completed, total):
                update_progress(completed / total, f"{completed}/{total}")
                return state["cancel"]

            result = run_batch_copy_dng(
                dng_files=dng_files,
                output_folder=output_path,
                mode="transcode" if transcode_checkbox.value else "copy",
                compression_name=compression_dropdown.value,
                jxl_distance=float(jxl_distance.value) if jxl_distance.value else None,
                jxl_effort=int(jxl_effort.value) if jxl_effort.value.strip() else None,
                scale=max(0.125, min(1.0, float(scale.value or 1.0))),
                demosaic=demosaic.value,
                demosaic_algorithm=None,
                do_preview=preview.value,
                do_fast_load=fast_load.value,
                strip_tags=strip_tags_set,
                extra_tags=extra_tags_obj,
                time_offset_seconds=time_offset,
                num_workers=max(1, min(8, int(num_workers.value))),
                on_task_done=on_task_done,
            )

            fps = result["completed"] / result["elapsed"] if result["elapsed"] > 0 else 0
            _log(
                f"Done: {result['completed']}/{result['total']} files in "
                f"{result['elapsed']:.1f}s ({fps:.1f} files/s, {num_workers.value} workers)"
            )
            stats = result.get("queue_stats", {})
            if "task_queue" in stats:
                q = stats["task_queue"]
                _log(f"  Task queue: avg_depth={q['avg_depth']:.1f}, empty={q['empty_time']:.1f}s")
            if "writer_queue" in stats:
                q = stats["writer_queue"]
                _log(f"  Writer queue: avg_depth={q['avg_depth']:.1f}, empty={q['empty_time']:.1f}s")
        except Exception as ex:
            _log(f"ERROR: {ex}")
        finally:
            _finish()

    # --- Helpers ---
    def update_progress(fraction, text):
        state["progress_fraction"] = fraction
        state["progress_text"] = text

    _MAX_LOG_LINES = 100

    def _build_display_log(current_log, old_log):
        parts = [p for p in [current_log.rstrip(), old_log.rstrip()] if p]
        combined = ("\n" + "─" * 40 + "\n").join(parts)
        lines = combined.split("\n")
        if len(lines) > _MAX_LOG_LINES:
            lines = lines[:_MAX_LOG_LINES]
        return "\n".join(lines)

    def _log(message):
        state["log"] = (state.get("log") or "") + message + "\n"

    def _finish():
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
            [ft.Icon(ft.Icons.FOLDER_OPEN), ft.Text("Select Output Folder")],
            alignment=ft.MainAxisAlignment.START, spacing=8,
        ),
        on_click=pick_output, style=_btn_style, width=220,
    )

    return ft.Column(
        controls=[
            ft.Container(
                content=ft.Row(
                    controls=[run_button, cancel_button, progress_text],
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
            ft.Text("Transcode Options", weight=ft.FontWeight.BOLD, size=13),
            transcode_checkbox,
            transcode_row,
            ft.Divider(height=8),
            ft.Text("Output Options", weight=ft.FontWeight.BOLD, size=13),
            ft.Row(
                controls=[preview, fast_load, num_workers],
                spacing=16,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            ft.Divider(height=8),
            ft.Text("Strip Tags", weight=ft.FontWeight.BOLD, size=13),
            strip_tags_field,
            ft.Divider(height=8),
            ft.Text("Set Tags", weight=ft.FontWeight.BOLD, size=13),
            ft.Row(
                controls=[tag_name_dropdown, tag_value_field, add_tag_btn],
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
                spacing=8,
            ),
            tag_list_column,
            ft.Divider(height=8),
            ft.Text("Adjust Time", weight=ft.FontWeight.BOLD, size=13),
            ft.Row(
                controls=[
                    time_offset_enabled,
                    time_sign,
                    time_hours,
                    time_minutes,
                    time_seconds,
                ],
                wrap=True,
            ),
            time_hint,
            ft.Divider(height=8),
            progress_bar,
            log_text,
        ],
        expand=True,
        scroll=ft.ScrollMode.ALWAYS,
    )
