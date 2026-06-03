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

    # --- Update Tags (unified metadata panel) ---
    # Operations: add, strip, adjust-time, adjust-time-offset
    metadata_ops_list = ft.Column(spacing=4, tight=True)  # Applied operations list

    # Operation selector
    metadata_op_dropdown = ft.Dropdown(
        label="Operation",
        value="add",
        options=[
            ft.dropdown.Option("add", "Add/Set Tag"),
            ft.dropdown.Option("strip", "Strip Tag"),
            ft.dropdown.Option("adjust-time", "Set DateTime (absolute)"),
            ft.dropdown.Option("adjust-time-offset", "Shift Time + Timezone"),
        ],
        width=220,
    )

    # Tag operation inputs (for add/strip)
    tag_op_name = ft.TextField(label="Tag Name", width=180)
    tag_op_value = ft.TextField(label="Value", width=200, visible=True)

    # Time operation inputs (for adjust-time)
    time_absolute = ft.TextField(
        label="YYYY:MM:DD HH:MM:SS",
        value="",
        width=220,
        hint_text="2024:01:15 14:30:00",
        visible=False,
    )

    # Time offset inputs (for adjust-time-offset)
    time_offset_sign = ft.Dropdown(
        value="+",
        options=[ft.dropdown.Option("+"), ft.dropdown.Option("−")],
        width=60,
        visible=False,
    )
    time_offset_hours = ft.TextField(label="H", value="0", width=70, keyboard_type=ft.KeyboardType.NUMBER, visible=False)
    time_offset_minutes = ft.TextField(label="M", value="0", width=70, keyboard_type=ft.KeyboardType.NUMBER, visible=False)
    time_offset_seconds = ft.TextField(label="S", value="0", width=70, keyboard_type=ft.KeyboardType.NUMBER, visible=False)
    time_offset_tz = ft.TextField(
        label="Timezone",
        value="",
        width=100,
        hint_text="+02:00",
        visible=False,
    )

    def on_metadata_op_changed(e):
        op = metadata_op_dropdown.value
        is_tag_op = op in ("add", "strip")
        is_time_absolute = op == "adjust-time"
        is_time_offset = op == "adjust-time-offset"

        # Show/hide tag inputs
        tag_op_name.visible = is_tag_op
        tag_op_value.visible = (op == "add")

        # Show/hide time inputs
        time_absolute.visible = is_time_absolute
        time_offset_sign.visible = is_time_offset
        time_offset_hours.visible = is_time_offset
        time_offset_minutes.visible = is_time_offset
        time_offset_seconds.visible = is_time_offset
        time_offset_tz.visible = is_time_offset

        page.update()

    metadata_op_dropdown.on_change = on_metadata_op_changed

    def rebuild_metadata_ops_list():
        """Rebuild the list of applied metadata operations."""
        metadata_ops_list.controls.clear()
        ops = state.get("metadata_ops", [])

        for i, op in enumerate(ops):
            idx = i

            def make_remove(captured_idx):
                def on_remove(e):
                    state["metadata_ops"].pop(captured_idx)
                    rebuild_metadata_ops_list()
                    _persist_settings()
                    page.update()
                return on_remove

            # Format display text based on operation type
            if op["type"] == "add":
                display = f"Set {op['name']} = {op['value']}"
            elif op["type"] == "strip":
                display = f"Strip {op['name']}"
            elif op["type"] == "adjust-time":
                display = f"Set DateTimeOriginal = {op['value']}"
            elif op["type"] == "adjust-time-offset":
                sign = op.get("sign", "+")
                h = op.get("hours", 0)
                m = op.get("minutes", 0)
                s = op.get("seconds", 0)
                tz = op.get("timezone", "")
                display = f"Shift time {sign}{h}h{m}m{s}s"
                if tz:
                    display += f", TZ={tz}"
            else:
                display = str(op)

            metadata_ops_list.controls.append(
                ft.Row(
                    controls=[
                        ft.Text(display, size=12, expand=True, overflow=ft.TextOverflow.ELLIPSIS),
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

    def on_apply_metadata_op(e):
        """Apply the current metadata operation to the list."""
        op = metadata_op_dropdown.value
        ops = state.setdefault("metadata_ops", [])

        if op == "add":
            name = tag_op_name.value.strip()
            value = tag_op_value.value
            if not name:
                return
            # TODO: Validate against tag registry
            # Remove existing add op for same tag
            ops = [o for o in ops if not (o["type"] == "add" and o.get("name") == name)]
            ops.append({"type": "add", "name": name, "value": value})
            tag_op_value.value = ""

        elif op == "strip":
            name = tag_op_name.value.strip()
            if not name:
                return
            # Remove existing strip op for same tag
            ops = [o for o in ops if not (o["type"] == "strip" and o.get("name") == name)]
            ops.append({"type": "strip", "name": name})

        elif op == "adjust-time":
            value = time_absolute.value.strip()
            if not value:
                return
            # Remove existing adjust-time op
            ops = [o for o in ops if o["type"] != "adjust-time"]
            ops.append({"type": "adjust-time", "value": value})

        elif op == "adjust-time-offset":
            try:
                h = int(time_offset_hours.value or 0)
                m = int(time_offset_minutes.value or 0)
                s = int(time_offset_seconds.value or 0)
            except (ValueError, TypeError):
                h = m = s = 0
            tz = time_offset_tz.value.strip()
            sign = time_offset_sign.value
            # Remove existing adjust-time-offset op
            ops = [o for o in ops if o["type"] != "adjust-time-offset"]
            ops.append({
                "type": "adjust-time-offset",
                "sign": sign,
                "hours": h,
                "minutes": m,
                "seconds": s,
                "timezone": tz,
            })

        state["metadata_ops"] = ops
        rebuild_metadata_ops_list()
        _persist_settings()
        page.update()

    apply_metadata_btn = ft.Button(
        content="Apply", icon=ft.Icons.ADD, style=_btn_style,
        on_click=on_apply_metadata_op,
    )

    # Load any saved metadata ops
    state["metadata_ops"] = _settings.get("metadata_ops", [])
    rebuild_metadata_ops_list()

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
            "metadata_ops": state.get("metadata_ops", []),
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

            # Process metadata_ops into strip_tags, extra_tags, and time adjustments
            metadata_ops = state.get("metadata_ops", [])
            strip_tags_set = set()
            extra_tags_obj = MetadataTags()
            time_offset_seconds = 0.0
            time_timezone = None
            has_time_adjust_absolute = False

            for op in metadata_ops:
                op_type = op.get("type")
                if op_type == "strip":
                    strip_tags_set.add(op.get("name", "").strip())
                elif op_type == "add":
                    extra_tags_obj.add_tag(op.get("name", ""), op.get("value", ""))
                elif op_type == "adjust-time-offset":
                    sign = -1 if op.get("sign") == "−" else 1
                    h = op.get("hours", 0)
                    m = op.get("minutes", 0)
                    s = op.get("seconds", 0)
                    time_offset_seconds += sign * (h * 3600 + m * 60 + s)
                    tz = op.get("timezone", "").strip()
                    if tz:
                        time_timezone = tz
                elif op_type == "adjust-time":
                    # Absolute time set - mark for special handling
                    has_time_adjust_absolute = True
                    _log(f"[warn] Absolute time adjustment not yet implemented: {op.get('value')}")

            # Remove empty tag names from strip set
            strip_tags_set = {t for t in strip_tags_set if t} or None
            if not extra_tags_obj._tags:
                extra_tags_obj = None

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
                time_offset_seconds=time_offset_seconds,
                time_timezone=time_timezone,
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
            ft.Text("Update Tags", weight=ft.FontWeight.BOLD, size=13),
            # Operation selector row
            ft.Row(
                controls=[
                    metadata_op_dropdown,
                    tag_op_name,
                    tag_op_value,
                    time_absolute,
                    time_offset_sign,
                    time_offset_hours,
                    time_offset_minutes,
                    time_offset_seconds,
                    time_offset_tz,
                    apply_metadata_btn,
                ],
                wrap=True,
                spacing=8,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            ),
            # Applied operations list
            metadata_ops_list,
            ft.Divider(height=8),
            progress_bar,
            log_text,
        ],
        expand=True,
        scroll=ft.ScrollMode.ALWAYS,
    )
