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
    metadata_ops_list = ft.Column(
        spacing=4,
        tight=True,
        expand=True,
    )

    # Operation selector
    metadata_op_dropdown = ft.Dropdown(
        label="Operation",
        value="set",
        options=[
            ft.dropdown.Option("set", "Set Tag"),
            ft.dropdown.Option("strip", "Strip Tag"),
            ft.dropdown.Option("shift-time", "Shift Time"),
            ft.dropdown.Option("shift-timezone", "Shift Timezone"),
        ],
        width=160,
    )

    # Tag/Time operation inputs
    # For set/strip: user enters name/value
    # For shift-time: name="AllDates", value=offset like "+1:00:00 00:00:00"
    # For shift-timezone: name="OffsetTimeOriginal", value=timezone like "+02:00"
    tag_op_name = ft.TextField(label="Tag Name", width=180, read_only=False)
    tag_op_value = ft.TextField(label="Value", width=220, visible=True)

    # Time shift inputs (for shift-time)
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
        is_set = op == "set"
        is_strip = op == "strip"
        is_shift_time = op == "shift-time"
        is_shift_tz = op == "shift-timezone"

        if is_shift_time:
            tag_op_name.value = "AllDates"
            tag_op_name.read_only = True
            tag_op_name.label = "Tag (AllDates)"
            tag_op_value.label = "Offset (+/-[Y:M:D] [H:M:S])"
            tag_op_value.hint_text = "5, 2:30, 0:15:30, or 0:0:1 12:00:00"
            tag_op_name.visible = True
            tag_op_value.visible = True
            # Hide old H/M/S fields - using unified value field instead
            time_offset_sign.visible = False
            time_offset_hours.visible = False
            time_offset_minutes.visible = False
            time_offset_seconds.visible = False
            time_offset_tz.visible = False
        elif is_shift_tz:
            tag_op_name.value = "OffsetTimeOriginal"
            tag_op_name.read_only = True
            tag_op_name.label = "Tag (OffsetTimeOriginal)"
            tag_op_value.label = "Timezone (+/-HH:MM)"
            tag_op_value.hint_text = "+02:00"
            tag_op_name.visible = True
            tag_op_value.visible = True
            time_offset_sign.visible = False
            time_offset_hours.visible = False
            time_offset_minutes.visible = False
            time_offset_seconds.visible = False
            time_offset_tz.visible = False
        else:
            # set or strip
            tag_op_name.read_only = False
            tag_op_name.label = "Tag Name"
            tag_op_value.label = "Value"
            tag_op_value.hint_text = ""
            tag_op_name.visible = is_set or is_strip
            tag_op_value.visible = is_set
            time_offset_sign.visible = False
            time_offset_hours.visible = False
            time_offset_minutes.visible = False
            time_offset_seconds.visible = False
            time_offset_tz.visible = False

        page.update()

    metadata_op_dropdown.on_select = on_metadata_op_changed

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
                    # Don't persist - metadata_ops are session-only
                    page.update()
                return on_remove

            # Format display text based on operation type - table columns
            if op["type"] == "set":
                cmd = "Set"
                tag = op['name']
                val = op['value']
            elif op["type"] == "strip":
                cmd = "Strip"
                tag = op['name']
                val = ""
            elif op["type"] == "shift-time":
                cmd = "Shift"
                tag = "AllDates"
                val = op.get("offset", "")
            elif op["type"] == "shift-timezone":
                cmd = "Set"
                tag = "OffsetTimeOriginal"
                val = op.get("timezone", "")
            else:
                cmd = str(op)
                tag = ""
                val = ""

            # Build row with fixed-width columns and dividers
            def col_divider():
                return ft.Container(width=1, height=20, bgcolor=ft.Colors.with_opacity(0.2, ft.Colors.ON_SURFACE))

            row_content = ft.Row(
                controls=[
                    ft.IconButton(
                        icon=ft.Icons.CLOSE,
                        icon_size=14,
                        on_click=make_remove(idx),
                        tooltip="Remove",
                        icon_color=ft.Colors.ON_SURFACE_VARIANT,
                        width=32,
                    ),
                    col_divider(),
                    ft.Container(
                        content=ft.Text(cmd, size=13, weight="w500"),
                        width=60,
                    ),
                    col_divider(),
                    ft.Container(
                        content=ft.Text(tag, size=13, overflow=ft.TextOverflow.ELLIPSIS),
                        width=160,
                    ),
                    col_divider(),
                    ft.Container(
                        content=ft.Text(val, size=13, overflow=ft.TextOverflow.ELLIPSIS),
                        expand=True,
                    ),
                ],
                spacing=4,
                vertical_alignment=ft.CrossAxisAlignment.CENTER,
            )

            # Wrap in container with subtle background
            metadata_ops_list.controls.append(
                ft.Container(
                    content=row_content,
                    bgcolor=ft.Colors.with_opacity(0.1, ft.Colors.ON_SURFACE),
                    border_radius=4,
                    padding=8,
                    expand=True,
                )
            )

    def on_apply_metadata_op(e):
        """Apply the current metadata operation to the list."""
        op = metadata_op_dropdown.value
        ops = state.setdefault("metadata_ops", [])

        if op == "set":
            name = tag_op_name.value.strip()
            value = tag_op_value.value
            if not name:
                return
            # Remove existing set op for same tag
            ops = [o for o in ops if not (o["type"] == "set" and o.get("name") == name)]
            ops.append({"type": "set", "name": name, "value": value})
            tag_op_value.value = ""

        elif op == "strip":
            name = tag_op_name.value.strip()
            if not name:
                return
            # Remove existing strip op for same tag
            ops = [o for o in ops if not (o["type"] == "strip" and o.get("name") == name)]
            ops.append({"type": "strip", "name": name})

        elif op == "shift-time":
            offset_str = tag_op_value.value.strip()
            if not offset_str:
                return
            # Validate format: +/- followed by H, H:M, H:M:S, or Y:M:D H:M:S
            import re
            # Pattern: sign, then numbers separated by colons, optional space for date/time separator
            pattern = r'^[+-](\d+(:\d+)*)$|^[+-](\d+(:\d+)* \d+(:\d+)*)$'
            if not re.match(pattern, offset_str):
                # Show error dialog
                from mu_dng_converter.dialogs import show_error
                async def _show_error():
                    await show_error(
                        page,
                        "Invalid Time Offset Format",
                        f"Expected format: +/-[Y:M:D] [H:M:S]\nExamples: +5 (5 hours), +2:30 (2h30m), +0:15:30 (15m30s), +0:0:1 12:00:00 (1d12h)"
                    )
                page.run_task(_show_error)
                return
            # Remove existing shift-time op
            ops = [o for o in ops if o["type"] != "shift-time"]
            ops.append({"type": "shift-time", "tag": "AllDates", "offset": offset_str})
            tag_op_value.value = ""

        elif op == "shift-timezone":
            tz = tag_op_value.value.strip()
            if not tz:
                return
            # Validate format: +/-HH:MM
            import re
            pattern = r'^[+-]\d{2}:\d{2}$'
            if not re.match(pattern, tz):
                from mu_dng_converter.dialogs import show_error
                async def _show_error():
                    await show_error(
                        page,
                        "Invalid Timezone Format",
                        f"Expected format: +/-HH:MM\nExample: +02:00 or -05:00"
                    )
                page.run_task(_show_error)
                return
            # Remove existing shift-timezone op
            ops = [o for o in ops if o["type"] != "shift-timezone"]
            ops.append({"type": "shift-timezone", "tag": "OffsetTimeOriginal", "timezone": tz})
            tag_op_value.value = ""

        state["metadata_ops"] = ops
        rebuild_metadata_ops_list()
        # Note: metadata_ops are session-only, don't persist
        page.update()

    apply_metadata_btn = ft.Button(
        content="Apply", icon=ft.Icons.ADD, style=_btn_style,
        on_click=on_apply_metadata_op,
    )

    # Metadata ops are not persisted - start fresh each run
    state["metadata_ops"] = []
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
            # Note: metadata_ops are session-only, not persisted
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

    # --- Copy/transcode logic (worker function for BatchRunner) ---
    def run_copy_worker(output_path, worker_state):
        """Worker function that runs in background thread."""
        import setproctitle
        from mu_dng_converter.batch_runner import make_state_logger
        setproctitle.setproctitle("mu-dng-converter: DNG → DNG")

        _log = make_state_logger(worker_state)

        try:
            from muimg.cli import run_batch_copy_dng
            from muimg.tiff_metadata import MetadataTags

            dng_files = worker_state["_dng_files"]
            _log(f"Input: {len(dng_files)} DNG files → {output_path}")

            # Process metadata_ops into strip_tags, extra_tags, and time adjustments
            metadata_ops = worker_state.get("metadata_ops", [])
            strip_tags_set = set()
            extra_tags_obj = MetadataTags()
            time_offset_seconds = 0.0
            time_timezone = None

            for op in metadata_ops:
                op_type = op.get("type")
                if op_type == "strip":
                    strip_tags_set.add(op.get("name", "").strip())
                elif op_type == "set":
                    extra_tags_obj.add_tag(op.get("name", ""), op.get("value", ""))
                elif op_type == "shift-time":
                    # Parse offset string following exiftool convention using timedelta
                    # "5" = 5 hours, "2:30" = 2h 30m, "0:15:30" = 15m 30s, "0:0:1 12:00:00" = 1d 12h
                    from datetime import timedelta
                    offset_str = op.get("offset", "").strip()
                    if offset_str:
                        try:
                            sign = -1 if offset_str[0] == "-" else 1
                            body = offset_str[1:]  # Remove sign

                            # Parse date part (if present) and time part
                            if " " in body:
                                date_part, time_part = body.split(" ", 1)
                                # Date: Y:M:D → days only (years/months converted to days)
                                y_m_d = [int(x) for x in date_part.split(":")]
                                years = y_m_d[0] if len(y_m_d) > 0 else 0
                                months = y_m_d[1] if len(y_m_d) > 1 else 0
                                days = y_m_d[2] if len(y_m_d) > 2 else 0
                                total_days = years * 365 + months * 30 + days
                            else:
                                time_part = body
                                total_days = 0

                            # Time: H:M:S (partial allowed)
                            h_m_s = [int(x) for x in time_part.split(":")]
                            hours = h_m_s[0] if len(h_m_s) > 0 else 0
                            minutes = h_m_s[1] if len(h_m_s) > 1 else 0
                            seconds = h_m_s[2] if len(h_m_s) > 2 else 0

                            delta = timedelta(days=total_days, hours=hours, minutes=minutes, seconds=seconds)
                            time_offset_seconds += sign * delta.total_seconds()
                        except (ValueError, IndexError):
                            _log(f"[warn] Could not parse time offset: {offset_str}")
                elif op_type == "shift-timezone":
                    tz = op.get("timezone", "").strip()
                    if tz:
                        time_timezone = tz

            # Remove empty tag names from strip set
            strip_tags_set = {t for t in strip_tags_set if t} or None
            if not extra_tags_obj._tags:
                extra_tags_obj = None

            def on_task_done(completed, total):
                worker_state["progress_fraction"] = completed / total if total > 0 else 0
                worker_state["progress_text"] = f"{completed}/{total}"
                return worker_state["cancel"]

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
            nw = max(1, min(8, int(num_workers.value)))
            _log(
                f"Done: {result['completed']}/{result['total']} files in "
                f"{result['elapsed']:.1f}s ({fps:.1f} files/s, {nw} workers)"
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
            worker_state["running"] = False
            worker_state["finished"] = True

    # --- Setup BatchRunner for run/cancel/polling logic ---
    from mu_dng_converter.batch_runner import BatchRunner

    runner = BatchRunner(
        page=page,
        state=state,
        worker_fn=run_copy_worker,
        run_button=run_button,
        cancel_button=cancel_button,
        progress_bar=progress_bar,
        log_text=log_text,
    )

    async def on_run_wrapper(e):
        await runner.on_run(
            input_path_text=input_path_text,
            output_path_text=output_path_text,
            persist_settings_fn=_persist_settings,
        )

    run_button.on_click = on_run_wrapper
    cancel_button.on_click = runner.on_cancel

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
