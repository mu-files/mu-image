"""DNG → TIF/Video conversion view."""

import threading
from pathlib import Path

import flet as ft


def build_dng_view(page: ft.Page) -> ft.Control:
    """Build the DNG conversion tab content."""

    state = {"running": False, "cancel": False}

    # --- Controls ---
    input_path_text = ft.Text("No folder selected", size=13)
    output_path_text = ft.Text("No folder selected", size=13)

    mode_dropdown = ft.Dropdown(
        label="Mode",
        value="tif",
        options=[
            ft.dropdown.Option("tif", "DNG → TIF (with metadata)"),
            ft.dropdown.Option("jxl", "DNG → JXL"),
            ft.dropdown.Option("jpg", "DNG → JPG"),
            ft.dropdown.Option("video", "DNG → Video (MP4)"),
        ],
        width=250,
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
        label="Exposure (EV)", value="0", width=130, keyboard_type=ft.KeyboardType.NUMBER
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
    use_xmp = ft.Checkbox(label="Use XMP metadata", value=True)

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
    log_text = ft.TextField(multiline=True, read_only=True, text_size=11, min_lines=12, expand=True)
    log_row = ft.Row(controls=[log_text])

    run_button = ft.Button(content="Run", icon=ft.Icons.PLAY_ARROW)
    cancel_button = ft.Button(content="Cancel", icon=ft.Icons.STOP, visible=False)

    # --- Async event handlers ---
    async def pick_input(e):
        result = await ft.FilePicker().get_directory_path(dialog_title="Select DNG input folder")
        if result:
            input_path_text.value = result
            page.update()

    async def pick_output(e):
        result = await ft.FilePicker().get_directory_path(dialog_title="Select output folder")
        if result:
            output_path_text.value = result
            page.update()

    def on_mode_changed(e):
        is_video = mode_dropdown.value == "video"
        video_options.visible = is_video
        log_text.min_lines = 4 if is_video else 12
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

    def on_cancel(e):
        state["cancel"] = True
        log_text.value = (log_text.value or "") + "Cancellation requested...\n"
        page.update()

    def on_run(e):
        inp = input_path_text.value
        out = output_path_text.value
        if inp == "No folder selected" or out == "No folder selected":
            log_text.value = "ERROR: Select input and output folders first.\n"
            page.update()
            return

        state["running"] = True
        state["cancel"] = False
        run_button.visible = False
        cancel_button.visible = True
        progress_bar.visible = True
        progress_bar.value = 0
        log_text.value = ""
        page.update()

        thread = threading.Thread(target=run_conversion, args=(inp, out), daemon=True)
        thread.start()

    run_button.on_click = on_run
    cancel_button.on_click = on_cancel

    # --- Conversion logic (runs in thread) ---
    def run_conversion(input_path, output_path):
        import time
        import numpy as np

        try:
            from muimg.dngio import decode_dng, DngFile, DemosaicAlgorithm
            from muimg.imgio import write_image

            mode = mode_dropdown.value
            input_dir = Path(input_path)
            dng_files = sorted(input_dir.glob("*.dng"))
            if not dng_files:
                log(f"No .dng files found in {input_path}")
                finish()
                return

            total = len(dng_files)
            log(f"Found {total} DNG files.")

            rendering_params = build_rendering_params()
            output_dtype = np.uint16 if bit_depth.value == "16" else np.uint8

            output_dir = Path(output_path)
            output_dir.mkdir(parents=True, exist_ok=True)

            start_time = time.perf_counter()
            completed = 0

            for i, dng_path in enumerate(dng_files):
                if state["cancel"]:
                    log("Cancelled.")
                    break
                try:
                    dng_file = DngFile(dng_path)
                    img, metadata = decode_dng(
                        file=dng_file,
                        output_dtype=output_dtype,
                        demosaic_algorithm=DemosaicAlgorithm.OPENCV_EA,
                        use_coreimage_if_available=False,
                        use_xmp=use_xmp.value,
                        rendering_params=rendering_params,
                        strict=False,
                    )
                    if img is not None:
                        out_file = output_dir / f"{dng_path.stem}.{mode}"
                        write_image(img, out_file, metadata=metadata)
                        completed += 1
                    else:
                        log(f"  WARN: Failed to decode {dng_path.name}")
                except Exception as ex:
                    log(f"  ERROR: {dng_path.name}: {ex}")

                elapsed = time.perf_counter() - start_time
                fps = (i + 1) / elapsed if elapsed > 0 else 0
                update_progress((i + 1) / total, f"{i+1}/{total} ({fps:.1f} files/s)")

            elapsed = time.perf_counter() - start_time
            log(f"Done: {completed}/{total} files in {elapsed:.1f}s")
        except Exception as ex:
            log(f"ERROR: {ex}")
        finally:
            finish()

    # --- Helpers ---
    def build_rendering_params():
        params = {}
        if temperature.value:
            params["Temperature"] = float(temperature.value)
        if tint.value:
            params["Tint"] = float(tint.value)
        if exposure.value:
            params["Exposure2012"] = float(exposure.value)
        return params

    def update_progress(fraction, text):
        progress_bar.value = fraction
        progress_text.value = text
        page.update()

    def log(message):
        log_text.value = (log_text.value or "") + message + "\n"
        page.update()

    def finish():
        state["running"] = False
        run_button.visible = True
        cancel_button.visible = False
        page.update()

    # --- Build layout ---
    input_btn = ft.Button(content="Select Input Folder", icon=ft.Icons.FOLDER_OPEN, on_click=pick_input)
    output_btn = ft.Button(content="Select Output Folder", icon=ft.Icons.FOLDER, on_click=pick_output)

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
            ft.Row(controls=[input_btn, input_path_text]),
            ft.Row(controls=[output_btn, output_path_text]),
            ft.Divider(height=8),
            ft.Text("Rendering Parameters", weight=ft.FontWeight.BOLD, size=13),
            ft.Row(controls=[wb_dropdown, temperature, tint, exposure], wrap=True),
            ft.Divider(height=8),
            ft.Text("Output Options", weight=ft.FontWeight.BOLD, size=13),
            ft.Row(controls=[bit_depth, num_workers, use_xmp], wrap=True),
            video_options,
            ft.Divider(height=8),
            progress_bar,
            log_row,
        ],
        expand=True,
    )
