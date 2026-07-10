"""Bridge module for PyWebView JavaScript-Python communication."""

import webview
from webview import FileDialog
import asyncio
from pathlib import Path
import json
import platform
from typing import Dict, Any, Optional


def _get_config_dir() -> Path:
    """Get the application config directory (platform-specific)."""
    if platform.system() == "Darwin":
        config_dir = Path.home() / "Library" / "Application Support" / "mu-dng-converter"
    elif platform.system() == "Windows":
        config_dir = Path.home() / "AppData" / "Local" / "mu-dng-converter"
    else:
        config_dir = Path.home() / ".config" / "mu-dng-converter"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir


def _get_folder_cache_path() -> Path:
    """Get the path to the folder cache JSON file."""
    return _get_config_dir() / "folder_cache.json"


class WebViewBridge:
    """Bridge class to handle JavaScript-Python communication."""
    
    def __init__(self):
        self.window = None
        self.cancel_flags = {}  # Cancel flags per tab
        self.folder_cache = self._load_folder_cache()

    def _load_folder_cache(self) -> Dict[str, str]:
        """Load folder cache from disk."""
        try:
            cache_path = _get_folder_cache_path()
            if cache_path.exists():
                return json.loads(cache_path.read_text())
        except Exception as e:
            print(f"Failed to load folder cache: {e}")
        return {}

    def _save_folder_cache(self):
        """Save folder cache to disk."""
        try:
            cache_path = _get_folder_cache_path()
            cache_path.write_text(json.dumps(self.folder_cache, indent=2))
        except Exception as e:
            print(f"Failed to save folder cache: {e}")

    def _get_cached_folder(self, key: str) -> str:
        """Get cached folder for a button key, verifying it exists."""
        folder = self.folder_cache.get(key)
        if folder is not None and Path(folder).is_dir():
            return folder
        return ""

    def _set_cached_folder(self, key: str, folder: Optional[str]):
        """Set cached folder for a button key and persist."""
        if folder is not None and Path(folder).is_dir():
            self.folder_cache[key] = folder
            self._save_folder_cache()
        
    def set_window(self, window):
        """Set the webview window instance."""
        self.window = window
        
    def select_input(
        self,
        tab: str,
        mode: str = "folder",
        file_type: str = "dng",
    ) -> str:
        """Handle input folder/file selection."""
        try:
            cache_key = f"{tab}-input"
            directory = self._get_cached_folder(cache_key) or None

            if mode == "folder":
                result = self.window.create_file_dialog(
                    FileDialog.FOLDER,
                    directory=directory,
                ) if directory else self.window.create_file_dialog(FileDialog.FOLDER)
            else:
                if file_type == "fits":
                    file_types = ('FITS files (*.fits;*.fit)', 'All files (*.*)')
                else:
                    file_types = ('DNG files (*.dng)', 'All files (*.*)')
                if directory:
                    result = self.window.create_file_dialog(
                        FileDialog.OPEN,
                        directory=directory,
                        allow_multiple=True,
                        file_types=file_types,
                    )
                else:
                    result = self.window.create_file_dialog(
                        FileDialog.OPEN,
                        allow_multiple=True,
                        file_types=file_types,
                    )

            if result:
                paths = result if isinstance(result, (list, tuple)) else [result]
                # Cache the folder (first path's parent for files, or the folder itself)
                if paths and paths[0]:
                    folder = paths[0] if mode == "folder" else str(Path(paths[0]).parent)
                    self._set_cached_folder(cache_key, folder)
                if isinstance(result, (list, tuple)):
                    return "\n".join(str(p) for p in result if p)
                return result
            return ""
        except Exception as e:
            import traceback
            print(f"Error selecting input for {tab}: {e}")
            traceback.print_exc()
            return ""

    def select_output(
        self,
        tab: str,
        mode: str = "folder",
    ) -> str:
        """Handle output folder (or save-file) selection."""
        try:
            cache_key = f"{tab}-output"
            directory = self._get_cached_folder(cache_key) or None

            if mode == "file":
                if directory:
                    result = self.window.create_file_dialog(
                        FileDialog.SAVE,
                        directory=directory,
                        save_filename="output.mp4",
                    )
                else:
                    result = self.window.create_file_dialog(
                        FileDialog.SAVE,
                        save_filename="output.mp4",
                    )
                if result:
                    path = result[0] if isinstance(result, (list, tuple)) else result
                    if path:
                        if not path.lower().endswith(".mp4"):
                            path += ".mp4"
                        # Cache the parent folder
                        self._set_cached_folder(cache_key, str(Path(path).parent))
                        return path
                return ""
            if directory:
                result = self.window.create_file_dialog(
                    FileDialog.FOLDER,
                    directory=directory,
                )
            else:
                result = self.window.create_file_dialog(FileDialog.FOLDER)
            if result:
                folder = result[0] if isinstance(result, (list, tuple)) else result
                if folder:
                    self._set_cached_folder(cache_key, folder)
                    return folder
            return ""
        except Exception as e:
            import traceback
            print(f"Error selecting output for {tab}: {e}")
            traceback.print_exc()
            return ""
    
    def _js(self, fn: str, *args):
        """Call a global JS function with JSON-encoded arguments."""
        payload = ", ".join(json.dumps(a) for a in args)
        self.window.evaluate_js(f"{fn}({payload})")

    def validate_tag(self, name: str) -> Optional[str]:
        """Check whether a tag name exists in the TIFF tag registry.
        
        Returns:
            The correct case-sensitive tag name if found (exact or case-insensitive),
            or None if not found.
        """
        try:
            from muimg.tiff_metadata import TIFF_TAG_TYPE_REGISTRY
            
            # Exact match
            if name in TIFF_TAG_TYPE_REGISTRY:
                return name
            
            # Case-insensitive search
            name_lower = name.lower()
            for registry_name in TIFF_TAG_TYPE_REGISTRY:
                if registry_name.lower() == name_lower:
                    return registry_name
            
            return None
        except Exception as e:
            print(f"Error validating tag {name}: {e}")
            return None

    def cancel_conversion(self, tab: str):
        """Handle cancel button click for a specific tab."""
        self.cancel_flags[tab] = True
        self._js("updateProgress", tab, "Cancelling...")

    def run_conversion(self, tab: str, settings: Dict[str, Any]):
        """Run a conversion for a tab. Blocks until done (pywebview runs
        each JS API call in its own thread)."""

        def log(msg):
            self._js("appendLog", tab, msg)

        def progress_bar(value):
            self._js("updateProgressBar", tab, value)

        self.cancel_flags[tab] = False
        try:
            if tab == "create-dng":
                if settings.get("inputType") == "fits":
                    return self._run_create_fits(tab, settings, log, progress_bar)
                return self._run_create_dng(tab, settings, log, progress_bar)
            if tab == "render-dng":
                return self._run_render(tab, settings, log, progress_bar)
            log(f"ERROR: Unknown tab {tab}")
            return "error"
        except Exception as e:
            import traceback
            log(f"ERROR: {e}\n{traceback.format_exc()}")
            return "error"

    def _gather_input_files(self, settings, suffixes, log):
        """Resolve the input file list from settings. Returns (files, output)
        or (None, None) on error (already logged)."""
        inputs = settings.get("input") or []
        mode = settings.get("inputMode", "folder")
        output = settings.get("output") or ""
        if not inputs or not output:
            log("ERROR: Select input and output folders first.")
            return None, None

        if mode == "folder":
            folder = Path(inputs[0])
            files = sorted(
                p for p in folder.iterdir()
                if p.suffix.lower() in suffixes
            ) if folder.is_dir() else []
        else:
            files = sorted(Path(f) for f in inputs)

        if not files:
            log(f"ERROR: No {'/'.join('*' + s for s in suffixes)} files found in {inputs[0]}")
            return None, None
        return files, output

    def _apply_overwrite_policy(self, files, existing, settings, log):
        """Check for existing files and apply overwrite action. Returns
        (files, response); a non-None response should be returned as-is.
        
        Always prompts if files exist, regardless of previous overwriteAction.
        The action only applies if it was set during THIS run (from the modal).
        """
        action = settings.get("overwriteAction")
        
        # If files exist and no action was provided THIS run, prompt
        if existing and not action:
            return files, {
                "status": "confirm-overwrite",
                "existing": len(existing),
                "total": len(files),
            }
        
        # If user chose 'skip' in the modal, filter out existing files
        if action == "skip" and existing:
            files = [f for f in files if f not in existing]
            if not files:
                log("All files already exist — nothing to do.")
                return files, "ok"
        
        # If action is 'overwrite' or no files exist, proceed with all files
        return files, None

    def _make_on_task_done(self, tab, progress_bar):
        def on_task_done(completed, total):
            if completed == 0 and total > 0:
                # Show tiny progress (1%) to indicate work has started
                progress_bar(1.)
            else:
                frac = completed / total if total > 0 else 0
                progress_bar(round(frac * 100, 1))
            return self.cancel_flags.get(tab, False)
        return on_task_done

    def _parse_metadata_ops(self, metadata_ops, log):
        """Parse UI metadata ops into engine arguments (see common.parse_metadata_ops)."""
        from mu_dng_converter.common import parse_metadata_ops
        return parse_metadata_ops(metadata_ops, log)

    @staticmethod
    def _resolve_demosaic_algorithm(settings):
        from muimg.raw_render import DemosaicAlgorithm
        try:
            return DemosaicAlgorithm.lookup(settings.get("demosaicAlgo") or "DNGSDK_BILINEAR")
        except Exception:
            return DemosaicAlgorithm.DNGSDK_BILINEAR

    def _run_create_dng(self, tab, settings, log, progress_bar):
        """DNG → DNG copy/transcode (Create DNG tab, DNG input)."""
        from muimg.cli import run_batch_copy_dng

        dng_files, output = self._gather_input_files(settings, (".dng",), log)
        if dng_files is None:
            return "error"

        output_dir = Path(output)
        existing = [f for f in dng_files if (output_dir / f.name).exists()]
        dng_files, response = self._apply_overwrite_policy(dng_files, existing, settings, log)
        if response is not None:
            return response
        
        # Clear overwrite action after it's been used
        settings.pop("overwriteAction", None)

        strip_tags_set, extra_tags_obj, time_offset_seconds, time_timezone = \
            self._parse_metadata_ops(settings.get("metadataOps"), log)

        log(f"Input: {len(dng_files)} DNG files → {output}")

        jxl_distance_val = settings.get("jxlDistance")
        num_workers = max(1, min(8, int(settings.get("numWorkers") or 4)))
        result = run_batch_copy_dng(
            dng_files=dng_files,
            output_folder=output,
            mode="transcode" if settings.get("transcode") else "copy",
            compression_name=settings.get("compression", "uncompressed"),
            jxl_distance=float(jxl_distance_val) if jxl_distance_val not in (None, "") else None,
            jxl_effort=int(settings.get("jxlEffort") or 5),
            scale=max(0.125, min(1.0, float(settings.get("scale") or 1.0))),
            demosaic=bool(settings.get("demosaic")),
            demosaic_algorithm=self._resolve_demosaic_algorithm(settings),
            do_preview=bool(settings.get("preview")),
            do_fast_load=bool(settings.get("fastLoad")),
            strip_tags=strip_tags_set,
            extra_tags=extra_tags_obj,
            time_offset_seconds=time_offset_seconds,
            time_timezone=time_timezone,
            num_workers=num_workers,
            on_task_done=self._make_on_task_done(tab, progress_bar),
        )

        fps = result["completed"] / result["elapsed"] if result["elapsed"] > 0 else 0
        log(
            f"Done: {result['completed']}/{result['total']} files in "
            f"{result['elapsed']:.1f}s ({fps:.1f} files/s, {num_workers} workers)"
        )
        return "ok"

    def _run_create_fits(self, tab, settings, log, progress_bar):
        """FITS → DNG conversion (Create DNG tab, FITS input)."""
        from mu_dng_converter.fits2dng import run_batch_fits_to_dng
        from muimg.raw_render import temp_tint_to_xy

        fits_files, output = self._gather_input_files(settings, (".fits", ".fit"), log)
        if fits_files is None:
            return "error"

        output_dir = Path(output)
        existing = [f for f in fits_files if (output_dir / (f.stem + ".dng")).exists()]
        fits_files, response = self._apply_overwrite_policy(fits_files, existing, settings, log)
        if response is not None:
            return response
        
        # Clear overwrite action after it's been used
        settings.pop("overwriteAction", None)

        strip_tags_set, extra_tags_obj, time_offset_seconds, time_timezone = \
            self._parse_metadata_ops(settings.get("metadataOps"), log)

        transcode = bool(settings.get("transcode"))
        compression = (
            settings.get("compression", "uncompressed") if transcode else "uncompressed"
        )
        jxl_distance_val = settings.get("jxlDistance")

        wb_preset = settings.get("wbPreset", "d50")
        if wb_preset == "d50":
            wb_xy = None
        else:
            try:
                temp_val = float(settings.get("temperature") or 5000)
                tint_val = float(settings.get("tint") or 0)
            except (TypeError, ValueError):
                temp_val, tint_val = 5000.0, 0.0
            wb_xy = temp_tint_to_xy(temp_val, tint_val)

        num_workers = max(1, min(8, int(settings.get("numWorkers") or 4)))
        log(f"Input: {len(fits_files)} FITS files → {output}")

        result = run_batch_fits_to_dng(
            fits_files=fits_files,
            output_folder=output,
            compression_name=compression,
            jxl_distance=float(jxl_distance_val) if transcode and jxl_distance_val not in (None, "") else None,
            jxl_effort=int(settings.get("jxlEffort") or 5) if transcode else None,
            demosaic=bool(settings.get("demosaic")),
            demosaic_algorithm=self._resolve_demosaic_algorithm(settings),
            scale=max(0.125, min(1.0, float(settings.get("scale") or 1.0))),
            auto_exposure=bool(settings.get("autoExposure", True)),
            ev_value=settings.get("exposure") or 0,
            use_tone_curve=bool(settings.get("toneCurve", True)),
            wb_xy=wb_xy,
            strip_tags=strip_tags_set,
            extra_tags=extra_tags_obj,
            time_offset_seconds=time_offset_seconds,
            time_timezone=time_timezone,
            do_preview=bool(settings.get("preview")),
            do_fast_load=bool(settings.get("fastLoad")),
            num_workers=num_workers,
            on_task_done=self._make_on_task_done(tab, progress_bar),
            log_callback=log,
        )

        elapsed = result["elapsed"]
        fps = result["written"] / elapsed if elapsed > 0 else 0
        parts = [f"{result['written']} written"]
        if result["skipped"]:
            parts.append(f"{result['skipped']} skipped")
        if result["errored"]:
            parts.append(f"{result['errored']} errors")
        log(
            f"Done: {', '.join(parts)} / {result['total']} total "
            f"in {elapsed:.1f}s ({fps:.1f} files/s)"
        )
        return "ok"

    def _run_render(self, tab, settings, log, progress_bar):
        """DNG → TIF/JPG/Video conversion (Render DNG tab)."""
        mode = settings.get("mode", "tif")
        dng_files, output = self._gather_input_files(settings, (".dng",), log)
        if dng_files is None:
            return "error"

        if mode != "video":
            output_dir = Path(output)
            existing = [
                f for f in dng_files
                if (output_dir / (f.stem + "." + mode)).exists()
            ]
            dng_files, response = self._apply_overwrite_policy(dng_files, existing, settings, log)
            if response is not None:
                return response
            
            # Clear overwrite action after it's been used
            settings.pop("overwriteAction", None)

        # Build rendering parameters from settings
        use_xmp = bool(settings.get("useXmp", True))
        rendering_params = {}
        if not use_xmp:
            if settings.get("temperature"):
                rendering_params["Temperature"] = float(settings["temperature"])
            if settings.get("tint"):
                rendering_params["Tint"] = float(settings["tint"])
            rendering_params["Exposure2012"] = float(settings.get("exposure") or 0)

        num_workers = max(1, min(8, int(settings.get("numWorkers") or 4)))
        on_task_done = self._make_on_task_done(tab, progress_bar)

        if mode == "video":
            from muimg.cli import run_batch_to_video
            output_mp4 = Path(output)
            log(f"Input: {len(dng_files)} DNG files → {output_mp4.name}")
            w, h = (settings.get("resolution") or "1920x1080").lower().split("x")
            result = run_batch_to_video(
                dng_files=dng_files,
                output_mp4=output_mp4,
                rendering_params=rendering_params,
                use_xmp=use_xmp,
                resolution=(int(w), int(h)),
                codec=settings.get("codec", "h264"),
                crf=int(settings.get("crf") or 20),
                bit_depth=settings.get("videoBitDepth", "8"),
                frame_rate=float(settings.get("frameRate") or 30),
                num_workers=num_workers,
                overlay_txt=bool(settings.get("overlay")),
                on_task_done=on_task_done,
            )
        else:
            from muimg.cli import run_batch_convert
            log(f"Input: {len(dng_files)} DNG files → .{mode}")
            result = run_batch_convert(
                dng_files=dng_files,
                output_folder=output,
                output_format=mode,
                bit_depth=settings.get("bitDepth", "16"),
                rendering_params=rendering_params,
                use_xmp=use_xmp,
                scale=max(0.125, min(1.0, float(settings.get("scale") or 1.0))),
                num_workers=num_workers,
                on_task_done=on_task_done,
                log_callback=log,
            )

        fps = result["completed"] / result["elapsed"] if result["elapsed"] > 0 else 0
        log(
            f"Done: {result['completed']}/{result['total']} files in "
            f"{result['elapsed']:.1f}s ({fps:.1f} files/s, {num_workers} workers)"
        )
        return "ok"
    
    def get_settings(self, tab: str) -> Dict[str, Any]:
        """Get current settings for a tab."""
        # TODO: Implement settings retrieval from JavaScript
        return {}
    
    def set_settings(self, tab: str, settings: Dict[str, Any]):
        """Set settings for a tab."""
        # TODO: Implement settings persistence to JavaScript
        pass


# Global bridge instance
bridge = WebViewBridge()


def expose_to_window(window):
    """Expose bridge methods to the webview window."""
    bridge.set_window(window)
    
    # Expose all bridge methods to JavaScript
    window.expose(bridge.select_input)
    window.expose(bridge.select_output)
    window.expose(bridge.run_conversion)
    window.expose(bridge.cancel_conversion)
    window.expose(bridge.validate_tag)
    window.expose(bridge.get_settings)
    window.expose(bridge.set_settings)
