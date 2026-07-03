"""Bridge module for PyWebView JavaScript-Python communication."""

import webview
from webview import FileDialog
import asyncio
from pathlib import Path
import json
from typing import Dict, Any, Optional

class WebViewBridge:
    """Bridge class to handle JavaScript-Python communication."""
    
    def __init__(self):
        self.window = None
        self.cancel_flags = {}  # Cancel flags per tab
        
    def set_window(self, window):
        """Set the webview window instance."""
        self.window = window
        
    def select_input(self, tab: str, mode: str = "folder") -> str:
        """Handle input folder/file selection."""
        try:
            if mode == "folder":
                result = self.window.create_file_dialog(FileDialog.FOLDER)
            else:
                if tab == 'fits-dng':
                    file_types = ('FITS files (*.fits;*.fit)', 'All files (*.*)')
                else:
                    file_types = ('DNG files (*.dng)', 'All files (*.*)')
                result = self.window.create_file_dialog(
                    FileDialog.OPEN,
                    allow_multiple=True,
                    file_types=file_types,
                )

            if result:
                if isinstance(result, (list, tuple)):
                    return "\n".join(result)
                return result
            return ""
        except Exception as e:
            print(f"Error selecting input for {tab}: {e}")
            return ""

    def select_output(self, tab: str) -> str:
        """Handle output folder selection."""
        try:
            result = self.window.create_file_dialog(FileDialog.FOLDER)
            if result:
                return result[0] if isinstance(result, (list, tuple)) else result
            return ""
        except Exception as e:
            print(f"Error selecting output for {tab}: {e}")
            return ""
    
    def _js(self, fn: str, *args):
        """Call a global JS function with JSON-encoded arguments."""
        payload = ", ".join(json.dumps(a) for a in args)
        self.window.evaluate_js(f"{fn}({payload})")

    def cancel_conversion(self, tab: str):
        """Handle cancel button click for a specific tab."""
        self.cancel_flags[tab] = True
        self._js("updateProgress", tab, "Cancelling...")

    def run_conversion(self, tab: str, settings: Dict[str, Any]) -> str:
        """Run a conversion for a tab. Blocks until done (pywebview runs
        each JS API call in its own thread)."""

        def log(msg):
            self._js("appendLog", tab, msg)

        def progress(msg):
            self._js("updateProgress", tab, msg)

        def progress_bar(value):
            self._js("updateProgressBar", tab, value)

        if tab != "dng-dng":
            progress("Not implemented yet")
            return "not-implemented"

        self.cancel_flags[tab] = False
        try:
            from muimg.cli import run_batch_copy_dng
            from muimg.tiff_metadata import MetadataTags

            inputs = settings.get("input") or []
            mode = settings.get("inputMode", "folder")
            output = settings.get("output") or ""
            if not inputs or not output:
                progress("Error")
                log("ERROR: Select input and output folders first.")
                return "error"

            if mode == "folder":
                dng_files = sorted(Path(inputs[0]).glob("*.dng"))
            else:
                dng_files = sorted(Path(f) for f in inputs)

            if not dng_files:
                progress("Error")
                log(f"ERROR: No *.dng files found in {inputs[0]}")
                return "error"

            # Overwrite check — handled by an in-page modal on the JS side
            output_dir = Path(output)
            existing = [f for f in dng_files if (output_dir / f.name).exists()]
            if existing and not settings.get("overwriteConfirmed"):
                return {
                    "status": "confirm-overwrite",
                    "existing": len(existing),
                    "total": len(dng_files),
                }

            # Process metadata ops
            metadata_ops = settings.get("metadataOps", [])
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
                    from datetime import timedelta
                    offset_str = op.get("offset", "").strip()
                    if offset_str:
                        try:
                            sign = -1 if offset_str[0] == "-" else 1
                            body = offset_str[1:] if offset_str[0] in "+-" else offset_str
                            if " " in body:
                                days_part, time_part = body.split(" ", 1)
                                days = int(days_part)
                            else:
                                time_part = body
                                days = 0
                            h_m_s = [int(x) for x in time_part.split(":")]
                            hours = h_m_s[0] if len(h_m_s) > 0 else 0
                            minutes = h_m_s[1] if len(h_m_s) > 1 else 0
                            seconds = h_m_s[2] if len(h_m_s) > 2 else 0
                            delta = timedelta(days=days, hours=hours, minutes=minutes, seconds=seconds)
                            time_offset_seconds += sign * delta.total_seconds()
                        except (ValueError, IndexError):
                            log(f"[warn] Could not parse time offset: {offset_str}")
                elif op_type == "shift-timezone":
                    tz = op.get("timezone", "").strip()
                    if tz:
                        time_timezone = tz

            strip_tags_set = {t for t in strip_tags_set if t} or None
            if not extra_tags_obj._tags:
                extra_tags_obj = None

            def on_task_done(completed, total):
                frac = completed / total if total > 0 else 0
                progress_bar(round(frac * 100, 1))
                progress(f"{completed}/{total}")
                return self.cancel_flags.get(tab, False)

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
                demosaic_algorithm=None,
                do_preview=bool(settings.get("preview")),
                do_fast_load=bool(settings.get("fastLoad")),
                strip_tags=strip_tags_set,
                extra_tags=extra_tags_obj,
                time_offset_seconds=time_offset_seconds,
                time_timezone=time_timezone,
                num_workers=num_workers,
                on_task_done=on_task_done,
            )

            fps = result["completed"] / result["elapsed"] if result["elapsed"] > 0 else 0
            log(
                f"Done: {result['completed']}/{result['total']} files in "
                f"{result['elapsed']:.1f}s ({fps:.1f} files/s, {num_workers} workers)"
            )
            progress("")
            return "ok"
        except Exception as e:
            import traceback
            log(f"ERROR: {e}\n{traceback.format_exc()}")
            progress("Error")
            return "error"
    
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
    window.expose(bridge.get_settings)
    window.expose(bridge.set_settings)
