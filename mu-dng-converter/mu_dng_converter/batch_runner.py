"""Shared batch processing runner for DNG converter views.

Provides common run/cancel/progress logic across all conversion views.
"""

import threading
from pathlib import Path


class BatchRunner:
    """Handles common batch processing workflow: run, cancel, progress polling.
    
    Each view creates a BatchRunner instance with its specific worker function.
    """

    def __init__(
        self,
        page,
        state,
        worker_fn,
        run_button,
        cancel_button,
        progress_bar,
        log_text,
        on_complete=None,
    ):
        """Initialize the batch runner.
        
        Args:
            page: Flet page instance
            state: View's state dict (must have keys: running, cancel, finished, 
                   progress_fraction, progress_text, log, input_files, last_input_dir, 
                   last_output_dir)
            worker_fn: Function to run in background thread. Signature:
                      worker_fn(output_path, state, update_progress_fn) -> None
            run_button: Flet Button control for Run
            cancel_button: Flet Button control for Cancel
            progress_bar: Flet ProgressBar control
            log_text: Flet TextField control for logging
            on_complete: Optional callback when processing completes
        """
        self.page = page
        self.state = state
        self.worker_fn = worker_fn
        self.run_button = run_button
        self.cancel_button = cancel_button
        self.progress_bar = progress_bar
        self.log_text = log_text
        self.on_complete = on_complete

    def on_cancel(self, e):
        """Handle cancel button click."""
        self.state["cancel"] = True
        current_log = self.state.get("log") or ""
        self.state["log"] = current_log + "Cancellation requested...\n"
        self.page.update()

    async def on_run(
        self,
        input_path_text,
        output_path_text,
        persist_settings_fn,
        file_pattern="*.dng",
    ):
        """Handle run button click.
        
        Validates inputs, checks for overwrites, starts worker thread.
        
        Args:
            input_path_text: Text control with input path
            output_path_text: Text control with output path
            persist_settings_fn: Function to call to persist settings
            file_pattern: Glob pattern for finding files (default: "*.dng")
        """
        from mu_dng_converter.dialogs import check_overwrite

        inp = input_path_text.value
        out = output_path_text.value

        # Validate inputs
        if (
            not inp
            or inp == "No folder selected"
            or not out
            or out == "No folder selected"
        ):
            self.log_text.value = (
                "ERROR: Select input and output folders first.\n"
                + (self.log_text.value or "")
            )
            self.page.update()
            return

        # Get file list
        input_files = self.state.get("input_files")
        if input_files:
            dng_files = sorted(Path(f) for f in input_files)
        else:
            input_dir = Path(inp)
            dng_files = sorted(input_dir.glob(file_pattern))

        if not dng_files:
            self.log_text.value = (
                f"ERROR: No {file_pattern} files found in {inp}\n"
                + (self.log_text.value or "")
            )
            self.page.update()
            return

        # Check for existing files
        output_dir = Path(out)
        existing = [f for f in dng_files if (output_dir / f.name).exists()]
        if existing:
            action = await check_overwrite(self.page, len(existing), len(dng_files))
            if action == "cancel":
                return
            elif action == "skip":
                dng_files = [f for f in dng_files if f not in existing]
                if not dng_files:
                    self.log_text.value = (
                        "All files already exist — nothing to do.\n"
                        + (self.log_text.value or "")
                    )
                    self.page.update()
                    return

        # Start processing
        self.state["running"] = True
        self.state["cancel"] = False
        self.state["finished"] = False
        self.state["progress_fraction"] = 0
        self.state["progress_text"] = ""
        self.state["log"] = ""
        self.run_button.visible = False
        self.cancel_button.visible = True
        self.progress_bar.visible = True
        self.progress_bar.value = 0
        self.state["_old_log"] = self.log_text.value or ""
        self.state["_dng_files"] = dng_files
        persist_settings_fn()
        self.page.update()

        # Start worker thread
        thread = threading.Thread(
            target=self._worker_wrapper,
            args=(out,),
            daemon=True,
        )
        thread.start()
        await self._poll_ui()

    def _worker_wrapper(self, output_path):
        """Wrap the worker function to handle completion."""
        try:
            self.worker_fn(output_path, self.state)
        except Exception as e:
            import traceback
            current_log = self.state.get("log") or ""
            self.state["log"] = current_log + f"\nERROR: {e}\n{traceback.format_exc()}\n"
        finally:
            self.state["finished"] = True

    async def _poll_ui(self):
        """Poll state and update UI."""
        import asyncio

        prev_frac = -1
        prev_log = ""
        max_log_lines = 100

        def build_display_log(current, old):
            parts = [p for p in [current.rstrip(), old.rstrip()] if p]
            combined = ("\n" + "─" * 40 + "\n").join(parts)
            lines = combined.split("\n")
            if len(lines) > max_log_lines:
                lines = lines[:max_log_lines]
            return "\n".join(lines)

        while self.state.get("running"):
            frac = self.state.get("progress_fraction", 0)
            txt = self.state.get("progress_text", "")
            log = self.state.get("log", "")
            changed = False

            # Update progress bar
            if frac != prev_frac:
                self.progress_bar.value = frac
                prev_frac = frac
                changed = True

            # Update progress text
            if txt:
                self.progress_bar.visible = True
                changed = True

            # Update log
            if log != prev_log:
                self.log_text.value = build_display_log(
                    log, self.state.get("_old_log", "")
                )
                prev_log = log
                changed = True

            # Check if finished
            if self.state.get("finished"):
                break

            if changed:
                self.page.update()
            await asyncio.sleep(0.2)

        # Final update
        self.state["running"] = False
        self.state["finished"] = True
        self.run_button.visible = True
        self.cancel_button.visible = False
        self.progress_bar.visible = False

        # Append final log
        remaining = self.state.get("log", "")
        if remaining:
            self.log_text.value = build_display_log(
                remaining, self.state.get("_old_log", "")
            )
        self.progress_bar.value = self.state.get("progress_fraction", 0)
        self.page.update()

        if self.on_complete:
            self.on_complete()
