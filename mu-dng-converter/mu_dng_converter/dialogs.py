"""Cross-platform file/folder dialog helpers.

On macOS, uses pyobjc (NSOpenPanel) for native dialogs with full features
including "New Folder" button. Falls back to Flet's FilePicker on other
platforms.
"""

import asyncio
import sys
from pathlib import Path

import flet as ft

IS_MACOS = sys.platform == "darwin"


def _ensure_app_active():
    """Ensure NSApplication is active without showing a dock icon."""
    from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
    app = NSApplication.sharedApplication()
    # Accessory policy = no dock icon, no menu bar
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    app.activateIgnoringOtherApps_(True)


def _pick_directory_macos(title: str, initial_directory: str | None = None,
                         can_create_directories: bool = True) -> str | None:
    """Open native macOS NSOpenPanel for directory selection."""
    from AppKit import NSOpenPanel, NSModalResponseOK, NSFloatingWindowLevel
    from Foundation import NSURL

    _ensure_app_active()

    panel = NSOpenPanel.openPanel()
    panel.setCanChooseDirectories_(True)
    panel.setCanCreateDirectories_(can_create_directories)
    panel.setCanChooseFiles_(False)
    panel.setAllowsMultipleSelection_(False)
    panel.setTitle_(title)
    panel.setPrompt_("Choose")
    panel.setLevel_(NSFloatingWindowLevel)

    if initial_directory and Path(initial_directory).is_dir():
        url = NSURL.fileURLWithPath_(initial_directory)
        panel.setDirectoryURL_(url)

    response = panel.runModal()
    if response == NSModalResponseOK:
        urls = panel.URLs()
        if urls and len(urls) > 0:
            return str(urls[0].path())
    return None


def _pick_files_macos(
    title: str,
    initial_directory: str | None = None,
    allowed_extensions: list[str] | None = None,
    allow_multiple: bool = False,
) -> list[str] | None:
    """Open native macOS NSOpenPanel for file selection."""
    from AppKit import NSOpenPanel, NSModalResponseOK, NSFloatingWindowLevel
    from Foundation import NSURL

    _ensure_app_active()

    panel = NSOpenPanel.openPanel()
    panel.setCanChooseDirectories_(False)
    panel.setCanCreateDirectories_(False)
    panel.setCanChooseFiles_(True)
    panel.setAllowsMultipleSelection_(allow_multiple)
    panel.setTitle_(title)
    panel.setLevel_(NSFloatingWindowLevel)

    if initial_directory and Path(initial_directory).is_dir():
        url = NSURL.fileURLWithPath_(initial_directory)
        panel.setDirectoryURL_(url)

    if allowed_extensions:
        from AppKit import UTType
        types = []
        for ext in allowed_extensions:
            t = UTType.typeWithFilenameExtension_(ext)
            if t:
                types.append(t)
        if types:
            panel.setAllowedContentTypes_(types)

    response = panel.runModal()
    if response == NSModalResponseOK:
        urls = panel.URLs()
        if urls:
            return [str(u.path()) for u in urls]
    return None


async def pick_directory_async(title: str = "Select Folder",
                               initial_directory: str | None = None,
                               can_create_directories: bool = True) -> str | None:
    """Pick a directory using native dialog (macOS) or Flet fallback."""
    if IS_MACOS:
        try:
            return _pick_directory_macos(title, initial_directory, can_create_directories)
        except ImportError:
            pass
    return await ft.FilePicker().get_directory_path(
        dialog_title=title, initial_directory=initial_directory)


async def pick_files_async(
    title: str = "Select Files",
    initial_directory: str | None = None,
    allowed_extensions: list[str] | None = None,
    allow_multiple: bool = False,
) -> list[str] | None:
    """Pick files using native dialog (macOS) or Flet fallback."""
    if IS_MACOS:
        try:
            return _pick_files_macos(
                title, initial_directory, allowed_extensions, allow_multiple
            )
        except ImportError:
            pass
    files = await ft.FilePicker().pick_files(
        dialog_title=title,
        initial_directory=initial_directory,
        allowed_extensions=allowed_extensions,
        file_type=ft.FilePickerFileType.CUSTOM if allowed_extensions else ft.FilePickerFileType.ANY,
        allow_multiple=allow_multiple,
    )
    return [f.path for f in files] if files else None


async def check_overwrite(page, existing_count: int, total_count: int) -> str:
    """Show overwrite confirmation dialog.

    Args:
        page: Flet page instance.
        existing_count: Number of output files that already exist.
        total_count: Total number of files to process.

    Returns:
        "overwrite", "skip", or "cancel".
    """
    result_future: asyncio.Future[str] = asyncio.get_event_loop().create_future()

    def _close_dlg():
        dlg.open = False
        page.update()

    def on_overwrite(e):
        _close_dlg()
        if not result_future.done():
            result_future.set_result("overwrite")

    def on_skip(e):
        _close_dlg()
        if not result_future.done():
            result_future.set_result("skip")

    def on_cancel(e):
        _close_dlg()
        if not result_future.done():
            result_future.set_result("cancel")

    dlg = ft.AlertDialog(
        modal=True,
        title=ft.Text("Files already exist"),
        content=ft.Text(
            f"{existing_count} of {total_count} output files already exist "
            f"in the destination folder."
        ),
        actions=[
            ft.TextButton("Overwrite All", on_click=on_overwrite),
            ft.TextButton("Skip Existing", on_click=on_skip),
            ft.TextButton("Cancel", on_click=on_cancel),
        ],
        actions_alignment=ft.MainAxisAlignment.END,
    )

    page.overlay.append(dlg)
    dlg.open = True
    page.update()

    return await result_future
