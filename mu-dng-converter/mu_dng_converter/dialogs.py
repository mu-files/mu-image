"""Cross-platform file/folder dialog helpers using Flet's FilePicker."""

import asyncio
from pathlib import Path

import flet as ft


async def pick_directory_async(title: str = "Select Folder",
                               initial_directory: str | None = None,
                               can_create_directories: bool = True,
                               picker: ft.FilePicker | None = None) -> str | None:
    """Pick a directory using Flet's FilePicker.

    Validates initial_directory exists before passing it — an invalid path
    silently breaks get_directory_path on Windows.
    """
    import os
    if initial_directory:
        initial_directory = os.path.normpath(initial_directory)
        if not Path(initial_directory).exists():
            initial_directory = None
    if picker is None:
        picker = ft.FilePicker()
    return await picker.get_directory_path(
        dialog_title=title, initial_directory=initial_directory)


async def pick_files_async(
    title: str = "Select Files",
    initial_directory: str | None = None,
    allowed_extensions: list[str] | None = None,
    allow_multiple: bool = False,
    picker: ft.FilePicker | None = None,
) -> list[str] | None:
    """Pick files using Flet's FilePicker."""
    import os
    if initial_directory:
        initial_directory = os.path.normpath(initial_directory)
    if picker is None:
        picker = ft.FilePicker()
    files = await picker.pick_files(
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
