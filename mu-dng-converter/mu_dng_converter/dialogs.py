"""Cross-platform file/folder dialog helpers.

On macOS, uses pyobjc (NSOpenPanel) for native dialogs with full features
including "New Folder" button. Falls back to Flet's FilePicker on other
platforms.
"""

import sys
from pathlib import Path


def _ensure_app_active():
    """Ensure NSApplication is active without showing a dock icon."""
    from AppKit import NSApplication, NSApplicationActivationPolicyAccessory
    app = NSApplication.sharedApplication()
    # Accessory policy = no dock icon, no menu bar
    app.setActivationPolicy_(NSApplicationActivationPolicyAccessory)
    app.activateIgnoringOtherApps_(True)


def _pick_directory_macos(title: str, initial_directory: str | None = None) -> str | None:
    """Open native macOS NSOpenPanel for directory selection."""
    from AppKit import NSOpenPanel, NSModalResponseOK, NSFloatingWindowLevel
    from Foundation import NSURL

    _ensure_app_active()

    panel = NSOpenPanel.openPanel()
    panel.setCanChooseDirectories_(True)
    panel.setCanCreateDirectories_(True)
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


def pick_directory(title: str = "Select Folder", initial_directory: str | None = None) -> str | None:
    """Pick a directory using native dialog (macOS) or return None to signal fallback.

    Returns:
        Selected directory path, or None if cancelled or not on macOS.
    """
    if sys.platform == "darwin":
        try:
            return _pick_directory_macos(title, initial_directory)
        except ImportError:
            return None
    return None


def pick_files(
    title: str = "Select Files",
    initial_directory: str | None = None,
    allowed_extensions: list[str] | None = None,
    allow_multiple: bool = False,
) -> list[str] | None:
    """Pick files using native dialog (macOS) or return None to signal fallback.

    Returns:
        List of selected file paths, or None if cancelled or not on macOS.
    """
    if sys.platform == "darwin":
        try:
            return _pick_files_macos(
                title, initial_directory, allowed_extensions, allow_multiple
            )
        except ImportError:
            return None
    return None


IS_MACOS = sys.platform == "darwin"
