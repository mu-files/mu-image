"""Entry point for PyWebView build."""

import webview
import os
import sys
from pathlib import Path
from mu_dng_converter.webview_bridge import expose_to_window


def _set_macos_icon():
    """Set the macOS Dock icon to the app icon."""
    try:
        from AppKit import NSApplication, NSImage
        icon_path = Path(__file__).parent / "assets" / "icon.png"
        if icon_path.exists():
            app = NSApplication.sharedApplication()
            image = NSImage.alloc().initWithContentsOfFile_(str(icon_path))
            app.setApplicationIconImage_(image)
    except Exception:
        pass


def _set_macos_window_constraints(title: str, width: int, min_height: int, max_height: int):
    """Constrain the native NSWindow to a fixed width while allowing vertical resize."""
    try:
        from AppKit import NSApplication
        app = NSApplication.sharedApplication()
        for window in app.windows():
            if window.title() == title:
                window.setMinSize_((width, min_height))
                window.setMaxSize_((width, max_height))
                break
    except Exception:
        pass


def main():
    """Main PyWebView application."""
    import setproctitle
    setproctitle.setproctitle("mu-dng-converter")

    # Set macOS Dock icon immediately, before any window is created
    if sys.platform == "darwin":
        _set_macos_icon()

    # Get the absolute path to the web directory
    web_dir = Path(__file__).parent / "web"
    index_file = web_dir / "index.html"
    
    # Create the main application window
    window = webview.create_window(
        title="mu DNG Converter",
        url=str(index_file),
        width=760,
        height=780,
        resizable=True,
        min_size=(760, 500),
        text_select=True,  # allow selecting/copying text (scoped to the log via CSS)
        js_api=expose_to_window  # This will expose the bridge methods
    )

    # Apply native macOS window constraints after PyWebView initialises
    def _on_loaded():
        if sys.platform == "darwin":
            _set_macos_window_constraints("mu DNG Converter", 760, 500, 2000)

    window.events.loaded += _on_loaded

    # Expose the bridge methods to JavaScript
    expose_to_window(window)

    # Start the webview
    webview.start()

if __name__ == "__main__":
    main()
