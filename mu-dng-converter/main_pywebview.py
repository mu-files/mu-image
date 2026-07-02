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


def main():
    """Main PyWebView application."""
    if sys.platform == "darwin":
        _set_macos_icon()

    # Get the absolute path to the web directory
    web_dir = Path(__file__).parent / "web"
    index_file = web_dir / "index.html"
    
    # Create a simple window with the same dimensions as Flet
    window = webview.create_window(
        title="mu DNG Converter",
        url=str(index_file),
        width=760,
        height=780,
        resizable=True,
        min_size=(600, 500),
        js_api=expose_to_window  # This will expose the bridge methods
    )
    
    # Expose the bridge methods to JavaScript
    expose_to_window(window)
    
    # Start the webview
    webview.start()

if __name__ == "__main__":
    main()
