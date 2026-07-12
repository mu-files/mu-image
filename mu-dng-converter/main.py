"""Entry point for PyWebView build."""

import webview
import os
import sys
import platform
from pathlib import Path
from mu_dng_converter.webview_bridge import expose_to_window

# App version - updated by CI from git tag
__version__ = "0.0.0"

EULA_TEXT = """mu-files LLC — End User License Agreement

By using mu DNG Converter, you agree to the following terms:

1. RISK OF DATA LOSS AND BACKUP REQUIREMENT
This software modifies digital image files. By using this software, you acknowledge and agree that file alteration, damage, or corruption is an inherent risk of data processing. You agree that you are solely responsible for maintaining complete, independent backups of all original data and source files prior to using this software.

2. LIMITATION OF LIABILITY
THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND. IN NO EVENT SHALL MU-FILES LLC OR ITS DEVELOPERS BE LIABLE FOR ANY DAMAGES, LOSS OF DATA, LOSS OF PROFITS, OR COSTS OF PROCUREMENT OF SUBSTITUTE GOODS ARISING OUT OF THE USE OR INABILITY TO USE THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

3. ACCEPTANCE
By clicking "I Agree" you confirm that you have read and accept these terms."""


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


def _check_eula_accepted() -> bool:
    """Check if EULA has been accepted for the current version."""
    eula_file = _get_config_dir() / "eula_accepted_version"
    if eula_file.exists():
        accepted_version = eula_file.read_text().strip()
        return accepted_version == __version__
    return False


def _save_eula_accepted():
    """Save that EULA has been accepted for the current version."""
    eula_file = _get_config_dir() / "eula_accepted_version"
    eula_file.write_text(__version__)


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
        text_select=True,
        js_api=expose_to_window
    )

    # Apply native macOS window constraints after PyWebView initialises
    def _on_loaded():
        if sys.platform == "darwin":
            _set_macos_window_constraints("mu DNG Converter", 760, 500, 2000)
        if not _check_eula_accepted():
            window.evaluate_js("showEulaModal('acceptance')")

    window.events.loaded += _on_loaded
    expose_to_window(window)

    # Start the webview
    webview.start()

if __name__ == "__main__":
    main()
