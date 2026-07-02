"""Bridge module for PyWebView JavaScript-Python communication."""

import webview
import asyncio
from pathlib import Path
import json
from typing import Dict, Any, Optional

class WebViewBridge:
    """Bridge class to handle JavaScript-Python communication."""
    
    def __init__(self):
        self.window = None
        self.runners = {}  # Store runner instances for each tab
        
    def set_window(self, window):
        """Set the webview window instance."""
        self.window = window
        
    async def select_input(self, tab: str) -> str:
        """Handle input folder/file selection."""
        try:
            # Use webview's file dialog
            result = await self.window.create_file_dialog(
                webview.OPEN_DIALOG,
                allow_multiple=False,
                file_types=('*.dng', '*.fits', '*.fit') if tab == 'fits-dng' else ('*.dng',)
            )
            
            if result:
                # Return the selected path
                return result[0] if isinstance(result, list) else result
            return ""
        except Exception as e:
            print(f"Error selecting input for {tab}: {e}")
            return ""
    
    async def select_output(self, tab: str) -> str:
        """Handle output folder selection."""
        try:
            # Use webview's folder dialog
            result = await self.window.create_file_dialog(webview.FOLDER_DIALOG)
            
            if result:
                # Return the selected folder path
                return result[0] if isinstance(result, list) else result
            return ""
        except Exception as e:
            print(f"Error selecting output for {tab}: {e}")
            return ""
    
    async def handle_run(self, tab: str):
        """Handle run button click for a specific tab."""
        try:
            # Update progress in JavaScript
            self.window.evaluate_js(f"updateProgress('{tab}', 'Initializing...')")
            
            # Get the appropriate runner based on tab
            if tab == 'dng-image':
                from .views.dng_image_view import get_runner
                runner = get_runner()
            elif tab == 'fits-dng':
                from .views.fits_dng_view import get_runner
                runner = get_runner()
            elif tab == 'dng-dng':
                from .views.dng_dng_view import get_runner
                runner = get_runner()
            else:
                raise ValueError(f"Unknown tab: {tab}")
            
            self.runners[tab] = runner
            
            # Set up progress callbacks
            def progress_callback(message: str):
                self.window.evaluate_js(f"updateProgress('{tab}', '{message}')")
            
            def progress_bar_callback(value: float):
                self.window.evaluate_js(f"updateProgressBar('{tab}', {value})")
            
            def log_callback(message: str):
                # Escape message for JavaScript
                escaped = json.dumps(message)
                self.window.evaluate_js(f"appendLog('{tab}', {escaped})")
            
            # Run the conversion
            await runner.run_async(
                progress_callback=progress_callback,
                progress_bar_callback=progress_bar_callback,
                log_callback=log_callback
            )
            
            # Update completion status
            self.window.evaluate_js(f"updateProgress('{tab}', 'Complete')")
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.window.evaluate_js(f"updateProgress('{tab}', '{error_msg}')")
            print(f"Error running {tab}: {e}")
    
    def handle_cancel(self, tab: str):
        """Handle cancel button click for a specific tab."""
        try:
            if tab in self.runners:
                self.runners[tab].cancel()
                self.window.evaluate_js(f"updateProgress('{tab}', 'Cancelling...')")
        except Exception as e:
            print(f"Error cancelling {tab}: {e}")
    
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
    window.expose(bridge.handle_run)
    window.expose(bridge.handle_cancel)
    window.expose(bridge.get_settings)
    window.expose(bridge.set_settings)
