# Released under a modified PolyForm Small Business License.
# Free for small businesses, individuals, and academics. See LICENSE for details.

"""Background module loaders for slow-loading dependencies.

A single daemon thread imports deferred modules sequentially, ordered by
how soon they are likely to be needed.  Attribute access on a proxy blocks
only until *that* module's import is done (per-proxy Event).

Usage:
    from .deps import cv2_proxy as cv2
    # cv2.resize(...) blocks only if background import hasn't finished yet
"""

import importlib
import threading
import types


class LibraryBackgroundLoader(types.ModuleType):
    """Module proxy that resolves to the real module on first attribute access.

    Modules marked ``immediate=True`` are imported synchronously on the
    calling thread.  All others are left for the single background thread
    (started at the bottom of this file) to import in priority order.
    """

    def __init__(self, name: str, immediate: bool = False):
        super().__init__(name)
        object.__setattr__(self, "_bg_name", name)
        object.__setattr__(self, "_bg_module", None)
        object.__setattr__(self, "_bg_event", threading.Event())
        object.__setattr__(self, "_bg_exception", None)

        if immediate:
            self._bg_load()

    def _bg_load(self):
        """Import the target module and signal completion."""
        try:
            mod = importlib.import_module(object.__getattribute__(self, "_bg_name"))
            object.__setattr__(self, "_bg_module", mod)
        except Exception as e:
            object.__setattr__(self, "_bg_exception", e)
        finally:
            object.__getattribute__(self, "_bg_event").set()

    def _wait_for_load(self):
        mod = object.__getattribute__(self, "_bg_module")
        if mod is not None:
            return mod

        object.__getattribute__(self, "_bg_event").wait()

        exc = object.__getattribute__(self, "_bg_exception")
        if exc is not None:
            raise exc

        return object.__getattribute__(self, "_bg_module")

    def __getattr__(self, item):
        mod = self._wait_for_load()
        try:
            return getattr(mod, item)
        except AttributeError:
            # Try importing as a submodule (e.g. defusedxml.ElementTree)
            name = object.__getattribute__(self, "_bg_name")
            return importlib.import_module(f"{name}.{item}")

    def __dir__(self):
        mod = self._wait_for_load()
        return dir(mod)

    def __repr__(self):
        mod = object.__getattribute__(self, "_bg_module")
        if mod is not None:
            return repr(mod)
        name = object.__getattribute__(self, "_bg_name")
        return f"<LibraryBackgroundLoader proxy for {name}>"


# ---------------------------------------------------------------------------
# Proxy instances
# ---------------------------------------------------------------------------
# immediate=True  → imported right now on the main thread (needed at module-load)
# immediate=False → deferred to the single background thread below

# immediate=True — loaded on main thread (needed at module-load for class inheritance).
tifffile_proxy = LibraryBackgroundLoader("tifffile", immediate=True)

imagecodecs_proxy = LibraryBackgroundLoader("imagecodecs")
defusedxml_proxy = LibraryBackgroundLoader("defusedxml")
cv2_proxy = LibraryBackgroundLoader("cv2")

# All proxies in load-priority order. The background thread skips any
# that were already loaded with immediate=True.
_ALL_PROXIES = (tifffile_proxy, imagecodecs_proxy, defusedxml_proxy, cv2_proxy)

def _background_load_all():
    for proxy in _ALL_PROXIES:
        if not object.__getattribute__(proxy, "_bg_event").is_set():
            proxy._bg_load()

threading.Thread(
    target=_background_load_all, daemon=True, name="BackgroundModuleLoader"
).start()
