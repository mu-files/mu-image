# PyInstaller hook for muimg

from PyInstaller.utils.hooks import collect_submodules, collect_dynamic_libs

# muimg dependencies (excluding those with dedicated hooks: astropy, imagecodecs)
DEPS = ['cv2', 'numpy', 'tifffile', 'defusedxml', 'click', 'setproctitle']

hiddenimports = []
binaries = []

for dep in DEPS:
    hiddenimports += collect_submodules(dep)
    binaries += collect_dynamic_libs(dep)
