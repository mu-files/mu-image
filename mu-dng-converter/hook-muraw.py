# PyInstaller hook for muraw

from PyInstaller.utils.hooks import collect_submodules, collect_dynamic_libs

# muraw / muimage plus deps (excluding those with dedicated hooks: astropy, imagecodecs)
DEPS = ['muraw', 'muimage', 'cv2', 'numpy', 'tifffile', 'defusedxml', 'click', 'setproctitle']

hiddenimports = []
binaries = []

for dep in DEPS:
    hiddenimports += collect_submodules(dep)
    binaries += collect_dynamic_libs(dep)
