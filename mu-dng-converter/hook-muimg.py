# PyInstaller hook for muimg
# muimg uses lazy loading proxies for heavy dependencies, so we need to explicitly include them

# Just add the lazy-loaded dependencies that PyInstaller can't auto-detect
# imagecodecs and astropy have their own custom hooks
hiddenimports = [
    'cv2',
    'numpy',
    'tifffile',
    'defusedxml',
    'click',
    'setproctitle',
]
