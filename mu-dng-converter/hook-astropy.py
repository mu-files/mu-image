# Override the default astropy hook to avoid collecting visualization modules
# that require matplotlib. We only need astropy.io.fits and astropy.coordinates.

from PyInstaller.utils.hooks import collect_data_files

# Don't collect all submodules - only specify what we actually need
hiddenimports = [
    'astropy.io.fits',
    'astropy.io.fits.hdu',
    'astropy.io.fits.header',
    'astropy.coordinates',
    'astropy.units',
]

# Explicitly exclude visualization modules that require matplotlib
excludedimports = [
    'astropy.visualization',
    'matplotlib',
]

# Include astropy data files (CITATION, etc.)
datas = collect_data_files('astropy', include_py_files=False)
