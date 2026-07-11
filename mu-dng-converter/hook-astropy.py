# Override the default astropy hook to avoid collecting visualization modules
# that require matplotlib. We only need astropy.io.fits and astropy.coordinates.

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# Collect submodules we need (excluding visualization to avoid matplotlib)
hiddenimports = collect_submodules('astropy', filter=lambda name: 'visualization' not in name)

# Explicitly exclude visualization modules that require matplotlib
excludedimports = [
    'astropy.visualization',
    'matplotlib',
]

# Include astropy data files AND Python files (needed for parser tables like generic_parsetab.py)
datas = collect_data_files('astropy', include_py_files=True)
