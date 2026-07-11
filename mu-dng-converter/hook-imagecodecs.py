# PyInstaller hook for imagecodecs to include all binary extensions

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files, collect_submodules

# Collect all submodules (including _jpeg8, _jpeg12, etc.)
hiddenimports = collect_submodules('imagecodecs')

# Collect all dynamic libraries (compiled extensions)
binaries = collect_dynamic_libs('imagecodecs')

# Collect data files if any
datas = collect_data_files('imagecodecs', include_py_files=False)
