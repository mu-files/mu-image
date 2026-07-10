# PyInstaller hook for imagecodecs to include all binary extensions

from PyInstaller.utils.hooks import collect_dynamic_libs, collect_data_files

# Collect all dynamic libraries (compiled extensions)
binaries = collect_dynamic_libs('imagecodecs')

# Collect data files if any
datas = collect_data_files('imagecodecs', include_py_files=False)
