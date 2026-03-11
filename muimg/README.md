# muimg

This package is part of the mu-image project and provides utilities for working with raw image data, including DNG file generation.

## Installation

```bash
# Install in development mode
pip install -e .
```

## Optional: RCD Demosaicing (GPL-Licensed)

The RCD (Ratio Corrected Demosaicing) algorithm provides the best balance of quality and speed for Bayer demosaicing. However, RCD is licensed under **GPL v3**, which is incompatible with muimg's PolyForm Noncommercial license.

RCD is disabled by default. To enable it:

1. Rename `src/demosaic/rcd.txt` to `src/demosaic/rcd.c`
2. Rebuild: `pip install -e .`

By enabling RCD, you accept the GPL v3 license terms for that component. The RCD source is based on [Luis Sanz Rodríguez's implementation](https://github.com/LuisSR/RCD-Demosaicing).

**Alternative demosaicing algorithms** (always available):
- `DNGSDK_BILINEAR` - Good quality, fast
- `VNG` - High quality, slower  
- `OPENCV_EA` - Fastest, lower quality

## Usage

```python
from muimg import dng

# Write a DNG file
dng.write_dng(
    raw_data=your_raw_data,
    destination_file=Path("output.dng"),
    bits_per_pixel=16,
    camera_make="Your Camera",
    camera_model="Model XYZ",
    cfa_pattern='RGGB',
    jxl_distance=0.175,
    generate_thumbnail=True
)
```
