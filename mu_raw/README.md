# mu_raw

This package is part of the mu-image project and provides utilities for working with raw image data, including DNG file generation.

## Installation

```bash
pip install mu-raw
```

## Installation

```bash
# Install in development mode
pip install -e .
```

## Usage

```python
from mu_raw import dng

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
