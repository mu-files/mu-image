<p align="center" style="background-color: white; padding: 20px 0; margin: 0 0 30px 0;">
  <img src="assets/muIcon_muimg_full.svg" alt="muimg logo" width="400">
</p>

# muimg

This package is part of the mu-image project and provides utilities for working with raw image data, including DNG file generation.

## Installation

### From GitHub

```bash
pip install git+https://github.com/mu-files/mu-image.git#subdirectory=muimg
```

### For Development

```bash
git clone https://github.com/mu-files/mu-image.git
cd mu-image/muimg
pip install -e .
```

## Optional: RCD Demosaicing (GPL-Licensed)

The RCD (Ratio Corrected Demosaicing) algorithm provides the best balance of quality and speed for Bayer demosaicing. However, RCD is licensed under **GPL v3**, which is separate from muimg's PolyForm Small Business license.

RCD is disabled by default. To enable it:

1. Rename `c-src/demosaic/rcd.txt` to `c-src/demosaic/rcd.c`
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

## License

This software is released under a modified **PolyForm Small Business License 1.0.0**.

**Free for:**
- Small businesses (<100 employees, <$10M revenue)
- Individuals
- Academic institutions
- Non-profit organizations

**Large enterprises** require a commercial license. Contact: license@mu-files.com

See [LICENSE](LICENSE) for full terms.

### Third-Party Components

- **Adobe DNG SDK**: Adobe DNG SDK License (permissive, royalty-free)
- **VNG Demosaicing**: LGPL v2.1 / CDDL v1.0
- **RCD Demosaicing** (optional): GPL v3

See [LICENSES/README.md](LICENSES/README.md) for details.
