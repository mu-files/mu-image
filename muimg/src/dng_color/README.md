# DNG Color Processing Extension

Standalone Python C++ extension implementing DNG SDK color processing algorithms.
No external SDK dependencies — all algorithms are reimplemented in `dng_color_standalone.cpp`.

## Functions

### Color Temperature
- `temp_to_xy(temperature, tint)`: Convert color temperature/tint to xy chromaticity
- `xy_to_temp(x, y)`: Convert xy chromaticity to color temperature/tint

### Matrix Operations
- `interpolate_matrices(m1, m2, t1, t2, target_t)`: Dual-illuminant interpolation (1/T weighting)
- `matrix_transform(rgb, matrix, clip=True)`: Apply 3x3 color matrix to RGB image
- `bradford_adapt(src_x, src_y, dst_x, dst_y)`: Compute Bradford chromatic adaptation matrix

### Tone Curves
- `get_acr3_curve(num_points)`: Get ACR3 default tone curve as LUT
- `apply_tone_curve(rgb, curve)`: Apply custom tone curve (ProfileToneCurve)
- `apply_rgb_tone(rgb, curve)`: Hue-preserving RGB tone mapping (RefBaselineRGBTone)
- `srgb_gamma(rgb)`: Apply sRGB gamma encoding

### Color Adjustments
- `apply_hue_sat_map(rgb, map, hue_divs, sat_divs, val_divs)`: Apply HueSatMap 3D LUT

### RAW Processing (Stage 1)
- `linearize(data, table, max_val)`: Apply linearization table
- `normalize_raw(data, black_level, white_level)`: Normalize using black/white levels
- `apply_gain_map(data, gain_map)`: Flat-field correction

### Lens Corrections (Stage 2)
- `warp_rectilinear(rgb, radial_params, ...)`: Lens distortion correction
- `fix_vignette(rgb, params, center_x, center_y)`: Radial vignette correction

### High-Level Pipeline
- `process_rgb(rgb, color_matrix, white_xy, ...)`: Full color pipeline

## Build

Built via setuptools with C++17. No external dependencies beyond NumPy.

## Usage

```python
import muimg._dng_color as dng

# Convert temperature to white point
x, y = dng.temp_to_xy(5500, 0)

# Get ACR3 tone curve
curve = dng.get_acr3_curve(256)

# Apply sRGB gamma
result = dng.srgb_gamma(linear_rgb)

# Matrix transform
xyz = dng.matrix_transform(camera_rgb, camera_to_xyz_matrix)
```
