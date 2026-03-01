# DNG RAW Development Pipeline

This document describes the rendering pipeline used by the Adobe DNG SDK (`dng_render.cpp`) to convert RAW CFA data to a final display-referred RGB image. It maps how each DNG tag is interpreted and when it is applied during processing.

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            RAW IMAGE DATA                                   │
│                                                                             │
│  CFA: Single-channel Bayer mosaic                                           │
│    Tags: CFAPattern, CFARepeatPatternDim, CFAPlaneColor                     │
│                                                                             │
│  LinearRaw: 3-channel linear RGB (pre-demosaiced)                           │
│    Tags: SamplesPerPixel (=3)                                               │
│                                                                             │
│  Common: PhotometricInterpretation, Compression, PlanarConfiguration        │
│          ColumnInterleaveFactor, RowInterleaveFactor, JXLDistance, JXLEffort│
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 1: LINEARIZATION & BLACK SUBTRACTION               │
│  Tags: BlackLevel, WhiteLevel, ActiveArea                                   │
│  • Subtract BlackLevel from each pixel                                      │
│  • Scale to [0, 1] range using WhiteLevel                                   │
│  • Apply zero-offset ramp function: (x - blackLevel) / (1 - blackLevel)     │
│  (LinearRaw: typically BlackLevel=0, making this an identity operation)     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         STAGE 2: DEMOSAICING                                │
│  (Converts single-channel CFA to 3-channel RGB)                             │
│  Tags: CFAPattern determines interpolation algorithm                        │
│                                                                             │
│  PhotometricInterpretation determines path:                                 │
│  • CFA: Demosaic using CFAPattern → Camera RGB                              │
│  • LinearRaw: Already 3-channel linear RGB, skip demosaic                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│               STAGE 3: CAMERA RGB → LINEAR ProPhoto RGB                     │
│                                                                             │
│  Combined Matrix = AnalogBalance × CameraCalibration × ColorMatrix          │
│                                                                             │
│  Tags Applied:                                                              │
│  • AnalogBalance - Per-channel scaling before color conversion              │
│  • CameraCalibration1/2 - Per-image calibration adjustment                  │
│  • ColorMatrix1/2/3 - Maps camera RGB to XYZ (based on illuminant)          │
│  • CalibrationIlluminant1/2/3 - Identifies light source for each matrix    │
│  • AsShotNeutral / AsShotWhiteXY - White balance reference                  │
│                                                                             │
│  Process:                                                                   │
│  1. Determine white point from AsShotNeutral or AsShotWhiteXY               │
│  2. Interpolate ColorMatrix based on white point temperature                │
│  3. Apply: CameraToRGB = ProPhotoFromPCS × CameraToPCS                      │
│  4. White balance by dividing by camera white vector                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 4: HUE/SAT MAP (Profile LUT)                       │
│  Tags: HueSatMap (from camera profile)                                      │
│  • 3D LUT that adjusts hue, saturation, and value                           │
│  • Interpolated based on white point                                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│               STAGE 5: PROFILE GAIN TABLE MAP                               │
│  Tags: ProfileGainTableMap (from profile or negative)                       │
│  • Spatially-varying exposure adjustment                                    │
│  • Applied with BaselineExposure weighting                                  │
│  • Supports HDR/overrange values                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 6: EXPOSURE ADJUSTMENT                             │
│  Tags: BaselineExposure                                                     │
│                                                                             │
│  exposure = UserExposure + BaselineExposure - log2(Stage3Gain)              │
│  white = 1.0 / pow(2, max(0, exposure))                                     │
│  black = Shadows × ShadowScale × Stage3Gain × 0.001                         │
│                                                                             │
│  • Applies exposure ramp function                                           │
│  • Soft clipping near black point with quadratic toe                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 7: LOOK TABLE                                      │
│  Tags: LookTable (from camera profile)                                      │
│  • Creative 3D LUT for color grading                                        │
│  • Applied after exposure, before tone curve                                │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 8: TONE CURVE                                      │
│  Tags: ProfileToneCurve (from profile, optional)                            │
│                                                                             │
│  • Default: ACR3 baseline tone curve (S-curve)                              │
│  • Or custom ProfileToneCurve from camera profile                           │
│  • Applied per-channel but ratio-preserving (RGBTone)                       │
│  • Negative exposure compensation baked into curve                          │
│  • HDR: slope extension beyond 1.0 for overrange                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 9: RGB TABLES                                      │
│  Tags: RGBTables (masked 3D LUTs with semantic masks)                       │
│  • Multiple 3D LUTs with associated masks                                   │
│  • Weighted sum or sequential application                                   │
│  • Background table for unmasked regions                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│               STAGE 10: OUTPUT COLOR SPACE CONVERSION                       │
│                                                                             │
│  RGBtoFinal = FinalSpaceFromPCS × ProPhotoToPCS                             │
│                                                                             │
│  • Convert from linear ProPhoto RGB to final color space                    │
│  • Common targets: sRGB, Adobe RGB, Display P3, Rec.2020                    │
│  • HDR profiles get linear gamma variant of color space                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 11: GAMMA ENCODING                                 │
│                                                                             │
│  • Apply output space gamma function                                        │
│  • sRGB: ~2.2 gamma with linear toe                                         │
│  • HDR: Linear (gamma 1.0)                                                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FINAL DISPLAY IMAGE                                 │
│                    (8-bit or 16-bit RGB per channel)                        │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Tag Reference by Pipeline Stage

### Input Tags (CFA SubIFD)

| Tag | Purpose | Pipeline Stage |
|-----|---------|----------------|
| `PhotometricInterpretation` | CFA or LinearRaw | Input/demosaic routing |
| `Compression` | Decode algorithm (1=none, 7=JPEG, 52546=JXL) | Decompression |
| `SamplesPerPixel` | 1 for CFA, 3 for LinearRaw | Input validation |
| `PlanarConfiguration` | Data layout | Reading |
| `CFARepeatPatternDim` | CFA pattern size (usually 2×2) | Demosaicing |
| `CFAPattern` | Bayer pattern (RGGB, BGGR, etc.) | Demosaicing |
| `CFAPlaneColor` | Color assignment for each CFA plane | Demosaicing |
| `BitsPerSample` | Bit depth (8, 12, 14, 16) | WhiteLevel default |
| `BlackLevel` | Sensor black level pattern | Stage 1 |
| `BlackLevelRepeatDim` | Black level pattern size (rows, cols) | Stage 1 |
| `BlackLevelDeltaH` | Per-column black level offset | Stage 1 |
| `BlackLevelDeltaV` | Per-row black level offset | Stage 1 |
| `WhiteLevel` | Sensor saturation level (default: 2^BitsPerSample - 1) | Stage 1 |
| `ColumnInterleaveFactor` | Column interleaving | De-interleaving |
| `RowInterleaveFactor` | Row interleaving | De-interleaving |
| `JXLDistance` | JPEG XL quality parameter | Decompression |
| `JXLEffort` | JPEG XL encode effort | Decompression |
| `ActiveArea` | Valid pixel region | Cropping |
| `DefaultCropOrigin` | Crop region start | Output cropping |
| `DefaultCropSize` | Crop region dimensions | Output cropping |

### Global Tags (IFD0)

| Tag | Purpose | Pipeline Stage |
|-----|---------|----------------|
| `Make` | Camera manufacturer | Metadata |
| `Model` | Camera model | Metadata |
| `DNGVersion` | DNG specification version | Validation |
| `DNGBackwardVersion` | Minimum reader version | Validation |
| `Orientation` | Image orientation | Final output |
| `ColorMatrix1` | XYZ → Camera RGB for illuminant 1 (inverted for rendering) | Stage 3 |
| `ColorMatrix2` | XYZ → Camera RGB for illuminant 2 (inverted for rendering) | Stage 3 |
| `ColorMatrix3` | XYZ → Camera RGB for illuminant 3 (inverted for rendering) | Stage 3 |
| `CalibrationIlluminant1` | Light source for ColorMatrix1 | Stage 3 |
| `CalibrationIlluminant2` | Light source for ColorMatrix2 | Stage 3 |
| `CalibrationIlluminant3` | Light source for ColorMatrix3 | Stage 3 |
| `AnalogBalance` | Per-channel analog gain | Stage 3 |
| `AsShotNeutral` | White balance as camera neutral | Stage 3 |
| `AsShotWhiteXY` | White balance as xy chromaticity | Stage 3 |
| `BaselineExposure` | Exposure compensation (EV) | Stage 6 |

### Profile-Embedded Tags

| Tag | Purpose | Notes |
|-----|---------|-------|
| `ProfileGainTableMap` | Spatially-varying gain | Stage 5 |
| `ProfileToneCurve` | Custom tone curve | Stage 8 |
| `ForwardMatrix1/2` | Camera RGB → XYZ | Direct (avoids inverting ColorMatrix) |
| `CameraCalibration1/2` | Per-camera fine-tuning | Stage 3 |
| `OpcodeList1` | Pre-demosaic opcodes | Before Stage 1 (lens corrections, hot pixel removal) |
| `OpcodeList2` | Post-demosaic opcodes | After Stage 2 (gain maps, vignette correction) |
| `OpcodeList3` | Post-color opcodes | After Stage 3 (noise reduction, sharpening) |

## Color Matrix Interpolation

The DNG SDK interpolates between calibration matrices based on the scene white point temperature:

```
If ColorMatrix2 exists AND illuminant temperatures differ:
    
    Temperature → fraction g:
    - If T ≤ T1: g = 1.0 (use ColorMatrix1)
    - If T ≥ T2: g = 0.0 (use ColorMatrix2)  
    - Otherwise: g = (1/T - 1/T2) / (1/T1 - 1/T2)
    
    Final ColorMatrix = g × ColorMatrix1 + (1-g) × ColorMatrix2
```

For triple-illuminant profiles (ColorMatrix3), a more complex triangular interpolation is used.

## White Balance Application

The white balance is determined by (in priority order):
1. User-specified WhiteXY
2. `AsShotNeutral` → converted to XY via `NeutralToXY()`
3. `AsShotWhiteXY` → used directly
4. Default to D55 (5500K daylight)

The camera white vector scales each channel to neutralize the white point:
```cpp
fCameraWhite = spec->CameraWhite();  // Per-channel multipliers
```

## Tone Curve Details

The default ACR3 tone curve is a lookup table with 421 entries that:
- Lifts shadows (input 0.0 → output 0.0)
- Applies an S-curve for contrast
- Compresses highlights (input 1.0 → output 1.0)

For HDR profiles, the tone curve extends beyond 1.0 using slope continuation.

## Processing Order Summary

1. **Black/White Level** → Linearize raw data
2. **Demosaic** → Interpolate to full RGB
3. **AnalogBalance × CameraCalibration × ColorMatrix** → Camera → XYZ → ProPhoto
4. **HueSatMap** → Profile color adjustments  
5. **ProfileGainTableMap** → Local exposure adjustments
6. **Exposure Ramp** → Apply BaselineExposure
7. **LookTable** → Creative color grading
8. **Tone Curve** → Scene → Display mapping
9. **RGBTables** → Masked 3D LUTs
10. **Color Space** → ProPhoto → Output space
11. **Gamma** → Linear → Encoded

## muimg Implementation Status

Current implementation in `muimg.color.process_raw()`:

| Stage | SDK Function | muimg Status | Notes |
|-------|--------------|--------------|-------|
| 1. Black/White Level | `DoBaseline1DTable` | ✓ Implemented | |
| 2. Demosaic | — | ✓ Implemented | CFA + LinearRaw paths |
| 3. Camera → ProPhoto | `DoBaselineABCtoRGB` | ✓ Implemented | Dual-illuminant, ForwardMatrix, CameraCalibration |
| 4. HueSatMap | `DoBaselineHueSatMap` | ✓ Implemented | ProfileHueSatMap with dual-illuminant interpolation |
| 5. ProfileGainTableMap | `DoBaselineProfileGainTableMap` | ✗ Not implemented | |
| 6. Exposure Ramp | `DoBaseline1DFunction` | ✓ Implemented | Uses `BaselineExposure`, `ShadowScale` |
| 7. LookTable | `DoBaselineHueSatMap` | ✓ Implemented | ProfileLookTable |
| 8. Tone Curve | `DoBaselineRGBTone` | ✓ Implemented | ACR3 default only; no `ProfileToneCurve` |
| 9. RGBTables | `ProcessRGBTables` | ✗ Not implemented | |
| 10. Color Space | `DoBaselineRGBtoRGB` | ✓ Implemented | ProPhoto → sRGB only |
| 11. Gamma | `DoBaseline1DTable` | ✓ Implemented | sRGB gamma |

### Stage 3 Detail

| Tag | muimg Status |
|-----|--------------|
| `ColorMatrix1` | ✓ Implemented |
| `ColorMatrix2` | ✓ Implemented (dual-illuminant interpolation) |
| `ColorMatrix3` | ✗ Not implemented (triple illuminant) |
| `CalibrationIlluminant1/2` | ✓ Implemented |
| `CalibrationIlluminant3` | ✗ Not implemented |
| `AnalogBalance` | ✓ Implemented |
| `CameraCalibration1/2` | ✓ Implemented |
| `ForwardMatrix1/2` | ✓ Implemented (with NormalizeForwardMatrix) |
| `AsShotNeutral` | ✓ Implemented |
| `AsShotWhiteXY` | ✓ Implemented |
| `BaselineExposure` | ✓ Implemented |
| `ShadowScale` | ✓ Implemented |
