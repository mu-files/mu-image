# DNG RAW Development Pipeline

This document describes the rendering pipeline implemented in `muimg.dngio.DngPage.render`,
which is a port of the Adobe DNG SDK (`dng_render.cpp`). It converts RAW CFA
data to a final display-referred RGB image, mapping how each DNG tag is
interpreted and applied during processing.

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
│                         OPCODELIST1 (Pre-linearization)                     │
│  Tags: OpcodeList1                                                          │
│  • Applied to raw sensor data before any processing                         │
│  • Hot pixel removal, defect correction                                     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 1: LINEARIZATION & BLACK SUBTRACTION               │
│  Tags: LinearizationTable, BlackLevel, BlackLevelRepeatDim,                 │
│        BlackLevelDeltaH, BlackLevelDeltaV, WhiteLevel                       │
│  • Apply LinearizationTable LUT (if present) - maps ADC values to linear    │
│  • Compute total black: BlackLevel[r%rR][c%rC][s] + DeltaH[c] + DeltaV[r]   │
│  • Normalize: (pixel - totalBlack) / (WhiteLevel[s] - totalBlack)           │
│  • Clamp result to [0, 1]                                                   │
│  (LinearRaw: typically BlackLevel=0, making this an identity operation)     │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                  OPCODELIST2 (Post-linearization, Pre-demosaic)             │
│  Tags: OpcodeList2                                                          │
│  DNG Spec: "applied to the raw image, just after it has been mapped to      │
│             linear reference values"                                        │
│  • CFA: GainMap, MapPolynomial                                              │
│  • LinearRaw: GainMap, MapPolynomial, FixVignetteRadial, WarpRectilinear    │
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
│                    OPCODELIST3 (Post-demosaic)                              │
│  Tags: OpcodeList3                                                          │
│  DNG Spec: "applied to the raw image, just after it has been demosaiced"    │
│  • Supported: WarpRectilinear, FixVignetteRadial, MapPolynomial, GainMap    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DEFAULT CROP                                        │
│  Tags: DefaultCropOrigin, DefaultCropSize                                   │
│  • Trim image to final crop area                                            │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│               STAGE 3: CAMERA RGB → LINEAR ProPhoto RGB                     │
│                                                                             │
│  Tags: ColorMatrix1/2, ForwardMatrix1/2, CameraCalibration1/2,              │
│        CalibrationIlluminant1/2, AnalogBalance, AsShotNeutral/AsShotWhiteXY │
│                                                                             │
│  Process:                                                                   │
│  1. Apply: ColorMatrix = AnalogBalance × CameraCalibration × ColorMatrix    │
│  2. Interpolate matrices based on scene temperature (dual-illuminant)       │
│  3. Determine white point from AsShotNeutral or AsShotWhiteXY               │
│  4. If ForwardMatrix present: CameraToPCS = FM × inv(cameraWhite) × inv(AB) │
│     Else: CameraToPCS = inv(ColorMatrix × BradfordAdapt)                    │
│  5. CameraToProPhoto = ProPhotoFromPCS × CameraToPCS                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 4: HUE/SAT MAP (Profile LUT)                       │
│  Tags: ProfileHueSatMapDims, ProfileHueSatMapData1/2                        │
│  • 3D LUT that adjusts hue, saturation, and value                           │
│  • Dual-illuminant: interpolate Data1/Data2 based on scene temperature      │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│               STAGE 5: PROFILE GAIN TABLE MAP                               │
│  Tags: ProfileGainTableMap, ProfileGainTableMap2 (v2 takes precedence)      │
│        BaselineExposure, BaselineExposureOffset                             │
│  • 3D gain table: (row, col, brightness) - spatially & tonally varying      │
│  • Brightness index scaled by pow(2, BaselineExposure + Offset)             │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 6: EXPOSURE ADJUSTMENT                             │
│  Tags: BaselineExposure, BaselineExposureOffset, DefaultBlackRender,        │
│        ShadowScale                                                          │
│                                                                             │
│  exposure = BaselineExposure + BaselineExposureOffset                       │
│  white = 1.0 / pow(2, max(0, exposure))                                     │
│  black = Shadows × ShadowScale × 0.001, clamped to < 0.99 × white           │
│   (Shadows = 5.0 if DefaultBlackRender=Auto, 0.0 if None)                   │
│                                                                             │
│  • Applies exposure ramp function                                           │
│  • Soft clipping near black point with quadratic toe                        │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 7: LOOK TABLE                                      │
│  Tags: ProfileLookTableDims, ProfileLookTableData                           │
│  • Creative 3D LUT for color grading (same format as HueSatMap)             │
│  • Applied after exposure ramp, before tone curve                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 8: TONE CURVE                                      │
│  Tags: ProfileToneCurve (from profile, optional)                            │
│                                                                             │
│  • Default: ACR3 baseline tone curve (S-curve)                              │
│  • Or custom ProfileToneCurve from camera profile                           │
│  • Hue-preserving: curve applied to max/min, middle interpolated            │
│  • Negative exposure: apply_exposure_tone() darkening before curve          │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 9: RGB TABLES (NOT IMPLEMENTED)                    │
│  Tags: RGBTables (DNG 1.6+)                                                 │
│  • Listed in UNSUPPORTED_RENDERING_TAGS - skipped                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│               STAGE 10: OUTPUT COLOR SPACE CONVERSION                       │
│                                                                             │
│  prophoto_to_srgb = XYZ_D65_TO_SRGB × BradfordAdapt(D50→D65) × ProPhotoToXYZ│
│                                                                             │
│  • Convert from linear ProPhoto RGB (D50) to linear sRGB (D65)              │
│  • Only sRGB output currently implemented                                   │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                    STAGE 11: GAMMA ENCODING                                 │
│                                                                             │
│  sRGB transfer: x <= 0.0031308 ? 12.92*x : 1.055*x^(1/2.4) - 0.055          │
│                                                                             │
│  • Clips to [0, 1] and applies sRGB gamma                                   │
│  • Only sRGB gamma currently implemented                                    │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FINAL OUTPUT IMAGE                                  │
│  Output dtype options: uint8, uint16, float16, float32                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Tag Reference by Pipeline Stage

### Input Tags (CFA SubIFD)

| Tag | Purpose | Pipeline Stage |
|-----|---------|----------------|
| `PhotometricInterpretation` | CFA or LinearRaw | Input/demosaic |
| `Compression` | Decode algorithm (1=none, 7=JPEG, 52546=JXL) | Decompress |
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
| `WhiteLevel` | Sensor saturation level (default: 2^BPS - 1) | Stage 1 |
| `LinearizationTable` | Sensor linearization LUT | Stage 1 |
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
| `ColorMatrix1` | XYZ → Camera RGB for illuminant 1 (inverted) | Stage 3 |
| `ColorMatrix2` | XYZ → Camera RGB for illuminant 2 (inverted) | Stage 3 |
| `ColorMatrix3` | XYZ → Camera RGB for illuminant 3 (inverted) | Stage 3 |
| `CalibrationIlluminant1` | Light source for ColorMatrix1 | Stage 3 |
| `CalibrationIlluminant2` | Light source for ColorMatrix2 | Stage 3 |
| `CalibrationIlluminant3` | Light source for ColorMatrix3 | Stage 3 |
| `AnalogBalance` | Per-channel analog gain | Stage 3 |
| `AsShotNeutral` | White balance as camera neutral | Stage 3 |
| `AsShotWhiteXY` | White balance as xy chromaticity | Stage 3 |
| `BaselineExposure` | Exposure compensation (EV) | Stage 6 |
| `BaselineExposureOffset` | Profile exposure offset (EV) | Stage 6 |
| `DefaultBlackRender` | Shadow mapping mode (0=Auto, 1=None) | Stage 6 |

### Profile-Embedded Tags

| Tag | Purpose | Notes |
|-----|---------|-------|
| `ProfileGainTableMap` | Spatially-varying gain | Stage 5 |
| `ProfileToneCurve` | Custom tone curve | Stage 8 |
| `ForwardMatrix1/2` | Camera RGB → XYZ | Direct (avoids inverting ColorMatrix) |
| `CameraCalibration1/2` | Per-camera fine-tuning | Stage 3 |
| `OpcodeList1` | Pre-linearization opcodes | Before Stage 1 (defect correction) |
| `OpcodeList2` | Post-linearization opcodes | After Stage 1 (GainMap, MapPolynomial) |
| `OpcodeList3` | Post-color opcodes | After Stage 3 (additional corrections) |

## muimg Implementation Status

Current implementation in `muimg.dngio.DngPage.render()`:

| Stage | muimg Status | Notes |
|-------|--------------|-------|
| OpcodeList1 | 🔴 Not implemented | Pre-linearization (defect correction) |
| 1. Linearization | ✓ Implemented | LinearizationTable + black/white |
| OpcodeList2 | ✓ Implemented | GainMap, MapPolynomial, etc. |
| 2. Demosaic | ✓ Implemented | CFA + LinearRaw paths |
| OpcodeList3 | ✓ Implemented | WarpRectilinear, etc. |
| DefaultCrop | ✓ Implemented | After OpcodeList3 |
| 3. Camera → ProPhoto | ✓ Implemented | 🔴 Not implemented: Triple-illuminant |
| 4. HueSatMap | ✓ Implemented | ProfileHueSatMap |
| 5. ProfileGainTableMap | ✓ Implemented | PGTM1 + PGTM2 |
| 6. Exposure Ramp | ✓ Implemented | BaselineExposure, ShadowScale |
| 7. LookTable | ✓ Implemented | ProfileLookTable |
| 8. Tone Curve | ✓ Implemented | ACR3 + ProfileToneCurve |
| 9. RGBTables | 🔴 Not implemented | DNG 1.6+ |
| 10. Color Space | ✓ Implemented | ProPhoto → sRGB only |
| 11. Gamma | ✓ Implemented | sRGB only |
| ReductionMatrix | 🔴 Not implemented | >3 color channels |
| SemanticMasks | 🔴 Not implemented | DNG 1.6+ depth/masks |
| HDR/Overrange | 🔴 Not implemented | ProfileDynamicRange |

