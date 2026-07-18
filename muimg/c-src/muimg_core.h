/*
 * muimg_core - Core image processing library
 *
 * Pure C interface for image processing operations.
 * No Python or NumPy dependencies.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 * Copyright (c) 2026 mu-files
 */

#ifndef MUIMG_CORE_H
#define MUIMG_CORE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stddef.h>
#include <stdint.h>

// Export when building the shared library. Consumers load via dlopen/LoadLibrary
// and should not link against an import library, so non-export builds leave
// MUIMG_API empty (declarations are for types / documentation only).
#if defined(MUIMG_CORE_EXPORTS)
#if defined(_WIN32) || defined(_WIN64)
#define MUIMG_API __declspec(dllexport)
#else
#define MUIMG_API __attribute__((visibility("default")))
#endif
#else
#define MUIMG_API
#endif

// Return codes
#define MUIMG_SUCCESS 0
#define MUIMG_ERROR_INVALID_ARGUMENT 1
#define MUIMG_ERROR_UNSUPPORTED_DTYPE 2
#define MUIMG_ERROR_DIMENSION_MISMATCH 3
#define MUIMG_ERROR_OUT_OF_MEMORY 4

// Data types
typedef enum {
  MUIMG_DTYPE_UINT8 = 0,
  MUIMG_DTYPE_UINT16 = 1,
  MUIMG_DTYPE_FLOAT16 = 2,
  MUIMG_DTYPE_FLOAT32 = 3
} MuImgDType;

// Buffer descriptor for passing image data
typedef struct {
  void *data;       // Pointer to raw buffer
  size_t height;    // Image height in pixels
  size_t width;     // Image width in pixels
  size_t channels;  // Number of channels (1 for CFA, 3 for RGB)
  MuImgDType dtype; // Data type
  size_t stride;    // Row stride in bytes (0 = packed/contiguous)
} MuImgBuffer;

//=============================================================================
// Demosaicing operations
//=============================================================================

// Bilinear demosaic
//
// Converts CFA (Bayer) data to RGB using bilinear interpolation.
// Based on Adobe DNG SDK dng_mosaic_info::InterpolateGeneric
//
// Args:
//   input: CFA data, float32, shape (H, W, 1), values in [0,1]
//   output: RGB data, float32, shape (H, W, 3), values in [0,1]
//   cfa_pattern: 2x2 pattern of color indices, row-major order
//                [row0col0, row0col1, row1col0, row1col1]
//                Values: 0=Red, 1=Green, 2=Blue
//
// Returns: MUIMG_SUCCESS on success, error code otherwise
MUIMG_API int muimg_bilinear_demosaic(const MuImgBuffer *input, MuImgBuffer *output,
                            const int cfa_pattern[4]);

MUIMG_API int muimg_convert_dtype(const MuImgBuffer *input, MuImgBuffer *output,
                        int src_bits, int dst_bits, float clip_max);

MUIMG_API int muimg_mono_lut(const MuImgBuffer *input, MuImgBuffer *output,
                   const float *lut, size_t lut_size, int src_bits,
                   int dst_bits);

MUIMG_API int muimg_transform_color(const MuImgBuffer *input, MuImgBuffer *output,
                          const float *input_lut, size_t input_lut_size,
                          const float *matrix, const float *output_lut,
                          size_t output_lut_size, int src_bits, int dst_bits,
                          bool hue_preserving);

MUIMG_API int muimg_clip_and_transform_color(const MuImgBuffer *input,
                                   MuImgBuffer *output, const float clip_max[3],
                                   const float matrix[9]);

// Normalize RAW data using black/white levels per DNG spec Chapter 5
//
// Implements: linear = (raw - BlackLevel - DeltaH - DeltaV) / (WhiteLevel -
// BlackLevel - ...)
//
// Args:
//   input: RAW pixel data, uint16 or float32, shape (H, W) or (H, W,
//   samples_per_pixel) output: Normalized float32 pixel data, same shape as
//   input black_level: BlackLevel pattern, float32, length = repeat_rows *
//   repeat_cols * samples_per_pixel black_repeat_rows: Number of rows in
//   repeating pattern black_repeat_cols: Number of cols in repeating pattern
//   samples_per_pixel: 1 for CFA, 3 for LinearRaw
//   white_level: WhiteLevel per sample, float32, length = samples_per_pixel
//   white_count: Length of white_level array (must equal samples_per_pixel)
//   black_delta_h: Optional per-column delta, float32, length = width (or NULL)
//   delta_h_count: Length of black_delta_h array (or 0 if NULL)
//   black_delta_v: Optional per-row delta, float32, length = height (or NULL)
//   delta_v_count: Length of black_delta_v array (or 0 if NULL)
//   linearization_table: Optional uint16 LUT (or NULL)
//   linearization_table_size: Length of linearization_table (or 0 if NULL)
//
// Returns: MUIMG_SUCCESS on success, error code otherwise
MUIMG_API int muimg_normalize_raw(const MuImgBuffer *input, MuImgBuffer *output,
                        const float *black_level, int black_repeat_rows,
                        int black_repeat_cols, int samples_per_pixel,
                        const float *white_level, int white_count,
                        const float *black_delta_h, int delta_h_count,
                        const float *black_delta_v, int delta_v_count,
                        const uint16_t *linearization_table,
                        int linearization_table_size);

// Apply HueSatMap (3D LUT) to RGB image for camera profile color adjustments
//
// SDK ref: dng_hue_sat_map.cpp
//
// Args:
//   rgb: RGB image data, float32, shape (H, W, 3) - modified in-place
//   map_data: HueSatMap data, float32, flattened array of (hue_shift,
//   sat_scale, val_scale) triplets hue_divs: Number of hue divisions in the map
//   sat_divs: Number of saturation divisions in the map
//   val_divs: Number of value divisions in the map (0 or 1 for 2.5D table)
//
// Returns: MUIMG_SUCCESS on success, error code otherwise
MUIMG_API int muimg_apply_hue_sat_map(MuImgBuffer *rgb, const float *map_data,
                            int hue_divs, int sat_divs, int val_divs);

// Apply exposure ramp function (dng_function_exposure_ramp)
//
// SDK ref: dng_render.cpp lines 50-103
// 3 regions: below black-radius=0, above black+radius=linear, between=quadratic
//
// Args:
//   input: Input RGB image, float32, shape (H, W, 3)
//   output: Output RGB image, float32, shape (H, W, 3)
//   white: White point (1.0 / pow(2, max(0, exposure)))
//   black: Black point (shadows * shadowScale * 0.001)
//   minBlack: Minimum black for radius calculation
//   supportOverrange: Allow values > 1.0
//
// Returns: MUIMG_SUCCESS on success, error code otherwise
MUIMG_API int muimg_apply_exposure_ramp(const MuImgBuffer *input, MuImgBuffer *output,
                              double white, double black, double minBlack,
                              bool supportOverrange);

// Apply ProfileGainTableMap - spatially-varying per-plane gain adjustment
//
// SDK ref: dng_reference.cpp RefBaselineProfileGainTableMap() lines 3260-3460
// 3D trilinear interpolation: (spatial_v, spatial_h, exposure_weight)
//
// Args:
//   rgb: RGB image data, float32, shape (H, W, 3) - modified in-place
//   gains: Gain table, float32, shape (points_v, points_h, num_table_points)
//   flattened points_v: Number of vertical grid points points_h: Number of
//   horizontal grid points spacing_v: Vertical grid spacing in normalized
//   coords spacing_h: Horizontal grid spacing in normalized coords origin_v:
//   Vertical grid origin in normalized coords origin_h: Horizontal grid origin
//   in normalized coords num_table_points: Number of exposure weight table
//   points weights: MapInputWeights (5 floats): [R, G, B, min, max]
//   coefficients gamma: Gamma applied to exposure weight exposure_weight_gain:
//   pow(2, baseline_exposure)
//
// Returns: MUIMG_SUCCESS on success, error code otherwise
MUIMG_API int muimg_apply_profile_gain_table_map(MuImgBuffer *rgb, const float *gains,
                                       int points_v, int points_h,
                                       float spacing_v, float spacing_h,
                                       float origin_v, float origin_h,
                                       int num_table_points,
                                       const float *weights, float gamma,
                                       float exposure_weight_gain);

// Apply radial vignette correction to RGB or CFA data
MUIMG_API int muimg_fix_vignette(MuImgBuffer *img, const double *params, int num_params,
                       double center_x, double center_y);

// Apply lens distortion correction using WarpRectilinear opcode
// radial_params: flattened array of size num_planes * num_coeffs
// tangential_params: flattened array of size num_planes * 2 (or NULL)
MUIMG_API int muimg_warp_rectilinear(
    const MuImgBuffer* input,
    MuImgBuffer* output,
    const double* radial_params,
    int num_planes,
    int num_coeffs,
    const double* tangential_params,
    double center_x,
    double center_y,
    bool use_bicubic
);

// Apply GainMap opcode to RGB image data (OpcodeList2)
// gain_values: flattened (points_v, points_h, map_planes) float32 array
// top/left/bottom/right: area bounds (0 = full image for bottom/right)
MUIMG_API int muimg_apply_gain_map(
    MuImgBuffer* img,
    const float* gain_values,
    int points_v, int points_h, int map_planes,
    int top, int left, int bottom, int right,
    int start_plane, int num_planes,
    int row_pitch, int col_pitch,
    double spacing_v, double spacing_h,
    double origin_v, double origin_h
);

// Apply GainMap opcode to CFA image data (OpcodeList1)
// gain_values: flattened (points_v, points_h, map_planes) float32 array
MUIMG_API int muimg_apply_gain_map_cfa(
    MuImgBuffer* img,
    const float* gain_values,
    int points_v, int points_h, int map_planes,
    int top, int left, int bottom, int right,
    int row_pitch, int col_pitch,
    double spacing_v, double spacing_h,
    double origin_v, double origin_h
);

// Apply simple 2D flat-field gain map to CFA float32 image (in-place).
// gain_map: row-major (gain_h, gain_w) float32; bilinearly scaled to image size.
MUIMG_API int muimg_apply_flat_gain_map(
    MuImgBuffer* img,
    const float* gain_map,
    int gain_h, int gain_w
);

// Apply MapPolynomial opcode (OpcodeList2) in-place to float32 image.
// Supports 1-channel CFA or 3-channel RGB (from img->channels).
// top/left/bottom/right: area bounds (0,0 for bottom/right means full image).
// coefficients: length degree+1
MUIMG_API int muimg_map_polynomial(
    MuImgBuffer* img,
    int top, int left, int bottom, int right,
    int start_plane, int num_planes,
    int row_pitch, int col_pitch,
    const float* coefficients, int degree
);

// FixBadPixelsConstant opcode (OpcodeList1).
// input/output: uint16 CFA, shape (H, W, 1). Neighbors are read from input;
// replacements are written to output (non-bad pixels should already match input).
MUIMG_API int muimg_fix_bad_pixels_constant(
    const MuImgBuffer* input,
    MuImgBuffer* output,
    uint32_t constant,
    uint32_t bayer_phase
);

// VNG (Variable Number of Gradients) demosaic
//
// High-quality demosaicing algorithm.
//
// Args:
//   input: CFA data, uint16, shape (H, W, 1)
//   output: RGB data, uint16, shape (H, W, 3)
//   cfa_pattern: String pattern like "RGGB", "BGGR", etc.
//
// Returns: MUIMG_SUCCESS on success, error code otherwise
MUIMG_API int muimg_vng_demosaic(const MuImgBuffer *input, MuImgBuffer *output,
                       const char *cfa_pattern);

//=============================================================================
// Version information
//=============================================================================

MUIMG_API const char *muimg_version(void);

#ifdef __cplusplus
}
#endif

#endif // MUIMG_CORE_H
