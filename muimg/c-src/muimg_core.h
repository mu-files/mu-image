/*
 * muimg_core - Core image processing library
 * 
 * Pure C interface for image processing operations.
 * No Python or NumPy dependencies.
 * 
 * Copyright (c) 2024 mu-files
 * Licensed under a modified PolyForm Small Business License
 */

#ifndef MUIMG_CORE_H
#define MUIMG_CORE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stddef.h>

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
    void* data;          // Pointer to raw buffer
    size_t height;       // Image height in pixels
    size_t width;        // Image width in pixels
    size_t channels;     // Number of channels (1 for CFA, 3 for RGB)
    MuImgDType dtype;    // Data type
    size_t stride;       // Row stride in bytes (0 = packed/contiguous)
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
int muimg_bilinear_demosaic(
    const MuImgBuffer* input,
    MuImgBuffer* output,
    const int cfa_pattern[4]
);

int muimg_convert_dtype(
    const MuImgBuffer* input,
    MuImgBuffer* output,
    int src_bits,
    int dst_bits,
    float clip_max
);

int muimg_mono_lut(
    const MuImgBuffer* input,
    MuImgBuffer* output,
    const float* lut,
    size_t lut_size,
    int src_bits,
    int dst_bits
);

int muimg_transform_color(
    const MuImgBuffer* input,
    MuImgBuffer* output,
    const float* input_lut,
    size_t input_lut_size,
    const float* matrix,
    const float* output_lut,
    size_t output_lut_size,
    int src_bits,
    int dst_bits,
    bool hue_preserving
);

int muimg_clip_and_transform_color(
    const MuImgBuffer* input,
    MuImgBuffer* output,
    const float clip_max[3],
    const float matrix[9]
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
int muimg_vng_demosaic(
    const MuImgBuffer* input,
    MuImgBuffer* output,
    const char* cfa_pattern
);

//=============================================================================
// Version information
//=============================================================================

const char* muimg_version(void);

#ifdef __cplusplus
}
#endif

#endif // MUIMG_CORE_H
