/*
 * DNG SDK Color Processing - Standalone Python wrapper
 * 
 * This module provides Python bindings for DNG SDK pixel-level operations,
 * implemented as standalone code extracted from the SDK:
 * - ProfileHueSatMap (3D HSV LUT)
 * - Tone curve application (hue-preserving and per-channel)
 * - Exposure ramp and tone functions
 * - sRGB gamma encoding
 * - Color matrix transforms
 * - RAW linearization and normalization
 * - Lens distortion correction (WarpRectilinear)
 * - Vignette correction (FixVignette)
 * - GainMap and MapPolynomial opcodes
 * - Bilinear demosaicing
 * - ProfileGainTableMap
 * 
 * Non-pixel-processing code (temperature conversion, tone curve generation,
 * Bradford adaptation, matrix interpolation) is implemented in Python color.py.
 * 
 * Based on Adobe DNG SDK 1.7.1
 * Original Copyright 2006-2024 Adobe Systems Incorporated
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>

//=============================================================================
// RGB <-> HSV Conversion (from dng_utils.h)
// H range: 0-6, S/V range: 0-1
//=============================================================================

static inline void rgb_to_hsv(float r, float g, float b, float& h, float& s, float& v) {
    v = std::max(r, std::max(g, b));
    float gap = v - std::min(r, std::min(g, b));
    
    if (gap > 0.0f) {
        if (r == v) {
            h = (g - b) / gap;
            if (h < 0.0f) h += 6.0f;
        } else if (g == v) {
            h = 2.0f + (b - r) / gap;
        } else {
            h = 4.0f + (r - g) / gap;
        }
        s = gap / v;
    } else {
        h = 0.0f;
        s = 0.0f;
    }
}

static inline void hsv_to_rgb(float h, float s, float v, float& r, float& g, float& b) {
    if (s > 0.0f) {
        h = fmodf(h, 6.0f);
        if (h < 0.0f) h += 6.0f;
        
        int i = (int)h;
        float f = h - (float)i;
        float p = v * (1.0f - s);
        float q = v * (1.0f - s * f);
        float t = v * (1.0f - s * (1.0f - f));
        
        switch (i) {
            case 0: r = v; g = t; b = p; break;
            case 1: r = q; g = v; b = p; break;
            case 2: r = p; g = v; b = t; break;
            case 3: r = p; g = q; b = v; break;
            case 4: r = t; g = p; b = v; break;
            case 5: r = v; g = p; b = q; break;
            case 6: r = v; g = t; b = p; break;  // Edge case
        }
    } else {
        r = v;
        g = v;
        b = v;
    }
}

//=============================================================================
// HueSatMap - 3D LUT for HSV adjustments (from dng_hue_sat_map.cpp)
//=============================================================================

struct HSBModify {
    float hue_shift;   // Hue shift in degrees
    float sat_scale;   // Saturation scale factor
    float val_scale;   // Value scale factor
};

// Apply HueSatMap to a single pixel
// map_data: flattened 3D array of HSBModify [val][hue][sat]
// hue_divs, sat_divs, val_divs: table dimensions
static void apply_hue_sat_map(
    float& r, float& g, float& b,
    const HSBModify* map_data,
    uint32_t hue_divs, uint32_t sat_divs, uint32_t val_divs
) {
    // Convert to HSV
    float h, s, v;
    r = std::max(0.0f, r);
    g = std::max(0.0f, g);
    b = std::max(0.0f, b);
    rgb_to_hsv(r, g, b, h, s, v);
    
    // Scale factors for indexing
    float h_scale = (hue_divs < 2) ? 0.0f : (hue_divs * (1.0f / 6.0f));
    float s_scale = (float)((int32_t)sat_divs - 1);
    float v_scale = (float)((int32_t)val_divs - 1);
    
    int32_t max_hue_idx = (int32_t)hue_divs - 1;
    int32_t max_sat_idx = (int32_t)sat_divs - 2;
    int32_t max_val_idx = (int32_t)val_divs - 2;
    
    int32_t hue_step = sat_divs;
    int32_t val_step = hue_divs * hue_step;
    
    float hue_shift, sat_scale, val_scale;
    
    if (val_divs < 2) {
        // 2.5D table (most common)
        float h_scaled = h * h_scale;
        float s_scaled = s * s_scale;
        
        int32_t h_idx0 = (int32_t)h_scaled;
        int32_t s_idx0 = (int32_t)s_scaled;
        s_idx0 = std::min(s_idx0, max_sat_idx);
        
        int32_t h_idx1 = h_idx0 + 1;
        if (h_idx0 >= max_hue_idx) {
            h_idx0 = max_hue_idx;
            h_idx1 = 0;
        }
        
        float h_fract1 = h_scaled - (float)h_idx0;
        float s_fract1 = s_scaled - (float)s_idx0;
        float h_fract0 = 1.0f - h_fract1;
        float s_fract0 = 1.0f - s_fract1;
        
        const HSBModify* e00 = map_data + h_idx0 * hue_step + s_idx0;
        const HSBModify* e01 = map_data + h_idx1 * hue_step + s_idx0;
        
        float hs0 = h_fract0 * e00->hue_shift + h_fract1 * e01->hue_shift;
        float ss0 = h_fract0 * e00->sat_scale + h_fract1 * e01->sat_scale;
        float vs0 = h_fract0 * e00->val_scale + h_fract1 * e01->val_scale;
        
        e00++; e01++;
        float hs1 = h_fract0 * e00->hue_shift + h_fract1 * e01->hue_shift;
        float ss1 = h_fract0 * e00->sat_scale + h_fract1 * e01->sat_scale;
        float vs1 = h_fract0 * e00->val_scale + h_fract1 * e01->val_scale;
        
        hue_shift = s_fract0 * hs0 + s_fract1 * hs1;
        sat_scale = s_fract0 * ss0 + s_fract1 * ss1;
        val_scale = s_fract0 * vs0 + s_fract1 * vs1;
    } else {
        // Full 3D table - trilinear interpolation
        float h_scaled = h * h_scale;
        float s_scaled = s * s_scale;
        float v_scaled = v * v_scale;
        
        int32_t h_idx0 = (int32_t)h_scaled;
        int32_t s_idx0 = (int32_t)s_scaled;
        int32_t v_idx0 = (int32_t)v_scaled;
        
        s_idx0 = std::min(s_idx0, max_sat_idx);
        v_idx0 = std::min(v_idx0, max_val_idx);
        
        int32_t h_idx1 = h_idx0 + 1;
        if (h_idx0 >= max_hue_idx) {
            h_idx0 = max_hue_idx;
            h_idx1 = 0;
        }
        
        float h_fract1 = h_scaled - (float)h_idx0;
        float s_fract1 = s_scaled - (float)s_idx0;
        float v_fract1 = v_scaled - (float)v_idx0;
        float h_fract0 = 1.0f - h_fract1;
        float s_fract0 = 1.0f - s_fract1;
        float v_fract0 = 1.0f - v_fract1;
        
        const HSBModify* e00 = map_data + v_idx0 * val_step + h_idx0 * hue_step + s_idx0;
        const HSBModify* e01 = map_data + v_idx0 * val_step + h_idx1 * hue_step + s_idx0;
        const HSBModify* e10 = e00 + val_step;
        const HSBModify* e11 = e01 + val_step;
        
        float hs0 = v_fract0 * (h_fract0 * e00->hue_shift + h_fract1 * e01->hue_shift) +
                    v_fract1 * (h_fract0 * e10->hue_shift + h_fract1 * e11->hue_shift);
        float ss0 = v_fract0 * (h_fract0 * e00->sat_scale + h_fract1 * e01->sat_scale) +
                    v_fract1 * (h_fract0 * e10->sat_scale + h_fract1 * e11->sat_scale);
        float vs0 = v_fract0 * (h_fract0 * e00->val_scale + h_fract1 * e01->val_scale) +
                    v_fract1 * (h_fract0 * e10->val_scale + h_fract1 * e11->val_scale);
        
        e00++; e01++; e10++; e11++;
        float hs1 = v_fract0 * (h_fract0 * e00->hue_shift + h_fract1 * e01->hue_shift) +
                    v_fract1 * (h_fract0 * e10->hue_shift + h_fract1 * e11->hue_shift);
        float ss1 = v_fract0 * (h_fract0 * e00->sat_scale + h_fract1 * e01->sat_scale) +
                    v_fract1 * (h_fract0 * e10->sat_scale + h_fract1 * e11->sat_scale);
        float vs1 = v_fract0 * (h_fract0 * e00->val_scale + h_fract1 * e01->val_scale) +
                    v_fract1 * (h_fract0 * e10->val_scale + h_fract1 * e11->val_scale);
        
        hue_shift = s_fract0 * hs0 + s_fract1 * hs1;
        sat_scale = s_fract0 * ss0 + s_fract1 * ss1;
        val_scale = s_fract0 * vs0 + s_fract1 * vs1;
    }
    
    // Apply adjustments
    hue_shift *= (6.0f / 360.0f);  // Convert degrees to H range
    h += hue_shift;
    s = std::min(s * sat_scale, 1.0f);
    v = std::max(0.0f, std::min(v * val_scale, 1.0f));
    
    // Convert back to RGB
    hsv_to_rgb(h, s, v, r, g, b);
}

//=============================================================================
// 1D Tone Curve Interpolation
//=============================================================================

// Interpolate a custom tone curve (for ProfileToneCurve)
static float interpolate_tone_curve(float x, const float* curve, int curve_size) {
    if (curve_size < 2) return x;
    
    x = std::max(0.0f, std::min(1.0f, x));
    float idx = x * (float)(curve_size - 1);
    int i = (int)idx;
    if (i >= curve_size - 1) i = curve_size - 2;
    
    float fract = idx - (float)i;
    return curve[i] * (1.0f - fract) + curve[i + 1] * fract;
}

//=============================================================================
// Stage 1: Pre-Demosaic Operations (on RAW CFA data)
//=============================================================================

// Apply linearization table to RAW data
// Converts sensor ADC values to linear light values
// table: LUT mapping input [0, max_val] to output
// max_val: maximum input value (e.g., 16383 for 14-bit)
static void apply_linearization_table(
    float* data, npy_intp count,
    const float* table, int table_size, float max_val
) {
    float scale = (float)(table_size - 1) / max_val;
    for (npy_intp i = 0; i < count; i++) {
        float val = data[i];
        float idx = val * scale;
        int i0 = (int)idx;
        if (i0 < 0) i0 = 0;
        if (i0 >= table_size - 1) i0 = table_size - 2;
        float fract = idx - (float)i0;
        data[i] = table[i0] * (1.0f - fract) + table[i0 + 1] * fract;
    }
}

// Normalize RAW data using black and white levels per DNG spec Chapter 5.
// Implements: linear = (raw - BlackLevel[r%rR][c%rC][s] - DeltaH[c] - DeltaV[r]) / (WhiteLevel[s] - BlackLevel[...])
//
// Parameters:
//   data: raw pixel data, shape (height, width, samples_per_pixel) or (height, width) if samples=1
//   black_level: 3D array [repeat_rows][repeat_cols][samples_per_pixel] stored row-major
//   black_repeat_rows, black_repeat_cols: dimensions of the repeating black level pattern
//   black_delta_h: per-column delta array, length=width (or NULL if not used)
//   black_delta_v: per-row delta array, length=height (or NULL if not used)
//   white_level: per-sample white level array, length=samples_per_pixel
//   samples_per_pixel: number of samples (1 for CFA, 3 for LinearRaw)
//
// SDK ref: dng_linearize_plane.cpp, dng_linearization_info
static void normalize_black_white(
    float* data, npy_intp height, npy_intp width, int samples_per_pixel,
    const float* black_level, int black_repeat_rows, int black_repeat_cols,
    const float* black_delta_h, npy_intp delta_h_count,
    const float* black_delta_v, npy_intp delta_v_count,
    const float* white_level, int white_count,
    const uint16_t* linearization_table = nullptr, int linearization_table_size = 0
) {
    for (npy_intp row = 0; row < height; row++) {
        // Get row delta (0 if not provided)
        float delta_v = 0.0f;
        if (black_delta_v != NULL && delta_v_count > 0) {
            delta_v = black_delta_v[row % delta_v_count];
        }
        
        // Black level row index in repeating pattern
        int black_row = (int)(row % black_repeat_rows);
        
        for (npy_intp col = 0; col < width; col++) {
            // Get column delta (0 if not provided)
            float delta_h = 0.0f;
            if (black_delta_h != NULL && delta_h_count > 0) {
                delta_h = black_delta_h[col % delta_h_count];
            }
            
            // Black level column index in repeating pattern
            int black_col = (int)(col % black_repeat_cols);
            
            for (int sample = 0; sample < samples_per_pixel; sample++) {
                npy_intp pixel_idx = (row * width + col) * samples_per_pixel + sample;
                
                // Apply LinearizationTable if present (before black/white normalization)
                // SDK ref: dng_linearize_plane::Process - LUT lookup on raw ADC values
                float pixel_val = data[pixel_idx];
                if (linearization_table != nullptr && linearization_table_size > 0) {
                    int lut_idx = (int)pixel_val;
                    // Clamp to table bounds (upper bound common when table < max pixel value)
                    if (lut_idx >= linearization_table_size) lut_idx = linearization_table_size - 1;
                    pixel_val = (float)linearization_table[lut_idx];
                }
                
                // BlackLevel index: [row][col][sample] in row-major order
                int black_idx = (black_row * black_repeat_cols + black_col) * samples_per_pixel + sample;
                float black = black_level[black_idx];
                
                // WhiteLevel per sample
                float white = white_level[sample % white_count];
                
                // Total black level including deltas
                float total_black = black + delta_h + delta_v;
                
                // Normalize to [0, 1]
                float range = white - total_black;
                if (range > 0.0f) {
                    data[pixel_idx] = (pixel_val - total_black) / range;
                } else {
                    data[pixel_idx] = 0.0f;
                }
                data[pixel_idx] = std::max(0.0f, std::min(1.0f, data[pixel_idx]));
            }
        }
    }
}

//=============================================================================
// Stage 2: Post-Demosaic Operations (on RGB data)
//=============================================================================

// SDK ref: dng_resample.h lines 192-194
const uint32_t kResampleSubsampleBits2D = 5;
const uint32_t kResampleSubsampleCount2D = 1 << kResampleSubsampleBits2D;  // 32

// SDK ref: dng_resample.cpp dng_resample_bicubic::Evaluate lines 33-48
static inline double dng_resample_bicubic_Evaluate(double x) {
    const double A = -0.75;
    x = std::abs(x);
    
    if (x >= 2.0)
        return 0.0;
    else if (x >= 1.0)
        return (((A * x - 5.0 * A) * x + 8.0 * A) * x - 4.0 * A);
    else
        return (((A + 2.0) * x - (A + 3.0)) * x * x + 1.0);
}

// SDK ref: dng_resample.cpp dng_resample_weights_2d::Initialize lines 308-441
// Precomputed 2D bicubic weights: 32x32 fractional positions, 4x4 weights each
// Total: 32 * 32 * 16 = 16384 weights
static float* g_bicubic_weights_2d = nullptr;
static const uint32_t kBicubicWidth = 4;  // 2 * radius where radius = 2
static const uint32_t kBicubicWidthSqr = 16;

static void init_bicubic_weights_2d() {
    if (g_bicubic_weights_2d != nullptr) return;
    
    g_bicubic_weights_2d = new float[kResampleSubsampleCount2D * kResampleSubsampleCount2D * kBicubicWidthSqr];
    
    // SDK ref: dng_resample.cpp lines 369-441
    for (uint32_t y = 0; y < kResampleSubsampleCount2D; y++) {
        double yFract = y * (1.0 / (double)kResampleSubsampleCount2D);
        
        for (uint32_t x = 0; x < kResampleSubsampleCount2D; x++) {
            double xFract = x * (1.0 / (double)kResampleSubsampleCount2D);
            
            float* w32 = g_bicubic_weights_2d + 
                         (y * kResampleSubsampleCount2D + x) * kBicubicWidthSqr;
            
            // SDK ref: lines 386-428
            double t32 = 0.0;
            uint32_t index = 0;
            
            for (uint32_t i = 0; i < kBicubicWidth; i++) {
                int32_t yInt = ((int32_t)i) - 2 + 1;  // fRadius = 2, so -1, 0, 1, 2
                double yPos = yInt - yFract;
                
                for (uint32_t j = 0; j < kBicubicWidth; j++) {
                    int32_t xInt = ((int32_t)j) - 2 + 1;  // -1, 0, 1, 2
                    double xPos = xInt - xFract;
                    
                    // SDK ref: lines 415-418 - Separable kernel
                    w32[index] = (float)(dng_resample_bicubic_Evaluate(xPos) *
                                         dng_resample_bicubic_Evaluate(yPos));
                    t32 += w32[index];
                    index++;
                }
            }
            
            // SDK ref: lines 430-438 - Normalize weights to sum to 1.0
            float s32 = (float)(1.0 / t32);
            for (uint32_t i = 0; i < kBicubicWidthSqr; i++) {
                w32[i] *= s32;
            }
        }
    }
}

// WarpRectilinear lens distortion correction with per-plane coefficients
// SDK ref: dng_lens_correction.cpp dng_filter_warp::GetSrcPixelPosition, EvaluateRatio
// Uses radial polynomial model: ratio = kr0 + kr2*r^2 + kr4*r^4 + kr6*r^6 (EVEN powers)
// Each color plane has its own coefficients for lateral CA correction
// center_x, center_y: optical center in normalized [0,1] coordinates
static void warp_rectilinear(
    const float* src, float* dst,
    npy_intp height, npy_intp width, int channels,
    const double* radial_params, int num_planes, int num_coeffs,  // [num_planes][num_coeffs]
    const double* tangential_params,  // [num_planes][2] or NULL
    double center_x, double center_y,
    bool use_bicubic = true  // SDK ref: dng_lens_correction.cpp line 1251 uses dng_resample_bicubic
) {
    // SDK ref: dng_lens_correction.cpp line 1253 - Initialize weights
    if (use_bicubic) {
        init_bicubic_weights_2d();
    }
    
    // SDK ref: dng_lens_correction.cpp lines 1200-1236
    double cx = center_x * width;
    double cy = center_y * height;
    
    // SDK: fNormRadius = MaxDistancePointToRect(squareCenter, squareBounds)
    double corner_dists[4] = {
        std::sqrt(cx*cx + cy*cy),
        std::sqrt((width-cx)*(width-cx) + cy*cy),
        std::sqrt(cx*cx + (height-cy)*(height-cy)),
        std::sqrt((width-cx)*(width-cx) + (height-cy)*(height-cy))
    };
    double fNormRadius = *std::max_element(corner_dists, corner_dists + 4);
    if (fNormRadius < 1.0) fNormRadius = 1.0;
    double fInvNormRadius = 1.0 / fNormRadius;
    
    for (npy_intp y = 0; y < height; y++) {
        for (npy_intp x = 0; x < width; x++) {
            // SDK ref: GetSrcPixelPosition lines 1601-1604
            double diff_h = (double)x - cx;
            double diff_v = (double)y - cy;
            double diffNorm_h = diff_h * fInvNormRadius;
            double diffNorm_v = diff_v * fInvNormRadius;
            
            // SDK: rr = Min(diffNormSqr.v + diffNormSqr.h, 1.0)
            double rr = diffNorm_h * diffNorm_h + diffNorm_v * diffNorm_v;
            if (rr > 1.0) rr = 1.0;
            
            npy_intp dst_idx = (y * width + x) * channels;
            
            // SDK applies different warp per color plane for lateral CA correction
            for (int c = 0; c < channels && c < num_planes; c++) {
                const double* plane_radial = radial_params + c * num_coeffs;
                const double* plane_tan = tangential_params ? tangential_params + c * 2 : NULL;
                
                // SDK ref: WarpRectilinear v1 stores coeffs for EVEN powers (r^0, r^2, r^4, r^6)
                // dng_lens_correction.cpp lines 1908-1911: fData[0], fData[2], fData[4], fData[6]
                // Since rr = r^2, polynomial is: kr0 + kr2*rr + kr4*rr^2 + kr6*rr^3
                double ratio = plane_radial[0];
                double r_pow = rr;
                for (int i = 1; i < num_coeffs && i < 4; i++) {
                    ratio += plane_radial[i] * r_pow;
                    r_pow *= rr;
                }
                
                // SDK ref: GetSrcPixelPosition lines 1625-1626
                double dSrc_h = diff_h * ratio;
                double dSrc_v = diff_v * ratio;
                
                // Apply tangential warp if provided
                if (plane_tan && (plane_tan[0] != 0.0 || plane_tan[1] != 0.0)) {
                    double kt0 = plane_tan[0];
                    double kt1 = plane_tan[1];
                    double tan_h = 2*kt0*diffNorm_h*diffNorm_v + kt1*(rr + 2*diffNorm_h*diffNorm_h);
                    double tan_v = 2*kt1*diffNorm_h*diffNorm_v + kt0*(rr + 2*diffNorm_v*diffNorm_v);
                    dSrc_h += fNormRadius * tan_h;
                    dSrc_v += fNormRadius * tan_v;
                }
                
                // SDK ref: line 1663
                double src_x = cx + dSrc_h;
                double src_y = cy + dSrc_v;
                
                // SDK ref: lines 1511-1514 - clamp to image bounds
                src_x = std::max(0.0, std::min(src_x, (double)(width - 1)));
                src_y = std::max(0.0, std::min(src_y, (double)(height - 1)));
                
                if (use_bicubic) {
                    // SDK ref: dng_lens_correction.cpp lines 1516-1577
                    // Decompose into integer and fractional parts
                    int32_t sInt_v = (int32_t)std::floor(src_y);
                    int32_t sInt_h = (int32_t)std::floor(src_x);
                    
                    // SDK ref: line 1521-1522 - fractional part scaled to subsample count
                    int32_t sFct_v = (int32_t)((src_y - (double)sInt_v) * kResampleSubsampleCount2D);
                    int32_t sFct_h = (int32_t)((src_x - (double)sInt_h) * kResampleSubsampleCount2D);
                    
                    // SDK ref: line 1526 - add resample offset (1 - fRadius = 1 - 2 = -1)
                    sInt_v += -1;
                    sInt_h += -1;
                    
                    // SDK ref: lines 1530-1552 - clip
                    int32_t hMin = 0;
                    int32_t hMax = (int32_t)(width - kBicubicWidth);
                    int32_t vMin = 0;
                    int32_t vMax = (int32_t)(height - kBicubicWidth);
                    
                    if (sInt_h < hMin) { sInt_h = hMin; sFct_h = 0; }
                    else if (sInt_h > hMax) { sInt_h = hMax; sFct_h = 0; }
                    if (sInt_v < vMin) { sInt_v = vMin; sFct_v = 0; }
                    else if (sInt_v > vMax) { sInt_v = vMax; sFct_v = 0; }
                    
                    // SDK ref: line 1556 - get precomputed weights
                    const float* w = g_bicubic_weights_2d + 
                                     (sFct_v * kResampleSubsampleCount2D + sFct_h) * kBicubicWidthSqr;
                    
                    // SDK ref: lines 1562-1577 - perform 2D resample
                    float total = 0.0f;
                    int32_t wIdx = 0;
                    for (int32_t i = 0; i < (int32_t)kBicubicWidth; i++) {
                        for (int32_t j = 0; j < (int32_t)kBicubicWidth; j++) {
                            int32_t sy = sInt_v + i;
                            int32_t sx = sInt_h + j;
                            total += w[wIdx] * src[(sy * width + sx) * channels + c];
                            wIdx++;
                        }
                    }
                    
                    // SDK ref: line 1581 - Pin_real32
                    dst[dst_idx + c] = std::max(0.0f, std::min(1.0f, total));
                } else {
                    // Bilinear interpolation (fallback)
                    int x0 = (int)std::floor(src_x);
                    int y0 = (int)std::floor(src_y);
                    int x1 = std::min(x0 + 1, (int)(width - 1));
                    int y1 = std::min(y0 + 1, (int)(height - 1));
                    double fx = src_x - x0;
                    double fy = src_y - y0;
                    
                    double v00 = src[(y0 * width + x0) * channels + c];
                    double v01 = src[(y0 * width + x1) * channels + c];
                    double v10 = src[(y1 * width + x0) * channels + c];
                    double v11 = src[(y1 * width + x1) * channels + c];
                    
                    dst[dst_idx + c] = (float)(
                        v00 * (1-fx) * (1-fy) +
                        v01 * fx * (1-fy) +
                        v10 * (1-fx) * fy +
                        v11 * fx * fy
                    );
                }
            }
        }
    }
}

// Radial vignette correction
// Applies gain = 1 + k0*r^2 + k1*r^4 + k2*r^6 + k3*r^8 + k4*r^10
static void fix_vignette_radial(
    float* data, npy_intp height, npy_intp width, int channels,
    const double* params, int num_params,
    double center_x, double center_y
) {
    double cx = center_x * width;
    double cy = center_y * height;
    double corner_dists[4] = {
        std::sqrt(cx*cx + cy*cy),
        std::sqrt((width-cx)*(width-cx) + cy*cy),
        std::sqrt(cx*cx + (height-cy)*(height-cy)),
        std::sqrt((width-cx)*(width-cx) + (height-cy)*(height-cy))
    };
    double max_dist = *std::max_element(corner_dists, corner_dists + 4);
    if (max_dist < 1.0) max_dist = 1.0;
    
    for (npy_intp y = 0; y < height; y++) {
        for (npy_intp x = 0; x < width; x++) {
            double dx = ((double)x - cx) / max_dist;
            double dy = ((double)y - cy) / max_dist;
            double r2 = dx*dx + dy*dy;
            
            // Evaluate polynomial: gain = 1 + k0*r^2 + k1*r^4 + ...
            double gain = 1.0;
            double r2p = r2;
            for (int i = 0; i < num_params && i < 5; i++) {
                gain += params[i] * r2p;
                r2p *= r2;
            }
            
            // Apply gain
            npy_intp idx = (y * width + x) * channels;
            for (int c = 0; c < channels; c++) {
                data[idx + c] *= (float)gain;
            }
        }
    }
}

// Apply gain map (flat-field correction) to CFA data
// gain_map: 2D or 4D array of per-pixel gains
// For Bayer CFA: can be (H/2, W/2, 4) for 2x2 pattern or (H, W) for single plane
static void apply_gain_map_cfa(
    float* data, npy_intp height, npy_intp width,
    const float* gain_map, npy_intp gain_h, npy_intp gain_w,
    int cfa_pattern_width, int cfa_pattern_height
) {
    // Scale gain map coordinates to data coordinates
    float scale_y = (float)gain_h / (float)height;
    float scale_x = (float)gain_w / (float)width;
    
    for (npy_intp y = 0; y < height; y++) {
        for (npy_intp x = 0; x < width; x++) {
            // Bilinear interpolation of gain map
            float gy = y * scale_y;
            float gx = x * scale_x;
            
            int gy0 = (int)gy;
            int gx0 = (int)gx;
            if (gy0 >= gain_h - 1) gy0 = gain_h - 2;
            if (gx0 >= gain_w - 1) gx0 = gain_w - 2;
            
            float fy = gy - gy0;
            float fx = gx - gx0;
            
            float g00 = gain_map[gy0 * gain_w + gx0];
            float g01 = gain_map[gy0 * gain_w + gx0 + 1];
            float g10 = gain_map[(gy0 + 1) * gain_w + gx0];
            float g11 = gain_map[(gy0 + 1) * gain_w + gx0 + 1];
            
            float gain = g00 * (1-fx) * (1-fy) + g01 * fx * (1-fy) +
                        g10 * (1-fx) * fy + g11 * fx * fy;
            
            data[y * width + x] *= gain;
        }
    }
}

//=============================================================================
// sRGB Gamma Encoding
//=============================================================================

static inline float srgb_gamma(float linear) {
    if (linear <= 0.0031308f) {
        return linear * 12.92f;
    } else {
        return 1.055f * powf(linear, 1.0f / 2.4f) - 0.055f;
    }
}

//=============================================================================
// Python Module Functions
//=============================================================================

// Apply HueSatMap to RGB image
static PyObject* dng_color_apply_hue_sat_map(PyObject* self, PyObject* args) {
    PyArrayObject* rgb_array = NULL;
    PyArrayObject* map_array = NULL;
    int hue_divs, sat_divs, val_divs;
    
    if (!PyArg_ParseTuple(args, "O!O!iii",
            &PyArray_Type, &rgb_array,
            &PyArray_Type, &map_array,
            &hue_divs, &sat_divs, &val_divs)) {
        return NULL;
    }
    
    // Validate RGB array
    if (PyArray_NDIM(rgb_array) != 3 || PyArray_DIM(rgb_array, 2) != 3) {
        PyErr_SetString(PyExc_ValueError, "rgb must be shape (H, W, 3)");
        return NULL;
    }
    if (PyArray_TYPE(rgb_array) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "rgb must be float32");
        return NULL;
    }
    
    // Validate map array - should be (val_divs, hue_divs, sat_divs, 3)
    // or flattened with 3 values per entry (hue_shift, sat_scale, val_scale)
    npy_intp expected_entries = (npy_intp)hue_divs * sat_divs * val_divs;
    npy_intp map_entries = PyArray_SIZE(map_array) / 3;
    if (map_entries != expected_entries) {
        PyErr_Format(PyExc_ValueError, 
            "Map size mismatch: expected %ld entries, got %ld",
            (long)expected_entries, (long)map_entries);
        return NULL;
    }
    
    npy_intp height = PyArray_DIM(rgb_array, 0);
    npy_intp width = PyArray_DIM(rgb_array, 1);
    
    // Ensure contiguous input
    PyArrayObject* rgb_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)rgb_array, NPY_FLOAT32, 3, 3);
    if (!rgb_cont) return NULL;
    
    PyArrayObject* map_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)map_array, NPY_FLOAT32, 1, 4);
    if (!map_cont) {
        Py_DECREF(rgb_cont);
        return NULL;
    }
    
    // Create output array
    npy_intp dims[3] = {height, width, 3};
    PyObject* result = PyArray_SimpleNew(3, dims, NPY_FLOAT32);
    if (!result) {
        Py_DECREF(rgb_cont);
        Py_DECREF(map_cont);
        return NULL;
    }
    
    float* src_data = (float*)PyArray_DATA(rgb_cont);
    float* dst_data = (float*)PyArray_DATA((PyArrayObject*)result);
    const HSBModify* map_data = (const HSBModify*)PyArray_DATA(map_cont);
    
    // Copy source to dest first
    memcpy(dst_data, src_data, height * width * 3 * sizeof(float));
    
    // Apply HueSatMap to each pixel
    npy_intp total_pixels = height * width;
    for (npy_intp p = 0; p < total_pixels; p++) {
        npy_intp idx = p * 3;
        apply_hue_sat_map(
            dst_data[idx + 0], dst_data[idx + 1], dst_data[idx + 2],
            map_data, hue_divs, sat_divs, val_divs
        );
    }
    
    Py_DECREF(rgb_cont);
    Py_DECREF(map_cont);
    return result;
}

// Apply sRGB gamma encoding (linear to sRGB)
// sRGB spec: if x <= 0.0031308: 12.92 * x, else: 1.055 * x^(1/2.4) - 0.055
static PyObject* dng_color_srgb_gamma(PyObject* self, PyObject* args) {
    PyArrayObject* rgb_array = NULL;
    
    if (!PyArg_ParseTuple(args, "O!", &PyArray_Type, &rgb_array)) {
        return NULL;
    }
    
    if (PyArray_NDIM(rgb_array) != 3 || PyArray_DIM(rgb_array, 2) != 3) {
        PyErr_SetString(PyExc_ValueError, "rgb must be shape (H, W, 3)");
        return NULL;
    }
    if (PyArray_TYPE(rgb_array) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "rgb must be float32");
        return NULL;
    }
    
    npy_intp height = PyArray_DIM(rgb_array, 0);
    npy_intp width = PyArray_DIM(rgb_array, 1);
    
    PyArrayObject* rgb_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)rgb_array, NPY_FLOAT32, 3, 3);
    if (!rgb_cont) return NULL;
    
    npy_intp dims[3] = {height, width, 3};
    PyObject* result = PyArray_SimpleNew(3, dims, NPY_FLOAT32);
    if (!result) {
        Py_DECREF(rgb_cont);
        return NULL;
    }
    
    float* src_data = (float*)PyArray_DATA(rgb_cont);
    float* dst_data = (float*)PyArray_DATA((PyArrayObject*)result);
    
    npy_intp total = height * width * 3;
    const float threshold = 0.0031308f;
    const float inv_gamma = 1.0f / 2.4f;
    
    for (npy_intp i = 0; i < total; i++) {
        float x = src_data[i];
        float y;
        if (x <= threshold) {
            y = 12.92f * x;
        } else {
            y = 1.055f * std::pow(x, inv_gamma) - 0.055f;
        }
        dst_data[i] = std::max(0.0f, std::min(1.0f, y));
    }
    
    Py_DECREF(rgb_cont);
    return result;
}

// Apply 3x3 color matrix transform to RGB image with optional clipping
static PyObject* dng_color_matrix_transform(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyArrayObject* rgb_array = NULL;
    PyArrayObject* matrix_array = NULL;
    int clip = 1;  // Default: clip to [0,1]
    
    static char* kwlist[] = {"rgb", "matrix", "clip", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!|p", kwlist,
            &PyArray_Type, &rgb_array,
            &PyArray_Type, &matrix_array,
            &clip)) {
        return NULL;
    }
    
    if (PyArray_NDIM(rgb_array) != 3 || PyArray_DIM(rgb_array, 2) != 3) {
        PyErr_SetString(PyExc_ValueError, "rgb must be shape (H, W, 3)");
        return NULL;
    }
    if (PyArray_TYPE(rgb_array) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "rgb must be float32");
        return NULL;
    }
    if (PyArray_SIZE(matrix_array) != 9) {
        PyErr_SetString(PyExc_ValueError, "matrix must be 3x3 (9 elements)");
        return NULL;
    }
    
    npy_intp height = PyArray_DIM(rgb_array, 0);
    npy_intp width = PyArray_DIM(rgb_array, 1);
    
    PyArrayObject* rgb_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)rgb_array, NPY_FLOAT32, 3, 3);
    PyArrayObject* mat_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)matrix_array, NPY_FLOAT32, 1, 2);
    
    if (!rgb_cont || !mat_cont) {
        Py_XDECREF(rgb_cont);
        Py_XDECREF(mat_cont);
        return NULL;
    }
    
    npy_intp dims[3] = {height, width, 3};
    PyObject* result = PyArray_SimpleNew(3, dims, NPY_FLOAT32);
    if (!result) {
        Py_DECREF(rgb_cont);
        Py_DECREF(mat_cont);
        return NULL;
    }
    
    float* src_data = (float*)PyArray_DATA(rgb_cont);
    float* dst_data = (float*)PyArray_DATA((PyArrayObject*)result);
    const float* m = (const float*)PyArray_DATA(mat_cont);
    
    // Matrix is row-major: m[row*3 + col]
    float m00 = m[0], m01 = m[1], m02 = m[2];
    float m10 = m[3], m11 = m[4], m12 = m[5];
    float m20 = m[6], m21 = m[7], m22 = m[8];
    
    npy_intp total_pixels = height * width;
    
    if (clip) {
        for (npy_intp i = 0; i < total_pixels; i++) {
            float r = src_data[i * 3 + 0];
            float g = src_data[i * 3 + 1];
            float b = src_data[i * 3 + 2];
            
            dst_data[i * 3 + 0] = std::max(0.0f, std::min(1.0f, m00 * r + m01 * g + m02 * b));
            dst_data[i * 3 + 1] = std::max(0.0f, std::min(1.0f, m10 * r + m11 * g + m12 * b));
            dst_data[i * 3 + 2] = std::max(0.0f, std::min(1.0f, m20 * r + m21 * g + m22 * b));
        }
    } else {
        for (npy_intp i = 0; i < total_pixels; i++) {
            float r = src_data[i * 3 + 0];
            float g = src_data[i * 3 + 1];
            float b = src_data[i * 3 + 2];
            
            dst_data[i * 3 + 0] = m00 * r + m01 * g + m02 * b;
            dst_data[i * 3 + 1] = m10 * r + m11 * g + m12 * b;
            dst_data[i * 3 + 2] = m20 * r + m21 * g + m22 * b;
        }
    }
    
    Py_DECREF(rgb_cont);
    Py_DECREF(mat_cont);
    return result;
}

// ============================================================================
// Exposure Ramp (dng_function_exposure_ramp from dng_render.cpp lines 50-103)
// Direct port of SDK code
// ============================================================================

// SDK ref: dng_render.cpp dng_function_exposure_ramp::Evaluate() lines 81-103
static inline float exposure_ramp_evaluate(float x, float black, float slope, 
                                           float radius, float qScale, 
                                           bool supportOverrange) {
    // Region 1: x <= black - radius → 0
    if (x <= black - radius)
        return 0.0f;
    
    // Region 2: x >= black + radius → linear ramp
    if (x >= black + radius) {
        float y = (x - black) * slope;
        if (!supportOverrange)
            y = std::min(y, 1.0f);
        return y;
    }
    
    // Region 3: quadratic blend
    float y = x - (black - radius);
    return qScale * y * y;
}

// Apply exposure ramp to RGB image
// SDK ref: dng_render.cpp lines 50-103, 1907-1928
static PyObject* dng_color_apply_exposure_ramp(PyObject* self, PyObject* args) {
    PyArrayObject* rgb_array = NULL;
    double white, black, minBlack;
    int supportOverrange = 0;
    
    if (!PyArg_ParseTuple(args, "O!ddd|p",
            &PyArray_Type, &rgb_array,
            &white, &black, &minBlack,
            &supportOverrange)) {
        return NULL;
    }
    
    if (PyArray_NDIM(rgb_array) != 3 || PyArray_DIM(rgb_array, 2) != 3) {
        PyErr_SetString(PyExc_ValueError, "rgb must be shape (H, W, 3)");
        return NULL;
    }
    if (PyArray_TYPE(rgb_array) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "rgb must be float32");
        return NULL;
    }
    
    npy_intp height = PyArray_DIM(rgb_array, 0);
    npy_intp width = PyArray_DIM(rgb_array, 1);
    
    PyArrayObject* rgb_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)rgb_array, NPY_FLOAT32, 3, 3);
    if (!rgb_cont) return NULL;
    
    npy_intp dims[3] = {height, width, 3};
    PyObject* result = PyArray_SimpleNew(3, dims, NPY_FLOAT32);
    if (!result) {
        Py_DECREF(rgb_cont);
        return NULL;
    }
    
    const float* src_data = (const float*)PyArray_DATA(rgb_cont);
    float* dst_data = (float*)PyArray_DATA((PyArrayObject*)result);
    
    // SDK ref: dng_render.cpp lines 55-75 (constructor)
    float slope = 1.0f / (float)(white - black);
    
    // Compute radius for quadratic blend region
    const float kMaxCurveX = 0.5f;      // Fraction of minBlack
    const float kMaxCurveY = 1.0f / 16.0f;  // Fraction of white
    
    float radius = std::min(kMaxCurveX * (float)minBlack, kMaxCurveY / slope);
    
    float qScale = 0.0f;
    if (radius > 0.0f)
        qScale = slope / (4.0f * radius);
    
    // Process all pixels
    npy_intp total = height * width * 3;
    for (npy_intp i = 0; i < total; i++) {
        dst_data[i] = exposure_ramp_evaluate(src_data[i], (float)black, slope, 
                                              radius, qScale, supportOverrange != 0);
    }
    
    Py_DECREF(rgb_cont);
    return result;
}

// Apply simple per-pixel curve via LUT interpolation
// No hue preservation - each channel processed independently
static PyObject* dng_color_apply_curve(PyObject* self, PyObject* args) {
    PyArrayObject* rgb_array = NULL;
    PyArrayObject* curve_array = NULL;
    
    if (!PyArg_ParseTuple(args, "O!O!",
            &PyArray_Type, &rgb_array,
            &PyArray_Type, &curve_array)) {
        return NULL;
    }
    
    if (PyArray_NDIM(rgb_array) != 3 || PyArray_DIM(rgb_array, 2) != 3) {
        PyErr_SetString(PyExc_ValueError, "rgb must be shape (H, W, 3)");
        return NULL;
    }
    if (PyArray_TYPE(rgb_array) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "rgb must be float32");
        return NULL;
    }
    
    npy_intp height = PyArray_DIM(rgb_array, 0);
    npy_intp width = PyArray_DIM(rgb_array, 1);
    int curve_size = (int)PyArray_SIZE(curve_array);
    
    if (curve_size < 2) {
        PyErr_SetString(PyExc_ValueError, "Curve must have at least 2 points");
        return NULL;
    }
    
    PyArrayObject* rgb_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)rgb_array, NPY_FLOAT32, 3, 3);
    PyArrayObject* curve_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)curve_array, NPY_FLOAT32, 1, 1);
    
    if (!rgb_cont || !curve_cont) {
        Py_XDECREF(rgb_cont);
        Py_XDECREF(curve_cont);
        return NULL;
    }
    
    npy_intp dims[3] = {height, width, 3};
    PyObject* result = PyArray_SimpleNew(3, dims, NPY_FLOAT32);
    if (!result) {
        Py_DECREF(rgb_cont);
        Py_DECREF(curve_cont);
        return NULL;
    }
    
    const float* src_data = (const float*)PyArray_DATA(rgb_cont);
    float* dst_data = (float*)PyArray_DATA((PyArrayObject*)result);
    const float* curve = (const float*)PyArray_DATA(curve_cont);
    
    // Simple per-pixel LUT interpolation
    npy_intp total = height * width * 3;
    for (npy_intp i = 0; i < total; i++) {
        dst_data[i] = interpolate_tone_curve(src_data[i], curve, curve_size);
    }
    
    Py_DECREF(rgb_cont);
    Py_DECREF(curve_cont);
    return result;
}

// Apply hue-preserving RGB curve (RefBaselineRGBTone from dng_reference.cpp)
// This preserves color relationships by interpolating the middle channel
static PyObject* dng_color_apply_curve_hue_preserving(PyObject* self, PyObject* args) {
    PyArrayObject* rgb_array = NULL;
    PyArrayObject* curve_array = NULL;
    
    if (!PyArg_ParseTuple(args, "O!O!",
            &PyArray_Type, &rgb_array,
            &PyArray_Type, &curve_array)) {
        return NULL;
    }
    
    if (PyArray_NDIM(rgb_array) != 3 || PyArray_DIM(rgb_array, 2) != 3) {
        PyErr_SetString(PyExc_ValueError, "rgb must be shape (H, W, 3)");
        return NULL;
    }
    if (PyArray_TYPE(rgb_array) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "rgb must be float32");
        return NULL;
    }
    
    npy_intp height = PyArray_DIM(rgb_array, 0);
    npy_intp width = PyArray_DIM(rgb_array, 1);
    int curve_size = (int)PyArray_SIZE(curve_array);
    
    if (curve_size < 2) {
        PyErr_SetString(PyExc_ValueError, "Tone curve must have at least 2 points");
        return NULL;
    }
    
    PyArrayObject* rgb_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)rgb_array, NPY_FLOAT32, 3, 3);
    PyArrayObject* curve_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)curve_array, NPY_FLOAT32, 1, 1);
    
    if (!rgb_cont || !curve_cont) {
        Py_XDECREF(rgb_cont);
        Py_XDECREF(curve_cont);
        return NULL;
    }
    
    npy_intp dims[3] = {height, width, 3};
    PyObject* result = PyArray_SimpleNew(3, dims, NPY_FLOAT32);
    if (!result) {
        Py_DECREF(rgb_cont);
        Py_DECREF(curve_cont);
        return NULL;
    }
    
    float* src_data = (float*)PyArray_DATA(rgb_cont);
    float* dst_data = (float*)PyArray_DATA((PyArrayObject*)result);
    const float* curve = (const float*)PyArray_DATA(curve_cont);
    
    npy_intp total_pixels = height * width;
    
    // Process each pixel with hue-preserving tone mapping
    // SDK ref: dng_reference.cpp RefBaselineRGBTone lines 1868-1990
    for (npy_intp i = 0; i < total_pixels; i++) {
        float r = std::max(0.0f, std::min(1.0f, src_data[i * 3 + 0]));
        float g = std::max(0.0f, std::min(1.0f, src_data[i * 3 + 1]));
        float b = std::max(0.0f, std::min(1.0f, src_data[i * 3 + 2]));
        
        float rr, gg, bb;
        
        // Macro to apply curve and preserve hue for sorted r >= g >= b case
        #define RGBTone(r_in, g_in, b_in, r_out, g_out, b_out) \
            { \
                r_out = interpolate_tone_curve(r_in, curve, curve_size); \
                b_out = interpolate_tone_curve(b_in, curve, curve_size); \
                float denom = r_in - b_in; \
                if (denom > 1e-10f) { \
                    g_out = b_out + ((r_out - b_out) * (g_in - b_in) / denom); \
                } else { \
                    g_out = b_out; \
                } \
            }
        
        if (r >= g) {
            if (g > b) {
                // Case 1: r >= g > b
                RGBTone(r, g, b, rr, gg, bb);
            } else if (b > r) {
                // Case 2: b > r >= g
                RGBTone(b, r, g, bb, rr, gg);
            } else if (b > g) {
                // Case 3: r >= b > g
                RGBTone(r, b, g, rr, bb, gg);
            } else {
                // Case 4: r >= g == b
                rr = interpolate_tone_curve(r, curve, curve_size);
                gg = interpolate_tone_curve(g, curve, curve_size);
                bb = gg;
            }
        } else {
            if (r >= b) {
                // Case 5: g > r >= b
                RGBTone(g, r, b, gg, rr, bb);
            } else if (b > g) {
                // Case 6: b > g > r
                RGBTone(b, g, r, bb, gg, rr);
            } else {
                // Case 7: g >= b > r
                RGBTone(g, b, r, gg, bb, rr);
            }
        }
        
        #undef RGBTone
        
        dst_data[i * 3 + 0] = rr;
        dst_data[i * 3 + 1] = gg;
        dst_data[i * 3 + 2] = bb;
    }
    
    Py_DECREF(rgb_cont);
    Py_DECREF(curve_cont);
    return result;
}

// Apply custom tone curve to image (simple per-channel, no hue preservation)
static PyObject* dng_color_apply_tone_curve(PyObject* self, PyObject* args) {
    PyArrayObject* rgb_array = NULL;
    PyArrayObject* curve_array = NULL;
    
    if (!PyArg_ParseTuple(args, "O!O!",
            &PyArray_Type, &rgb_array,
            &PyArray_Type, &curve_array)) {
        return NULL;
    }
    
    if (PyArray_NDIM(rgb_array) != 3 || PyArray_DIM(rgb_array, 2) != 3) {
        PyErr_SetString(PyExc_ValueError, "rgb must be shape (H, W, 3)");
        return NULL;
    }
    if (PyArray_TYPE(rgb_array) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "rgb must be float32");
        return NULL;
    }
    
    npy_intp height = PyArray_DIM(rgb_array, 0);
    npy_intp width = PyArray_DIM(rgb_array, 1);
    int curve_size = (int)PyArray_SIZE(curve_array);
    
    if (curve_size < 2) {
        PyErr_SetString(PyExc_ValueError, "Tone curve must have at least 2 points");
        return NULL;
    }
    
    PyArrayObject* rgb_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)rgb_array, NPY_FLOAT32, 3, 3);
    PyArrayObject* curve_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)curve_array, NPY_FLOAT32, 1, 1);
    
    if (!rgb_cont || !curve_cont) {
        Py_XDECREF(rgb_cont);
        Py_XDECREF(curve_cont);
        return NULL;
    }
    
    npy_intp dims[3] = {height, width, 3};
    PyObject* result = PyArray_SimpleNew(3, dims, NPY_FLOAT32);
    if (!result) {
        Py_DECREF(rgb_cont);
        Py_DECREF(curve_cont);
        return NULL;
    }
    
    float* src_data = (float*)PyArray_DATA(rgb_cont);
    float* dst_data = (float*)PyArray_DATA((PyArrayObject*)result);
    const float* curve = (const float*)PyArray_DATA(curve_cont);
    
    npy_intp total = height * width * 3;
    for (npy_intp i = 0; i < total; i++) {
        dst_data[i] = interpolate_tone_curve(src_data[i], curve, curve_size);
    }
    
    Py_DECREF(rgb_cont);
    Py_DECREF(curve_cont);
    return result;
}

// Apply linearization table to RAW CFA data
static PyObject* dng_color_linearize(PyObject* self, PyObject* args) {
    PyArrayObject* data_array = NULL;
    PyArrayObject* table_array = NULL;
    float max_val;
    
    if (!PyArg_ParseTuple(args, "O!O!f",
            &PyArray_Type, &data_array,
            &PyArray_Type, &table_array,
            &max_val)) {
        return NULL;
    }
    
    if (PyArray_TYPE(data_array) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "data must be float32");
        return NULL;
    }
    
    PyArrayObject* data_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)data_array, NPY_FLOAT32, 1, 3);
    PyArrayObject* table_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)table_array, NPY_FLOAT32, 1, 1);
    
    if (!data_cont || !table_cont) {
        Py_XDECREF(data_cont);
        Py_XDECREF(table_cont);
        return NULL;
    }
    
    // Copy data for output
    PyObject* result = PyArray_NewCopy(data_cont, NPY_CORDER);
    if (!result) {
        Py_DECREF(data_cont);
        Py_DECREF(table_cont);
        return NULL;
    }
    
    float* result_data = (float*)PyArray_DATA((PyArrayObject*)result);
    const float* table = (const float*)PyArray_DATA(table_cont);
    npy_intp count = PyArray_SIZE(data_cont);
    int table_size = (int)PyArray_SIZE(table_cont);
    
    apply_linearization_table(result_data, count, table, table_size, max_val);
    
    Py_DECREF(data_cont);
    Py_DECREF(table_cont);
    return result;
}

// Normalize RAW data using black/white levels per DNG spec Chapter 5.
// SDK ref: dng_linearize_plane.cpp, dng_linearization_info
//
// Args:
//   data: RAW pixel data, float32, shape (H, W) or (H, W, samples_per_pixel)
//   black_level: BlackLevel pattern, float32, shape (repeat_rows, repeat_cols, samples_per_pixel)
//                or flattened 1D array in row-col-sample order
//   black_repeat_rows: number of rows in repeating pattern (from BlackLevelRepeatDim[0])
//   black_repeat_cols: number of cols in repeating pattern (from BlackLevelRepeatDim[1])
//   samples_per_pixel: 1 for CFA, 3 for LinearRaw
//   white_level: WhiteLevel per sample, float32, shape (samples_per_pixel,)
//   black_delta_h: optional per-column delta, float32, shape (width,) or None
//   black_delta_v: optional per-row delta, float32, shape (height,) or None
static PyObject* dng_color_normalize_raw(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyArrayObject* data_array = NULL;
    PyArrayObject* black_array = NULL;
    int black_repeat_rows = 1;
    int black_repeat_cols = 1;
    int samples_per_pixel = 1;
    PyArrayObject* white_array = NULL;
    PyObject* delta_h_obj = Py_None;
    PyObject* delta_v_obj = Py_None;
    PyObject* linearization_table_obj = Py_None;
    
    static const char* kwlist[] = {
        "data", "black_level", "black_repeat_rows", "black_repeat_cols",
        "samples_per_pixel", "white_level", "black_delta_h", "black_delta_v",
        "linearization_table", NULL
    };
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!iiiO!|OOO",
            const_cast<char**>(kwlist),
            &PyArray_Type, &data_array,
            &PyArray_Type, &black_array,
            &black_repeat_rows,
            &black_repeat_cols,
            &samples_per_pixel,
            &PyArray_Type, &white_array,
            &delta_h_obj,
            &delta_v_obj,
            &linearization_table_obj)) {
        return NULL;
    }
    
    // Validate data array
    if (PyArray_TYPE(data_array) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "data must be float32");
        return NULL;
    }
    
    int ndim = PyArray_NDIM(data_array);
    if (ndim < 2 || ndim > 3) {
        PyErr_SetString(PyExc_ValueError, "data must be 2D (H,W) or 3D (H,W,C)");
        return NULL;
    }
    
    npy_intp height = PyArray_DIM(data_array, 0);
    npy_intp width = PyArray_DIM(data_array, 1);
    int data_samples = (ndim == 3) ? (int)PyArray_DIM(data_array, 2) : 1;
    
    if (data_samples != samples_per_pixel) {
        PyErr_SetString(PyExc_ValueError, "data channels must match samples_per_pixel");
        return NULL;
    }
    
    // Make contiguous copies
    PyArrayObject* data_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)data_array, NPY_FLOAT32, 2, 3);
    PyArrayObject* black_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)black_array, NPY_FLOAT32, 1, 3);
    PyArrayObject* white_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)white_array, NPY_FLOAT32, 1, 1);
    
    if (!data_cont || !black_cont || !white_cont) {
        Py_XDECREF(data_cont);
        Py_XDECREF(black_cont);
        Py_XDECREF(white_cont);
        return NULL;
    }
    
    // Validate black_level size matches repeat pattern
    npy_intp expected_black_size = black_repeat_rows * black_repeat_cols * samples_per_pixel;
    if (PyArray_SIZE(black_cont) != expected_black_size) {
        PyErr_Format(PyExc_ValueError, 
            "black_level size (%zd) must equal repeat_rows * repeat_cols * samples_per_pixel (%zd)",
            (Py_ssize_t)PyArray_SIZE(black_cont), (Py_ssize_t)expected_black_size);
        Py_DECREF(data_cont);
        Py_DECREF(black_cont);
        Py_DECREF(white_cont);
        return NULL;
    }
    
    // Handle optional delta arrays
    PyArrayObject* delta_h_cont = NULL;
    PyArrayObject* delta_v_cont = NULL;
    npy_intp delta_h_count = 0;
    npy_intp delta_v_count = 0;
    
    if (delta_h_obj != Py_None && delta_h_obj != NULL) {
        delta_h_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
            delta_h_obj, NPY_FLOAT32, 1, 1);
        if (!delta_h_cont) {
            Py_DECREF(data_cont);
            Py_DECREF(black_cont);
            Py_DECREF(white_cont);
            return NULL;
        }
        delta_h_count = PyArray_SIZE(delta_h_cont);
    }
    
    if (delta_v_obj != Py_None && delta_v_obj != NULL) {
        delta_v_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
            delta_v_obj, NPY_FLOAT32, 1, 1);
        if (!delta_v_cont) {
            Py_DECREF(data_cont);
            Py_DECREF(black_cont);
            Py_DECREF(white_cont);
            Py_XDECREF(delta_h_cont);
            return NULL;
        }
        delta_v_count = PyArray_SIZE(delta_v_cont);
    }
    
    // Handle optional linearization table (uint16 LUT)
    PyArrayObject* lin_table_cont = NULL;
    int lin_table_size = 0;
    if (linearization_table_obj != Py_None && linearization_table_obj != NULL) {
        lin_table_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
            linearization_table_obj, NPY_UINT16, 1, 1);
        if (!lin_table_cont) {
            Py_DECREF(data_cont);
            Py_DECREF(black_cont);
            Py_DECREF(white_cont);
            Py_XDECREF(delta_h_cont);
            Py_XDECREF(delta_v_cont);
            return NULL;
        }
        lin_table_size = (int)PyArray_SIZE(lin_table_cont);
    }
    
    // Copy data for output
    PyObject* result = PyArray_NewCopy(data_cont, NPY_CORDER);
    if (!result) {
        Py_DECREF(data_cont);
        Py_DECREF(black_cont);
        Py_DECREF(white_cont);
        Py_XDECREF(delta_h_cont);
        Py_XDECREF(delta_v_cont);
        Py_XDECREF(lin_table_cont);
        return NULL;
    }
    
    float* result_data = (float*)PyArray_DATA((PyArrayObject*)result);
    const float* black = (const float*)PyArray_DATA(black_cont);
    const float* white = (const float*)PyArray_DATA(white_cont);
    const float* delta_h = delta_h_cont ? (const float*)PyArray_DATA(delta_h_cont) : NULL;
    const float* delta_v = delta_v_cont ? (const float*)PyArray_DATA(delta_v_cont) : NULL;
    int white_count = (int)PyArray_SIZE(white_cont);
    
    const uint16_t* lin_table = lin_table_cont ? (const uint16_t*)PyArray_DATA(lin_table_cont) : nullptr;
    
    normalize_black_white(result_data, height, width, samples_per_pixel,
                         black, black_repeat_rows, black_repeat_cols,
                         delta_h, delta_h_count,
                         delta_v, delta_v_count,
                         white, white_count,
                         lin_table, lin_table_size);
    
    Py_DECREF(data_cont);
    Py_DECREF(black_cont);
    Py_DECREF(white_cont);
    Py_XDECREF(delta_h_cont);
    Py_XDECREF(delta_v_cont);
    Py_XDECREF(lin_table_cont);
    return result;
}

// Apply gain map (flat-field correction) to RAW CFA
static PyObject* dng_color_apply_gain_map(PyObject* self, PyObject* args) {
    PyArrayObject* data_array = NULL;
    PyArrayObject* gain_array = NULL;
    
    if (!PyArg_ParseTuple(args, "O!O!",
            &PyArray_Type, &data_array,
            &PyArray_Type, &gain_array)) {
        return NULL;
    }
    
    if (PyArray_TYPE(data_array) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "data must be float32");
        return NULL;
    }
    if (PyArray_NDIM(data_array) != 2) {
        PyErr_SetString(PyExc_ValueError, "data must be 2D (H,W) CFA");
        return NULL;
    }
    if (PyArray_NDIM(gain_array) != 2) {
        PyErr_SetString(PyExc_ValueError, "gain_map must be 2D");
        return NULL;
    }
    
    npy_intp height = PyArray_DIM(data_array, 0);
    npy_intp width = PyArray_DIM(data_array, 1);
    npy_intp gain_h = PyArray_DIM(gain_array, 0);
    npy_intp gain_w = PyArray_DIM(gain_array, 1);
    
    PyArrayObject* data_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)data_array, NPY_FLOAT32, 2, 2);
    PyArrayObject* gain_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)gain_array, NPY_FLOAT32, 2, 2);
    
    if (!data_cont || !gain_cont) {
        Py_XDECREF(data_cont);
        Py_XDECREF(gain_cont);
        return NULL;
    }
    
    // Copy data for output
    PyObject* result = PyArray_NewCopy(data_cont, NPY_CORDER);
    if (!result) {
        Py_DECREF(data_cont);
        Py_DECREF(gain_cont);
        return NULL;
    }
    
    float* result_data = (float*)PyArray_DATA((PyArrayObject*)result);
    const float* gain = (const float*)PyArray_DATA(gain_cont);
    
    apply_gain_map_cfa(result_data, height, width, gain, gain_h, gain_w, 2, 2);
    
    Py_DECREF(data_cont);
    Py_DECREF(gain_cont);
    return result;
}

// ============================================================================
// ProfileGainTableMap application
// Direct port from DNG SDK dng_reference.cpp RefBaselineProfileGainTableMap()
// ============================================================================

static inline float Lerp_real32(float a, float b, float t) {
    return a + (b - a) * t;
}

static inline float Pin_real32(float lo, float x, float hi) {
    return x < lo ? lo : (x > hi ? hi : x);
}

static inline float Min_real32(float a, float b) {
    return a < b ? a : b;
}

static inline float Max_real32(float a, float b) {
    return a > b ? a : b;
}

static inline int Min_int32(int a, int b) {
    return a < b ? a : b;
}

// Direct port of RefBaselineProfileGainTableMap from dng_reference.cpp lines 3260-3460
static void RefBaselineProfileGainTableMap(
    const float* rSrcPtr,
    const float* gSrcPtr,
    const float* bSrcPtr,
    float* rDstPtr,
    float* gDstPtr,
    float* bDstPtr,
    const int cols,
    const int top,
    const int left,
    const int imageAreaL,
    const int imageAreaT,
    const int imageAreaW,
    const int imageAreaH,
    const float exposureWeightGain,
    // Gain table map parameters
    const int points_v,
    const int points_h,
    const float mapSpacingV,
    const float mapSpacingH,
    const float mapOriginV,
    const float mapOriginH,
    const int numTablePoints,
    const float* mapInputWeights,
    const float gamma,
    const float* gains,  // gains[row][col][tablePoint] flattened
    const bool supportOverrange
) {
    const float miw0 = mapInputWeights[0];
    const float miw1 = mapInputWeights[1];
    const float miw2 = mapInputWeights[2];
    const float miw3 = mapInputWeights[3];
    const float miw4 = mapInputWeights[4];
    
    const float mapOriginH32 = mapOriginH;
    const float mapOriginV32 = mapOriginV;
    const float mapSpacingH32 = mapSpacingH;
    const float mapSpacingV32 = mapSpacingV;
    
    const float xLimitLo = 0.0f;
    const float yLimitLo = 0.0f;
    const float xLimitHi = (float)(points_h - 1);
    const float yLimitHi = (float)(points_v - 1);
    
    const int xPixelLimit = points_h - 1;
    const int yPixelLimit = points_v - 1;
    
    const int tableSize = numTablePoints;
    const int tableLimit = tableSize - 1;
    
    // For gain table indexing: gains[row * rowStep + col * colStep + tableIdx]
    const int colStep = numTablePoints;
    const int rowStep = points_h * numTablePoints;
    
    // Initialize sample position. Note the half-pixel offset.
    float y = (float)top + 0.5f;
    float x = (float)left + 0.5f;
    
    // Process each pixel in this row.
    for (int col = 0; col < cols; col++) {
        
        // Transform to image-relative coordinates.
        float u_image = (x - (float)imageAreaL) / (float)imageAreaW;
        float v_image = (y - (float)imageAreaT) / (float)imageAreaH;
        
        // Transform to map-relative coordinates.
        float x_map = (u_image - mapOriginH32) / mapSpacingH32;
        float y_map = (v_image - mapOriginV32) / mapSpacingV32;
        
        // Clamp to valid sample positions.
        x_map = Pin_real32(xLimitLo, x_map, xLimitHi);
        y_map = Pin_real32(yLimitLo, y_map, yLimitHi);
        
        // Compute integer 2D indices.
        int x0 = (int)x_map;
        int x1 = Min_int32(x0 + 1, xPixelLimit);
        
        int y0 = (int)y_map;
        int y1 = Min_int32(y0 + 1, yPixelLimit);
        
        // Compute fractional weights.
        float xf = x_map - (float)x0;
        float yf = y_map - (float)y0;
        
        // Read linear RGB values in RIMM space.
        float r = rSrcPtr[col];
        float g = gSrcPtr[col];
        float b = bSrcPtr[col];
        
        // Apply MapInputWeights (5-element dot product).
        float minValue = Min_real32(r, Min_real32(g, b));
        float maxValue = Max_real32(r, Max_real32(g, b));
        
        float weight = ((miw0 * r) +
                       (miw1 * g) +
                       (miw2 * b) +
                       (miw3 * minValue) +
                       (miw4 * maxValue));
        
        // Scale weight by baseline exposure.
        weight = weight * exposureWeightGain;
        
        // Clamp weight to [0,1].
        weight = Pin_real32(0.0f, weight, 1.0f);
        
        // Apply gamma parameter.
        if (gamma != 1.0f)
            weight = powf(weight, gamma);
        
        // Scale weight by table size and compute table indices.
        float weightScaled = weight * (float)tableSize;
        
        int w0 = Min_int32((int)weightScaled, tableLimit);
        int w1 = Min_int32(w0 + 1, tableLimit);
        
        float wf = weightScaled - (float)w0;
        
        // Look up 8 gains.
        float gain000 = gains[y0 * rowStep + x0 * colStep + w0];
        float gain001 = gains[y0 * rowStep + x0 * colStep + w1];
        float gain010 = gains[y0 * rowStep + x1 * colStep + w0];
        float gain011 = gains[y0 * rowStep + x1 * colStep + w1];
        float gain100 = gains[y1 * rowStep + x0 * colStep + w0];
        float gain101 = gains[y1 * rowStep + x0 * colStep + w1];
        float gain110 = gains[y1 * rowStep + x1 * colStep + w0];
        float gain111 = gains[y1 * rowStep + x1 * colStep + w1];
        
        // Interpolate in table (w) direction.
        float gain00_ = Lerp_real32(gain000, gain001, wf);
        float gain01_ = Lerp_real32(gain010, gain011, wf);
        float gain10_ = Lerp_real32(gain100, gain101, wf);
        float gain11_ = Lerp_real32(gain110, gain111, wf);
        
        // Interpolate in column (x) direction.
        float gain0__ = Lerp_real32(gain00_, gain01_, xf);
        float gain1__ = Lerp_real32(gain10_, gain11_, xf);
        
        // Interpolate in row (y) direction.
        float gain = Lerp_real32(gain0__, gain1__, yf);
        
        // Apply gain.
        r *= gain;
        g *= gain;
        b *= gain;
        
        // Optionally clamp to [0,1].
        if (!supportOverrange) {
            r = Pin_real32(0.0f, r, 1.0f);
            g = Pin_real32(0.0f, g, 1.0f);
            b = Pin_real32(0.0f, b, 1.0f);
        }
        
        // Store the result.
        rDstPtr[col] = r;
        gDstPtr[col] = g;
        bDstPtr[col] = b;
        
        // Increment sample position for next column.
        x += 1.0f;
    }
}

// Wrapper that processes entire image using the SDK row-by-row approach
static void apply_profile_gain_table_map(
    float* rgb,  // Input/output RGB image, shape (H, W, 3), interleaved
    int height, int width,
    int points_v, int points_h,
    float spacing_v, float spacing_h,
    float origin_v, float origin_h,
    int num_table_points,
    const float* weights,
    float gamma,
    const float* gains,
    float exposure_weight_gain
) {
    // Allocate temporary planar buffers for one row
    std::vector<float> rRow(width), gRow(width), bRow(width);
    
    // Image area is the full image (0, 0, width, height)
    const int imageAreaL = 0;
    const int imageAreaT = 0;
    const int imageAreaW = width;
    const int imageAreaH = height;
    
    for (int row = 0; row < height; row++) {
        // Deinterleave this row
        for (int col = 0; col < width; col++) {
            int idx = (row * width + col) * 3;
            rRow[col] = rgb[idx + 0];
            gRow[col] = rgb[idx + 1];
            bRow[col] = rgb[idx + 2];
        }
        
        // Process using exact SDK function
        RefBaselineProfileGainTableMap(
            rRow.data(), gRow.data(), bRow.data(),  // src
            rRow.data(), gRow.data(), bRow.data(),  // dst (in-place)
            width,      // cols
            row,        // top
            0,          // left
            imageAreaL, imageAreaT, imageAreaW, imageAreaH,
            exposure_weight_gain,
            points_v, points_h,
            spacing_v, spacing_h,
            origin_v, origin_h,
            num_table_points,
            weights,
            gamma,
            gains,
            false  // supportOverrange = false for standard rendering
        );
        
        // Interleave back
        for (int col = 0; col < width; col++) {
            int idx = (row * width + col) * 3;
            rgb[idx + 0] = rRow[col];
            rgb[idx + 1] = gRow[col];
            rgb[idx + 2] = bRow[col];
        }
    }
}

// Python wrapper for ProfileGainTableMap
static PyObject* dng_color_apply_profile_gain_table_map(PyObject* self, PyObject* args) {
    PyArrayObject* rgb_array = NULL;
    PyArrayObject* gains_array = NULL;
    PyArrayObject* weights_array = NULL;
    int points_v, points_h, num_table_points;
    double spacing_v, spacing_h, origin_v, origin_h;
    double gamma;
    double baseline_exposure;
    
    if (!PyArg_ParseTuple(args, "O!O!O!iiddddidd",
            &PyArray_Type, &rgb_array,
            &PyArray_Type, &gains_array,
            &PyArray_Type, &weights_array,
            &points_v, &points_h,
            &spacing_v, &spacing_h,
            &origin_v, &origin_h,
            &num_table_points,
            &gamma,
            &baseline_exposure)) {
        return NULL;
    }
    
    // Validate input
    if (PyArray_NDIM(rgb_array) != 3 || PyArray_DIM(rgb_array, 2) != 3) {
        PyErr_SetString(PyExc_ValueError, "rgb must be shape (H, W, 3)");
        return NULL;
    }
    if (PyArray_TYPE(rgb_array) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "rgb must be float32");
        return NULL;
    }
    if (PyArray_NDIM(gains_array) != 3) {
        PyErr_SetString(PyExc_ValueError, "gains must be 3D (points_v, points_h, num_table_points)");
        return NULL;
    }
    if (PyArray_SIZE(weights_array) != 5) {
        PyErr_SetString(PyExc_ValueError, "weights must have 5 elements");
        return NULL;
    }
    
    npy_intp height = PyArray_DIM(rgb_array, 0);
    npy_intp width = PyArray_DIM(rgb_array, 1);
    
    // Get contiguous arrays
    PyArrayObject* rgb_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)rgb_array, NPY_FLOAT32, 3, 3);
    PyArrayObject* gains_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)gains_array, NPY_FLOAT32, 3, 3);
    PyArrayObject* weights_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)weights_array, NPY_FLOAT32, 1, 1);
    
    if (!rgb_cont || !gains_cont || !weights_cont) {
        Py_XDECREF(rgb_cont);
        Py_XDECREF(gains_cont);
        Py_XDECREF(weights_cont);
        return NULL;
    }
    
    // Copy RGB for output
    PyObject* result = PyArray_NewCopy(rgb_cont, NPY_CORDER);
    if (!result) {
        Py_DECREF(rgb_cont);
        Py_DECREF(gains_cont);
        Py_DECREF(weights_cont);
        return NULL;
    }
    
    float* result_data = (float*)PyArray_DATA((PyArrayObject*)result);
    const float* gains = (const float*)PyArray_DATA(gains_cont);
    const float* weights = (const float*)PyArray_DATA(weights_cont);
    
    float exposure_weight_gain = powf(2.0f, (float)baseline_exposure);
    
    apply_profile_gain_table_map(
        result_data, (int)height, (int)width,
        points_v, points_h,
        (float)spacing_v, (float)spacing_h,
        (float)origin_v, (float)origin_h,
        num_table_points,
        weights,
        (float)gamma,
        gains,
        exposure_weight_gain
    );
    
    Py_DECREF(rgb_cont);
    Py_DECREF(gains_cont);
    Py_DECREF(weights_cont);
    return result;
}

// Warp rectilinear lens distortion correction (Stage 2)
static PyObject* dng_color_op_warp_rectilinear(PyObject* self, PyObject* args, PyObject* kwargs) {
    static const char* kwlist[] = {
        "rgb", "radial_params", "center_x", "center_y", "tangential_params", "use_bicubic", NULL
    };
    
    PyArrayObject* rgb_array = NULL;
    PyArrayObject* radial_array = NULL;
    PyArrayObject* tangential_array = NULL;
    double center_x = 0.5, center_y = 0.5;
    int use_bicubic = 1;  // default True (SDK uses bicubic)
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!dd|O!p",
            const_cast<char**>(kwlist),
            &PyArray_Type, &rgb_array,
            &PyArray_Type, &radial_array,
            &center_x, &center_y,
            &PyArray_Type, &tangential_array,
            &use_bicubic)) {
        return NULL;
    }
    
    if (PyArray_NDIM(rgb_array) != 3 || PyArray_DIM(rgb_array, 2) != 3) {
        PyErr_SetString(PyExc_ValueError, "rgb must be shape (H, W, 3)");
        return NULL;
    }
    if (PyArray_TYPE(rgb_array) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "rgb must be float32");
        return NULL;
    }
    
    npy_intp height = PyArray_DIM(rgb_array, 0);
    npy_intp width = PyArray_DIM(rgb_array, 1);
    
    PyArrayObject* rgb_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)rgb_array, NPY_FLOAT32, 3, 3);
    // radial_array is 2D: (num_planes, num_coeffs)
    PyArrayObject* radial_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)radial_array, NPY_FLOAT64, 2, 2);
    PyArrayObject* tan_cont = NULL;
    
    if (tangential_array) {
        // tangential_array is 2D: (num_planes, 2)
        tan_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
            (PyObject*)tangential_array, NPY_FLOAT64, 2, 2);
    }
    
    if (!rgb_cont || !radial_cont) {
        Py_XDECREF(rgb_cont);
        Py_XDECREF(radial_cont);
        Py_XDECREF(tan_cont);
        return NULL;
    }
    
    // Get dimensions from 2D radial array
    int num_planes = (int)PyArray_DIM(radial_cont, 0);
    int num_coeffs = (int)PyArray_DIM(radial_cont, 1);
    
    npy_intp dims[3] = {height, width, 3};
    PyObject* result = PyArray_SimpleNew(3, dims, NPY_FLOAT32);
    if (!result) {
        Py_DECREF(rgb_cont);
        Py_DECREF(radial_cont);
        Py_XDECREF(tan_cont);
        return NULL;
    }
    
    const float* src = (const float*)PyArray_DATA(rgb_cont);
    float* dst = (float*)PyArray_DATA((PyArrayObject*)result);
    const double* radial = (const double*)PyArray_DATA(radial_cont);
    const double* tangential = tan_cont ? (const double*)PyArray_DATA(tan_cont) : NULL;
    
    warp_rectilinear(src, dst, height, width, 3, radial, num_planes, num_coeffs, tangential, center_x, center_y, (bool)use_bicubic);
    
    Py_DECREF(rgb_cont);
    Py_DECREF(radial_cont);
    Py_XDECREF(tan_cont);
    return result;
}

// Fix vignette radial (Stage 2)
static PyObject* dng_color_op_fix_vignette(PyObject* self, PyObject* args) {
    PyArrayObject* rgb_array = NULL;
    PyArrayObject* params_array = NULL;
    double center_x = 0.5, center_y = 0.5;
    
    if (!PyArg_ParseTuple(args, "O!O!dd",
            &PyArray_Type, &rgb_array,
            &PyArray_Type, &params_array,
            &center_x, &center_y)) {
        return NULL;
    }
    
    if (PyArray_NDIM(rgb_array) != 3 || PyArray_DIM(rgb_array, 2) != 3) {
        PyErr_SetString(PyExc_ValueError, "rgb must be shape (H, W, 3)");
        return NULL;
    }
    if (PyArray_TYPE(rgb_array) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "rgb must be float32");
        return NULL;
    }
    
    npy_intp height = PyArray_DIM(rgb_array, 0);
    npy_intp width = PyArray_DIM(rgb_array, 1);
    
    PyArrayObject* rgb_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)rgb_array, NPY_FLOAT32, 3, 3);
    PyArrayObject* params_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)params_array, NPY_FLOAT64, 1, 1);
    
    if (!rgb_cont || !params_cont) {
        Py_XDECREF(rgb_cont);
        Py_XDECREF(params_cont);
        return NULL;
    }
    
    // Copy for output
    PyObject* result = PyArray_NewCopy(rgb_cont, NPY_CORDER);
    if (!result) {
        Py_DECREF(rgb_cont);
        Py_DECREF(params_cont);
        return NULL;
    }
    
    float* data = (float*)PyArray_DATA((PyArrayObject*)result);
    const double* params = (const double*)PyArray_DATA(params_cont);
    int num_params = (int)PyArray_SIZE(params_cont);
    
    fix_vignette_radial(data, height, width, 3, params, num_params, center_x, center_y);
    
    Py_DECREF(rgb_cont);
    Py_DECREF(params_cont);
    return result;
}

//=============================================================================
// Bilinear Demosaic (from dng_mosaic_info.cpp dng_bilinear_interpolator)
//
// This is a port of the DNG SDK's bilinear demosaicing algorithm.
// For a 2x2 Bayer CFA pattern, each output pixel needs R, G, B values.
// At positions where that color exists in the CFA, copy directly.
// At other positions, interpolate from neighbors using bilinear weights.
//
// SDK reference: dng_mosaic_info.cpp lines 31-1090, dng_reference.cpp
// RefBilinearRow16/RefBilinearRow32 lines 1293-1385
//=============================================================================

// CFA pattern is passed as a 2x2 array of color plane indices
// SDK ref: dng_mosaic_info::fCFAPattern[row][col] contains plane index
// Color plane indices: 0=Red, 1=Green, 2=Blue (per fCFAPlaneColor)

// Get CFA color at position (row, col) for 2x2 pattern
static inline int get_cfa_color(const int* cfa_colors, int row, int col) {
    int pr = row & 1;
    int pc = col & 1;
    return cfa_colors[pr * 2 + pc];
}

// Bilinear demosaic for 2x2 Bayer pattern
// SDK ref: dng_mosaic_info.cpp dng_bilinear_pattern::Calculate() and
//          dng_bilinear_interpolator::Interpolate()
//
// For each output color plane, the kernel weights for a 2x2 Bayer pattern are:
//
// Red plane (planeColor=0):
//   At R position: copy (weight 1.0)
//   At G position in R row: average of E and W red neighbors
//   At G position in B row: average of N and S red neighbors  
//   At B position: average of 4 diagonal red neighbors (NW, NE, SW, SE)
//
// Blue plane (planeColor=2): same logic but for blue
//
// Green plane (planeColor=1):
//   At G position: copy (weight 1.0)
//   At R or B position: average of 4 adjacent green neighbors (N, S, E, W)
//
// SDK operates on float32 throughout (dng_pixel_buffer with ttFloat)
//
static void bilinear_demosaic_kernel(
    const float* src, float* dst,
    npy_intp height, npy_intp width,
    const int* cfa_colors
) {
    // Process each output pixel
    for (npy_intp row = 0; row < height; row++) {
        for (npy_intp col = 0; col < width; col++) {
            npy_intp dst_idx = (row * width + col) * 3;
            int this_color = get_cfa_color(cfa_colors, row, col);
            
            // For each output color plane (R=0, G=1, B=2)
            for (int plane = 0; plane < 3; plane++) {
                float sum = 0.0f;
                int count = 0;
                
                if (this_color == plane) {
                    // This position has the color we want - copy directly
                    // SDK ref: lines 606-614 "Special case no interpolation case"
                    dst[dst_idx + plane] = src[row * width + col];
                }
                else if (plane == 1) {
                    // Green interpolation at R or B position
                    // SDK ref: lines 634-646 "All sides" - average N,S,E,W
                    // Check N
                    if (row > 0) {
                        sum += src[(row - 1) * width + col];
                        count++;
                    }
                    // Check S
                    if (row < height - 1) {
                        sum += src[(row + 1) * width + col];
                        count++;
                    }
                    // Check W
                    if (col > 0) {
                        sum += src[row * width + (col - 1)];
                        count++;
                    }
                    // Check E
                    if (col < width - 1) {
                        sum += src[row * width + (col + 1)];
                        count++;
                    }
                    dst[dst_idx + plane] = sum / (float)count;
                }
                else {
                    // Red or Blue interpolation
                    // Determine if we're interpolating R at B position or B at R position
                    // or R/B at G position
                    
                    // Find which color is at this position
                    if (this_color == 1) {
                        // Green position - need to interpolate R or B
                        // SDK ref: lines 648-670 "N & S" or "E & W"
                        // Determine if horizontal or vertical neighbors have this color
                        
                        // Check if horizontal neighbors (E,W) have the target color
                        int west_color = (col > 0) ? get_cfa_color(cfa_colors, row, col - 1) : -1;
                        int east_color = (col < width - 1) ? get_cfa_color(cfa_colors, row, col + 1) : -1;
                        
                        if (west_color == plane || east_color == plane) {
                            // Horizontal interpolation
                            if (col > 0 && get_cfa_color(cfa_colors, row, col - 1) == plane) {
                                sum += src[row * width + (col - 1)];
                                count++;
                            }
                            if (col < width - 1 && get_cfa_color(cfa_colors, row, col + 1) == plane) {
                                sum += src[row * width + (col + 1)];
                                count++;
                            }
                        } else {
                            // Vertical interpolation
                            if (row > 0 && get_cfa_color(cfa_colors, row - 1, col) == plane) {
                                sum += src[(row - 1) * width + col];
                                count++;
                            }
                            if (row < height - 1 && get_cfa_color(cfa_colors, row + 1, col) == plane) {
                                sum += src[(row + 1) * width + col];
                                count++;
                            }
                        }
                        if (count > 0) {
                            dst[dst_idx + plane] = sum / (float)count;
                        } else {
                            dst[dst_idx + plane] = 0.0f;
                        }
                    }
                    else {
                        // R position needing B, or B position needing R
                        // SDK ref: lines 724-736 "Four corners" - diagonal average
                        if (row > 0 && col > 0) {
                            sum += src[(row - 1) * width + (col - 1)];
                            count++;
                        }
                        if (row > 0 && col < width - 1) {
                            sum += src[(row - 1) * width + (col + 1)];
                            count++;
                        }
                        if (row < height - 1 && col > 0) {
                            sum += src[(row + 1) * width + (col - 1)];
                            count++;
                        }
                        if (row < height - 1 && col < width - 1) {
                            sum += src[(row + 1) * width + (col + 1)];
                            count++;
                        }
                        if (count > 0) {
                            dst[dst_idx + plane] = sum / (float)count;
                        } else {
                            dst[dst_idx + plane] = 0.0f;
                        }
                    }
                }
            }
        }
    }
}

// Python wrapper for bilinear demosaic
// SDK ref: dng_mosaic_info::InterpolateGeneric
//
// Args:
//   cfa: Normalized CFA data, float32, shape (H, W), values in [0,1]
//   cfa_pattern: 2x2 array of color plane indices [row0col0, row0col1, row1col0, row1col1]
//                SDK ref: dng_mosaic_info::fCFAPattern[row][col]
//                Values: 0=Red, 1=Green, 2=Blue (per fCFAPlaneColor)
//
static PyObject* dng_color_bilinear_demosaic(PyObject* self, PyObject* args, PyObject* kwargs) {
    PyArrayObject* cfa_array = NULL;
    PyArrayObject* cfa_pattern_array = NULL;
    
    static char* kwlist[] = {(char*)"cfa", (char*)"cfa_pattern", NULL};
    
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "O!O!", kwlist,
            &PyArray_Type, &cfa_array,
            &PyArray_Type, &cfa_pattern_array)) {
        return NULL;
    }
    
    // Validate CFA pattern array - must be 4 elements (2x2 flattened)
    if (PyArray_SIZE(cfa_pattern_array) != 4) {
        PyErr_SetString(PyExc_ValueError, 
            "cfa_pattern must be 4 elements (2x2 pattern flattened row-major)");
        return NULL;
    }
    
    // Validate CFA array
    if (PyArray_NDIM(cfa_array) != 2) {
        PyErr_SetString(PyExc_ValueError, "cfa must be 2D array (H, W)");
        return NULL;
    }
    if (PyArray_TYPE(cfa_array) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "cfa must be float32");
        return NULL;
    }
    
    npy_intp height = PyArray_DIM(cfa_array, 0);
    npy_intp width = PyArray_DIM(cfa_array, 1);
    
    if (height < 2 || width < 2) {
        PyErr_SetString(PyExc_ValueError, "Image must be at least 2x2");
        return NULL;
    }
    
    // Ensure contiguous input
    PyArrayObject* cfa_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)cfa_array, NPY_FLOAT32, 2, 2);
    if (!cfa_cont) return NULL;
    
    // Get CFA pattern as int array
    PyArrayObject* pattern_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)cfa_pattern_array, NPY_INT32, 1, 2);
    if (!pattern_cont) {
        Py_DECREF(cfa_cont);
        return NULL;
    }
    
    const int32_t* pattern_data = (const int32_t*)PyArray_DATA(pattern_cont);
    int cfa_colors[4] = {pattern_data[0], pattern_data[1], pattern_data[2], pattern_data[3]};
    
    // Validate color indices (must be 0, 1, or 2)
    for (int i = 0; i < 4; i++) {
        if (cfa_colors[i] < 0 || cfa_colors[i] > 2) {
            Py_DECREF(cfa_cont);
            Py_DECREF(pattern_cont);
            PyErr_SetString(PyExc_ValueError, 
                "cfa_pattern values must be 0 (Red), 1 (Green), or 2 (Blue)");
            return NULL;
        }
    }
    
    // Create output array (float32, same as SDK)
    npy_intp dims[3] = {height, width, 3};
    PyObject* result = PyArray_SimpleNew(3, dims, NPY_FLOAT32);
    if (!result) {
        Py_DECREF(cfa_cont);
        Py_DECREF(pattern_cont);
        return NULL;
    }
    
    const float* src_data = (const float*)PyArray_DATA(cfa_cont);
    float* dst_data = (float*)PyArray_DATA((PyArrayObject*)result);
    
    // Run bilinear demosaic
    bilinear_demosaic_kernel(src_data, dst_data, height, width, cfa_colors);
    
    Py_DECREF(cfa_cont);
    Py_DECREF(pattern_cont);
    return result;
}

// =============================================================================
// MapPolynomial opcode (OpcodeList2)
// SDK ref: dng_reference.cpp RefBaselineMapPoly32
// =============================================================================

// Port of RefBaselineMapPoly32 from dng_reference.cpp
// Note: Uses existing Pin_real32() defined above
// Applies polynomial: y = c0 + c1*x + c2*x^2 + ... 
// For negative x, alternates signs on even powers
static void RefBaselineMapPoly32(
    float* dPtr,
    const int32_t rowStep,
    const uint32_t rows,
    const uint32_t cols,
    const uint32_t rowPitch,
    const uint32_t colPitch,
    const float* coefficients,
    const uint32_t degree,
    uint16_t blackLevel)
{
    float blackScale1 = 1.0f;
    float blackScale2 = 1.0f;
    float blackOffset1 = 0.0f;
    float blackOffset2 = 0.0f;

    if (blackLevel != 0) {
        blackOffset2 = ((float)blackLevel) / 65535.0f;
        blackScale2 = 1.0f - blackOffset2;
        blackScale1 = (blackScale2 != 0.0f) ? 1.0f / blackScale2 : 0.0f;
        blackOffset1 = 1.0f - blackScale1;
    }

    for (uint32_t row = 0; row < rows; row += rowPitch) {
        
        if (blackLevel != 0) {
            for (uint32_t col = 0; col < cols; col += colPitch) {
                dPtr[col] = dPtr[col] * blackScale1 + blackOffset1;
            }
        }

        switch (degree) {
            case 0: {
                float y = Pin_real32(-1.0f, coefficients[0], 1.0f);
                for (uint32_t col = 0; col < cols; col += colPitch) {
                    dPtr[col] = y;
                }
                break;
            }

            case 1: {
                for (uint32_t col = 0; col < cols; col += colPitch) {
                    float x = dPtr[col];
                    float y = coefficients[0] + x * coefficients[1];
                    dPtr[col] = Pin_real32(-1.0f, y, 1.0f);
                }
                break;
            }

            case 2: {
                for (uint32_t col = 0; col < cols; col += colPitch) {
                    float x = dPtr[col];
                    float y;
                    if (x < 0.0f) {
                        y = coefficients[0] + x * (coefficients[1] - x * coefficients[2]);
                    } else {
                        y = coefficients[0] + x * (coefficients[1] + x * coefficients[2]);
                    }
                    dPtr[col] = Pin_real32(-1.0f, y, 1.0f);
                }
                break;
            }

            case 3: {
                for (uint32_t col = 0; col < cols; col += colPitch) {
                    float x = dPtr[col];
                    float y;
                    if (x < 0.0f) {
                        y = coefficients[0] + x * (coefficients[1] - x * (coefficients[2] - x * coefficients[3]));
                    } else {
                        y = coefficients[0] + x * (coefficients[1] + x * (coefficients[2] + x * coefficients[3]));
                    }
                    dPtr[col] = Pin_real32(-1.0f, y, 1.0f);
                }
                break;
            }

            case 4: {
                for (uint32_t col = 0; col < cols; col += colPitch) {
                    float x = dPtr[col];
                    float y;
                    if (x < 0.0f) {
                        y = coefficients[0] + x * (coefficients[1] - x * (coefficients[2] - x * (coefficients[3] - x * coefficients[4])));
                    } else {
                        y = coefficients[0] + x * (coefficients[1] + x * (coefficients[2] + x * (coefficients[3] + x * coefficients[4])));
                    }
                    dPtr[col] = Pin_real32(-1.0f, y, 1.0f);
                }
                break;
            }

            default: {
                for (uint32_t col = 0; col < cols; col += colPitch) {
                    float x = dPtr[col];
                    float y = coefficients[0];

                    if (x < 0.0f) {
                        x = -x;
                        float xx = x;
                        for (uint32_t j = 1; j <= degree; j++) {
                            y -= coefficients[j] * xx;
                            xx *= x;
                        }
                    } else {
                        float xx = x;
                        for (uint32_t j = 1; j <= degree; j++) {
                            y += coefficients[j] * xx;
                            xx *= x;
                        }
                    }
                    dPtr[col] = Pin_real32(-1.0f, y, 1.0f);
                }
                break;
            }
        }

        if (blackLevel != 0) {
            for (uint32_t col = 0; col < cols; col += colPitch) {
                dPtr[col] = dPtr[col] * blackScale2 + blackOffset2;
            }
        }

        dPtr += rowStep;
    }
}

// ============================================================================
// MapPolynomial opcode - shared implementation
// ============================================================================

// Shared core implementation for MapPolynomial opcode
static void apply_map_polynomial_impl(
    float* data, int height, int width, int num_channels,
    int top, int left, int bottom, int right,
    int start_plane, int num_planes,
    int row_pitch, int col_pitch,
    const float* coefficients, int degree
) {
    // Handle area bounds (0 means full image per SDK)
    if (top == 0 && bottom == 0) { top = 0; bottom = height; }
    if (left == 0 && right == 0) { left = 0; right = width; }
    
    int area_height = bottom - top;
    int area_width = right - left;
    
    for (int p = start_plane; p < start_plane + num_planes && p < num_channels; p++) {
        // Get pointer to start of area for this plane
        float* plane_ptr = data + (top * width + left) * num_channels + p;
        
        // rowStep: elements to advance per row (width * num_channels)
        int32_t rowStep = (int32_t)(width * num_channels * row_pitch);
        
        // cols: total elements in row (area_width * num_channels)
        // colPitch: skip between same-plane values (num_channels * col_pitch)
        RefBaselineMapPoly32(
            plane_ptr,
            rowStep,
            (uint32_t)area_height,
            (uint32_t)(area_width * num_channels),
            (uint32_t)row_pitch,
            (uint32_t)(col_pitch * num_channels),
            coefficients,
            (uint32_t)degree,
            0  // blackLevel - stage 2 is already normalized
        );
    }
}

// MapPolynomial opcode for RGB data (OpcodeList2)
static PyObject* dng_color_op_map_polynomial(PyObject* self, PyObject* args) {
    PyObject* rgb_obj;
    PyObject* coeffs_obj;
    int top, left, bottom, right;
    int plane, planes, row_pitch, col_pitch;
    int degree;
    
    if (!PyArg_ParseTuple(args, "OOiiiiiiiii",
            &rgb_obj, &coeffs_obj,
            &top, &left, &bottom, &right,
            &plane, &planes, &row_pitch, &col_pitch,
            &degree)) {
        return NULL;
    }
    
    // Get contiguous arrays
    PyObject* rgb_cont = PyArray_ContiguousFromAny(rgb_obj, NPY_FLOAT32, 3, 3);
    PyObject* coeffs_cont = PyArray_ContiguousFromAny(coeffs_obj, NPY_FLOAT32, 1, 1);
    if (!rgb_cont || !coeffs_cont) {
        Py_XDECREF(rgb_cont);
        Py_XDECREF(coeffs_cont);
        return NULL;
    }
    
    npy_intp* dims = PyArray_DIMS((PyArrayObject*)rgb_cont);
    int height = (int)dims[0];
    int width = (int)dims[1];
    
    // Create output (copy input)
    PyObject* result = PyArray_Copy((PyArrayObject*)rgb_cont);
    if (!result) {
        Py_DECREF(rgb_cont);
        Py_DECREF(coeffs_cont);
        return NULL;
    }
    
    float* dst_data = (float*)PyArray_DATA((PyArrayObject*)result);
    const float* coefficients = (const float*)PyArray_DATA((PyArrayObject*)coeffs_cont);
    
    apply_map_polynomial_impl(dst_data, height, width, 3,
                              top, left, bottom, right,
                              plane, planes, row_pitch, col_pitch,
                              coefficients, degree);
    
    Py_DECREF(rgb_cont);
    Py_DECREF(coeffs_cont);
    return result;
}

// MapPolynomial opcode for CFA data (OpcodeList2, pre-demosaic)
static PyObject* dng_color_op_map_polynomial_cfa(PyObject* self, PyObject* args) {
    PyObject* cfa_obj;
    PyObject* coeffs_obj;
    int top, left, bottom, right;
    int row_pitch, col_pitch;
    int degree;
    
    if (!PyArg_ParseTuple(args, "OOiiiiiiii",
            &cfa_obj, &coeffs_obj,
            &top, &left, &bottom, &right,
            &row_pitch, &col_pitch,
            &degree)) {
        return NULL;
    }
    
    // Get contiguous arrays
    PyObject* cfa_cont = PyArray_ContiguousFromAny(cfa_obj, NPY_FLOAT32, 2, 2);
    PyObject* coeffs_cont = PyArray_ContiguousFromAny(coeffs_obj, NPY_FLOAT32, 1, 1);
    if (!cfa_cont || !coeffs_cont) {
        Py_XDECREF(cfa_cont);
        Py_XDECREF(coeffs_cont);
        return NULL;
    }
    
    npy_intp* dims = PyArray_DIMS((PyArrayObject*)cfa_cont);
    int height = (int)dims[0];
    int width = (int)dims[1];
    
    // Create output (copy input)
    PyObject* result = PyArray_Copy((PyArrayObject*)cfa_cont);
    if (!result) {
        Py_DECREF(cfa_cont);
        Py_DECREF(coeffs_cont);
        return NULL;
    }
    
    float* dst_data = (float*)PyArray_DATA((PyArrayObject*)result);
    const float* coefficients = (const float*)PyArray_DATA((PyArrayObject*)coeffs_cont);
    
    apply_map_polynomial_impl(dst_data, height, width, 1,
                              top, left, bottom, right,
                              0, 1, row_pitch, col_pitch,
                              coefficients, degree);
    
    Py_DECREF(cfa_cont);
    Py_DECREF(coeffs_cont);
    return result;
}

// ============================================================================
// GainMap opcode - Direct copy from DNG SDK dng_gain_map.cpp
// ============================================================================

// dng_gain_map_interpolator - copied from SDK lines 23-249
class dng_gain_map_interpolator {
private:
    const float* fGainValues;
    npy_intp fPointsV;
    npy_intp fPointsH;
    npy_intp fMapPlanes;
    double fOriginV;
    double fOriginH;
    double fSpacingV;
    double fSpacingH;
    
    double fScaleV;
    double fScaleH;
    double fOffsetV;
    double fOffsetH;
    
    int32_t fColumn;
    uint32_t fPlane;
    
    uint32_t fRowIndex1;
    uint32_t fRowIndex2;
    float fRowFract;
    
    int32_t fResetColumn;
    
    float fValueBase;
    float fValueStep;
    float fValueIndex;
    
    float Entry(uint32_t row, uint32_t col, uint32_t plane) const {
        return fGainValues[row * fPointsH * fMapPlanes + col * fMapPlanes + plane];
    }
    
    float InterpolateEntry(uint32_t colIndex) {
        return Entry(fRowIndex1, colIndex, fPlane) * (1.0f - fRowFract) +
               Entry(fRowIndex2, colIndex, fPlane) * (       fRowFract);
    }
    
    void ResetColumn() {
        double colIndexF = ((fScaleH * (fColumn + fOffsetH)) - 
                            fOriginH) / fSpacingH;
        
        if (colIndexF <= 0.0) {
            fValueBase = InterpolateEntry(0);
            fValueStep = 0.0f;
            fResetColumn = (int32_t)ceil(fOriginH / fScaleH - fOffsetH);
        } else {
            uint32_t lastCol = static_cast<uint32_t>(fPointsH - 1);
            
            if (colIndexF >= static_cast<double>(lastCol)) {
                fValueBase = InterpolateEntry(lastCol);
                fValueStep = 0.0f;
                fResetColumn = 0x7FFFFFFF;
            } else {
                uint32_t colIndex = static_cast<uint32_t>(colIndexF);
                double base = InterpolateEntry(colIndex);
                double delta = InterpolateEntry(colIndex + 1) - base;
                
                fValueBase = (float)(base + delta * (colIndexF - (double)colIndex));
                fValueStep = (float)((delta * fScaleH) / fSpacingH);
                fResetColumn = (int32_t)ceil(((colIndex + 1) * fSpacingH +
                                              fOriginH) / fScaleH - fOffsetH);
            }
        }
        fValueIndex = 0.0f;
    }
    
public:
    dng_gain_map_interpolator(const float* gainValues,
                              npy_intp pointsV, npy_intp pointsH, npy_intp mapPlanes,
                              double originV, double originH,
                              double spacingV, double spacingH,
                              npy_intp imageBoundsT, npy_intp imageBoundsL,
                              npy_intp imageBoundsH, npy_intp imageBoundsW,
                              int32_t row, int32_t column, uint32_t plane)
        : fGainValues(gainValues)
        , fPointsV(pointsV), fPointsH(pointsH), fMapPlanes(mapPlanes)
        , fOriginV(originV), fOriginH(originH)
        , fSpacingV(spacingV), fSpacingH(spacingH)
        , fScaleV(1.0 / imageBoundsH)
        , fScaleH(1.0 / imageBoundsW)
        , fOffsetV(0.5 - imageBoundsT)
        , fOffsetH(0.5 - imageBoundsL)
        , fColumn(column)
        , fPlane(plane)
        , fRowIndex1(0), fRowIndex2(0), fRowFract(0.0f)
        , fResetColumn(0)
        , fValueBase(0.0f), fValueStep(0.0f), fValueIndex(0.0f)
    {
        double rowIndexF = (fScaleV * (row + fOffsetV) -
                            fOriginV) / fSpacingV;
        
        if (rowIndexF <= 0.0) {
            fRowIndex1 = 0;
            fRowIndex2 = 0;
            fRowFract = 0.0f;
        } else {
            uint32_t lastRow = static_cast<uint32_t>(fPointsV - 1);
            
            if (rowIndexF >= static_cast<double>(lastRow)) {
                fRowIndex1 = lastRow;
                fRowIndex2 = fRowIndex1;
                fRowFract = 0.0f;
            } else {
                fRowIndex1 = static_cast<uint32_t>(rowIndexF);
                fRowIndex2 = fRowIndex1 + 1;
                fRowFract = (float)(rowIndexF - (double)fRowIndex1);
            }
        }
        ResetColumn();
    }
    
    float Interpolate() const {
        return fValueBase + fValueStep * fValueIndex;
    }
    
    void Increment() {
        if (++fColumn >= fResetColumn) {
            ResetColumn();
        } else {
            fValueIndex += 1.0f;
        }
    }
};

// ============================================================================
// GainMap opcode - shared implementation
// SDK ref: dng_gain_map.cpp dng_opcode_GainMap::ProcessArea
// ============================================================================

// Shared core implementation for GainMap opcode
// is_cfa: true for 2D CFA data, false for 3D RGB data
static void apply_gain_map_impl(
    float* data, npy_intp height, npy_intp width, int num_channels,
    const float* gain_values,
    npy_intp points_v, npy_intp points_h, npy_intp map_planes,
    int32_t top, int32_t left, int32_t bottom, int32_t right,
    int row_pitch, int col_pitch,
    double spacing_v, double spacing_h, double origin_v, double origin_h
) {
    // Handle area bounds (0 means full image) - SDK dng_rect overlap
    int32_t overlap_t = std::max(0, top);
    int32_t overlap_l = std::max(0, left);
    int32_t overlap_b = (bottom > 0) ? std::min((int32_t)height, bottom) : (int32_t)height;
    int32_t overlap_r = (right > 0) ? std::min((int32_t)width, right) : (int32_t)width;
    
    if (overlap_t >= overlap_b || overlap_l >= overlap_r) {
        return;
    }
    
    uint32_t cols = overlap_r - overlap_l;
    uint32_t colPitch = std::min((uint32_t)col_pitch, cols);
    uint32_t mapPlane = 0;
    
    // SDK: for row in overlap.t to overlap.b by rowPitch
    for (int32_t row = overlap_t; row < overlap_b; row += row_pitch) {
        
        // SDK: dng_gain_map_interpolator interp(*fGainMap, imageBounds, row, overlap.l, mapPlane)
        dng_gain_map_interpolator interp(
            gain_values, points_v, points_h, map_planes,
            origin_v, origin_h, spacing_v, spacing_h,
            0, 0, height, width,  // imageBounds: t=0, l=0, h=height, w=width
            row, overlap_l, mapPlane
        );
        
        // SDK: for col in 0 to cols by colPitch
        for (uint32_t col = 0; col < cols; col += colPitch) {
            float gain = interp.Interpolate();
            
            // Apply gain to pixel at (row, overlap_l + col)
            npy_intp pixel_idx = (row * width + overlap_l + col) * num_channels;
            
            for (int c = 0; c < num_channels; c++) {
                data[pixel_idx + c] = std::min(data[pixel_idx + c] * gain, 1.0f);
            }
            
            for (uint32_t j = 0; j < colPitch; j++) {
                interp.Increment();
            }
        }
    }
}

// GainMap opcode for RGB data (OpcodeList2)
static PyObject* dng_color_op_gain_map(PyObject* self, PyObject* args) {
    PyArrayObject* rgb_array = NULL;
    PyArrayObject* gain_array = NULL;
    int top, left, bottom, right;
    int start_plane, num_planes, row_pitch, col_pitch;
    double spacing_v, spacing_h, origin_v, origin_h;
    
    if (!PyArg_ParseTuple(args, "O!O!iiiiiiiidddd",
            &PyArray_Type, &rgb_array,
            &PyArray_Type, &gain_array,
            &top, &left, &bottom, &right,
            &start_plane, &num_planes, &row_pitch, &col_pitch,
            &spacing_v, &spacing_h, &origin_v, &origin_h)) {
        return NULL;
    }
    
    if (PyArray_NDIM(rgb_array) != 3 || PyArray_DIM(rgb_array, 2) != 3) {
        PyErr_SetString(PyExc_ValueError, "rgb must be shape (H, W, 3)");
        return NULL;
    }
    if (PyArray_TYPE(rgb_array) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "rgb must be float32");
        return NULL;
    }
    if (PyArray_NDIM(gain_array) != 3) {
        PyErr_SetString(PyExc_ValueError, "gain_values must be shape (points_v, points_h, planes)");
        return NULL;
    }
    
    npy_intp height = PyArray_DIM(rgb_array, 0);
    npy_intp width = PyArray_DIM(rgb_array, 1);
    npy_intp points_v = PyArray_DIM(gain_array, 0);
    npy_intp points_h = PyArray_DIM(gain_array, 1);
    npy_intp map_planes = PyArray_DIM(gain_array, 2);
    
    PyArrayObject* rgb_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)rgb_array, NPY_FLOAT32, 3, 3);
    PyArrayObject* gain_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)gain_array, NPY_FLOAT32, 3, 3);
    
    if (!rgb_cont || !gain_cont) {
        Py_XDECREF(rgb_cont);
        Py_XDECREF(gain_cont);
        return NULL;
    }
    
    PyObject* result = PyArray_NewCopy(rgb_cont, NPY_CORDER);
    if (!result) {
        Py_DECREF(rgb_cont);
        Py_DECREF(gain_cont);
        return NULL;
    }
    
    float* data = (float*)PyArray_DATA((PyArrayObject*)result);
    const float* gain_values = (const float*)PyArray_DATA(gain_cont);
    
    apply_gain_map_impl(data, height, width, 3,
                        gain_values, points_v, points_h, map_planes,
                        top, left, bottom, right,
                        row_pitch, col_pitch,
                        spacing_v, spacing_h, origin_v, origin_h);
    
    Py_DECREF(rgb_cont);
    Py_DECREF(gain_cont);
    return result;
}

// GainMap opcode for CFA data (OpcodeList2, pre-demosaic)
static PyObject* dng_color_op_gain_map_cfa(PyObject* self, PyObject* args) {
    PyArrayObject* cfa_array = NULL;
    PyArrayObject* gain_array = NULL;
    int top, left, bottom, right;
    int row_pitch, col_pitch;
    double spacing_v, spacing_h, origin_v, origin_h;
    
    if (!PyArg_ParseTuple(args, "O!O!iiiiiidddd",
            &PyArray_Type, &cfa_array,
            &PyArray_Type, &gain_array,
            &top, &left, &bottom, &right,
            &row_pitch, &col_pitch,
            &spacing_v, &spacing_h, &origin_v, &origin_h)) {
        return NULL;
    }
    
    if (PyArray_NDIM(cfa_array) != 2) {
        PyErr_SetString(PyExc_ValueError, "cfa must be shape (H, W)");
        return NULL;
    }
    if (PyArray_TYPE(cfa_array) != NPY_FLOAT32) {
        PyErr_SetString(PyExc_TypeError, "cfa must be float32");
        return NULL;
    }
    if (PyArray_NDIM(gain_array) != 3) {
        PyErr_SetString(PyExc_ValueError, "gain_values must be shape (points_v, points_h, planes)");
        return NULL;
    }
    
    npy_intp height = PyArray_DIM(cfa_array, 0);
    npy_intp width = PyArray_DIM(cfa_array, 1);
    npy_intp points_v = PyArray_DIM(gain_array, 0);
    npy_intp points_h = PyArray_DIM(gain_array, 1);
    npy_intp map_planes = PyArray_DIM(gain_array, 2);
    
    PyArrayObject* cfa_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)cfa_array, NPY_FLOAT32, 2, 2);
    PyArrayObject* gain_cont = (PyArrayObject*)PyArray_ContiguousFromAny(
        (PyObject*)gain_array, NPY_FLOAT32, 3, 3);
    
    if (!cfa_cont || !gain_cont) {
        Py_XDECREF(cfa_cont);
        Py_XDECREF(gain_cont);
        return NULL;
    }
    
    PyObject* result = PyArray_NewCopy(cfa_cont, NPY_CORDER);
    if (!result) {
        Py_DECREF(cfa_cont);
        Py_DECREF(gain_cont);
        return NULL;
    }
    
    float* data = (float*)PyArray_DATA((PyArrayObject*)result);
    const float* gain_values = (const float*)PyArray_DATA(gain_cont);
    
    apply_gain_map_impl(data, height, width, 1,
                        gain_values, points_v, points_h, map_planes,
                        top, left, bottom, right,
                        row_pitch, col_pitch,
                        spacing_v, spacing_h, origin_v, origin_h);
    
    Py_DECREF(cfa_cont);
    Py_DECREF(gain_cont);
    return result;
}

// Module method definitions
static PyMethodDef DngColorMethods[] = {
    {"apply_hue_sat_map", dng_color_apply_hue_sat_map, METH_VARARGS,
     "Apply HueSatMap (3D LUT) to RGB image for camera profile color adjustments.\n\n"
     "Args:\n"
     "    rgb (ndarray): Input RGB image, float32, shape (H, W, 3)\n"
     "    hue_sat_map (ndarray): HueSatMap data, float32, shape (V, H, S, 3) or flattened\n"
     "        Each entry contains (hue_shift_degrees, sat_scale, val_scale)\n"
     "    hue_divs (int): Number of hue divisions in the map\n"
     "    sat_divs (int): Number of saturation divisions in the map\n"
     "    val_divs (int): Number of value divisions in the map\n\n"
     "Returns:\n"
     "    ndarray: Processed RGB image with HueSatMap adjustments applied"},
    
    {"apply_tone_curve", dng_color_apply_tone_curve, METH_VARARGS,
     "Apply a custom tone curve (ProfileToneCurve) to RGB image.\n\n"
     "Args:\n"
     "    rgb (ndarray): Input RGB image, float32, shape (H, W, 3)\n"
     "    curve (ndarray): 1D tone curve LUT, float32, maps [0,1] -> [0,1]\n\n"
     "Returns:\n"
     "    ndarray: Tone-mapped RGB image"},
    
    {"apply_curve", dng_color_apply_curve, METH_VARARGS,
     "Apply simple per-pixel curve via LUT interpolation.\n\n"
     "Each channel processed independently (no hue preservation).\n\n"
     "Args:\n"
     "    rgb (ndarray): Input RGB image, float32, shape (H, W, 3)\n"
     "    curve (ndarray): 1D curve LUT, float32, maps [0,1] -> [0,1]\n\n"
     "Returns:\n"
     "    ndarray: Curve-mapped RGB image"},
    
    {"apply_curve_hue_preserving", dng_color_apply_curve_hue_preserving, METH_VARARGS,
     "Apply hue-preserving RGB curve (RefBaselineRGBTone).\n\n"
     "This preserves color relationships by applying the curve to max/min\n"
     "channels and interpolating the middle channel.\n\n"
     "Args:\n"
     "    rgb (ndarray): Input RGB image, float32, shape (H, W, 3)\n"
     "    curve (ndarray): 1D curve LUT, float32, maps [0,1] -> [0,1]\n\n"
     "Returns:\n"
     "    ndarray: Hue-preserving curve-mapped RGB image"},
    
    {"apply_exposure_ramp", dng_color_apply_exposure_ramp, METH_VARARGS,
     "Apply exposure ramp function (dng_function_exposure_ramp).\n\n"
     "SDK ref: dng_render.cpp lines 50-103\n"
     "3 regions: below black-radius=0, above black+radius=linear, between=quadratic\n\n"
     "Args:\n"
     "    rgb (ndarray): Input RGB image, float32, shape (H, W, 3)\n"
     "    white (float): White point (1.0 / pow(2, max(0, exposure)))\n"
     "    black (float): Black point (shadows * shadowScale * 0.001)\n"
     "    minBlack (float): Minimum black for radius calculation\n"
     "    supportOverrange (bool, optional): Allow values > 1.0\n\n"
     "Returns:\n"
     "    ndarray: Exposure-adjusted RGB image"},
    
    {"srgb_gamma", dng_color_srgb_gamma, METH_VARARGS,
     "Apply sRGB gamma encoding (linear to sRGB).\n\n"
     "Args:\n"
     "    rgb (ndarray): Input linear RGB image, float32, shape (H, W, 3)\n\n"
     "Returns:\n"
     "    ndarray: sRGB gamma-encoded image"},
    
    {"matrix_transform", (PyCFunction)dng_color_matrix_transform, METH_VARARGS | METH_KEYWORDS,
     "Apply 3x3 color matrix transform to RGB image.\n\n"
     "Args:\n"
     "    rgb (ndarray): Input RGB image, float32, shape (H, W, 3)\n"
     "    matrix (ndarray): 3x3 color transform matrix, float32\n"
     "    clip (bool): Clip output to [0,1] (default: True)\n\n"
     "Returns:\n"
     "    ndarray: Transformed RGB image"},
    
    {"linearize", dng_color_linearize, METH_VARARGS,
     "Apply linearization table to RAW sensor data (Stage 1).\n\n"
     "Converts non-linear sensor ADC values to linear light values.\n\n"
     "Args:\n"
     "    data (ndarray): RAW sensor data, float32\n"
     "    table (ndarray): Linearization LUT, float32\n"
     "    max_val (float): Maximum input value (e.g., 16383 for 14-bit)\n\n"
     "Returns:\n"
     "    ndarray: Linearized data"},
    
    {"normalize_raw", (PyCFunction)dng_color_normalize_raw, METH_VARARGS | METH_KEYWORDS,
     "Normalize RAW data using black and white levels per DNG spec Chapter 5.\n\n"
     "Implements: linear = (raw - BlackLevel[r%rR][c%rC][s] - DeltaH[c] - DeltaV[r]) / (WhiteLevel[s] - BlackLevel)\n\n"
     "Args:\n"
     "    data (ndarray): RAW pixel data, float32, (H,W) or (H,W,samples_per_pixel)\n"
     "    black_level (ndarray): BlackLevel pattern, float32, flattened in row-col-sample order\n"
     "    black_repeat_rows (int): Number of rows in repeating pattern (from BlackLevelRepeatDim[0])\n"
     "    black_repeat_cols (int): Number of cols in repeating pattern (from BlackLevelRepeatDim[1])\n"
     "    samples_per_pixel (int): 1 for CFA, 3 for LinearRaw\n"
     "    white_level (ndarray): WhiteLevel per sample, float32\n"
     "    black_delta_h (ndarray, optional): Per-column delta, float32, shape (width,)\n"
     "    black_delta_v (ndarray, optional): Per-row delta, float32, shape (height,)\n\n"
     "Returns:\n"
     "    ndarray: Normalized data in [0,1] range"},
    
    {"apply_gain_map", dng_color_apply_gain_map, METH_VARARGS,
     "Apply gain map (flat-field correction) to RAW CFA data (Stage 1).\n\n"
     "Multiplies each CFA pixel by interpolated gain value.\n\n"
     "Args:\n"
     "    data (ndarray): RAW CFA data, float32, (H,W)\n"
     "    gain_map (ndarray): 2D gain map, float32\n\n"
     "Returns:\n"
     "    ndarray: Gain-corrected CFA data"},
    
    {"apply_profile_gain_table_map", dng_color_apply_profile_gain_table_map, METH_VARARGS,
     "Apply ProfileGainTableMap to RGB image (Stage 3).\n\n"
     "SDK ref: dng_reference.cpp RefBaselineProfileGainTableMap()\n"
     "Applied after HueSatMap, before exposure ramp.\n\n"
     "Args:\n"
     "    rgb (ndarray): RGB image, float32, (H,W,3)\n"
     "    gains (ndarray): Gain table, float32, (points_v, points_h, num_table_points)\n"
     "    weights (ndarray): MapInputWeights, float32, (5,)\n"
     "    points_v (int): Map height\n"
     "    points_h (int): Map width\n"
     "    spacing_v (float): Vertical spacing\n"
     "    spacing_h (float): Horizontal spacing\n"
     "    origin_v (float): Vertical origin\n"
     "    origin_h (float): Horizontal origin\n"
     "    num_table_points (int): Table depth\n"
     "    gamma (float): Gamma parameter\n"
     "    baseline_exposure (float): BaselineExposure value\n\n"
     "Returns:\n"
     "    ndarray: RGB with gain applied"},
    
    {"op_warp_rectilinear", (PyCFunction)dng_color_op_warp_rectilinear, METH_VARARGS | METH_KEYWORDS,
     "Apply lens distortion correction using WarpRectilinear opcode (Stage 2).\n\n"
     "Uses polynomial radial model: r_src = r_dst * f(r_dst)\n"
     "where f(r) = k0 + k1*r + k2*r^2 + k3*r^3\n\n"
     "Args:\n"
     "    rgb (ndarray): Input RGB image, float32, (H,W,3)\n"
     "    radial_params (ndarray): Radial polynomial coefficients [k0,k1,k2,k3]\n"
     "    center_x (float): Optical center x in [0,1] (default: 0.5)\n"
     "    center_y (float): Optical center y in [0,1] (default: 0.5)\n"
     "    tangential_params (ndarray): Optional tangential coefficients [kt0,kt1]\n\n"
     "Returns:\n"
     "    ndarray: Distortion-corrected RGB image"},
    
    {"op_fix_vignette", dng_color_op_fix_vignette, METH_VARARGS,
     "Apply radial vignette correction (Stage 2).\n\n"
     "Applies gain = 1 + k0*r^2 + k1*r^4 + k2*r^6 + ...\n\n"
     "Args:\n"
     "    rgb (ndarray): Input RGB image, float32, (H,W,3)\n"
     "    params (ndarray): Vignette polynomial coefficients [k0,k1,k2,...]\n"
     "    center_x (float): Optical center x in [0,1]\n"
     "    center_y (float): Optical center y in [0,1]\n\n"
     "Returns:\n"
     "    ndarray: Vignette-corrected RGB image"},
    
    {"bilinear_demosaic", (PyCFunction)dng_color_bilinear_demosaic, METH_VARARGS | METH_KEYWORDS,
     "Demosaic CFA data using DNG SDK bilinear interpolation.\n\n"
     "This is a port of the DNG SDK's bilinear demosaicing algorithm from\n"
     "dng_mosaic_info.cpp (InterpolateGeneric with dng_bilinear_interpolator).\n"
     "SDK operates on float32 throughout.\n\n"
     "Args:\n"
     "    cfa (ndarray): Normalized CFA data, float32, shape (H, W), values in [0,1]\n"
     "    cfa_pattern (ndarray): 2x2 CFA pattern as color plane indices, int32,\n"
     "        flattened row-major [row0col0, row0col1, row1col0, row1col1].\n"
     "        SDK ref: dng_mosaic_info::fCFAPattern[row][col]\n"
     "        Values: 0=Red, 1=Green, 2=Blue (per fCFAPlaneColor)\n\n"
     "Returns:\n"
     "    ndarray: Demosaiced RGB image, float32, shape (H, W, 3)"},
    
    {"op_map_polynomial", dng_color_op_map_polynomial, METH_VARARGS,
     "Apply MapPolynomial opcode to RGB image (OpcodeList2 Stage 2).\n\n"
     "SDK ref: dng_reference.cpp RefBaselineMapPoly32\n"
     "Applies polynomial: y = c0 + c1*x + c2*x^2 + ...\n"
     "For negative x, alternates signs on even powers.\n\n"
     "Args:\n"
     "    rgb (ndarray): RGB image, float32, (H,W,3)\n"
     "    coefficients (ndarray): Polynomial coefficients, float32\n"
     "    top, left, bottom, right (int): Area bounds (0 = full)\n"
     "    plane (int): First plane to process\n"
     "    planes (int): Number of planes\n"
     "    row_pitch (int): Row stride\n"
     "    col_pitch (int): Column stride\n"
     "    degree (int): Polynomial degree\n\n"
     "Returns:\n"
     "    ndarray: RGB with polynomial applied"},
    
    {"op_gain_map", dng_color_op_gain_map, METH_VARARGS,
     "Apply GainMap opcode to RGB image (OpcodeList2).\n\n"
     "SDK ref: dng_gain_map.cpp dng_opcode_GainMap::ProcessArea\n"
     "Applies bilinearly-interpolated 2D gain map for lens shading correction.\n\n"
     "Args:\n"
     "    rgb (ndarray): RGB image, float32, (H,W,3)\n"
     "    gain_values (ndarray): Gain map, float32, (points_v, points_h, planes)\n"
     "    top, left, bottom, right (int): Area bounds (0 = full)\n"
     "    plane (int): First plane to process\n"
     "    planes (int): Number of planes\n"
     "    row_pitch (int): Row stride\n"
     "    col_pitch (int): Column stride\n"
     "    spacing_v, spacing_h (float): Map sample spacing\n"
     "    origin_v, origin_h (float): Map origin\n\n"
     "Returns:\n"
     "    ndarray: RGB with gain map applied"},
    
    {"op_gain_map_cfa", dng_color_op_gain_map_cfa, METH_VARARGS,
     "Apply GainMap opcode to CFA image (OpcodeList2, pre-demosaic).\n\n"
     "SDK ref: dng_gain_map.cpp dng_opcode_GainMap::ProcessArea\n"
     "Applies bilinearly-interpolated 2D gain map for lens shading correction.\n\n"
     "Args:\n"
     "    cfa (ndarray): CFA image, float32, (H,W)\n"
     "    gain_values (ndarray): Gain map, float32, (points_v, points_h, planes)\n"
     "    top, left, bottom, right (int): Area bounds (0 = full)\n"
     "    row_pitch (int): Row stride\n"
     "    col_pitch (int): Column stride\n"
     "    spacing_v, spacing_h (float): Map sample spacing\n"
     "    origin_v, origin_h (float): Map origin\n\n"
     "Returns:\n"
     "    ndarray: CFA with gain map applied"},
    
    {NULL, NULL, 0, NULL}
};

// Module definition
static struct PyModuleDef raw_render_module = {
    PyModuleDef_HEAD_INIT,
    "_raw_render",
    "DNG SDK color processing (standalone implementation).\n\n"
    "This module provides access to Adobe DNG SDK's color processing algorithms,\n"
    "including color temperature conversion, camera color space transforms,\n"
    "and the ACR3 default tone curve.\n\n"
    "Based on Adobe DNG SDK 1.7.1",
    -1,
    DngColorMethods
};

// Module initialization
PyMODINIT_FUNC PyInit__raw_render(void) {
    import_array();
    return PyModule_Create(&raw_render_module);
}
