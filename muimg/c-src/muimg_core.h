/*
 * muimg_core - Shared types and the library's exported C ABI surface.
 *
 * Public exports from the shared library are only:
 *   - muimg_version
 *   - muimg_execute_graph  (declared in muimg_compute_graph.h)
 *
 * Per-op buffer wrappers live in src/muimg_ops.h (internal; not exported).
 *
 * Copyright (c) 2024 mu-files
 * Licensed under a modified PolyForm Small Business License
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
  size_t channels;  // Number of channels (1 mono/CFA, 3 RGB, 4 RGBA)
  MuImgDType dtype; // Data type
  size_t stride;    // Row stride in bytes (0 = packed/contiguous)
} MuImgBuffer;

//=============================================================================
// Version information
//=============================================================================

MUIMG_API const char *muimg_version(void);

#ifdef __cplusplus
}
#endif

#endif // MUIMG_CORE_H
