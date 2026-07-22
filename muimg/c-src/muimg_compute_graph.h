/*
 * muimg_compute_graph - Shared library C ABI and compute-graph IR.
 *
 * Public exports from the shared library:
 *   - muimg_version
 *   - muimg_execute_graph
 *
 * Copyright (c) 2024–2026 mu-files
 * Licensed under a modified PolyForm Small Business License
 */

#ifndef MUIMG_COMPUTE_GRAPH_H
#define MUIMG_COMPUTE_GRAPH_H

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

//=============================================================================
// Return codes
//=============================================================================

#define MUIMG_SUCCESS 0
#define MUIMG_ERROR_INVALID_ARGUMENT 1
#define MUIMG_ERROR_UNSUPPORTED_DTYPE 2
#define MUIMG_ERROR_DIMENSION_MISMATCH 3
#define MUIMG_ERROR_OUT_OF_MEMORY 4
#define MUIMG_ERROR_UNKNOWN_OP 10
#define MUIMG_ERROR_GRAPH_INVALID 11
#define MUIMG_ERROR_NOT_IMPLEMENTED 12
#define MUIMG_ERROR_INTERNAL 13

//=============================================================================
// Buffers
//=============================================================================

typedef enum {
  MUIMG_DTYPE_UINT8 = 0,
  MUIMG_DTYPE_UINT16 = 1,
  MUIMG_DTYPE_FLOAT16 = 2,
  MUIMG_DTYPE_FLOAT32 = 3
} MuImgDType;

typedef struct {
  void *data;       // Pointer to raw buffer
  size_t height;    // Image height in pixels
  size_t width;     // Image width in pixels
  size_t channels;  // Number of channels (1 mono/CFA, 3 RGB, 4 RGBA)
  MuImgDType dtype; // Data type
  size_t stride;    // Row stride in bytes (0 = packed/contiguous)
} MuImgBuffer;

//=============================================================================
// Tensor descriptors (geometry only; pixels live in bindings / executor)
//=============================================================================

typedef uint32_t MuImgTensorId;

/*
 * Identifies a value in the graph and reuses MuImgBuffer for dtype / shape.
 *
 * In the graph's tensor_descs list, buffer.data must be NULL (and is ignored).
 * height, width, channels, dtype, and stride describe the value. Pixel storage
 * is supplied later via MuImgGraphBinding (inputs/outputs) or allocated by the
 * executor (intermediates).
 *
 * channels is the layout contract (1 = mono / CFA mosaic, 3 = RGB, 4 = RGBA).
 * CFA pattern and other semantics are op attrs, not fields here.
 */
typedef struct {
  MuImgTensorId id;
  MuImgBuffer buffer;
} MuImgTensorDesc;

//=============================================================================
// Node attributes (POD / borrowed arrays; valid for duration of execute)
//=============================================================================

typedef enum {
  MUIMG_ATTR_NONE = 0,
  MUIMG_ATTR_I32 = 1,
  MUIMG_ATTR_F32 = 2,
  MUIMG_ATTR_F64 = 3,
  MUIMG_ATTR_I32_ARRAY = 4,
  MUIMG_ATTR_F32_ARRAY = 5,
  MUIMG_ATTR_F64_ARRAY = 6,
  MUIMG_ATTR_STRING = 7 /* NUL-terminated; count == 1 */
} MuImgAttrType;

typedef struct {
  const char *key; /* NUL-terminated; e.g. "value", "matrix", "lut" */
  MuImgAttrType type;
  size_t count; /* 1 for scalars/strings; element count for arrays */
  union {
    int32_t i32;
    float f32;
    double f64;
    const int32_t *i32_array;
    const float *f32_array;
    const double *f64_array;
    const char *string;
  } value;
} MuImgAttr;

//=============================================================================
// Graph nodes and segment
//=============================================================================

/*
 * One op node. Op name is a stable string matched by the generated catalog
 * dispatcher (e.g. "sub_scalar", "bilinear_demosaic").
 *
 * inputs/outputs are MuImgTensorId values (not positions in tensor_descs).
 * Pointers are borrowed; valid for the execute call.
 */
typedef struct {
  uint32_t id;
  const char *op;
  const MuImgTensorId *inputs;
  size_t num_inputs;
  const MuImgTensorId *outputs;
  size_t num_outputs;
  const MuImgAttr *attrs;
  size_t num_attrs;
} MuImgGraphNode;

/*
 * One engine segment: all nodes are engine affinity. Graph inputs/outputs
 * list tensor ids bound by the caller via MuImgGraphBinding.
 */
typedef struct {
  const MuImgTensorDesc *tensor_descs;
  size_t num_tensor_descs;
  const MuImgTensorId *inputs;
  size_t num_inputs;
  const MuImgTensorId *outputs;
  size_t num_outputs;
  const MuImgGraphNode *nodes;
  size_t num_nodes;
} MuImgGraph;

typedef struct {
  MuImgTensorId tensor_id;
  MuImgBuffer buffer; /* real storage; geometry must match the MuImgTensorDesc */
} MuImgGraphBinding;

//=============================================================================
// Exported ABI
//=============================================================================

MUIMG_API const char *muimg_version(void);

/*
 * Run one engine segment.
 *
 * - input_bindings: one entry per graph.inputs id (order need not match;
 *   matched by tensor_id). Buffers are borrowed; not freed by the executor.
 * - output_bindings: one entry per graph.outputs id; caller provides
 *   pre-allocated MuImgBuffer storage matching the tensor descriptor. On
 *   success the executor writes pixel data into those buffers.
 *
 * Intermediate values (neither graph inputs nor outputs) are allocated and
 * freed by the executor for the duration of the call.
 *
 * Error model: returns MUIMG_SUCCESS or MUIMG_ERROR_*. Does not throw across
 * this boundary (exceptions from std:: / bugs are caught and mapped to
 * MUIMG_ERROR_OUT_OF_MEMORY or MUIMG_ERROR_INTERNAL). Op callbacks must
 * likewise return error codes and must not throw.
 */
MUIMG_API int muimg_execute_graph(const MuImgGraph *graph,
                                  const MuImgGraphBinding *input_bindings,
                                  size_t num_input_bindings,
                                  MuImgGraphBinding *output_bindings,
                                  size_t num_output_bindings);

#ifdef __cplusplus
}
#endif

#endif /* MUIMG_COMPUTE_GRAPH_H */
