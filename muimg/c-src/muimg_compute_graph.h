/*
 * muimg_compute_graph - Full-image compute graph segment API (Phase A)
 *
 * In-memory IR for one engine segment. Python owns affinity splitting and
 * only submits engine nodes here. No Python / NumPy dependencies.
 *
 * Serialization of this IR across the FFI is a later step (binding layer);
 * this header defines the structs the executor runs.
 *
 * Copyright (c) 2026 mu-files
 */

#ifndef MUIMG_COMPUTE_GRAPH_H
#define MUIMG_COMPUTE_GRAPH_H

#ifdef __cplusplus
extern "C" {
#endif

#include "muimg_core.h"

#include <stddef.h>
#include <stdint.h>

//=============================================================================
// Extra return codes (reuse MUIMG_SUCCESS and existing MUIMG_ERROR_* too)
//=============================================================================

#define MUIMG_ERROR_UNKNOWN_OP 10
#define MUIMG_ERROR_GRAPH_INVALID 11
#define MUIMG_ERROR_NOT_IMPLEMENTED 12
#define MUIMG_ERROR_INTERNAL 13

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
 * Full-image op only in Phase A. Op name is a stable string matched by the
 * hard-wired dispatcher (e.g. "sub_scalar", "mul_scalar", "bilinear_demosaic").
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
// Full-frame operator callback (registry / hard-wired dispatch target)
//=============================================================================

/*
 * Contract: read num_inputs buffers, write num_outputs buffers (pre-sized by
 * the executor to match tensor descriptors). Attr arrays are borrowed.
 * Return MUIMG_SUCCESS or an MUIMG_ERROR_* code.
 */
typedef int (*MuImgFullImageOpFn)(const MuImgBuffer *inputs, size_t num_inputs,
                                  MuImgBuffer *outputs, size_t num_outputs,
                                  const MuImgAttr *attrs, size_t num_attrs);

//=============================================================================
// Execute
//=============================================================================

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
 *
 * Phase A.2: implemented for hard-wired full-image ops (see engine sources).
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
