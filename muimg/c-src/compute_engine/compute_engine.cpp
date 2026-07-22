/*
 * muimg._compute_engine - Python binding for muimg_execute_graph
 *
 * Structured Python dict/list IR is converted to in-memory MuImgGraph
 * structs here (no binary blob).
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <numpy/arrayscalars.h>

#include <cstdio>
#include <cstring>
#include <deque>
#include <memory>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#else
#include <dlfcn.h>
#endif

#include "../muimg_core.h"
#include "../muimg_compute_graph.h"
#include "../py_ptr.h"

//=============================================================================
// Core library loading (same pattern as raw_render_ops.cpp)
//=============================================================================

static void *core_dlopen(const char *path) {
#if defined(_WIN32)
  return reinterpret_cast<void *>(LoadLibraryA(path));
#else
  return dlopen(path, RTLD_LAZY);
#endif
}

static void *core_dlsym(void *handle, const char *name) {
#if defined(_WIN32)
  return reinterpret_cast<void *>(GetProcAddress(static_cast<HMODULE>(handle), name));
#else
  return dlsym(handle, name);
#endif
}

static void core_dlclose(void *handle) {
#if defined(_WIN32)
  FreeLibrary(static_cast<HMODULE>(handle));
#else
  dlclose(handle);
#endif
}

static void *g_core_lib = nullptr;
static int (*muimg_execute_graph_fn)(const MuImgGraph *, const MuImgGraphBinding *,
                                     size_t, MuImgGraphBinding *, size_t) = nullptr;

static const char *core_lib_basename() {
#if defined(_WIN32)
  return "muimg_core.windows-amd64.dll";
#elif defined(__APPLE__)
#if defined(__aarch64__) || defined(__arm64__)
  return "libmuimg_core.macos-arm64.dylib";
#else
  return "libmuimg_core.macos-x86_64.dylib";
#endif
#else
#if defined(__aarch64__) || defined(__arm64__)
  return "libmuimg_core.linux-aarch64.so";
#else
  return "libmuimg_core.linux-x86_64.so";
#endif
#endif
}

static bool resolve_core_lib_path(char *out, size_t out_len) {
  const char *basename = core_lib_basename();
  PyObject *mod = PyImport_ImportModule("muimg");
  if (mod) {
    PyObject *file_obj = PyObject_GetAttrString(mod, "__file__");
    Py_DECREF(mod);
    if (file_obj && PyUnicode_Check(file_obj)) {
      const char *package_file = PyUnicode_AsUTF8(file_obj);
      if (package_file) {
        const char *slash = std::strrchr(package_file, '/');
#if defined(_WIN32)
        const char *bslash = std::strrchr(package_file, '\\');
        if (bslash && (!slash || bslash > slash)) {
          slash = bslash;
        }
#endif
        if (slash) {
          int n = std::snprintf(out, out_len, "%.*s/_binaries/%s",
                                (int)(slash - package_file), package_file,
                                basename);
          Py_DECREF(file_obj);
          if (n > 0 && static_cast<size_t>(n) < out_len) {
            return true;
          }
        }
      }
    }
    Py_XDECREF(file_obj);
    PyErr_Clear();
  }
  int n = std::snprintf(out, out_len, "muimg/_binaries/%s", basename);
  return n > 0 && static_cast<size_t>(n) < out_len;
}

static bool load_core_library() {
  if (g_core_lib && muimg_execute_graph_fn) {
    return true;
  }

  char lib_path[4096];
  if (!resolve_core_lib_path(lib_path, sizeof(lib_path))) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to resolve muimg_core library path");
    return false;
  }

  g_core_lib = core_dlopen(lib_path);
  if (!g_core_lib) {
    PyErr_Format(PyExc_RuntimeError, "Failed to load muimg_core library: %s",
                 lib_path);
    return false;
  }

  muimg_execute_graph_fn =
      reinterpret_cast<int (*)(const MuImgGraph *, const MuImgGraphBinding *, size_t,
                               MuImgGraphBinding *, size_t)>(
          core_dlsym(g_core_lib, "muimg_execute_graph"));
  if (!muimg_execute_graph_fn) {
    PyErr_SetString(PyExc_RuntimeError,
                    "Failed to load muimg_execute_graph from muimg_core "
                    "(rebuild/copy a core binary that includes compute_graph)");
    core_dlclose(g_core_lib);
    g_core_lib = nullptr;
    return false;
  }
  return true;
}

static void set_python_error_from_muimg(int code) {
  const char *name = "UNKNOWN";
  switch (code) {
    case MUIMG_ERROR_INVALID_ARGUMENT:
      name = "INVALID_ARGUMENT";
      break;
    case MUIMG_ERROR_UNSUPPORTED_DTYPE:
      name = "UNSUPPORTED_DTYPE";
      break;
    case MUIMG_ERROR_DIMENSION_MISMATCH:
      name = "DIMENSION_MISMATCH";
      break;
    case MUIMG_ERROR_OUT_OF_MEMORY:
      name = "OUT_OF_MEMORY";
      break;
    case MUIMG_ERROR_UNKNOWN_OP:
      name = "UNKNOWN_OP";
      break;
    case MUIMG_ERROR_GRAPH_INVALID:
      name = "GRAPH_INVALID";
      break;
    case MUIMG_ERROR_NOT_IMPLEMENTED:
      name = "NOT_IMPLEMENTED";
      break;
    case MUIMG_ERROR_INTERNAL:
      name = "INTERNAL";
      break;
  }
  PyErr_Format(PyExc_RuntimeError, "muimg_execute_graph failed: %s (%d)", name,
               code);
}

//=============================================================================
// Graph build scratch (keeps owned storage alive for the execute call)
//=============================================================================

struct GraphScratch {
  std::vector<MuImgTensorDesc> descs;
  std::vector<MuImgTensorId> inputs;
  std::vector<MuImgTensorId> outputs;
  std::vector<MuImgGraphNode> nodes;

  /* deque: push_back must not invalidate c_str() pointers into elements. */
  std::deque<std::string> op_names;
  std::deque<std::string> attr_keys;
  std::deque<std::string> attr_strings;
  std::vector<std::vector<MuImgTensorId>> node_ios;
  std::vector<std::vector<MuImgAttr>> node_attrs;
  std::vector<std::vector<float>> f32_arrays;
  std::vector<std::vector<double>> f64_arrays;
  std::vector<std::vector<int32_t>> i32_arrays;
};

static bool parse_dtype(PyObject *obj, MuImgDType &out) {
  if (PyUnicode_Check(obj)) {
    const char *s = PyUnicode_AsUTF8(obj);
    if (!s) {
      return false;
    }
    if (std::strcmp(s, "float32") == 0) {
      out = MUIMG_DTYPE_FLOAT32;
      return true;
    }
    if (std::strcmp(s, "uint8") == 0) {
      out = MUIMG_DTYPE_UINT8;
      return true;
    }
    if (std::strcmp(s, "uint16") == 0) {
      out = MUIMG_DTYPE_UINT16;
      return true;
    }
    if (std::strcmp(s, "float16") == 0) {
      out = MUIMG_DTYPE_FLOAT16;
      return true;
    }
    PyErr_Format(PyExc_ValueError, "unsupported dtype string: %s", s);
    return false;
  }
  if (PyLong_Check(obj)) {
    long v = PyLong_AsLong(obj);
    if (v < 0 || v > 3) {
      PyErr_SetString(PyExc_ValueError, "dtype enum out of range");
      return false;
    }
    out = static_cast<MuImgDType>(v);
    return true;
  }
  PyErr_SetString(PyExc_TypeError, "dtype must be str or int");
  return false;
}

static bool parse_id_list(PyObject *obj, std::vector<MuImgTensorId> &out) {
  if (!PyList_Check(obj) && !PyTuple_Check(obj)) {
    PyErr_SetString(PyExc_TypeError, "expected list/tuple of tensor ids");
    return false;
  }
  Py_ssize_t n = PySequence_Size(obj);
  out.resize(static_cast<size_t>(n));
  for (Py_ssize_t i = 0; i < n; ++i) {
    PyObject *item = PySequence_GetItem(obj, i);
    if (!item) {
      return false;
    }
    if (!PyLong_Check(item)) {
      Py_DECREF(item);
      PyErr_SetString(PyExc_TypeError, "tensor id must be int");
      return false;
    }
    out[static_cast<size_t>(i)] =
        static_cast<MuImgTensorId>(PyLong_AsUnsignedLong(item));
    Py_DECREF(item);
    if (PyErr_Occurred()) {
      return false;
    }
  }
  return true;
}

static bool parse_attr_value(PyObject *key_obj, PyObject *val,
                             GraphScratch &scratch, MuImgAttr &attr) {
  if (!PyUnicode_Check(key_obj)) {
    PyErr_SetString(PyExc_TypeError, "attr key must be str");
    return false;
  }
  scratch.attr_keys.emplace_back(PyUnicode_AsUTF8(key_obj));
  attr.key = scratch.attr_keys.back().c_str();

  if (PyUnicode_Check(val)) {
    const char *s = PyUnicode_AsUTF8(val);
    if (!s) {
      return false;
    }
    scratch.attr_strings.emplace_back(s);
    attr.type = MUIMG_ATTR_STRING;
    attr.count = 1;
    attr.value.string = scratch.attr_strings.back().c_str();
    return true;
  }
  if (PyBool_Check(val)) {
    attr.type = MUIMG_ATTR_I32;
    attr.count = 1;
    attr.value.i32 = (val == Py_True) ? 1 : 0;
    return true;
  }
  /* NumPy scalar floats preserve f32 vs f64 from mc coercion. */
  if (PyArray_IsScalar(val, Float64)) {
    attr.type = MUIMG_ATTR_F64;
    attr.count = 1;
    attr.value.f64 = PyArrayScalar_VAL(val, Float64);
    return true;
  }
  if (PyArray_IsScalar(val, Float32)) {
    attr.type = MUIMG_ATTR_F32;
    attr.count = 1;
    attr.value.f32 = PyArrayScalar_VAL(val, Float32);
    return true;
  }
  if (PyFloat_Check(val)) {
    attr.type = MUIMG_ATTR_F32;
    attr.count = 1;
    attr.value.f32 = static_cast<float>(PyFloat_AsDouble(val));
    return true;
  }
  if (PyLong_Check(val)) {
    attr.type = MUIMG_ATTR_I32;
    attr.count = 1;
    attr.value.i32 = static_cast<int32_t>(PyLong_AsLong(val));
    return !PyErr_Occurred();
  }

  /* Typed 1-D arrays: keep dtype (f64 / f32 / i32). */
  if (PyArray_Check(val)) {
    PyArrayObject *src = reinterpret_cast<PyArrayObject *>(val);
    if (PyArray_NDIM(src) != 1) {
      PyErr_SetString(PyExc_ValueError, "attr array must be 1-D");
      return false;
    }
    const int typenum = PyArray_TYPE(src);
    if (typenum == NPY_FLOAT64) {
      PyObject *obj = PyArray_ContiguousFromAny(val, NPY_FLOAT64, 1, 1);
      if (!obj) {
        return false;
      }
      PyArrayObject *arr = reinterpret_cast<PyArrayObject *>(obj);
      npy_intp n = PyArray_DIM(arr, 0);
      if (n < 1) {
        Py_DECREF(obj);
        PyErr_SetString(PyExc_ValueError, "attr array must be non-empty");
        return false;
      }
      scratch.f64_arrays.emplace_back(
          static_cast<double *>(PyArray_DATA(arr)),
          static_cast<double *>(PyArray_DATA(arr)) + n);
      attr.type = MUIMG_ATTR_F64_ARRAY;
      attr.count = static_cast<size_t>(n);
      attr.value.f64_array = scratch.f64_arrays.back().data();
      Py_DECREF(obj);
      return true;
    }
    if (typenum == NPY_FLOAT32) {
      PyObject *obj = PyArray_ContiguousFromAny(val, NPY_FLOAT32, 1, 1);
      if (!obj) {
        return false;
      }
      PyArrayObject *arr = reinterpret_cast<PyArrayObject *>(obj);
      npy_intp n = PyArray_DIM(arr, 0);
      if (n < 1) {
        Py_DECREF(obj);
        PyErr_SetString(PyExc_ValueError, "attr array must be non-empty");
        return false;
      }
      scratch.f32_arrays.emplace_back(
          static_cast<float *>(PyArray_DATA(arr)),
          static_cast<float *>(PyArray_DATA(arr)) + n);
      attr.type = MUIMG_ATTR_F32_ARRAY;
      attr.count = static_cast<size_t>(n);
      attr.value.f32_array = scratch.f32_arrays.back().data();
      Py_DECREF(obj);
      return true;
    }
    if (typenum == NPY_INT32 || typenum == NPY_INT64 || typenum == NPY_UINT16 ||
        typenum == NPY_INT16 || typenum == NPY_UINT8) {
      PyObject *obj = PyArray_ContiguousFromAny(val, NPY_INT32, 1, 1);
      if (!obj) {
        return false;
      }
      PyArrayObject *arr = reinterpret_cast<PyArrayObject *>(obj);
      npy_intp n = PyArray_DIM(arr, 0);
      if (n < 1) {
        Py_DECREF(obj);
        PyErr_SetString(PyExc_ValueError, "attr array must be non-empty");
        return false;
      }
      scratch.i32_arrays.emplace_back(
          static_cast<int32_t *>(PyArray_DATA(arr)),
          static_cast<int32_t *>(PyArray_DATA(arr)) + n);
      attr.type = MUIMG_ATTR_I32_ARRAY;
      attr.count = static_cast<size_t>(n);
      attr.value.i32_array = scratch.i32_arrays.back().data();
      Py_DECREF(obj);
      return true;
    }
  }

  /* Sequence fallback → float32 (legacy). */
  PyObject *f_obj = PyArray_ContiguousFromAny(val, NPY_FLOAT32, 1, 1);
  if (f_obj) {
    PyArrayObject *arr = reinterpret_cast<PyArrayObject *>(f_obj);
    npy_intp n = PyArray_DIM(arr, 0);
    if (n < 1) {
      Py_DECREF(f_obj);
      PyErr_SetString(PyExc_ValueError, "attr array must be non-empty");
      return false;
    }
    scratch.f32_arrays.emplace_back(
        static_cast<float *>(PyArray_DATA(arr)),
        static_cast<float *>(PyArray_DATA(arr)) + n);
    attr.type = MUIMG_ATTR_F32_ARRAY;
    attr.count = static_cast<size_t>(n);
    attr.value.f32_array = scratch.f32_arrays.back().data();
    Py_DECREF(f_obj);
    return true;
  }
  PyErr_Clear();

  PyErr_SetString(PyExc_TypeError,
                  "attr value must be str, bool, float, int, or 1-D array");
  return false;
}

static bool parse_attrs(PyObject *obj, GraphScratch &scratch,
                        std::vector<MuImgAttr> &attrs_out) {
  attrs_out.clear();
  if (obj == Py_None || obj == nullptr) {
    return true;
  }
  if (!PyDict_Check(obj)) {
    PyErr_SetString(PyExc_TypeError, "node attrs must be a dict");
    return false;
  }
  PyObject *key;
  PyObject *value;
  Py_ssize_t pos = 0;
  while (PyDict_Next(obj, &pos, &key, &value)) {
    MuImgAttr attr{};
    if (!parse_attr_value(key, value, scratch, attr)) {
      return false;
    }
    attrs_out.push_back(attr);
  }
  return true;
}

static bool parse_tensor_desc(PyObject *obj, MuImgTensorDesc &desc) {
  if (!PyDict_Check(obj)) {
    PyErr_SetString(PyExc_TypeError, "tensor_desc must be a dict");
    return false;
  }
  PyObject *id_obj = PyDict_GetItemString(obj, "id");
  PyObject *dtype_obj = PyDict_GetItemString(obj, "dtype");
  PyObject *h_obj = PyDict_GetItemString(obj, "height");
  PyObject *w_obj = PyDict_GetItemString(obj, "width");
  PyObject *c_obj = PyDict_GetItemString(obj, "channels");
  if (!id_obj || !dtype_obj || !h_obj || !w_obj || !c_obj) {
    PyErr_SetString(PyExc_ValueError,
                    "tensor_desc requires id, dtype, height, width, channels");
    return false;
  }
  desc.id = static_cast<MuImgTensorId>(PyLong_AsUnsignedLong(id_obj));
  if (PyErr_Occurred()) {
    return false;
  }
  MuImgDType dtype;
  if (!parse_dtype(dtype_obj, dtype)) {
    return false;
  }
  desc.buffer.data = nullptr;
  desc.buffer.dtype = dtype;
  desc.buffer.height = static_cast<size_t>(PyLong_AsSize_t(h_obj));
  desc.buffer.width = static_cast<size_t>(PyLong_AsSize_t(w_obj));
  desc.buffer.channels = static_cast<size_t>(PyLong_AsSize_t(c_obj));
  desc.buffer.stride = 0;
  PyObject *stride_obj = PyDict_GetItemString(obj, "stride");
  if (stride_obj) {
    desc.buffer.stride = static_cast<size_t>(PyLong_AsSize_t(stride_obj));
  }
  return !PyErr_Occurred();
}

static bool parse_graph(PyObject *graph_obj, GraphScratch &scratch,
                        MuImgGraph &graph) {
  if (!PyDict_Check(graph_obj)) {
    PyErr_SetString(PyExc_TypeError, "graph must be a dict");
    return false;
  }

  PyObject *descs_obj = PyDict_GetItemString(graph_obj, "tensor_descs");
  PyObject *inputs_obj = PyDict_GetItemString(graph_obj, "inputs");
  PyObject *outputs_obj = PyDict_GetItemString(graph_obj, "outputs");
  PyObject *nodes_obj = PyDict_GetItemString(graph_obj, "nodes");
  if (!descs_obj || !inputs_obj || !outputs_obj || !nodes_obj) {
    PyErr_SetString(PyExc_ValueError,
                    "graph requires tensor_descs, inputs, outputs, nodes");
    return false;
  }
  if (!PyList_Check(descs_obj) || !PyList_Check(nodes_obj)) {
    PyErr_SetString(PyExc_TypeError, "tensor_descs and nodes must be lists");
    return false;
  }

  Py_ssize_t nd = PyList_Size(descs_obj);
  scratch.descs.resize(static_cast<size_t>(nd));
  for (Py_ssize_t i = 0; i < nd; ++i) {
    if (!parse_tensor_desc(PyList_GetItem(descs_obj, i),
                           scratch.descs[static_cast<size_t>(i)])) {
      return false;
    }
  }

  if (!parse_id_list(inputs_obj, scratch.inputs) ||
      !parse_id_list(outputs_obj, scratch.outputs)) {
    return false;
  }

  Py_ssize_t nn = PyList_Size(nodes_obj);
  scratch.nodes.resize(static_cast<size_t>(nn));
  scratch.node_ios.resize(static_cast<size_t>(nn) * 2);
  scratch.node_attrs.resize(static_cast<size_t>(nn));

  for (Py_ssize_t i = 0; i < nn; ++i) {
    PyObject *node_obj = PyList_GetItem(nodes_obj, i);
    if (!PyDict_Check(node_obj)) {
      PyErr_SetString(PyExc_TypeError, "node must be a dict");
      return false;
    }
    MuImgGraphNode &node = scratch.nodes[static_cast<size_t>(i)];
    PyObject *id_obj = PyDict_GetItemString(node_obj, "id");
    PyObject *op_obj = PyDict_GetItemString(node_obj, "op");
    PyObject *nin_obj = PyDict_GetItemString(node_obj, "inputs");
    PyObject *nout_obj = PyDict_GetItemString(node_obj, "outputs");
    PyObject *attrs_obj = PyDict_GetItemString(node_obj, "attrs");
    if (!id_obj || !op_obj || !nin_obj || !nout_obj) {
      PyErr_SetString(PyExc_ValueError,
                      "node requires id, op, inputs, outputs");
      return false;
    }
    if (!PyUnicode_Check(op_obj)) {
      PyErr_SetString(PyExc_TypeError, "node op must be str");
      return false;
    }
    node.id = static_cast<uint32_t>(PyLong_AsUnsignedLong(id_obj));
    scratch.op_names.emplace_back(PyUnicode_AsUTF8(op_obj));
    node.op = scratch.op_names.back().c_str();

    auto &ins = scratch.node_ios[static_cast<size_t>(i) * 2];
    auto &outs = scratch.node_ios[static_cast<size_t>(i) * 2 + 1];
    if (!parse_id_list(nin_obj, ins) || !parse_id_list(nout_obj, outs)) {
      return false;
    }
    node.inputs = ins.data();
    node.num_inputs = ins.size();
    node.outputs = outs.data();
    node.num_outputs = outs.size();

    auto &attrs = scratch.node_attrs[static_cast<size_t>(i)];
    if (!parse_attrs(attrs_obj, scratch, attrs)) {
      return false;
    }
    node.attrs = attrs.empty() ? nullptr : attrs.data();
    node.num_attrs = attrs.size();
  }

  graph.tensor_descs = scratch.descs.data();
  graph.num_tensor_descs = scratch.descs.size();
  graph.inputs = scratch.inputs.data();
  graph.num_inputs = scratch.inputs.size();
  graph.outputs = scratch.outputs.data();
  graph.num_outputs = scratch.outputs.size();
  graph.nodes = scratch.nodes.data();
  graph.num_nodes = scratch.nodes.size();
  return true;
}

static bool buffer_from_array(PyArrayObject *arr, const MuImgTensorDesc &desc,
                              MuImgBuffer &buf) {
  if (PyArray_TYPE(arr) == NPY_FLOAT32) {
    if (desc.buffer.dtype != MUIMG_DTYPE_FLOAT32) {
      PyErr_SetString(PyExc_TypeError, "binding dtype mismatch (float32)");
      return false;
    }
  } else if (PyArray_TYPE(arr) == NPY_UINT16) {
    if (desc.buffer.dtype != MUIMG_DTYPE_UINT16) {
      PyErr_SetString(PyExc_TypeError, "binding dtype mismatch (uint16)");
      return false;
    }
  } else if (PyArray_TYPE(arr) == NPY_UINT8) {
    if (desc.buffer.dtype != MUIMG_DTYPE_UINT8) {
      PyErr_SetString(PyExc_TypeError, "binding dtype mismatch (uint8)");
      return false;
    }
  } else {
    PyErr_SetString(PyExc_TypeError, "binding array must be uint8/uint16/float32");
    return false;
  }

  if (!PyArray_IS_C_CONTIGUOUS(arr)) {
    PyErr_SetString(PyExc_ValueError, "binding array must be C-contiguous");
    return false;
  }

  int nd = PyArray_NDIM(arr);
  size_t h = desc.buffer.height;
  size_t w = desc.buffer.width;
  size_t c = desc.buffer.channels;
  if (c == 1 && nd == 2) {
    if (static_cast<size_t>(PyArray_DIM(arr, 0)) != h ||
        static_cast<size_t>(PyArray_DIM(arr, 1)) != w) {
      PyErr_SetString(PyExc_ValueError, "binding shape mismatch (H,W)");
      return false;
    }
  } else if (nd == 3) {
    if (static_cast<size_t>(PyArray_DIM(arr, 0)) != h ||
        static_cast<size_t>(PyArray_DIM(arr, 1)) != w ||
        static_cast<size_t>(PyArray_DIM(arr, 2)) != c) {
      PyErr_SetString(PyExc_ValueError, "binding shape mismatch (H,W,C)");
      return false;
    }
  } else {
    PyErr_SetString(PyExc_ValueError, "binding array must be (H,W) or (H,W,C)");
    return false;
  }

  buf = desc.buffer;
  buf.data = PyArray_DATA(arr);
  return true;
}

static const MuImgTensorDesc *find_desc(const GraphScratch &scratch,
                                        MuImgTensorId id) {
  for (const auto &d : scratch.descs) {
    if (d.id == id) {
      return &d;
    }
  }
  return nullptr;
}

static bool parse_bindings(PyObject *obj, const GraphScratch &scratch,
                           bool is_output, std::vector<MuImgGraphBinding> &out,
                           std::vector<PyPtr<>> &keep_alive) {
  if (!PyDict_Check(obj)) {
    PyErr_SetString(PyExc_TypeError, "bindings must be a dict id -> ndarray");
    return false;
  }
  out.clear();
  PyObject *key;
  PyObject *value;
  Py_ssize_t pos = 0;
  while (PyDict_Next(obj, &pos, &key, &value)) {
    if (!PyLong_Check(key)) {
      PyErr_SetString(PyExc_TypeError, "binding key must be tensor id int");
      return false;
    }
    MuImgTensorId tid = static_cast<MuImgTensorId>(PyLong_AsUnsignedLong(key));
    const MuImgTensorDesc *desc = find_desc(scratch, tid);
    if (!desc) {
      PyErr_Format(PyExc_ValueError, "binding for unknown tensor id %u", tid);
      return false;
    }
    /* Packed (C-contiguous) buffers only. */
    int requirements = NPY_ARRAY_C_CONTIGUOUS;
    if (is_output) {
      requirements |= NPY_ARRAY_WRITEABLE;
    }
    PyPtr<> arr = make_pyptr(PyArray_FROM_OTF(value, NPY_NOTYPE, requirements));
    if (!arr) {
      return false;
    }
    MuImgGraphBinding bind{};
    bind.tensor_id = tid;
    if (!buffer_from_array(reinterpret_cast<PyArrayObject *>(arr.get()), *desc,
                           bind.buffer)) {
      return false;
    }
    keep_alive.push_back(std::move(arr));
    out.push_back(bind);
  }
  return true;
}

//=============================================================================
// execute_graph(graph, inputs, outputs) -> None
//=============================================================================

static PyObject *py_execute_graph(PyObject * /*self*/, PyObject *args) {
  PyObject *graph_obj = nullptr;
  PyObject *inputs_obj = nullptr;
  PyObject *outputs_obj = nullptr;
  if (!PyArg_ParseTuple(args, "OOO", &graph_obj, &inputs_obj, &outputs_obj)) {
    return nullptr;
  }
  if (!load_core_library()) {
    return nullptr;
  }

  GraphScratch scratch;
  MuImgGraph graph{};
  if (!parse_graph(graph_obj, scratch, graph)) {
    return nullptr;
  }

  std::vector<MuImgGraphBinding> in_binds;
  std::vector<MuImgGraphBinding> out_binds;
  std::vector<PyPtr<>> keep_alive;

  if (!parse_bindings(inputs_obj, scratch, false, in_binds, keep_alive) ||
      !parse_bindings(outputs_obj, scratch, true, out_binds, keep_alive)) {
    return nullptr;
  }

  int rc = muimg_execute_graph_fn(&graph, in_binds.data(), in_binds.size(),
                                  out_binds.data(), out_binds.size());
  if (rc != MUIMG_SUCCESS) {
    set_python_error_from_muimg(rc);
    return nullptr;
  }
  Py_RETURN_NONE;
}

static PyMethodDef ComputeEngineMethods[] = {
    {"execute_graph", py_execute_graph, METH_VARARGS,
     "execute_graph(graph, inputs, outputs) -> None\n\n"
     "Run one engine segment.\n\n"
     "graph: dict with tensor_descs, inputs, outputs, nodes\n"
     "inputs/outputs: dict[tensor_id] -> C-contiguous ndarray\n"},
    {nullptr, nullptr, 0, nullptr}};

static struct PyModuleDef compute_engine_module = {
    PyModuleDef_HEAD_INIT,
    "_compute_engine",
    "Compute-graph engine binding.",
    -1,
    ComputeEngineMethods};

PyMODINIT_FUNC PyInit__compute_engine(void) {
  import_array();
  if (PyErr_Occurred()) {
    return nullptr;
  }
  return PyModule_Create(&compute_engine_module);
}
