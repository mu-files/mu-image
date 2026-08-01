#pragma once

/*
 * RAII wrapper for PyObject reference counting.
 * Requires Python.h before this header.
 */

#include <memory>

struct PyObjectDeleter {
  template <typename T>
  void operator()(T *obj) const {
    Py_XDECREF(reinterpret_cast<PyObject *>(obj));
  }
};

template <typename T = PyObject>
using PyPtr = std::unique_ptr<T, PyObjectDeleter>;

template <typename T = PyObject>
PyPtr<T> make_pyptr(T *obj) {
  return PyPtr<T>(obj);
}
