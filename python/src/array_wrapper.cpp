// Copyright © 2023-2024 Apple Inc.
#include <memory>
#include <new>
#include <stdexcept>
#include <utility>

#include <Python.h>
#include <nanobind/nanobind.h>

#include "mlx/array.h"

namespace mx = mlx::core;
namespace nb = nanobind;

extern "C" __attribute__((visibility("default"))) PyObject*
mlx_core_wrap_mx_array_move(mx::array* array) noexcept {
  std::unique_ptr<mx::array> owned(array);
  if (owned == nullptr) {
    PyErr_SetString(PyExc_ValueError, "mlx_core_wrap_mx_array_move received a null array");
    return nullptr;
  }

  try {
    nb::object array_type = nb::module_::import_("mlx.core").attr("array");
    nb::object py_array = array_type.attr("__new__")(array_type);
    auto* storage = nb::inst_ptr<mx::array>(py_array);
    if (storage == nullptr) {
      PyErr_SetString(PyExc_RuntimeError, "failed to allocate an mlx.core.array instance");
      return nullptr;
    }
    new (storage) mx::array(std::move(*owned));
    nb::inst_set_state(py_array, true, true);
    return py_array.release().ptr();
  } catch (nb::python_error& exc) {
    exc.restore();
  } catch (const std::exception& exc) {
    PyErr_SetString(PyExc_RuntimeError, exc.what());
  } catch (...) {
    PyErr_SetString(PyExc_RuntimeError, "unknown error wrapping mlx.core.array");
  }
  return nullptr;
}
