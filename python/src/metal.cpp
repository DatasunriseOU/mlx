// Copyright © 2023-2024 Apple Inc.
#include <cstdint>
#include <iostream>

#include <nanobind/nanobind.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/variant.h>
#include <nanobind/stl/vector.h>

#include "mlx/backend/metal/device.h"
#include "mlx/backend/metal/metal.h"
#include "mlx/device.h"
#include "mlx/memory.h"
#include "mlx/utils.h"
#include "python/src/small_vector.h"

namespace mx = mlx::core;
namespace nb = nanobind;
using namespace nb::literals;

bool DEPRECATE(const char* old_fn, const char* new_fn) {
  std::cerr << old_fn << " is deprecated and will be removed in a future "
            << "version. Use " << new_fn << " instead." << std::endl;
  return true;
}

#define DEPRECATE(oldfn, newfn) static bool dep = DEPRECATE(oldfn, newfn)

void init_metal(nb::module_& m) {
  nb::module_ metal = m.def_submodule("metal", "mlx.metal");
  metal.def(
      "is_available",
      &mx::metal::is_available,
      R"pbdoc(
      Check if the Metal back-end is available.
      )pbdoc");
  metal.def("get_active_memory", []() {
    DEPRECATE("mx.metal.get_active_memory", "mx.get_active_memory");
    return mx::get_active_memory();
  });
  metal.def("get_peak_memory", []() {
    DEPRECATE("mx.metal.get_peak_memory", "mx.get_peak_memory");
    return mx::get_peak_memory();
  });
  metal.def("reset_peak_memory", []() {
    DEPRECATE("mx.metal.reset_peak_memory", "mx.reset_peak_memory");
    mx::reset_peak_memory();
  });
  metal.def("get_cache_memory", []() {
    DEPRECATE("mx.metal.get_cache_memory", "mx.get_cache_memory");
    return mx::get_cache_memory();
  });
  metal.def(
      "set_memory_limit",
      [](size_t limit) {
        DEPRECATE("mx.metal.set_memory_limit", "mx.set_memory_limit");
        return mx::set_memory_limit(limit);
      },
      "limit"_a);
  metal.def(
      "set_cache_limit",
      [](size_t limit) {
        DEPRECATE("mx.metal.set_cache_limit", "mx.set_cache_limit");
        return mx::set_cache_limit(limit);
      },
      "limit"_a);
  metal.def(
      "set_wired_limit",
      [](size_t limit) {
        DEPRECATE("mx.metal.set_wired_limit", "mx.set_wired_limit");
        return mx::set_wired_limit(limit);
      },
      "limit"_a);
  metal.def("clear_cache", []() {
    DEPRECATE("mx.metal.clear_cache", "mx.clear_cache");
    mx::clear_cache();
  });
  metal.def(
      "start_capture",
      &mx::metal::start_capture,
      "path"_a,
      R"pbdoc(
      Start a Metal capture.

      Args:
        path (str): The path to save the capture which should have
          the extension ``.gputrace``.
      )pbdoc");
  metal.def(
      "stop_capture",
      &mx::metal::stop_capture,
      R"pbdoc(
      Stop a Metal capture.
      )pbdoc");
  metal.def(
      "_current_command_buffer",
      [](mx::StreamOrDevice s) {
        auto stream = mx::to_stream(s);
        if (stream.device != mx::Device::gpu) {
          throw std::invalid_argument(
              "[metal._current_command_buffer] Expected a GPU stream or device.");
        }
        return reinterpret_cast<uintptr_t>(
            mx::metal::current_command_buffer(stream));
      },
      "stream"_a = std::monostate{},
      R"pbdoc(
      Return the current borrowed Metal command buffer pointer for interop.

      This is an internal scheduling hook for zero-copy framework interop.
      The pointer is owned and committed by MLX; consumers may encode work
      into it but must not retain, release, or commit it.
      )pbdoc");
  metal.def("device_info", []() {
    DEPRECATE("mx.metal.device_info", "mx.device_info");
    return mx::device_info(mx::Device(mx::Device::gpu, 0));
  });
}
