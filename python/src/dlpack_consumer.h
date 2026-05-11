// Copyright © 2026 Apple Inc.
#pragma once

#include <cstddef>
#include <memory>
#include <optional>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include "mlx/array.h"
#include "mlx/dtype.h"

namespace mx = mlx::core;
namespace nb = nanobind;

// Convert a DLPack capsule (or a Python object exposing __dlpack__) into an
// mx::array.
//
// Supported device types:
//   * kDLCPU (1)        : zero-copy; wraps the producer pointer when the
//                         active allocator can expose it directly.
//   * kDLMetal (8)      : zero-copy; wraps a foreign MTL::Buffer in shared
//                         storage mode. Non-shared buffers are rejected.
//
// All other device types raise std::invalid_argument.
//
// Accepted inputs are wrapped zero-copy, so the returned mx::array keeps the
// capsule deleter alive until the array and any aliases are destroyed.
// Rejected capsules are left unconsumed. Dtype conversion is rejected because it
// would require a hidden copy/cast.
mx::array dlpack_to_mlx(
    nb::object obj,
    std::optional<mx::Dtype> dtype = std::nullopt);

mx::Dtype dlpack_to_mlx_dtype(const nb::dlpack::dtype& dt);
mx::Shape validate_and_extract_shape(const nb::dlpack::dltensor& t);
bool is_row_contiguous(const mx::Shape& shape, const int64_t* strides);
size_t checked_num_bytes(const mx::Shape& shape, mx::Dtype dtype);

// A small reference-counted holder that drives the DLPack capsule's deleter
// exactly once after ownership has been committed. Kept here so the metal-glue
// translation unit can reach it.
class DLPackOwner {
 public:
  DLPackOwner(bool versioned, void* mt) : versioned_(versioned), mt_(mt) {}

  ~DLPackOwner() {
    invoke();
  }

  void activate() {
    active_ = true;
  }

  void invoke();

  // Disable copies: only one DLPackOwner may exist per managed tensor.
  DLPackOwner(const DLPackOwner&) = delete;
  DLPackOwner& operator=(const DLPackOwner&) = delete;

 private:
  bool versioned_;
  void* mt_;
  bool active_ = false;
};

// Build an mx::array from a DLPack tensor whose data is a foreign MTL::Buffer.
// Defined in dlpack_consumer_metal.cpp when MLX_BUILD_METAL is on, and in
// dlpack_consumer_no_metal.cpp otherwise.
mx::array build_dlpack_metal_array(
    nb::dlpack::dltensor& t,
    std::shared_ptr<DLPackOwner> owner);
