// Copyright © 2025 Apple Inc.

#include <stdexcept>

#include "mlx/backend/cuda/cuda.h"
#include "mlx/fast.h"

namespace mlx::core {

namespace cu {

bool is_available() {
  return false;
}

int current_device() {
  return -1;
}

void synchronize_device() {}

allocator::Buffer import_external_buffer(void*, size_t) {
  throw std::runtime_error(
      "[import_external_buffer] No CUDA back-end: cannot import a CUDA DLPack "
      "buffer.");
}

void free_external_buffer(allocator::Buffer) {}

} // namespace cu

namespace fast {

CustomKernelFunction cuda_kernel(
    const std::string&,
    const std::vector<std::string>&,
    const std::vector<std::string>&,
    const std::string&,
    const std::string&,
    bool,
    int) {
  throw std::runtime_error("[cuda_kernel] No CUDA back-end.");
}

std::vector<array> precompiled_cuda_kernel(
    const std::string&,
    const std::string&,
    const std::vector<array>&,
    const std::vector<Shape>&,
    const std::vector<Dtype>&,
    const std::vector<ScalarArg>&,
    std::tuple<int, int, int>,
    std::tuple<int, int, int>,
    int shared_memory,
    std::optional<float> init_value,
    bool ensure_row_contiguous,
    StreamOrDevice) {
  throw std::runtime_error("[cuda_kernel] No CUDA back-end.");
}

} // namespace fast

} // namespace mlx::core
