// Copyright © 2025 Apple Inc.

#pragma once

#include <string>
#include <unordered_map>
#include <variant>

#include "mlx/api.h"

namespace mlx::core::cu {

/* Check if the CUDA backend is available. */
MLX_API bool is_available();

/* Get information about a CUDA device. */
MLX_API const
    std::unordered_map<std::string, std::variant<std::string, size_t>>&
    device_info(int device_index = 0);

/* Ordinal of the CUDA device backing MLX's current allocations.
 * Returns -1 if the CUDA backend is unavailable. Used by DLPack export so the
 * exported kDLCUDA tensor advertises the correct device_id (multi-GPU safe). */
MLX_API int current_device();

/* Block until all CUDA work submitted on the current device has completed.
 * No-op when the CUDA backend is unavailable. Used by DLPack export to honor
 * the producer-side readiness contract before handing a device pointer to a
 * foreign consumer (tvm-ffi / TileLang). */
MLX_API void synchronize_device();

} // namespace mlx::core::cu
