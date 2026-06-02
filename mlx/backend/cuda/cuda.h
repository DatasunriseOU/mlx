// Copyright © 2025 Apple Inc.

#pragma once

#include <string>
#include <unordered_map>
#include <variant>

#include "mlx/allocator.h"
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

/* Wrap a foreign CUDA device pointer (e.g. imported from a DLPack kDLCUDA /
 * kDLCUDAManaged capsule produced by torch/tvm-ffi) in an MLX allocator::Buffer
 * WITHOUT taking ownership of the underlying CUDA memory.
 *
 * MLX's CUDA Buffer wraps an internal `CudaBuffer` struct (not the raw device
 * pointer): allocator::Buffer::raw_ptr()/ptr() dereference it as `CudaBuffer*`,
 * so a raw foreign pointer cannot be handed to the Buffer ctor directly (that
 * was the "allocator/lifetime mismatch" that made CUDA DLPack import unsupported
 * on this fork). This helper heap-allocates a `CudaBuffer{ptr, nbytes, -1}`
 * wrapper so MLX reads the foreign pointer back correctly. device=-1 marks it as
 * unified-style so raw_ptr()/move_to_unified_memory() are no-ops (they neither
 * copy nor free the foreign allocation).
 *
 * The wrapper MUST be released via cu::free_external_buffer (NOT allocator::free,
 * which would recycle the foreign pointer into MLX's pool). The caller is
 * responsible for keeping the real owner (DLPack capsule / torch tensor) alive
 * for as long as any mx::array reads this buffer; the underlying CUDA memory is
 * never freed by MLX. */
MLX_API allocator::Buffer import_external_buffer(void* ptr, size_t nbytes);

/* Release a Buffer created by cu::import_external_buffer: deletes ONLY the
 * heap-allocated `CudaBuffer` wrapper struct, never the underlying (foreign)
 * CUDA allocation. */
MLX_API void free_external_buffer(allocator::Buffer buffer);

} // namespace mlx::core::cu
