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

/* Copy `nbytes` from a FOREIGN CUDA device pointer (e.g. a DLPack kDLCUDA /
 * kDLCUDAManaged capsule produced by torch / tvm-ffi) into a FRESH MLX-managed
 * allocation and return it.
 *
 * Unlike import_external_buffer (which wraps the foreign pointer and therefore
 * forces an mx::array deleter to keep the foreign owner alive), this performs a
 * single on-device cudaMemcpy(cudaMemcpyDefault) into a real allocator::malloc()
 * buffer. The returned Buffer is owned by MLX and is freed through the normal
 * allocator::free() path — NO foreign pointer and NO Python/DLPack owner ever
 * enters MLX's graph or scheduler. This is the deadlock-free import primitive:
 * the foreign capsule/torch tensor is only READ during the copy (on the calling
 * thread) and may be released immediately afterwards on that same thread, so its
 * (GIL-needing) deleter never runs on MLX's scheduler thread.
 *
 * The copy is synchronous on the current stream/thread; on return the data is
 * fully resident in the MLX buffer. */
MLX_API allocator::Buffer copy_external_to_mlx_buffer(
    const void* src,
    size_t nbytes);

} // namespace mlx::core::cu
