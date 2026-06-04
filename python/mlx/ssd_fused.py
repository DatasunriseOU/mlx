# Copyright © 2026 DatasunriseOU.
"""Custom FUSED SSD chunk-scan MLX primitive (CUDA + Metal).

HONEST LABEL: this is a **custom-primitive mono kernel, NOT mx.compile
auto-fusion.** MLX's ``mx.compile`` auto-fusion is elementwise-only
(``mlx/compile.cpp`` ``is_fusable`` is a typeid allowlist of unary/binary/
ternary/broadcast ops; a Matmul/Reduce/Scan node is a hard fusion boundary and
``Compiled::eval_gpu`` only emits a per-element contiguous/strided loop — no
shared memory, no tensor cores, no sequential state). MLX therefore *cannot*
auto-fuse a stateful tensor-core chunk-scan into one kernel body. The reachable
mono-fusion is to author the whole SSD chunk region as ONE user kernel body and
register it as ONE MLX op via the existing escape hatch:

  * CUDA  : ``mx.fast.cuda_kernel``   (JIT, sm_121 mma.sync m16n8k16-capable)
  * Metal : ``mx.fast.metal_kernel``  (MSL threadgroup memory + simdgroup)

Both register exactly ONE ``CustomKernel`` primitive => ONE launch => ONE node
in the MLX graph. That is the genuine mono-fused kernel BODY (state resident in
shared memory across the chunk axis, never round-tripped through global memory),
distinct from cudaGraph launch-batching (which merely orders the 6 separate
F0/F1/F2/B0/B1/B2 kernels without merging their bodies).

This REPLACES the 6-kernel chain (F0/F1/F2 fwd + B0/B1/B2 bwd) with ONE fused
forward op. It is **default OFF** behind ``MLX_SSD_FUSED`` — callers keep the
existing path byte-identical unless they opt in (or call this module directly).

RULE #1 (no silent fallback): when invoked, this is the ONE path. On any
compile / shape / backend-availability failure it RAISES with where+what; it
never degrades to a slow / wrong / zeroed path and never silently re-tiles.

PER-(batch,head) ALGORITHM — ONE threadgroup/CTA owns ``state[P,N]`` resident in
shared memory and loops the chunks serially (state never leaves shared):

  state[P,N] = h0
  for c in 0..nchunks-1:
    dA[l]      = inclusive_cumsum_l( A[h] * dt[c,l] )           (F0 cumsum)
    Y[c,i,p]   = D[h]*x[c,i,p]
               + exp(dA[i]) * sum_n C[c,i,n] * state[p,n]       (F2 inter-chunk, RESIDENT state)
               + sum_{s<=i} ( sum_n C[c,i,n]*B[c,s,n] )         (F0 cb)
                            * exp(dA[i]-dA[s]) * dt[c,s] * x[c,s,p]   (F2 intra-chunk)
    state[p,n] = exp(dA[L-1]) * state[p,n]                      (F1 carry)
               + sum_l exp(dA[L-1]-dA[l]) * dt[c,l] * x[c,l,p] * B[c,l,n]  (F0 summary)
  final_state[p,n] = state[p,n]

INPUT ABI (raw per-position SSD tensors, NOT the F2 handoff buffers — this op IS
F0+F1+F2), ngroups assumed 1 (each head shares the single B/C group):
  x   : (batch, seqlen, nheads, headdim)   fp16/fp32
  B   : (batch, seqlen, 1, dstate)         fp16/fp32
  C   : (batch, seqlen, 1, dstate)         fp16/fp32
  A   : (nheads,)                          fp32   A = -softplus(...) <= 0
  dt  : (batch, seqlen, nheads)            fp32
  D   : (nheads,)                          fp32
  h0  : (batch, nheads, headdim, dstate)   fp32   initial inter-chunk state
OUTPUTS:
  Y           : (batch, seqlen, nheads, headdim)   (input x dtype)
  final_state : (batch, nheads, headdim, dstate)   fp32

Numerical contract: reproduces the serial per-timestep diagonal recurrence to
fp16 ``< 5e-4`` over all elements. The carried state is fp32 (never downcast).
"""

from __future__ import annotations

import os
from typing import Tuple

import mlx.core as mx

__all__ = [
    "SSD_FUSED_OP_NAME",
    "SSD_FUSED_ENV",
    "ssd_fused_enabled",
    "ssd_chunk_scan_fused",
    "auto_fusion_status",
]

# Stable op name surfaced in the MLX graph for this fused SSD chunk op.
SSD_FUSED_OP_NAME = "ssd_chunk_scan_fused"

# Env gate (default OFF). When unset/falsey the existing 6-kernel path is used by
# callers; nothing here runs unless explicitly invoked.
SSD_FUSED_ENV = "MLX_SSD_FUSED"


def ssd_fused_enabled() -> bool:
    """True iff the fused SSD primitive is env-gated ON (default OFF)."""
    return os.environ.get(SSD_FUSED_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def auto_fusion_status() -> str:
    """Honest one-line provenance label (RULE #1: no fake auto-fusion claim)."""
    return (
        "custom-primitive (one mx.fast.{cuda,metal}_kernel op, mono kernel body, "
        "state resident in shared mem across chunks) — NOT mx.compile auto-fusion "
        "(MLX auto-fusion is elementwise-only; a GEMM/scan node is a hard fusion "
        "boundary). Distinct from cudaGraph launch-batching."
    )


# 1/ln(2): exp2(x*LOG2E) == exp(x). Kept explicit so both backends match.
_LOG2E = 1.4426950408889634


# --------------------------------------------------------------------------- #
# Metal mono-kernel body (MSL). ONE threadgroup per (batch, head); thread p in  #
# [0,P) owns row p of the resident state[P,N] in threadgroup memory and the      #
# output column for headdim p. State never leaves threadgroup memory across the  #
# chunk loop. dstate N and chunk L are compile-time template ints.               #
# --------------------------------------------------------------------------- #
_METAL_SRC = r"""
  // grid = (P, 1, batch*nheads); threadgroup = (P, 1, 1). One TG per (b,h).
  const uint p   = thread_position_in_threadgroup.x;          // headdim row [0,P)
  const uint bh  = threadgroup_position_in_grid.z;            // (b*nheads + h)
  const uint h   = bh % NHEADS;
  const uint b   = bh / NHEADS;

  if (p >= P) return;

  // Resident state row: state[p, 0..N). One thread owns one headdim row.
  threadgroup float st[P * N];
  // Shared per-chunk scratch shared by all threads in the TG.
  threadgroup float dA[L];        // inclusive cumsum of A[h]*dt over the chunk
  threadgroup float Csh[L * N];   // C[c, i, :]
  threadgroup float Bsh[L * N];   // B[c, s, :]
  threadgroup float dtsh[L];      // dt[c, l]

  const float Ah = (float)A[h];
  const float Dh = (float)D[h];

  // ---- seed resident state from h0 (b,h,p,:) ----
  const uint h0_base = ((b * NHEADS + h) * P + p) * N;
  for (uint n = 0; n < N; ++n) {
    st[p * N + n] = (float)h0[h0_base + n];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const uint NCHUNKS = SEQLEN / L;

  for (uint c = 0; c < NCHUNKS; ++c) {
    const uint t0 = c * L;  // first timestep of this chunk

    // Thread 0 loads dt and builds the inclusive cumsum dA (sequential, L small).
    if (p == 0) {
      float acc = 0.0f;
      for (uint l = 0; l < L; ++l) {
        const uint t = t0 + l;
        const float dtl = (float)dt[(b * SEQLEN + t) * NHEADS + h];
        dtsh[l] = dtl;
        acc += Ah * dtl;
        dA[l] = acc;
      }
    }
    // Every thread cooperatively stages C and B for the chunk (ngroups == 1).
    for (uint l = p; l < L; l += P) {
      const uint t = t0 + l;
      const uint bc_base = (b * SEQLEN + t) * N;  // (b,t,0,:) group 0
      for (uint n = 0; n < N; ++n) {
        Csh[l * N + n] = (float)C[bc_base + n];
        Bsh[l * N + n] = (float)B[bc_base + n];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const float dA_last = dA[L - 1];

    // ---- outputs for this chunk: row p over all i in [0,L) ----
    for (uint i = 0; i < L; ++i) {
      // Y_off: exp(dA[i]) * sum_n C[i,n] * state[p,n]   (resident state)
      float yoff = 0.0f;
      for (uint n = 0; n < N; ++n) {
        yoff += Csh[i * N + n] * st[p * N + n];
      }
      float y = exp(dA[i]) * yoff;

      // Y_diag: sum_{s<=i} cb[i,s] * exp(dA[i]-dA[s]) * dt[s] * x[c,s,p]
      for (uint s = 0; s <= i; ++s) {
        float cb = 0.0f;
        for (uint n = 0; n < N; ++n) {
          cb += Csh[i * N + n] * Bsh[s * N + n];
        }
        const uint xs = ((b * SEQLEN + (t0 + s)) * NHEADS + h) * P + p;
        y += cb * exp(dA[i] - dA[s]) * dtsh[s] * (float)x[xs];
      }

      // D skip on x[c,i,p]
      const uint xi = ((b * SEQLEN + (t0 + i)) * NHEADS + h) * P + p;
      y += Dh * (float)x[xi];

      Y[xi] = (T)y;
    }

    // ---- advance resident state (F1 carry + F0 summary) for row p ----
    // state[p,n] = exp(dA_last)*state[p,n]
    //            + sum_l exp(dA_last - dA[l]) * dt[l] * x[c,l,p] * B[c,l,n]
    float decay = exp(dA_last);
    float xcol[L];
    for (uint l = 0; l < L; ++l) {
      const uint xl = ((b * SEQLEN + (t0 + l)) * NHEADS + h) * P + p;
      xcol[l] = exp(dA_last - dA[l]) * dtsh[l] * (float)x[xl];
    }
    for (uint n = 0; n < N; ++n) {
      float acc = decay * st[p * N + n];
      for (uint l = 0; l < L; ++l) {
        acc += xcol[l] * Bsh[l * N + n];
      }
      st[p * N + n] = acc;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // ---- write final_state[b,h,p,:] ----
  const uint fs_base = ((b * NHEADS + h) * P + p) * N;
  for (uint n = 0; n < N; ++n) {
    final_state[fs_base + n] = st[p * N + n];
  }
"""


# --------------------------------------------------------------------------- #
# CUDA mono-kernel body. Same per-(batch,head) resident-state structure; ONE    #
# block per (b,h), thread p in [0,P) owns state row p. Authored for sm_121      #
# (no wgmma/tcgen05); the contraction loops below are scalar — a follow-up may   #
# swap the cb / state contractions for mma.sync m16n8k16 fragments (the kernel   #
# body is the ONE place that lives, exactly as the wave8 chunk-owner reference). #
# Validated/measured on gb10 in the Test phase (no CUDA on this Mac).            #
# --------------------------------------------------------------------------- #
_CUDA_SRC = r"""
  // grid = (P,1, batch*nheads); block = (P,1,1). One block per (b,h).
  const unsigned p  = threadIdx.x;                 // headdim row [0,P)
  const unsigned bh = blockIdx.z;                  // b*NHEADS + h
  const unsigned h  = bh % NHEADS;
  const unsigned b  = bh / NHEADS;
  if (p >= P) return;

  __shared__ float st[P * N];
  __shared__ float dA[L];
  __shared__ float Csh[L * N];
  __shared__ float Bsh[L * N];
  __shared__ float dtsh[L];

  const float Ah = (float)A[h];
  const float Dh = (float)D[h];

  const unsigned h0_base = ((b * NHEADS + h) * P + p) * N;
  for (unsigned n = 0; n < N; ++n) st[p * N + n] = (float)h0[h0_base + n];
  __syncthreads();

  const unsigned NCHUNKS = SEQLEN / L;
  for (unsigned c = 0; c < NCHUNKS; ++c) {
    const unsigned t0 = c * L;
    if (p == 0) {
      float acc = 0.0f;
      for (unsigned l = 0; l < L; ++l) {
        const unsigned t = t0 + l;
        const float dtl = (float)dt[(b * SEQLEN + t) * NHEADS + h];
        dtsh[l] = dtl;
        acc += Ah * dtl;
        dA[l] = acc;
      }
    }
    for (unsigned l = p; l < L; l += P) {
      const unsigned t = t0 + l;
      const unsigned bc_base = (b * SEQLEN + t) * N;
      for (unsigned n = 0; n < N; ++n) {
        Csh[l * N + n] = (float)C[bc_base + n];
        Bsh[l * N + n] = (float)B[bc_base + n];
      }
    }
    __syncthreads();

    const float dA_last = dA[L - 1];
    for (unsigned i = 0; i < L; ++i) {
      float yoff = 0.0f;
      for (unsigned n = 0; n < N; ++n) yoff += Csh[i * N + n] * st[p * N + n];
      float y = expf(dA[i]) * yoff;
      for (unsigned s = 0; s <= i; ++s) {
        float cb = 0.0f;
        for (unsigned n = 0; n < N; ++n) cb += Csh[i * N + n] * Bsh[s * N + n];
        const unsigned xs = ((b * SEQLEN + (t0 + s)) * NHEADS + h) * P + p;
        y += cb * expf(dA[i] - dA[s]) * dtsh[s] * (float)x[xs];
      }
      const unsigned xi = ((b * SEQLEN + (t0 + i)) * NHEADS + h) * P + p;
      y += Dh * (float)x[xi];
      Y[xi] = (T)y;
    }

    const float decay = expf(dA_last);
    float xcol[L];
    for (unsigned l = 0; l < L; ++l) {
      const unsigned xl = ((b * SEQLEN + (t0 + l)) * NHEADS + h) * P + p;
      xcol[l] = expf(dA_last - dA[l]) * dtsh[l] * (float)x[xl];
    }
    for (unsigned n = 0; n < N; ++n) {
      float acc = decay * st[p * N + n];
      for (unsigned l = 0; l < L; ++l) acc += xcol[l] * Bsh[l * N + n];
      st[p * N + n] = acc;
    }
    __syncthreads();
  }

  const unsigned fs_base = ((b * NHEADS + h) * P + p) * N;
  for (unsigned n = 0; n < N; ++n) final_state[fs_base + n] = st[p * N + n];
"""


_HEADER = r"""
#include <metal_stdlib>
using namespace metal;
"""

_CUDA_HEADER = r"""
#include <cuda_fp16.h>
"""


# Build-once caches keyed by (backend, P, N, L, NHEADS, SEQLEN, dtype-str).
_KERNEL_CACHE: dict = {}


def _backend() -> str:
    """Return 'metal' or 'cuda'; RAISE if neither (RULE #1: no silent CPU path)."""
    dev = mx.default_device()
    dtype = str(getattr(dev, "type", dev))
    if "gpu" in dtype.lower() or mx.metal.is_available():
        # mx.metal.is_available() is True on a Metal build even though the device
        # enum is 'gpu'; on a CUDA build cuda_kernel is the path.
        if hasattr(mx.fast, "cuda_kernel") and not mx.metal.is_available():
            return "cuda"
        if mx.metal.is_available():
            return "metal"
    raise RuntimeError(
        "[ssd_chunk_scan_fused] no GPU backend available — the fused SSD "
        "primitive requires Metal or CUDA. (RULE #1: refusing a silent CPU "
        f"fallback.) default_device={dev!r}"
    )


def _make_kernel(backend: str, P: int, N: int, L: int, nheads: int, seqlen: int):
    key = (backend, P, N, L, nheads, seqlen)
    k = _KERNEL_CACHE.get(key)
    if k is not None:
        return k

    # Substitute the compile-time ints into the body (template macros). We inline
    # them as `#define`s in the header so both backends see identical constants.
    defines = (
        f"#define P {P}\n#define N {N}\n#define L {L}\n"
        f"#define NHEADS {nheads}\n#define SEQLEN {seqlen}\n"
    )
    if backend == "metal":
        k = mx.fast.metal_kernel(
            name=f"ssd_fused_metal_P{P}_N{N}_L{L}",
            input_names=["x", "B", "C", "A", "dt", "D", "h0"],
            output_names=["Y", "final_state"],
            source=_METAL_SRC,
            header=_HEADER + defines,
            ensure_row_contiguous=True,
        )
    elif backend == "cuda":
        k = mx.fast.cuda_kernel(
            name=f"ssd_fused_cuda_P{P}_N{N}_L{L}",
            input_names=["x", "B", "C", "A", "dt", "D", "h0"],
            output_names=["Y", "final_state"],
            source=_CUDA_SRC,
            header=_CUDA_HEADER + defines,
            ensure_row_contiguous=True,
            shared_memory=0,  # all shared arrays are static __shared__ above.
        )
    else:
        raise RuntimeError(f"[ssd_chunk_scan_fused] unknown backend {backend!r}")
    _KERNEL_CACHE[key] = k
    return k


def ssd_chunk_scan_fused(
    x: mx.array,
    B: mx.array,
    C: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
    *,
    chunk_size: int = 64,
) -> Tuple[mx.array, mx.array]:
    """Mono-fused SSD chunk-scan forward as ONE MLX custom-primitive op.

    Replaces the 6-kernel F0/F1/F2 chain with ONE kernel launch; the chunk state
    stays resident in shared/threadgroup memory across the whole chunk axis.

    Returns ``(Y, final_state)``. RAISES (RULE #1) on any shape mismatch,
    missing backend, or unsupported config — never a silent fallback.
    """
    if x.ndim != 4:
        raise ValueError(
            f"[ssd_chunk_scan_fused] x must be (batch,seqlen,nheads,headdim); "
            f"got shape {x.shape}"
        )
    batch, seqlen, nheads, headdim = x.shape
    if B.ndim != 4 or C.ndim != 4 or B.shape[2] != 1 or C.shape[2] != 1:
        raise ValueError(
            "[ssd_chunk_scan_fused] B and C must be (batch,seqlen,1,dstate) "
            f"(ngroups==1); got B={B.shape} C={C.shape}"
        )
    dstate = B.shape[3]
    if C.shape[3] != dstate:
        raise ValueError(
            f"[ssd_chunk_scan_fused] B/C dstate mismatch: {B.shape[3]} vs {C.shape[3]}"
        )
    L = int(chunk_size)
    if seqlen % L != 0:
        raise ValueError(
            f"[ssd_chunk_scan_fused] seqlen={seqlen} not divisible by chunk_size={L}"
        )
    if A.shape != (nheads,) or D.shape != (nheads,):
        raise ValueError(
            f"[ssd_chunk_scan_fused] A and D must be (nheads,)={nheads}; "
            f"got A={A.shape} D={D.shape}"
        )
    if h0.shape != (batch, nheads, headdim, dstate):
        raise ValueError(
            "[ssd_chunk_scan_fused] h0 must be (batch,nheads,headdim,dstate)="
            f"{(batch, nheads, headdim, dstate)}; got {h0.shape}"
        )

    backend = _backend()
    kernel = _make_kernel(backend, headdim, dstate, L, nheads, seqlen)

    # ONE launch: grid in threads = (P, 1, batch*nheads), threadgroup = (P,1,1).
    grid = (headdim, 1, batch * nheads)
    threadgroup = (headdim, 1, 1)

    outputs = kernel(
        inputs=[x, B, C, A.astype(mx.float32), dt.astype(mx.float32),
                D.astype(mx.float32), h0.astype(mx.float32)],
        template=[("T", x.dtype)],
        grid=grid,
        threadgroup=threadgroup,
        output_shapes=[x.shape, (batch, nheads, headdim, dstate)],
        output_dtypes=[x.dtype, mx.float32],
    )
    Y, final_state = outputs[0], outputs[1]
    return Y, final_state
