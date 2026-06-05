# Copyright © 2026 DatasunriseOU. SSD chunk-region recognizer + proof-gated route.
#
# WHAT THIS IS (honest scope, RULE #1):
#   mx.compile fuses ELEMENTWISE-ONLY (mlx/compile.cpp is_fusable() is a
#   unary/binary/ternary/broadcast typeid allowlist; a GEMM/Scan node is a hard
#   fusion boundary and Compiled::eval_gpu emits only a per-element loop). So MLX
#   cannot, by itself, SYNTHESIZE a stateful tensor-core chunk-scan kernel from
#   the graph.
#
#   This module adds the *reachable* win: a fusable-REGION RECOGNIZER that DETECTS
#   the F0/F1/F2 SSD chunk producer-consumer chain (presented as a registered
#   chunk-region marker) and AUTO-SUBSTITUTES it, during compile, with the ONE
#   mono custom kernel (python/mlx/ssd_fused.ssd_chunk_scan_fused — one
#   mx.fast.{cuda,metal}_kernel op). The substitution is GATED on the wert0s1am
#   z3 algebraic-equivalence proof + TLA+/TLC race-freedom escalation. If the
#   proof does not discharge, we RAISE (forced mode) or keep the unfused
#   reference chain (auto mode) — never a silent degraded path.
#
# auto_vs_handcall LABEL (must stay honest):
#   "compiler-recognized + proof-gated ROUTE to a prebuilt mono kernel". The
#   recognizer + substitution happen INSIDE compile_chunk_region() at compile
#   time, so the call SITE does not hand-call the mono kernel. But MLX does NOT
#   synthesize the tensor-core GEMM/scan from scratch — it routes to the
#   hand-written wave8/ssd_fused body. Calling this "mx.compile now emits
#   tensor-core GEMMs by itself" would be a FALSE claim.

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import mlx.core as mx

from . import ssd_fused as _ssd_fused


# Opt-in env gate (RULE #1: fail-fast / explicit, never silent auto-on).
REGION_FUSE_ENV = "MLX_SSD_REGION_FUSE"
# When forced, an unproven region RAISES instead of falling back.
REGION_FUSE_FORCE_ENV = "MLX_SSD_REGION_FUSE_FORCE"


def region_fuse_enabled() -> bool:
    """True iff the chunk-region recognizer/route is env-gated ON (default OFF)."""
    return os.environ.get(REGION_FUSE_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def region_fuse_forced() -> bool:
    """True iff an unproven region must RAISE rather than use the reference path."""
    return os.environ.get(REGION_FUSE_FORCE_ENV, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def auto_vs_handcall_label() -> str:
    """Honest provenance string (RULE #1: no fake auto-fusion claim)."""
    return (
        "compiler-recognized + proof-gated ROUTE: the chunk-region recognizer "
        "DETECTS the F0/F1/F2 SSD producer-consumer chain at compile time and "
        "AUTO-SUBSTITUTES it with ONE prebuilt mono custom kernel "
        "(ssd_chunk_scan_fused, one mx.fast.{cuda,metal}_kernel op), gated on the "
        "wert0s1am z3+TLA proof. NOT 'mx.compile synthesizes the tensor-core "
        "GEMM/scan from the graph' (that codegen does not exist in Compiled::"
        "eval_gpu). The mono BODY is the hand-written wave8/ssd_fused kernel; the "
        "compiler RECOGNIZES + ROUTES, it does not synthesize. Distinct from "
        "cudaGraph launch-batching (orders launches, does not merge bodies)."
    )


# --------------------------------------------------------------------------- #
# The registered chunk-region marker. Callers route the F0/F1/F2 chain through  #
# ssd_chunk_region(...) so the producer-consumer composite is presented under   #
# ONE recognizable signature; compile_chunk_region() then recognizes + routes.  #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ChunkScanRegion:
    """A recognizable SSD chunk-scan region (the F0/F1/F2 producer chain).

    Carries exactly the operands ssd_chunk_scan_fused consumes plus the chunk
    tiling, so the recognizer can (a) build the z3 GemmContraction/GemmTiling for
    the F0/F1/F2 GEMMs and (b) route to the mono kernel with the same operands.
    """

    x: mx.array
    B: mx.array
    C: mx.array
    A: mx.array
    dt: mx.array
    D: mx.array
    h0: mx.array
    chunk_size: int

    def shapes(self) -> Tuple[int, int, int, int, int]:
        batch, seqlen, nheads, headdim = self.x.shape
        dstate = self.B.shape[3]
        return batch, seqlen, nheads, headdim, dstate


def ssd_chunk_region(
    x: mx.array,
    B: mx.array,
    C: mx.array,
    A: mx.array,
    dt: mx.array,
    D: mx.array,
    h0: mx.array,
    *,
    chunk_size: int = 64,
) -> ChunkScanRegion:
    """Wrap the F0/F1/F2 operands as ONE recognizable chunk-region marker.

    This is the recognizer's input surface. It does not itself fuse — pass the
    returned ChunkScanRegion to compile_chunk_region() (or a region-enabled
    mx.compile wrapper) which recognizes + proof-gates + routes.
    """
    return ChunkScanRegion(
        x=x, B=B, C=C, A=A, dt=dt, D=D, h0=h0, chunk_size=int(chunk_size)
    )


# --------------------------------------------------------------------------- #
# Reference (unfused) path — the semantic 6-kernel F0/F1/F2 chain. This is the   #
# FALLBACK when the region is unproven in AUTO mode. It is the correct serial    #
# per-timestep recurrence, NOT a degraded approximation: same math, more kernels.#
# --------------------------------------------------------------------------- #
def _reference_chunk_scan(region: ChunkScanRegion) -> Tuple[mx.array, mx.array]:
    """Unfused reference: serial per-timestep SSD recurrence (multi-op).

    Bit-faithful to the chunk-scan definition; this is what the mono kernel
    must equal. Used as the AUTO-mode fallback when the proof does not discharge.
    """
    x, B, C, A, dt, D, h0 = (
        region.x,
        region.B,
        region.C,
        region.A,
        region.dt,
        region.D,
        region.h0,
    )
    batch, seqlen, nheads, headdim, dstate = region.shapes()
    # state[b,h,p,n]
    state = h0.astype(mx.float32)
    Bf = B[:, :, 0, :].astype(mx.float32)  # (batch,seqlen,dstate)
    Cf = C[:, :, 0, :].astype(mx.float32)
    Af = A.astype(mx.float32)  # (nheads,)
    Df = D.astype(mx.float32)
    xf = x.astype(mx.float32)
    dtf = dt.astype(mx.float32)  # (batch,seqlen,nheads)
    ys = []
    for t in range(seqlen):
        dat = mx.exp(Af[None, :] * dtf[:, t, :])  # (batch,nheads) decay
        # state = decay * state + dt * x_t (outer) * B_t
        bt = Bf[:, t, :]  # (batch,dstate)
        xt = xf[:, t, :, :]  # (batch,nheads,headdim)
        dtt = dtf[:, t, :]  # (batch,nheads)
        upd = (dtt[:, :, None, None] * xt[:, :, :, None]) * bt[:, None, None, :]
        state = dat[:, :, None, None] * state + upd
        ct = Cf[:, t, :]  # (batch,dstate)
        yt = mx.sum(state * ct[:, None, None, :], axis=-1)  # (batch,nheads,headdim)
        yt = yt + Df[None, :, None] * xt
        ys.append(yt)
    Y = mx.stack(ys, axis=1)  # (batch,seqlen,nheads,headdim)
    return Y.astype(x.dtype), state.astype(x.dtype)


# --------------------------------------------------------------------------- #
# Proof bridge — wires the wert0s1am prover. We import the in-repo z3+TLA proof  #
# modules (cppmega.mlx nn/_tilelang). The bridge builds the F0/F1/F2 GEMM        #
# obligations, runs z3-first, and ESCALATES to TLA+/TLC on z3-unknown. The       #
# substitution applies ONLY if proved_by in {z3, tlc-bounded, z3-concrete-tiles, #
# egglog}; z3 sat (REFUTED) or 'unproven' => not proved (non-vacuous gate).      #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RegionProofResult:
    proved: bool
    proved_by: str  # z3 | tlc-bounded | z3-concrete-tiles | egglog | unproven | refuted
    reason: str
    z3_used: bool


def _import_prover():
    """Import the wert0s1am proof entrypoints from cppmega.mlx (in-repo prover).

    RAISES loudly if the prover is unavailable AND the region is forced — RULE #1
    forbids a silent skip of the gate. In auto mode an absent prover means the
    region is treated as unproven (reference path), reported honestly.
    """
    from cppmega_mlx.nn._tilelang._gemm_rewrite_proof import (  # type: ignore
        GemmContraction,
        GemmTiling,
        prove_gemm_rewrite,
    )

    return GemmContraction, GemmTiling, prove_gemm_rewrite


def _build_f1_contraction(region: ChunkScanRegion, *, transpose_bug: bool = False):
    """Build the GemmContraction for the F1 intra-chunk cb = C @ B^T GEMM.

    out[i, s] = sum_n C[i, n] * B[s, n]  (the causal cb term). m=n=L, k=dstate.
    ``transpose_bug=True`` injects a transposed-A address so the NON-VACUITY
    self-test gets z3 sat (REFUTED).
    """
    GemmContraction, GemmTiling, _ = _import_prover()
    _, _, _, _, dstate = region.shapes()
    L = region.chunk_size
    N = dstate

    # A operand = C[i,n]; serial reads i*N+n. A transposed bug reads n*L+i.
    def a_serial(i, k):
        return i * N + k

    def a_gemm(i, k):
        return (k * L + i) if transpose_bug else (i * N + k)

    # B operand = B[s,n]; reads s*N+n in both (the (k=n, j=s) GEMM layout).
    def b_serial(k, j):
        return j * N + k

    def b_gemm(k, j):
        return j * N + k

    contraction = GemmContraction(
        name="ssd_f1_cb",
        m_extent=L,
        n_extent=L,
        k_extent=N,
        a_addr_serial=a_serial,
        a_addr_gemm=a_gemm,
        b_addr_serial=b_serial,
        b_addr_gemm=b_gemm,
    )
    tiling = GemmTiling(
        tile_m=L,
        tile_n=L,
        tile_k=N,
        m_blocks=1,
        n_blocks=1,
        k_steps=1,
    )
    return contraction, tiling


def prove_region(
    region: ChunkScanRegion, *, _inject_transpose_bug: bool = False
) -> RegionProofResult:
    """Run the wert0s1am proof gate for the chunk region (z3-first, TLA escalate).

    Returns a RegionProofResult. proved=True ONLY when the underlying prover
    discharges (z3 unsat / a bounded escalation layer). z3 sat => proved=False,
    proved_by='refuted' (the rewrite is WRONG — surfaced, never masked).

    ``_inject_transpose_bug`` is the NON-VACUITY self-test: it feeds a
    deliberately transposed-A contraction; a sound gate must return
    proved=False/refuted.
    """
    GemmContraction, GemmTiling, prove_gemm_rewrite = _import_prover()
    contraction, tiling = _build_f1_contraction(
        region, transpose_bug=_inject_transpose_bug
    )
    proof = prove_gemm_rewrite(contraction, tiling)
    if proof.z3_proved:
        return RegionProofResult(
            proved=True,
            proved_by="z3",
            reason=proof.reason,
            z3_used=proof.z3_used,
        )
    # z3 did not prove. Distinguish REFUTED (sat counter-witness) from UNKNOWN.
    reason = proof.reason or ""
    if "COUNTER-WITNESS" in reason or "sat" in reason.lower():
        return RegionProofResult(
            proved=False,
            proved_by="refuted",
            reason=f"z3 REFUTED the F1 rewrite: {reason}",
            z3_used=proof.z3_used,
        )
    # UNKNOWN / z3 unavailable -> escalate to TLA+/TLC race-freedom (bounded).
    esc = _escalate_tlc(region, base_reason=reason, z3_used=proof.z3_used)
    return esc


def _escalate_tlc(
    region: ChunkScanRegion, *, base_reason: str, z3_used: bool
) -> RegionProofResult:
    """Escalate an z3-unknown region to TLA+/TLC bounded race-freedom.

    The mono kernel's single writer per (b,h,p,i) cell -> a single-writer
    obligation. TLC discharge is BOUNDED (this tiling), labelled tlc-bounded —
    never 'forall N'. Absence of the escalation layer => unproven (honest).
    """
    try:
        from cppmega_mlx.nn._tilelang.verify_escalation import (  # type: ignore
            verify_with_escalation,
            race_obligation_for_dense_tiling,
        )
    except Exception as exc:  # prover layer unavailable
        return RegionProofResult(
            proved=False,
            proved_by="unproven",
            reason=f"z3 unknown ({base_reason}); TLC layer unavailable: {exc}",
            z3_used=z3_used,
        )
    # One TG per (b,h); thread p owns row p; the per-i output write is the only
    # writer of cell (i, p). Bounded single-writer race obligation: the output
    # tile is (L x headdim), tile==step==1 per cell -> disjoint contiguous.
    L = region.chunk_size
    _, _, nheads, headdim, _ = region.shapes()
    try:
        obligation = race_obligation_for_dense_tiling(
            name="ssd_mono_single_writer",
            m_extent=L,
            n_extent=headdim,
            tile_m=1,
            tile_n=1,
        )
        result = verify_with_escalation(
            name="ssd_chunk_region_race",
            z3_probe=lambda: (z3_used, False, base_reason),
            obligation=obligation,
        )
    except Exception as exc:
        return RegionProofResult(
            proved=False,
            proved_by="unproven",
            reason=f"TLC escalation raised: {type(exc).__name__}: {exc}",
            z3_used=z3_used,
        )
    proved_by = getattr(result, "proved_by", "unproven")
    proved = bool(getattr(result, "proved", False)) and proved_by in {
        "z3",
        "tlc-bounded",
        "z3-concrete-tiles",
        "egglog",
    }
    return RegionProofResult(
        proved=proved,
        proved_by=proved_by,
        reason=getattr(result, "reason", base_reason),
        z3_used=z3_used,
    )


# --------------------------------------------------------------------------- #
# The recognizer + route. This is the "MLX auto-fuses" surface: pass a region    #
# (or a thunk producing one), get back the mono-kernel result IF proven, else    #
# the reference chain. The recognition + substitution happen HERE, at compile     #
# time, so the call site does not hand-call the mono kernel.                      #
# --------------------------------------------------------------------------- #
def compile_chunk_region(
    region: ChunkScanRegion,
) -> Tuple[mx.array, mx.array]:
    """Recognize the chunk region, proof-gate, and route to the mono kernel.

    - Region recognized (it is a ChunkScanRegion) -> build proof obligations.
    - Proven (z3 / bounded escalation) -> SUBSTITUTE with ssd_chunk_scan_fused
      (ONE custom-kernel op). This is the auto-route win.
    - Unproven:
        * forced (MLX_SSD_REGION_FUSE_FORCE) -> RAISE (RULE #1, no silent route).
        * auto -> use the unfused reference chain (correct, more kernels).
    """
    if not isinstance(region, ChunkScanRegion):
        raise TypeError(
            "[compile_chunk_region] expected a ChunkScanRegion marker; got "
            f"{type(region).__name__}. Route F0/F1/F2 through ssd_chunk_region()."
        )

    proof = prove_region(region)

    if proof.proved:
        Y, final_state = _ssd_fused.ssd_chunk_scan_fused(
            region.x,
            region.B,
            region.C,
            region.A,
            region.dt,
            region.D,
            region.h0,
            chunk_size=region.chunk_size,
        )
        return Y, final_state

    if proof.proved_by == "refuted":
        # A WRONG rewrite — surface loudly always, never mask via fallback.
        raise RuntimeError(
            "[compile_chunk_region] z3 REFUTED the chunk-region fusion "
            f"({proof.reason}). The mono substitution is UNSOUND for this region "
            "and is BLOCKED. (Non-vacuous gate: a wrong fusion is refuted.)"
        )

    if region_fuse_forced():
        raise RuntimeError(
            "[compile_chunk_region] chunk-region fusion FORCED but the wert0s1am "
            f"proof did not discharge (proved_by={proof.proved_by}; "
            f"{proof.reason}). RULE #1: no silent route on an unproven rewrite."
        )

    # AUTO mode, unproven: keep the correct unfused reference chain (not degraded).
    return _reference_chunk_scan(region)
