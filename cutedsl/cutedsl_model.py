"""
cutedsl_model.py
CuteDSL implementations of DDiT components.
Step 1: Pre-Attention Norm + Modulate (DynTanh variant).
Step 2: Input Depthwise Convs.
"""

import math
import time
from typing import Optional

import torch
import torch.nn as nn

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack


# ---------------------------------------------------------------------------
# Step 1: Pre-Attention Norm + Modulate (DynTanh + AdaLN)
# ---------------------------------------------------------------------------

@cute.kernel
def _cute_fused_modulate_dyntanh_kernel(
    mX: cute.Tensor,
    mShift: cute.Tensor,
    mScale: cute.Tensor,
    mAlpha: cute.Tensor,
    mGamma: cute.Tensor,
    mBeta: cute.Tensor,
    mOut: cute.Tensor,
    T: cutlass.Int32,
    thr_layout: cute.Layout,
    val_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    gX = cute.zipped_divide(mX, tiler_mn)
    gOut = cute.zipped_divide(mOut, tiler_mn)

    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mX.element_type)
    tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
    thr_copy = tiled_copy.get_slice(tidx)

    blk_coord = ((None, None), bidx)
    blkX = gX[blk_coord]
    blkOut = gOut[blk_coord]

    thrX = thr_copy.partition_S(blkX)
    thrOut = thr_copy.partition_S(blkOut)

    frgX = cute.make_fragment_like(thrX)
    frgOut = cute.make_fragment_like(thrOut)

    # Predication
    idX = cute.make_identity_tensor(mX.shape)
    gId = cute.zipped_divide(idX, tiler_mn)
    blkId = gId[blk_coord]
    thrId = thr_copy.partition_S(blkId)
    frgPred = cute.make_rmem_tensor(thrId.shape, cutlass.Boolean)
    for i in range(0, cute.size(frgPred), 1):
        frgPred[i] = cute.elem_less(thrId[i], mX.shape)

    # Load x
    cute.copy(copy_atom, thrX, frgX, pred=frgPred)
    x_val = frgX.load().to(cutlass.Float32)

    # Load alpha scalar
    alpha_val = mAlpha[0].to(cutlass.Float32)

    # Compute batch index from block index (each block = 1 row)
    batch_idx = bidx // T

    # Load shift (vectorized via tiled copy + local_tile)
    blkShift = cute.local_tile(mShift, tiler_mn, (batch_idx, 0))
    thrShift = thr_copy.partition_S(blkShift)
    frgShift = cute.make_fragment_like(thrShift)
    cute.copy(copy_atom, thrShift, frgShift, pred=frgPred)
    shift_val = frgShift.load().to(cutlass.Float32)

    # Load scale
    blkScale = cute.local_tile(mScale, tiler_mn, (batch_idx, 0))
    thrScale = thr_copy.partition_S(blkScale)
    frgScale = cute.make_fragment_like(thrScale)
    cute.copy(copy_atom, thrScale, frgScale, pred=frgPred)
    scale_val = frgScale.load().to(cutlass.Float32)

    # Load gamma (1D viewed as 2D row, always row 0)
    gamma_2d = cute.make_tensor(mGamma.iterator, cute.make_layout((1, mGamma.shape[0]), stride=(0, 1)))
    blkGamma = cute.local_tile(gamma_2d, tiler_mn, (0, 0))
    thrGamma = thr_copy.partition_S(blkGamma)
    frgGamma = cute.make_fragment_like(thrGamma)
    cute.copy(copy_atom, thrGamma, frgGamma, pred=frgPred)
    gamma_val = frgGamma.load().to(cutlass.Float32)

    # Load beta
    beta_2d = cute.make_tensor(mBeta.iterator, cute.make_layout((1, mBeta.shape[0]), stride=(0, 1)))
    blkBeta = cute.local_tile(beta_2d, tiler_mn, (0, 0))
    thrBeta = thr_copy.partition_S(blkBeta)
    frgBeta = cute.make_fragment_like(thrBeta)
    cute.copy(copy_atom, thrBeta, frgBeta, pred=frgPred)
    beta_val = frgBeta.load().to(cutlass.Float32)

    # DynTanh + Modulate elementwise on fragment
    dt = cute.math.tanh(x_val * alpha_val) * gamma_val + beta_val
    out_val = dt * (cutlass.Float32(1.0) + scale_val) + shift_val
    frgOut.store(out_val.to(mOut.element_type))

    cute.copy(copy_atom, frgOut, thrOut, pred=frgPred)


@cute.jit
def _cute_fused_modulate_dyntanh_jit(
    x_tensor, shift_tensor, scale_tensor,
    alpha_tensor, gamma_tensor, beta_tensor, out_tensor,
    T_val,
    copy_bits: cutlass.Constexpr = 128
):
    dtype = x_tensor.element_type
    vector_size = copy_bits // dtype.width

    # Row-wise tiling: 1 row per block, threads vectorize over C
    thr_layout = cute.make_layout((1, 128), stride=(0, 1))
    val_layout = cute.make_layout((1, vector_size), stride=(0, 1))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    gX = cute.zipped_divide(x_tensor, tiler_mn)

    _cute_fused_modulate_dyntanh_kernel.set_name_prefix("cute_fused_modulate_dyntanh")
    _cute_fused_modulate_dyntanh_kernel(
        x_tensor, shift_tensor, scale_tensor, alpha_tensor, gamma_tensor, beta_tensor, out_tensor,
        T_val,
        thr_layout, val_layout
    ).launch(
        grid=[cute.size(gX, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


def cute_fused_modulate_dyntanh(
    x: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
    alpha: torch.Tensor,
    gamma: torch.Tensor,
    beta: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Fused DynTanh + Modulate using CuteDSL.
    x: (B, T, C)
    shift, scale: (B, 1, C)
    alpha: (1,)
    gamma: (C,)
    beta: (C,) or None
    returns: (B, T, C)
    """
    B, T, C = x.shape
    x_flat = x.view(-1, C)
    shift_flat = shift.squeeze(1)  # (B, C)
    scale_flat = scale.squeeze(1)   # (B, C)
    out = torch.empty_like(x_flat)

    if beta is None:
        beta = torch.zeros(C, device=x.device, dtype=x.dtype)

    x_ct = from_dlpack(x_flat.detach(), assumed_align=16)
    shift_ct = from_dlpack(shift_flat.detach(), assumed_align=16)
    scale_ct = from_dlpack(scale_flat.detach(), assumed_align=16)
    alpha_ct = from_dlpack(alpha.detach(), assumed_align=16)
    gamma_ct = from_dlpack(gamma.detach(), assumed_align=16)
    beta_ct = from_dlpack(beta.detach(), assumed_align=16)
    out_ct = from_dlpack(out, assumed_align=16)

    if not hasattr(cute_fused_modulate_dyntanh, "_compiled"):
        print("[CuteDSL] Compiling fused_modulate_dyntanh ...")
        start = time.time()
        cute_fused_modulate_dyntanh._compiled = cute.compile(
            _cute_fused_modulate_dyntanh_jit,
            x_ct, shift_ct, scale_ct, alpha_ct, gamma_ct, beta_ct, out_ct,
            cutlass.Int32(T),
            options="--opt-level 2 --gpu-arch sm_86"
        )
        print(f"[CuteDSL] Compilation took {time.time() - start:.2f}s")

    cute_fused_modulate_dyntanh._compiled(
        x_ct, shift_ct, scale_ct, alpha_ct, gamma_ct, beta_ct, out_ct,
        cutlass.Int32(T),
    )
    return out.view(B, T, C)


# ---------------------------------------------------------------------------
# Step 2: Input Depthwise Convs
# ---------------------------------------------------------------------------

@cute.kernel
def _cute_fused_depthwise_convs_kernel(
    mX: cute.Tensor,
    mW1: cute.Tensor,
    mB1: cute.Tensor,
    mW2: cute.Tensor,
    mB2: cute.Tensor,
    mOut: cute.Tensor,
    T: cutlass.Int32,
    num_c_tiles: cutlass.Int32,
    thr_layout: cute.Layout,
    val_layout: cute.Layout,
):
    tidx, _, _ = cute.arch.thread_idx()
    bidx, _, _ = cute.arch.block_idx()

    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)
    gX = cute.zipped_divide(mX, tiler_mn)
    gOut = cute.zipped_divide(mOut, tiler_mn)

    copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), mX.element_type)
    tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
    thr_copy = tiled_copy.get_slice(tidx)

    # Compute row and c_tile from flat block index
    row = bidx // num_c_tiles
    c_tile = bidx % num_c_tiles
    t = row % T

    blk_coord = ((None, None), (row, c_tile))
    blkX = gX[blk_coord]
    blkOut = gOut[blk_coord]

    thrX = thr_copy.partition_S(blkX)
    thrOut = thr_copy.partition_S(blkOut)

    frgX = cute.make_fragment_like(thrX)
    frgOut = cute.make_fragment_like(thrOut)

    # Predication
    idX = cute.make_identity_tensor(mX.shape)
    gId = cute.zipped_divide(idX, tiler_mn)
    blkId = gId[blk_coord]
    thrId = thr_copy.partition_S(blkId)
    frgPred = cute.make_rmem_tensor(thrId.shape, cutlass.Boolean)
    for i in range(0, cute.size(frgPred), 1):
        frgPred[i] = cute.elem_less(thrId[i], mX.shape)

    # Load current row
    cute.copy(copy_atom, thrX, frgX, pred=frgPred)
    x_0 = frgX.load().to(cutlass.Float32)

    # Boundary masks for inter values
    mask_m1 = cutlass.Float32(1.0)
    if t < 1:
        mask_m1 = cutlass.Float32(0.0)
    mask_p1 = cutlass.Float32(1.0)
    if t >= T - 1:
        mask_p1 = cutlass.Float32(0.0)

    # Load neighbors with zero-fill for OOB
    frg_m2 = cute.make_fragment_like(frgX)
    frg_m1 = cute.make_fragment_like(frgX)
    frg_p1 = cute.make_fragment_like(frgX)
    frg_p2 = cute.make_fragment_like(frgX)
    frg_m2.fill(0.0)
    frg_m1.fill(0.0)
    frg_p1.fill(0.0)
    frg_p2.fill(0.0)

    if t >= 2:
        blk_m2 = gX[((None, None), (row - 2, c_tile))]
        thr_m2 = thr_copy.partition_S(blk_m2)
        cute.copy(copy_atom, thr_m2, frg_m2, pred=frgPred)
    if t >= 1:
        blk_m1 = gX[((None, None), (row - 1, c_tile))]
        thr_m1 = thr_copy.partition_S(blk_m1)
        cute.copy(copy_atom, thr_m1, frg_m1, pred=frgPred)
    if t < T - 1:
        blk_p1 = gX[((None, None), (row + 1, c_tile))]
        thr_p1 = thr_copy.partition_S(blk_p1)
        cute.copy(copy_atom, thr_p1, frg_p1, pred=frgPred)
    if t < T - 2:
        blk_p2 = gX[((None, None), (row + 2, c_tile))]
        thr_p2 = thr_copy.partition_S(blk_p2)
        cute.copy(copy_atom, thr_p2, frg_p2, pred=frgPred)

    x_m2 = frg_m2.load().to(cutlass.Float32)
    x_m1 = frg_m1.load().to(cutlass.Float32)
    x_p1 = frg_p1.load().to(cutlass.Float32)
    x_p2 = frg_p2.load().to(cutlass.Float32)

    for i in range(0, cute.size(frgX), 1):
        if frgPred[i]:
            coord = thrId[i]
            col = coord[1]

            w1_0 = mW1[col * 3 + 0].to(cutlass.Float32)
            w1_1 = mW1[col * 3 + 1].to(cutlass.Float32)
            w1_2 = mW1[col * 3 + 2].to(cutlass.Float32)
            b1 = mB1[col].to(cutlass.Float32)

            conv1_m1 = x_m2[i] * w1_0 + x_m1[i] * w1_1 + x_0[i] * w1_2 + b1
            conv1_0  = x_m1[i] * w1_0 + x_0[i] * w1_1 + x_p1[i] * w1_2 + b1
            conv1_p1 = x_0[i] * w1_0 + x_p1[i] * w1_1 + x_p2[i] * w1_2 + b1

            inter_m1 = (x_m1[i] + conv1_m1) * mask_m1
            inter_0  = x_0[i] + conv1_0
            inter_p1 = (x_p1[i] + conv1_p1) * mask_p1

            sig_m1 = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.math.exp(-inter_m1))
            sig_0  = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.math.exp(-inter_0))
            sig_p1 = cutlass.Float32(1.0) / (cutlass.Float32(1.0) + cute.math.exp(-inter_p1))
            silu_m1 = inter_m1 * sig_m1
            silu_0  = inter_0  * sig_0
            silu_p1 = inter_p1 * sig_p1

            w2_0 = mW2[col * 3 + 0].to(cutlass.Float32)
            w2_1 = mW2[col * 3 + 1].to(cutlass.Float32)
            w2_2 = mW2[col * 3 + 2].to(cutlass.Float32)
            b2 = mB2[col].to(cutlass.Float32)

            conv2_0 = silu_m1 * w2_0 + silu_0 * w2_1 + silu_p1 * w2_2 + b2
            out_0 = inter_0 + conv2_0

            frgOut[i] = out_0.to(mOut.element_type)

    cute.copy(copy_atom, frgOut, thrOut, pred=frgPred)


@cute.jit
def _cute_fused_depthwise_convs_jit(
    x_tensor, w1_tensor, b1_tensor, w2_tensor, b2_tensor, out_tensor,
    T_val,
    num_c_tiles: cutlass.Constexpr,
    num_threads: cutlass.Constexpr = 128,
    copy_bits: cutlass.Constexpr = 128,
):
    dtype = x_tensor.element_type
    vector_size = copy_bits // dtype.width

    thr_layout = cute.make_layout((1, num_threads), stride=(0, 1))
    val_layout = cute.make_layout((1, vector_size), stride=(0, 1))
    tiler_mn, tv_layout = cute.make_layout_tv(thr_layout, val_layout)

    gX = cute.zipped_divide(x_tensor, tiler_mn)

    _cute_fused_depthwise_convs_kernel.set_name_prefix("cute_fused_depthwise_convs")
    _cute_fused_depthwise_convs_kernel(
        x_tensor, w1_tensor, b1_tensor, w2_tensor, b2_tensor, out_tensor,
        T_val, num_c_tiles,
        thr_layout, val_layout
    ).launch(
        grid=[cute.size(gX, mode=[1]), 1, 1],
        block=[cute.size(tv_layout, mode=[0]), 1, 1],
    )


# Cache for compiled conv variants
_convs_compiled_cache = {}


def cute_fused_depthwise_convs(
    x: torch.Tensor,
    w1: torch.Tensor,
    b1: Optional[torch.Tensor],
    w2: torch.Tensor,
    b2: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Fused depthwise convs: x + conv1(x) + conv2(SiLU(x + conv1(x)))
    x: (B, T, C)
    w1, w2: (C, 1, 3)
    b1, b2: (C,) or None
    returns: (B, T, C)
    """
    B, T, C = x.shape

    # Dynamic configuration to ensure num_c_tiles == 1 and avoid multi-tile bugs
    NUM_THREADS = 128
    dtype_width = 32 if x.dtype == torch.float32 else 16
    # Need NUM_THREADS * (COPY_BITS // dtype_width) >= C
    # => COPY_BITS >= C * dtype_width / NUM_THREADS
    min_copy_bits = 16
    while min_copy_bits < (C * dtype_width + NUM_THREADS - 1) // NUM_THREADS:
        min_copy_bits *= 2
    COPY_BITS = min_copy_bits

    x_flat = x.view(-1, C)
    out = torch.empty_like(x_flat)

    if b1 is None:
        b1 = torch.zeros(C, device=x.device, dtype=x.dtype)
    if b2 is None:
        b2 = torch.zeros(C, device=x.device, dtype=x.dtype)

    # Flatten weights to 1D as expected by kernel: (C*3,)
    w1_flat = w1.view(-1)
    w2_flat = w2.view(-1)

    x_ct = from_dlpack(x_flat.detach(), assumed_align=16)
    w1_ct = from_dlpack(w1_flat.detach(), assumed_align=16)
    b1_ct = from_dlpack(b1.detach(), assumed_align=16)
    w2_ct = from_dlpack(w2_flat.detach(), assumed_align=16)
    b2_ct = from_dlpack(b2.detach(), assumed_align=16)
    out_ct = from_dlpack(out, assumed_align=16)

    vector_size = COPY_BITS // dtype_width
    tile_c = NUM_THREADS * vector_size
    num_c_tiles_val = (C + tile_c - 1) // tile_c

    cache_key = (NUM_THREADS, COPY_BITS, C)
    if cache_key not in _convs_compiled_cache:
        print(f"[CuteDSL] Compiling fused_depthwise_convs (threads={NUM_THREADS}, copy_bits={COPY_BITS}, num_c_tiles={num_c_tiles_val}) ...")
        start = time.time()
        _convs_compiled_cache[cache_key] = cute.compile(
            _cute_fused_depthwise_convs_jit,
            x_ct, w1_ct, b1_ct, w2_ct, b2_ct, out_ct,
            cutlass.Int32(T),
            num_c_tiles_val,
            NUM_THREADS,
            COPY_BITS,
            options="--opt-level 2 --gpu-arch sm_86"
        )
        print(f"[CuteDSL] Compilation took {time.time() - start:.2f}s")

    _convs_compiled_cache[cache_key](
        x_ct, w1_ct, b1_ct, w2_ct, b2_ct, out_ct,
        cutlass.Int32(T),
    )
    return out.view(B, T, C)


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------
def test_conv():
    import torch.nn.functional as F
    device = "cuda"
    B, T, C = 2, 3, 4
    x = torch.tensor([[[1.0, 2.0, 3.0, 4.0],
                       [5.0, 6.0, 7.0, 8.0],
                       [9.0, 10.0, 11.0, 12.0]]], device=device, dtype=torch.bfloat16)
    w1 = torch.ones(C, 1, 3, device=device, dtype=torch.bfloat16)
    b1 = torch.zeros(C, device=device, dtype=torch.bfloat16)
    w2 = torch.ones(C, 1, 3, device=device, dtype=torch.bfloat16)
    b2 = torch.zeros(C, device=device, dtype=torch.bfloat16)

    # Reference
    x_ref = x.clone()
    conv1 = torch.nn.Conv1d(C, C, kernel_size=3, padding=1, groups=C, bias=True).cuda().to(torch.bfloat16)
    conv1.weight.data = w1
    conv1.bias.data = b1
    conv2 = torch.nn.Conv1d(C, C, kernel_size=3, padding=1, groups=C, bias=True).cuda().to(torch.bfloat16)
    conv2.weight.data = w2
    conv2.bias.data = b2
    ref = x_ref + conv1(x_ref.transpose(1, 2)).transpose(1, 2)
    ref = ref + conv2(F.silu(ref).transpose(1, 2)).transpose(1, 2)

    # CuteDSL
    out = cute_fused_depthwise_convs(x, w1, b1, w2, b2)
    err = (out.to(torch.float32) - ref.to(torch.float32)).abs().max().item()
    print(f"Conv Max Abs Error (Cute vs Ref): {err:.6f}")
    assert err <= 0.05, f"Error too large: {err}"
    print("Conv correctness PASSED")


# ---------------------------------------------------------------------------
# Main test entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    device = "cuda"
    B, T, C = 4, 16, 64
    x = torch.randn(B, T, C, device=device, dtype=torch.bfloat16)
    shift = torch.randn(B, 1, C, device=device, dtype=torch.bfloat16)
    scale = torch.randn(B, 1, C, device=device, dtype=torch.bfloat16)
    alpha = torch.ones(1, device=device, dtype=torch.bfloat16)
    gamma = torch.ones(C, device=device, dtype=torch.bfloat16)
    beta = torch.zeros(C, device=device, dtype=torch.bfloat16)

    # Reference
    ref = torch.tanh(alpha * x) * gamma + beta
    ref = ref * (1 + scale) + shift

    # CuteDSL
    out = cute_fused_modulate_dyntanh(x, shift, scale, alpha, gamma, beta)
    err = (out.to(torch.float32) - ref.to(torch.float32)).abs().max().item()
    print(f"Max Abs Error (Cute vs Ref): {err:.6f}")
    assert err <= 0.03125, f"Error too large: {err}"
    print("Step 1 correctness PASSED")
    test_conv()
