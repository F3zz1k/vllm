# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 routed experts from the checkpoint's NVFP4 codes (Triton, XPU).

Weights stay 4-bit: E2M1 codes, an E4M3 scale per 16 values along K and a
per-expert fp32 factor. At load time the codes are repacked (`repack`) so
that byte b of a 32-value block holds value b in its low nibble and value
b + 16 in its high nibble: each nibble plane of a 16-byte load is one K=16
tile with one scale per row, i.e. one bf16 DPAS step, with no interleave or
reshape in the kernel. Tiles are decoded to bf16 in registers (W4A16).

Pipeline (graph-capturable: sizes are worst-case bounds, no host sync):
  1. route slots (token, k) sorted by expert; each expert's slots form
     `cdiv(count, BLOCK_M)` tiles; tile t finds its expert by binary search;
  2. GEMM1 on the gathered tokens, gate and up columns in one program, SiTU
     epilogue -> act (slots, I) in sorted order;
  3. GEMM2 -> times the routing weight, stored at the slot's (token, k) row;
     the k rows of a token are summed in fp32.
"""

from __future__ import annotations

import os

import torch
import triton
import triton.language as tl

INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"
NODEC = os.environ.get("INKLING2_MOE_NODEC", "0") == "1"


GROUP = 128  # values per repack group: byte b holds value b (low) and b + 64 (high)


def repack(packed: torch.Tensor) -> torch.Tensor:
    """Checkpoint nibble order (value 2i low, 2i+1 high) -> the kernel's: per
    128-value group, byte b holds value b (low nibble) and b + 64 (high), so
    each nibble plane of a 64-byte load matches 64 contiguous activations.
    (..., C/2) uint8."""
    lo = packed & 15
    hi = packed >> 4
    codes = torch.stack((lo, hi), -1).flatten(-2)  # (..., C) in value order
    blk = codes.unflatten(-1, (-1, 2, GROUP // 2))  # (..., C/128, half, 64)
    return (blk[..., 0, :] | (blk[..., 1, :] << 4)).flatten(-2).contiguous()


@triton.jit
def _e2m1(code):
    """E2M1 code (int32, 0..15) -> exact bf16, built from bits."""
    e = (code >> 1) & 3
    m = code & 1
    bits = tl.where(e > 0, ((e + 126) << 7) | (m << 6), m * (126 << 7))
    bits = bits | ((code & 8) << 12)
    return bits.to(tl.int16).to(tl.bfloat16, bitcast=True)


@triton.jit
def _mma_t(acc, Xs, r0, r_end, s_x, Wq, Ws, e, n0, NT, K, s_we, s_wn, s_se, s_sn, j0, j1,
           BN: tl.constexpr, BM: tl.constexpr, DOT32: tl.constexpr, NODEC: tl.constexpr):
    """acc (BN, BM) += W_e[n0:n0+BN, :K] @ Xs[r0:r0+BM, :K]^T, 128 values per step.

    The decoded weights are the dot's A operand (row-major, decoded in
    registers); the activation tile is transposed in registers: on the B70 an
    elementwise op on a transposed B-operand tile compiles ~50x slower. The
    per-16 scales are gathered to the tile's shape (cached loads), no reshape."""
    n = n0 + tl.arange(0, BN)
    nm = n < NT
    srow = Ws + e * s_se + n * s_sn
    for j in range(j0, j1):
        for q in tl.static_range(4):  # 16-value pieces of each nibble plane
            wp = tl.make_block_ptr(Wq + e * s_we, shape=(NT, K // 2), strides=(s_wn, 1),
                                   offsets=(n0, j * 64 + q * 16), block_shape=(BN, 16),
                                   order=(1, 0))
            byte = tl.load(wp, boundary_check=(0,), padding_option="zero")
            ap = tl.make_block_ptr(Xs, shape=(r_end, K), strides=(s_x, 1),
                                   offsets=(r0, j * 128 + q * 16), block_shape=(BM, 16),
                                   order=(1, 0))
            a0 = tl.load(ap, boundary_check=(0,), padding_option="zero")
            a1 = tl.load(tl.advance(ap, (0, 64)), boundary_check=(0,), padding_option="zero")
            if NODEC:
                wl = byte.to(tl.bfloat16)
                wh = wl
            else:
                b32 = byte.to(tl.int32)
                wl = _e2m1(b32 & 15)
                wh = _e2m1(b32 >> 4)
            slo = tl.load(srow + j * 8 + q, mask=nm, other=0.0).to(tl.float32)
            shi = tl.load(srow + j * 8 + 4 + q, mask=nm, other=0.0).to(tl.float32)
            if DOT32:
                pl = tl.dot(wl.to(tl.float32), tl.trans(a0.to(tl.float32)))
                ph = tl.dot(wh.to(tl.float32), tl.trans(a1.to(tl.float32)))
            else:
                pl = tl.dot(wl, tl.trans(a0))
                ph = tl.dot(wh, tl.trans(a1))
            acc += pl * slo[:, None] + ph * shi[:, None]
    return acc


@triton.jit
def _find_expert(TileStart, prog, E):
    lo = 0
    hi = E - 1
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if tl.load(TileStart + mid) <= prog:
            lo = mid
        else:
            hi = mid - 1
    return lo


@triton.jit
def _gemm1_kernel(Xs, Wq, Ws, P1, SlotStart, Count, TileStart, NumTiles,
                  H, I, E, s_x, s_we, s_wn, s_se, s_sn, s_ps, s_pr, J_PER,
                  BM: tl.constexpr, BN: tl.constexpr, DOT32: tl.constexpr,
                  NODEC: tl.constexpr):
    """Partial gate|up sums of K-split `program_id(2)` -> P1[split, slot, 2I] fp32."""
    prog = tl.program_id(0)
    if prog >= tl.load(NumTiles):
        return
    e = _find_expert(TileStart, prog, E)
    seg = tl.load(SlotStart + e)
    r_end = seg + tl.load(Count + e)
    r0 = seg + (prog - tl.load(TileStart + e)) * BM
    n0 = tl.program_id(1) * BN
    sp = tl.program_id(2)
    j0 = sp * J_PER
    j1 = tl.minimum(j0 + J_PER, H // 128)
    acc = _mma_t(tl.zeros((BN, BM), tl.float32), Xs, r0, r_end, s_x, Wq, Ws, e, n0, 2 * I, H,
                 s_we, s_wn, s_se, s_sn, j0, j1, BN, BM, DOT32, NODEC)
    n = n0 + tl.arange(0, BN)
    r = r0 + tl.arange(0, BM)
    tl.store(P1 + sp * s_ps + r[None, :] * s_pr + n[:, None], acc,
             mask=(r < r_end)[None, :] & (n < 2 * I)[:, None])


@triton.jit
def _situ_kernel(P1, Act, F, SlotExpert, splits, I, s_ps, s_pr, s_a, gate_cap, up_cap,
                 BLOCK: tl.constexpr):
    """act[slot] = SiTU(f * sum_split gate, f * sum_split up), GEMM1 output rounded to bf16."""
    r = tl.program_id(0)
    c = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    cm = c < I
    g = tl.zeros((BLOCK,), tl.float32)
    u = tl.zeros((BLOCK,), tl.float32)
    for sp in range(splits):
        g += tl.load(P1 + sp * s_ps + r * s_pr + c, mask=cm, other=0.0)
        u += tl.load(P1 + sp * s_ps + r * s_pr + I + c, mask=cm, other=0.0)
    f = tl.load(F + tl.load(SlotExpert + r))
    g = (g * f).to(tl.bfloat16).to(tl.float32)
    u = (u * f).to(tl.bfloat16).to(tl.float32)
    tg = 2.0 / (1.0 + tl.exp(-2.0 * g / gate_cap)) - 1.0
    tu = 2.0 / (1.0 + tl.exp(-2.0 * u / up_cap)) - 1.0
    act = gate_cap * tg / (1.0 + tl.exp(-g)) * (up_cap * tu)
    tl.store(Act + r * s_a + c, act.to(tl.bfloat16), mask=cm)


@triton.jit
def _gemm2_kernel(Act, Wq, Ws, F, P2, SortedSlot, SlotStart, Count, TileStart, NumTiles,
                  TopW, H, I, E, s_a, s_we, s_wn, s_se, s_sn, s_ps, s_po, J_PER,
                  BM: tl.constexpr, BN: tl.constexpr, DOT32: tl.constexpr,
                  NODEC: tl.constexpr):
    """Partial down-proj of K-split `program_id(2)`, times f and the routing
    weight -> P2[split, token * K + k, H] fp32."""
    prog = tl.program_id(0)
    if prog >= tl.load(NumTiles):
        return
    e = _find_expert(TileStart, prog, E)
    seg = tl.load(SlotStart + e)
    r_end = seg + tl.load(Count + e)
    r0 = seg + (prog - tl.load(TileStart + e)) * BM
    n0 = tl.program_id(1) * BN
    sp = tl.program_id(2)
    j0 = sp * J_PER
    j1 = tl.minimum(j0 + J_PER, I // 128)
    y = _mma_t(tl.zeros((BN, BM), tl.float32), Act, r0, r_end, s_a, Wq, Ws, e, n0, H, I,
               s_we, s_wn, s_se, s_sn, j0, j1, BN, BM, DOT32, NODEC)
    r = r0 + tl.arange(0, BM)
    rm = r < r_end
    orig = tl.load(SortedSlot + r, mask=rm, other=0)  # token * K + k
    tw = tl.load(TopW + orig, mask=rm, other=0.0) * tl.load(F + e)
    n = n0 + tl.arange(0, BN)
    tl.store(P2 + sp * s_ps + orig[None, :] * s_po + n[:, None], y * tw[None, :],
             mask=rm[None, :] & (n < H)[:, None])


def _cfg(S: int, E: int):
    small = S <= 4 * E
    g = lambda k, d: int(os.environ.get(k, d))
    if small:
        return g("INKLING2_MOE_BM", 16), g("INKLING2_MOE_BN", 64), g("INKLING2_MOE_WARPS", 4), \
            g("INKLING2_MOE_SPLIT", 6)
    return g("INKLING2_MOE_BM_L", 64), g("INKLING2_MOE_BN_L", 128), g("INKLING2_MOE_WARPS_L", 8), 1


def nvfp4_moe(x: torch.Tensor, topk_w: torch.Tensor, topk_ids: torch.Tensor,
              w13: torch.Tensor, s13: torch.Tensor, f13: torch.Tensor,
              w2: torch.Tensor, s2: torch.Tensor, f2: torch.Tensor,
              gate_cap: float, up_cap: float) -> torch.Tensor:
    """x (T, H) bf16; topk (T, K); w13 repacked uint8 (E, 2I, H/2) [gate; up],
    s13 e4m3 (E, 2I, H/16), f13 (E,); w2 repacked (E, H, I/2), s2, f2."""
    T, H = x.shape
    K = topk_ids.shape[1]
    E, I = w13.shape[0], w13.shape[1] // 2
    S = T * K
    bm, bn, nw, split = _cfg(S, E)
    flat = topk_ids.reshape(-1).long()
    order = torch.argsort(flat, stable=True)
    xs = x.index_select(0, order // K)  # tokens in sorted slot order (block loads)
    sorted_slot = order.to(torch.int32)
    slot_expert = flat.index_select(0, order).to(torch.int32)
    # bincount sizes its output on the host: not graph-safe; scatter_add is.
    count = torch.zeros(E, dtype=torch.int32, device=x.device).scatter_add_(
        0, flat, torch.ones_like(flat, dtype=torch.int32))
    slot_start = (torch.cumsum(count, 0) - count).to(torch.int32)
    tiles = (count + bm - 1) // bm
    tile_start = (torch.cumsum(tiles, 0) - tiles).to(torch.int32)
    num_tiles = tiles.sum().to(torch.int32).reshape(1)
    max_tiles = triton.cdiv(S, bm) + E
    s1 = min(split, H // 128)
    j1 = triton.cdiv(H // 128, s1)
    p1 = torch.empty((s1, S, 2 * I), dtype=torch.float32, device=x.device)
    _gemm1_kernel[(max_tiles, triton.cdiv(2 * I, bn), s1)](
        xs, w13, s13, p1, slot_start, count, tile_start, num_tiles,
        H, I, E, xs.stride(0), w13.stride(0), w13.stride(1), s13.stride(0), s13.stride(1),
        p1.stride(0), p1.stride(1), j1, BM=bm, BN=bn, DOT32=INTERP, NODEC=NODEC, num_warps=nw)
    act = torch.empty((S, I), dtype=torch.bfloat16, device=x.device)
    _situ_kernel[(S, triton.cdiv(I, 256))](p1, act, f13, slot_expert, s1, I, p1.stride(0),
                                           p1.stride(1), act.stride(0), gate_cap, up_cap,
                                           BLOCK=256)
    s2n = min(split, I // 128)
    j2 = triton.cdiv(I // 128, s2n)
    p2 = torch.empty((s2n, S, H), dtype=torch.float32, device=x.device)
    _gemm2_kernel[(max_tiles, triton.cdiv(H, bn), s2n)](
        act, w2, s2, f2, p2, sorted_slot, slot_start, count, tile_start, num_tiles,
        topk_w.reshape(-1).float().contiguous(), H, I, E, act.stride(0), w2.stride(0),
        w2.stride(1), s2.stride(0), s2.stride(1), p2.stride(0), p2.stride(1), j2,
        BM=bm, BN=bn, DOT32=INTERP, NODEC=NODEC, num_warps=nw)
    return p2.view(s2n, T, K, H).sum((0, 2)).to(torch.bfloat16)
