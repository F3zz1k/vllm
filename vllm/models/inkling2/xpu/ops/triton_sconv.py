# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 short conv over the paged conv state (Triton, XPU).

`out[t] = (sum_lag w[lag] * x[t - lag] + x[t]) / sqrt(k + 1)` (static taps), or
with dynamic per-token per-head taps added to the static ones and
`/ sqrt(2k + 1)` (full-attention K/V). Past inputs of the same step come from
`x` (rows of the same request, `query_start[t]` tells where it starts), older
ones from the conv state, which this kernel never writes and reads in the
same slots: each token stores its own input at its own conv slot.

Optional fused epilogues: `NORM` 1 = L2 clip per head, 2 = unweighted RMS
norm per head (one head per program then), and a store of the result into
the visit's attention KV cache at the token's KV slot (`KV_OFF` = 0 for K,
`D` for V in the K|V content dim).
"""

from __future__ import annotations

import math
import torch
import triton
import triton.language as tl


@triton.jit
def _sconv_kernel(
    X, Out, Conv, Slots, BlockTable, SeqIdx, QStart, Pos, W, Dyn, KV, KVSlots,
    C, hph, off, eps, clip, inv_norm, dyn_heads_w,
    s_x, s_o, s_c_b, s_c_h, s_c_n, s_bt, s_w0, s_w1, s_dyn, s_kv_b, s_kv_h, s_kv_n, KV_OFF,
    K: tl.constexpr, BLOCK_C: tl.constexpr, DYNAMIC: tl.constexpr, NORM: tl.constexpr,
    STORE_KV: tl.constexpr, PAGE: tl.constexpr, KV_PAGE: tl.constexpr,
):
    t = tl.program_id(0)
    cb = tl.program_id(1)
    c = cb * BLOCK_C + tl.arange(0, BLOCK_C)
    cm = c < C
    head = c // hph
    j = c % hph
    slot = tl.load(Slots + t)
    req = tl.load(SeqIdx + t)
    q0 = tl.load(QStart + t)
    p = tl.load(Pos + t)
    x = tl.load(X + t * s_x + c, mask=cm, other=0.0).to(tl.float32)
    acc = tl.zeros([BLOCK_C], tl.float32)
    for lag in tl.static_range(K):
        pp = p - lag
        if lag == 0:
            v = x
        else:
            in_batch = t - lag >= q0
            if in_batch:
                v = tl.load(X + (t - lag) * s_x + c, mask=cm, other=0.0).to(tl.float32)
            else:
                ppc = tl.maximum(pp, 0)
                blk = tl.load(BlockTable + req * s_bt + ppc // PAGE).to(tl.int64)
                v = tl.load(Conv + blk * s_c_b + head * s_c_h + (ppc % PAGE) * s_c_n + off + j,
                            mask=cm & (pp >= 0), other=0.0).to(tl.float32)
        if DYNAMIC:
            # static_w (K, C): tap lag = backward distance
            w = tl.load(W + lag * s_w0 + c * s_w1, mask=cm, other=0.0).to(tl.float32)
            w += tl.load(Dyn + t * s_dyn + (c // dyn_heads_w) * K + lag, mask=cm, other=0.0).to(
                tl.float32)
        else:
            # conv1d weight (C, K): tap K-1 is the current token
            w = tl.load(W + c * s_w0 + (K - 1 - lag) * s_w1, mask=cm, other=0.0).to(tl.float32)
        acc += w * v
    y = ((acc + x) * inv_norm).to(tl.bfloat16).to(tl.float32)  # the conv output is bf16
    if NORM == 1:
        n = tl.sqrt(tl.sum(y * y, 0))
        y = y * tl.minimum(clip / (n + 1e-6), 1.0)
    elif NORM == 2:
        y = y * tl.rsqrt(tl.sum(y * y, 0) / BLOCK_C + eps)
    tl.store(Out + t * s_o + c, y.to(Out.dtype.element_ty), mask=cm)
    if slot >= 0:
        cs = slot
        tl.store(Conv + (cs // PAGE).to(tl.int64) * s_c_b + head * s_c_h + (cs % PAGE) * s_c_n
                 + off + j, x.to(Conv.dtype.element_ty), mask=cm)
    if STORE_KV:
        ks = tl.load(KVSlots + t)
        if ks >= 0:
            tl.store(KV + (ks // KV_PAGE).to(tl.int64) * s_kv_b + head * s_kv_h
                     + (ks % KV_PAGE) * s_kv_n + KV_OFF + j, y.to(KV.dtype.element_ty), mask=cm)


def short_conv(x: torch.Tensor, conv: torch.Tensor, stream_off: int, hph: int,
               slots: torch.Tensor, block_table: torch.Tensor, seq_idx: torch.Tensor,
               query_start: torch.Tensor, positions: torch.Tensor, weight: torch.Tensor,
               dyn: torch.Tensor | None = None, norm: int = 0, clip: float = 0.0,
               eps: float = 1e-6, kv: torch.Tensor | None = None,
               kv_slots: torch.Tensor | None = None, kv_off: int = 0) -> torch.Tensor:
    """x (T, C) bf16 (C = heads * hph); conv [B, H, page, W]; weight (C, k)
    static or (k, C) with `dyn` (T, H * k); returns (T, C) bf16."""
    T, C = x.shape
    k = weight.shape[1] if dyn is None else weight.shape[0]
    out = torch.empty((T, C), dtype=torch.bfloat16, device=x.device)
    if T == 0:
        return out
    if norm:
        assert C % hph == 0
        block_c = hph
        assert block_c & (block_c - 1) == 0, "per-head norm needs a power-of-two head"
    else:
        block_c = min(1024, triton.next_power_of_2(C))
    inv = 1.0 / math.sqrt((2 * k + 1) if dyn is not None else (k + 1))
    store_kv = kv is not None
    kvt = kv if store_kv else conv
    _sconv_kernel[(T, triton.cdiv(C, block_c))](
        x, out, conv, slots, block_table, seq_idx, query_start, positions, weight,
        dyn if dyn is not None else x, kvt, kv_slots if store_kv else slots,
        C, hph, stream_off, eps, clip, inv, hph if dyn is not None else 1,
        x.stride(0), out.stride(0), conv.stride(0), conv.stride(1), conv.stride(2),
        block_table.stride(0), weight.stride(0), weight.stride(1),
        dyn.stride(0) if dyn is not None else 0,
        kvt.stride(0), kvt.stride(1), kvt.stride(2), kv_off,
        K=k, BLOCK_C=block_c, DYNAMIC=dyn is not None, NORM=norm, STORE_KV=store_kv,
        PAGE=conv.shape[2], KV_PAGE=kvt.shape[2],
    )
    return out
