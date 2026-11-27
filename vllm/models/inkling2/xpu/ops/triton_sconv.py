# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 short conv over the per-sequence conv state (Triton, XPU).

`out[t] = (sum_lag w[lag] * x[t - lag] + x[t]) / sqrt(k + 1)` (static taps), or
with dynamic per-token per-head taps added to the static ones and
`/ sqrt(2k + 1)` (full-attention K/V). Past inputs of the same step come from
`x`; the `k - 1` before the request's first row of this step from its state
slot (`[k - 1, H * W]`, oldest first; zero when the slot holds no history).

One program per (token, column block). The program of a request's first row
also computes the rows that read the state (the first `k - 1`) and then
writes the request's new state, so the state is read and written by one
program only and the kernel needs no second pass. Programs of the other
state-reading rows do nothing.

Optional fused epilogues: `NORM` 1 = L2 clip per head, 2 = unweighted RMS
norm per head (one head per program then), a store of the result into the
visit's attention KV cache at the token's KV slot (`KV_OFF` = 0 for K, `D` for
V in the K|V content dim), and the residual sum `last (+ addend) + y`.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _conv_row(t, r, st_base, has_init, X, Out, W, Dyn, KV, KVSlots, Last, Addend,
              c, cm, ROW, eps, clip, inv_norm, dyn_heads_w,
              s_x, s_o, s_w0, s_w1, s_dyn, s_kv_b, s_kv_h, s_kv_n, KV_OFF, hph,
              K: tl.constexpr, BLOCK_C: tl.constexpr, DYNAMIC: tl.constexpr,
              NORM: tl.constexpr, STORE_KV: tl.constexpr, KV_PAGE: tl.constexpr,
              RESID: tl.constexpr, HAS_ADD: tl.constexpr):
    x = tl.load(X + t * s_x + c, mask=cm, other=0.0).to(tl.float32)
    acc = tl.zeros([BLOCK_C], tl.float32)
    for lag in tl.static_range(K):
        if lag == 0:
            v = x
        else:
            rr = r - lag
            if rr >= 0:
                v = tl.load(X + (t - lag) * s_x + c, mask=cm, other=0.0).to(tl.float32)
            else:
                v = tl.load(st_base + (K - 1 + rr) * ROW + c, mask=cm & has_init,
                            other=0.0).to(tl.float32)
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
    if RESID:  # the layer output: (last (+ addend) + conv out) in fp32, rounded once
        res = tl.load(Last + t * s_o + c, mask=cm, other=0.0).to(tl.float32)
        if HAS_ADD:
            res += tl.load(Addend + t * s_o + c, mask=cm, other=0.0).to(tl.float32)
        tl.store(Out + t * s_o + c, (res + y).to(Out.dtype.element_ty), mask=cm)
    else:
        tl.store(Out + t * s_o + c, y.to(Out.dtype.element_ty), mask=cm)
    if STORE_KV:
        ks = tl.load(KVSlots + t)
        if ks >= 0:
            head = c // hph
            tl.store(KV + (ks // KV_PAGE).to(tl.int64) * s_kv_b + head * s_kv_h
                     + (ks % KV_PAGE) * s_kv_n + KV_OFF + c % hph, y.to(KV.dtype.element_ty),
                     mask=cm)


@triton.jit
def _sconv_kernel(
    X, Out, State, SeqIdx, QStart, QSL, BlockTable, SeqLens, NumAcc, W, Dyn, KV, KVSlots, Last, Addend,
    C, hph, W_row, off, eps, clip, inv_norm, dyn_heads_w,
    s_x, s_o, s_st, ROW, s_bt, state_bs, s_w0, s_w1, s_dyn, s_kv_b, s_kv_h, s_kv_n, KV_OFF,
    K: tl.constexpr, KP: tl.constexpr, BLOCK_C: tl.constexpr, DYNAMIC: tl.constexpr,
    NORM: tl.constexpr, STORE_KV: tl.constexpr, KV_PAGE: tl.constexpr,
    RESID: tl.constexpr, HAS_ADD: tl.constexpr, ALIGN: tl.constexpr, S: tl.constexpr,
):
    t = tl.program_id(0)
    cb = tl.program_id(1)
    c = cb * BLOCK_C + tl.arange(0, BLOCK_C)
    cm = c < C
    # state column of channel c: head-major rows of width W_row, stream at `off`
    sc = (c // hph) * W_row + off + c % hph
    req = tl.load(SeqIdx + t)
    q0 = tl.load(QStart + t)
    r = t - q0
    if r == 0:
        L = tl.load(QSL + req + 1) - q0
        seq_len = tl.load(SeqLens + req)
        has_init = seq_len > L  # tokens computed before this step
        if ALIGN:
            col = tl.maximum(seq_len - 1, 0) // state_bs
        else:
            col = 0
        si = tl.load(BlockTable + req * s_bt + col).to(tl.int64)
        st = State + si * s_st
        # speculative decode (S > 0): the state holds K - 1 + S rows; a decode row reads
        # its K - 1 prior inputs from row num_accepted - 1 on (vLLM's rolling conv state)
        if S > 0:
            is_dec = has_init & (L <= 1 + S)
            w = tl.load(NumAcc + req, mask=is_dec, other=1) - 1
        else:
            is_dec = False
            w = 0
        st_w = st + w * ROW
        st_c = st_w + sc - c  # so that st_c + row * ROW + c addresses channel c
        for u in tl.static_range(K - 1):
            if u < L:
                _conv_row(t + u, u, st_c, has_init, X, Out, W, Dyn, KV, KVSlots, Last, Addend,
                          c, cm, ROW, eps, clip, inv_norm, dyn_heads_w, s_x, s_o, s_w0, s_w1,
                          s_dyn, s_kv_b, s_kv_h, s_kv_n, KV_OFF, hph, K=K, BLOCK_C=BLOCK_C,
                          DYNAMIC=DYNAMIC, NORM=NORM, STORE_KV=STORE_KV, KV_PAGE=KV_PAGE,
                          RESID=RESID, HAS_ADD=HAS_ADD)
        # new state, all loads before the stores. Prefill rows: rows 0..K-2 = the last K - 1
        # inputs (older ones from the old state). Decode rows (S > 0): the K - 2 inputs before
        # the step (old rows w + 1 + j), then the step's L inputs at rows K - 2 .. K - 3 + L, so
        # the next step's window from row num_accepted - 1 ends at its last accepted token for
        # any L <= 1 + S (a verify of fewer drafts: dynamic draft lengths, or drafts cut at
        # max_model_len); rows past the step's inputs are not read and stay as they were.
        R = K - 1 + S
        rows = tl.arange(0, KP)
        if is_dec:
            from_old = rows < K - 2
            old_row = tl.minimum(w + 1 + rows, R - 1)
            src = rows - (K - 2)
            rm = rows < K - 2 + L
        else:
            src = L - (K - 1) + rows
            from_old = src < 0
            old_row = tl.maximum(K - 1 + src, 0)
            rm = rows < K - 1
        from_x = tl.load(X + (q0 + tl.maximum(src, 0))[:, None] * s_x + c[None, :],
                         mask=(rm & ~from_old)[:, None] & cm[None, :], other=0.0)
        from_st = tl.load(st + old_row[:, None] * ROW + sc[None, :],
                          mask=(rm & from_old)[:, None] & cm[None, :] & has_init, other=0.0)
        new = tl.where(from_old[:, None], from_st.to(tl.float32), from_x.to(tl.float32))
        tl.store(st + rows[:, None] * ROW + sc[None, :], new.to(State.dtype.element_ty),
                 mask=rm[:, None] & cm[None, :])
    elif r >= K - 1:
        _conv_row(t, r, State, False, X, Out, W, Dyn, KV, KVSlots, Last, Addend,
                  c, cm, ROW, eps, clip, inv_norm, dyn_heads_w, s_x, s_o, s_w0, s_w1,
                  s_dyn, s_kv_b, s_kv_h, s_kv_n, KV_OFF, hph, K=K, BLOCK_C=BLOCK_C,
                  DYNAMIC=DYNAMIC, NORM=NORM, STORE_KV=STORE_KV, KV_PAGE=KV_PAGE,
                  RESID=RESID, HAS_ADD=HAS_ADD)


def short_conv(x: torch.Tensor, state: torch.Tensor, stream_off: int, hph: int, row_width: int,
               seq_idx: torch.Tensor, query_start: torch.Tensor, query_start_loc: torch.Tensor,
               block_table: torch.Tensor, seq_lens: torch.Tensor, weight: torch.Tensor,
               dyn: torch.Tensor | None = None, norm: int = 0, clip: float = 0.0,
               eps: float = 1e-6, kv: torch.Tensor | None = None,
               kv_slots: torch.Tensor | None = None, kv_off: int = 0,
               last: torch.Tensor | None = None,
               addend: torch.Tensor | None = None, align: bool = False,
               state_block_size: int = 1, num_accepted: torch.Tensor | None = None,
               spec: int = 0) -> torch.Tensor:
    """x (T, C) bf16 (C = heads * hph); state [slots, k - 1, heads * row_width]
    (the stream at column `stream_off` of each head's row); weight (C, k) static
    or (k, C) with `dyn` (T, H * k); returns (T, C) bf16, or with `last` (and
    `addend`), contiguous (T, C), the residual sum last (+ addend) + conv.
    A request's slot is block_table[req, 0], or in align mode the block of its
    last token; it holds history when seq_lens[req] exceeds its rows in the
    step. Updates the state slots of the requests in the step."""
    T, C = x.shape
    k = weight.shape[1] if dyn is None else weight.shape[0]
    out = torch.empty((T, C), dtype=torch.bfloat16, device=x.device)
    if T == 0:
        return out
    assert state.shape[1] == k - 1 + spec and state.stride(2) == 1
    assert spec == 0 or num_accepted is not None
    if norm:
        assert C % hph == 0
        block_c = hph
        assert block_c & (block_c - 1) == 0, "per-head norm needs a power-of-two head"
    else:
        block_c = min(1024, triton.next_power_of_2(C))
    inv = 1.0 / math.sqrt((2 * k + 1) if dyn is not None else (k + 1))
    store_kv = kv is not None
    kvt = kv if store_kv else x.view(1, 1, T, C)
    _sconv_kernel[(T, triton.cdiv(C, block_c))](
        x, out, state, seq_idx, query_start, query_start_loc, block_table, seq_lens,
        num_accepted if num_accepted is not None else seq_lens, weight,
        dyn if dyn is not None else x, kvt, kv_slots if store_kv else seq_idx,
        last if last is not None else x, addend if addend is not None else x,
        C, hph, row_width, stream_off, eps, clip, inv, hph if dyn is not None else 1,
        x.stride(0), out.stride(0), state.stride(0), state.stride(1), block_table.stride(0),
        state_block_size,
        weight.stride(0), weight.stride(1), dyn.stride(0) if dyn is not None else 0,
        kvt.stride(0), kvt.stride(1), kvt.stride(2), kv_off,
        K=k, KP=triton.next_power_of_2(k - 1 + spec), BLOCK_C=block_c, DYNAMIC=dyn is not None,
        NORM=norm, STORE_KV=store_kv, KV_PAGE=kvt.shape[2] if store_kv else 1,
        RESID=last is not None, HAS_ADD=addend is not None, ALIGN=align, S=spec,
    )
    return out
