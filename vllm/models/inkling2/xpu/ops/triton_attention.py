# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 relative-bias paged attention (Triton, XPU).

One kernel for prefill, chunked prefill, decode and mixed batches:

* a program owns `BLOCK_M` rows = up to `BLOCK_M // G` consecutive query
  tokens of one request times the `G` query heads of one KV head (rows
  `r -> (token r // G, head r % G)`), so GQA's 3 heads per KV head pack
  without padding to a power of two;
* the bias of key `j` for a query at position `p` is `rel[t, h, p - j]` for
  `0 <= p - j < rel_extent`, else zero; keys at distance `>= window` are
  masked on sliding layers;
* the per-token sink is one extra logit with a zero value: the online
  softmax starts from `m = sink, l = 1`;
* with `num_splits > 1` every program takes keys `s, s + S, ...` (in units
  of `BLOCK_N`) and writes fp32 partials + log2-sum-exp; `_combine` merges
  them (decode of long contexts). The split count only depends on the
  launch, so a graph replays it for any context length.

Programs map to requests with the usual varlen trick: program `i` of the
first grid axis is query tile `i - b` of request `b`, where `b` is the
largest request with `cu_q[b] // TOKENS + b <= i`.
"""

from __future__ import annotations

import os

import torch

import triton
import triton.language as tl

LOG2E = tl.constexpr(1.4426950408889634)
# The Triton interpreter's bf16 tl.dot is wrong in this build: dot in fp32 there.
INTERP = os.environ.get("TRITON_INTERPRET", "0") == "1"


@triton.jit
def _find_req(cu_q, prog, num_reqs, TOKENS: tl.constexpr):
    lo = 0
    hi = num_reqs
    while lo < hi:
        mid = (lo + hi + 1) // 2
        start = tl.load(cu_q + mid) // TOKENS + mid
        if start <= prog:
            lo = mid
        else:
            hi = mid - 1
    return lo


@triton.jit
def _rel_attn_kernel(
    Q, KV, Rel, Sink, BlockTable, SeqLens, CuQ, Out, PartO, PartL,
    scale, num_reqs, window, rel_extent, num_splits,
    s_q_t, s_q_h, s_kv_b, s_kv_h, s_kv_n, s_rel_t, s_rel_h, s_sink_t, s_bt, s_o_t, s_o_h,
    s_po_s, s_po_t, s_po_h, s_pl_s, s_pl_t, s_pl_h,
    G: tl.constexpr, D: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    PAGE: tl.constexpr, SPLIT: tl.constexpr, DOT32: tl.constexpr, REV: tl.constexpr,
    BLOCKPTR: tl.constexpr,
):
    TOKENS: tl.constexpr = BLOCK_M // G
    prog = tl.program_id(0)
    kvh = tl.program_id(1)
    split = tl.program_id(2)
    req = _find_req(CuQ, prog, num_reqs, TOKENS)
    q0 = tl.load(CuQ + req)
    qlen = tl.load(CuQ + req + 1) - q0
    tile = prog - (q0 // TOKENS + req)
    if tile * TOKENS >= qlen:
        return
    seqlen = tl.load(SeqLens + req)
    ctx = seqlen - qlen  # tokens before this step

    rows = tl.arange(0, BLOCK_M)
    tok = tile * TOKENS + rows // G
    head = kvh * G + rows % G
    row_ok = (rows < TOKENS * G) & (tok < qlen)
    t = q0 + tok
    pos = ctx + tok
    d = tl.arange(0, D)
    q = tl.load(Q + t[:, None] * s_q_t + head[:, None] * s_q_h + d[None, :],
                mask=row_ok[:, None], other=0.0)
    sink = tl.load(Sink + t * s_sink_t + head, mask=row_ok, other=0.0).to(tl.float32)

    # visible keys of the tile: [lo, hi)
    last_pos = ctx + tl.minimum(tile * TOKENS + TOKENS, qlen) - 1
    first_pos = ctx + tile * TOKENS
    hi = last_pos + 1
    lo = 0
    if window > 0:
        lo = tl.maximum(first_pos - window + 1, 0)
    lo = (lo // BLOCK_N) * BLOCK_N

    if split == 0:
        m_i = tl.where(row_ok, sink * LOG2E, 0.0)
        l_i = tl.full([BLOCK_M], 1.0, tl.float32)
    else:  # the sink is counted once, in split 0
        m_i = tl.full([BLOCK_M], -1.0e30, tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
    acc = tl.zeros([BLOCK_M, D], tl.float32)
    qk_scale = scale * LOG2E
    n_off = tl.arange(0, BLOCK_N)
    start = lo + split * BLOCK_N
    for kb in range(start, hi, BLOCK_N * SPLIT):
        key = kb + n_off
        key_ok = key < hi
        if BLOCKPTR:  # one page per step: 2D block loads (BLOCK_N == PAGE, aligned)
            blk = tl.load(BlockTable + req * s_bt + kb // PAGE).to(tl.int64)
            kp = tl.make_block_ptr(KV + blk * s_kv_b + kvh * s_kv_h, shape=(PAGE, 2 * D),
                                   strides=(s_kv_n, 1), offsets=(0, 0), block_shape=(BLOCK_N, D),
                                   order=(1, 0))
            k = tl.load(kp)
            v = tl.load(tl.advance(kp, (0, D)))
        else:
            blk = tl.load(BlockTable + req * s_bt + key // PAGE, mask=key_ok, other=0).to(tl.int64)
            base = blk * s_kv_b + kvh * s_kv_h + (key % PAGE) * s_kv_n
            k = tl.load(KV + base[:, None] + d[None, :], mask=key_ok[:, None], other=0.0)
            v = tl.load(KV + base[:, None] + D + d[None, :], mask=key_ok[:, None], other=0.0)
        if DOT32:
            s = tl.dot(q.to(tl.float32), tl.trans(k.to(tl.float32)))
        else:
            s = tl.dot(q, tl.trans(k.to(q.dtype)))
        dist = pos[:, None] - key[None, :]
        in_rel = (dist >= 0) & (dist < rel_extent) & row_ok[:, None]
        if REV:  # rel stored reversed: column E-1-d, forward-contiguous along keys
            roff = rel_extent - 1 - dist
        else:
            roff = dist
        bias = tl.load(Rel + t[:, None] * s_rel_t + head[:, None] * s_rel_h + roff,
                       mask=in_rel, other=0.0).to(tl.float32)
        s = s * qk_scale + bias * LOG2E
        ok = (dist >= 0) & key_ok[None, :]
        if window > 0:
            ok = ok & (dist < window)
        s = tl.where(ok, s, float("-inf"))
        m_new = tl.maximum(m_i, tl.max(s, 1))
        alpha = tl.exp2(m_i - m_new)
        p = tl.exp2(s - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, 1)
        if DOT32:
            pv = tl.dot(p.to(tl.bfloat16).to(tl.float32), v.to(tl.float32))
        else:
            pv = tl.dot(p.to(q.dtype), v.to(q.dtype))
        acc = acc * alpha[:, None] + pv
        m_i = m_new

    if SPLIT == 1:
        o = acc / l_i[:, None]
        tl.store(Out + t[:, None] * s_o_t + head[:, None] * s_o_h + d[None, :],
                 o.to(Out.dtype.element_ty), mask=row_ok[:, None])
    else:
        tl.store(PartO + split * s_po_s + t[:, None] * s_po_t + head[:, None] * s_po_h + d[None, :],
                 acc, mask=row_ok[:, None])
        lb = PartL + split * s_pl_s + t * s_pl_t + head * s_pl_h
        tl.store(lb, m_i, mask=row_ok)
        tl.store(lb + 1, l_i, mask=row_ok)


@triton.jit
def _combine_kernel(PartO, PartL, Out, s_po_s, s_po_t, s_po_h, s_pl_s, s_pl_t, s_pl_h,
                    s_o_t, s_o_h, D: tl.constexpr, SPLITS: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, D)
    sp = tl.arange(0, SPLITS)
    lb = PartL + sp * s_pl_s + t * s_pl_t + h * s_pl_h
    m = tl.load(lb)
    ls = tl.load(lb + 1)
    m_all = tl.max(m, 0)
    w = tl.exp2(m - m_all)
    o = tl.load(PartO + sp[:, None] * s_po_s + t * s_po_t + h * s_po_h + d[None, :])
    num = tl.sum(o * w[:, None], 0)
    den = tl.sum(ls * w, 0)
    tl.store(Out + t * s_o_t + h * s_o_h + d, (num / den).to(Out.dtype.element_ty))


def _launch_opts() -> dict:
    o = {}
    if os.environ.get("INKLING2_ATTN_WARPS"):
        o["num_warps"] = int(os.environ["INKLING2_ATTN_WARPS"])
    if os.environ.get("INKLING2_ATTN_GRF"):
        o["grf_mode"] = os.environ["INKLING2_ATTN_GRF"]
    return o


def rel_attention(q: torch.Tensor, kv: torch.Tensor, rel: torch.Tensor, sinks: torch.Tensor,
                  block_table: torch.Tensor, seq_lens: torch.Tensor, cu_q: torch.Tensor,
                  scale: float, window: int | None, rel_extent: int, num_reqs: int,
                  out: torch.Tensor | None = None, num_splits: int = 1,
                  block_m: int | None = None, block_n: int = 64,
                  rel_reversed: bool = False) -> torch.Tensor:
    """q (T, Hq, D) bf16; kv [B, Hk, page, 2D]; rel (T, Hq, E) (column d = bias at
    distance d, or with `rel_reversed` column E-1-d: the kernel then reads it
    forward-contiguous along keys, which the B70 needs); sinks (T, Hq) (row
    stride free); returns (T, Hq, D) bf16. Rows of padded tokens beyond
    `cu_q[num_reqs]` are left untouched."""
    T, Hq, D = q.shape
    Hk, page = kv.shape[1], kv.shape[2]
    assert kv.stride(3) == 1 and q.stride(2) == 1 and rel.stride(2) == 1 and sinks.stride(1) == 1
    G = Hq // Hk
    if block_m is None:
        block_m = 16 if T <= num_reqs * 4 else 64
    tokens = block_m // G
    if out is None:
        out = torch.empty_like(q)
    grid0 = T // tokens + num_reqs
    if num_splits > 1:
        assert num_splits & (num_splits - 1) == 0
        part_o = torch.empty((num_splits, T, Hq, D), dtype=torch.float32, device=q.device)
        part_l = torch.zeros((num_splits, T, Hq, 2), dtype=torch.float32, device=q.device)
        po, pl = part_o, part_l
        pstr = (*part_o.stride()[:3], *part_l.stride()[:3])
    else:
        po, pl, pstr = out, out, (0, 0, 0, 0, 0, 0)
    _rel_attn_kernel[(grid0, Hk, num_splits)](
        q, kv, rel, sinks, block_table, seq_lens, cu_q, out, po, pl,
        scale, num_reqs, window or 0, rel_extent, num_splits,
        q.stride(0), q.stride(1), kv.stride(0), kv.stride(1), kv.stride(2),
        rel.stride(0), rel.stride(1), sinks.stride(0), block_table.stride(0),
        out.stride(0), out.stride(1), *pstr,
        G=G, D=D, BLOCK_M=block_m, BLOCK_N=block_n, PAGE=page, SPLIT=num_splits, DOT32=INTERP, REV=rel_reversed,
        BLOCKPTR=(os.environ.get("INKLING2_ATTN_BLOCKPTR") == "1" and block_n == page
                  and not INTERP), **_launch_opts(),
    )
    if num_splits > 1:
        # every token row; padded rows (beyond the real tokens) get garbage
        _combine_kernel[(T, Hq)](part_o, part_l, out, *pstr, out.stride(0), out.stride(1),
                                 D=D, SPLITS=num_splits)
    return out
