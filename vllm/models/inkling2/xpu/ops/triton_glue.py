# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused elementwise glue of an Inkling2 visit (Triton, XPU).

Each kernel replaces a chain of small torch ops with one launch and keeps the
reference's rounding points (bf16 where the reference rounds):

  qkr_prep    q and r: learned scale, L2 clip (bf16), log scaling (bf16), and
              the relative logits r @ rel_proj (bf16)
  attn_post   per-token sinks from the attention LSE and the sigmoid output gate
  route       sigmoid router with bias, top-k, normalised x route_scale
  blend_prep  [rms(dest), rms(message)] (bf16) for the loop controllers
  blend_out   (1 - tanh(forget)) * dest + tanh(gain) * message, then the next
              sublayer's RMSNorm (bf16)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _clip_round(x, clip):
    n = tl.sqrt(tl.sum(x * x, 0))
    return (x * tl.minimum(clip / (n + 1e-6), 1.0)).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _qkr_kernel(Y, QS, RS, Tau, Proj, Q, Rel, Rout, s_y, r_off, q_clip, r_clip,
                H: tl.constexpr, D: tl.constexpr, R: tl.constexpr, E: tl.constexpr,
                BE: tl.constexpr, TAU: tl.constexpr, REL: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, D)
    q = tl.load(Y + t * s_y + h * D + d).to(tl.float32) * tl.load(QS + h * D + d).to(tl.float32)
    q = _clip_round(q, q_clip)
    j = tl.arange(0, R)
    r = tl.load(Y + t * s_y + r_off + h * R + j).to(tl.float32) * tl.load(RS + h * R + j).to(
        tl.float32)
    r = _clip_round(r, r_clip)
    if TAU:
        tau = tl.load(Tau + t)
        q = (q * tau).to(tl.bfloat16).to(tl.float32)
        r = (r * tau).to(tl.bfloat16).to(tl.float32)
    tl.store(Q + (t * H + h) * D + d, q.to(tl.bfloat16))
    if not REL:  # large batches: r out, the logits as one bf16 GEMM
        tl.store(Rout + (t * H + h) * R + j, r.to(tl.bfloat16))
        return
    for e0 in tl.static_range(0, E, BE):
        e = e0 + tl.arange(0, BE)
        p = tl.load(Proj + j[:, None] * E + e[None, :]).to(tl.float32)
        rel = tl.sum(r[:, None] * p, 0)
        tl.store(Rel + (t * H + h) * E + e, rel.to(tl.bfloat16))


def qkr_prep(y, q_scale, r_scale, tau, proj, H, D, R, r_off, q_clip, r_clip):
    """y (T, cols) the fused projection output; q at columns [0, H*D), r at
    [r_off, r_off + H*R). Returns q (T, H, D) bf16 and rel logits (T, H, E) bf16."""
    T = y.shape[0]
    E = proj.shape[1]
    q = torch.empty((T, H, D), dtype=torch.bfloat16, device=y.device)
    # in-kernel logits win at decode sizes; above that the GEMM does (2.2 ms vs a
    # fraction at T = 2048)
    in_kernel = T <= 64
    rel = torch.empty((T, H, E) if in_kernel else (T, H, R), dtype=torch.bfloat16,
                      device=y.device)
    if T:
        _qkr_kernel[(T, H)](y, q_scale, r_scale, tau if tau is not None else y, proj, q, rel,
                            rel, y.stride(0), r_off, q_clip, r_clip, H=H, D=D, R=R, E=E,
                            BE=min(E, 256), TAU=tau is not None, REL=in_kernel, num_warps=4)
    return q, rel if in_kernel else torch.matmul(rel, proj)


@triton.jit
def _attn_post_kernel(O, LSE, Sink, Gate, Out, n, s_lse, s_sink, s_gate,
                      H: tl.constexpr, D: tl.constexpr):
    t = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, D)
    if t >= n:
        tl.store(Out + t * (H * D) + h * D + d, tl.zeros((D,), tl.bfloat16))
        return
    o = tl.load(O + (t * H + h) * D + d).to(tl.float32)
    z = tl.load(LSE + h * s_lse + t) - tl.load(Sink + t * s_sink + h).to(tl.float32)
    o = (o / (1.0 + tl.exp(-z))).to(tl.bfloat16).to(tl.float32)
    g = tl.load(Gate + t * s_gate + h * D + d).to(tl.float32)
    tl.store(Out + t * (H * D) + h * D + d, (o / (1.0 + tl.exp(-g))).to(tl.bfloat16))


def attn_post(o, lse, sinks, gate, out, n):
    """o (T, H, D) raw attention output, lse (H, >=n) fp32, sinks (T, H) and gate
    (T, H*D) views of the projection -> out (T, H*D) bf16 (rows >= n zeroed)."""
    T, H, D = o.shape
    if T:
        _attn_post_kernel[(T, H)](o, lse, sinks, gate, out, n, lse.stride(0), sinks.stride(0),
                                  gate.stride(0), H=H, D=D, num_warps=2)
    return out


@triton.jit
def _attn_post_q_kernel(O, LSE, Sink, Gate, Out, XQ, XS, n, s_lse, s_sink, s_gate,
                        H: tl.constexpr, HP: tl.constexpr, D: tl.constexpr):
    # one program per token: the gated row (H * D) and its per-token int8 quant
    t = tl.program_id(0)
    h = tl.arange(0, HP)
    d = tl.arange(0, D)
    hm = h < H
    m = hm[:, None]
    if t >= n:
        tl.store(Out + t * (H * D) + h[:, None] * D + d[None, :], tl.zeros((HP, D), tl.bfloat16),
                 mask=m)
        tl.store(XQ + t * (H * D) + h[:, None] * D + d[None, :], tl.zeros((HP, D), tl.int8), mask=m)
        tl.store(XS + t, 1e-10)
        return
    o = tl.load(O + (t * H + h[:, None]) * D + d[None, :], mask=m, other=0.0).to(tl.float32)
    z = tl.load(LSE + h * s_lse + t, mask=hm, other=0.0) - tl.load(
        Sink + t * s_sink + h, mask=hm, other=0.0).to(tl.float32)
    o = (o / (1.0 + tl.exp(-z))[:, None]).to(tl.bfloat16).to(tl.float32)
    g = tl.load(Gate + t * s_gate + h[:, None] * D + d[None, :], mask=m, other=0.0).to(tl.float32)
    y = (o / (1.0 + tl.exp(-g))).to(tl.bfloat16)
    tl.store(Out + t * (H * D) + h[:, None] * D + d[None, :], y, mask=m)
    yf = tl.where(m, y.to(tl.float32), 0.0)
    sc = tl.maximum(tl.max(tl.max(tl.abs(yf), 1), 0) / 127.0, 1e-10)
    x = yf / sc
    f = tl.floor(x)
    r = x - f
    up = (r > 0.5) | ((r == 0.5) & ((f - 2.0 * tl.floor(f * 0.5)) != 0.0))  # half to even
    tl.store(XQ + t * (H * D) + h[:, None] * D + d[None, :], tl.where(up, f + 1.0, f).to(tl.int8),
             mask=m)
    tl.store(XS + t, sc)


def attn_post_q(o, lse, sinks, gate, out, n, xq, xs):
    """attn_post plus the per-token int8 rows (xq (T, H*D) int8, xs (T,) fp32) of the
    output, for an int8 GEMM; decode sizes (one program per token)."""
    T, H, D = o.shape
    if T:
        _attn_post_q_kernel[(T,)](o, lse, sinks, gate, out, xq, xs, n, lse.stride(0),
                                  sinks.stride(0), gate.stride(0), H=H,
                                  HP=triton.next_power_of_2(H), D=D, num_warps=8)
    return out


@triton.jit
def _route_kernel(X, W, Logits, Bias, Ids, Wts, H, scale, E: tl.constexpr, TOPK: tl.constexpr,
                  BK: tl.constexpr, LOGITS_IN: tl.constexpr):
    t = tl.program_id(0)
    lanes = tl.arange(0, E)
    if LOGITS_IN:
        acc = tl.load(Logits + t * E + lanes)
    else:
        acc = tl.zeros((E,), tl.float32)
        for k0 in range(0, H, BK):
            k = k0 + tl.arange(0, BK)
            x = tl.load(X + t * H + k).to(tl.float32)
            w = tl.load(W + lanes[:, None] * H + k[None, :])
            acc += tl.sum(w * x[None, :], 1)
    sc = 1.0 / (1.0 + tl.exp(-acc))
    sel = sc + tl.load(Bias + lanes)
    chosen = tl.zeros((E,), tl.int32)
    for _ in tl.static_range(TOPK):
        idx = tl.argmax(sel, 0)
        chosen = tl.where(lanes == idx, 1, chosen)
        sel = tl.where(lanes == idx, float("-inf"), sel)
    norm = scale / (tl.sum(tl.where(chosen == 1, sc, 0.0), 0) + 1e-20)
    rem = chosen
    for k in tl.static_range(TOPK):
        idx = tl.argmax(rem, 0)
        rem = tl.where(lanes == idx, 0, rem)
        tl.store(Ids + t * TOPK + k, idx)
        tl.store(Wts + t * TOPK + k, tl.sum(tl.where(lanes == idx, sc, 0.0), 0) * norm)


def route(x, weight, bias, top_k, route_scale):
    """Sigmoid router (fp32): (weights fp32 (T, k), ids int32 (T, k)); slot order is
    by expert index."""
    T, H = x.shape
    E = weight.shape[0]
    ids = torch.empty((T, top_k), dtype=torch.int32, device=x.device)
    wts = torch.empty((T, top_k), dtype=torch.float32, device=x.device)
    # the logits GEMM is far faster than a per-token GEMV inside the kernel
    # (measured 126 us/call for that at T=1)
    logits = torch.nn.functional.linear(x.float(), weight)
    if T:
        _route_kernel[(T,)](x, weight, logits if logits is not None else weight, bias, ids, wts,
                            H, route_scale, E=E, TOPK=top_k, BK=32,
                            LOGITS_IN=logits is not None, num_warps=4)
    return wts, ids


@triton.jit
def _blend_prep_kernel(Dst, Msg, Cat, T, s_ds, s_dt, s_ms, s_mt, D: tl.constexpr,
                       BLOCK: tl.constexpr, eps):
    i = tl.program_id(0)
    s = i // T
    t = i % T
    ssd = 0.0
    ssm = 0.0
    for c in tl.static_range(0, D, BLOCK):
        d = c + tl.arange(0, BLOCK)
        dm = d < D
        a = tl.load(Dst + s * s_ds + t * s_dt + d, mask=dm, other=0.0)
        b = tl.load(Msg + s * s_ms + t * s_mt + d, mask=dm, other=0.0)
        ssd += tl.sum(a * a, 0)
        ssm += tl.sum(b * b, 0)
    rd = tl.rsqrt(ssd / D + eps)
    rm = tl.rsqrt(ssm / D + eps)
    for c in tl.static_range(0, D, BLOCK):
        d = c + tl.arange(0, BLOCK)
        dm = d < D
        a = tl.load(Dst + s * s_ds + t * s_dt + d, mask=dm, other=0.0)
        b = tl.load(Msg + s * s_ms + t * s_mt + d, mask=dm, other=0.0)
        tl.store(Cat + i * (2 * D) + d, (a * rd).to(tl.bfloat16), mask=dm)
        tl.store(Cat + i * (2 * D) + D + d, (b * rm).to(tl.bfloat16), mask=dm)


def blend_prep(dests, messages, eps=1e-6):
    """fp32 (S, T, D) dests / messages (any stride along S and T, unit along D)
    -> bf16 (S, T, 2D) [rms(dest), rms(message)]."""
    S, T, D = dests.shape
    cat = torch.empty((S, T, 2 * D), dtype=torch.bfloat16, device=dests.device)
    if S * T:
        _blend_prep_kernel[(S * T,)](dests, messages, cat, T, dests.stride(0), dests.stride(1),
                                     messages.stride(0), messages.stride(1), D=D, BLOCK=512,
                                     eps=eps, num_warps=4)
    return cat


@triton.jit
def _blend_out_kernel(GF, Dst, Msg, W, Scr, Out, T, s_ds, s_dt, s_ms, s_mt, s_w, eps,
                      D: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0)
    s = i // T
    t = i % T
    ss = 0.0
    for c in tl.static_range(0, D, BLOCK):
        d = c + tl.arange(0, BLOCK)
        dm = d < D
        g = tl.load(GF + i * (2 * D) + d, mask=dm, other=0.0)
        f = tl.load(GF + i * (2 * D) + D + d, mask=dm, other=0.0)
        a = tl.load(Dst + s * s_ds + t * s_dt + d, mask=dm, other=0.0)
        b = tl.load(Msg + s * s_ms + t * s_mt + d, mask=dm, other=0.0)
        tg = 2.0 / (1.0 + tl.exp(-2.0 * g)) - 1.0
        tf = 2.0 / (1.0 + tl.exp(-2.0 * f)) - 1.0
        x = (1.0 - tf) * a + tg * b
        ss += tl.sum(x * x, 0)
        tl.store(Scr + i * D + d, x, mask=dm)  # fp32 scratch row, normalised below
    r = tl.rsqrt(ss / D + eps)
    for c in tl.static_range(0, D, BLOCK):
        d = c + tl.arange(0, BLOCK)
        dm = d < D
        x = tl.load(Scr + i * D + d, mask=dm, other=0.0)
        w = tl.load(W + s * s_w + d, mask=dm, other=0.0).to(tl.float32)
        tl.store(Out + i * D + d, (w * (x * r)).to(tl.bfloat16), mask=dm)


def blend_out(gf, dests, messages, weight, eps):
    """gf fp32 (S, T, 2D) [gain; forget] pre-activations; weight (S, D) RMSNorm rows
    (stride 0 to share one) -> bf16 (S, T, D) normalised blends."""
    S, T, D = dests.shape
    scr = torch.empty((S, T, D), dtype=torch.float32, device=dests.device)
    out = torch.empty((S, T, D), dtype=torch.bfloat16, device=dests.device)
    if S * T:
        _blend_out_kernel[(S * T,)](gf, dests, messages, weight, scr, out, T, dests.stride(0),
                                    dests.stride(1), messages.stride(0), messages.stride(1),
                                    weight.stride(0), eps, D=D, BLOCK=512, num_warps=4)
    return out
