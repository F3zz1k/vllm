# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Torch references of the Inkling2 ops (XPU port).

Every op here follows the arithmetic of transformers' `Inkling2Loop` model
(fp32 where it computes in fp32, one bf16 rounding where it rounds) and works
on vLLM's paged caches, so it runs on CPU for the model skeleton and serves as
the oracle of the Triton kernels in `ops/`.

Paged cache views used throughout:
  attention KV, one visit: `kv` [num_blocks, H, block_size, 2 * D] (K | V in
    the content dim, the logical layout of vLLM's KV cache groups);
  conv state, one visit: `conv` [num_blocks, H, 4, W], per head the streams
    K, V, attention output and MLP output at the offsets of `ConvLayout`.
A token's slot is `block * block_size + offset`; -1 skips the token.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------- norms


def rms_norm(x: torch.Tensor, weight: torch.Tensor | None, eps: float,
             out_dtype: torch.dtype | None = torch.bfloat16) -> torch.Tensor:
    """fp32 RMS norm (weighted when `weight` is given), rounded once."""
    h = x.float()
    h = h * torch.rsqrt(h.square().mean(-1, keepdim=True) + eps)
    if weight is not None:
        h = weight.float() * h
    return h if out_dtype is None else h.to(out_dtype)


def unweighted_rms(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    h = x.float()
    return h * torch.rsqrt(h.square().mean(-1, keepdim=True) + eps)


def l2_clip(x: torch.Tensor, threshold: float) -> torch.Tensor:
    """fp32 in, fp32 out: scale each head vector to `threshold` if longer."""
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    return x * (threshold / (norm + 1e-6)).clamp(max=1.0)


def log_scaling_tau(positions: torch.Tensor, n_floor: int, alpha: float) -> torch.Tensor:
    """`1 + alpha * log(max((pos + 1) / n_floor, 1))`, fp32 `(T,)`."""
    return 1.0 + alpha * torch.log(((positions + 1).float() / n_floor).clamp(min=1.0))


# --------------------------------------------------- block attention residuals


def attn_res(blocks: torch.Tensor, partial: torch.Tensor, queries: torch.Tensor) -> torch.Tensor:
    """Block AttnRes with normalized values.

    blocks `(N, T, D)` (N may be 0), partial `(T, D)`, queries `(Q, D)` fp32
    already scaled by `D**-0.5`. Returns fp32 `(Q, T, D)`: per query, the
    softmax mix of the unweighted-RMS-normalized `[blocks..., partial]`.
    """
    stacked = torch.cat([blocks.float(), partial.float()[None]], 0)  # (N+1, T, D)
    stacked = unweighted_rms(stacked)
    scores = torch.einsum("qd,ntd->tqn", queries.float(), stacked)
    probs = torch.softmax(scores, -1)
    return torch.einsum("tqn,ntd->qtd", probs, stacked)


# --------------------------------------------------------- loop controllers


def message_blend(dest: torch.Tensor, message: torch.Tensor, down: torch.Tensor,
                  gain: torch.Tensor, forget: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """`(1 - tanh(forget(h))) * dest + tanh(gain(h)) * message`, fp32.

    h = down(cat(norm(dest), norm(message))) with a bf16 GEMM (bf16 output, as
    transformers' bf16 `down_proj`); gain/forget run in fp32 on that output.
    """
    dest, message = dest.float(), message.float()
    cat = torch.cat([unweighted_rms(dest, eps), unweighted_rms(message, eps)], -1)
    h = F.linear(cat.to(down.dtype), down).float()
    return (1 - torch.tanh(F.linear(h, forget.float()))) * dest + torch.tanh(
        F.linear(h, gain.float())) * message


# --------------------------------------------------------------- short conv


@dataclass(frozen=True)
class ConvLayout:
    """Per-head offsets of the four conv streams in a conv-state row."""

    num_heads: int
    k_off: int
    k_width: int
    v_off: int
    v_width: int
    attn_off: int
    mlp_off: int
    hidden_per_head: int
    width: int  # row width W per head

    @staticmethod
    def make(num_kv_heads: int, head_dim: int, hidden: int) -> ConvLayout:
        hph = hidden // num_kv_heads
        # unpadded: the conv-state page must stay under the attention page so the KV
        # block stays at 64 tokens (with speculative rows at TP=2 a padded row did not)
        width = 2 * head_dim + 2 * hph
        return ConvLayout(num_kv_heads, 0, head_dim, head_dim, head_dim,
                          2 * head_dim, 2 * head_dim + hph, hph, width)

    def stream(self, name: str) -> tuple[int, int]:
        """(offset, width per head) of a stream."""
        return {
            "k": (self.k_off, self.k_width),
            "v": (self.v_off, self.v_width),
            "attn": (self.attn_off, self.hidden_per_head),
            "mlp": (self.mlp_off, self.hidden_per_head),
        }[name]


def _slot_split(slots: torch.Tensor, block_size: int):
    return slots // block_size, slots % block_size


def conv_history_state(state: torch.Tensor, layout: ConvLayout, stream: str, x: torch.Tensor,
                       query_start_loc: list[int], state_idx: list[int], has_init: list[bool],
                       kernel: int) -> torch.Tensor:
    """fp32 `(T, kernel, H * w)` oldest first: the step's own inputs `x`, and
    before each request's first row its state slot (`[k - 1, H * W]`, zero
    without history). Then writes each request's new state (its last k - 1
    inputs)."""
    off, w = layout.stream(stream)
    H, W = layout.num_heads, layout.width
    T, C = x.shape
    hist = x.new_zeros((T, kernel, C), dtype=torch.float32)
    for r in range(len(query_start_loc) - 1):
        q0, q1 = query_start_loc[r], query_start_loc[r + 1]
        st = state[state_idx[r]].view(kernel - 1, H, W)[..., off:off + w].reshape(kernel - 1, C)
        old = st.float() if has_init[r] else torch.zeros_like(st, dtype=torch.float32)
        seq = torch.cat([old, x[q0:q1].float()])  # (k - 1 + n, C)
        for i in range(q1 - q0):
            hist[q0 + i] = seq[i:i + kernel]
        new = seq[-(kernel - 1):].to(state.dtype)
        state[state_idx[r]].view(kernel - 1, H, W)[..., off:off + w] = new.view(kernel - 1, H, w)
    return hist


def static_conv(hist: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Short conv with residual: `(conv(x) + x) / sqrt(k + 1)`, fp32.

    hist `(T, k, C)` oldest first; weight `(C, k)` as nn.Conv1d (tap k-1 is the
    current token)."""
    k = hist.shape[1]
    out = (hist * weight.float().t()[None]).sum(1)
    return (out + hist[:, -1]) / math.sqrt(k + 1)


def dynamic_conv(hist: torch.Tensor, static_w: torch.Tensor, dyn: torch.Tensor,
                 num_heads: int) -> torch.Tensor:
    """Dynamic short conv of the full-attention K/V, fp32.

    hist `(T, k, C)` oldest first, static_w `(k, C)` (tap 0 = current token),
    dyn `(T, H * k)` per-token per-head taps (tap 0 = current token)."""
    T, k, C = hist.shape
    lagged = hist.flip(1)  # lag 0 = current
    w = static_w.float()[None] .view(1, k, num_heads, C // num_heads)
    w = w + dyn.float().view(T, num_heads, k).transpose(1, 2)[..., None]
    out = (lagged.view(T, k, num_heads, -1) * w).sum(1).reshape(T, C)
    return (out + hist[:, -1]) / math.sqrt(2 * k + 1)


# --------------------------------------------------------------- attention


def kv_write(kv: torch.Tensor, k: torch.Tensor, v: torch.Tensor, slots: torch.Tensor) -> None:
    """k, v `(T, H, D)` into `kv` [B, H, N, 2D] at `slots`."""
    d = k.shape[-1]
    valid = slots >= 0
    blk, pos = _slot_split(slots[valid], kv.shape[2])
    kv[blk, :, pos, :d] = k[valid].to(kv.dtype)
    kv[blk, :, pos, d:] = v[valid].to(kv.dtype)


def rel_attention(q: torch.Tensor, rel_logits: torch.Tensor, sinks: torch.Tensor,
                  kv: torch.Tensor, block_table: torch.Tensor, seq_lens: torch.Tensor,
                  query_start_loc: torch.Tensor, scale: float, window: int | None,
                  rel_extent: int) -> torch.Tensor:
    """Relative-bias paged attention with sinks, fp32 math, bf16 output.

    q `(T, Hq, D)`, rel_logits `(T, Hq, E)` (E = rel_extent; column d is the
    bias at backward distance d), sinks `(T, Hq)`. Keys at backward distance
    `>= window` are masked (sliding window); the bias is zero for distances
    `>= rel_extent`."""
    T, Hq, D = q.shape
    Hk = kv.shape[1]
    g = Hq // Hk
    bs = kv.shape[2]
    out = torch.empty(T, Hq, D, dtype=torch.float32, device=q.device)
    qsl = query_start_loc.tolist()
    for r in range(len(qsl) - 1):
        a, b = qsl[r], qsl[r + 1]
        if b == a:
            continue
        L = int(seq_lens[r])
        qlen = b - a
        q_pos = torch.arange(L - qlen, L, device=q.device)
        lo = 0 if window is None else max(0, L - qlen - window + 1)
        k_pos = torch.arange(lo, L, device=q.device)
        blocks = block_table[r, k_pos // bs]
        kk = kv[blocks, :, k_pos % bs, :D].float()  # (L', Hk, D)
        vv = kv[blocks, :, k_pos % bs, D:].float()
        kk = kk.repeat_interleave(g, 1)
        vv = vv.repeat_interleave(g, 1)
        dist = q_pos[:, None] - k_pos[None, :]  # (qlen, L')
        s = torch.einsum("thd,khd->htk", q[a:b].float(), kk) * scale
        bias_idx = dist.clamp(0, rel_extent - 1)
        rl = rel_logits[a:b].float().permute(1, 0, 2)  # (H, qlen, E)
        bias = rl.gather(-1, bias_idx[None].expand(Hq, -1, -1))
        bias = bias.masked_fill(((dist < 0) | (dist >= rel_extent))[None], 0.0)
        s = s + bias
        mask = dist < 0
        if window is not None:
            mask = mask | (dist >= window)
        s = s.masked_fill(mask[None], float("-inf"))
        sink = sinks[a:b].float().t()[..., None]  # (H, qlen, 1)
        p = torch.softmax(torch.cat([s, sink], -1), -1)[..., :-1]
        out[a:b] = torch.einsum("htk,khd->thd", p, vv)
    return out.to(torch.bfloat16)


# --------------------------------------------------------------------- MoE


def route(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, top_k: int,
          route_scale: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Sigmoid router with correction bias (fp32): (weights fp32, ids int32)."""
    logits = F.linear(x.float(), weight.float())
    scores = torch.sigmoid(logits)
    ids = torch.topk(scores + bias.float(), top_k, dim=-1, sorted=False).indices
    w = scores.gather(1, ids)
    w = w / (w.sum(-1, keepdim=True) + 1e-20)
    return w * route_scale, ids.to(torch.int32)


def situ(gate_up: torch.Tensor, gate_cap: float, up_cap: float) -> torch.Tensor:
    d = gate_up.shape[-1] // 2
    g, u = gate_up[..., :d].float(), gate_up[..., d:].float()
    return (gate_cap * torch.tanh(g / gate_cap) * torch.sigmoid(g)
            * (up_cap * torch.tanh(u / up_cap))).to(gate_up.dtype)


def moe_experts(x: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor, expert_weights,
                num_experts: int, gate_cap: float, up_cap: float) -> torch.Tensor:
    """bf16 expert MLPs; `expert_weights(e)` -> (w13 `(2I, H)` [gate; up], w2 `(H, I)`)."""
    out = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
    flat = ids.long()
    for e in torch.unique(flat).tolist():
        tok, slot = torch.where(flat == e)
        w13, w2 = expert_weights(e)
        h = situ(F.linear(x[tok], w13), gate_cap, up_cap)
        h = F.linear(h, w2)
        out.index_add_(0, tok, h * weights[tok, slot, None])
    return out.to(x.dtype)
