# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 op dispatch (XPU port): torch references (`ref.py`) for now."""

from __future__ import annotations

import torch

from . import ref
from .ref import log_scaling_tau, message_blend, rms_norm, route  # noqa: F401


def attn_res(blocks: list[torch.Tensor], partial: torch.Tensor,
             queries: torch.Tensor) -> torch.Tensor:
    if blocks:
        stack = torch.stack(blocks, 0)
    else:
        stack = partial.new_empty((0, *partial.shape))
    return ref.attn_res(stack, partial, queries)


def qr_norm(x: torch.Tensor, scale: torch.Tensor, clip: float,
            tau: torch.Tensor | None) -> torch.Tensor:
    """bf16 `(T, H, d)`: learned scale + L2 clip (rounded), then the log
    scaling `tau` (rounded again), as the reference's two modules."""
    y = ref.l2_clip(x.float() * scale.float(), clip).to(torch.bfloat16)
    if tau is not None:
        y = (y.float() * tau[:, None, None]).to(torch.bfloat16)
    return y


def k_norm(k: torch.Tensor, clip: float) -> torch.Tensor:
    return ref.l2_clip(k.float(), clip).to(torch.bfloat16)


def v_norm(v: torch.Tensor, eps: float) -> torch.Tensor:
    return ref.unweighted_rms(v, eps).to(torch.bfloat16)


def _history(owner, md, stream: str, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """fp32 `(T, k, C)` conv window of `x`'s stream, oldest first; stores `x`
    in the conv state first. Without metadata (dummy run) the past is zero."""
    T = x.shape[0]
    k = owner.kernel_size
    if md is None or owner.kv_cache.numel() == 0:
        hist = x.new_zeros((T, k, x.shape[1]), dtype=torch.float32)
        hist[:, -1] = x.float()
        return hist
    ref.conv_write(owner.kv_cache, owner.layout, stream, x, md.slot_mapping[:T])
    return ref.conv_history(owner.kv_cache, owner.layout, stream, positions[:T],
                            md.block_table, md.seq_idx[:T].long(), k)


def short_conv(owner, md, stream: str, x: torch.Tensor, positions: torch.Tensor,
               weight: torch.Tensor) -> torch.Tensor:
    return ref.static_conv(_history(owner, md, stream, x, positions), weight).to(torch.bfloat16)


def dynamic_conv(owner, md, stream: str, x: torch.Tensor, positions: torch.Tensor,
                 static_w: torch.Tensor, dyn: torch.Tensor, num_heads: int) -> torch.Tensor:
    hist = _history(owner, md, stream, x, positions)
    return ref.dynamic_conv(hist, static_w, dyn, num_heads).to(torch.bfloat16)


def attention(owner, md, q, k, v, rel_logits, sinks, scale: float, window: int | None,
              rel_extent: int) -> torch.Tensor:
    T = q.shape[0]
    ref.kv_write(owner.kv_cache, k, v, md.slot_mapping[:T])
    n = md.num_actual_tokens
    out = torch.zeros_like(q)
    out[:n] = ref.rel_attention(q[:n], rel_logits[:n], sinks[:n], owner.kv_cache,
                                md.block_table, md.seq_lens, md.query_start_loc, scale,
                                window, rel_extent)
    return out


def experts(moe, x: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    return ref.moe_experts(x, weights, ids, moe.expert_weights, moe.num_experts,
                           moe.gate_cap, moe.up_cap)
