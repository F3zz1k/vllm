# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 op dispatch (XPU port).

`VLLM_INKLING2_OPS=triton` (default on XPU, and on CPU under
TRITON_INTERPRET=1) runs the Triton kernels; `ref` the torch references of
`ref.py` (CPU default; not graph-capturable).
"""

from __future__ import annotations

import os

import torch

from . import ref
from .ref import log_scaling_tau, message_blend, rms_norm, route  # noqa: F401


def _backend() -> str:
    b = os.environ.get("VLLM_INKLING2_OPS")
    if b:
        return b
    if os.environ.get("TRITON_INTERPRET") == "1":
        return "triton"
    return "triton" if hasattr(torch, "xpu") and torch.xpu.is_available() else "ref"


BACKEND = _backend()


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


def _ref_history(owner, md, stream, x, positions):
    T, k = x.shape[0], owner.kernel_size
    if md is None or owner.kv_cache.numel() == 0:
        hist = x.new_zeros((T, k, x.shape[1]), dtype=torch.float32)
        hist[:, -1] = x.float()
        return hist
    ref.conv_write(owner.kv_cache, owner.layout, stream, x, md.slot_mapping[:T])
    return ref.conv_history(owner.kv_cache, owner.layout, stream, positions[:T],
                            md.block_table, md.seq_idx[:T].long(), k)


def conv(owner, md, stream: str, x: torch.Tensor, positions: torch.Tensor,
         weight: torch.Tensor, dyn: torch.Tensor | None = None, norm: int = 0,
         clip: float = 0.0, eps: float = 1e-6, kv_owner=None, kv_md=None) -> torch.Tensor:
    """Short conv of one stream over the visit's conv state (static taps, or
    dynamic with `dyn`), then optionally a per-head norm (1 = L2 clip, 2 =
    unweighted RMS) and a store into the visit's KV cache (K for stream 'k',
    V for 'v'). Returns bf16 `(T, C)`."""
    T = x.shape[0]
    hph = owner.layout.stream(stream)[1]
    have_md = md is not None and owner.kv_cache.numel() > 0
    if BACKEND == "triton" and have_md:
        from .triton_sconv import short_conv

        kv = kv_owner.kv_cache if (kv_owner is not None and kv_md is not None) else None
        return short_conv(x, owner.kv_cache, owner.layout.stream(stream)[0], hph,
                          md.slot_mapping[:T], md.block_table, md.seq_idx[:T],
                          md.query_start[:T], positions[:T], weight, dyn=dyn, norm=norm,
                          clip=clip, eps=eps, kv=kv,
                          kv_slots=None if kv is None else kv_md.slot_mapping[:T],
                          kv_off=0 if stream == "k" else hph)
    hist = _ref_history(owner, md if have_md else None, stream, x, positions)
    if dyn is None:
        y = ref.static_conv(hist, weight).to(torch.bfloat16)
    else:
        y = ref.dynamic_conv(hist, weight, dyn, owner.layout.num_heads).to(torch.bfloat16)
    if norm:
        yh = y.view(T, -1, hph).float()
        yh = ref.l2_clip(yh, clip) if norm == 1 else ref.unweighted_rms(yh, eps)
        y = yh.to(torch.bfloat16).view(T, -1)
    if kv_owner is not None and kv_md is not None and kv_owner.kv_cache.numel() > 0:
        slots = kv_md.slot_mapping[:T]
        valid = slots >= 0
        cache = kv_owner.kv_cache
        page = cache.shape[2]
        o = 0 if stream == "k" else hph
        cache[slots[valid] // page, :, slots[valid] % page, o:o + hph] = (
            y[valid].view(-1, owner.layout.num_heads, hph).to(cache.dtype))
    return y


def attention(owner, md, q, rel_logits, sinks, scale: float, window: int | None,
              rel_extent: int) -> torch.Tensor:
    """Attention over the visit's KV cache (K/V already stored by `conv`)."""
    n_req = md.query_start_loc.shape[0] - 1
    if BACKEND == "triton":
        from .triton_attention import rel_attention

        decode = md.max_query_len <= 8
        splits = (8 if window is None else 2) if decode else 1
        return rel_attention(q, owner.kv_cache, rel_logits, sinks, md.block_table, md.seq_lens,
                             md.query_start_loc, scale, window, rel_extent, n_req,
                             out=torch.zeros_like(q), num_splits=splits)
    n = md.num_actual_tokens
    out = torch.zeros_like(q)
    out[:n] = ref.rel_attention(q[:n], rel_logits[:n], sinks[:n], owner.kv_cache,
                                md.block_table, md.seq_lens, md.query_start_loc, scale,
                                window, rel_extent)
    return out


def _fused_moe(moe):
    k = getattr(moe, "_kernel", None)
    if k is None:
        from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe

        t = moe.store.t
        k = XpuFusedMoe(w13=t["w13"], w13_scales=t.get("w13_scale"), w13_bias=None,
                        w2=t["w2"], w2_scales=t.get("w2_scale"), w2_bias=None,
                        n_experts_per_token=moe.top_k, activation="situ",
                        num_experts=moe.num_experts, activation_situ_beta=moe.gate_cap,
                        activation_situ_linear_beta=moe.up_cap)
        moe._kernel = k
    return k


def experts(moe, x: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    if BACKEND == "triton" and x.is_xpu and moe.store.fmt in ("bf16", "fp8"):
        out = torch.empty_like(x)
        _fused_moe(moe).apply(out, x, weights, ids)
        return out
    return ref.moe_experts(x, weights, ids, moe.expert_weights, moe.num_experts,
                           moe.gate_cap, moe.up_cap)
