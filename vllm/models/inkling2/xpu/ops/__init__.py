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


def ref_interp() -> bool:
    return os.environ.get("TRITON_INTERPRET") == "1"


def attn_res(stack: torch.Tensor, n: int, last: torch.Tensor, addend: torch.Tensor | None,
             queries: torch.Tensor, norm_weight: torch.Tensor | None = None,
             norm_eps: float = 1e-6) -> torch.Tensor:
    """Block AttnRes over `[*stack[:n], last (+ addend)]` with normalized values,
    one mix per query row: fp32 `(Q, T, D)`, or bf16 RMS-normalized with the
    matching row of `norm_weight` `(Q, D)` (the next sublayer's norm, fused)."""
    if BACKEND == "triton" and last.is_xpu or (BACKEND == "triton" and ref_interp()):
        from .triton_attn_res import attn_res as tri

        return tri(stack[:n], last.contiguous(), None if addend is None else addend.contiguous(),
                   queries.contiguous(), None if norm_weight is None else norm_weight.contiguous(),
                   norm_eps, True)
    partial = last.float() if addend is None else last.float() + addend.float()
    mixes = ref.attn_res(stack[:n], partial, queries)
    if norm_weight is None:
        return mixes
    return torch.stack([ref.rms_norm(mixes[q], norm_weight[q], norm_eps)
                        for q in range(mixes.shape[0])])


def add3(last: torch.Tensor, addend: torch.Tensor | None, m: torch.Tensor) -> torch.Tensor:
    """(last (+ addend) + m) in fp32, rounded to bf16 once (the layer output)."""
    y = last.float() if addend is None else last.float() + addend.float()
    return (y + m.float()).to(torch.bfloat16)


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


def rel_logits(r: torch.Tensor, proj: torch.Tensor, attn) -> torch.Tensor:
    """(T, H, E) relative logits r @ proj (bf16, as the reference). The Triton
    kernel takes them reversed along E (from a flipped copy of proj, free)."""
    if BACKEND == "triton":
        rev = getattr(attn, "_rel_proj_rev", None)
        if rev is None or rev.data_ptr() == 0 or rev.shape != proj.shape:
            rev = attn._rel_proj_rev = proj.flip(-1).contiguous()
        return torch.matmul(r, rev)
    return torch.matmul(r, proj)


def attention(owner, md, q, rel_logits, sinks, scale: float, window: int | None,
              rel_extent: int) -> torch.Tensor:
    """Attention over the visit's KV cache (K/V already stored by `conv`)."""
    n_req = md.query_start_loc.shape[0] - 1
    if BACKEND == "triton":
        from .triton_attention import rel_attention

        decode = md.max_query_len <= 8
        splits = (16 if window is None else 8) if decode else 1
        return rel_attention(q, owner.kv_cache, rel_logits, sinks, md.block_table, md.seq_lens,
                             md.query_start_loc, scale, window, rel_extent, n_req,
                             out=torch.zeros_like(q), num_splits=splits, rel_reversed=True)
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


_SCRATCH: dict = {}


def _nvfp4_via_scratch(moe, x, weights, ids, fp8: bool):
    """NVFP4 storage, compute on XpuFusedMoe: decode the hit experts into one
    scratch shared by every layer (fp8 with per-expert scales, or bf16)."""
    from .triton_nvfp4_dequant import dequant_hit

    t = moe.store.t
    E, I2, H2 = t["w13_packed"].shape
    H, I = 2 * H2, I2 // 2
    key = (E, H, I, fp8, x.device)
    sc = _SCRATCH.get(key)
    if sc is None:
        from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe

        dt = torch.float8_e4m3fn if fp8 else torch.bfloat16
        w13 = torch.empty((E, H, 2 * I), dtype=dt, device=x.device)
        w2 = torch.empty((E, I, H), dtype=dt, device=x.device)
        s13 = torch.ones(E, dtype=torch.float32, device=x.device) if fp8 else None
        s2 = torch.ones(E, dtype=torch.float32, device=x.device) if fp8 else None
        k = XpuFusedMoe(w13=w13, w13_scales=s13, w13_bias=None, w2=w2, w2_scales=s2,
                        w2_bias=None, n_experts_per_token=moe.top_k, activation="situ",
                        num_experts=E, activation_situ_beta=moe.gate_cap,
                        activation_situ_linear_beta=moe.up_cap)
        ones = torch.ones(E, dtype=torch.float32, device=x.device)
        sc = _SCRATCH[key] = (k, w13, w2, s13, s2, ones)
    k, w13, w2, s13, s2, ones = sc
    hit = torch.zeros(E, dtype=torch.int32, device=x.device).scatter_(
        0, ids.reshape(-1).long(), 1)
    dequant_hit(t["w13_packed"], t["w13_bscale"], t["w13_factor"],
                t["w13_inv8"] if fp8 else ones, hit, w13)
    dequant_hit(t["w2_packed"], t["w2_bscale"], t["w2_factor"],
                t["w2_inv8"] if fp8 else ones, hit, w2)
    if fp8:
        s13.copy_(t["w13_s8"])
        s2.copy_(t["w2_s8"])
    out = torch.empty_like(x)
    k.apply(out, x, weights, ids)
    return out


def experts(moe, x: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    if BACKEND == "triton" and x.is_xpu and moe.store.fmt in ("bf16", "fp8"):
        out = torch.empty_like(x)
        _fused_moe(moe).apply(out, x, weights, ids)
        return out
    # NVFP4 storage: VLLM_INKLING2_NVFP4_COMPUTE = auto (direct 4-bit Triton
    # kernel for small batches, fp8 scratch + XpuFusedMoe for large ones;
    # measured crossover ~64-128 routed rows), triton, fp8 or bf16.
    mode = os.environ.get("VLLM_INKLING2_NVFP4_COMPUTE", "auto")
    if mode == "auto":
        thr = int(os.environ.get("VLLM_INKLING2_NVFP4_SCRATCH_MIN_TOKENS", "256"))
        mode = "fp8" if x.shape[0] >= thr else "triton"
    if BACKEND == "triton" and moe.store.fmt == "nvfp4" and x.is_xpu and mode in ("fp8", "bf16"):
        return _nvfp4_via_scratch(moe, x, weights, ids, mode == "fp8")
    if BACKEND == "triton" and moe.store.fmt == "nvfp4" and (x.is_xpu or ref_interp()):
        from .triton_moe_nvfp4 import nvfp4_moe

        t = moe.store.t
        return nvfp4_moe(x, weights, ids, t["w13_packed"], t["w13_bscale"], t["w13_factor"],
                         t["w2_packed"], t["w2_bscale"], t["w2_factor"], moe.gate_cap, moe.up_cap)
    return ref.moe_experts(x, weights, ids, moe.expert_weights, moe.num_experts,
                           moe.gate_cap, moe.up_cap)
