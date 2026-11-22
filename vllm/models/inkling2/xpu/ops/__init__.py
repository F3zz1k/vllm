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


def qkr(attn, yq: torch.Tensor, tau: torch.Tensor | None, fused: bool):
    """q (T, H, D) and the relative logits (T, H, E) from the projection's q / r
    columns (`yq` rows: [q | r | ...]). `fused`: one Triton kernel (distance-order
    logits, for the SYCL attention); else the torch ops."""
    T = yq.shape[0]
    H, D, R = attn.num_heads, attn.head_dim, attn.d_rel
    if fused:
        from .triton_glue import qkr_prep

        return qkr_prep(yq, attn.q_scale, attn.r_scale, tau, attn.rel_proj, H, D, R, H * D,
                        attn.qk_clip, attn.rel_clip)
    q = qr_norm(yq[:, :H * D].view(T, H, D), attn.q_scale, attn.qk_clip, tau)
    r = qr_norm(yq[:, H * D:H * D + H * R].view(T, H, R), attn.r_scale, attn.rel_clip, tau)
    return q, rel_logits(r, attn.rel_proj, attn)


def blend(layer, s: slice, dests: torch.Tensor, messages: torch.Tensor, norm_w: torch.Tensor,
          eps: float) -> torch.Tensor:
    """Loop controllers of streams `s` then the next sublayer's RMSNorm: fp32 `(S, T, D)`
    dests / messages -> bf16 `(S, T, D)`. h = down(cat(rms(dest), rms(message))) (bf16
    GEMM), [gain; forget](h) (fp32 GEMM), (1 - tanh(forget)) dest + tanh(gain) message."""
    if BACKEND == "triton" and dests.is_xpu:
        from .triton_glue import blend_out, blend_prep

        cat = blend_prep(dests, messages)
        h = torch.bmm(cat, layer.ctrl_down[s].transpose(1, 2)).float()
        gf = torch.bmm(h, layer.ctrl_gf[s].transpose(1, 2))
        return blend_out(gf, dests, messages, norm_w, eps)
    rms = ref.unweighted_rms
    cat = torch.cat([rms(dests), rms(messages)], -1).to(layer.ctrl_down.dtype)
    h = torch.bmm(cat, layer.ctrl_down[s].transpose(1, 2)).float()
    gf = torch.bmm(h, layer.ctrl_gf[s].transpose(1, 2))
    D = dests.shape[-1]
    x = (1 - torch.tanh(gf[..., D:])) * dests + torch.tanh(gf[..., :D]) * messages
    return ref.rms_norm(x, norm_w[:, None, :], eps)


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
         clip: float = 0.0, eps: float = 1e-6, kv_owner=None, kv_md=None,
         last: torch.Tensor | None = None, addend: torch.Tensor | None = None) -> torch.Tensor:
    """Short conv of one stream over the visit's conv state (static taps, or
    dynamic with `dyn`), then optionally a per-head norm (1 = L2 clip, 2 =
    unweighted RMS) and a store into the visit's KV cache (K for stream 'k',
    V for 'v'). Returns bf16 `(T, C)`; with `last` (and `addend`), the layer output
    last (+ addend) + conv in fp32, rounded once."""
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
                          kv_off=0 if stream == "k" else hph,
                          last=None if last is None else last.contiguous(),
                          addend=None if addend is None else addend.contiguous())
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
    return y if last is None else add3(last, addend, y)


def _use_sycl_attention() -> bool:
    if os.environ.get("VLLM_INKLING2_ATTN", "sycl") != "sycl" or BACKEND != "triton":
        return False
    from . import sycl_attention

    return sycl_attention.available()


def rel_logits(r: torch.Tensor, proj: torch.Tensor, attn) -> torch.Tensor:
    """(T, H, E) relative logits r @ proj (bf16, as the reference). The Triton
    kernel takes them reversed along E (from a flipped copy of proj, free);
    the SYCL kernel in distance order."""
    if BACKEND == "triton" and not _use_sycl_attention():
        rev = getattr(attn, "_rel_proj_rev", None)
        if rev is None or rev.data_ptr() == 0 or rev.shape != proj.shape:
            rev = attn._rel_proj_rev = proj.flip(-1).contiguous()
        return torch.matmul(r, rev)
    return torch.matmul(r, proj)


def sycl_ready(owner) -> bool:
    """The SYCL attention (and the glue fused around it) serves this KV cache."""
    return _use_sycl_attention() and owner.kv_cache.shape[2] == 64


def attention(owner, md, q, rel_logits, sinks, scale: float, window: int | None,
              rel_extent: int, gate: torch.Tensor | None = None) -> torch.Tensor:
    """Attention over the visit's KV cache (K/V already stored by `conv`): (T, H, D)
    bf16, or with `gate` (T, H*D) the sigmoid-gated output (T, H*D) bf16."""
    T, H, D = q.shape
    out = torch.zeros_like(q)
    gated = None if gate is None else torch.empty((T, H * D), dtype=q.dtype, device=q.device)
    _attention_into(out, q, rel_logits, sinks, gated, gate, owner=owner, scale=scale,
                    window=window, rel_extent=rel_extent)
    return out if gate is None else gated


def _attention_into(out, q, rel_logits, sinks, gated, gate, *, owner, scale, window,
                    rel_extent):
    """Writes `out` in place. The kernels take host-side lengths (max query / key
    length, token count), so under breakable XPU graphs this is an eager break: on
    replay it runs with the current step's metadata, looked up here, not the
    capture's."""
    from .. import cache  # the harness patches cache.layer_metadata

    md = cache.layer_metadata(owner.prefix)
    n_req = md.query_start_loc.shape[0] - 1
    n = md.num_actual_tokens
    # the SYCL kernels are built for 64-token KV pages only (vLLM's XPU block size)
    if _use_sycl_attention() and owner.kv_cache.shape[2] == 64:
        from . import sycl_attention

        decode = md.max_query_len <= 1
        # Fixed per layer type (not per context length): a graph replays the
        # split count it was captured with.
        splits = (int(os.environ.get("VLLM_INKLING2_DECODE_SPLITS_GLOBAL", "16"))
                  if window is None else 4) if decode else 1
        if gated is not None:
            from .triton_glue import attn_post

            _, lse = sycl_attention.rel_attention(
                q[:n], owner.kv_cache, rel_logits[:n], sinks[:n], md.block_table, md.seq_lens,
                md.query_start_loc, scale, window, md.max_query_len, md.max_seq_len,
                num_splits=splits, out=out[:n], return_lse=True)
            attn_post(out, lse, sinks, gate, gated, n)
            return
        sycl_attention.rel_attention(q[:n], owner.kv_cache, rel_logits[:n], sinks[:n],
                                     md.block_table, md.seq_lens, md.query_start_loc,
                                     scale, window, md.max_query_len, md.max_seq_len,
                                     num_splits=splits, out=out[:n])
        return
    if BACKEND == "triton":
        from .triton_attention import rel_attention

        decode = md.max_query_len <= 8
        splits = (16 if window is None else 8) if decode else 1
        rel_attention(q, owner.kv_cache, rel_logits, sinks, md.block_table, md.seq_lens,
                      md.query_start_loc, scale, window, rel_extent, n_req, out=out,
                      num_splits=splits, rel_reversed=True)
    else:
        out[:n] = ref.rel_attention(q[:n], rel_logits[:n], sinks[:n], owner.kv_cache,
                                    md.block_table, md.seq_lens, md.query_start_loc, scale,
                                    window, rel_extent)
    if gated is not None:
        gated.copy_((out.float().view(gated.shape) * torch.sigmoid(gate.float())).to(gated.dtype))


try:
    from vllm.compilation.breakable_cudagraph import eager_break_during_capture

    _attention_into = eager_break_during_capture(_attention_into)
except ImportError:  # pragma: no cover
    pass


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


def route_fused(x, weight, bias, top_k: int, route_scale: float):
    """Router: the fused Triton kernel on XPU, else the reference."""
    if BACKEND == "triton" and x.is_xpu:
        from .triton_glue import route as tri

        return tri(x, weight, bias, top_k, route_scale)
    return ref.route(x, weight, bias, top_k, route_scale)


def experts(moe, x: torch.Tensor, weights: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    if BACKEND == "triton" and x.is_xpu and moe.store.fmt in ("bf16", "fp8"):
        out = torch.empty_like(x)
        _fused_moe(moe).apply(out, x, weights, ids)
        return out
    if BACKEND == "triton" and x.is_xpu and moe.store.fmt == "nvfp4":
        from . import moe_nvfp4

        t = moe.store.t
        return moe_nvfp4.moe(x, weights, ids, t["w13_q"], t["w13_bscale"], t["w13_factor"],
                             t["w2_q"], t["w2_bscale"], t["w2_factor"], moe.gate_cap, moe.up_cap)
    return ref.moe_experts(x, weights, ids, moe.expert_weights, moe.num_experts,
                           moe.gate_cap, moe.up_cap)
