# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 relative-bias paged attention on the SYCL flash-attention kernels.

`_inkling2_C.rel_attention` (vllm-xpu-kernels with the Inkling2 patch: its own
library, the vendor attention untouched) runs the vendor chunk-prefill /
paged-decode kernels with the relative bias added to the scores. The per-token
sink is applied here from the returned log-sum-exp:
`out *= sigmoid(lse - sink)` (the sink is one extra logit with a zero value).
"""

from __future__ import annotations

import os

import torch

_LOADED = None


def available() -> bool:
    global _LOADED
    if _LOADED is None:
        _LOADED = False
        try:
            import vllm_xpu_kernels._inkling2_C  # noqa: F401
            _LOADED = True
        except ImportError:
            path = os.environ.get("VLLM_INKLING2_C")
            if path and os.path.exists(path):
                import ctypes

                # its kernel libraries sit next to it (dev builds): open them by
                # path first, so LD_LIBRARY_PATH stays the image's
                for name in ("attn", "moe"):
                    lib = os.path.join(os.path.dirname(path), f"libinkling2_{name}_kernels_xe_2.so")
                    if os.path.exists(lib):
                        ctypes.CDLL(lib, mode=ctypes.RTLD_GLOBAL)
                torch.ops.load_library(path)
                _LOADED = True
    return _LOADED


def rel_attention(q: torch.Tensor, kv: torch.Tensor, rel: torch.Tensor, sinks: torch.Tensor,
                  block_table: torch.Tensor, seq_lens: torch.Tensor, cu_q: torch.Tensor,
                  scale: float, window: int | None, max_q: int, max_k: int,
                  num_splits: int = 1, out: torch.Tensor | None = None,
                  return_lse: bool = False, kv_scales=(None, None)):
    """q (T, Hq, D) bf16; kv [B, Hk, page, 2D] (K | V) bf16, or fp8 with `kv_scales`
    (k_scale, v_scale: one-element fp32 tensors); rel (T, Hq, E) bf16 with
    column d = bias at distance d; sinks (T, Hq); returns (T, Hq, D) bf16. With
    `return_lse`, the sinks are not applied: returns (out, lse (Hq, T) fp32)."""
    D = q.shape[-1]
    kvt = kv.transpose(1, 2)  # [B, page, Hk, 2D]
    k, v = kvt[..., :D], kvt[..., D:]
    if out is None:
        out = torch.empty_like(q)
    lse = torch.empty((q.shape[1], q.shape[0]), dtype=torch.float32, device=q.device)
    torch.ops._inkling2_C.rel_attention(
        q, k, v, out, cu_q, seq_lens, block_table, max_q, max_k, scale,
        -1 if window is None else window - 1, rel, lse, num_splits,
        *(() if kv_scales[0] is None else kv_scales))  # bf16: the pre-fp8 op signature too
    if return_lse:
        return out, lse
    gate = torch.sigmoid(lse.t() - sinks.float())  # (T, Hq)
    out.copy_((out.float() * gate[..., None]).to(out.dtype))
    return out
