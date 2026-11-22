# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVFP4 routed experts on XPU: the vendor routing ops around our NVFP4 grouped GEMM.

remap (group token rows by expert) -> grouped GEMM w13 -> SiTU -> grouped GEMM w2 ->
weighted gather. The GEMM (`_inkling2_C.grouped_gemm_nvfp4`) reads the checkpoint
bytes directly: e2m1 pairs (low nibble first), an E4M3 scale per 16 values applied
per 16-wide MMA k-block (exact), the per-expert factor in the epilogue. No scratch.
"""

from __future__ import annotations

import torch

from . import sycl_attention


def available() -> bool:
    return sycl_attention.available() and hasattr(torch.ops._inkling2_C, "grouped_gemm_nvfp4")


def gemm(a, w, s, f, out, rows_per_expert):
    torch.ops._inkling2_C.grouped_gemm_nvfp4(a, w, s, f, out, rows_per_expert, w.shape[1],
                                             w.shape[2] * 2)
    return out


def moe(x, weights, ids, w13, s13, f13, w2, s2, f2, gate_cap, up_cap):
    """x bf16 (T, H); weights fp32 / ids int32 (T, k); w13 (E, 2I, H/2) [gate; up] rows,
    s13 (E, 2I, H/16) e4m3 bytes, f13 (E,); w2 (E, H, I/2), s2, f2 -> bf16 (T, H)."""
    import vllm_xpu_kernels.fused_moe_interface  # noqa: F401  (loads _moe_C / _C)

    assert available(), "_inkling2_C with grouped_gemm_nvfp4 is required for nvfp4 experts"

    T, H = x.shape
    E, I2, _ = w13.shape
    k = ids.shape[1]
    rows = T * k
    dev = x.device
    remapped = torch.empty((rows, H), dtype=x.dtype, device=dev)
    rpe = torch.zeros((E,), dtype=torch.int32, device=dev)
    unp = torch.empty((T, k), dtype=torch.int32, device=dev)
    torch.ops._moe_C.remap_hidden_states(
        hidden_states=x, hidden_states_scales=None, remapped_hidden_states=remapped,
        remapped_hidden_states_scales=None, expert_map=None, rows_per_expert=rpe,
        unpermuted_row_to_permuted_row=unp, topk_ids=ids, total_experts_num=E,
        local_experts_num=E)
    g1 = gemm(remapped, w13, s13, f13, torch.empty((rows, I2), dtype=x.dtype, device=dev), rpe)
    act = torch.empty((rows, I2 // 2), dtype=x.dtype, device=dev)
    torch.ops._C.situ_and_mul(act, g1, gate_cap, up_cap)
    g2 = gemm(act, w2, s2, f2, torch.empty((rows, H), dtype=x.dtype, device=dev), rpe)
    out = torch.empty_like(x)
    torch.ops._moe_C.moe_gather(out, g2, weights, unp, E)
    return out
