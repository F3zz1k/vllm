# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Decode the hit NVFP4 experts into an fp8 (or bf16) scratch for XpuFusedMoe.

Storage stays 4-bit (repacked codes + E4M3 scale per 16 + per-expert factor,
see `triton_moe_nvfp4.repack`); every MoE call decodes only the experts that
some token routes to, into one scratch shared by all layers, laid out as
XpuFusedMoe wants it ([E, K, N], K-major). An fp8 scratch uses one scale per
expert (amax over the expert / 448, computed at load). Elementwise, so it
runs near the read+write bandwidth; graph-capturable (unhit experts exit).
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .triton_moe_nvfp4 import GROUP, _e2m1


@triton.jit
def _dequant_kernel(Wq, Ws, F, Inv, Hit, Out, N, K, s_we, s_wn, s_se, s_sn, s_oe, s_ok,
                    BN: tl.constexpr, FP8: tl.constexpr):
    e = tl.program_id(0)
    if tl.load(Hit + e) == 0:
        return
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    nm = n < N
    j = tl.program_id(2)  # 128-value group along K
    c = tl.arange(0, 64)
    byte = tl.load(Wq + e * s_we + n[:, None] * s_wn + (j * 64 + c)[None, :],
                   mask=nm[:, None], other=0).to(tl.int32)
    sb = Ws + e * s_se + n[:, None] * s_sn + j * 8 + (c // 16)[None, :]
    slo = tl.load(sb, mask=nm[:, None], other=0.0).to(tl.float32)
    shi = tl.load(sb + 4, mask=nm[:, None], other=0.0).to(tl.float32)
    mul = tl.load(F + e) * tl.load(Inv + e)
    lo = _e2m1(byte & 15).to(tl.float32) * slo * mul
    hi = _e2m1(byte >> 4).to(tl.float32) * shi * mul
    k = j * 128 + c
    # [E, K, N]: transpose in registers so the stores run along contiguous N
    dst = Out + e * s_oe + k[:, None] * s_ok + n[None, :]
    if FP8:
        tl.store(dst, tl.trans(lo).to(tl.float8e4nv), mask=nm[None, :])
        tl.store(dst + 64 * s_ok, tl.trans(hi).to(tl.float8e4nv), mask=nm[None, :])
    else:
        tl.store(dst, tl.trans(lo).to(tl.bfloat16), mask=nm[None, :])
        tl.store(dst + 64 * s_ok, tl.trans(hi).to(tl.bfloat16), mask=nm[None, :])


def dequant_hit(wq: torch.Tensor, ws: torch.Tensor, f: torch.Tensor, inv: torch.Tensor,
                hit: torch.Tensor, out: torch.Tensor) -> None:
    """wq (E, N, K/2) repacked codes, ws (E, N, K/16), f (E,) factor, inv (E,)
    1/scratch-scale (1 for bf16), hit (E,) int32 -> out (E, K, N) for hit experts."""
    E, N, K2 = wq.shape
    K = 2 * K2
    assert K % GROUP == 0
    bn = 64
    _dequant_kernel[(E, triton.cdiv(N, bn), K // GROUP)](
        wq, ws, f, inv, hit, out, N, K, wq.stride(0), wq.stride(1), ws.stride(0), ws.stride(1),
        out.stride(0), out.stride(1), BN=bn, FP8=out.dtype == torch.float8_e4m3fn,
        num_warps=4)
