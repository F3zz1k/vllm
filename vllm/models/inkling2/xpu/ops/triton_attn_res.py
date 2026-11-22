# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Block attention residual (Triton, XPU): softmax mix of residual blocks.

For each token and pseudo-query `q` (fp32, pre-scaled by `D**-0.5`) the
sources `x_n` = stored blocks, then the partial block (+ an optional
addend), in fp32, are scored `(q . x_n) * rsqrt(mean(x_n^2) + eps)`; the
output is `sum_n softmax_n * x_n * rsqrt(mean(x_n^2) + eps)` (normalized
values), optionally followed by the next sublayer's weighted RMSNorm
(bf16 out) in the same kernel. One program per (query, token); two passes
over the sources (scores, then the mix), D in chunks.

(Upstream's NVIDIA kernel does not build for Intel GPUs: module build failure.)
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _src(Stack, s_st, Last, Addend, t, s_row, d, dm, n: tl.constexpr, N_STACK: tl.constexpr,
         HAS_ADDEND: tl.constexpr):
    if n < N_STACK:
        x = tl.load(Stack + n * s_st + t * s_row + d, mask=dm, other=0.0).to(tl.float32)
    else:
        x = tl.load(Last + t * s_row + d, mask=dm, other=0.0).to(tl.float32)
        if HAS_ADDEND:
            x += tl.load(Addend + t * s_row + d, mask=dm, other=0.0).to(tl.float32)
    return x


@triton.jit
def _attn_res_kernel(Stack, Last, Addend, Q, W, Out, s_st, s_row, s_oq, s_ot, eps, norm_eps,
                     D: tl.constexpr, N_STACK: tl.constexpr, HAS_ADDEND: tl.constexpr,
                     APPLY_NORM: tl.constexpr, BLOCK: tl.constexpr, NP2: tl.constexpr):
    qi = tl.program_id(0)
    t = tl.program_id(1)
    N: tl.constexpr = N_STACK + 1
    lanes = tl.arange(0, NP2)
    scores = tl.full([NP2], float("-inf"), tl.float32)
    inv = tl.zeros([NP2], tl.float32)
    for n in tl.static_range(N):
        dot = 0.0
        ss = 0.0
        for c in tl.static_range(0, D, BLOCK):
            d = c + tl.arange(0, BLOCK)
            dm = d < D
            x = _src(Stack, s_st, Last, Addend, t, s_row, d, dm, n, N_STACK, HAS_ADDEND)
            q = tl.load(Q + qi * D + d, mask=dm, other=0.0)
            dot += tl.sum(x * q, 0)
            ss += tl.sum(x * x, 0)
        r = 1.0 / tl.sqrt(ss / D + eps)
        scores = tl.where(lanes == n, dot * r, scores)
        inv = tl.where(lanes == n, r, inv)
    p = tl.exp(scores - tl.max(scores, 0))
    p = p / tl.sum(p, 0)
    wgt = p * inv  # weight of the raw source n
    # pass 2: the mix, chunk by chunk (and its sum of squares for the norm)
    ss_out = 0.0
    for c in tl.static_range(0, D, BLOCK):
        d = c + tl.arange(0, BLOCK)
        dm = d < D
        acc = tl.zeros([BLOCK], tl.float32)
        for n in tl.static_range(N):
            x = _src(Stack, s_st, Last, Addend, t, s_row, d, dm, n, N_STACK, HAS_ADDEND)
            acc += x * tl.sum(tl.where(lanes == n, wgt, 0.0), 0)
        if APPLY_NORM:
            ss_out += tl.sum(acc * acc, 0)
            tl.store(Out + qi * s_oq + t * s_ot + d, acc, mask=dm)  # fp32 scratch row
        else:
            tl.store(Out + qi * s_oq + t * s_ot + d, acc, mask=dm)
    if APPLY_NORM:
        r = 1.0 / tl.sqrt(ss_out / D + norm_eps)
        for c in tl.static_range(0, D, BLOCK):
            d = c + tl.arange(0, BLOCK)
            dm = d < D
            acc = tl.load(Out + qi * s_oq + t * s_ot + d, mask=dm, other=0.0)
            w = tl.load(W + qi * D + d, mask=dm, other=0.0).to(tl.float32)
            tl.store(Out + qi * s_oq + t * s_ot + d, w * (acc * r), mask=dm)


def attn_res(stack: torch.Tensor, last: torch.Tensor, addend: torch.Tensor | None,
             queries: torch.Tensor, norm_weight: torch.Tensor | None, norm_eps: float,
             normalize_values: bool = True, eps: float = 1e-6) -> torch.Tensor:
    """stack (N, T, D) (N may be 0), last/addend (T, D), queries fp32 (Q, D);
    returns fp32 (Q, T, D), or bf16 normalized with norm_weight (Q, D)."""
    assert normalize_values
    T, D = last.shape
    Q = queries.shape[0]
    N = stack.shape[0]
    assert last.stride(1) == 1 and (N == 0 or (stack.stride(1) == last.stride(0)
                                               and stack.stride(2) == 1))
    out = torch.empty((Q, T, D), dtype=torch.float32, device=last.device)
    if T:
        _attn_res_kernel[(Q, T)](
            stack if N else last, last, addend if addend is not None else last, queries,
            norm_weight if norm_weight is not None else queries, out,
            stack.stride(0) if N else 0, last.stride(0), out.stride(0), out.stride(1), eps,
            norm_eps, D=D, N_STACK=N, HAS_ADDEND=addend is not None,
            APPLY_NORM=norm_weight is not None, BLOCK=512, NP2=triton.next_power_of_2(N + 1),
            num_warps=4)
    return out if norm_weight is None else out.to(torch.bfloat16)
