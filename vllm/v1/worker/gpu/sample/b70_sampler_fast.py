# SPDX-License-Identifier: Apache-2.0
"""B70-SM: top-k-first temperature sampling and spec-decode verification.

Gate: B70_SAMPLER_FAST=1 (default off; unset keeps the stock path bit for bit).

The served config samples with top-k 20 (generation_config), so after top-k the
target distribution lives on <= 20 tokens, and the DFlash2 drafter's proposal q
lives on its selector_top_k (16) cached candidates. Stock work per verify step is
full-vocabulary: an fp32 copy of [rows, V], the temperature kernel, the pivot
search of the Qrita top-k/top-p kernel (one program per row, serial sweeps), a
stats kernel over V/8192 blocks, the rejection kernel, a V/1024-block resample
kernel and an insert kernel. This module replaces that with three launches:

  A. _split_topk_kernel   grid (rows, V/BLOCK): temperature, then the exact top-k of
                          each tile by iterative extraction, ties to the lower id.
  B. _merge_kernel        grid (rows,): merge the per-tile candidates into the row's
                          top-k (value desc, id asc), softmax over those k, top-p on
                          that renormalised distribution, normalised log-probs.
  C. _fast_reject_kernel  grid (reqs,): the stock Leviathan test and residual
                          resample, evaluated on the <=KP-token support; the draft
                          side reads its cached candidates (DFlash2 CandidateSampler).

Semantics held against the stock path (see tests/v1/worker/test_b70_sampler_fast.py):
  * top-k keeps exactly k tokens, ties at the k-th value broken to the lower token id
    (stock Qrita: "duplicate_keep_mask", index order);
  * top-p is taken on softmax(top-k masked logits), keep while exclusive cumulative
    mass < p, at least one token (stock apply_top_k_top_p_pytorch / Qrita);
  * temperature divides before the selection, greedy rows (T == 0) keep raw logits;
  * the uniform u and the Gumbel noise use the stock murmur3 streams keyed by
    (seed, position, token id), so a given draw lands on the same token as stock.
Exactness is "same distribution, same RNG draw": log-sum-exp and cumulative sums are
summed in a different order, so a draw exactly at a float boundary can differ.

Anything outside the supported envelope falls back to the stock path for the whole
step: min_p, logprobs in a processed mode, a top-k-disabled sampled request, top-k
above KMAX, grammar bitmasks, block/synthetic/adaptive verification, watermarking,
fp64 Gumbel, drafters without cached candidates.
"""

import os

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, tldevice, triton
from vllm.utils.math_utils import cdiv, next_power_of_2
from vllm.v1.worker.gpu.sample.gumbel import gumbel_noised_argmax, tl_rand32

logger = init_logger(__name__)

KMAX = 32  # largest top-k handled; above this the stock path runs
BLOCK = 4096  # vocab tile per program of kernel A (the stock XPU tile)
_BIG = 2147483647
_logged: set[str] = set()


def enabled() -> bool:
    return os.environ.get("B70_SAMPLER_FAST", "0") == "1"


def _log_once(tag: str, msg: str) -> None:
    if tag not in _logged:
        _logged.add(tag)
        logger.info("B70-SM: %s", msg)


@triton.jit
def _split_topk_kernel(
    cand_val_ptr,
    cand_idx_ptr,
    logits_ptr,
    logits_stride,
    eim_ptr,
    temp_ptr,
    topk_ptr,
    vocab_size,
    KP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    S: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    s = tl.program_id(1)
    req = tl.load(eim_ptr + row)
    temp = tl.load(temp_ptr + req).to(tl.float32)
    k = tl.minimum(tl.load(topk_ptr + req), KP)
    k = tl.where(temp == 0.0, 1, k)
    offs = s * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs < vocab_size
    x = tl.load(logits_ptr + row * logits_stride + offs, mask=mask, other=-float("inf"))
    x = x.to(tl.float32)
    if temp != 0.0:
        x = x / temp
    base = (row * S + s) * KP
    for j in range(k):
        m = tl.max(x, axis=0)
        i = tl.min(tl.where(x == m, offs, 2147483647), axis=0)
        tl.store(cand_val_ptr + base + j, m)
        tl.store(cand_idx_ptr + base + j, i)
        x = tl.where(offs == i, -float("inf"), x)


@triton.jit
def _merge_kernel(
    ids_ptr,
    logit_ptr,
    logp_ptr,
    cand_val_ptr,
    cand_idx_ptr,
    eim_ptr,
    temp_ptr,
    topk_ptr,
    topp_ptr,
    S: tl.constexpr,
    KP: tl.constexpr,
    NC: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    req = tl.load(eim_ptr + row)
    temp = tl.load(temp_ptr + req).to(tl.float32)
    k = tl.minimum(tl.load(topk_ptr + req), KP)
    k = tl.where(temp == 0.0, 1, k)
    p = tl.load(topp_ptr + req).to(tl.float32)

    f = tl.arange(0, NC)
    j = f % KP
    valid = (f < S * KP) & (j < k)
    v = tl.load(cand_val_ptr + row * S * KP + f, mask=valid, other=-float("inf"))
    ix = tl.load(cand_idx_ptr + row * S * KP + f, mask=valid, other=2147483647)

    kk = tl.arange(0, KP)
    acc_v = tl.full([KP], -float("inf"), tl.float32)
    acc_i = tl.zeros([KP], tl.int32)
    for jj in range(k):
        m = tl.max(v, axis=0)
        sel = tl.min(tl.where(v == m, ix, 2147483647), axis=0)
        sel = tl.where(sel == 2147483647, 0, sel)
        acc_v = tl.where(kk == jj, m, acc_v)
        acc_i = tl.where(kk == jj, sel, acc_i)
        v = tl.where(ix == sel, -float("inf"), v)

    valid_k = kk < k
    lv = tl.where(valid_k, acc_v, -float("inf"))
    mx = tl.max(lv, axis=0)
    mx = tl.where(mx > -float("inf"), mx, 0.0)
    e = tl.exp(lv - mx)
    prob = e / tl.sum(e, axis=0)
    cs_excl = tl.cumsum(prob, axis=0) - prob
    thr = tl.where(p >= 1.0, 2.0, p)  # p == 1: nucleus is the whole top-k
    keep = valid_k & ((cs_excl < thr) | (kk == 0))
    lm = tl.where(keep, lv, -float("inf"))
    lse = mx + tl.log(tl.sum(tl.where(keep, e, 0.0), axis=0))
    out = row * KP + kk
    tl.store(ids_ptr + out, acc_i)
    tl.store(logit_ptr + out, lm)
    tl.store(logp_ptr + out, lm - lse)


@triton.jit
def _draft_lse(dbase, cand_ptr, kd, kd_mask, temp):
    cand = tl.load(cand_ptr + kd, mask=kd_mask, other=0)
    dl = (
        tl.load(dbase + cand, mask=kd_mask, other=-float("inf")).to(tl.float32) / temp
    )
    dm = tl.max(dl, axis=0)
    return dm + tl.log(tl.sum(tl.exp(dl - dm), axis=0))


@triton.jit
def _fast_reject_kernel(
    sampled_ptr,
    sampled_stride,
    num_sampled_ptr,
    ids_ptr,
    logit_ptr,
    logp_ptr,
    draft_logits_ptr,
    draft_stride_0,
    draft_stride_1,
    cand_ids_ptr,
    cand_stride_0,
    cand_stride_1,
    draft_sampled_ptr,
    cu_num_logits_ptr,
    idx_mapping_ptr,
    temp_ptr,
    seed_ptr,
    pos_ptr,
    num_steps,
    KP: tl.constexpr,
    KD: tl.constexpr,
    KD_REAL: tl.constexpr,
):
    req_idx = tl.program_id(0)
    req_state = tl.load(idx_mapping_ptr + req_idx).to(tl.int64)
    start = tl.load(cu_num_logits_ptr + req_idx).to(tl.int64)
    end = tl.load(cu_num_logits_ptr + req_idx + 1).to(tl.int64)
    n = end - start - 1
    seed = tl.load(seed_ptr + req_state)
    temp = tl.load(temp_ptr + req_state).to(tl.float32)
    is_greedy = temp == 0.0
    kk = tl.arange(0, KP)
    kd = tl.arange(0, KD)
    kd_mask = kd < KD_REAL

    accepted_length = tl.zeros((), tl.int64)
    verifying = True
    for i in range(n):
        logit_idx = start + i
        d = tl.load(draft_sampled_ptr + logit_idx + 1).to(tl.int64)
        is_valid = d >= 0
        d = tl.maximum(0, d)
        if not is_greedy:
            verifying &= is_valid
        if verifying:
            pos = tl.load(pos_ptr + logit_idx)
            u = tl_rand32(seed, pos, includes_zero=False)
            row_ids = tl.load(ids_ptr + logit_idx * KP + kk).to(tl.int64)
            if is_greedy:
                top = tl.load(ids_ptr + logit_idx * KP).to(tl.int64)
                accepted = (top == d) & is_valid
                verifying = accepted
                accepted_length += accepted
                tl.store(
                    sampled_ptr + req_idx * sampled_stride + i,
                    tl.where(accepted, d, top),
                )
            else:
                lp = tl.load(logp_ptr + logit_idx * KP + kk)
                t_lp = tl.max(tl.where(row_ids == d, lp, -float("inf")), axis=0)
                dbase = draft_logits_ptr + req_state * draft_stride_0 + i * draft_stride_1
                dlse = _draft_lse(
                    dbase,
                    cand_ids_ptr + req_state * cand_stride_0 + i * cand_stride_1,
                    kd,
                    kd_mask,
                    temp,
                )
                d_lp = tl.load(dbase + d).to(tl.float32) / temp - dlse
                accepted = t_lp > tl.log(u) + d_lp
                verifying = accepted
                accepted_length += accepted
                tl.store(sampled_ptr + req_idx * sampled_stride + i, d)

    rtok = start + accepted_length
    is_bonus = rtok == end - 1
    if is_greedy and not is_bonus:
        # The target argmax is already in `sampled`.
        tl.store(num_sampled_ptr + req_idx, (accepted_length + 1).to(tl.int32))
        return
    rej = tl.load(draft_sampled_ptr + rtok + 1, mask=not is_bonus, other=0)
    use_target = is_bonus | (rej < 0) | is_greedy
    row_ids = tl.load(ids_ptr + rtok * KP + kk)
    lm = tl.load(logit_ptr + rtok * KP + kk)
    lp = tl.load(logp_ptr + rtok * KP + kk)
    step = tl.minimum(accepted_length, num_steps - 1)
    dbase = draft_logits_ptr + req_state * draft_stride_0 + step * draft_stride_1
    dlse = _draft_lse(
        dbase,
        cand_ids_ptr + req_state * cand_stride_0 + step * cand_stride_1,
        kd,
        kd_mask,
        temp,
    )
    dl = (
        tl.load(dbase + row_ids.to(tl.int64), mask=~use_target, other=-float("inf")).to(
            tl.float32
        )
        / temp
    )
    d_lp = dl - dlse
    ratio = tl.exp(d_lp - lp)
    resid = tl.where(ratio < 1.0, lp + tldevice.log1p(-ratio), -float("inf"))
    res = tl.where(use_target, lm, resid).to(tl.float32)
    pos = tl.load(pos_ptr + rtok)
    _, sel = gumbel_noised_argmax(
        res,
        row_ids,
        kk < KP,
        seed,
        pos,
        temp,
        IS_DRAFTING=False,
        USE_FP64=False,
        APPLY_TEMPERATURE=False,
    )
    token = tl.load(ids_ptr + rtok * KP + sel).to(tl.int64)
    tl.store(sampled_ptr + req_idx * sampled_stride + accepted_length, token)
    tl.store(num_sampled_ptr + req_idx, (accepted_length + 1).to(tl.int32))


@triton.jit
def _fast_sample_kernel(
    sampled_ptr,
    ids_ptr,
    logit_ptr,
    eim_ptr,
    temp_ptr,
    seed_ptr,
    pos_ptr,
    KP: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    req = tl.load(eim_ptr + row).to(tl.int64)
    temp = tl.load(temp_ptr + req).to(tl.float32)
    seed = tl.load(seed_ptr + req)
    pos = tl.load(pos_ptr + row)
    kk = tl.arange(0, KP)
    row_ids = tl.load(ids_ptr + row * KP + kk)
    lm = tl.load(logit_ptr + row * KP + kk)
    _, sel = gumbel_noised_argmax(
        lm,
        row_ids,
        kk < KP,
        seed,
        pos,
        temp,
        IS_DRAFTING=False,
        USE_FP64=False,
        APPLY_TEMPERATURE=False,
    )
    tl.store(sampled_ptr + row, tl.load(ids_ptr + row * KP + sel).to(tl.int64))


def _kp(kmax: int) -> int:
    # Two compiled shapes only (16 and 32), so no mid-serving JIT for a new top-k.
    return 16 if kmax <= 16 else 32


def batch_params(
    states, idx_mapping_np: np.ndarray, vocab_size: int
) -> int | None:
    """Host-side eligibility on the sampling params; returns the batch's max top-k
    (>= 1) or None when the stock path must run. No device access."""
    temp = states.temperature.np[idx_mapping_np]
    if np.any(states.min_p.np[idx_mapping_np] != 0.0):
        return None
    sampled = temp != 0.0
    if not sampled.any():
        return 1
    k = states.top_k.np[idx_mapping_np][sampled]
    kmax = int(k.max())
    if kmax > KMAX or kmax >= vocab_size:
        return None
    return kmax


def topk_support(
    logits: torch.Tensor,  # [N, V] any float dtype, stride(-1) == 1
    expanded_idx_mapping: torch.Tensor,
    sampling_states,
    kmax: int,
    vocab_size: int | None = None,
    block: int = BLOCK,
    num_warps: int = 4,
):
    """Kernels A and B. Returns (ids int32, masked scaled logits fp32, log-probs
    fp32), each [N, KP]; entries past a row's k, or dropped by top-p, are -inf."""
    n, v_full = logits.shape
    v = v_full if vocab_size is None else min(v_full, vocab_size)
    kp = _kp(kmax)
    s = cdiv(v, block)
    dev = logits.device
    cand_val = torch.empty(n, s, kp, dtype=torch.float32, device=dev)
    cand_idx = torch.empty(n, s, kp, dtype=torch.int32, device=dev)
    ids = torch.empty(n, kp, dtype=torch.int32, device=dev)
    lmask = torch.empty(n, kp, dtype=torch.float32, device=dev)
    logp = torch.empty(n, kp, dtype=torch.float32, device=dev)
    _split_topk_kernel[(n, s)](
        cand_val,
        cand_idx,
        logits,
        logits.stride(0),
        expanded_idx_mapping,
        sampling_states.temperature.gpu,
        sampling_states.top_k.gpu,
        v,
        KP=kp,
        BLOCK_SIZE=block,
        S=s,
        num_warps=num_warps,
    )
    _merge_kernel[(n,)](
        ids,
        lmask,
        logp,
        cand_val,
        cand_idx,
        expanded_idx_mapping,
        sampling_states.temperature.gpu,
        sampling_states.top_k.gpu,
        sampling_states.top_p.gpu,
        S=s,
        KP=kp,
        NC=next_power_of_2(s * kp),
        num_warps=num_warps,
    )
    return ids, lmask, logp


def fast_reject(
    logits: torch.Tensor,
    draft_logits: torch.Tensor,
    draft_candidate_ids: torch.Tensor,
    draft_sampled: torch.Tensor,
    pos: torch.Tensor,
    cu_num_logits: torch.Tensor,
    idx_mapping: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    sampling_states,
    num_speculative_steps: int,
    kmax: int,
    block: int = BLOCK,
):
    """Drop-in for rejection_sample(): returns (sampled int64 [R, K+1],
    num_sampled int32 [R])."""
    num_reqs = cu_num_logits.shape[0] - 1
    ids, lmask, logp = topk_support(
        logits,
        expanded_idx_mapping,
        sampling_states,
        kmax,
        vocab_size=draft_logits.shape[-1],
        block=block,
    )
    kp = ids.shape[1]
    sampled = draft_sampled.new_empty(
        num_reqs, num_speculative_steps + 1, dtype=torch.int64
    )
    num_sampled = sampled.new_empty(num_reqs, dtype=torch.int32)
    kd_real = draft_candidate_ids.shape[-1]
    _fast_reject_kernel[(num_reqs,)](
        sampled,
        sampled.stride(0),
        num_sampled,
        ids,
        lmask,
        logp,
        draft_logits,
        draft_logits.stride(0),
        draft_logits.stride(1),
        draft_candidate_ids,
        draft_candidate_ids.stride(0),
        draft_candidate_ids.stride(1),
        draft_sampled,
        cu_num_logits,
        idx_mapping,
        sampling_states.temperature.gpu,
        sampling_states.seeds.gpu,
        pos,
        draft_logits.shape[1],
        KP=kp,
        KD=max(2, next_power_of_2(kd_real)),
        KD_REAL=kd_real,
        num_warps=1,
    )
    return sampled, num_sampled


def fast_sample(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    sampling_states,
    pos: torch.Tensor,
    kmax: int,
    block: int = BLOCK,
) -> torch.Tensor:
    """Plain (non-speculative) sampling: one token per row, int64 [N]."""
    ids, lmask, _ = topk_support(
        logits, expanded_idx_mapping, sampling_states, kmax, block=block
    )
    sampled = torch.empty(logits.shape[0], dtype=torch.int64, device=logits.device)
    _fast_sample_kernel[(logits.shape[0],)](
        sampled,
        ids,
        lmask,
        expanded_idx_mapping,
        sampling_states.temperature.gpu,
        sampling_states.seeds.gpu,
        pos,
        KP=ids.shape[1],
        num_warps=1,
    )
    return sampled


def warmup(device, vocab_size: int, num_steps: int, num_candidates: int) -> None:
    """JIT-compile every kernel shape the fast path can take (KP in {16, 32}) on
    dedicated initialised tensors, so no first request pays for compilation."""
    from types import SimpleNamespace

    def t(x):
        return SimpleNamespace(gpu=x)

    r = 1
    n = r * (num_steps + 1)
    logits = (
        torch.arange(vocab_size, device=device, dtype=torch.float32)
        .unsqueeze(0)
        .repeat(n, 1)
        / vocab_size
    ).to(torch.bfloat16)
    states = SimpleNamespace(
        temperature=t(torch.full((r,), 0.7, device=device)),
        top_k=t(torch.full((r,), 20, dtype=torch.int32, device=device)),
        top_p=t(torch.full((r,), 0.9, device=device)),
        seeds=t(torch.ones(r, dtype=torch.int64, device=device)),
    )
    eim = torch.zeros(n, dtype=torch.int32, device=device)
    pos = torch.arange(n, dtype=torch.int64, device=device)
    draft = torch.full((r, num_steps, vocab_size), -float("inf"), device=device)
    cand = torch.zeros(r, num_steps, num_candidates, dtype=torch.int64, device=device)
    ds = torch.full((n,), -1, dtype=torch.int64, device=device)
    cu = torch.tensor([0, n], dtype=torch.int32, device=device)
    idx = torch.zeros(r, dtype=torch.int32, device=device)
    for kmax in (16, KMAX):
        states.top_k.gpu.fill_(kmax)
        fast_reject(logits, draft, cand, ds, pos, cu, idx, eim, states, num_steps, kmax)
        fast_sample(logits, eim, states, pos, kmax)
    if torch.device(device).type != "cpu":
        torch.accelerator.synchronize()
