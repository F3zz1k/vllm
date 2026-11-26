# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 XPU short-conv kernel vs the torch reference over several steps
(prefill chunks longer and shorter than the kernel, then decode, a fresh
request and one restored from a state slot), checking outputs, the KV store
and the per-sequence conv state."""

import os

import pytest
import torch

from vllm.models.inkling2.xpu.ops import ref
from vllm.models.inkling2.xpu.ops.triton_sconv import short_conv

DEV = os.environ.get("INKLING2_TEST_DEVICE", "cpu")


@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("mode", ["static", "dynamic_k", "dynamic_v", "attn", "resid"])
def test_short_conv_steps(mode, fp8):
    if fp8 and mode not in ("dynamic_k", "dynamic_v"):
        pytest.skip("fp8 KV: only the K / V streams store")
    g = torch.Generator().manual_seed(1)
    H, hd, hidden, k = 2, 32, 128, 4
    layout = ref.ConvLayout.make(H, hd, hidden)
    stream = {"static": "k", "dynamic_k": "k", "dynamic_v": "v", "attn": "attn",
              "resid": "mlp"}[mode]
    off, w = layout.stream(stream)
    C = H * w
    nslots = 6
    st_ref = (torch.randn(nslots, k - 1, H * layout.width, generator=g)).bfloat16()
    st_tri = st_ref.clone().to(DEV)
    kv_ref = torch.zeros(20, H, 16, 2 * hd).bfloat16()
    kv_tri = kv_ref.clone().to(DEV)
    if fp8:  # the model's view of vLLM's uint8 fp8 pages
        kv_tri = torch.zeros(kv_ref.shape, dtype=torch.uint8, device=DEV).view(
            torch.float8_e4m3fn)
    dyn_mode = mode.startswith("dynamic")
    weight = (torch.randn((k, C) if dyn_mode else (C, k), generator=g) * 0.4)
    if dyn_mode:
        weight = weight.bfloat16()
    norm = {"dynamic_k": 1, "dynamic_v": 2}.get(mode, 0)
    # (state slot, has history, tokens) per request per step
    steps = [[(1, 0, 5), (2, 0, 2)], [(1, 1, 1), (2, 1, 2)], [(1, 1, 1), (2, 1, 1), (3, 1, 9)],
             [(4, 0, 1), (1, 1, 6), (3, 1, 1)], [(4, 1, 2), (2, 1, 1)]]
    kv_next = 0
    for step in steps:
        qsl = [0]
        for _, _, n in step:
            qsl.append(qsl[-1] + n)
        T = qsl[-1]
        seq_idx = torch.cat([torch.full((n,), r) for r, (_, _, n) in enumerate(step)]).int()
        qs = torch.tensor([qsl[r] for r in seq_idx.tolist()]).int()
        sidx = [s for s, _, _ in step]
        hinit = [bool(h) for _, h, _ in step]
        kvs = torch.arange(kv_next, kv_next + T) % 320
        kv_next += T
        x = torch.randn(T, C, generator=g).bfloat16()
        dyn = (torch.randn(T, H * k, generator=g) * 0.3).bfloat16() if dyn_mode else None
        last = torch.randn(T, C, generator=g).bfloat16() if mode == "resid" else None
        hist = ref.conv_history_state(st_ref, layout, stream, x, qsl, sidx, hinit, k)
        if dyn_mode:
            want = ref.dynamic_conv(hist, weight, dyn, H).to(torch.bfloat16)
        else:
            want = ref.static_conv(hist, weight).to(torch.bfloat16)
        if norm == 1:
            want = ref.l2_clip(want.view(T, H, hd).float(), 22.6).to(torch.bfloat16).view(T, C)
        elif norm == 2:
            want = ref.unweighted_rms(want.view(T, H, hd)).to(torch.bfloat16).view(T, C)
        if last is not None:
            want = (last.float() + want.float()).bfloat16()
        d = lambda t: t.to(DEV)
        i32 = lambda v: torch.tensor(v, dtype=torch.int32, device=DEV)
        # slot via the block table (column 0); history when seq_len > rows in the step
        bt = i32([[s, 0] for s in sidx])
        lens = i32([n + (7 if h else 0) for (_, h, n) in step])
        got = short_conv(d(x), st_tri, off, w, layout.width, d(seq_idx), d(qs), i32(qsl),
                         bt, lens, d(weight),
                         dyn=None if dyn is None else d(dyn), norm=norm, clip=22.6,
                         kv=kv_tri if norm else None, kv_slots=d(kvs) if norm else None,
                         kv_off=0 if mode == "dynamic_k" else hd,
                         last=None if last is None else d(last)).cpu()
        err = (got.float() - want.float()).abs().max().item()
        assert err <= 1.6e-2, (step, err)
        if norm:
            o = 0 if mode == "dynamic_k" else hd
            stored = kv_tri.cpu()[kvs // 16, :, kvs % 16, o:o + hd]
            want_kv = got.view(T, H, hd).float()
            if fp8:  # one e4m3 rounding of the stored value (3 mantissa bits)
                diff = (stored.float() - want_kv).abs()
                assert (diff <= want_kv.abs() * 2.0**-4 + 2.0**-9).all(), diff.max()
            else:
                assert (stored.float() - want_kv).abs().max() == 0
        assert torch.equal(st_tri.cpu(), st_ref), step


@pytest.mark.parametrize("mode", ["attn", "dynamic_k"])
def test_short_conv_spec(mode):
    """Speculative decode: state of k - 1 + S rows; verify steps of 1 + S rows read from
    row num_accepted - 1. Reference: the conv over the truly accepted history."""
    g = torch.Generator().manual_seed(2)
    H, hd, hidden, k, S = 2, 32, 128, 4, 3
    layout = ref.ConvLayout.make(H, hd, hidden)
    stream = "attn" if mode == "attn" else "k"
    off, w = layout.stream(stream)
    C = H * w
    dyn_mode = mode.startswith("dynamic")
    weight = torch.randn((k, C) if dyn_mode else (C, k), generator=g) * 0.4
    if dyn_mode:
        weight = weight.bfloat16()
    st = torch.zeros(4, k - 1 + S, H * layout.width).bfloat16().to(DEV)
    hist = {0: torch.zeros(0, C), 1: torch.zeros(0, C)}  # truly accepted inputs per request
    acc = {0: 1, 1: 1}
    d = lambda t: t.to(DEV)
    i32 = lambda v: torch.tensor(v, dtype=torch.int32, device=DEV)
    # per step: (req, rows); decode rows (1 + S) first, as vLLM orders them
    steps = [[(0, 6)], [(0, 4), (1, 9)], [(0, 4), (1, 4)], [(0, 4), (1, 4)], [(1, 4), (0, 2)]]
    for step in steps:
        qsl = [0]
        for _, n in step:
            qsl.append(qsl[-1] + n)
        T = qsl[-1]
        x = torch.randn(T, C, generator=g).bfloat16()
        dyn = (torch.randn(T, H * k, generator=g) * 0.3).bfloat16() if dyn_mode else None
        want = []
        for r, (req, n) in enumerate(step):
            prior = hist[req][-(k - 1):].float()
            prior = torch.cat([torch.zeros(k - 1 - prior.shape[0], C), prior])
            seq = torch.cat([prior, x[qsl[r]:qsl[r + 1]].float()])
            hh = torch.stack([seq[i:i + k] for i in range(n)])
            if dyn_mode:
                want.append(ref.dynamic_conv(hh, weight, dyn[qsl[r]:qsl[r + 1]], H))
            else:
                want.append(ref.static_conv(hh, weight))
        want = torch.cat(want).bfloat16()
        if dyn_mode:
            want = ref.l2_clip(want.view(T, H, hd).float(), 22.6).to(torch.bfloat16).view(T, C)
        seq_idx = torch.cat([torch.full((n,), r) for r, (_, n) in enumerate(step)]).int()
        qs = torch.tensor([qsl[r] for r in seq_idx.tolist()]).int()
        lens = [hist[req].shape[0] + n for req, n in step]
        nacc = [acc[req] for req, _ in step]
        got = short_conv(d(x), st, off, w, layout.width, d(seq_idx), d(qs), i32(qsl),
                         i32([[req] for req, _ in step]), i32(lens), d(weight),
                         dyn=None if dyn is None else d(dyn), norm=1 if dyn_mode else 0,
                         clip=22.6, num_accepted=i32(nacc), spec=S).cpu()
        err = (got.float() - want.float()).abs().max().item()
        assert err <= 1.6e-2, (step, err)
        for r, (req, n) in enumerate(step):  # accept a random prefix of each verify row
            xs = x[qsl[r]:qsl[r + 1]].float()
            is_dec = hist[req].shape[0] > 0 and n <= 1 + S
            a = int(torch.randint(1, n + 1, (1,), generator=g)) if is_dec else n
            hist[req] = torch.cat([hist[req], xs[:a]])
            acc[req] = a if is_dec else 1
