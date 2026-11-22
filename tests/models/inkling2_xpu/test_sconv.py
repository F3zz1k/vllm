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


@pytest.mark.parametrize("mode", ["static", "dynamic_k", "dynamic_v", "attn", "resid"])
def test_short_conv_steps(mode):
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
            assert (stored.float() - got.view(T, H, hd).float()).abs().max() == 0
        assert torch.equal(st_tri.cpu(), st_ref), step
