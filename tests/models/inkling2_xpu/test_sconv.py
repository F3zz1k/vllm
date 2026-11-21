# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 XPU short-conv kernel vs the torch reference, over several steps
(prefill chunks then decode) so the conv-state path is exercised."""

import os

import pytest
import torch

from vllm.models.inkling2.xpu.ops import ref
from vllm.models.inkling2.xpu.ops.triton_sconv import short_conv

DEV = os.environ.get("INKLING2_TEST_DEVICE", "cpu")


@pytest.mark.parametrize("mode", ["static", "dynamic_k", "dynamic_v", "attn"])
def test_short_conv_steps(mode):
    g = torch.Generator().manual_seed(1)
    H, hd, hidden, k, page = 2, 32, 128, 4, 4
    layout = ref.ConvLayout.make(H, hd, hidden)
    stream = {"static": "k", "dynamic_k": "k", "dynamic_v": "v", "attn": "attn"}[mode]
    off, w = layout.stream(stream)
    C = H * w
    nreq, per = 2, 16
    nb = nreq * per + 1
    conv_ref = torch.zeros(nb, H, page, layout.width).bfloat16()
    conv_tri = conv_ref.clone().to(DEV)
    kv_ref = torch.zeros(20, H, 16, 2 * hd).bfloat16()
    kv_tri = kv_ref.clone().to(DEV)
    bt = (torch.arange(nreq * per) + 1).view(nreq, per).int()
    kv_bt = torch.arange(20).view(2, 10)
    dyn_mode = mode.startswith("dynamic")
    weight = (torch.randn((k, C) if dyn_mode else (C, k), generator=g) * 0.4)
    if dyn_mode:
        weight = weight.bfloat16()
    norm = {"dynamic_k": 1, "dynamic_v": 2}.get(mode, 0)
    steps = [[5, 3], [1, 2], [1, 1], [6, 1]]  # tokens per request per step
    done = [0, 0]
    for step in steps:
        pos, sidx, qs, x, slots, kvs = [], [], [], [], [], []
        t0 = 0
        for r, n in enumerate(step):
            p = torch.arange(done[r], done[r] + n)
            pos.append(p); sidx.append(torch.full((n,), r)); qs.append(torch.full((n,), t0))
            slots.append(bt[r, p // page].long() * page + p % page)
            kvs.append(kv_bt[r, p // 16] * 16 + p % 16)
            t0 += n
            done[r] += n
        pos, sidx, qs = torch.cat(pos), torch.cat(sidx).int(), torch.cat(qs).int()
        slots, kvs = torch.cat(slots), torch.cat(kvs)
        T = pos.numel()
        x = torch.randn(T, C, generator=g).bfloat16()
        dyn = (torch.randn(T, H * k, generator=g) * 0.3).bfloat16() if dyn_mode else None
        ref.conv_write(conv_ref, layout, stream, x, slots)
        hist = ref.conv_history(conv_ref, layout, stream, pos, bt, sidx.long(), k)
        if dyn_mode:
            want = ref.dynamic_conv(hist, weight, dyn, H).to(torch.bfloat16)
        else:
            want = ref.static_conv(hist, weight).to(torch.bfloat16)
        if norm == 1:
            want = ref.l2_clip(want.view(T, H, hd).float(), 22.6).to(torch.bfloat16).view(T, C)
        elif norm == 2:
            want = ref.unweighted_rms(want.view(T, H, hd)).to(torch.bfloat16).view(T, C)
        d = lambda t: t.to(DEV)
        got = short_conv(d(x), conv_tri, off, w, d(slots), d(bt), d(sidx), d(qs), d(pos),
                         d(weight), dyn=None if dyn is None else d(dyn), norm=norm, clip=22.6,
                         kv=kv_tri if norm else None, kv_slots=d(kvs) if norm else None,
                         kv_off=0 if mode == "dynamic_k" else hd).cpu()
        err = (got.float() - want.float()).abs().max().item()
        assert err <= 1.6e-2, (step, err)
        if norm:
            o = 0 if mode == "dynamic_k" else hd
            stored = kv_tri.cpu()[kvs // 16, :, kvs % 16, o:o + hd]
            assert (stored.float() - got.view(T, H, hd).float()).abs().max() == 0
    assert (conv_tri.cpu()[..., off:off + w].float() - conv_ref[..., off:off + w].float()).abs().max() == 0
