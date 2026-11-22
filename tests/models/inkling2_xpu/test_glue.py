# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused glue kernels (ops/triton_glue.py, the short-conv residual) vs the torch
ops they replace. XPU only."""

import os
from types import SimpleNamespace

import pytest
import torch

from vllm.models.inkling2.xpu import ops
from vllm.models.inkling2.xpu.ops import ref

DEV = os.environ.get("INKLING2_TEST_DEVICE", "cpu")
H, D, R, E, DM = 24, 128, 16, 1024, 2304


def _skip():
    if DEV == "cpu":
        pytest.skip("XPU only")


def _rel(a, b):
    return ((a.float() - b.float()).norm() / b.float().norm().clamp(min=1e-12)).item()


@pytest.mark.parametrize("T", [1, 7, 300])
@pytest.mark.parametrize("tau", [False, True])
def test_qkr(T, tau):
    _skip()
    from vllm.models.inkling2.xpu.ops.triton_glue import qkr_prep

    g = torch.Generator(device="cpu").manual_seed(1)
    y = (torch.randn(T, H * D + H * R + 40, generator=g) * 3).bfloat16().to(DEV)
    qs = (torch.rand(H, D, generator=g) + 0.5).bfloat16().to(DEV)
    rs = (torch.rand(H, R, generator=g) + 0.5).bfloat16().to(DEV)
    proj = torch.randn(R, E, generator=g).bfloat16().to(DEV)
    t = (1 + torch.rand(T, generator=g)).to(DEV) if tau else None
    q, rel = qkr_prep(y, qs, rs, t, proj, H, D, R, H * D, 22.627, 8.0)
    q0 = ops.qr_norm(y[:, :H * D].view(T, H, D), qs, 22.627, t)
    r0 = ops.qr_norm(y[:, H * D:H * D + H * R].view(T, H, R), rs, 8.0, t)
    assert _rel(q, q0) < 1e-2
    assert _rel(rel, torch.matmul(r0, proj)) < 1e-2


@pytest.mark.parametrize("T,n", [(1, 1), (8, 5), (300, 300)])
def test_attn_post(T, n):
    _skip()
    from vllm.models.inkling2.xpu.ops.triton_glue import attn_post

    o = torch.randn(T, H, D, device=DEV).bfloat16()
    lse = torch.randn(H, T, device=DEV)
    y = torch.randn(T, 64 + H * D, device=DEV).bfloat16()
    sinks, gate = y[:, :H], y[:, 64:]
    out = torch.empty(T, H * D, device=DEV, dtype=torch.bfloat16)
    attn_post(o, lse, sinks, gate, out, n)
    w = (o.float() * torch.sigmoid(lse.t() - sinks.float())[..., None]).bfloat16()
    want = (w.float().view(T, -1) * torch.sigmoid(gate.float())).bfloat16()
    assert _rel(out[:n], want[:n]) < 1e-2
    assert (out[n:] == 0).all()


@pytest.mark.parametrize("T", [1, 16, 200])
def test_route(T):
    _skip()
    from vllm.models.inkling2.xpu.ops.triton_glue import route

    x = torch.randn(T, DM, device=DEV).bfloat16()
    w = torch.randn(128, DM, device=DEV) * 0.02
    b = torch.randn(128, device=DEV) * 0.01
    wt, ids = route(x, w, b, 8, 8.0)
    wr, idr = ref.route(x, w, b, 8, 8.0)
    assert (ids.sort(1).values == idr.sort(1).values).all()
    o, orr = ids.long().argsort(1), idr.long().argsort(1)
    assert (wt.gather(1, o) - wr.gather(1, orr)).abs().max().item() < 1e-5


@pytest.mark.parametrize("S,T", [(3, 1), (1, 9), (3, 50)])
def test_blend(S, T):
    _skip()
    r = 64
    layer = SimpleNamespace(ctrl_down=(torch.randn(4, r, 2 * DM, device=DEV) * 0.02).bfloat16(),
                            ctrl_gf=torch.randn(4, 2 * DM, r, device=DEV) * 0.05)
    dests = torch.randn(S, T, DM, device=DEV)
    msgs = torch.randn(4, T, DM, device=DEV)[:S]
    w = torch.rand(S, DM, device=DEV).bfloat16() + 0.5
    got = ops.blend(layer, slice(0, S), dests, msgs, w, 1e-6)
    rms = ref.unweighted_rms
    cat = torch.cat([rms(dests), rms(msgs)], -1).bfloat16()
    h = torch.bmm(cat, layer.ctrl_down[:S].transpose(1, 2)).float()
    gf = torch.bmm(h, layer.ctrl_gf[:S].transpose(1, 2))
    x = (1 - torch.tanh(gf[..., DM:])) * dests + torch.tanh(gf[..., :DM]) * msgs
    want = ref.rms_norm(x, w[:, None, :], 1e-6)
    assert _rel(got, want) < 1e-2
