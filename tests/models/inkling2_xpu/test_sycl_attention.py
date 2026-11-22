# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 SYCL relative-bias attention vs the torch reference (XPU only)."""

import os

import pytest
import torch

from vllm.models.inkling2.xpu.ops import ref

DEV = os.environ.get("INKLING2_TEST_DEVICE", "cpu")


def _case(qlens, ctxs, E, Hq=24, Hk=8, D=128, page=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = len(qlens)
    seqs = [c + q for c, q in zip(ctxs, qlens)]
    per = max((s + page - 1) // page for s in seqs)
    nb = n * per + 3
    kv = (torch.randn(nb, Hk, page, 2 * D, generator=g) * 0.5).bfloat16()
    perm = torch.randperm(nb - 1, generator=g)[: n * per] + 1
    bt = perm.view(n, per).int()
    T = sum(qlens)
    q = torch.randn(T, Hq, D, generator=g).bfloat16()
    rel = (torch.randn(T, Hq, E, generator=g) * 2).bfloat16()
    sinks = (torch.randn(T, Hq, generator=g) * 2).bfloat16()
    cu = torch.tensor([0] + list(torch.tensor(qlens).cumsum(0)), dtype=torch.int32)
    sl = torch.tensor(seqs, dtype=torch.int32)
    return q, kv, rel, sinks, bt, sl, cu


@pytest.mark.parametrize("window", [None, 100])
@pytest.mark.parametrize("page", [16, 64])
@pytest.mark.parametrize("qlens,ctxs,splits", [([37], [0], 1), ([1, 1, 1], [5, 300, 0], 1),
                                               ([1, 1], [700, 40], 4), ([9, 1, 200], [40, 3, 170], 1)])
def test_sycl_rel_attention(window, page, qlens, ctxs, splits):
    if DEV == "cpu":
        pytest.skip("XPU only")
    from vllm.models.inkling2.xpu.ops import sycl_attention as S
    assert S.available()
    E = 128
    q, kv, rel, sinks, bt, sl, cu = _case(qlens, ctxs, E, page=page)
    want = ref.rel_attention(q, rel, sinks, kv, bt, sl, cu, 128 ** -0.5, window, E)
    d = lambda t: t.to(DEV)
    got = S.rel_attention(d(q), d(kv), d(rel), d(sinks), d(bt), d(sl), d(cu), 128 ** -0.5,
                          window, max(qlens), int(sl.max()), num_splits=splits).cpu()
    err = (got.float() - want.float()).abs().max().item()
    assert err < 3e-2, err
