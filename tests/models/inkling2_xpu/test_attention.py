# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 XPU relative attention kernel vs the torch reference.
Runs on CPU with TRITON_INTERPRET=1 or on an XPU device."""

import os

import pytest
import torch

from vllm.models.inkling2.xpu.ops import ref
from vllm.models.inkling2.xpu.ops.triton_attention import rel_attention

DEV = os.environ.get("INKLING2_TEST_DEVICE", "cpu")


def _case(qlens, ctxs, window, E, Hq=6, Hk=2, D=32, page=16, seed=0):
    g = torch.Generator().manual_seed(seed)
    n = len(qlens)
    seqs = [c + q for c, q in zip(ctxs, qlens)]
    per = max((s + page - 1) // page for s in seqs)
    nb = n * per + 3
    kv = (torch.randn(nb, Hk, page, 2 * D, generator=g)).bfloat16()
    perm = torch.randperm(nb - 1, generator=g)[: n * per] + 1
    bt = perm.view(n, per).int()
    T = sum(qlens)
    q = torch.randn(T, Hq, D, generator=g).bfloat16()
    rel = (torch.randn(T, Hq, E, generator=g) * 2).bfloat16()
    sinks_full = (torch.randn(T, Hq + 5, generator=g) * 2).bfloat16()
    sinks = sinks_full[:, 2:2 + Hq]  # strided view like the projection slice
    cu = torch.tensor([0] + list(torch.tensor(qlens).cumsum(0)), dtype=torch.int32)
    sl = torch.tensor(seqs, dtype=torch.int32)
    return q, kv, rel, sinks, bt, sl, cu


@pytest.mark.parametrize("rev", [False, True])
@pytest.mark.parametrize("window", [None, 24])
@pytest.mark.parametrize("splits", [1, 4])
@pytest.mark.parametrize("qlens,ctxs", [([37], [0]), ([1, 1, 1], [5, 70, 0]),
                                        ([9, 1, 20], [40, 3, 17])])
def test_rel_attention(rev, window, splits, qlens, ctxs):
    E = 32
    q, kv, rel, sinks, bt, sl, cu = _case(qlens, ctxs, window, E)
    want = ref.rel_attention(q, rel, sinks, kv, bt, sl, cu, 32 ** -0.5, window, E)
    rel_in = rel.flip(-1).contiguous() if rev else rel
    args = [x.to(DEV) for x in (q, kv, rel_in, sinks, bt, sl, cu)]
    got = rel_attention(*args, 32 ** -0.5, window, E, len(qlens), num_splits=splits,
                        block_n=16, rel_reversed=rev).cpu()
    err = (got.float() - want.float()).abs().max().item()
    assert err < 2e-2, err
