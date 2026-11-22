# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 XPU block AttnRes kernel (fused residual add and RMSNorm) vs reference."""

import os

import pytest
import torch

from vllm.models.inkling2.xpu.ops import ref
from vllm.models.inkling2.xpu.ops.triton_attn_res import attn_res

DEV = os.environ.get("INKLING2_TEST_DEVICE", "cpu")


@pytest.mark.parametrize("n,q,tokens,addend,norm", [(0, 1, 5, False, True), (3, 3, 7, True, True),
                                                    (4, 1, 300, True, False), (1, 4, 3, False, False)])
def test_attn_res(n, q, tokens, addend, norm):
    g = torch.Generator().manual_seed(n * 10 + q)
    D = 2304
    stack = torch.randn(5, tokens, D, generator=g).bfloat16()
    last = torch.randn(tokens, D, generator=g).bfloat16()
    add = torch.randn(tokens, D, generator=g).bfloat16() if addend else None
    queries = torch.randn(q, D, generator=g) * D ** -0.5
    w = (1 + 0.1 * torch.randn(q, D, generator=g)).bfloat16() if norm else None
    partial = last.float() if add is None else last.float() + add.float()
    want = ref.attn_res(stack[:n], partial, queries)
    if norm:
        want = torch.stack([ref.rms_norm(want[i], w[i], 1e-6) for i in range(q)])
    d = lambda t: None if t is None else t.to(DEV)
    got = attn_res(d(stack)[:n], d(last), d(add), d(queries), d(w), 1e-6, True).cpu()
    assert got.dtype == want.dtype
    err = (got.float() - want.float()).abs().max().item()
    assert err < (1.6e-2 if norm else 1e-4), err
