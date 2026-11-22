# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 experts on XpuFusedMoe (situ, layouts, fp8 scales) vs the torch
reference, at Inkling 2 Mini expert shapes. XPU only."""

import os
from types import SimpleNamespace

import pytest
import torch

from vllm.models.inkling2.xpu import ops
from vllm.models.inkling2.xpu.experts import ExpertStore
from vllm.models.inkling2.xpu.ops import ref

DEV = os.environ.get("INKLING2_TEST_DEVICE", "cpu")


@pytest.mark.parametrize("fmt", ["bf16", "fp8", "nvfp4/fp8", "nvfp4/bf16", "nvfp4/triton"])
@pytest.mark.parametrize("tokens", [1, 7, 64, 300])
def test_experts_kernel(fmt, tokens):
    if DEV == "cpu" and fmt != "nvfp4":
        pytest.skip("XPU only")
    g = torch.Generator().manual_seed(2)
    E, H, I, K = 128, 2304, 1152, 8
    # random NVFP4-format checkpoint tensors (interleaved gate/up rows)
    parts = {}
    for n, (r, c) in (("w13", (2 * I, H)), ("w2", (H, I))):
        parts[f"{n}_weight"] = torch.randint(0, 256, (E, r, c // 2), generator=g, dtype=torch.int32).to(torch.uint8)
        parts[f"{n}_weight_scale"] = (torch.rand(E, r, c // 16, generator=g) * 2 + 0.25).to(torch.float8_e4m3fn)
        parts[f"{n}_weight_amax"] = torch.rand(E, generator=g) * 0.5 + 0.5
        parts[f"{n}_weight_outer_scale_denominator"] = torch.tensor([2688.0 / 3])
    fmt, _, compute = fmt.partition("/")
    os.environ["VLLM_INKLING2_NVFP4_COMPUTE"] = compute or "auto"
    store = ExpertStore(fmt)
    store.build(parts, DEV)
    moe = SimpleNamespace(store=store, top_k=K, num_experts=E, gate_cap=4.0, up_cap=25.0,
                          expert_weights=store.expert_weights)
    x = (torch.randn(tokens, H, generator=g) * 0.5).bfloat16().to(DEV)
    gw = (torch.randn(E, H, generator=g) * 0.05).to(DEV)
    w, ids = ref.route(x, gw, torch.zeros(E, device=DEV), K, 8.0)
    got = ops.experts(moe, x, w, ids)
    want = ref.moe_experts(x, w, ids, store.expert_weights, E, 4.0, 25.0)
    rel = ((got.float() - want.float()).norm() / want.float().norm()).item()
    # the fp8 scratch carries the per-expert fp8 re-encode (3% on random codes,
    # 1.4% measured on the checkpoint's NVFP4 layers)
    assert rel < (8e-2 if compute == "fp8" else 2e-2), rel
