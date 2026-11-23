# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 experts (XpuFusedMoe for bf16/fp8, the NVFP4 grouped GEMM for nvfp4)
vs the torch reference, at Inkling 2 Mini expert shapes. XPU only."""

import os
from types import SimpleNamespace

import pytest
import torch

from vllm.models.inkling2.xpu import ops
from vllm.models.inkling2.xpu.experts import ExpertStore
from vllm.models.inkling2.xpu.ops import ref

DEV = os.environ.get("INKLING2_TEST_DEVICE", "cpu")


def _moe(store, top_k, num_experts):
    """The attributes of Inkling2MoE that ops.experts reads (single rank, no prefill store)."""
    return SimpleNamespace(store=store, top_k=top_k, num_experts=num_experts, gate_cap=4.0,
                           up_cap=25.0, expert_weights=store.expert_weights, ep_size=1,
                           ep_rank=0, expert_map=None, prefill_store=None)


@pytest.mark.parametrize("fmt", ["bf16", "fp8", "nvfp4"])
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
        sc = torch.rand(E, r, c // 16, generator=g) * 2 + 0.25
        # a quarter of the blocks get E4M3 subnormal scales (< 2^-6), as real checkpoints have
        sc = torch.where(torch.rand(sc.shape, generator=g) < 0.25, sc * 4e-3, sc)
        parts[f"{n}_weight_scale"] = sc.to(torch.float8_e4m3fn)
        parts[f"{n}_weight_amax"] = torch.rand(E, generator=g) * 0.5 + 0.5
        parts[f"{n}_weight_outer_scale_denominator"] = torch.tensor([2688.0 / 3])
    store = ExpertStore(fmt)
    store.build(parts, DEV)
    moe = _moe(store, K, E)
    x = (torch.randn(tokens, H, generator=g) * 0.5).bfloat16().to(DEV)
    gw = (torch.randn(E, H, generator=g) * 0.05).to(DEV)
    w, ids = ref.route(x, gw, torch.zeros(E, device=DEV), K, 8.0)
    got = ops.experts(moe, x, w, ids)
    want = ref.moe_experts(x, w, ids, store.expert_weights, E, 4.0, 25.0)
    rel = ((got.float() - want.float()).norm() / want.float().norm()).item()
    assert rel < 2e-2, rel


@pytest.mark.parametrize("mx", ["bf16", "fp8", "mxfp8"])
@pytest.mark.parametrize("tokens", [1, 7, 64, 300])
def test_experts_mxfp8_layers(mx, tokens, monkeypatch):
    """Layers 20-23 (MXFP8 checkpoint bytes) under each VLLM_INKLING2_MXFP8 store,
    against weights decoded straight from the checkpoint parts (not the store)."""
    if DEV == "cpu":
        pytest.skip("XPU only")
    monkeypatch.setenv("VLLM_INKLING2_MXFP8", mx)
    g = torch.Generator().manual_seed(3)
    E, H, I, K = 128, 2304, 1152, 8
    parts = {}
    for n, (r, c) in (("w13", (2 * I, H)), ("w2", (H, I))):
        parts[f"{n}_weight"] = (torch.randn(E, r, c, generator=g) * 64).to(torch.float8_e4m3fn)
        parts[f"{n}_weight_scale"] = torch.randint(115, 121, (E, r, c // 32), generator=g,
                                                   dtype=torch.int32).to(torch.uint8)
    store = ExpertStore("nvfp4")  # MXFP8 parts switch it to VLLM_INKLING2_MXFP8
    store.build(parts, DEV)
    assert store.fmt == mx
    full = {n: ExpertStore.decode({k: v.to(DEV) for k, v in parts.items()}, n, torch.bfloat16)
            for n in ("w13", "w2")}
    moe = _moe(store, K, E)
    x = (torch.randn(tokens, H, generator=g) * 0.5).bfloat16().to(DEV)
    gw = (torch.randn(E, H, generator=g) * 0.05).to(DEV)
    w, ids = ref.route(x, gw, torch.zeros(E, device=DEV), K, 8.0)
    got = ops.experts(moe, x, w, ids)
    want = ref.moe_experts(x, w, ids, store.expert_weights, E, 4.0, 25.0)
    rel = ((got.float() - want.float()).norm() / want.float().norm()).item()
    assert rel < 2e-2, rel  # the kernel computes what the store holds
    for e in (0, E - 1):  # the store holds the checkpoint's experts
        for got_w, want_w in zip(store.expert_weights(e), (full["w13"][e], full["w2"][e])):
            if mx == "fp8":  # one scale per expert instead of per 32: lossy, bounded
                err = ((got_w.float() - want_w.float()).norm() / want_w.float().norm()).item()
                assert err < 0.1, err
            else:  # bf16 and mxfp8 keep the checkpoint values exactly
                assert torch.equal(got_w, want_w)
