# SPDX-License-Identifier: Apache-2.0
"""B70-IP-K: CPU tests (TRITON_INTERPRET=1) for the W4A8 MoE kernels in b70_moe_w4a8.

Run: TRITON_INTERPRET=1 python -m pytest tests/kernels/moe/test_b70_moe_w4a8.py
The weights are built as GPTQ int32 and pushed through the same transforms the
XPU loader applies (transpose, int32 -> uint8 view, implement_zp), so the test
checks that the kernel reads the bytes XpuFusedMoe holds."""

import os

import pytest
import torch

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
pytestmark = pytest.mark.skipif(not INTERPRET,
                                reason='CPU test: needs TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES=""')


def _patch_interpreter_bf16() -> None:
    """Test harness only (as IP-F's test): the Triton 3.8 interpreter keeps bf16 as raw
    uint16 and casts int<->bf16 on the bit patterns; route bf16 casts through fp32."""
    import triton.language as tl
    from triton.runtime import interpreter as interp

    builder = interp.InterpreterBuilder
    if getattr(builder, "_b70_ip_bf16", False):
        return
    cast_impl = builder.cast_impl

    def bf16_cast(self, src, dst_type):
        s, d = src.dtype.scalar, dst_type.scalar
        if (s == tl.bfloat16) != (d == tl.bfloat16) and tl.float32 not in (s, d):
            return cast_impl(self, cast_impl(self, src, tl.float32), dst_type)
        return cast_impl(self, src, dst_type)

    builder.cast_impl = bf16_cast
    builder._b70_ip_bf16 = True


if INTERPRET:
    _patch_interpreter_bf16()


def implement_zp(qweight: torch.Tensor) -> torch.Tensor:
    """Copy of vllm_xpu_kernels.fused_moe_interface.implement_zp (u4 -> s4, nibble positions kept)."""
    high = ((qweight >> 4) & 0x0F).to(torch.int8) - 8
    low = (qweight & 0x0F).to(torch.int8) - 8

    def enc(x):
        return ((x < 0).to(torch.uint8) << 3) | (x.view(torch.uint8) & 0x7)

    return (enc(high) << 4) | enc(low)


def gptq_random(e, k, n, gen):
    """GPTQ qweight [E, K/8, N] int32 (8 nibbles per word, ascending K), scales [E, K/128, N]."""
    nib = torch.randint(0, 16, (e, k, n), generator=gen, dtype=torch.int64)
    q = torch.zeros((e, k // 8, n), dtype=torch.int64)
    for i in range(8):
        q |= nib[:, i::8, :] << (4 * i)
    q = q.to(torch.int64).where(q < 2**31, q - 2**32).to(torch.int32)
    scales = (torch.rand((e, k // 128, n), generator=gen) * 0.02 + 0.001).to(torch.bfloat16)
    return q, scales, nib - 8


def to_xpu_layout(q, scales):
    """int_wna16._process_weights_xpu + implement_zp: [E, N, K/2] int8, scales [E, N, K/128]."""
    packed = q.transpose(1, 2).contiguous().view(torch.uint8)
    return implement_zp(packed).view(torch.int8), scales.transpose(1, 2).contiguous()


def deinterleave_inverse(xq):
    """Undo the per-128-group (evens, odds) layout."""
    r, k = xq.shape
    g = xq.view(r, k // 128, 2, 64)
    return torch.stack((g[:, :, 0], g[:, :, 1]), dim=-1).reshape(r, k)


@pytest.fixture(scope="module")
def mod():
    from vllm.model_executor.layers.fused_moe import b70_moe_w4a8

    return b70_moe_w4a8


def test_quant_rows_layout_and_scale(mod):
    gen = torch.Generator().manual_seed(0)
    x = (torch.randn((5, 256), generator=gen) * 3).to(torch.bfloat16)
    x[3] = 0
    xq, xs = mod.quant_rows(x)
    plain = deinterleave_inverse(xq).to(torch.float32)
    xf = x.float()
    amax = xf.abs().amax(1)
    assert torch.allclose(xs[:3], amax[:3] / 127, rtol=1e-6)
    ref = torch.clamp(torch.round(xf / xs[:, None]), -127, 127)
    assert (plain - ref).abs().max() <= 1  # only half-way ties may differ
    assert plain.abs().max() == 127 and torch.all(plain[3] == 0)


def test_quant_rows_silu(mod):
    gen = torch.Generator().manual_seed(1)
    inter = 256
    g1 = (torch.randn((7, 2 * inter), generator=gen) * 2).to(torch.bfloat16)
    n_valid = torch.tensor([5], dtype=torch.int32)
    xq, xs = mod.quant_rows(g1, n_valid, silu_inter=inter)
    act = torch.nn.functional.silu(g1[:, :inter].float()) * g1[:, inter:].float()
    a5 = act[:5]
    assert torch.allclose(xs[:5], a5.abs().amax(1) / 127, rtol=1e-5)
    ref = torch.clamp(torch.round(a5 / xs[:5, None]), -127, 127)
    plain = deinterleave_inverse(xq)[:5].float()
    assert (plain - ref).abs().max() <= 1
    # the int8 noise of a heavy-tailed SiLU*up row: a few % (rows 5, 6 are past n_valid: not written)
    rel = (plain * xs[:5, None] - a5).norm() / a5.norm()
    assert rel < 0.03, rel


def test_tile_map(mod):
    counts = torch.tensor([0, 5, 64, 65, 0, 130, 1], dtype=torch.int32)
    tile_e, tile_r0, seg_end = mod.tile_map(counts, int(counts.sum()), 64)
    got = [(int(e), int(r)) for e, r in zip(tile_e, tile_r0) if e >= 0]
    starts = [0, 0, 5, 69, 134, 134, 264]
    want = []
    for e, c in enumerate(counts.tolist()):
        for t in range((c + 63) // 64):
            want.append((e, starts[e] + 64 * t))
    assert got == want
    assert seg_end.tolist() == torch.cumsum(counts, 0).tolist()


@pytest.mark.parametrize("wload", [0, 1, 2])
@pytest.mark.parametrize("bm,bn", [(16, 32), (32, 64)])
def test_grouped_gemm_exact(mod, bm, bn, wload):
    """Kernel == float64 reference of (int8 A) x (int4 W from the GPTQ words) with both scales."""
    gen = torch.Generator().manual_seed(2)
    e, k, n = 3, 256, 64
    q, scales, w_int = gptq_random(e, k, n, gen)  # w_int [E, K, N] in [-8, 7]
    w, ws = to_xpu_layout(q, scales)
    counts = torch.tensor([3, 0, 37], dtype=torch.int32)
    rows = int(counts.sum())
    x = (torch.randn((rows + 4, k), generator=gen)).to(torch.bfloat16)
    aq, a_s = mod.quant_rows(x)
    tile_e, tile_r0, seg_end = mod.tile_map(counts, rows + 4, bm)
    out = torch.zeros((rows + 4, n), dtype=torch.float32)
    mod.grouped_gemm(aq, a_s, w, ws, out, tile_e, tile_r0, seg_end, (bm, bn, 4, 0, wload))
    a_plain = deinterleave_inverse(aq).double()
    start = 0
    for ex, c in enumerate(counts.tolist()):
        if not c:
            continue
        a = a_plain[start:start + c]
        acc = torch.zeros((c, n), dtype=torch.float64)
        for g in range(k // 128):
            sl = slice(g * 128, (g + 1) * 128)
            acc += (a[:, sl] @ w_int[ex, sl].double()) * scales[ex, g].double()[None, :]
        ref = acc * a_s[start:start + c].double()[:, None]
        torch.testing.assert_close(out[start:start + c].double(), ref, rtol=2e-6, atol=1e-6)
        start += c
    assert torch.all(out[rows:] == 0)  # rows past the last expert untouched


@pytest.mark.parametrize("wload", [0, 1, 2])
def test_grouped_gemm_dual_silu(mod, wload):
    """gemm1 + SiLU*up fused (dual accumulators) == the plain kernel followed by SiLU*up."""
    gen = torch.Generator().manual_seed(3)
    e, k, inter = 2, 256, 64
    q, scales, _ = gptq_random(e, k, 2 * inter, gen)
    w, ws = to_xpu_layout(q, scales)
    counts = torch.tensor([20, 9], dtype=torch.int32)
    rows = int(counts.sum())
    x = torch.randn((rows, k), generator=gen).to(torch.bfloat16)
    aq, a_s = mod.quant_rows(x)
    tile_e, tile_r0, seg_end = mod.tile_map(counts, rows, 16)
    g1 = torch.zeros((rows, 2 * inter), dtype=torch.float32)
    mod.grouped_gemm(aq, a_s, w, ws, g1, tile_e, tile_r0, seg_end, (16, 32, 4))
    act = torch.zeros((rows, inter), dtype=torch.float32)
    mod.grouped_gemm(aq, a_s, w, ws, act, tile_e, tile_r0, seg_end, (16, 32, 4, 0, wload), dual=True)
    ref = torch.nn.functional.silu(g1[:, :inter]) * g1[:, inter:]
    torch.testing.assert_close(act, ref, rtol=1e-5, atol=1e-5)


def test_gate_and_eligibility(mod, monkeypatch):
    """Default off; below the token threshold the stock path runs; non-g128 / non-SiLU impls are refused."""
    from types import SimpleNamespace

    def impl(**kw):
        d = dict(is_int4=True, activation="silu", gemm1_clamp_limit=None, w13_bias=None, w2_bias=None,
                 w13=torch.zeros((2, 512, 128), dtype=torch.int8),
                 gemm1_wei_scales=torch.zeros((2, 512, 2), dtype=torch.bfloat16),
                 w2=torch.zeros((2, 256, 128), dtype=torch.int8),
                 gemm2_wei_scales=torch.zeros((2, 256, 2), dtype=torch.bfloat16), inter_size=256)
        d.update(kw)
        return SimpleNamespace(**d)

    assert mod.eligible(impl()) is None
    assert mod.eligible(impl(activation="gelu")) is not None
    assert mod.eligible(impl(gemm1_wei_scales=torch.zeros((2, 512, 4), dtype=torch.bfloat16))) is not None
    assert mod.eligible(impl(is_int4=False)) is not None
    x = torch.zeros((8, 256), dtype=torch.bfloat16)
    monkeypatch.delenv("B70_MOE_W4A8", raising=False)
    assert not mod.maybe_apply(impl(), x, x, None, None)
    monkeypatch.setenv("B70_MOE_W4A8", "1")
    monkeypatch.setenv("B70_MOE_W4A8_MIN_TOKENS", "64")
    assert not mod.maybe_apply(impl(), x, x, None, None)  # 8 tokens < 64
    monkeypatch.setenv("B70_MOE_W4A8_MIN_TOKENS", "1")
    assert not mod.maybe_apply(impl(activation="gelu"), x, x, None, None)
