# SPDX-License-Identifier: Apache-2.0
"""B70-IP-K7: native int8 dots in the Qwen4Exp QSA kernels, CPU checks.

Part 1 needs no Triton: the B70_QSA_I8DOT gate, the launch kwargs, and the
int8-dot math (int8 Q per (token, head), int32 Q.K, optional int8 P.V) emulated
in torch against the dense fp32 reference over the dequantized cache.

Part 2 runs the real kernels (Q quantizer, attn2 and split-K with I8DOT 1/2)
on CPU tensors through the Triton interpreter; it reuses the IP-F harness
(test_b70_ip_qsa_int8) and is skipped unless started with
``TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES=""``.
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import test_b70_ip_qsa_int8 as ipf  # noqa: E402  (also patches the interpreter)
from test_b70_ip_qsa_int8 import pth_layout  # noqa: E402, F401  (pad16 / tight)

from vllm.models.qwen4_exp.nvidia.ops import qsa_attn2  # noqa: E402

HD = ipf.HD
INTERPRET = ipf.INTERPRET
requires_interpreter = ipf.requires_interpreter


def quantize_q(q: torch.Tensor):
    """Torch replica of _qsa_quant_q_kernel."""
    x = q.float()
    scale = torch.clamp(x.abs().amax(-1) / 127.0, min=1e-10)
    y = x * (1.0 / scale)[..., None]
    y = torch.where(y >= 0, (y + 0.5).floor(), (y - 0.5).ceil()).clamp(-127, 127)
    return y.to(torch.int8), scale


# --------------------------------------------------------------------------
# Part 1
# --------------------------------------------------------------------------


@pytest.mark.parametrize("raw,mode", [(None, 0), ("", 0), ("0", 0), ("1", 1), ("2", 2)])
def test_i8dot_mode_parses(raw, mode, monkeypatch) -> None:
    if raw is None:
        monkeypatch.delenv(qsa_attn2.I8DOT_ENV, raising=False)
    else:
        monkeypatch.setenv(qsa_attn2.I8DOT_ENV, raw)
    assert qsa_attn2.i8dot_mode() == mode


def test_i8dot_mode_rejects_junk(monkeypatch) -> None:
    monkeypatch.setenv(qsa_attn2.I8DOT_ENV, "yes")
    with pytest.raises(ValueError):
        qsa_attn2.i8dot_mode()


def test_i8dot_kwargs() -> None:
    assert qsa_attn2.i8dot_kwargs(None, 0) == {}
    s = torch.zeros(5, 24)
    kw = qsa_attn2.i8dot_kwargs(s, 2)
    assert kw["I8DOT"] == 2 and kw["stride_qs_row"] == 24 and kw["q_scale_ptr"] is s


def _emulate(q, kq, ks, vq, vs, mode):
    """One row, one tile = all tokens: the kernel's int8 algebra in torch."""
    hq, nkv = q.shape[0], kq.shape[1]
    rep = hq // nkv
    kq, vq = kq.repeat_interleave(rep, 1), vq.repeat_interleave(rep, 1)
    ks, vs = ks.repeat_interleave(rep, 1), vs.repeat_interleave(rep, 1)
    qi, qs = quantize_q(q)
    s_int = torch.einsum("hd,khd->hk", qi.double(), kq.double())  # exact int32
    s = s_int.float() * qs[:, None] * ks.T * HD**-0.5
    p = torch.exp(s - s.amax(-1, keepdim=True))
    pv = p * vs.T
    if mode == 2:
        ps = (pv.amax(-1) / 127.0).clamp(min=1e-30)
        pi = (pv / ps[:, None] + 0.5).floor().clamp(max=127)
        o = torch.einsum("hk,khd->hd", pi.double(), vq.double()).float() * ps[:, None]
    else:
        o = torch.einsum("hk,khd->hd", pv.bfloat16().float(), vq.float())
    return o / p.sum(-1, keepdim=True)


@pytest.mark.parametrize("mode", [1, 2])
@pytest.mark.parametrize("hq,nkv", [(24, 2), (12, 1)])
@pytest.mark.parametrize("ntok", [32, 2051])
def test_i8dot_math_vs_dequant_reference(mode, hq, nkv, ntok) -> None:
    torch.manual_seed(ntok + hq + mode)
    key = torch.randn(ntok, nkv, HD) * torch.rand(ntok, nkv, 1) * 3
    value = torch.randn(ntok, nkv, HD) * torch.rand(ntok, nkv, 1)
    key[::97] *= 8
    kq, ks = ipf.quantize_pth(key.bfloat16())
    vq, vs = ipf.quantize_pth(value.bfloat16())
    q = torch.randn(hq, HD).bfloat16()
    kd, vd = kq.float() * ks[..., None], vq.float() * vs[..., None]
    ref = ipf.dense_reference(q, kd, vd, HD**-0.5).float()
    got = _emulate(q, kq, ks, vq, vs, mode)
    rel, cos = ipf._metrics(got, ref)
    print(
        f"[IP-I] emulate mode={mode} hq={hq} ntok={ntok}: rel={rel:.2e} cos={cos:.6f}"
    )
    # The fixture is sharp (outlier keys x8, logits up to ~80, mean top
    # probability ~0.9): the int8 Q error lands on the peak logit. Measured
    # (6 seeds): vs this reference rel 0.4e-2..6.5e-2 (the tail is a
    # near-tie of two peak logits ~60-80); vs the original bf16 K/V 1.0-3.5x
    # the int8-storage error. Envelope gate here; GPU numbers are measured separately.
    assert rel < 8e-2 and cos > 0.997


# --------------------------------------------------------------------------
# Part 2: real kernels through the interpreter
# --------------------------------------------------------------------------


@requires_interpreter
def test_quant_q_kernel_matches_formula() -> None:
    torch.manual_seed(3)
    q = torch.randn(5, 12, HD).bfloat16() * 3
    q[1, 2] = 0  # all-zero head
    q[2, 3, 7] = 40  # outlier
    qi, qs = qsa_attn2.quantize_q_i8(q)
    ri, rs = quantize_q(q)
    assert torch.equal(qi, ri)
    torch.testing.assert_close(qs, rs, rtol=0, atol=0)


@requires_interpreter
@pytest.mark.parametrize("mode", ["1", "2"])
@pytest.mark.parametrize("impl", ["attn2", "splitk"])
@pytest.mark.parametrize("hq,nkv", [(24, 2), (12, 1)])
@pytest.mark.parametrize("use_prefill_config", [False, True])
def test_qsa_kernels_i8dot_vs_reference(
    mode, impl, hq, nkv, use_prefill_config, monkeypatch
) -> None:
    case = ipf._paged_case(hq, nkv, page=16, rows=3, width=67, seed=hq + nkv)
    ipf._write_int8(case)
    k_ref, ks_ref = ipf.quantize_pth(case["key"])
    v_ref, vs_ref = ipf.quantize_pth(case["value"])
    deq = ipf._reference_rows(
        case, k_ref.float() * ks_ref[..., None], v_ref.float() * vs_ref[..., None]
    )
    monkeypatch.setenv(qsa_attn2.I8DOT_ENV, "0")
    out_s = ipf._run(case, impl, monkeypatch, True, use_prefill_config)
    monkeypatch.setenv(qsa_attn2.I8DOT_ENV, mode)
    out_i = ipf._run(case, impl, monkeypatch, True, use_prefill_config)
    rel_d, cos_d = ipf._metrics(out_i.flatten(1), deq.flatten(1))
    rel_s, cos_s = ipf._metrics(out_i, out_s)
    print(
        f"[IP-I] {impl} I8DOT={mode} hq={hq} nkv={nkv} "
        f"prefill_cfg={use_prefill_config}: "
        f"vs dequant ref rel={rel_d:.2e} cos={cos_d:.6f}; vs int8-storage "
        f"rel={rel_s:.2e} cos={cos_s:.6f}"
    )
    assert torch.isfinite(out_i).all()
    assert rel_d < 4e-2 and cos_d > 0.999


@requires_interpreter
@pytest.mark.parametrize("impl", ["attn2", "splitk"])
def test_i8dot_ignored_for_bf16_cache(impl, monkeypatch) -> None:
    case = ipf._paged_case(12, 1, page=16, rows=2, width=35, seed=7)
    monkeypatch.setenv(qsa_attn2.I8DOT_ENV, "0")
    a = ipf._run(case, impl, monkeypatch, False, False)
    monkeypatch.setenv(qsa_attn2.I8DOT_ENV, "2")
    b = ipf._run(case, impl, monkeypatch, False, False)
    assert torch.equal(a, b)


# --------------------------------------------------------------------------
# B70-IP-K8: int8-dot launch regions and the ALIGN8 decision
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "mode,stem",
    [(0, "sparse_attn2_"), (1, "sparse_attn2_i8_"), (2, "sparse_attn2_i8pv_")],
)
@pytest.mark.parametrize(
    "rows,pre,tail", [(64, False, "decode"), (4096, True, "prefill")]
)
def test_attn2_i8_regions(mode, stem, rows, pre, tail) -> None:
    cfg = qsa_attn2.select_attn2_config(rows, 1, pre, 2051, mode)
    assert cfg["region"] == stem + tail
    if mode:
        # The int8 shapes bf16 cannot compile (IGC SIGFPE guard) are allowed here.
        assert cfg["num_warps"] > 1 and cfg["block_n"] >= 64
        assert cfg["extra"] == {"grf_mode": "auto"}
        assert HD // cfg["num_dchunk"] >= 32


def test_attn2_i8_profiles_cover_both_regions() -> None:
    prof = qsa_attn2.attn2_profiles(1, 2051, max_rows=4096, i8dot=2)
    assert {p[0] for p in prof} == {128} and {p[3] for p in prof} == {4}


@pytest.mark.parametrize("layout", ["NHD", "padded"])
@pytest.mark.parametrize("nkv", [1, 2])
def test_align8_true_for_inline_cells(layout, nkv) -> None:
    from vllm.models.qwen4_exp.nvidia.ops import qsa as qsa_ops

    nb, page = 3, 16
    content = 2 * ipf._half()
    if layout == "NHD":
        cache = torch.zeros(nb, page, nkv, content, dtype=torch.int8).transpose(1, 2)
    else:
        page_elems = page * nkv * content + 4096
        raw = torch.zeros(nb * page_elems + 64, dtype=torch.int8)
        cache = torch.as_strided(
            raw, (nb, nkv, page, content), (page_elems, content, nkv * content, 1), 64
        )
    k, v, _, _ = qsa_ops.qsa_pth_kv_views(cache, HD)
    assert qsa_attn2.i8dot_align8(k, v, HD)
    # B70-IP-PAD16: V offset from the cell, 16-B gathers on the padded cell.
    kw = qsa_attn2.i8dot_align_kwargs(k, v, HD)
    assert kw["V_OFF"] == ipf._half()
    assert kw["KV_ALIGN"] == (16 if os.environ["B70_PTH_PAD16"] == "1" else 8)
    # Not the inline cell (separate tensors) or an odd base: no alignment claim.
    assert not qsa_attn2.i8dot_align8(k, k.clone(), HD)
    # A 4-byte (not 8-byte) aligned base: scale views still work, ALIGN8 must not.
    under = torch.zeros(cache.untyped_storage().nbytes() + 8, dtype=torch.int8)
    off = cache.storage_offset() + 4
    odd = torch.as_strided(under, cache.shape, cache.stride(), off)
    k2, v2, _, _ = qsa_ops.qsa_pth_kv_views(odd, HD)
    assert not qsa_attn2.i8dot_align8(k2, v2, HD)
