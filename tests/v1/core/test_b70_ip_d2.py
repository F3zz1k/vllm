# SPDX-License-Identifier: Apache-2.0
"""B70-IP-D2: int8 per-token-head KV on the XPU FLASH_ATTN backend, CPU tests.

1. Spec: with ``B70_PTH_FA_DECODE=1`` FLASH_ATTN packs int8_per_token_head
   exactly like TRITON_ATTN (same page bytes, so hybrid unify / K3 agree); off,
   nothing changes. The QSA backend keeps its inline packing either way.
2. Views: ``split_layout_views`` gives the vllm-xpu-kernels contract (int8
   [B, N, H, D] with strides (page, H*2D, 2D, 1), V at +D, fp32 scales
   [B, N, H] with strides (page/4, H, 1) after the data region), equal to the
   kernel test's own reference over raw page bytes, for NHD/HND logical caches,
   padded pages and a layer offset; it refuses non-compact and unaligned caches.
3. Writer (``TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES=``): the FA impl's
   do_kv_cache_update (TRITON_ATTN's per-token-head writer through the split
   views) puts the reference quantization at the reference byte positions,
   ignores slot -1 and padded key rows, and leaves every other byte alone.
4. K4 parity: the same tokens in an inline (TRITON_ATTN) cache and a split
   (D2) cache dequantize to the same bf16 blocks, and the forced K4 prefill over
   the split cache equals the stock K4 prefill over the inline cache.
5. Routing: fa_int8_forward sends q 1 / causal q <= 16 / non-causal q <= 16 to
   the INT8 decode call (int8 K/V + scale views) and longer batches to the
   dequant path; all agree with an fp32 reference on the dequantized cache.

    TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= pytest -q tests/v1/core/test_b70_ip_d2.py
"""

import os
import sys
import types
from types import SimpleNamespace

import pytest
import torch

from vllm.v1.attention.backends import b70_pth_fa
from vllm.v1.attention.backends.triton_attn import pth_half_bytes

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"
HS = 256


@pytest.fixture(autouse=True, params=["pad16", "tight"])
def pth_layout(request, monkeypatch):
    """B70-IP-PAD16: every test runs on vllm#62113's 16-B padded inline halves
    (default; 544-B slot at HS 256) and on the tight 520-B slot
    (B70_PTH_PAD16=0). D2 keeps the same page bytes as TRITON_ATTN either way;
    its split layout uses the first H*N*(2D+8) bytes of each page."""
    monkeypatch.setenv("B70_PTH_PAD16", "1" if request.param == "pad16" else "0")
    return request.param


class _XPU:
    @staticmethod
    def is_xpu():
        return True


@pytest.fixture
def d2_on(monkeypatch):
    monkeypatch.setattr(b70_pth_fa, "current_platform", _XPU())
    monkeypatch.setenv("B70_PTH_FA_DECODE", "1")


# --------------------------------------------------------------------------
# references
# --------------------------------------------------------------------------
def quantize_ref(x):
    """The int8_per_token_head writer semantics of vLLM (fp32)."""
    xf = x.float()
    scale = torch.clamp(xf.abs().amax(-1) / 127.0, min=1e-6)
    y = xf * (1.0 / scale)[..., None]
    y = torch.where(y >= 0, y + 0.5, y - 0.5).clamp(-128.0, 127.0).trunc()
    return y.to(torch.int8), scale


def kernel_ref_views(raw, block_size, nkv, hs):
    """vllm-xpu-kernels tests/flash_attn/int8_kv_b70_check.py split_layout_views,
    over raw [nb, page] uint8 (page may be padded)."""
    nb, page = raw.shape
    data = block_size * nkv * 2 * hs
    sb = block_size * nkv * 4
    d = torch.as_strided(
        raw.view(torch.int8),
        (nb, block_size, nkv, 2 * hs),
        (page, nkv * 2 * hs, 2 * hs, 1),
    )
    f = raw.view(torch.float32)

    def scales(off):
        return torch.as_strided(
            f, (nb, block_size, nkv), (page // 4, nkv, 1), f.storage_offset() + off // 4
        )

    return d[..., :hs], d[..., hs:], scales(data), scales(data + sb)


def make_layer_cache(nb, nkv, bs, hs=HS, nhd=True, pad=0, layer_off=0, fill=0x5A):
    """A vLLM-style logical (B, H, N, C) int8 layer view over a flat buffer.
    Returns (raw [nb, page] uint8 over the same bytes, kv_cache view)."""
    c = 2 * pth_half_bytes(hs)
    page = bs * nkv * c + pad
    buf = torch.full((layer_off + nb * page,), fill, dtype=torch.uint8)
    raw = buf[layer_off:].view(nb, page)
    i8 = buf.view(torch.int8)
    # physical [B, N, H, C] (NHD) or [B, H, N, C] (HND)
    stride = (page, c, nkv * c, 1) if nhd else (page, bs * c, c, 1)
    kv = torch.as_strided(i8, (nb, nkv, bs, c), stride, layer_off)
    return raw, kv


def inline_views(kv_cache):
    """TRITON_ATTN / K4 views over the inline layout (TritonAttentionImpl)."""
    from vllm.v1.attention.backends.triton_attn import TritonAttentionImpl
    from vllm.v1.kv_cache_interface import KVQuantMode

    impl = object.__new__(TritonAttentionImpl)
    impl._k_scale_cache = None
    impl._v_scale_cache = None
    impl.head_size = HS
    impl._kv_quant_mode = KVQuantMode.INT8_PER_TOKEN_HEAD
    impl._ensure_scale_caches(kv_cache)
    padded = kv_cache.shape[-1] // 2
    k, v = kv_cache.transpose(1, 2).split(padded, dim=-1)
    return k, v, impl._k_scale_cache, impl._v_scale_cache


# --------------------------------------------------------------------------
# 1. spec / gate
# --------------------------------------------------------------------------
def _spec(block=64, nkv=2):
    from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode

    return FullAttentionSpec(
        block_size=block,
        num_kv_heads=nkv,
        head_size=HS,
        dtype=torch.int8,
        kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
    )


def test_spec_matches_triton_when_on(d2_on, pth_layout):
    from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
    from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend

    for block, nkv in ((64, 2), (3264, 1), (832, 4)):
        spec = _spec(block, nkv)
        fa = FlashAttentionBackend.customize_spec(spec)
        tr = TritonAttentionBackend.customize_spec(spec)
        assert fa == tr
        slot = 544 if pth_layout == "pad16" else 520
        assert fa.page_size_bytes == block * nkv * slot
    assert FlashAttentionBackend.supports_kv_cache_dtype("int8_per_token_head")


def test_spec_unchanged_when_off(monkeypatch):
    from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend

    monkeypatch.setattr(b70_pth_fa, "current_platform", _XPU())
    monkeypatch.delenv("B70_PTH_FA_DECODE", raising=False)
    spec = _spec()
    assert FlashAttentionBackend.customize_spec(spec) == spec
    assert not FlashAttentionBackend.supports_kv_cache_dtype("int8_per_token_head")
    monkeypatch.setenv("B70_PTH_FA_DECODE", "1")
    monkeypatch.setattr(
        b70_pth_fa, "current_platform", SimpleNamespace(is_xpu=lambda: False)
    )
    assert not FlashAttentionBackend.supports_kv_cache_dtype("int8_per_token_head")


def test_qsa_keeps_inline_packing(d2_on):
    # import order: model before qsa
    pytest.importorskip("vllm.models.qwen4_exp.nvidia.model")
    from vllm.models.qwen4_exp.nvidia import qsa
    from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend

    spec = _spec(3264, 1)
    b = qsa.Qwen4ExpQSAFlashAttentionBackend
    assert b.customize_spec(spec) == TritonAttentionBackend.customize_spec(spec)
    assert b.supports_kv_cache_dtype("int8_per_token_head")


# --------------------------------------------------------------------------
# 2. views
# --------------------------------------------------------------------------
@pytest.mark.parametrize("nhd", [True, False])
@pytest.mark.parametrize(
    "nkv,bs,pad,off", [(1, 64, 0, 0), (2, 64, 128, 0), (4, 128, 0, 4096)]
)
def test_split_views_contract(nhd, nkv, bs, pad, off):
    nb = 5
    raw, kv = make_layer_cache(nb, nkv, bs, nhd=nhd, pad=pad, layer_off=off)
    k, v, ks, vs = b70_pth_fa.split_layout_views(kv, nkv, HS)
    rk, rv, rks, rvs = kernel_ref_views(raw, bs, nkv, HS)
    page = raw.shape[1]
    for got, ref in ((k, rk), (v, rv), (ks, rks), (vs, rvs)):
        assert got.shape == ref.shape and got.stride() == ref.stride()
        assert got.data_ptr() == ref.data_ptr() and got.dtype == ref.dtype
    assert k.stride() == (page, nkv * 2 * HS, 2 * HS, 1)
    assert v.data_ptr() - k.data_ptr() == HS
    assert ks.stride() == vs.stride() == (page // 4, nkv, 1)
    assert ks.data_ptr() - k.data_ptr() == bs * nkv * 2 * HS
    assert vs.data_ptr() - ks.data_ptr() == bs * nkv * 4
    # check_int8_kv_cache's alignment rules
    assert k.stride(0) % 64 == 0 and k.stride(2) % 64 == 0
    assert (k.data_ptr() - raw.data_ptr()) % 64 == 0
    # views only: no byte changed
    assert bool((raw == 0x5A).all())


def test_split_views_refuse_bad_layouts():
    # head dims outside the block (LHBNC-like): not block-compact
    nb, nkv, bs = 3, 2, 64
    c = 2 * pth_half_bytes(HS)
    buf = torch.zeros(nb * nkv * bs * c, dtype=torch.int8)
    kv = torch.as_strided(buf, (nb, nkv, bs, c), (bs * c, nb * bs * c, c, 1))
    with pytest.raises(ValueError, match="block-compact"):
        b70_pth_fa.split_layout_views(kv, nkv, HS)
    # padded page not a multiple of 64
    _, kv = make_layer_cache(3, 2, 64, pad=8)
    with pytest.raises(ValueError, match="64-byte"):
        b70_pth_fa.split_layout_views(kv, 2, HS)
    # head size not a multiple of 64
    _, kv = make_layer_cache(3, 2, 64, hs=96)
    with pytest.raises(ValueError, match="64-byte"):
        b70_pth_fa.split_layout_views(kv, 2, 96)


@pytest.mark.parametrize("content", [2 * HS, 2 * (HS + 4) + 4, 2 * (HS + 24)])
def test_split_views_refuse_wrong_content(content):
    """B70-IP-PAD16: the content width must be a per-token-head cell for this
    head size (tight or 16-B padded), not a fixed 2 * (D + 4)."""
    buf = torch.zeros(3 * 2 * 64 * content, dtype=torch.int8)
    kv = buf.view(3, 64, 2, content).transpose(1, 2)
    with pytest.raises(ValueError, match="does not match"):
        b70_pth_fa.split_layout_views(kv, 2, HS)


def test_views_cache_follows_runner_reset():
    _, kv = make_layer_cache(3, 2, 64)
    impl = SimpleNamespace(num_kv_heads=2, head_size=HS, _k_scale_cache=None)
    a = b70_pth_fa.pth_split_views(impl, kv)
    assert b70_pth_fa.pth_split_views(impl, kv) is a
    impl._k_scale_cache = None  # what the runner does after profiling
    b = b70_pth_fa.pth_split_views(impl, kv)
    assert b is not a and b[0].data_ptr() == a[0].data_ptr()
    _, kv2 = make_layer_cache(3, 2, 64)
    assert b70_pth_fa.pth_split_views(impl, kv2)[0].data_ptr() == kv2.data_ptr()


# --------------------------------------------------------------------------
# 3. writer
# --------------------------------------------------------------------------
@pytest.mark.skipif(
    not INTERPRET, reason='needs TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES=""'
)
@pytest.mark.parametrize("nhd", [True, False])
@pytest.mark.parametrize("nkv,pad", [(1, 0), (2, 128)])
def test_writer_roundtrip(nhd, nkv, pad):
    from vllm.v1.attention.backend import AttentionType
    from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl

    torch.manual_seed(0)
    nb, bs = 4, 64
    raw, kv = make_layer_cache(nb, nkv, bs, nhd=nhd, pad=pad)
    before = raw.clone()
    impl = object.__new__(FlashAttentionImpl)
    impl.attn_type = AttentionType.DECODER
    impl._b70_fa_int8 = True
    impl._k_scale_cache = None
    impl.num_kv_heads, impl.head_size = nkv, HS

    nt = 9
    key = torch.randn(nt + 3, nkv, HS) * 2  # 3 padded rows past slot_mapping
    key[:, :, 7] *= 20
    val = torch.randn(nt + 3, nkv, HS)
    slots = torch.tensor([0, 1, 63, 64, 130, -1, 200, 255, 77], dtype=torch.int64)
    layer = SimpleNamespace()
    impl.do_kv_cache_update(layer, key, val, kv, slots)

    rk, rv, rks, rvs = kernel_ref_views(raw, bs, nkv, HS)
    exp = before.clone()
    ek, ev, eks, evs = kernel_ref_views(exp, bs, nkv, HS)
    kq, kscale = quantize_ref(key[:nt])
    vq, vscale = quantize_ref(val[:nt])
    for t, s in enumerate(slots.tolist()):
        if s < 0:
            continue
        b, o = divmod(s, bs)
        ek[b, o], ev[b, o], eks[b, o], evs[b, o] = kq[t], vq[t], kscale[t], vscale[t]
    assert torch.equal(rk, ek) and torch.equal(rv, ev)
    assert torch.equal(rks, eks) and torch.equal(rvs, evs)
    assert torch.equal(raw, exp)  # nothing else (padding, other slots) touched


# --------------------------------------------------------------------------
# 4. K4 parity
# --------------------------------------------------------------------------
def _fill_both(nkv, bs, nb, seqs, seed=0):
    """Same quantized tokens into an inline cache and a split cache."""
    g = torch.Generator().manual_seed(seed)
    _, kv_in = make_layer_cache(nb, nkv, bs, fill=0)
    _, kv_sp = make_layer_cache(nb, nkv, bs, fill=0)
    ik, iv, iks, ivs = inline_views(kv_in)
    sk, sv, sks, svs = b70_pth_fa.split_layout_views(kv_sp, nkv, HS)
    perm = (torch.randperm(nb - 1, generator=g) + 1).tolist()
    bt = torch.zeros(len(seqs), max(-(-s // bs) for s in seqs), dtype=torch.int32)
    dense_k, dense_v, c = [], [], 0
    for i, L in enumerate(seqs):
        n = -(-L // bs)
        bt[i, :n] = torch.tensor(perm[c : c + n])
        c += n
        key = torch.randn(L, nkv, HS, generator=g) * 2
        val = torch.randn(L, nkv, HS, generator=g)
        kq, ksc = quantize_ref(key)
        vq, vsc = quantize_ref(val)
        pos = torch.arange(L)
        b, o = bt[i][pos // bs].long(), pos % bs
        for kk, vv, kss, vss in ((ik, iv, iks, ivs), (sk, sv, sks, svs)):
            kk[b, o, :, :HS] = kq
            vv[b, o, :, :HS] = vq
            kss[b, o], vss[b, o] = ksc, vsc
        dense_k.append(kq.float() * ksc[..., None])
        dense_v.append(vq.float() * vsc[..., None])
    return kv_in, kv_sp, bt, dense_k, dense_v


def test_dequant_blocks_parity_with_k4():
    nkv, bs, nb = 2, 64, 9
    kv_in, kv_sp, bt, _, _ = _fill_both(nkv, bs, nb, [70, 130, 5])
    ik, iv, iks, ivs = inline_views(kv_in)
    sk, sv, sks, svs = b70_pth_fa.split_layout_views(kv_sp, nkv, HS)
    ids = torch.arange(1, nb, dtype=torch.long)
    for dt in (torch.bfloat16, torch.float16):
        a = torch.empty(ids.numel(), bs, nkv, HS, dtype=dt)
        b = torch.empty_like(a)
        b70_pth_fa._dequant_blocks(ik, iks, ids, a, HS)
        b70_pth_fa._dequant_blocks(sk, sks, ids, b, HS)
        assert torch.equal(a, b)
        b70_pth_fa._dequant_blocks(iv, ivs, ids, a, HS)
        b70_pth_fa._dequant_blocks(sv, svs, ids, b, HS)
        assert torch.equal(a, b)


# --------------------------------------------------------------------------
# kernel stand-ins (no vllm_xpu_kernels on CPU)
# --------------------------------------------------------------------------
class _FakeXpuOps:
    """flash_attn_varlen_func stand-in: paged causal attention in fp32 over
    either a bf16 cache or an int8 cache with per-token-head descale views."""

    def __init__(self):
        self.calls = []

    def flash_attn_varlen_func(
        self,
        q,
        k,
        v,
        cu_seqlens_q,
        max_seqlen_q,
        max_seqlen_k,
        softmax_scale=None,
        causal=False,
        out=None,
        block_table=None,
        seqused_k=None,
        window_size=None,
        k_descale=None,
        v_descale=None,
        s_aux=None,
        **kw,
    ):
        self.calls.append(
            SimpleNamespace(kdtype=k.dtype, max_q=max_seqlen_q, causal=causal)
        )
        if k.dtype == torch.int8:
            assert k_descale is not None and k_descale.shape == k.shape[:3]
            kf = k.float() * k_descale[..., None]
            vf = v.float() * v_descale[..., None]
        else:
            kf, vf = k.float(), v.float()
        bs, nkv = k.shape[1], k.shape[2]
        cu = cu_seqlens_q.tolist()
        used = seqused_k.tolist()
        for i in range(len(cu) - 1):
            q0, q1, L = cu[i], cu[i + 1], used[i]
            if q1 == q0:
                continue
            nblk = -(-L // bs)
            blocks = block_table[i, :nblk].long()
            K = kf[blocks].reshape(-1, nkv, kf.shape[-1])[:L]
            V = vf[blocks].reshape(-1, nkv, vf.shape[-1])[:L]
            out[q0:q1] = _ref_attn(q[q0:q1].float(), K, V, softmax_scale, causal).to(
                out.dtype
            )
        return out


def _ref_attn(q, K, V, scale, causal):
    g = q.shape[1] // K.shape[1]
    K, V = K.repeat_interleave(g, 1), V.repeat_interleave(g, 1)
    s = torch.einsum("qhd,khd->hqk", q, K) * scale
    if causal:
        L, ql = K.shape[0], q.shape[0]
        qpos = torch.arange(L - ql, L)
        mask = torch.arange(L)[None, :] <= qpos[:, None]
        s = s.masked_fill(~mask[None], float("-inf"))
    return torch.einsum("hqk,khd->qhd", torch.softmax(s, -1), V)


@pytest.fixture
def fake_kernels(monkeypatch):
    ops = _FakeXpuOps()
    mod = types.ModuleType("vllm._xpu_ops")
    mod.xpu_ops = ops
    monkeypatch.setitem(sys.modules, "vllm._xpu_ops", mod)
    pkg = types.ModuleType("vllm_xpu_kernels")
    fai = types.ModuleType("vllm_xpu_kernels.flash_attn_interface")
    fai._SPEC_DECODE_MAX_QLEN = 16
    pkg.flash_attn_interface = fai
    monkeypatch.setitem(sys.modules, "vllm_xpu_kernels", pkg)
    monkeypatch.setitem(sys.modules, "vllm_xpu_kernels.flash_attn_interface", fai)
    monkeypatch.setattr(b70_pth_fa, "_scratch", {})
    return ops


def _impl(nkv, hq, max_len=4096):
    return SimpleNamespace(
        num_kv_heads=nkv,
        head_size=HS,
        scale=HS**-0.5,
        alibi_slopes=None,
        sinks=None,
        logits_soft_cap=0,
        sliding_window=(-1, -1),
        _b70_max_model_len=max_len,
        _k_scale_cache=None,
        _v_scale_cache=None,
    )


def _meta(q_lens, seqs, bt, causal=True):
    qsl = torch.tensor([0] + torch.tensor(q_lens).cumsum(0).tolist(), dtype=torch.int32)
    return SimpleNamespace(
        query_start_loc=qsl,
        seq_lens=torch.tensor(seqs, dtype=torch.int32),
        block_table=bt,
        max_query_len=max(q_lens),
        max_seq_len=max(seqs),
        causal=causal,
    )


def test_forced_k4_split_equals_k4_inline(fake_kernels, monkeypatch):
    monkeypatch.setattr(b70_pth_fa, "current_platform", _XPU())
    monkeypatch.delenv("B70_PTH_FA_PREFILL", raising=False)
    nkv, hq, bs, nb = 2, 8, 64, 12
    seqs, q_lens = [200, 90, 130], [100, 90, 70]
    kv_in, kv_sp, bt, _, _ = _fill_both(nkv, bs, nb, seqs, seed=1)
    meta = _meta(q_lens, seqs, bt)
    q = torch.randn(sum(q_lens), hq, HS).to(torch.bfloat16)

    # stock K4 over the inline cache (TRITON_ATTN path)
    imp_a = _impl(nkv, hq)
    ik, iv, iks, ivs = inline_views(kv_in)
    imp_a._k_scale_cache, imp_a._v_scale_cache = iks, ivs
    out_a = torch.zeros_like(q)
    assert b70_pth_fa.maybe_fa_prefill(imp_a, q, ik, iv, out_a, meta, q.shape[0])

    # forced K4 over the split cache (D2 path)
    imp_b = _impl(nkv, hq)
    sk, sv, sks, svs = b70_pth_fa.split_layout_views(kv_sp, nkv, HS)
    out_b = torch.zeros_like(q)
    assert b70_pth_fa.maybe_fa_prefill(
        imp_b, q, sk, sv, out_b, meta, q.shape[0], pth_scales=(sks, svs), force=True
    )
    assert torch.equal(out_a, out_b)
    assert all(c.kdtype == torch.bfloat16 for c in fake_kernels.calls)


# --------------------------------------------------------------------------
# 5. routing
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "q_lens,causal,want",
    [
        ([1, 1, 1], True, "int8"),
        ([8, 8, 8], True, "int8"),  # uniform DFlash K=7 verify
        ([3, 1, 5], True, "int8"),  # adaptive verification (ragged)
        ([8, 8, 8], False, "int8"),  # non-causal draft block
        ([1, 40, 1], True, "bf16"),  # mixed with a prefill chunk
        ([20, 20, 20], False, "bf16"),
    ],
)
def test_fa_int8_forward_routes(fake_kernels, monkeypatch, q_lens, causal, want):
    monkeypatch.setattr(b70_pth_fa, "current_platform", _XPU())
    monkeypatch.setenv("B70_PTH_FA_PREFILL", "0")  # forced path must ignore it
    nkv, hq, bs, nb = 2, 8, 64, 12
    seqs = [150, 64, 129]
    _, kv_sp, bt, dk, dv = _fill_both(nkv, bs, nb, seqs, seed=2)
    meta = _meta(q_lens, seqs, bt, causal)
    q = torch.randn(sum(q_lens), hq, HS).to(torch.bfloat16)
    out = torch.zeros_like(q)
    impl = _impl(nkv, hq)
    b70_pth_fa.fa_int8_forward(impl, q, kv_sp, meta, out, q.shape[0], (-1, -1))
    kinds = {("int8" if c.kdtype == torch.int8 else "bf16") for c in fake_kernels.calls}
    assert kinds == {want}
    if want == "int8":
        assert all(c.causal for c in fake_kernels.calls)
    qsl = meta.query_start_loc.tolist()
    for i in range(len(seqs)):
        ref = _ref_attn(
            q[qsl[i] : qsl[i + 1]].float(), dk[i], dv[i], impl.scale, causal
        )
        got = out[qsl[i] : qsl[i + 1]].float()
        rel = (got - ref).norm() / ref.norm()
        assert rel < 1.5e-2, (i, rel.item())
