# SPDX-License-Identifier: Apache-2.0
"""B70-IP-K1 / K2: per-token-head KV on the Triton backend, CPU-only checks.

K1: unified_attention launches the kernel with USE_TD=False for per-token-head
    KV (inline fp32 scales make the K/V pitch unaligned for Xe2 2D block
    reads), unless B70_PTH_TD_GUARD=0.
K2: building the scale views does not write the cache (hybrid models share the
    pages with GDN state; vllm#63944).
PAD16: vllm#62113's 16-B padded halves (B70_PTH_PAD16, default 1) and the tight
    layout both round-trip writer -> scale views -> kernel_unified_attention
    (interpreter) against an fp32 reference over the dequantized cache.
"""

import os

import pytest
import torch

import vllm.v1.attention.ops.triton_unified_attention as ua
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    pth_half_bytes,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode

INTERPRET = os.environ.get("TRITON_INTERPRET") == "1"


@pytest.fixture(params=["pad16", "tight"])
def pth_layout(request, monkeypatch):
    monkeypatch.setenv("B70_PTH_PAD16", "1" if request.param == "pad16" else "0")
    return request.param


def _impl(hs, mode=KVQuantMode.INT8_PER_TOKEN_HEAD):
    impl = TritonAttentionImpl.__new__(TritonAttentionImpl)
    impl._k_scale_cache = None
    impl._v_scale_cache = None
    impl.head_size = hs
    impl._kv_quant_mode = mode
    return impl


@pytest.mark.parametrize("hs", [64, 128, 256])
def test_spec_half_matches_upstream_62113(pth_layout, hs):
    spec = TritonAttentionBackend.customize_spec(
        FullAttentionSpec(
            block_size=16,
            num_kv_heads=2,
            head_size=hs,
            dtype=torch.int8,
            kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
        )
    )
    half = {64: 80, 128: 144, 256: 272} if pth_layout == "pad16" else None
    want = half[hs] if half else hs + 4
    assert pth_half_bytes(hs) == want
    assert spec.state_content_bytes == 2 * want
    assert spec.page_size_bytes == 16 * 2 * 2 * want


def test_scale_views_leave_cache_bytes_unchanged(pth_layout):
    nb, nkv, bs, hs = 4, 2, 16, 64
    h = pth_half_bytes(hs)
    cache = torch.full((nb, nkv, bs, 2 * h), 37, dtype=torch.int8)
    before = cache.clone()
    impl = _impl(hs)
    impl._ensure_scale_caches(cache)
    assert torch.equal(cache, before)
    # the views address the inline scale bytes
    assert impl._k_scale_cache.shape == (nb, bs, nkv)
    impl._k_scale_cache[1, 3, 1] = 2.5
    raw = cache[1, 1, 3, hs : hs + 4].contiguous().view(torch.float32)
    assert raw.item() == 2.5
    impl._v_scale_cache[2, 0, 0] = -1.0
    raw = cache[2, 0, 0, h + hs : h + hs + 4].contiguous()
    assert raw.view(torch.float32).item() == -1.0


def test_scale_views_refuse_mismatched_head(pth_layout):
    # A cache whose half cannot hold this impl's head is refused, not misread.
    cache = torch.zeros(2, 1, 16, 2 * pth_half_bytes(64), dtype=torch.int8)
    with pytest.raises(ValueError):
        _impl(128)._ensure_scale_caches(cache)


def _quant_ref(x):
    xf = x.float()
    scale = torch.clamp(xf.abs().amax(-1) / 127.0, min=1e-6)
    y = xf * (1.0 / scale)[..., None]
    y = torch.where(y >= 0, y + 0.5, y - 0.5).clamp(-128.0, 127.0).trunc()
    return y.to(torch.int8), scale


@pytest.mark.skipif(
    not INTERPRET, reason='needs TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES=""'
)
@pytest.mark.parametrize("nhd", [False, True])
@pytest.mark.parametrize("hs", [128, 256])
def test_triton_attn_roundtrip_writer_to_kernel(pth_layout, nhd, hs, monkeypatch):
    """TRITON_ATTN round trip under the active layout: the per-token-head
    writer through _pth_key_value_caches, then the unified kernel through the
    same views, vs an fp32 reference over the dequantized tokens. The pad
    bytes are never written."""
    from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
        triton_reshape_and_cache_flash_per_token_head_quant,
    )

    monkeypatch.delenv("B70_PTH_TD_GUARD", raising=False)
    torch.manual_seed(hs)
    nb, nkv, nq, bs = 6, 2, 4, 16
    h = pth_half_bytes(hs)
    shape = (nb, bs, nkv, 2 * h) if nhd else (nb, nkv, bs, 2 * h)
    cache = torch.zeros(shape, dtype=torch.int8)
    kv_cache = cache.transpose(1, 2) if nhd else cache  # logical (B, H, N, C)
    impl = _impl(hs)
    impl.fp8_dtype = None
    key_cache, value_cache = impl._pth_key_value_caches(kv_cache)
    assert value_cache.data_ptr() - key_cache.data_ptr() == h
    if pth_layout == "pad16":
        assert h % 16 == 0 and key_cache.stride(1 if nhd else 2) % 16 == 0

    seq_lens, q_lens = [37, 20], [3, 1]
    ntok = sum(seq_lens)
    key = torch.randn(ntok, nkv, hs) * 2
    val = torch.randn(ntok, nkv, hs)
    block_table = torch.tensor([[4, 1, 5], [0, 3, 2]], dtype=torch.int32)
    slots = []
    for r, n in enumerate(seq_lens):
        for t in range(n):
            slots.append(int(block_table[r, t // bs]) * bs + t % bs)
    slot_mapping = torch.tensor(slots, dtype=torch.int64)
    triton_reshape_and_cache_flash_per_token_head_quant(
        key,
        val,
        key_cache,
        value_cache,
        impl._k_scale_cache,
        impl._v_scale_cache,
        slot_mapping,
        kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
    )
    flat = kv_cache.transpose(1, 2).reshape(nb * bs, nkv, 2 * h)
    # pad bytes untouched
    assert not flat[..., hs + 4 : h].any() and not flat[..., h + hs + 4 :].any()
    kq, ks = _quant_ref(key)
    vq, vs = _quant_ref(val)
    assert torch.equal(flat[slot_mapping, :, :hs], kq)
    assert torch.equal(flat[slot_mapping, :, h : h + hs], vq)

    q = torch.randn(sum(q_lens), nq, hs, dtype=torch.float16)
    out = torch.zeros_like(q)
    cu_q = torch.tensor([0, q_lens[0], sum(q_lens)], dtype=torch.int32)
    scale = hs**-0.5
    ua.unified_attention(
        q=q,
        k=key_cache,
        v=value_cache,
        out=out,
        cu_seqlens_q=cu_q,
        max_seqlen_q=max(q_lens),
        seqused_k=torch.tensor(seq_lens, dtype=torch.int32),
        max_seqlen_k=max(seq_lens),
        softmax_scale=scale,
        causal=True,
        alibi_slopes=None,
        window_size=(-1, -1),
        block_table=block_table,
        softcap=0,
        q_descale=None,
        k_descale=None,
        v_descale=None,
        kv_quant_mode=KVQuantMode.INT8_PER_TOKEN_HEAD,
        k_scale_cache=impl._k_scale_cache,
        v_scale_cache=impl._v_scale_cache,
    )
    kd = kq.float() * ks[..., None]
    vd = vq.float() * vs[..., None]
    ref, t0, q0 = [], 0, 0
    for n, ql in zip(seq_lens, q_lens):
        k_r = kd[t0 : t0 + n].repeat_interleave(nq // nkv, dim=1)
        v_r = vd[t0 : t0 + n].repeat_interleave(nq // nkv, dim=1)
        for i in range(ql):
            pos = n - ql + i  # causal: query i sees keys [0, pos]
            s_ = torch.einsum("hd,thd->ht", q[q0 + i].float(), k_r[: pos + 1]) * scale
            ref.append(torch.einsum("ht,thd->hd", s_.softmax(-1), v_r[: pos + 1]))
        t0 += n
        q0 += ql
    ref = torch.stack(ref)
    got = out.float()
    rel = ((got - ref).norm() / ref.norm()).item()
    cos = torch.nn.functional.cosine_similarity(
        got.flatten(), ref.flatten(), dim=0
    ).item()
    print(f"[PAD16] triton {pth_layout} nhd={nhd} hs={hs}: rel={rel:.2e} cos={cos:.6f}")
    assert rel < 5e-3 and cos > 0.99999


class _FakeKernel:
    def __init__(self):
        self.kwargs = None

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.kwargs = kwargs

        return launch


def _call(monkeypatch, mode, use_td):
    fake = _FakeKernel()
    monkeypatch.setattr(ua, "kernel_unified_attention", fake)
    # With no device, vllm.triton_utils hands out a placeholder module.
    monkeypatch.setattr(
        ua.triton,
        "next_power_of_2",
        lambda n: 1 << (n - 1).bit_length(),
        raising=False,
    )
    monkeypatch.setattr(ua.triton, "cdiv", lambda a, b: -(-a // b), raising=False)
    nkv, nq, hs, bs = 2, 4, 64, 16
    q = torch.zeros(3, nq, hs, dtype=torch.bfloat16)
    pth = mode.is_per_token_head
    width = hs + (4 if pth else 0)
    kv = torch.zeros(8, bs, nkv, width, dtype=torch.int8 if pth else torch.bfloat16)
    scale = torch.ones(8, bs, nkv) if pth else None
    one = torch.ones(1)
    ua.unified_attention(
        q=q,
        k=kv[..., :hs],
        v=kv[..., :hs],
        out=torch.empty_like(q),
        cu_seqlens_q=torch.tensor([0, 3], dtype=torch.int32),
        max_seqlen_q=3,
        seqused_k=torch.tensor([3], dtype=torch.int32),
        max_seqlen_k=3,
        softmax_scale=0.125,
        causal=True,
        alibi_slopes=None,
        window_size=(-1, -1),
        block_table=torch.zeros(1, 1, dtype=torch.int32),
        softcap=0,
        q_descale=None,
        k_descale=one.expand(1, nkv),
        v_descale=one.expand(1, nkv),
        kv_quant_mode=mode,
        k_scale_cache=scale,
        v_scale_cache=scale,
        use_td=use_td,
    )
    return fake.kwargs


@pytest.mark.parametrize("guard", [None, "0"])
def test_td_guard_per_token_head(monkeypatch, guard):
    if guard is None:
        monkeypatch.delenv("B70_PTH_TD_GUARD", raising=False)
    else:
        monkeypatch.setenv("B70_PTH_TD_GUARD", guard)
    kw = _call(monkeypatch, KVQuantMode.INT8_PER_TOKEN_HEAD, use_td=True)
    assert kw["USE_TD"] is (guard == "0")


def test_td_untouched_without_per_token_head(monkeypatch):
    monkeypatch.delenv("B70_PTH_TD_GUARD", raising=False)
    kw = _call(monkeypatch, KVQuantMode.NONE, use_td=True)
    assert kw["USE_TD"] is True
