# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B70-IP-K4: prefill over an int8 per-token-head KV cache through the XPU FA kernel.

On XPU the Triton unified attention kernel is ~100x slower than the SYCL
flash-attention kernel for prefill chunks (B70, hs=256: 4096 queries over 32K
context 2,705-3,034 ms vs 28.9 ms). Dequantizing the int8 context to bf16 costs
<1 ms per layer at 128K, so for batches that carry a real prefill we gather the
blocks each sequence references, dequantize them into a one-layer bf16 scratch,
and call ``flash_attn_varlen_func`` over a remapped block table. Decode-only
batches (and spec-decode verify batches, q <= B70_PTH_FA_MIN_Q) stay on the
Triton kernel and its CUDA graphs.

The scratch holds max_model_len tokens of one layer (K and V, bf16), shared by
all layers (they run in sequence), allocated in the first eager forward (the
memory-profile run) so the KV pool is sized around it. Sequences whose total
context exceeds it are processed in groups.

Env: B70_PTH_FA_PREFILL=1 turns it on (default off: it halved 16-parallel prefill throughput on the Flash-Next engine and cost KV memory per card; on a single-card 27B it cost ~2.1 GiB and could block boot); B70_PTH_FA_MIN_Q (default 64).

B70-IP-D2 (``B70_PTH_FA_DECODE=1``, default off): FLASH_ATTN on XPU accepts
``int8_per_token_head`` and decodes natively with the vllm-xpu-kernels INT8
per-token-head paged decode (wiesion, intel/llm-scaler#744). Its layers keep the
same bytes per block as TRITON_ATTN's inline layout (page sizes, hybrid unify
and B70-IP-K3 unchanged) but arrange them as the kernel's block-local split
layout ``[int8 K|V for all (slot, head) | K scales | V scales]``, so every K/V
surface base is 64-byte aligned. Each layer's pages are only ever touched by its
own backend, so the inline layout of TRITON_ATTN / QSA layers (drafter, QSA) is
untouched. Batches with max q <= the kernels' spec-decode bound run the native
decode (causal rows through the kernels' per-position split-K route,
non-causal rows as one decode row each); everything else goes through the K4
dequant-to-bf16 scratch, forced on (no B70_PTH_FA_MIN_Q threshold).
"""

import os

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_scratch: dict = {}
_logged = False


def _enabled() -> bool:
    return (
        current_platform.is_xpu() and os.environ.get("B70_PTH_FA_PREFILL", "0") == "1"
    )


def fa_decode_enabled() -> bool:
    """B70-IP-D2 gate: FLASH_ATTN serves int8_per_token_head on XPU."""
    return current_platform.is_xpu() and os.environ.get("B70_PTH_FA_DECODE", "0") == "1"


def _get_scratch(block_size, nkv, hs, dtype, device, max_len):
    key = (block_size, nkv, hs, dtype, str(device))
    s = _scratch.get(key)
    if s is None:
        nblk = -(-max_len // block_size) + 1
        k = torch.empty((nblk, block_size, nkv, hs), dtype=dtype, device=device)
        v = torch.empty_like(k)
        s = (k, v)
        _scratch[key] = s
        logger.info(
            "B70-IP-K4 int8 KV prefill via FA: bf16 scratch %d blocks x %d tokens "
            "(%.2f GiB)",
            nblk,
            block_size,
            2 * k.numel() * k.element_size() / 2**30,
        )
    return s


def _max_len(impl) -> int:
    # Captured in TritonAttentionImpl.__init__ (the config is not current at forward).
    v = getattr(impl, "_b70_max_model_len", None)
    return int(v) if v else 131072


def ensure_scratch(impl, kv_cache: torch.Tensor, dtype, force: bool = False) -> None:
    """Allocate at the first forward (profile run) so KV sizing sees it."""
    if not (force or _enabled()):
        return
    num_blocks, nkv, block_size, content = kv_cache.shape
    hs = impl.head_size
    _get_scratch(block_size, nkv, hs, dtype, kv_cache.device, _max_len(impl))


def _dequant_blocks(src_data, src_scale, ids, dst, hs, chunk=4):
    # src_data: (nb, bs, nkv, >= hs) int8 view (an inline half, padded or not,
    # or a D2 split view); src_scale: (nb, bs, nkv) fp32. Only [:hs] is read.
    n = ids.numel()
    for i in range(0, n, chunk):
        sel = ids[i : i + chunk]
        d = src_data[sel][..., :hs].to(dst.dtype)
        s = src_scale[sel].to(dst.dtype).unsqueeze(-1)
        dst[i : i + sel.numel()] = d * s


def _dequant_fp8_blocks(src, scale, ids, dst, chunk=4):
    # src: (nb, bs, nkv, hs) fp8 view; scale: per-tensor fp32 scalar tensor
    sc = scale.reshape(-1)[0].to(dst.dtype)
    n = ids.numel()
    for i in range(0, n, chunk):
        sel = ids[i : i + chunk]
        dst[i : i + sel.numel()] = src[sel].to(dst.dtype) * sc


# ---------------------------------------------------------------------------
# B70-V7-K4F (B70_PTH_FA_PREFILL_FAST=1): the K4 dequant without host reads
# ---------------------------------------------------------------------------
_k4f_logged = False
_k4f_fallback_logged = False


def k4_fast_enabled() -> bool:
    return os.environ.get("B70_PTH_FA_PREFILL_FAST", "0") == "1"


def attach_host_lens(attn_metadata, common_attn_metadata) -> None:
    """Builders call this so the K4F grouping reads host copies the scheduler
    already has (query_start_loc_cpu exact; seq_lens_cpu_upper_bound, exact
    for prefill rows and an upper bound for spec-decode rows) instead of
    ``.tolist()`` on device tensors."""
    if not k4_fast_enabled():
        return
    attn_metadata._b70_qsl_cpu = getattr(
        common_attn_metadata, "query_start_loc_cpu", None
    )
    attn_metadata._b70_seq_lens_ub_cpu = getattr(
        common_attn_metadata, "seq_lens_cpu_upper_bound", None
    )


@triton.jit
def _rtne_bf16(x):
    # fp32 -> bf16 bits, round to nearest even (torch's rounding for finite
    # values), done on the integer bits so every backend rounds alike.
    u = x.to(tl.uint32, bitcast=True)
    u = u + 0x7FFF + ((u >> 16) & 1)
    return u >> 16


@triton.jit
def _k4f_dequant(
    k_ptr,
    v_ptr,
    ks_ptr,
    vs_ptr,
    bt_ptr,
    sl_ptr,
    ok_ptr,
    ov_ptr,
    s_kb,
    s_kn,
    s_kh,
    s_vb,
    s_vn,
    s_vh,
    s_ksb,
    s_ksn,
    s_ksh,
    s_vsb,
    s_vsn,
    s_vsh,
    s_bt,
    stride_blocks,
    BS: tl.constexpr,
    NKV: tl.constexpr,
    HS: tl.constexpr,
    TOK: tl.constexpr,
    NKV_P2: tl.constexpr,
    HS_P2: tl.constexpr,
    OUT_BF16: tl.constexpr,
):
    r = tl.program_id(0)
    c = tl.program_id(1)
    t = tl.program_id(2)
    L = tl.load(sl_ptr + r)
    nblk = (L + BS - 1) // BS
    if c >= nblk:
        return
    src = tl.load(bt_ptr + r * s_bt + c).to(tl.int64)
    dst = (r * stride_blocks + c).to(tl.int64)
    tok = t * TOK + tl.arange(0, TOK)
    h = tl.arange(0, NKV_P2)
    d = tl.arange(0, HS_P2)
    m_t = tok < BS
    m3 = m_t[:, None, None] & (h < NKV)[None, :, None] & (d < HS)[None, None, :]
    m2 = m_t[:, None] & (h < NKV)[None, :]
    out_dt = ok_ptr.dtype.element_ty
    o_off = (
        (dst * BS + tok)[:, None, None] * (NKV * HS)
        + h[None, :, None] * HS
        + d[None, None, :]
    )
    # K
    a = tl.load(
        k_ptr
        + src * s_kb
        + tok[:, None, None] * s_kn
        + h[None, :, None] * s_kh
        + d[None, None, :],
        mask=m3,
        other=0,
    )
    sc = tl.load(
        ks_ptr + src * s_ksb + tok[:, None] * s_ksn + h[None, :] * s_ksh,
        mask=m2,
        other=0.0,
    )
    if OUT_BF16:
        sc = (_rtne_bf16(sc) << 16).to(tl.float32, bitcast=True)
        y = a.to(tl.float32) * sc[:, :, None]
        tl.store(ok_ptr + o_off, _rtne_bf16(y).to(tl.int16), mask=m3)
    else:
        sc = sc.to(out_dt).to(tl.float32)
        y = a.to(tl.float32) * sc[:, :, None]
        tl.store(ok_ptr + o_off, y.to(out_dt), mask=m3)
    # V
    a = tl.load(
        v_ptr
        + src * s_vb
        + tok[:, None, None] * s_vn
        + h[None, :, None] * s_vh
        + d[None, None, :],
        mask=m3,
        other=0,
    )
    sc = tl.load(
        vs_ptr + src * s_vsb + tok[:, None] * s_vsn + h[None, :] * s_vsh,
        mask=m2,
        other=0.0,
    )
    if OUT_BF16:
        sc = (_rtne_bf16(sc) << 16).to(tl.float32, bitcast=True)
        y = a.to(tl.float32) * sc[:, :, None]
        tl.store(ov_ptr + o_off, _rtne_bf16(y).to(tl.int16), mask=m3)
    else:
        sc = sc.to(out_dt).to(tl.float32)
        y = a.to(tl.float32) * sc[:, :, None]
        tl.store(ov_ptr + o_off, y.to(out_dt), mask=m3)


def _get_dequant_kernel():
    """One Triton launch per group dequantizes every referenced block of K and
    V: program (r, c, t) copies tokens [t*TOK, (t+1)*TOK) of block c of
    sequence r into scratch block r*stride + c. Numerics equal
    ``_dequant_blocks``: int8 -> bf16 is exact, the fp32 scale rounds once to
    the output dtype, and the product rounds once (bf16 x bf16 is exact in
    fp32); the bf16 roundings are done on the integer bits (RTNE)."""
    return _k4f_dequant


def dequant_group(
    key_cache, value_cache, k_sc, v_sc, bt, seq_lens, ks, vs, stride, width, hs
) -> None:
    """Dequantize the blocks of sequences ``bt.shape[0]`` (device block table
    rows and device seq_lens) into scratch blocks ``r*stride + c``."""
    import triton

    nb, bs, nkv, _ = key_cache.shape
    g = bt.shape[0]
    if g == 0 or width == 0:
        return
    TOK = 16 if hs * nkv <= 512 else 8
    kern = _get_dequant_kernel()
    bf16 = ks.dtype == torch.bfloat16
    kern[(g, width, triton.cdiv(bs, TOK))](
        key_cache,
        value_cache,
        k_sc,
        v_sc,
        bt,
        seq_lens,
        ks.view(torch.int16) if bf16 else ks,
        vs.view(torch.int16) if bf16 else vs,
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        k_sc.stride(0),
        k_sc.stride(1),
        k_sc.stride(2),
        v_sc.stride(0),
        v_sc.stride(1),
        v_sc.stride(2),
        bt.stride(0),
        stride,
        BS=bs,
        NKV=nkv,
        HS=hs,
        TOK=TOK,
        NKV_P2=triton.next_power_of_2(nkv),
        HS_P2=triton.next_power_of_2(hs),
        OUT_BF16=bf16,
    )


def _fa_prefill_fast(
    impl,
    query,
    key_cache,
    value_cache,
    output,
    attn_metadata,
    k_sc,
    v_sc,
    causal,
    sinks,
    window_size,
) -> bool | None:
    """K4 with no device-to-host reads: grouping from the builder's host
    copies, block table built on the device (a uniform per-group stride),
    one dequant launch and one FA call per group. Returns None when the host
    copies are missing (caller falls back to the stock loop)."""
    global _k4f_logged, _k4f_fallback_logged
    qsl_cpu = getattr(attn_metadata, "_b70_qsl_cpu", None)
    sl_cpu = getattr(attn_metadata, "_b70_seq_lens_ub_cpu", None)
    if qsl_cpu is None or sl_cpu is None:
        if not _k4f_fallback_logged:
            _k4f_fallback_logged = True
            logger.warning(
                "B70-V7-K4F: metadata has no host query_start_loc / seq_lens "
                "upper bound; K4 keeps the .tolist() loop"
            )
        return None
    from vllm._xpu_ops import xpu_ops

    nseq = attn_metadata.query_start_loc.numel() - 1
    qsl = [int(x) for x in qsl_cpu[: nseq + 1].tolist()]
    sl = [int(x) for x in sl_cpu[:nseq].tolist()]
    if len(qsl) != nseq + 1 or len(sl) != nseq:
        if not _k4f_fallback_logged:
            _k4f_fallback_logged = True
            logger.warning(
                "B70-V7-K4F: host lens cover %d/%d seqs; K4 keeps the loop",
                len(sl),
                nseq,
            )
        return None
    nb, bs, nkv, _ = key_cache.shape
    hs = impl.head_size
    ks, vs = _get_scratch(bs, nkv, hs, query.dtype, query.device, _max_len(impl))
    cap = ks.shape[0]
    nblks = [max(1, -(-s // bs)) for s in sl]
    dev = query.device
    i = 0
    groups = 0
    while i < nseq:
        j, stride = i, 0
        while j < nseq and (j - i + 1) * max(stride, nblks[j]) <= cap:
            stride = max(stride, nblks[j])
            j += 1
        if j == i:  # one sequence longer than the scratch: never (max_model_len)
            return False
        g = j - i
        bt = attn_metadata.block_table[i:j]
        seq_dev = attn_metadata.seq_lens[i:j]
        dequant_group(
            key_cache, value_cache, k_sc, v_sc, bt, seq_dev, ks, vs, stride, stride, hs
        )
        new_bt = (
            torch.arange(g, dtype=torch.int32, device=dev)[:, None] * stride
            + torch.arange(stride, dtype=torch.int32, device=dev)[None, :]
        )
        q0, q1 = qsl[i], qsl[j]
        cu = attn_metadata.query_start_loc[i : j + 1] - q0
        xpu_ops.flash_attn_varlen_func(
            query[q0:q1],
            ks,
            vs,
            cu_seqlens_q=cu,
            max_seqlen_q=max(qsl[r + 1] - qsl[r] for r in range(i, j)),
            max_seqlen_k=max(sl[i:j]),
            softmax_scale=impl.scale,
            causal=causal,
            out=output[q0:q1],
            block_table=new_bt,
            seqused_k=seq_dev,
            window_size=window_size,
            s_aux=sinks,
        )
        groups += 1
        i = j
    if not _k4f_logged:
        _k4f_logged = True
        logger.info(
            "B70-V7-K4F int8 KV prefill via FA bf16 without host reads: first "
            "batch %d seqs in %d groups, max q %d, max ctx (upper bound) %d, "
            "head_size %d, kv heads %d",
            nseq,
            groups,
            attn_metadata.max_query_len,
            max(sl) if sl else 0,
            hs,
            nkv,
        )
    return True


def maybe_fa_prefill(
    impl,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
    attn_metadata,
    num_actual_tokens: int,
    fp8_scales: tuple | None = None,
    pth_scales: tuple | None = None,
    force: bool = False,
    window_size: list[int] | None = None,
) -> bool:
    """int8 per-token-head (default) or, with fp8_scales=(k_scale, v_scale),
    an fp8 per-tensor cache (B70-IP-K5, FLASH_ATTN backend).

    pth_scales=(k_scale, v_scale) overrides the impl's inline scale views (the
    B70-IP-D2 split layout). force=True (B70-IP-D2) ignores B70_PTH_FA_PREFILL
    and B70_PTH_FA_MIN_Q, since the INT8 FA kernel has no prefill: it also
    accepts non-causal batches, sinks and the caller's window_size."""
    global _logged
    if not force:
        if not _enabled():
            return False
        if attn_metadata.max_query_len <= int(os.environ.get("B70_PTH_FA_MIN_Q", "64")):
            return False
    if (
        impl.alibi_slopes is not None
        or (getattr(impl, "logits_soft_cap", 0) or 0) != 0
        or getattr(attn_metadata, "mm_prefix_range_tensor", None) is not None
        or getattr(attn_metadata, "mm_prefix_query_range_tensor", None) is not None
        or getattr(attn_metadata, "rswa_prefix_lens", None) is not None
        or not isinstance(attn_metadata.causal, bool)
    ):
        return False
    if not force and (
        getattr(impl, "sinks", None) is not None
        or tuple(impl.sliding_window or (-1, -1)) != (-1, -1)
        or not attn_metadata.causal
    ):
        return False
    causal = attn_metadata.causal if force else True
    sinks = getattr(impl, "sinks", None) if force else None
    if not force:
        window_size = None
    if fp8_scales is None and k4_fast_enabled():
        k_sc, v_sc = pth_scales or (impl._k_scale_cache, impl._v_scale_cache)
        done = _fa_prefill_fast(
            impl,
            query,
            key_cache,
            value_cache,
            output,
            attn_metadata,
            k_sc,
            v_sc,
            causal,
            sinks,
            window_size,
        )
        if done is not None:
            return done
    from vllm._xpu_ops import xpu_ops

    nb, bs, nkv, _ = key_cache.shape
    hs = impl.head_size
    ks, vs = _get_scratch(bs, nkv, hs, query.dtype, query.device, _max_len(impl))
    cap = ks.shape[0]
    seq_lens = attn_metadata.seq_lens.tolist()
    qsl = attn_metadata.query_start_loc.tolist()
    nseq = len(seq_lens)
    bt = attn_metadata.block_table
    nblks = [-(-s // bs) for s in seq_lens]
    i = 0
    while i < nseq:
        j, used = i, 0
        while j < nseq and used + nblks[j] <= cap:
            used += nblks[j]
            j += 1
        if j == i:  # one sequence longer than the scratch (max_model_len): never
            return False
        ids = torch.cat([bt[r, : nblks[r]] for r in range(i, j)]).to(torch.long)
        if fp8_scales is None:
            k_sc, v_sc = pth_scales or (impl._k_scale_cache, impl._v_scale_cache)
            _dequant_blocks(key_cache, k_sc, ids, ks, hs)
            _dequant_blocks(value_cache, v_sc, ids, vs, hs)
        else:
            _dequant_fp8_blocks(key_cache, fp8_scales[0], ids, ks)
            _dequant_fp8_blocks(value_cache, fp8_scales[1], ids, vs)
        width = max(nblks[i:j])
        new_bt = torch.zeros((j - i, width), dtype=torch.int32, device=query.device)
        off = 0
        for r in range(i, j):
            new_bt[r - i, : nblks[r]] = torch.arange(
                off, off + nblks[r], dtype=torch.int32, device=query.device
            )
            off += nblks[r]
        q0, q1 = qsl[i], qsl[j]
        cu = torch.tensor(
            [q - q0 for q in qsl[i : j + 1]], dtype=torch.int32, device=query.device
        )
        sk = torch.tensor(seq_lens[i:j], dtype=torch.int32, device=query.device)
        xpu_ops.flash_attn_varlen_func(
            query[q0:q1],
            ks,
            vs,
            cu_seqlens_q=cu,
            max_seqlen_q=max(qsl[r + 1] - qsl[r] for r in range(i, j)),
            max_seqlen_k=max(seq_lens[i:j]),
            softmax_scale=impl.scale,
            causal=causal,
            out=output[q0:q1],
            block_table=new_bt,
            seqused_k=sk,
            window_size=window_size,
            s_aux=sinks,
        )
        i = j
    if not _logged:
        _logged = True
        logger.info(
            "B70-IP-K4/K5 %s KV prefill via FA bf16: first batch %d seqs, max q %d, "
            "max ctx %d",
            "int8" if fp8_scales is None else "fp8",
            nseq,
            attn_metadata.max_query_len,
            max(seq_lens),
        )
    return True


# ---------------------------------------------------------------------------
# B70-IP-D2: native INT8 per-token-head decode on the XPU FLASH_ATTN backend
# ---------------------------------------------------------------------------
_d2_logged = False


def log_fa_decode_once(head_size: int, num_kv_heads: int) -> None:
    global _d2_logged
    if not _d2_logged:
        _d2_logged = True
        logger.info(
            "B70-IP-D2 int8_per_token_head KV on XPU FLASH_ATTN: native INT8 paged "
            "decode (vllm-xpu-kernels, q <= %d per row), prefill via dequant to a "
            "bf16 scratch (K4), block-local split layout; head_size %d, kv heads %d",
            spec_decode_max_qlen(),
            head_size,
            num_kv_heads,
        )


def spec_decode_max_qlen() -> int:
    """Largest max_query_len that vllm-xpu-kernels runs through its split-K
    decode kernel (the only kernel that reads INT8 KV); 1 when the module or
    the bound is missing."""
    try:
        from vllm_xpu_kernels import flash_attn_interface as fai
    except ImportError:
        return 1
    return max(1, int(getattr(fai, "_SPEC_DECODE_MAX_QLEN", 1)))


def split_layout_views(
    kv_cache: torch.Tensor, num_kv_heads: int, head_size: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Block-local split views over one layer's int8 per-token-head cache.

    ``kv_cache`` is the logical ``(B, H, N, C)`` int8 layer view vLLM
    allocates for the spec (same bytes as TRITON_ATTN's inline layout: ``C =
    2 * half``, ``half`` = D + 4 padded to 16 B since vllm#62113, B70-IP-PAD16;
    the split layout uses ``H * N * (2D + 8)`` of the ``H * N * C`` bytes and
    leaves the rest of the page unused). Each
    block is re-read as ``[int8 K|V (N, H, 2D) | K scales fp32 (N, H) |
    V scales fp32 (N, H)]``. Returns key / value int8 ``[B, N, H, D]`` with
    strides ``(page, H*2D, 2D, 1)`` (value = key + D bytes) and k_scale /
    v_scale fp32 ``[B, N, H]`` with strides ``(page/4, H, 1)``: the contract of
    vllm-xpu-kernels ``check_int8_kv_cache``. ``page`` is the tensor's block
    stride, so padded pages work; the page itself must be one dense run of bytes
    (a block-compact KV cache layout, B outside H, N, C).
    """
    if kv_cache.dtype != torch.int8:
        raise ValueError(f"int8_per_token_head cache is int8, got {kv_cache.dtype}")
    num_blocks, nkv, block_size, content = kv_cache.shape
    from vllm.v1.attention.backends.triton_attn import pth_half_ok

    if nkv != num_kv_heads or content % 2 or not pth_half_ok(content // 2, head_size):
        raise ValueError(
            f"int8_per_token_head cache shape {tuple(kv_cache.shape)} does not match "
            f"num_kv_heads={num_kv_heads}, head_size={head_size}"
        )
    page_bytes = block_size * num_kv_heads * content
    span = 1 + sum((kv_cache.shape[d] - 1) * kv_cache.stride(d) for d in (1, 2, 3))
    if kv_cache.stride(3) != 1 or span != page_bytes:
        raise ValueError(
            "B70-IP-D2 needs a block-compact KV cache layout (each block's "
            f"(H, N, C) bytes dense); got strides {kv_cache.stride()} for shape "
            f"{tuple(kv_cache.shape)}"
        )
    page = kv_cache.stride(0) if num_blocks > 1 else page_bytes
    if page < page_bytes or page % 64 != 0 or head_size % 64 != 0:
        raise ValueError(
            "B70-IP-D2: the INT8 decode kernel needs 64-byte aligned K/V surfaces "
            f"(block stride {page} B, head_size {head_size}); both must be "
            "multiples of 64"
        )
    data_bytes = block_size * num_kv_heads * 2 * head_size
    scale_bytes = block_size * num_kv_heads * 4
    data = torch.as_strided(
        kv_cache,
        (num_blocks, block_size, num_kv_heads, 2 * head_size),
        (page, num_kv_heads * 2 * head_size, 2 * head_size, 1),
    )

    def scales(offset: int) -> torch.Tensor:
        raw = torch.as_strided(
            kv_cache,
            (num_blocks, scale_bytes),
            (page, 1),
            kv_cache.storage_offset() + offset,
        )
        # int8 [B, N*H*4] -> fp32 [B, N*H] (stride page/4, 1) -> [B, N, H]
        return raw.view(torch.float32).view(num_blocks, block_size, num_kv_heads)

    return (
        data[..., :head_size],
        data[..., head_size:],
        scales(data_bytes),
        scales(data_bytes + scale_bytes),
    )


def pth_split_views(impl, kv_cache: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """Cached split views for the bound cache. They live in ``_k_scale_cache``
    (a tuple here), which the runner clears after profiling, so no view keeps
    the profiling cache alive."""
    key = (kv_cache.data_ptr(), tuple(kv_cache.shape), tuple(kv_cache.stride()))
    cached = impl._k_scale_cache
    if not isinstance(cached, tuple) or cached[0] != key:
        views = split_layout_views(kv_cache, impl.num_kv_heads, impl.head_size)
        impl._k_scale_cache = cached = (key, views)
    return cached[1]


def _capturing() -> bool:
    fn = getattr(getattr(torch, "xpu", None), "is_current_stream_capturing", None)
    try:
        return bool(fn()) if fn is not None else False
    except Exception:  # noqa: BLE001
        return False


_dfa_swa_logged = False


def draft_swa_enabled() -> bool:
    """B70-V7-DFA: a non-causal sliding-window block (the DFlash drafter,
    window 2048) may take the INT8 decode route (B70_DRAFT_FA_INT8=1)."""
    return os.environ.get("B70_DRAFT_FA_INT8", "0") == "1"


def _noncausal_rows_decode(
    query,
    key_cache,
    value_cache,
    k_scale,
    v_scale,
    output,
    attn_metadata,
    impl,
    window_left: int = -1,
) -> None:
    """Non-causal small-q rows (DFlash draft blocks): every query row attends
    to its request's whole context, i.e. one q_len-1 decode row each. Graph-safe
    (no host sync); padded rows past query_start_loc[-1] read one slot of block 0
    and are zeroed.

    B70-V7-DFA: ``window_left >= 0`` (a symmetric non-causal window ``(w, w)``)
    gives every row the keys ``[seq_len - 1 - w, seq_len - 1]``, i.e. the
    window measured from the block's last row. TRITON_ATTN measures it from
    each row (``|q_pos - k_pos| <= w``), so a row ``j`` places before the block
    end also sees ``j`` older keys there. The two agree while
    ``seq_len <= w + 1``; past that each row ``j`` loses its ``j`` oldest keys
    (at most q_len - 1 = 7 of 2048 for the DFlash K=7 drafter). The decode
    kernel takes one scalar window, so the exact per-row edge is not
    expressible in one launch."""
    from vllm._xpu_ops import xpu_ops

    total = query.shape[0]
    qsl = attn_metadata.query_start_loc
    batch = qsl.numel() - 1
    cu = torch.arange(total + 1, dtype=torch.int32, device=query.device)
    pos = cu[:-1]
    req = torch.searchsorted(qsl[1:], pos, right=True).clamp_(max=batch - 1)
    is_pad = pos >= qsl[-1]
    used = (
        attn_metadata.seq_lens.index_select(0, req)
        .clamp(min=1)
        .masked_fill(is_pad, 1)
        .to(torch.int32)
    )
    bt = attn_metadata.block_table.index_select(0, req)
    bt.masked_fill_(is_pad[:, None], 0)
    xpu_ops.flash_attn_varlen_func(
        query,
        key_cache,
        value_cache,
        cu_seqlens_q=cu,
        max_seqlen_q=1,
        max_seqlen_k=attn_metadata.max_seq_len,
        softmax_scale=impl.scale,
        causal=True,
        out=output,
        block_table=bt,
        seqused_k=used,
        window_size=[window_left, 0] if window_left >= 0 else None,
        k_descale=k_scale,
        v_descale=v_scale,
        s_aux=impl.sinks,
    )
    output.masked_fill_(is_pad[:, None, None], 0)


def fa_int8_forward(
    impl,
    query: torch.Tensor,
    kv_cache: torch.Tensor,
    attn_metadata,
    output: torch.Tensor,
    num_actual_tokens: int,
    window: tuple[int, int] | None,
) -> torch.Tensor:
    """B70-IP-D2 forward for one FLASH_ATTN layer over an int8 per-token-head
    cache. ``window`` is the impl's window after _maybe_symmetrize_window."""
    from vllm._xpu_ops import xpu_ops

    causal = attn_metadata.causal
    if (
        not isinstance(causal, bool)
        or getattr(attn_metadata, "mm_prefix_query_range_tensor", None) is not None
        or getattr(attn_metadata, "rswa_prefix_lens", None) is not None
    ):
        raise NotImplementedError(
            "B70-IP-D2 int8_per_token_head FLASH_ATTN: per-request causal, "
            "mm_prefix and R-SWA are not supported"
        )
    # The prefill scratch is allocated in the first forward (the profile run),
    # so the KV pool is sized around it.
    ensure_scratch(impl, kv_cache, query.dtype, force=True)
    n = num_actual_tokens
    q, out = query[:n], output[:n]
    key_cache, value_cache, k_scale, v_scale = pth_split_views(impl, kv_cache)
    window_size = list(window) if window is not None and window[0] >= 0 else None
    max_q = attn_metadata.max_query_len

    if max_q <= spec_decode_max_qlen() and (causal or max_q == 1):
        # q 1: the INT8 paged decode. 1 < q <= bound, causal: the kernels'
        # spec-decode route (one decode row per (request, position), exact
        # per-position causal lengths; ragged q lengths included).
        xpu_ops.flash_attn_varlen_func(
            q,
            key_cache,
            value_cache,
            cu_seqlens_q=attn_metadata.query_start_loc,
            max_seqlen_q=max_q,
            max_seqlen_k=attn_metadata.max_seq_len,
            softmax_scale=impl.scale,
            causal=True,
            out=out,
            block_table=attn_metadata.block_table,
            seqused_k=attn_metadata.seq_lens,
            window_size=window_size,
            k_descale=k_scale,
            v_descale=v_scale,
            s_aux=impl.sinks,
        )
        return output
    if max_q <= spec_decode_max_qlen() and window_size is None:
        _noncausal_rows_decode(
            q, key_cache, value_cache, k_scale, v_scale, out, attn_metadata, impl
        )
        return output
    if (
        max_q <= spec_decode_max_qlen()
        and window_size is not None
        and window_size[0] == window_size[1]
        and draft_swa_enabled()
    ):
        # B70-V7-DFA: the DFlash drafter's non-causal sliding window.
        global _dfa_swa_logged
        if not _dfa_swa_logged:
            _dfa_swa_logged = True
            logger.info(
                "B70-V7-DFA non-causal sliding-window rows on the native INT8 "
                "decode: window %d from the block end, max q %d, head_size %d, "
                "kv heads %d",
                window_size[0] + 1,
                max_q,
                impl.head_size,
                impl.num_kv_heads,
            )
        _noncausal_rows_decode(
            q,
            key_cache,
            value_cache,
            k_scale,
            v_scale,
            out,
            attn_metadata,
            impl,
            window_left=window_size[0],
        )
        return output

    # Prefill or mixed batch: dequantize the referenced blocks into the K4
    # scratch and run FA bf16 (the INT8 kernel has no prefill path).
    if _capturing():
        raise RuntimeError(
            f"B70-IP-D2: a batch with max q {max_q} was captured into a full "
            "graph; the int8 prefill path syncs with the host. Use "
            "FULL_AND_PIECEWISE (full graphs for uniform decode only)."
        )
    if not maybe_fa_prefill(
        impl,
        q,
        key_cache,
        value_cache,
        out,
        attn_metadata,
        n,
        pth_scales=(k_scale, v_scale),
        force=True,
        window_size=window_size,
    ):
        raise NotImplementedError(
            "B70-IP-D2 int8_per_token_head FLASH_ATTN: this batch is neither a "
            "decode nor a dequantizable prefill (ALiBi / soft-cap / mm_prefix)"
        )
    return output
