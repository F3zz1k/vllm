# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B70-QSA-K: Xe2-shaped sparse paged GQA attention for Qwen4Exp QSA (Intel XPU).

Selected with ``B70_QSA_ATTN_IMPL=attn2`` (default ``splitk`` = the original
``_qsa_sparse_paged_gqa_splitk_kernel``). Same numerical contract as the
original: BF16 (or fp8-e4m3) K/V, BF16 Q, FP32 scores / softmax /
accumulation, FP32 partials merged by log-sum-exp, BF16 rounding of the
attention output before the FP32 sigmoid gate.

Why a second kernel (measured in the Triton tuning drill): the original holds a whole ``[BLOCK_M, HEAD_DIM]`` FP32 accumulator
plus gathered K and V tiles in one program. At head_dim 256 that alone is the
whole 256-GRF file of a Xe2 hardware thread, so every config spills 11-46 KB
per thread. Adding warps does not help: Triton's Intel backend lays a chained
(Q.K -> P.V) dot's warps along M, and M is only the 6 (padded 8) heads of one
KV group, so extra warps replicate the same work. The SYCL kernels at the bar
(analytics-zoo vllm-xpu-kernels ``token_split_attention_v3``) instead keep
each subgroup's live state small and get their parallelism from many short
token partials merged afterwards.

attn2 does the same within Triton:

1. **One program = one row x one KV group x TILES_PER_PROG tiles of BLOCK_N
   selected tokens, one subgroup (num_warps 1).** All 6 heads of the group are
   the dot's M (padded to 8), so each gathered K/V row serves every head.
2. **Each tile is finished in place**: Q.K over head_dim chunks (a runtime loop,
   so only one Q chunk and one K chunk are ever live), an exact softmax over the
   tile's BLOCK_N scores, then P.V chunk by chunk with each ``[BM, DC]`` output
   chunk stored as soon as it is computed. No full-head_dim accumulator exists.
3. **Each tile writes an FP32 partial** (normalised output + log2-sum-exp, the
   layout the original split-K uses), and ``_qsa_attn2_merge_kernel`` combines a
   row's tiles with an online log-sum-exp merge, rounds to BF16 and applies the
   gate. The merge reads the row's valid count, so tiles past it are never read.
4. **Rows go in chunks** (``row_chunk``) so the partial workspace stays bounded
   (prefill default: 256 rows x 6 heads x 33 tiles x 1 KB = 51 MB at BLOCK_N 64;
   decode 64 rows x 6 x 65 x 1 KB = 25 MB at BLOCK_N 32). It is a transient
   torch.empty outside the engine's profiled peak, like the original's
   partials.
5. Every dot is M 8, N = BLOCK_N or DC >= 16, K = DC or BLOCK_N >= 16: DPAS
   eligible and legal under Triton 3.8.0+xpu's N/K >= 16 minimum.

Launch choices go through ``B70_QSA_TUNE`` keys ``sparse_attn2_decode`` and
``sparse_attn2_prefill`` (fields in qsa_xpu_tuning); the merge reads ``merge``.
"""

from __future__ import annotations

import os

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

from . import qsa_xpu_tuning as xpu_tune

logger = init_logger(__name__)

ATTN_IMPL_ENV = "B70_QSA_ATTN_IMPL"
_IMPLS = ("splitk", "attn2")


def attn_impl() -> str:
    """Selected sparse attention implementation (read per call; host-side only)."""
    impl = os.environ.get(ATTN_IMPL_ENV, "splitk").strip() or "splitk"
    if impl not in _IMPLS:
        raise ValueError(f"{ATTN_IMPL_ENV}={impl!r}; expected one of {_IMPLS}")
    logger.info_once("B70-QSA: sparse attention implementation = %s", impl)
    return impl


I8DOT_ENV = "B70_QSA_I8DOT"
_I8DOT_MODES = {
    "0": 0,
    "1": 1,
    "2": 2,
}
_I8DOT_DESC = {
    1: "Q.K int8 x int8 -> int32 (DPAS), P.V bf16",
    2: "Q.K and P.V int8 x int8 -> int32 (DPAS)",
}


def i8dot_mode() -> int:
    """B70-IP-K7: native int8 attention over an int8 per-token-head QSA cache.

    ``B70_QSA_I8DOT`` = 0 (default, IP-F: int8 K/V widened to bf16 for the
    dots), 1 (Q quantized per (token, q head) to int8; Q.K is an int8 x int8
    -> int32 dot; P.V stays bf16), 2 (1 plus P.V as int8 x int8 -> int32:
    P * v_scale quantized per row of the tile). Only int8_per_token_head
    caches take it; bf16/fp8 caches ignore it. Read per call (host side).
    """
    raw = os.environ.get(I8DOT_ENV, "0").strip() or "0"
    if raw not in _I8DOT_MODES:
        raise ValueError(f"{I8DOT_ENV}={raw!r}; expected one of {tuple(_I8DOT_MODES)}")
    mode = _I8DOT_MODES[raw]
    if mode:
        logger.info_once(
            "B70-IP-K7: QSA native int8 attention %s=%d: %s "
            "(int8_per_token_head KV only)",
            I8DOT_ENV,
            mode,
            _I8DOT_DESC[mode],
        )
    return mode


@triton.jit
def _qsa_quant_q_kernel(
    q_ptr,
    q_i8_ptr,
    q_scale_ptr,
    stride_q_row,
    stride_q_head,
    NUM_QUERY_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
) -> None:
    """B70-IP-K7: symmetric int8 Q, one fp32 scale per (token, q head).

    Same rounding as the KV writer: scale = absmax / 127, x * (1 / scale),
    half away from zero, clamp to [-127, 127]. Output is contiguous
    [rows, heads, head_dim] int8 and [rows, heads] fp32.
    """
    row = tl.program_id(0)
    head = tl.program_id(1)
    dims = tl.arange(0, HEAD_DIM)
    x = tl.load(q_ptr + row * stride_q_row + head * stride_q_head + dims).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), axis=0) / 127.0, 1.0e-10)
    y = x * (1.0 / scale)
    y = tl.where(y >= 0, tl.floor(y + 0.5), tl.ceil(y - 0.5))
    y = tl.minimum(tl.maximum(y, -127.0), 127.0)
    out_row = row.to(tl.int64) * NUM_QUERY_HEADS + head
    tl.store(q_i8_ptr + out_row * HEAD_DIM + dims, y.to(tl.int8))
    tl.store(q_scale_ptr + out_row, scale)


def quantize_q_i8(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """B70-IP-K7: (int8 Q [rows, heads, head_dim], fp32 scale [rows, heads])."""
    rows, heads, head_dim = q.shape
    assert q.stride(2) == 1
    q_i8 = torch.empty((rows, heads, head_dim), dtype=torch.int8, device=q.device)
    q_scale = torch.empty((rows, heads), dtype=torch.float32, device=q.device)
    if rows:
        _qsa_quant_q_kernel[(rows, heads)](
            q,
            q_i8,
            q_scale,
            q.stride(0),
            q.stride(1),
            NUM_QUERY_HEADS=heads,
            HEAD_DIM=head_dim,
            num_warps=1,
        )
    return q_i8, q_scale


def i8dot_align_kwargs(
    k_cache: torch.Tensor | object, v_cache: torch.Tensor | object, head_dim: int
) -> dict:
    """B70-IP-K8 / B70-IP-PAD16: the aligned-gather kwargs of an int8-dot attn2
    launch. ALIGN8 is True when the int8 K/V views are the inline per-token-head
    cell (V = K + one cache half, same strides, half >= head_dim + 4) with
    K rows aligned to KV_ALIGN bytes (16 when the cell, its strides and its V
    offset are all 16-B multiples, as with vllm#62113's padded halves; else 8).
    V_OFF is the V offset in bytes. Works on warmup stand-ins (their data_ptr
    is 0 or 1, so the V offset check fails and ALIGN8 stays off)."""
    ks, vs = k_cache.stride(), v_cache.stride()
    v_off = v_cache.data_ptr() - k_cache.data_ptr()
    if not (
        tuple(ks) == tuple(vs)
        and head_dim + 4 <= v_off < head_dim + 4 + 16
        and v_off % 4 == 0
    ):
        return {"ALIGN8": False}
    for align in (16, 8):
        if k_cache.data_ptr() % align == 0 and all(s % align == 0 for s in ks[:3]):
            return {"ALIGN8": True, "KV_ALIGN": align, "V_OFF": v_off}
    return {"ALIGN8": False}


def i8dot_align8(
    k_cache: torch.Tensor | object, v_cache: torch.Tensor | object, head_dim: int
) -> bool:
    """B70-IP-K8: True when the int8-dot gathers may assume aligned cells."""
    return i8dot_align_kwargs(k_cache, v_cache, head_dim)["ALIGN8"]


def i8dot_kwargs(q_scale, mode: int) -> dict:
    """B70-IP-K7: the trailing kernel kwargs of an int8-dot launch ({} when off)."""
    if not mode:
        return {}
    return {
        "q_scale_ptr": q_scale,
        "stride_qs_row": q_scale.stride(0),
        "I8DOT": mode,
    }


@triton.jit
def _attn2_dequant(tile):
    # Same decode as qsa._maybe_dequant: e4m3 bytes -> fp16 of x * 2^-8 (the
    # host folds 2^8 into the scales); e4m3 view -> fp16; anything else as is.
    if tile.dtype == tl.uint8:
        code = tile.to(tl.uint16)
        bits = ((code & 0x80) << 8) | ((code & 0x7F) << 7)
        return bits.to(tl.float16, bitcast=True)
    elif tile.dtype == tl.float8e4nv:
        return tile.to(tl.float16)
    else:
        return tile


@triton.jit(do_not_specialize=["num_rows", "num_requests"])
def _qsa_attn2_tile_kernel(
    q_ptr,
    k_cache_ptr,
    v_cache_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_req_ptr,
    partial_output_ptr,
    partial_lse_ptr,
    softmax_scale,
    output_scale,
    stride_q_row,
    stride_q_head,
    stride_k_block,
    stride_k_token,
    stride_k_head,
    stride_v_block,
    stride_v_token,
    stride_v_head,
    stride_indices_row,
    stride_table_req,
    num_rows,
    num_cache_blocks,
    num_requests,
    TOPK: tl.constexpr,
    PAGE_SIZE: tl.constexpr,
    PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    NUM_DCHUNK: tl.constexpr,
    TILES_PER_PROG: tl.constexpr,
    k_scale_ptr=None,
    v_scale_ptr=None,
    stride_ks_block=0,
    stride_ks_token=0,
    stride_ks_head=0,
    stride_vs_block=0,
    stride_vs_token=0,
    stride_vs_head=0,
    KV_PTH: tl.constexpr = False,
    q_scale_ptr=None,
    stride_qs_row=0,
    I8DOT: tl.constexpr = 0,
    ALIGN8: tl.constexpr = False,
    KV_ALIGN: tl.constexpr = 8,
    V_OFF: tl.constexpr = 0,
) -> None:
    # B70-IP-K8: ALIGN8 (I8DOT on the inline int8 per-token-head cell, checked on
    # the host): Triton cannot prove the alignment of a gathered row and emits
    # byte loads. Every cell offset is a multiple of KV_ALIGN, and V sits V_OFF
    # bytes after K in the same cell; saying so (multiple_of KV_ALIGN, V
    # addressed from the K pointer) lets the gathers vectorize. B70-IP-PAD16:
    # with vllm#62113's 16-B halves (544-B cell, V_OFF 272 at head_dim 256)
    # KV_ALIGN is 16 for K and V; the tight 520-B cell gives 8-B K, 4-B V.
    # B70-IP-K7: I8DOT (needs KV_PTH) = q_ptr is int8 Q with one fp32 scale per
    # (token, q head) at q_scale_ptr. 1: S = dot(q_i8, k_i8) in int32 (DPAS
    # int8, no widening), scores = S * q_scale * k_scale in fp32. 2: also
    # P.V in int8: P' = P * v_scale per column, quantized per head row of the
    # tile (scale = rowmax / 127), dot(p_i8, v_i8) in int32, times the scale.
    # B70-IP-K6: KV_PTH = int8 K/V with one fp32 scale per (token, kv head)
    # (int8_per_token_head). The int8 rows are cast to bf16 (exact) for the
    # dots; the K scale multiplies the FP32 scores column-wise after Q.K, the
    # V scale multiplies P column-wise before P.V (fp32, then bf16), as
    # TRITON_ATTN does. Pointer loads only (B70-IP-K1). Defaults keep the
    # bf16/fp8 launches unchanged.
    DC: tl.constexpr = HEAD_DIM // NUM_DCHUNK
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    prog = tl.program_id(2)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    request_ok = (request >= 0) & (request < num_requests)
    # Trailing count column of the packed selection buffer (see qsa.py).
    valid_count = tl.load(indices_ptr + row * stride_indices_row + TOPK)
    num_tiles = tl.cdiv(tl.minimum(valid_count, TOPK), BLOCK_N)

    head_offsets = tl.arange(0, BLOCK_M)
    head_mask = head_offsets < GROUP_SIZE
    dc_offsets = tl.arange(0, DC)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    q_rows = (
        q_ptr
        + row * stride_q_row
        + (first_head + head_offsets[:, None]) * stride_q_head
    )
    score_scale = softmax_scale * 1.4426950408889634
    if I8DOT >= 1:
        q_token_scale = tl.load(
            q_scale_ptr + row * stride_qs_row + first_head + head_offsets,
            mask=head_mask,
            other=0.0,
        )

    for t in range(TILES_PER_PROG):
        tile = prog * TILES_PER_PROG + t
        if tile < num_tiles:
            columns = tile * BLOCK_N + column_offsets
            logical_token = tl.load(
                indices_ptr + row * stride_indices_row + columns,
                mask=columns < TOPK,
                other=-1,
            )
            safe_token = tl.maximum(logical_token, 0)
            logical_page = safe_token // PAGE_SIZE
            page_offset = safe_token % PAGE_SIZE
            valid = (
                request_ok & (logical_token >= 0) & (logical_page < PAGE_TABLE_WIDTH)
            )
            physical_page = tl.load(
                block_table_ptr
                + safe_request * stride_table_req
                + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
                mask=valid,
                other=-1,
            )
            valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
            safe_page = tl.maximum(physical_page, 0).to(tl.int64)
            # One int64 base per token; elements add a contiguous dim offset.
            k_base = (
                k_cache_ptr
                + safe_page * stride_k_block
                + page_offset * stride_k_token
                + kv_head * stride_k_head
            )
            v_base = (
                v_cache_ptr
                + safe_page * stride_v_block
                + page_offset * stride_v_token
                + kv_head * stride_v_head
            )
            if ALIGN8:
                cell_offset = tl.multiple_of(
                    safe_page * stride_k_block
                    + page_offset * stride_k_token
                    + kv_head * stride_k_head,
                    KV_ALIGN,
                )
                k_base = k_cache_ptr + cell_offset
                v_base = k_cache_ptr + cell_offset + V_OFF

            # Q.K, head_dim chunk by chunk (runtime loop: one chunk live).
            if I8DOT >= 1:
                # B70-IP-K7: int8 x int8 -> int32, exact; scales after.
                acc_qk = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.int32)
                for c in range(NUM_DCHUNK):
                    q_chunk = tl.load(
                        q_rows + c * DC + dc_offsets[None, :],
                        mask=head_mask[:, None],
                        other=0,
                    )
                    k_chunk = tl.load(
                        k_base[None, :] + c * DC + dc_offsets[:, None],
                        mask=valid[None, :],
                        other=0,
                    )
                    acc_qk = tl.dot(q_chunk, k_chunk, acc=acc_qk)
                k_token_scale = tl.load(
                    k_scale_ptr
                    + safe_page * stride_ks_block
                    + page_offset * stride_ks_token
                    + kv_head * stride_ks_head,
                    mask=valid,
                    other=0.0,
                )
                scores = acc_qk.to(tl.float32) * (
                    q_token_scale[:, None] * k_token_scale[None, :]
                )
            else:
                scores = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
                for c in range(NUM_DCHUNK):
                    q_chunk = tl.load(
                        q_rows + c * DC + dc_offsets[None, :],
                        mask=head_mask[:, None],
                        other=0.0,
                    )
                    k_chunk = tl.load(
                        k_base[None, :] + c * DC + dc_offsets[:, None],
                        mask=valid[None, :],
                        other=0.0,
                    )
                    scores = tl.dot(
                        q_chunk, _attn2_dequant(k_chunk).to(q_chunk.dtype), acc=scores
                    )
                if KV_PTH:
                    k_token_scale = tl.load(
                        k_scale_ptr
                        + safe_page * stride_ks_block
                        + page_offset * stride_ks_token
                        + kv_head * stride_ks_head,
                        mask=valid,
                        other=0.0,
                    )
                    scores = scores * k_token_scale[None, :]

            # Exact softmax over this tile.
            scores = tl.where(valid[None, :], scores * score_scale, -1.0e20)
            tile_max = tl.max(scores, axis=1)
            probabilities = tl.where(
                valid[None, :], tl.math.exp2(scores - tile_max[:, None]), 0.0
            )
            tile_sum = tl.sum(probabilities, axis=1)
            has_values = tile_sum > 0
            inv_sum = tl.where(
                has_values, output_scale / tl.maximum(tile_sum, 1.0e-20), 0.0
            )
            partial_row = (tile * num_rows + row).to(tl.int64)
            tile_lse = tl.where(
                has_values,
                tile_max + tl.math.log2(tl.maximum(tile_sum, 1.0e-20)),
                -float("inf"),
            )
            lse_rows = partial_lse_ptr + partial_row * NUM_QUERY_HEADS + first_head
            if I8DOT >= 1:
                # IGC 2.41.5 dies with SIGFPE (ZE_RESULT_ERROR_MODULE_BUILD_FAILURE)
                # building the 1-D store of this row vector when its parent
                # layout is the int8 DPAS (opsPerChan 4) one; the same values
                # stored as a [BM, DC] tile masked to column 0 build and run.
                tl.store(
                    lse_rows + head_offsets[:, None] + 0 * dc_offsets[None, :],
                    tile_lse[:, None] + tl.zeros((BLOCK_M, DC), dtype=tl.float32),
                    mask=head_mask[:, None] & (dc_offsets[None, :] == 0),
                )
            else:
                tl.store(lse_rows + head_offsets, tile_lse, mask=head_mask)
            part_rows = (
                partial_output_ptr
                + (partial_row * NUM_QUERY_HEADS + first_head + head_offsets[:, None])
                * HEAD_DIM
            )

            if KV_PTH:
                v_token_scale = tl.load(
                    v_scale_ptr
                    + safe_page * stride_vs_block
                    + page_offset * stride_vs_token
                    + kv_head * stride_vs_head,
                    mask=valid,
                    other=0.0,
                )
                if I8DOT == 2:
                    # B70-IP-K7: P' = P * v_scale >= 0; one scale per head row.
                    pv_scaled = probabilities * v_token_scale[None, :]
                    p_scale = tl.maximum(tl.max(pv_scaled, axis=1) / 127.0, 1.0e-30)
                    pv_weights = tl.minimum(
                        tl.floor(pv_scaled * (1.0 / p_scale)[:, None] + 0.5), 127.0
                    ).to(tl.int8)
                    inv_sum = inv_sum * p_scale
                else:
                    pv_weights = (probabilities * v_token_scale[None, :]).to(
                        tl.bfloat16
                    )
            else:
                pv_weights = probabilities

            # P.V, chunk by chunk; each [BM, DC] output chunk is stored at once.
            for c in range(NUM_DCHUNK):
                if I8DOT == 2:
                    v_chunk = tl.load(
                        v_base[:, None] + c * DC + dc_offsets[None, :],
                        mask=valid[:, None],
                        other=0,
                    )
                    out_chunk = tl.dot(pv_weights, v_chunk).to(tl.float32)
                else:
                    v_chunk = _attn2_dequant(
                        tl.load(
                            v_base[:, None] + c * DC + dc_offsets[None, :],
                            mask=valid[:, None],
                            other=0.0,
                        )
                    )
                    if KV_PTH:
                        v_chunk = v_chunk.to(tl.bfloat16)
                    out_chunk = tl.dot(pv_weights.to(v_chunk.dtype), v_chunk)
                tl.store(
                    part_rows + c * DC + dc_offsets[None, :],
                    out_chunk * inv_sum[:, None],
                    mask=head_mask[:, None],
                )


@triton.jit(do_not_specialize=["num_rows"])
def _qsa_attn2_merge_kernel(
    partial_output_ptr,
    partial_lse_ptr,
    indices_ptr,
    output_ptr,
    output_gate_ptr,
    stride_indices_row,
    stride_output_row,
    stride_output_head,
    stride_output_gate_row,
    stride_output_gate_head,
    num_rows,
    TOPK: tl.constexpr,
    BLOCK_N: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    NUM_QUERY_HEADS: tl.constexpr,
) -> None:
    """Online log-sum-exp merge of one (row, head)'s tile partials + gate."""
    row = tl.program_id(0)
    head = tl.program_id(1)
    valid_count = tl.load(indices_ptr + row * stride_indices_row + TOPK)
    num_tiles = tl.cdiv(tl.minimum(valid_count, TOPK), BLOCK_N)
    dim_offsets = tl.arange(0, HEAD_DIM)
    running_max = -float("inf")
    denominator = 0.0
    merged = tl.zeros((HEAD_DIM,), dtype=tl.float32)
    for tile in range(num_tiles):
        partial_row = (tile * num_rows + row).to(tl.int64)
        lse = tl.load(partial_lse_ptr + partial_row * NUM_QUERY_HEADS + head)
        if lse > -float("inf"):
            next_max = tl.maximum(running_max, lse)
            alpha = tl.math.exp2(running_max - next_max)
            weight = tl.math.exp2(lse - next_max)
            part = tl.load(
                partial_output_ptr
                + (partial_row * NUM_QUERY_HEADS + head) * HEAD_DIM
                + dim_offsets
            )
            merged = merged * alpha + part * weight
            denominator = denominator * alpha + weight
            running_max = next_max
    merged = tl.where(denominator > 0, merged / denominator, 0.0)
    # Original contract: BF16 attention output, then the FP32 sigmoid gate.
    merged = merged.to(output_ptr.dtype.element_ty).to(tl.float32)
    gate = tl.load(
        output_gate_ptr
        + row * stride_output_gate_row
        + head * stride_output_gate_head
        + dim_offsets
    ).to(tl.float32)
    tl.store(
        output_ptr + row * stride_output_row + head * stride_output_head + dim_offsets,
        merged * tl.sigmoid(gate),
    )


# --------------------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------------------
# Defaults: the fastest configs measured on a B70 (Triton 3.8.0+xpu, IGC 2.41.5,
# compute-runtime 26.35; run 2026-W43/2026-10-25-qsa-triton-xpu, B2a): BLOCK_N 32 x
# 8 head_dim chunks at 256 GRF, zero spill, for both regions. Decode 8 rows
# 54-56 us (rows 32: 84-202, rows 64: 191-372, 4K-131K); prefill 4,096 rows
# 9.3 ms at 32K, 12.6 ms at 131K. At 128 GRF the same tile spills 192 B and
# runs ~17 % slower, so grf_mode "256" is part of the default.
_DEFAULTS = {
    "sparse_attn2_decode": {
        "block_n": 32,
        "num_dchunk": 8,
        "tiles_per_prog": 1,
        "row_chunk": 4096,
        "num_warps": 1,
        "num_stages": 1,
    },
    "sparse_attn2_prefill": {
        "block_n": 32,
        "num_dchunk": 8,
        "tiles_per_prog": 1,
        "row_chunk": 512,
        "num_warps": 1,
        "num_stages": 1,
    },
}
# B70-IP-K8: the int8-dot launches (B70_QSA_I8DOT, aligned gathers). Measured on
# a B70 (card 2, TP2 12q/1kv, width 2051, 32K context, kernel-only, graph-timed
# decode): BLOCK_N 64 x 8 chunks (DC 32) with 2 subgroups and grf "auto":
# prefill 4096 rows 11.2-11.4 ms vs 15.0 ms for the bf16 default above, decode
# 64 rows 0.246 vs 0.30 ms. The same shape in bf16 does not compile (the IGC
# SIGFPE below); in int8 it does. Measured on a B70.
_DEFAULTS["sparse_attn2_i8_decode"] = {
    "block_n": 64,
    "num_dchunk": 8,
    "tiles_per_prog": 1,
    "row_chunk": 4096,
    "num_warps": 2,
    "num_stages": 1,
}
_DEFAULTS["sparse_attn2_i8_prefill"] = {
    "block_n": 64,
    "num_dchunk": 8,
    "tiles_per_prog": 1,
    "row_chunk": 512,
    "num_warps": 2,
    "num_stages": 1,
}
# B70_QSA_I8DOT=2 (int8 P.V too): BLOCK_N 128 with 4 subgroups. Measured: TP2
# prefill 11.0 ms (bf16 14.9), decode 64 rows 0.170 ms (bf16 0.307), decode 8
# rows 0.044 ms (bf16 0.091); TP1 prefill 22.1 ms (bf16 30.2).
_DEFAULTS["sparse_attn2_i8pv_decode"] = {
    **_DEFAULTS["sparse_attn2_i8_decode"],
    "block_n": 128,
    "num_warps": 4,
}
_DEFAULTS["sparse_attn2_i8pv_prefill"] = {
    **_DEFAULTS["sparse_attn2_i8_prefill"],
    "block_n": 128,
    "num_warps": 4,
}
_DEFAULT_GRF = "256"
_DEFAULT_GRF_I8 = "auto"
# Rows at/above which a use_prefill_config call takes the prefill region.
_PREFILL_ROWS = 2048


def _region(num_rows: int, use_prefill_config: bool, i8dot: int = 0) -> str:
    stem = {0: "sparse_attn2_", 1: "sparse_attn2_i8_", 2: "sparse_attn2_i8pv_"}[
        int(i8dot)
    ]
    if use_prefill_config and num_rows >= _PREFILL_ROWS:
        return stem + "prefill"
    return stem + "decode"


def select_attn2_config(
    num_rows: int,
    num_kv_heads: int,
    use_prefill_config: bool,
    selection_width: int,
    i8dot: int = 0,
) -> dict:
    """Launch config of one call: a pure function of shapes (graph-capture stable).

    i8dot (B70-IP-K8, the B70_QSA_I8DOT mode) selects the int8-dot regions
    (``sparse_attn2_i8_*`` for 1, ``sparse_attn2_i8pv_*`` for 2).
    """
    del num_kv_heads
    region = _region(num_rows, use_prefill_config, i8dot)

    def get(field):
        return xpu_tune.get(region, field, _DEFAULTS[region][field])

    block_n = get("block_n")
    num_dchunk = get("num_dchunk")
    tiles_per_prog = get("tiles_per_prog")
    num_tiles = triton.cdiv(selection_width, block_n)
    num_warps = get("num_warps")
    if num_warps > 1 and block_n >= 32 and not i8dot:
        # IGC 2.41.5 dies with SIGFPE ("Internal Compiler Error: Floating point
        # exception") compiling these, which kills the process (measured: every
        # BLOCK_N 32/64 config with num_warps 2, Triton 3.8.0 and Intel main).
        raise ValueError(
            f"B70_QSA attn2: num_warps {num_warps} with block_n {block_n} crashes "
            "IGC 2.41.5; use num_warps 1 (or block_n 16)"
        )
    extra = xpu_tune.launch_extra(region)
    if xpu_tune.get(region, "grf_mode", None) is None:
        extra = {"grf_mode": _DEFAULT_GRF_I8 if i8dot else _DEFAULT_GRF}
    return {
        "region": region,
        "block_n": block_n,
        "num_dchunk": num_dchunk,
        "tiles_per_prog": min(tiles_per_prog, num_tiles),
        "num_tiles": num_tiles,
        "row_chunk": max(1, min(get("row_chunk"), num_rows)),
        "num_warps": num_warps,
        "num_stages": get("num_stages"),
        "extra": extra,
    }


def pth_scale_kwargs(
    k_scale_cache: torch.Tensor | None, v_scale_cache: torch.Tensor | None
) -> dict:
    """B70-IP-K6: the per-token-head scale kwargs of a sparse attention launch.

    Empty for bf16/fp8 caches, so those launches keep the kernels' defaults.
    """
    if k_scale_cache is None:
        assert v_scale_cache is None
        return {}
    assert v_scale_cache is not None
    assert k_scale_cache.dtype == v_scale_cache.dtype == torch.float32
    assert len(k_scale_cache.shape) == len(v_scale_cache.shape) == 3
    return {
        "k_scale_ptr": k_scale_cache,
        "v_scale_ptr": v_scale_cache,
        "stride_ks_block": k_scale_cache.stride(0),
        "stride_ks_token": k_scale_cache.stride(1),
        "stride_ks_head": k_scale_cache.stride(2),
        "stride_vs_block": v_scale_cache.stride(0),
        "stride_vs_token": v_scale_cache.stride(1),
        "stride_vs_head": v_scale_cache.stride(2),
        "KV_PTH": True,
    }


def merge_kwargs() -> dict:
    return {
        "num_warps": xpu_tune.get("merge", "num_warps", 2),
        "num_stages": 1,
        **xpu_tune.launch_extra("merge"),
    }


def qsa_sparse_paged_attention_attn2(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    use_prefill_config: bool,
    out: torch.Tensor,
    output_gate_view: torch.Tensor,
    softmax_scale: float,
    output_scale: float,
    block_m: int,
    k_scale_cache: torch.Tensor | None = None,
    v_scale_cache: torch.Tensor | None = None,
    q_i8: torch.Tensor | None = None,
    q_scale: torch.Tensor | None = None,
    i8dot: int = 0,
) -> torch.Tensor:
    """Launch attn2; the caller (qsa.qsa_sparse_paged_attention) validated inputs.

    k_scale_cache/v_scale_cache (B70-IP-K6): fp32 [num_blocks, block_size,
    num_kv_heads] views of an int8 per-token-head cache, else None.
    q_i8/q_scale/i8dot (B70-IP-K7): int8 Q and its per-(token, head) scales
    from quantize_q_i8, and the B70_QSA_I8DOT mode; 0 = bf16 Q.
    """
    pth = pth_scale_kwargs(k_scale_cache, v_scale_cache)
    if i8dot:
        assert pth and q_i8 is not None and q_scale is not None
        assert q_i8.shape == q.shape and q_i8.is_contiguous()
    head_dim = q.shape[2]
    num_kv_heads = k_cache.shape[2]
    group_size = q.shape[1] // num_kv_heads
    selection_width = logical_indices.shape[1] - 1
    cfg = select_attn2_config(
        q.shape[0], num_kv_heads, use_prefill_config, selection_width, i8dot
    )
    align8 = i8dot_align_kwargs(k_cache, v_cache, head_dim) if i8dot else {}
    q_src = q_i8 if i8dot else q
    dc = head_dim // cfg["num_dchunk"]
    assert dc >= 16 and dc * cfg["num_dchunk"] == head_dim, "attn2 head_dim chunking"
    # int8 DPAS consumes K in steps of 32 (opsPerChan 4 x systolic depth 8).
    assert not i8dot or dc >= 32, "B70_QSA_I8DOT needs head_dim chunks >= 32"
    rows_total = q.shape[0]
    row_chunk = cfg["row_chunk"]
    num_tiles = cfg["num_tiles"]
    partial_output = torch.empty(
        (num_tiles, row_chunk, q.shape[1], head_dim),
        dtype=torch.float32,
        device=q.device,
    )
    partial_lse = torch.empty(
        (num_tiles, row_chunk, q.shape[1]), dtype=torch.float32, device=q.device
    )
    progs = triton.cdiv(num_tiles, cfg["tiles_per_prog"])
    for r0 in range(0, rows_total, row_chunk):
        rows = min(row_chunk, rows_total - r0)
        q_c = q_i8[r0 : r0 + rows] if i8dot else q[r0 : r0 + rows]
        i8_kw = i8dot_kwargs(q_scale[r0 : r0 + rows] if i8dot else None, i8dot)
        ind_c = logical_indices[r0 : r0 + rows]
        req_c = token_to_req[r0 : r0 + rows]
        _qsa_attn2_tile_kernel[(rows, num_kv_heads, progs)](
            q_c,
            k_cache,
            v_cache,
            ind_c,
            block_table,
            req_c,
            partial_output,
            partial_lse,
            softmax_scale,
            output_scale,
            q_src.stride(0),
            q_src.stride(1),
            k_cache.stride(0),
            k_cache.stride(1),
            k_cache.stride(2),
            v_cache.stride(0),
            v_cache.stride(1),
            v_cache.stride(2),
            logical_indices.stride(0),
            block_table.stride(0),
            rows,
            k_cache.shape[0],
            block_table.shape[0],
            TOPK=selection_width,
            PAGE_SIZE=k_cache.shape[1],
            PAGE_TABLE_WIDTH=block_table.shape[1],
            GROUP_SIZE=group_size,
            HEAD_DIM=head_dim,
            NUM_QUERY_HEADS=q.shape[1],
            BLOCK_M=block_m,
            BLOCK_N=cfg["block_n"],
            NUM_DCHUNK=cfg["num_dchunk"],
            TILES_PER_PROG=cfg["tiles_per_prog"],
            num_warps=cfg["num_warps"],
            num_stages=cfg["num_stages"],
            **cfg["extra"],
            **pth,
            **i8_kw,
            **align8,
        )
        _qsa_attn2_merge_kernel[(rows, q.shape[1])](
            partial_output,
            partial_lse,
            ind_c,
            out[r0 : r0 + rows],
            output_gate_view[r0 : r0 + rows],
            logical_indices.stride(0),
            out.stride(0),
            out.stride(1),
            output_gate_view.stride(0),
            output_gate_view.stride(1),
            rows,
            TOPK=selection_width,
            BLOCK_N=cfg["block_n"],
            HEAD_DIM=head_dim,
            NUM_QUERY_HEADS=q.shape[1],
            **merge_kwargs(),
        )
    return out


def attn2_profiles(
    num_kv_heads: int, selection_width: int, max_rows: int = 8192, i8dot: int = 0
) -> set[tuple]:
    """Every (block_n, num_dchunk, tiles_per_prog, warps, stages, extra) in use."""
    out = set()
    for num_rows in range(1, max_rows + 1):
        for use_prefill_config in (False, True):
            c = select_attn2_config(
                num_rows, num_kv_heads, use_prefill_config, selection_width, i8dot
            )
            out.add(
                (
                    c["block_n"],
                    c["num_dchunk"],
                    c["tiles_per_prog"],
                    c["num_warps"],
                    c["num_stages"],
                    tuple(sorted(c["extra"].items())),
                )
            )
    return out
