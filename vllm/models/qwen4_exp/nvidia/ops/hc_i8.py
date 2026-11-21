# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B70-K2: HyperConnection producers that also emit the per-token int8 rows.

Each kernel computes exactly what its ``ops/hc.py`` / ``b70_hc_gemm.py``
counterpart computes (same fp32 expressions, same bf16 store), then quantizes
the stored bf16 row with the XPU ``scaled_int8_quant`` rule
(``b70_i8_producer.i8_scale`` / ``i8_quant``). One program owns one token row,
because the per-token scale needs the whole row.

  hc_gate_mix_quant   = hc_gate_mix               -> in_proj_qkvz / qkv_proj
  hc_silu_quant       = hc_silu                    -> HC up (input_mix_weight_up)
  parallel_hc_gemm_silu_quant
                      = parallel_hc_gemm_silu (V2) -> HC up
"""

import torch

from vllm.model_executor.layers import b70_i8_producer as _pq
from vllm.model_executor.layers.b70_i8_producer import i8_quant, i8_scale
from vllm.triton_utils import tl, triton


@triton.jit
def _hc_gate_mix_q_kernel(
    x_ptr,
    g_ptr,
    y_ptr,
    q_ptr,
    s_ptr,
    stride_x,
    stride_g,
    stride_y,
    DIM: tl.constexpr,
    HC: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    NT_PAD: tl.constexpr,
):
    HC_DIM: tl.constexpr = DIM // HC
    row = tl.program_id(0).to(tl.int64)
    offs = (
        tl.arange(0, NT_PAD)[:, None] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    )
    mask = offs < HC_DIM
    # _hc_gate_mix_kernel, all tiles of the row in one program.
    acc = tl.zeros([NT_PAD, BLOCK_SIZE], dtype=tl.float32)
    for stream in tl.static_range(HC):
        offsets = stream * HC_DIM + offs
        g = tl.load(g_ptr + row * stride_g + offsets, mask, other=0.0)
        x = tl.load(x_ptr + row * stride_x + offsets, mask, other=0.0)
        acc += tl.sigmoid(g.to(tl.float32)) * x.to(tl.float32)
    acc /= HC
    yb = acc.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + row * stride_y + offs, yb, mask)
    yf = yb.to(tl.float32)
    amax = tl.max(tl.max(tl.where(mask, tl.abs(yf), 0.0), axis=1), axis=0)
    s = i8_scale(amax)
    tl.store(q_ptr + row * HC_DIM + offs, i8_quant(yf, s), mask)
    tl.store(s_ptr + row, s)


def hc_gate_mix_quant(
    x: torch.Tensor, gate: torch.Tensor, hc_count: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """hc_gate_mix(x, gate, hc_count) + its int8 rows [N, HC_DIM] and fp32 [N, 1]."""
    n, dim = gate.shape
    assert x.shape == gate.shape and dim % hc_count == 0
    assert x.stride(1) == 1 and gate.stride(1) == 1
    assert x.dtype == torch.bfloat16
    hc_dim = dim // hc_count
    out = x.new_empty(n, hc_dim)
    xq = torch.empty((n, hc_dim), dtype=torch.int8, device=x.device)
    xs = torch.empty((n, 1), dtype=torch.float32, device=x.device)
    block = 512
    if n:
        _hc_gate_mix_q_kernel[(n,)](
            x,
            gate,
            out,
            xq,
            xs,
            x.stride(0),
            gate.stride(0),
            out.stride(0),
            dim,
            hc_count,
            block,
            triton.next_power_of_2(triton.cdiv(hc_dim, block)),
            num_warps=_pq.NUM_WARPS,
        )
    return out, xq, xs


@triton.jit
def _hc_silu_q_kernel(
    x_ptr,
    y_ptr,
    q_ptr,
    s_ptr,
    stride_x,
    stride_y,
    DIM: tl.constexpr,
    HC: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = triton.next_power_of_2(DIM)
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK_SIZE)
    mask = offs < DIM
    # _hc_silu_kernel
    x = tl.load(x_ptr + row * stride_x + offs, mask).to(tl.float32) / HC
    y = x * tl.sigmoid(x)
    yb = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + row * stride_y + offs, yb, mask)
    yf = yb.to(tl.float32)
    amax = tl.max(tl.where(mask, tl.abs(yf), 0.0), axis=0)
    s = i8_scale(amax)
    tl.store(q_ptr + row * DIM + offs, i8_quant(yf, s), mask)
    tl.store(s_ptr + row, s)


def hc_silu_quant(
    x: torch.Tensor, hc_count: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """hc_silu(x, hc_count) + its int8 rows and fp32 [N, 1] scales."""
    num_tokens, dim = x.shape
    assert x.stride(1) == 1 and x.dtype == torch.bfloat16
    out = x.new_empty(x.shape)
    xq = torch.empty((num_tokens, dim), dtype=torch.int8, device=x.device)
    xs = torch.empty((num_tokens, 1), dtype=torch.float32, device=x.device)
    if num_tokens:
        _hc_silu_q_kernel[(num_tokens,)](
            x,
            out,
            xq,
            xs,
            x.stride(0),
            out.stride(0),
            DIM=dim,
            HC=hc_count,
            num_warps=_pq.NUM_WARPS,
        )
    return out, xq, xs


@triton.jit
def _hc_finish_silu_q_kernel(
    partials,
    output,
    lora_out,
    q_ptr,
    s_ptr,
    M: tl.constexpr,
    N: tl.constexpr,
    TILES: tl.constexpr,
    LORA: tl.constexpr,
    HC: tl.constexpr,
    N_PAD: tl.constexpr,
):
    """_hc_finish_silu (V2) for one row: the same fp32 tile sum in the same step
    order, BF16 store, hc_silu of the first LORA columns from the stored value;
    then the int8 quant of the silu row."""
    row = tl.program_id(0)
    col = tl.arange(0, N_PAD)
    offsets = row * N + col
    mask = col < N
    value = tl.full((N_PAD,), 0, tl.float32)
    for step in range(TILES):
        value += tl.load(partials + step * M * N + offsets, mask, other=0)
    stored = value.to(output.dtype.element_ty)
    tl.store(output + offsets, stored, mask)
    v = stored.to(tl.float32) / HC
    y = v * tl.sigmoid(v)
    lmask = col < LORA
    yb = y.to(lora_out.dtype.element_ty)
    tl.store(lora_out + row * LORA + col, yb, lmask)
    yf = yb.to(tl.float32)
    amax = tl.max(tl.where(lmask, tl.abs(yf), 0.0), axis=0)
    s = i8_scale(amax)
    tl.store(q_ptr + row * LORA + col, i8_quant(yf, s), lmask)
    tl.store(s_ptr + row, s)


def parallel_hc_gemm_silu_quant(x, weight, valid_columns, lora_rank, hc_count):
    """parallel_hc_gemm_silu (B70-V2-HC) + the int8 rows of the silu output.

    Returns (out [rows, columns], silu_lora [rows, lora_rank], xq, xs)."""
    from ..b70_hc_gemm import _parallel_partials

    rows, _ = x.shape
    columns = weight.shape[0]
    assert 0 < lora_rank <= columns and x.dtype == torch.bfloat16
    partials, tiles = _parallel_partials(x, weight, valid_columns)
    output = torch.empty((rows, columns), dtype=x.dtype, device=x.device)
    lora = torch.empty((rows, lora_rank), dtype=x.dtype, device=x.device)
    xq = torch.empty((rows, lora_rank), dtype=torch.int8, device=x.device)
    xs = torch.empty((rows, 1), dtype=torch.float32, device=x.device)
    _hc_finish_silu_q_kernel[(rows,)](
        partials,
        output,
        lora,
        xq,
        xs,
        rows,
        columns,
        tiles,
        lora_rank,
        hc_count,
        triton.next_power_of_2(columns),
        num_warps=_pq.NUM_WARPS,
    )
    return output, lora, xq, xs


# ---------------------------------------------------------------------------
# B70-K2 HC down int8 with per-(token, stream) activation scales
# (B70_HC_DOWN_INT8=1 + B70_HC_DOWN_INT8_STREAM_SCALE=1).
#
# The HC down input xn [R, HC*HS] is HC independently RMS-normalised streams, each
# with its own (1 + w) affine; the stream gains differ by up to ~5.7x in the deployed
# checkpoint. One per-token scale over all HC*HS columns gives the small streams
# few int8 levels. Here every (token, stream) gets its own scale, and the weight
# every (row, stream). The producer (combine+norm) owns exactly one (token, stream)
# per program, so the amax is local. The GEMM splits K at stream boundaries (and
# SPS times inside each stream): exact int32 per split, scaled to fp32 by that
# split's (token, stream) x (row, stream) scales, summed in a fixed split order by
# the finish kernel. A row's bits depend only on that row: deterministic and
# batch-invariant, as the ordered DFlash2 HC path requires.
# ---------------------------------------------------------------------------
@triton.jit
def _hc_combine_norm_q_kernel(
    block_ptr,
    res_ptr,
    inj_ptr,
    w_ptr,
    out_ptr,
    y_ptr,
    q_ptr,
    s_ptr,
    stride_block,
    stride_res,
    stride_inj,
    stride_out,
    stride_y,
    HC_DIM: tl.constexpr,
    HC: tl.constexpr,
    W_SHARED: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """ops/hc.py _hc_combine_norm_kernel, plus the int8 rows of this (row, stream)."""
    HC_PAD: tl.constexpr = triton.next_power_of_2(HC)
    NUM_TILES: tl.constexpr = triton.cdiv(HC_DIM, BLOCK_SIZE)
    NUM_TILES_PAD: tl.constexpr = triton.next_power_of_2(NUM_TILES)

    pid = tl.program_id(0)
    row = pid // HC
    stream = pid % HC
    offs_hc = tl.arange(0, HC_PAD)
    mask_hc = offs_hc < HC
    tile_ids = tl.arange(0, NUM_TILES_PAD)
    offs_inner = tile_ids[:, None] * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :]
    mask_inner = offs_inner < HC_DIM
    offs = stream * HC_DIM + offs_inner
    w_offs = offs_inner if W_SHARED else offs

    res = tl.load(res_ptr + row * stride_res + offs, mask_inner, other=0.0)
    if inj_ptr is not None:
        inj = tl.load(inj_ptr + row * stride_inj + offs_hc, mask_hc, other=0.0)
    block = tl.load(block_ptr + row * stride_block + offs_inner, mask_inner, other=0.0)
    if inj_ptr is not None:
        inj = 2.0 * tl.sigmoid(inj.to(tl.float32) / HC)
        block = block.to(tl.float32) * tl.sum(tl.where(offs_hc == stream, inj, 0.0))
    out = (res.to(tl.float32) + block.to(tl.float32)).to(out_ptr.dtype.element_ty)
    if inj_ptr is None:
        w = tl.load(w_ptr + w_offs, mask_inner, other=0.0)
    tl.store(out_ptr + row * stride_out + offs, out, mask=mask_inner)

    out = out.to(tl.float32)
    sum_sq = tl.sum(tl.sum(out * out, axis=1), axis=0)
    rrms = tl.rsqrt(sum_sq / HC_DIM + EPS)
    if inj_ptr is not None:
        w = tl.load(w_ptr + w_offs, mask_inner, other=0.0)
    y = out * rrms
    y += y * w.to(tl.float32)
    yb = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + row * stride_y + offs, yb, mask_inner)
    yf = yb.to(tl.float32)
    amax = tl.max(tl.max(tl.where(mask_inner, tl.abs(yf), 0.0), axis=1), axis=0)
    s = i8_scale(amax)
    tl.store(
        q_ptr + row.to(tl.int64) * (HC * HC_DIM) + offs, i8_quant(yf, s), mask_inner
    )
    tl.store(s_ptr + row * HC + stream, s)


def hc_combine_norm_quant(
    residual, block_output, injection_logits, norm_weight, eps, hc_count
):
    """hc_combine_norm + int8 rows [N, DIM] of y and fp32 scales [N, hc_count]."""
    n, dim = residual.shape
    assert dim % hc_count == 0 and residual.dtype == torch.bfloat16
    hc_dim = dim // hc_count
    assert block_output.shape == (n, hc_dim)
    assert residual.stride(1) == 1 and block_output.stride(1) == 1
    if injection_logits is not None:
        assert injection_logits.shape == (n, hc_count)
        assert injection_logits.stride(1) == 1
    assert norm_weight.is_contiguous() and norm_weight.numel() in (hc_dim, dim)
    stride_injection = injection_logits.stride(0) if injection_logits is not None else 0
    out = residual.new_empty(residual.shape)
    y = residual.new_empty(residual.shape)
    xq = torch.empty((n, dim), dtype=torch.int8, device=residual.device)
    xs = torch.empty((n, hc_count), dtype=torch.float32, device=residual.device)
    if n:
        _hc_combine_norm_q_kernel[(n * hc_count,)](
            block_output,
            residual,
            injection_logits,
            norm_weight,
            out,
            y,
            xq,
            xs,
            block_output.stride(0),
            residual.stride(0),
            stride_injection,
            out.stride(0),
            y.stride(0),
            hc_dim,
            hc_count,
            W_SHARED=norm_weight.numel() == hc_dim,
            EPS=eps,
            BLOCK_SIZE=512,
        )
    return out, y, xq, xs


@triton.jit
def _grouped_gemma_rmsnorm_q_kernel(
    x_ptr,
    w_ptr,
    y_ptr,
    q_ptr,
    s_ptr,
    stride_x,
    stride_y,
    DIM: tl.constexpr,
    NUM_GROUPS: tl.constexpr,
    W_SHARED: tl.constexpr,
    EPS: tl.constexpr,
):
    """ops/hc.py _grouped_gemma_rmsnorm_kernel, plus the int8 rows of this group."""
    GROUP_DIM: tl.constexpr = DIM // NUM_GROUPS
    BLOCK_SIZE: tl.constexpr = triton.next_power_of_2(GROUP_DIM)
    pid = tl.program_id(0)
    group_id = pid % NUM_GROUPS
    row = pid // NUM_GROUPS
    offs_g = tl.arange(0, BLOCK_SIZE)
    offsets = group_id * GROUP_DIM + offs_g
    mask = offs_g < GROUP_DIM
    w_offs = offs_g if W_SHARED else offsets
    x = tl.load(x_ptr + row * stride_x + offsets, mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + w_offs, mask, other=0.0)
    rrms = tl.rsqrt(tl.sum(x * x) / GROUP_DIM + EPS)
    y = x * rrms
    y += y * w.to(tl.float32)
    yb = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + row * stride_y + offsets, yb, mask)
    yf = yb.to(tl.float32)
    amax = tl.max(tl.where(mask, tl.abs(yf), 0.0), axis=0)
    s = i8_scale(amax)
    tl.store(q_ptr + row.to(tl.int64) * DIM + offsets, i8_quant(yf, s), mask)
    tl.store(s_ptr + row * NUM_GROUPS + group_id, s)


def grouped_gemma_rmsnorm_quant(x, weight, eps, num_groups):
    """grouped_gemma_rmsnorm + int8 rows and fp32 scales [N, num_groups]."""
    n, dim = x.shape
    assert x.stride(1) == 1 and weight.is_contiguous() and dim % num_groups == 0
    assert x.dtype == torch.bfloat16
    group_dim = dim // num_groups
    assert weight.numel() in (group_dim, dim)
    y = x.new_empty(x.shape)
    xq = torch.empty((n, dim), dtype=torch.int8, device=x.device)
    xs = torch.empty((n, num_groups), dtype=torch.float32, device=x.device)
    if n:
        _grouped_gemma_rmsnorm_q_kernel[(n * num_groups,)](
            x,
            weight,
            y,
            xq,
            xs,
            x.stride(0),
            y.stride(0),
            dim,
            num_groups,
            W_SHARED=weight.numel() == group_dim,
            EPS=eps,
        )
    return y, xq, xs


@triton.jit
def _group_quant_kernel(
    x_ptr, q_ptr, s_ptr, stride_x, DIM: tl.constexpr, G: tl.constexpr
):
    """Per-(row, group) int8 quant of bf16 [N, DIM] (when no producer did it)."""
    GD: tl.constexpr = DIM // G
    BLOCK: tl.constexpr = triton.next_power_of_2(GD)
    pid = tl.program_id(0)
    row = pid // G
    grp = pid % G
    offs = tl.arange(0, BLOCK)
    mask = offs < GD
    yf = tl.load(x_ptr + row * stride_x + grp * GD + offs, mask, other=0.0).to(
        tl.float32
    )
    s = i8_scale(tl.max(tl.where(mask, tl.abs(yf), 0.0), axis=0))
    tl.store(q_ptr + row.to(tl.int64) * DIM + grp * GD + offs, i8_quant(yf, s), mask)
    tl.store(s_ptr + row * G + grp, s)


def group_quant(x: torch.Tensor, groups: int):
    n, dim = x.shape
    assert x.stride(1) == 1 and dim % groups == 0
    xq = torch.empty((n, dim), dtype=torch.int8, device=x.device)
    xs = torch.empty((n, groups), dtype=torch.float32, device=x.device)
    if n:
        _group_quant_kernel[(n * groups,)](x, xq, xs, x.stride(0), dim, groups)
    return xq, xs


def quantize_weight_int8_grouped(w: torch.Tensor, groups: int):
    """[N, K] float -> (int8 [N, K], fp32 [N, groups]): RTN per (row, group)."""
    n, k = w.shape
    wg = w.float().view(n, groups, k // groups)
    s = (wg.abs().amax(dim=-1) / 127.0).clamp(min=1e-8)
    q = torch.round(wg / s[..., None]).clamp(-127, 127).to(torch.int8)
    return q.view(n, k).contiguous(), s.contiguous()


@triton.jit(do_not_specialize=["M"])
def _hc_i8s_partials(
    xq,
    xs,
    wq,
    ws,
    partials,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    G: tl.constexpr,
    SPS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    """Program (split, n tile, m tile): exact int32 over this split's K range, then
    fp32 x (token, stream) x (row, stream) scales -> partials[split, M, N]."""
    split = tl.program_id(0)
    stream = split // SPS
    KS: tl.constexpr = K // (G * SPS)
    rows = tl.program_id(2) * BM + tl.arange(0, BM)
    cols = tl.program_id(1) * BN + tl.arange(0, BN)
    rmask = rows < M
    cmask = cols < N
    k0 = split * KS
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for step in range(KS // BK):
        kk = k0 + step * BK + tl.arange(0, BK)
        a = tl.load(
            xq + rows[:, None].to(tl.int64) * K + kk[None, :], rmask[:, None], other=0
        )
        b = tl.load(
            wq + cols[None, :].to(tl.int64) * K + kk[:, None], cmask[None, :], other=0
        )
        acc += tl.dot(a, b, out_dtype=tl.int32)
    sa = tl.load(xs + rows * G + stream, rmask, other=0.0)
    sb = tl.load(ws + cols * G + stream, cmask, other=0.0)
    v = acc.to(tl.float32) * sa[:, None] * sb[None, :]
    tl.store(
        partials + split * M * N + rows[:, None] * N + cols[None, :],
        v,
        rmask[:, None] & cmask[None, :],
    )


def hc_down_i8s_partials(xq, xs, wq, ws, sps: int = 4):
    """fp32 partials [G*SPS, M, N] of the per-stream-scaled int8 HC down GEMM."""
    m, k = xq.shape
    n = wq.shape[0]
    g = xs.shape[1]
    assert ws.shape == (n, g) and k % (g * sps) == 0
    ks = k // (g * sps)
    bk = 128 if ks % 128 == 0 else 64
    assert ks % bk == 0
    bm = 16 if m <= 16 else (32 if m <= 32 else 64)
    bn = 32
    tiles = g * sps
    partials = torch.empty((tiles, m, n), dtype=torch.float32, device=xq.device)
    if m:
        _hc_i8s_partials[(tiles, triton.cdiv(n, bn), triton.cdiv(m, bm))](
            xq, xs, wq, ws, partials, m, n, k, g, sps, bm, bn, bk, num_warps=4
        )
    return partials, tiles


def hc_down_i8s_silu(xq, xs, wq, ws, lora_rank, hc_count, quant_lora, sps=4):
    """(down [M, N] bf16, silu(lora) [M, lora_rank], xq_lora | None, xs_lora | None)."""
    from ..b70_hc_gemm import _hc_finish_silu

    rows = xq.shape[0]
    columns = wq.shape[0]
    partials, tiles = hc_down_i8s_partials(xq, xs, wq, ws, sps)
    output = torch.empty((rows, columns), dtype=torch.bfloat16, device=xq.device)
    lora = torch.empty((rows, lora_rank), dtype=torch.bfloat16, device=xq.device)
    if quant_lora:
        lq = torch.empty((rows, lora_rank), dtype=torch.int8, device=xq.device)
        ls = torch.empty((rows, 1), dtype=torch.float32, device=xq.device)
        if rows:
            _hc_finish_silu_q_kernel[(rows,)](
                partials,
                output,
                lora,
                lq,
                ls,
                rows,
                columns,
                tiles,
                lora_rank,
                hc_count,
                triton.next_power_of_2(columns),
                num_warps=_pq.NUM_WARPS,
            )
        return output, lora, lq, ls
    if rows:
        _hc_finish_silu[(triton.cdiv(rows * columns, 256),)](
            partials,
            output,
            lora,
            rows,
            columns,
            tiles,
            lora_rank,
            hc_count,
            num_warps=4,
        )
    return output, lora, None, None


def hc_down_i8s(xq, xs, wq, ws, sps=4):
    """Down [M, N] bf16 only (no silu): the same partials and fixed-order sum."""
    from ..b70_hc_gemm import _hc_finish

    rows = xq.shape[0]
    columns = wq.shape[0]
    partials, tiles = hc_down_i8s_partials(xq, xs, wq, ws, sps)
    output = torch.empty((rows, columns), dtype=torch.bfloat16, device=xq.device)
    if rows:
        _hc_finish[(triton.cdiv(rows * columns, 256),)](
            partials, output, rows, columns, tiles, num_warps=4
        )
    return output
