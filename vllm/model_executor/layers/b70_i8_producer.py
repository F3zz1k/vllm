# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B70-K2: per-token int8 activation quant folded into the producer kernels.

With B70_IP-W1 (``b70_dense_int8``) every converted linear runs
``scaled_int8_quant`` (one launch) and then the int8 GEMM. Production stacks
fold that quant into the kernel that produces the activation (vLLM
``rms_norm_dynamic_per_token_quant``, TensorRT-LLM ``generalRmsNorm`` with
``scale_orig_quant_per_token``). B70-K2 does the same for the W1 linears of
Qwen4Exp (``B70_I8_PRODUCER_QUANT=1``, default off):

  producer                                consumer (W1)            per step
  HC down finish / hc_silu  (+ quant)  -> HC up        K320  N10240   97
  HC gate mix (attn HC)     (+ quant)  -> in_proj_qkvz K2560 N4096    36
                                          qkv_proj     K2560 N3584    12
  GDN gated RMSNorm         (+ quant)  -> out_proj     K1536 N2560    36
  attn output * sigmoid(gate)(+ quant) -> o_proj       K1536 N2560    12

The producer writes the bf16 output as before and, in the same kernel, the int8
rows and an fp32 per-token scale. The pair rides on the bf16 tensor object
(``attach``) to the consumer's ``B70Int8DenseMethod.apply`` (``take``), which
then skips the quant and runs, for M <= B70_I8_PQ_K1_MAX_M, the K1 GEMM
``_xpu_C.b70_w8a8_gemm(xq, xs, w, ws, bias)`` (tile from the wheel's
``B70_I8D_TILE``), else W1's own GEMM (Triton ``scaled_mm`` / oneDNN) on the
pre-quantized rows.

The quant rule is the one W1 runs today, i.e. the XPU ``scaled_int8_quant``
(native ``dynamic_per_token_scaled_int8_quant`` = ``xpu_ops.
dynamic_per_token_int8_quant_ref``), bit for bit:
  amax = max |y| over the token row of the bf16 output
  s    = bf16(max(amax / 127, 1e-5))          (IEEE division, RNE to bf16)
  q    = clamp(rint(y / s), -128, 127)        (IEEE division, half to even)
So with ``B70_I8_PQ_GEMM=w1`` the result equals the unfused W1 path bitwise.

Env (read at import, so one graph capture sees one path):
  B70_I8_PRODUCER_QUANT=1   turn it on
  B70_I8_PQ_SITES=<list>    producer sites, default all:
                            hc_up,hc_block,gdn_out,attn_o
  B70_I8_PQ_MAX_M=<n>       producers emit for rows <= n (default: every width)
  B70_I8_PQ_GEMM=k1|w1      GEMM on the pre-quantized rows (default k1)
  B70_I8_PQ_K1_MAX_M=<n>    K1 GEMM for M <= n (default 64), W1 GEMM above
  B70_I8_PQ_K1_INCLUDE=<re> consumer prefixes that take the K1 GEMM (default all)
  B70_I8_PQ_NUM_WARPS=<n>   num_warps of the producer kernels (default 4)

Off under torch.compile (B70_TORCH_COMPILE=1 or Dynamo tracing): the carrier
is a Python attribute, which a traced graph does not see.
"""

import os
import re

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

ALL_SITES = ("hc_up", "hc_block", "gdn_out", "attn_o")

ENABLED = os.environ.get("B70_I8_PRODUCER_QUANT", "0") == "1"
_SITES = frozenset(
    s.strip()
    for s in os.environ.get("B70_I8_PQ_SITES", ",".join(ALL_SITES)).split(",")
    if s.strip()
)
MAX_M = int(os.environ.get("B70_I8_PQ_MAX_M", str(1 << 30)))
_GEMM = os.environ.get("B70_I8_PQ_GEMM", "k1")
_K1_MAX_M = int(os.environ.get("B70_I8_PQ_K1_MAX_M", "64"))
_K1_INCLUDE = os.environ.get("B70_I8_PQ_K1_INCLUDE", "")
NUM_WARPS = int(os.environ.get("B70_I8_PQ_NUM_WARPS", "4"))
_TC = os.environ.get("B70_TORCH_COMPILE", "0") == "1"

_ATTR = "_b70_i8"
_logged_on = False
_logged_sites: set[str] = set()
_k1_gemm = None  # None = not probed, False = absent, else the op


def _log_on() -> None:
    global _logged_on
    if not _logged_on:
        _logged_on = True
        logger.info(
            "B70-K2: producer int8 act quant on (sites=%s, rows<=%d; GEMM %s: "
            "K1 b70_w8a8_gemm M<=%d include=%r tile=%s, W1 GEMM above)",
            ",".join(s for s in ALL_SITES if s in _SITES),
            MAX_M,
            _GEMM,
            _K1_MAX_M,
            _K1_INCLUDE or "all",
            os.environ.get("B70_I8D_TILE", "auto"),
        )


def site_on(site: str) -> bool:
    return ENABLED and not _TC and site in _SITES


def consumer_is_w1(linear: torch.nn.Module | None) -> bool:
    """True if ``linear`` runs B70-IP-W1 (int8 weights, act quant in apply)."""
    if linear is None:
        return False
    qm = getattr(linear, "quant_method", None)
    return type(qm).__name__ == "B70Int8DenseMethod"


def emits(site: str, rows: int) -> bool:
    """Call-time check at a producer whose consumer is a W1 linear."""
    if not site_on(site) or rows > MAX_M or rows <= 0:
        return False
    if torch.compiler.is_compiling():
        return False
    _log_on()
    return True


def attach(
    x: torch.Tensor, xq: torch.Tensor, xs: torch.Tensor, site: str
) -> torch.Tensor:
    """Hang (int8 rows, fp32 [M, 1] scale) of ``x`` on the tensor object."""
    setattr(x, _ATTR, (xq, xs))
    if site not in _logged_sites:
        _logged_sites.add(site)
        logger.info(
            "B70-K2: %s producer emits int8 + scale (first: rows=%d K=%d)",
            site,
            xq.shape[0],
            xq.shape[1],
        )
    return x


def take(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """(xq [M, K] int8, xs [M, 1] fp32) if the producer quantized exactly ``x``."""
    pq = x.__dict__.get(_ATTR) if hasattr(x, "__dict__") else None
    if pq is None:
        return None
    xq, xs = pq
    k = x.shape[-1]
    m = x.numel() // k if k else 0
    if xq.shape != (m, k) or xs.numel() != m:
        return None
    return xq, xs


# ----------------------------------------------------------------- quant math
@triton.jit
def i8_scale(amax):
    """bf16(max(amax / 127, 1e-5)) as fp32: the XPU scaled_int8_quant scale."""
    s = tl.div_rn(amax, 127.0)
    # torch.clamp(min=1e-5) keeps NaN; (NaN < 1e-5) is False, so does this.
    s = tl.where(s < 1e-5, 1e-5, s)
    return s.to(tl.bfloat16).to(tl.float32)


@triton.jit
def i8_quant(y, s):
    """clamp(rint(y / s), -128, 127) as int8; y holds bf16 values in fp32."""
    v = tl.div_rn(y, s)
    # Round half to even without libdevice (the interpreter has none): for
    # |v| < 2**22, v + 1.5 * 2**23 lands where the fp32 ulp is 1. |v| <= ~128.5.
    r = (v + 12582912.0) - 12582912.0
    r = tl.minimum(tl.maximum(r, -128.0), 127.0)
    return r.to(tl.int8)


# --------------------------------------------- attention: out * sigmoid(gate)
@triton.jit
def _attn_gate_q_kernel(
    a_ptr,
    g_ptr,
    y_ptr,
    q_ptr,
    s_ptr,
    stride_a,
    stride_g,
    stride_y,
    DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    offs = tl.arange(0, BLOCK)
    mask = offs < DIM
    a = tl.load(a_ptr + row * stride_a + offs, mask, other=0.0)
    g = tl.load(g_ptr + row * stride_g + offs, mask, other=0.0)
    # torch: sigmoid(gate) is a bf16 tensor, then bf16 * bf16 -> bf16.
    sg = tl.sigmoid(g.to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    yb = (a.to(tl.float32) * sg).to(tl.bfloat16)
    tl.store(y_ptr + row * stride_y + offs, yb, mask)
    yf = yb.to(tl.float32)
    amax = tl.max(tl.where(mask, tl.abs(yf), 0.0), axis=0)
    s = i8_scale(amax)
    tl.store(q_ptr + row * DIM + offs, i8_quant(yf, s), mask)
    tl.store(s_ptr + row, s)


def attn_gate_quant(
    attn_output: torch.Tensor, gate: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """attn_output * sigmoid(gate) (bf16, as torch) + its int8 rows and scale."""
    assert attn_output.dim() == 2 and attn_output.shape == gate.shape
    assert attn_output.stride(1) == 1 and gate.stride(1) == 1
    m, dim = attn_output.shape
    y = torch.empty((m, dim), dtype=attn_output.dtype, device=attn_output.device)
    xq = torch.empty((m, dim), dtype=torch.int8, device=attn_output.device)
    xs = torch.empty((m, 1), dtype=torch.float32, device=attn_output.device)
    if m:
        _attn_gate_q_kernel[(m,)](
            attn_output,
            gate,
            y,
            xq,
            xs,
            attn_output.stride(0),
            gate.stride(0),
            y.stride(0),
            DIM=dim,
            BLOCK=triton.next_power_of_2(dim),
            num_warps=NUM_WARPS,
        )
    return y, xq, xs


# -------------------------------------------- GDN: gated RMSNorm (+ quant)
@triton.jit
def _gated_rmsnorm_q_kernel(
    x_ptr,
    z_ptr,
    w_ptr,
    y_ptr,
    q_ptr,
    s_ptr,
    stride_xt,
    stride_xh,
    stride_zt,
    stride_zh,
    eps,
    H: tl.constexpr,
    H_PAD: tl.constexpr,
    N: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """One token: H heads x N, RMSNorm per head, * w, * act(z) (norm_before_gate),
    the FLA layer_norm_fwd_kernel's arithmetic; then the token-row int8 quant."""
    t = tl.program_id(0).to(tl.int64)
    heads = tl.arange(0, H_PAD)
    cols = tl.arange(0, N)
    mask = (heads[:, None] < H) & (cols[None, :] < N)
    x = tl.load(
        x_ptr + t * stride_xt + heads[:, None] * stride_xh + cols[None, :],
        mask,
        other=0.0,
    ).to(tl.float32)
    xbar = tl.where(mask, x, 0.0)
    var = tl.sum(xbar * xbar, axis=1) / N
    rstd = tl.rsqrt(var + eps)
    w = tl.load(w_ptr + cols).to(tl.float32)
    x_hat = x * rstd[:, None]
    y = x_hat * w[None, :]
    z = tl.load(
        z_ptr + t * stride_zt + heads[:, None] * stride_zh + cols[None, :],
        mask,
        other=0.0,
    ).to(tl.float32)
    if ACTIVATION == "swish" or ACTIVATION == "silu":
        y *= z * tl.sigmoid(z)
    elif ACTIVATION == "sigmoid":
        y *= tl.sigmoid(z)
    yb = y.to(tl.bfloat16)
    offs = heads[:, None] * N + cols[None, :]
    tl.store(y_ptr + t * (H * N) + offs, yb, mask)
    yf = yb.to(tl.float32)
    amax = tl.max(tl.max(tl.where(mask, tl.abs(yf), 0.0), axis=1), axis=0)
    s = i8_scale(amax)
    tl.store(q_ptr + t * (H * N) + offs, i8_quant(yf, s), mask)
    tl.store(s_ptr + t, s)


def gated_rmsnorm_quant(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    activation: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """RMSNormGated(norm_before_gate=True, group_size=None) over [T, H, N] bf16,
    returned flat [T, H*N], plus the int8 rows and fp32 [T, 1] scales."""
    assert x.dim() == 3 and z.shape == x.shape and x.dtype == torch.bfloat16
    assert x.stride(-1) == 1 and z.stride(-1) == 1 and weight.is_contiguous()
    t, h, n = x.shape
    assert weight.numel() == n and n == triton.next_power_of_2(n)
    y = torch.empty((t, h * n), dtype=x.dtype, device=x.device)
    xq = torch.empty((t, h * n), dtype=torch.int8, device=x.device)
    xs = torch.empty((t, 1), dtype=torch.float32, device=x.device)
    if t:
        _gated_rmsnorm_q_kernel[(t,)](
            x,
            z,
            weight,
            y,
            xq,
            xs,
            x.stride(0),
            x.stride(1),
            z.stride(0),
            z.stride(1),
            eps,
            H=h,
            H_PAD=triton.next_power_of_2(h),
            N=n,
            ACTIVATION=activation,
            num_warps=NUM_WARPS,
        )
    return y, xq, xs


# ------------------------------------------------------------- the consumer
def _k1_op():
    global _k1_gemm
    if _k1_gemm is None:
        _k1_gemm = False
        try:
            if hasattr(torch.ops._xpu_C, "b70_w8a8_gemm"):
                _k1_gemm = torch.ops._xpu_C.b70_w8a8_gemm
        except (AttributeError, RuntimeError):
            pass
        if _k1_gemm is False and _GEMM == "k1":
            logger.warning(
                "B70-K2: the installed vllm_xpu_kernels has no b70_w8a8_gemm; "
                "pre-quantized rows take the W1 GEMM"
            )
    return _k1_gemm or None


def layer_takes_k1(layer: torch.nn.Module) -> bool:
    """Per-layer K1 choice, decided once (prefix regex) and cached on the layer."""
    v = layer.__dict__.get("_b70_pq_k1")
    if v is None:
        prefix = getattr(layer, "prefix", "") or ""
        w = layer.weight
        v = (
            _GEMM == "k1"
            and (not _K1_INCLUDE or re.search(_K1_INCLUDE, prefix) is not None)
            and w.shape[1] % 32 == 0
            and w.shape[0] % 16 == 0
            and w.shape[0] >= 32
            and w.is_contiguous()
        )
        layer.__dict__["_b70_pq_k1"] = v
    return v


def _b70_int8_dense_mm_q(
    xq: torch.Tensor,
    xs: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    bias: torch.Tensor | None,
    triton_max_m: int,
    use_k1: bool,
) -> torch.Tensor:
    """W1's GEMM step on producer-quantized rows (bf16 out)."""
    m = xq.shape[0]
    if use_k1 and m <= _K1_MAX_M and (bias is None or bias.dtype == torch.bfloat16):
        op = _k1_op()
        if op is not None:
            return op(xq, xs, w_q, w_s, bias)
    if m > triton_max_m and hasattr(torch.ops._xpu_C, "int8_scaled_mm"):
        return torch.ops._xpu_C.int8_scaled_mm(
            xq, xs.reshape(-1), w_q.t(), w_s.reshape(-1), torch.bfloat16, bias
        )
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
        triton_scaled_mm,
    )

    return triton_scaled_mm(xq, w_q.t(), xs, w_s, torch.bfloat16, bias)


def _b70_int8_dense_mm_q_fake(
    xq: torch.Tensor,
    xs: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    bias: torch.Tensor | None,
    triton_max_m: int,
    use_k1: bool,
) -> torch.Tensor:
    return torch.empty(
        (xq.shape[0], w_q.shape[0]), dtype=torch.bfloat16, device=xq.device
    )


direct_register_custom_op(
    "b70_int8_dense_mm_q",
    _b70_int8_dense_mm_q,
    fake_impl=_b70_int8_dense_mm_q_fake,
)


_ATTR_G = "_b70_i8g"


def attach_grouped(x: torch.Tensor, xq: torch.Tensor, xs: torch.Tensor) -> torch.Tensor:
    """Hang (int8 rows, fp32 [M, G] per-(token, group) scales) of ``x`` on it."""
    setattr(x, _ATTR_G, (xq, xs))
    return x


def take_grouped(
    x: torch.Tensor, groups: int
) -> tuple[torch.Tensor, torch.Tensor] | None:
    pq = x.__dict__.get(_ATTR_G) if hasattr(x, "__dict__") else None
    if pq is None:
        return None
    xq, xs = pq
    if x.dim() != 2 or xq.shape != tuple(x.shape) or xs.shape != (x.shape[0], groups):
        return None
    return xq, xs
