# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B70-IP-W1: BF16 dense linears as W8A8 INT8 on XPU (opt-in, B70_DENSE_INT8=1).

Intel Xe2 XMX runs INT8 natively; BF16 linears read 2 bytes per weight in
decode and run BF16 DPAS in prefill. With B70_DENSE_INT8=1, every unquantized
linear whose prefix passes the include/exclude regexes is quantized once after
loading: weights int8 symmetric per output channel (RTN), activations int8
symmetric per token (dynamic, native SYCL op), GEMM on XMX. Measured on one
B70 under graph replay with rotating weights (no L2 reuse): 1.5-2.0x faster
than BF16 at M=1..4096, relative error ~1.2 % per GEMM.

Kernel by M (rows): Triton int8 up to B70_DENSE_INT8_TRITON_MAX_M (default 64,
decode; it wins there), oneDNN ``_xpu_C.int8_scaled_mm`` above (prefill; it
wins at M >= 2048). Missing oneDNN op -> Triton everywhere.

Env:
  B70_DENSE_INT8=1                    turn it on (XPU only)
  B70_DENSE_INT8_INCLUDE=<regex>      prefixes to convert (default: all)
  B70_DENSE_INT8_EXCLUDE=<regex>      prefixes to keep BF16 (default below)
  B70_DENSE_INT8_MIN_PARAMS=<n>       skip smaller weights (default 1<<20)
  B70_DENSE_INT8_TRITON_MAX_M=<n>     Triton for M <= n (default 64)

B70-K1 small-M kernels (needs the v7 vllm_xpu_kernels wheel; default off):
  B70_DENSE_INT8_FUSED=1              route M <= FUSED_MAX_M to the K1 ops, which
                                      read the same int8 weights + fp32 row scale:
                                      M <= W8A16_MAX_M -> _xpu_C.b70_w8a16_gemm
                                      (weight-only, bf16 act, no quant pass), else
                                      _xpu_C.b70_w8a8_fused_gemm (act quant in the
                                      GEMM, s8 x s8 DPAS). Missing ops -> W1 path.
  B70_DENSE_INT8_FUSED_MAX_M=<n>      default 64 (the fused op's ceiling)
  B70_DENSE_INT8_W8A16_MAX_M=<n>      default 8 (0 = fused W8A8 only)

B70-K2 producer quant (b70_i8_producer, B70_I8_PRODUCER_QUANT=1): when the
producing kernel already wrote x's int8 rows and per-token scales, apply()
skips the quant and runs b70_int8_dense_mm_q (K1 b70_w8a8_gemm for small M,
else this module's GEMMs on the same int8 rows).
  To reach the shared expert and in_proj_ba, also lower
  B70_DENSE_INT8_MIN_PARAMS (e.g. 0) and drop in_proj_ba from B70_DENSE_INT8_EXCLUDE.
"""

import os
import re

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers import b70_i8_producer
from vllm.model_executor.layers.linear import LinearMethodBase
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)

# Routers, the GDN a/b projections, the sparse-attention indexer, heads,
# embeddings and vision stay BF16 by default: small, precision-sensitive, or
# owned by another path (B70_HEAD_FP8).
DEFAULT_EXCLUDE = (
    r"(lm_head|embed|visual|vision|indexer|in_proj_a\b|in_proj_b\b|in_proj_ba\b"
    r"|\.gate$|shared_expert_gate|router|\.ple)"
)

_stats = {"layers": 0, "bytes_saved": 0, "skipped": 0}
_logged_first = False

# B70-K1: read once at import (graph capture must see one path).
_K1_FUSED = os.environ.get("B70_DENSE_INT8_FUSED", "0") == "1"
_K1_FUSED_MAX_M = int(os.environ.get("B70_DENSE_INT8_FUSED_MAX_M", "64"))
_K1_W8A16_MAX_M = int(os.environ.get("B70_DENSE_INT8_W8A16_MAX_M", "8"))
_k1_ops = None  # None = not probed, False = absent, else the module


def _k1():
    """vllm_xpu_kernels.b70_int8_dense if B70_DENSE_INT8_FUSED=1 and the ops exist."""
    global _k1_ops
    if _k1_ops is None:
        _k1_ops = False
        if _K1_FUSED:
            try:
                from vllm_xpu_kernels import b70_int8_dense as k1

                if k1.available():
                    _k1_ops = k1
                    logger.info(
                        "B70-K1 dense int8: small-M kernels on (W8A16 M<=%d, "
                        "fused W8A8 M<=%d, W1 path above)",
                        _K1_W8A16_MAX_M,
                        min(_K1_FUSED_MAX_M, 64),
                    )
                else:
                    logger.warning(
                        "B70-K1 dense int8: B70_DENSE_INT8_FUSED=1 but the "
                        "installed vllm_xpu_kernels has no b70_w8a16_gemm / "
                        "b70_w8a8_fused_gemm; keeping the W1 path"
                    )
            except ImportError:
                logger.warning(
                    "B70-K1 dense int8: vllm_xpu_kernels.b70_int8_dense missing; "
                    "keeping the W1 path"
                )
    return _k1_ops or None


def _enabled() -> bool:
    return os.environ.get("B70_DENSE_INT8", "0") == "1" and current_platform.is_xpu()


def _wanted(layer: torch.nn.Module) -> tuple[bool, str]:
    prefix = getattr(layer, "prefix", "") or ""
    inc = os.environ.get("B70_DENSE_INT8_INCLUDE", "")
    exc = os.environ.get("B70_DENSE_INT8_EXCLUDE", DEFAULT_EXCLUDE)
    if inc and not re.search(inc, prefix):
        return False, "not included"
    if exc and re.search(exc, prefix):
        return False, "excluded"
    if getattr(layer, "hc_ksplit_prepare", None) is not None:
        return False, "hyper-connection k-split layer"
    w = getattr(layer, "weight", None)
    if not isinstance(w, torch.Tensor) or w.dim() != 2:
        return False, "no 2-D weight"
    if w.dtype not in (torch.bfloat16, torch.float16):
        return False, f"weight dtype {w.dtype}"
    if w.numel() < int(os.environ.get("B70_DENSE_INT8_MIN_PARAMS", str(1 << 20))):
        return False, "small"
    return True, ""


def quantize_weight_int8(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """[N, K] float -> (int8 [N, K], fp32 scale [N, 1]); symmetric per row, RTN.

    Chunked over rows so the fp32 temporary stays small for big weights.
    """
    n = w.shape[0]
    q = torch.empty_like(w, dtype=torch.int8)
    s = torch.empty((n, 1), dtype=torch.float32, device=w.device)
    step = max(1, (64 << 20) // max(1, w.shape[1] * 4))
    for i in range(0, n, step):
        blk = w[i : i + step].float()
        sc = (blk.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-8)
        q[i : i + step] = torch.round(blk / sc).clamp(-127, 127).to(torch.int8)
        s[i : i + step] = sc
    return q, s


class B70Int8DenseMethod(LinearMethodBase):
    """Replaces UnquantizedLinearMethod for one converted layer.

    Deliberately not an UnquantizedLinearMethod: code that fuses or reads
    BF16 weights directly when a layer is unquantized must take its
    quantized fallback here.
    """

    def __init__(self, triton_max_m: int, has_onednn: bool):
        self.triton_max_m = triton_max_m
        self.has_onednn = has_onednn

    def create_weights(self, *args, **kwargs):
        raise RuntimeError("B70Int8DenseMethod is installed after loading")

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        return None

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pq = b70_i8_producer.take(x) if b70_i8_producer.ENABLED else None
        if pq is not None:
            # B70-K2: the producer already wrote x's int8 rows and scales.
            out = torch.ops.vllm.b70_int8_dense_mm_q(
                pq[0],
                pq[1],
                layer.weight,
                layer.b70_int8_scale,
                bias,
                self.triton_max_m,
                b70_i8_producer.layer_takes_k1(layer),
            )
            return out.view(*x.shape[:-1], layer.weight.shape[0])
        x_2d = x.reshape(-1, x.shape[-1])
        out = torch.ops.vllm.b70_int8_dense_mm(
            x_2d, layer.weight, layer.b70_int8_scale, bias, self.triton_max_m
        )
        return out.view(*x.shape[:-1], layer.weight.shape[0])


def _b70_int8_dense_mm(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    bias: torch.Tensor | None,
    triton_max_m: int,
) -> torch.Tensor:
    """One opaque op for torch.compile: native act quant + the int8 GEMM by M."""
    from vllm import _custom_ops as ops

    if _K1_FUSED and x.shape[0] <= max(_K1_FUSED_MAX_M, _K1_W8A16_MAX_M):
        k1 = _k1()
        if k1 is not None:
            out = k1.mm(
                x.contiguous(), w_q, w_s, bias, _K1_W8A16_MAX_M, _K1_FUSED_MAX_M
            )
            if out is not None:
                return out

    x_q, x_s, _ = ops.scaled_int8_quant(x.contiguous(), None, None, True)
    if x.shape[0] > triton_max_m and hasattr(torch.ops._xpu_C, "int8_scaled_mm"):
        return torch.ops._xpu_C.int8_scaled_mm(
            x_q, x_s.reshape(-1).to(torch.float32), w_q.t(), w_s.reshape(-1),
            x.dtype, bias,
        )
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
        triton_scaled_mm,
    )

    return triton_scaled_mm(x_q, w_q.t(), x_s, w_s, x.dtype, bias)


def _b70_int8_dense_mm_fake(
    x: torch.Tensor,
    w_q: torch.Tensor,
    w_s: torch.Tensor,
    bias: torch.Tensor | None,
    triton_max_m: int,
) -> torch.Tensor:
    return torch.empty((x.shape[0], w_q.shape[0]), dtype=x.dtype, device=x.device)


direct_register_custom_op(
    "b70_int8_dense_mm",
    _b70_int8_dense_mm,
    fake_impl=_b70_int8_dense_mm_fake,
)


def maybe_convert(layer: torch.nn.Module) -> bool:
    """Called from UnquantizedLinearMethod.process_weights_after_loading on XPU."""
    global _logged_first
    if not _enabled():
        return False
    ok, why = _wanted(layer)
    if not ok:
        _stats["skipped"] += 1
        logger.debug("B70 dense int8: keep %s BF16 (%s)", getattr(layer, "prefix", "?"), why)
        return False
    w = layer.weight.data
    q, s = quantize_weight_int8(w)
    saved = w.numel() * w.element_size() - q.numel() - s.numel() * 4
    layer.weight = torch.nn.Parameter(q, requires_grad=False)
    layer.b70_int8_scale = s
    del w
    has_onednn = hasattr(torch.ops._xpu_C, "int8_scaled_mm")
    tmax = int(os.environ.get("B70_DENSE_INT8_TRITON_MAX_M", "64"))
    layer.quant_method = B70Int8DenseMethod(tmax, has_onednn)
    _stats["layers"] += 1
    _stats["bytes_saved"] += saved
    if not _logged_first:
        _logged_first = True
        logger.info(
            "B70-IP-W1 dense int8: BF16 linears -> W8A8 int8 (Triton M<=%d, %s above); "
            "first: %s %s",
            tmax,
            "oneDNN" if has_onednn else "Triton",
            getattr(layer, "prefix", "?"),
            tuple(q.shape),
        )
    return True


def report() -> None:
    if _enabled():
        logger.info(
            "B70-IP-W1 dense int8: %d layers converted, %d kept BF16, %.2f GiB saved",
            _stats["layers"],
            _stats["skipped"],
            _stats["bytes_saved"] / 2**30,
        )
