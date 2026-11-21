# Modified by Wulf for B70 Flash-Next Disk; upstream terms are retained.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HyperConnection (Gated Residual) utilities — NVIDIA model variant.

Implements the HyperConnection residual scheme proposed in
"HyperConnections" (https://arxiv.org/abs/2409.19606). This NVIDIA variant
delays each HC combine to the following HC mix boundary. HC glue kernels,
including fused combine+RMSNorm, live in ``ops/hc.py``; projections remain
standard vLLM Linear modules.

Hidden states between layers have shape ``[..., HC*HS]`` with HS inner
(HC outer, HS inner — checkpoint-native layout).

Typical usage inside a transformer decoder layer::

    self.attn_hc = GatedResidual(hc_config)

    hidden_states, block_input, injection = self.attn_hc.mix(hidden_states)
    attention_output = attention(block_input)
    hidden_states, block_input, injection = self.mlp_hc.combine_and_mix(
        hidden_states, attention_output, injection
    )
"""

import os

import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.model_executor.layers import b70_i8_producer as b70_pq
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.models.utils import maybe_prefix

from ..common.hyperconnection import (
    GroupedGemmaRMSNorm,
    HyperConnectionConfig,
)
from . import b70_compile_ops
from .ops.hc import (
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_combine_norm,
    hc_gate_mix,
    hc_silu,
)
from .ops.hc_i8 import (
    group_quant,
    grouped_gemma_rmsnorm_quant,
    hc_combine_norm_quant,
    hc_down_i8s,
    hc_down_i8s_silu,
    hc_gate_mix_quant,
    hc_silu_quant,
    parallel_hc_gemm_silu_quant,
    quantize_weight_int8_grouped,
)

logger = init_logger(__name__)
# B70-V32-07: rows up to which the DFlash2 fixed-order HC down projection is used; wider batches (8-stream
# decode, prefill chunks) take the stock BF16 GEMM. The ordered Triton GEMM is 4.5x (64 rows) to 6.7x
# (4096 rows) slower than F.linear on B70. Default: unbounded (wu1ff behaviour).
_B70_HC_ORDERED_MAX_ROWS = int(os.environ.get("B70_HC_ORDERED_MAX_ROWS", str(1 << 30)))
# B70-V2-HC: fold hc_silu into the parallel HC down GEMM's finish kernel (bitwise-
# identical; one launch fewer per GatedResidual on the parallel route, i.e. rows <=
# B70_HC_PARALLEL_MAX_ROWS). Default off.
_B70_HC_SILU_FUSE = os.environ.get("B70_HC_SILU_FUSE", "0") == "1"
_B70_HC_SILU_FUSE_LOGGED = False
# B70-K1: HC down projection in int8 (v7 vllm_xpu_kernels wheel, b70_int8_dense.hc_down):
# per-token int8 quant + s8 x s8 DPAS GEMM over an int8 copy of the down weight (per-row RTN, made at
# load). A row's output depends only on that row (exact int32 accumulation for every K split), so it is
# run-to-run deterministic and batch-invariant, which the ordered DFlash2 path requires; it replaces the
# ordered / K-split / stock down GEMM for rows <= B70_HC_DOWN_INT8_MAX_ROWS (default: every width, so the
# invariance holds across widths). Default off. Costs one int8 copy of each down weight (N x 10240 bytes).
_B70_HC_DOWN_INT8 = os.environ.get("B70_HC_DOWN_INT8", "0") == "1"
_B70_HC_DOWN_INT8_MAX_ROWS = int(os.environ.get("B70_HC_DOWN_INT8_MAX_ROWS", str(1 << 30)))
_B70_HC_DOWN_INT8_LOGGED = False
# B70-K2: with B70_HC_DOWN_INT8=1, per-(token, stream) activation scales and per-(row,
# stream) weight scales instead of one scale over the 4 x 2560 HC streams. The norm
# producer emits the int8 rows; the GEMM is a vLLM Triton split-K (exact int32 per
# split, fixed-order fp32 sum: deterministic, batch-invariant). No wheel needed.
_B70_HC_DOWN_INT8_STREAM = (
    _B70_HC_DOWN_INT8
    and os.environ.get("B70_HC_DOWN_INT8_STREAM_SCALE", "0") == "1"
)
_B70_HC_DOWN_INT8_SPS = int(os.environ.get("B70_HC_DOWN_INT8_SPS", "4"))
# B70-K2 capture (diagnostics only, eager engines): dump HC-down inputs xn on TP rank 0,
# up to B70_K2_HC_CAPTURE_DECODE rows of decode calls (<= 64 rows) per module and the
# first B70_K2_HC_CAPTURE_PREFILL rows of one prefill call (>= 512 rows) for every
# B70_K2_HC_CAPTURE_EVERY-th layer (+ the final mixer). Unset = no code runs.
_B70_K2_CAP_DIR = os.environ.get("B70_K2_HC_CAPTURE_DIR", "")
_B70_K2_CAP_DECODE = int(os.environ.get("B70_K2_HC_CAPTURE_DECODE", "256"))
_B70_K2_CAP_PREFILL = int(os.environ.get("B70_K2_HC_CAPTURE_PREFILL", "2048"))
_B70_K2_CAP_EVERY = int(os.environ.get("B70_K2_HC_CAPTURE_EVERY", "8"))


def _is_xpu() -> bool:
    from vllm.platforms import current_platform

    return current_platform.is_xpu()


# ---------------------------------------------------------------------------
# XPU HC down-GEMM determinism (K-split)
#
# The skinny BF16 HC down projection (K = hyper_hidden_size = 10240 -> N in
# {336 merged, 320 final mixer}) is run-to-run racy on XPU for row widths
# > 64: the oneDNN accumulation re-samples its split-K draw every execution,
# which is what made PIECEWISE-256 prefill replay diverge from eager
# (bitwise-equal inputs, few-ULP output flips). Replacing the single GEMM
# with two K=5120 GEMMs over contiguous weight halves is bitwise-deterministic
# in eager, fresh-clone and graph capture+replay at every width tested
# (M in {64, 128, 256}; 0/512 flips at M=256). See
# docs/HC_DOWN_GEMM_FIX_RECON.md for the offline campaign.
#
# The v0.32 Torch 2.14 BF16 path also changes valid outputs at M=1.
# DFlash2 therefore uses an explicitly ordered projection at every width.
# The legacy target-only split policy below remains unchanged.
# CUDA is untouched: the halves are prepared only on XPU
# (UnquantizedLinearMethod.process_weights_after_loading calls the
# ``hc_ksplit_prepare`` marker set below), so the split branch is never
# armed on non-XPU platforms.
# ---------------------------------------------------------------------------
_HC_KSPLIT_MAX_STOCK_WIDTH = 64
_HC_KSPLIT_WIDTHS_LOGGED: set[int] = set()


# ---------------------------------------------------------------------------
# Gated-residual variant
# ---------------------------------------------------------------------------
class GatedResidual(nn.Module):
    """Gated HyperConnection with learnable low-rank mixing and injection.

    ``combine_and_mix()`` runs the pre pipeline (grouped GemmaRMSNorm -> merged
    low-rank down+inject GEMM -> silu -> up GEMM -> sigmoid -> gated mean
    over the HC streams). When passed a pending block output, it fuses its
    residual combine with the RMSNorm. A missing injection selects unit-weight
    combine. Final mixers use ``use_combine=False`` and do not produce a new
    injection.

    Weights: the norm owns the grouped GemmaRMSNorm affine; the projections
    are vLLM Linear modules (merged replicated linear for down+inject), so
    GEMM dispatch (e.g. the low-latency skinny GEMM) applies through the
    standard quant_method mechanism.
    """

    def __init__(
        self,
        config: HyperConnectionConfig,
        use_combine: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.lora_rank = config.hc_lowrank
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.use_combine = use_combine
        self._astra_hc_up_weight = None
        self._astra_hc_up_prepared = False
        # B70-TC: name for the opaque compile ops (no-op unless B70_TORCH_COMPILE=1).
        self._b70_layer_name = prefix
        b70_compile_ops.register(prefix, self)
        spec = get_current_vllm_config().speculative_config
        self._ordered_hc = (
            spec is not None
            and spec.method == "dflash"
            and spec.draft_model_config.architectures == ["DFlash2DraftModel"]
            and config.params_dtype == torch.bfloat16
            and _is_xpu()
        )
        if self._ordered_hc:
            logger.info_once(
                "DFlash2 target HC uses fixed-order BF16 down projection "
                "with FP32 accumulation"
            )

        norm_size = (
            self.hyper_hidden_size if config.hc_per_branch_norm else config.hidden_size
        )
        group_size = config.hidden_size if config.hc_per_branch_norm else None
        # Normalize each H-sized HC stream independently while retaining a
        # separate affine weight for every element of the HC*H layout.
        self.hc_norm = GroupedGemmaRMSNorm(
            norm_size,
            eps=config.rms_norm_eps,
            group_size=group_size,
            dtype=config.params_dtype,
        )

        # -- vLLM Linear weights --------------------------------------------
        # The merged skinny-GEMM shape is physically padded to 16 rows to ensure
        # good alignment and performant implementation chosen by CuBLAS heuristics.
        self.pad_size = (-(self.lora_rank + self.hc_count)) % 16 if use_combine else 0
        if use_combine:
            self.input_mix_weight_down_block_inject = MergedColumnParallelLinear(
                self.hyper_hidden_size,
                [self.lora_rank, self.hc_count]
                + ([self.pad_size] if self.pad_size else []),
                bias=False,
                params_dtype=config.params_dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "input_mix_weight_down_block_inject"),
                return_bias=False,
                disable_tp=True,
            )
        else:
            self.input_mix_weight_down = ReplicatedLinear(
                self.hyper_hidden_size,
                self.lora_rank,
                bias=False,
                params_dtype=config.params_dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "input_mix_weight_down"),
                return_bias=False,
            )
        self.input_mix_weight_up = ReplicatedLinear(
            self.lora_rank,
            self.hyper_hidden_size,
            bias=False,
            params_dtype=config.params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "input_mix_weight_up"),
            return_bias=False,
        )

        # -- XPU deterministic K-split state --------------------------------
        # Contiguous K=5120 halves of the down weight, prepared once at load
        # time on XPU only (see _prepare_kdown_ksplit). ``None`` everywhere
        # else, which keeps the stock single GEMM armed.
        self._kdown_lo: torch.Tensor | None = None
        self._kdown_hi: torch.Tensor | None = None
        self._kdown_k = 0
        # B70-K1 int8 down weight (B70_HC_DOWN_INT8=1), prepared with the K-split halves.
        self._kdown_q: torch.Tensor | None = None
        self._kdown_qs: torch.Tensor | None = None
        down_linear = (
            self.input_mix_weight_down_block_inject
            if use_combine
            else self.input_mix_weight_down
        )
        # Marker consumed by
        # UnquantizedLinearMethod.process_weights_after_loading (XPU only).
        down_linear.hc_ksplit_prepare = self._prepare_kdown_ksplit

    def _prepare_kdown_ksplit(self) -> None:
        """One-time contiguous K-half prep for the deterministic XPU split."""
        if _B70_HC_DOWN_INT8 and self._kdown_q is None:
            self._prepare_kdown_int8()
        if self._ordered_hc or self._kdown_lo is not None:
            return
        down_linear = (
            self.input_mix_weight_down_block_inject
            if self.use_combine
            else self.input_mix_weight_down
        )
        weight = down_linear.weight
        if weight is None or weight.ndim != 2 or weight.shape[1] % 2 != 0:
            logger.debug(
                "HC down GEMM %s: unexpected weight geometry %s; K-split left disarmed",
                type(down_linear).__name__,
                None if weight is None else tuple(weight.shape),
            )
            return
        self._kdown_k = weight.shape[1] // 2
        self._kdown_lo = weight.data[:, : self._kdown_k].contiguous()
        self._kdown_hi = weight.data[:, self._kdown_k :].contiguous()
        logger.debug(
            "HC down GEMM: K-split halves prepared (K=%d -> 2x%d, N=%d)",
            weight.shape[1],
            self._kdown_k,
            weight.shape[0],
        )

    def _prepare_kdown_int8(self) -> None:
        """B70-K1: int8 copy of the down weight for b70_int8_dense.hc_down (load time, before capture)."""
        down_linear = (
            self.input_mix_weight_down_block_inject
            if self.use_combine
            else self.input_mix_weight_down
        )
        weight = getattr(down_linear, "weight", None)
        if (
            weight is None
            or weight.ndim != 2
            or weight.dtype != torch.bfloat16
            or weight.shape[1] % 32
            or weight.shape[0] % 16
            or weight.shape[0] < 32
        ):
            logger.warning(
                "B70-K1 HC down int8: unsupported weight %s; keeping the BF16 path",
                None if weight is None else (tuple(weight.shape), weight.dtype),
            )
            return
        if _B70_HC_DOWN_INT8_STREAM:
            if weight.shape[1] % (self.hc_count * _B70_HC_DOWN_INT8_SPS * 64):
                logger.warning(
                    "B70-K2 HC down int8 stream scales: K=%d does not split; BF16 path",
                    weight.shape[1],
                )
                return
            self._kdown_q, self._kdown_qs = quantize_weight_int8_grouped(
                weight.data, self.hc_count
            )
            return
        try:
            from vllm_xpu_kernels import b70_int8_dense as k1
        except ImportError:
            logger.warning("B70-K1 HC down int8: vllm_xpu_kernels.b70_int8_dense missing")
            return
        if not hasattr(k1, "hc_down") or not hasattr(torch.ops._xpu_C, "b70_w8a8_fused_gemm"):
            logger.warning("B70-K1 HC down int8: wheel lacks hc_down / b70_w8a8_fused_gemm")
            return
        self._kdown_q, self._kdown_qs = k1.quantize_weight_int8(weight.data)

    def _down_int8_takes(self, xn: torch.Tensor) -> bool:
        return (
            self._kdown_q is not None
            and xn.dtype == torch.bfloat16
            and xn.shape[0] <= _B70_HC_DOWN_INT8_MAX_ROWS
        )

    def _log_stream_int8(self) -> None:
        global _B70_HC_DOWN_INT8_LOGGED
        if not _B70_HC_DOWN_INT8_LOGGED:
            _B70_HC_DOWN_INT8_LOGGED = True
            logger.info(
                "B70-K2: HC down int8 with per-(token, stream) and per-(row, stream) "
                "scales on (Triton split-K, %d splits per stream, batch-invariant; "
                "rows <= %d; N=%d K=%d)",
                _B70_HC_DOWN_INT8_SPS,
                _B70_HC_DOWN_INT8_MAX_ROWS,
                self._kdown_q.shape[0],
                self._kdown_q.shape[1],
            )

    def _k2_capture(self, xn: torch.Tensor) -> None:
        """B70-K2 diagnostics: save HC-down inputs (eager only, TP rank 0)."""
        import re

        if hasattr(torch, "xpu") and torch.xpu.is_current_stream_capturing():
            return
        from vllm.distributed import get_tensor_model_parallel_rank

        if get_tensor_model_parallel_rank() != 0:
            return
        st = self.__dict__.setdefault("_k2_cap", {"dec": [], "n": 0, "pre": False})
        name = self._b70_layer_name or "hc"
        m = re.search(r"layers\.(\d+)\.", name)
        rows = xn.shape[0]
        os.makedirs(_B70_K2_CAP_DIR, exist_ok=True)
        tag = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
        if rows <= 64 and st["n"] < _B70_K2_CAP_DECODE:
            st["dec"].append(xn.detach().reshape(rows, -1).to("cpu"))
            st["n"] += rows
            if st["n"] >= _B70_K2_CAP_DECODE:
                torch.save(
                    {"xn": torch.cat(st["dec"])[:_B70_K2_CAP_DECODE], "module": name,
                     "kind": "decode"},
                    os.path.join(_B70_K2_CAP_DIR, f"{tag}__decode.pt"),
                )
                st["dec"] = []
        sampled = m is None or int(m.group(1)) % _B70_K2_CAP_EVERY == 0
        if rows >= 512 and sampled and not st["pre"]:
            st["pre"] = True
            torch.save(
                {"xn": xn.detach().reshape(rows, -1)[:_B70_K2_CAP_PREFILL].to("cpu"),
                 "module": name, "kind": "prefill"},
                os.path.join(_B70_K2_CAP_DIR, f"{tag}__prefill.pt"),
            )

    def _stream_quant(self, xn: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """The producer's per-(token, stream) int8 rows, or quantize here."""
        pq = b70_pq.take_grouped(xn, self.hc_count)
        if pq is None:
            pq = group_quant(xn.contiguous(), self.hc_count)
        return pq

    def _stream_takes(self, rows: int) -> bool:
        return (
            _B70_HC_DOWN_INT8_STREAM
            and self._kdown_q is not None
            and 0 < rows <= _B70_HC_DOWN_INT8_MAX_ROWS
            and not torch.compiler.is_compiling()
        )

    def _down_int8(self, xn: torch.Tensor) -> torch.Tensor:
        global _B70_HC_DOWN_INT8_LOGGED
        if _B70_HC_DOWN_INT8_STREAM:
            self._log_stream_int8()
            shape = xn.shape
            xq, xs = self._stream_quant(xn.reshape(-1, shape[-1]))
            out = hc_down_i8s(
                xq, xs, self._kdown_q, self._kdown_qs, _B70_HC_DOWN_INT8_SPS
            )
            return out.view(*shape[:-1], out.shape[-1])
        from vllm_xpu_kernels import b70_int8_dense as k1

        if not _B70_HC_DOWN_INT8_LOGGED:
            _B70_HC_DOWN_INT8_LOGGED = True
            logger.info(
                "B70-K1: HC down int8 on (per-token int8 x int8 weight, batch-invariant; "
                "rows <= %d; N=%d K=%d)",
                _B70_HC_DOWN_INT8_MAX_ROWS,
                self._kdown_q.shape[0],
                self._kdown_q.shape[1],
            )
        shape = xn.shape
        out = k1.hc_down(xn.reshape(-1, shape[-1]), self._kdown_q, self._kdown_qs)
        return out.view(*shape[:-1], out.shape[-1])

    def _mix_up(self, lora: torch.Tensor, normalized: torch.Tensor) -> torch.Tensor:
        if self._ordered_hc and not self._astra_hc_up_prepared:
            from vllm_xpu_kernels.astra_structural import prepare_hc_weight

            # Resolve the supported checkpoint layout before graph capture.
            # This aliases the loaded weight and allocates no duplicate table.
            self._astra_hc_up_weight = prepare_hc_weight(self.input_mix_weight_up.weight)
            self._astra_hc_up_prepared = True
        if self._astra_hc_up_weight is not None and 1 <= lora.shape[0] <= 8:
            from vllm_xpu_kernels.astra_structural import hc_up_mix

            return hc_up_mix(lora, self._astra_hc_up_weight, normalized)
        gate = self.input_mix_weight_up(lora)
        if (
            self.__dict__.get("_b70_i8_block", False)
            and normalized.dtype == torch.bfloat16
            and b70_pq.emits("hc_block", normalized.shape[0])
        ):
            # B70-K2: the gated mix also writes the block input's int8 rows for its
            # W1 consumer (in_proj_qkvz / qkv_proj); the bf16 output is unchanged.
            block_input, xq, xs = hc_gate_mix_quant(normalized, gate, self.hc_count)
            return b70_pq.attach(block_input, xq, xs, "hc_block")
        return hc_gate_mix(normalized, gate, self.hc_count)

    def _i8_lora_emits(self, xn: torch.Tensor) -> bool:
        """B70-K2: the HC up GEMM is W1 int8 and not the astra BF16 fused up."""
        rows = xn.shape[0]
        return (
            b70_pq.site_on("hc_up")
            and xn.dtype == torch.bfloat16
            and not (self._astra_hc_up_weight is not None and 1 <= rows <= 8)
            and b70_pq.consumer_is_w1(self.input_mix_weight_up)
            and b70_pq.emits("hc_up", rows)
        )

    def _down_gemm(self, xn: torch.Tensor) -> torch.Tensor:
        """Use ordered DFlash2 projection or the legacy target-only policy."""
        if self._down_int8_takes(xn):
            return self._down_int8(xn)
        if self._ordered_hc and xn.shape[0] <= _B70_HC_ORDERED_MAX_ROWS:
            from .b70_hc_gemm import ordered_hc_gemm

            linear = (
                self.input_mix_weight_down_block_inject
                if self.use_combine
                else self.input_mix_weight_down
            )
            valid = self.lora_rank + (self.hc_count if self.use_combine else 0)
            return ordered_hc_gemm(xn, linear.weight, valid)
        if (
            self._kdown_lo is not None
            and _is_xpu()
            and xn.shape[0] > _HC_KSPLIT_MAX_STOCK_WIDTH
        ):
            width = xn.shape[0]
            if width not in _HC_KSPLIT_WIDTHS_LOGGED:
                _HC_KSPLIT_WIDTHS_LOGGED.add(width)
                logger.info(
                    "HC down GEMM: deterministic K-split engaged at row width "
                    "%d (stock single GEMM only at widths <= %d)",
                    width,
                    _HC_KSPLIT_MAX_STOCK_WIDTH,
                )
            return F.linear(xn[..., : self._kdown_k], self._kdown_lo) + F.linear(
                xn[..., self._kdown_k :], self._kdown_hi
            )
        if self.use_combine:
            return self.input_mix_weight_down_block_inject(xn)
        return self.input_mix_weight_down(xn)

    def _down_and_silu(
        self, xn: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """(hc_silu(lora), injection) of the down projection.

        B70-V2-HC fuses the silu when the ordered projection takes the parallel
        route; otherwise the stock sequence runs."""
        global _B70_HC_SILU_FUSE_LOGGED
        if _B70_K2_CAP_DIR:
            self._k2_capture(xn)
        if self._stream_takes(xn.shape[0]) and xn.dim() == 2:
            # B70-K2: per-stream int8 down + silu (+ HC up int8 rows) in the finish.
            self._log_stream_int8()
            xq, xs = self._stream_quant(xn)
            quant_lora = self._i8_lora_emits(xn)
            down, lora, lq, ls = hc_down_i8s_silu(
                xq,
                xs,
                self._kdown_q,
                self._kdown_qs,
                self.lora_rank,
                self.hc_count,
                quant_lora,
                _B70_HC_DOWN_INT8_SPS,
            )
            if lq is not None:
                b70_pq.attach(lora, lq, ls, "hc_up")
            if self.use_combine:
                return lora, down[:, self.lora_rank : self.lora_rank + self.hc_count]
            return lora, None
        if (
            _B70_HC_SILU_FUSE
            and self._ordered_hc
            and xn.shape[0] <= _B70_HC_ORDERED_MAX_ROWS
            and not self._down_int8_takes(xn)
        ):
            from .b70_hc_gemm import parallel_hc_gemm_silu, takes_parallel_route

            linear = (
                self.input_mix_weight_down_block_inject
                if self.use_combine
                else self.input_mix_weight_down
            )
            valid = self.lora_rank + (self.hc_count if self.use_combine else 0)
            if (
                xn.dtype == linear.weight.dtype == torch.bfloat16
                and takes_parallel_route(xn, linear.weight, valid)
            ):
                if not _B70_HC_SILU_FUSE_LOGGED:
                    _B70_HC_SILU_FUSE_LOGGED = True
                    logger.info(
                        "B70-V2-HC: hc_silu fused into the parallel HC down finish "
                        "(B70_HC_SILU_FUSE=1)"
                    )
                if self._i8_lora_emits(xn):
                    down, lora, xq, xs = parallel_hc_gemm_silu_quant(
                        xn, linear.weight, valid, self.lora_rank, self.hc_count
                    )
                    b70_pq.attach(lora, xq, xs, "hc_up")
                else:
                    down, lora = parallel_hc_gemm_silu(
                        xn, linear.weight, valid, self.lora_rank, self.hc_count
                    )
                if self.use_combine:
                    injection = down[:, self.lora_rank : self.lora_rank + self.hc_count]
                    return lora, injection
                return lora, None
        if self.use_combine:
            # produce injection logits for combine
            split_sizes = [self.lora_rank, self.hc_count, self.pad_size]
            down_and_injection = self._down_gemm(xn)
            lora, injection, _ = down_and_injection.split(split_sizes, dim=-1)
        else:
            lora = self._down_gemm(xn)
            injection = None
        if self._i8_lora_emits(xn):
            silu, xq, xs = hc_silu_quant(lora, self.hc_count)
            return b70_pq.attach(silu, xq, xs, "hc_up"), injection
        return hc_silu(lora, self.hc_count), injection

    def mix(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if b70_compile_ops.ENABLED and torch.compiler.is_compiling():
            # B70-TC: row-count dispatch runs at call time, not at trace time.
            block_input, injection = torch.ops.vllm.b70_hc_mix(
                hidden_states, self._b70_layer_name
            )
            return hidden_states, block_input, (injection if self.use_combine else None)
        if self._ordered_hc:
            from .b70_c1_graph import replay_hc

            return replay_hc(self, "mix", self._mix_eager, (hidden_states,))
        return self._mix_eager(hidden_states)

    def _mix_eager(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self._stream_takes(hidden_states.shape[0]) and hidden_states.dim() == 2:
            xn, xq, xs = grouped_gemma_rmsnorm_quant(
                hidden_states,
                self.hc_norm.weight,
                self.config.rms_norm_eps,
                self.hc_count,
            )
            b70_pq.attach_grouped(xn, xq, xs)
        else:
            xn = grouped_gemma_rmsnorm(
                hidden_states,
                self.hc_norm.weight,
                self.config.rms_norm_eps,
                self.hc_count,
            )

        lora, injection = self._down_and_silu(xn)
        block_input = self._mix_up(lora, xn)

        return hidden_states, block_input, injection

    def combine_and_mix(
        self,
        hidden_states: torch.Tensor,
        prev_block_output: torch.Tensor,
        prev_injection: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if b70_compile_ops.ENABLED and torch.compiler.is_compiling():
            out, block_input, injection = torch.ops.vllm.b70_hc_combine_and_mix(
                hidden_states, prev_block_output, prev_injection, self._b70_layer_name
            )
            return out, block_input, (injection if self.use_combine else None)
        if self._ordered_hc:
            from .b70_c1_graph import replay_hc

            return replay_hc(
                self,
                "combine_and_mix",
                self._combine_and_mix_eager,
                (hidden_states, prev_block_output, prev_injection),
            )
        return self._combine_and_mix_eager(
            hidden_states, prev_block_output, prev_injection
        )

    def _combine_and_mix_eager(
        self,
        hidden_states: torch.Tensor,
        prev_block_output: torch.Tensor,
        prev_injection: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Consume a pending combine, then prepare the next block input.

        ``hidden_states`` is the multi-stream state from before the pending
        block's mix. Its combine with ``block_output`` is fused with this
        module's input RMSNorm. A missing injection applies the block output
        to every stream with unit weight.
        """
        if self._stream_takes(hidden_states.shape[0]):
            hidden_states, xn, xq, xs = hc_combine_norm_quant(
                hidden_states,
                prev_block_output,
                prev_injection,
                self.hc_norm.weight,
                self.config.rms_norm_eps,
                self.hc_count,
            )
            b70_pq.attach_grouped(xn, xq, xs)
        else:
            hidden_states, xn = hc_combine_norm(
                hidden_states,
                prev_block_output,
                prev_injection,
                self.hc_norm.weight,
                self.config.rms_norm_eps,
                self.hc_count,
            )

        lora, injection = self._down_and_silu(xn)
        block_input = self._mix_up(lora, xn)

        return hidden_states, block_input, injection

    def combine(
        self,
        hidden_states: torch.Tensor,
        block_output: torch.Tensor,
        injection: torch.Tensor | None,
    ) -> torch.Tensor:
        return hc_combine(hidden_states, block_output, injection, self.hc_count)

    @property
    def hyper_hidden_size(self) -> int:
        return self.hc_count * self.hidden_size


__all__ = [
    "GatedResidual",
    "GroupedGemmaRMSNorm",
    "HyperConnectionConfig",
]
