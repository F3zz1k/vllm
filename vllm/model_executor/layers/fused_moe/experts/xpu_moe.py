# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import torch

import vllm.model_executor.layers.fused_moe.modular_kernel as mk
from vllm.model_executor.layers.fused_moe.activation import MoEActivation
from vllm.model_executor.layers.fused_moe.config import (
    FusedMoEConfig,
    FusedMoEParallelConfig,
    FusedMoEQuantConfig,
)
from vllm.model_executor.layers.fused_moe.topk_weight_and_reduce import (
    TopKWeightAndReduceNoOP,
)
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8Dynamic128Sym,
    kFp8DynamicTensorSym,
    kFp8Static128BlockSym,
    kFp8StaticTensorSym,
    kInt4Static,
    kInt4Static32,
    kMxfp4Dynamic,
    kMxfp4Static,
    kMxfp8Dynamic,
    kMxfp8Static,
)
from vllm.platforms import current_platform

if current_platform.is_xpu():
    from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe


def prepare_fp8_moe_layer_for_xpu(
    w13: torch.Tensor,
    w13_scale: torch.Tensor,
    w2: torch.Tensor,
    w2_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if w13_scale is not None and w13_scale.ndim == 3:
        w13_scale = w13_scale.transpose(-1, -2).contiguous()
    if w2_scale is not None and w2_scale.ndim == 3:
        w2_scale = w2_scale.transpose(-1, -2).contiguous()
    return (
        w13.transpose(-1, -2).contiguous(),
        w13_scale,
        w2.transpose(-1, -2).contiguous(),
        w2_scale,
    )


class XPUExperts(mk.FusedMoEExpertsModular):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config,
            quant_config,
            max_num_tokens,
            num_dispatchers,
        )
        self.gemm1_clamp_limit = quant_config.gemm1_clamp_limit
        self.fused_moe_impl: XpuFusedMoe | None = None
        is_xe2_or_xe3 = torch.ops._xpu_C.is_xe2_arch() or torch.ops._xpu_C.is_xe3_arch()
        if not is_xe2_or_xe3:
            raise NotImplementedError(
                "XPUExperts is only supported on Intel Xe2/Xe3 GPUs"
            )
        self._expects_unquantized_inputs = is_xe2_or_xe3

    @property
    def expects_unquantized_inputs(self) -> bool:
        return self._expects_unquantized_inputs

    @staticmethod
    def activation_format() -> mk.FusedMoEActivationFormat:
        return mk.FusedMoEActivationFormat.Standard

    @staticmethod
    def _supports_current_device() -> bool:
        return current_platform.is_xpu()

    @staticmethod
    def _supports_no_act_and_mul() -> bool:
        return True

    @staticmethod
    def _supports_activation(activation: MoEActivation) -> bool:
        return activation in [
            MoEActivation.SILU,
            MoEActivation.GELU,
            MoEActivation.GELU_TANH,
            MoEActivation.SWIGLUOAI,
            MoEActivation.RELU2_NO_MUL,
        ]

    @staticmethod
    def _supports_parallel_config(moe_parallel_config: FusedMoEParallelConfig) -> bool:
        return True

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (None, None),
            (kFp8StaticTensorSym, None),
            (kFp8StaticTensorSym, kFp8DynamicTensorSym),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A

    def finalize_weight_and_reduce_impl(self) -> mk.TopKWeightAndReduce:
        return TopKWeightAndReduceNoOP()

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        workspace1 = (0,)
        workspace2 = (0,)
        output = (M, K)
        return (workspace1, workspace2, output)

    def apply(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
        global_num_experts: int,
        expert_map: torch.Tensor | None,
        a1q_scale: torch.Tensor | None,
        a2_scale: torch.Tensor | None,
        workspace13: torch.Tensor,
        workspace2: torch.Tensor,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        apply_router_weight_on_input: bool,
    ):
        if getattr(self, "b70_hotcold", None) is not None:
            self.b70_apply_hotcold(
                output, hidden_states, w1, w2, topk_weights, topk_ids, activation
            )
            return
        if self.fused_moe_impl is None:
            topk = topk_ids.size(-1)
            if (
                self.quant_config is not None
                and self.quant_config.weight_quant_dtype == "mxfp4"
            ):
                w1 = w1.view(torch.float4_e2m1fn_x2)
                w2 = w2.view(torch.float4_e2m1fn_x2)
            self.fused_moe_impl = XpuFusedMoe(
                w13=w1,
                w13_scales=self.w1_scale,
                w13_bias=self.w1_bias,
                w2=w2,
                w2_scales=self.w2_scale,
                w2_bias=self.w2_bias,
                n_experts_per_token=topk,
                activation=activation.value,
                num_experts=self.moe_config.num_local_experts,
                ep_rank=self.moe_config.ep_rank,
                ep_size=self.moe_config.ep_size,
                expert_map=expert_map,
                gemm1_clamp_limit=self.gemm1_clamp_limit,
            )
        assert self.fused_moe_impl is not None
        if self._b70_resident_decode_op(
            output, hidden_states, topk_weights, topk_ids, activation, expert_map
        ):
            return
        from vllm.model_executor.layers.fused_moe import b70_moe_kcut as _kcut

        # B70-V2-MOE: rows_per_expert zeroed by the sentinel kernel (None: stock).
        rows_per_expert = None
        if (self.moe_config.ep_size > 1
                and getattr(self.fused_moe_impl, "_astra_groups", None) is None):
            # Avoid topk_ids = -1, remap them to an
            # expert not owned by this rank so they are skipped.
            sentinel = global_num_experts - 1 if self.moe_config.ep_rank == 0 else 0
            if _kcut.enabled():
                topk_ids, rows_per_expert = _kcut.sentinel_and_zero(
                    topk_ids, sentinel, self.fused_moe_impl.num_experts
                )
            else:
                topk_ids = torch.where(topk_ids < 0, sentinel, topk_ids)
        from vllm.model_executor.layers.fused_moe import b70_moe_w4a8 as _w4a8

        if _w4a8.maybe_apply(
            self.fused_moe_impl, output, hidden_states, topk_weights, topk_ids,
            rows_per_expert=rows_per_expert,
        ):
            return
        if rows_per_expert is not None and _kcut.stock_kernel_ok(
            self.fused_moe_impl, a1q_scale
        ):
            _kcut.apply_kernel(
                self.fused_moe_impl, output, hidden_states, topk_weights, topk_ids,
                rows_per_expert, a1q_scale=a1q_scale,
            )
            return
        self.fused_moe_impl.apply(
            output=output,
            hidden_states=hidden_states,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            a1q_scale=a1q_scale,
        )

    # B70-V32-0054: None = not decided yet, False = stock path, tuple = the op's weight arguments
    _b70_decode_args = None

    def _b70_resident_decode_op(
        self, output, hidden_states, topk_weights, topk_ids, activation, expert_map
    ) -> bool:
        """B70-0054 on 0.32 (B70_MOE_DECODE_OP_RESIDENT=1 + B70_MOE_DECODE_OP=<lib>): decode batches of at most
        B70_MOE_DECODE_OP_MAX_ROWS tokens of the all-resident EP int4 MoE through b70moe.moe_decode. Refuses (stock
        path, one log line) under wu1ff's group ownership (weights re-laid out) or anything unexpected."""
        args = self._b70_decode_args
        if args is False:
            return False
        if args is None:
            if torch.xpu.is_current_stream_capturing():
                return False
            args = self._b70_resident_setup(activation, expert_map)
            self._b70_decode_args = args
            if args is False:
                return False
        from vllm.model_executor.layers.fused_moe import b70_decode_op as _op

        m = hidden_states.shape[0]
        if (m == 0 or m > _op.decode_op_max_rows() or m * topk_ids.size(-1) > 256
                or hidden_states.dtype != torch.bfloat16 or output.dtype != torch.bfloat16
                or not output.is_contiguous()):
            return False
        ids = topk_ids if topk_ids.dtype == torch.int32 else topk_ids.to(torch.int32)
        torch.ops.b70moe.moe_decode(output, hidden_states.contiguous(), topk_weights.contiguous(),
                                    ids.contiguous(), *args)
        return True

    def _b70_resident_setup(self, activation, expert_map):
        import os

        from vllm.logger import init_logger
        from vllm.model_executor.layers.fused_moe import b70_decode_op as _op

        logger = init_logger(__name__)
        if os.environ.get("B70_MOE_DECODE_OP_RESIDENT", "0") != "1":
            return False
        impl = self.fused_moe_impl
        why = None
        if getattr(impl, "_astra_groups", None) is not None:
            why = "wu1ff group ownership re-laid out the expert weights"
        elif _op.decode_op_max_rows() <= 0:
            why = "B70_MOE_DECODE_OP not set or the library did not load"
        elif not getattr(impl, "is_int4", False):
            why = "not an int4 MoE"
        elif activation.value != "silu" or self.gemm1_clamp_limit is not None:
            why = f"activation {activation.value} / clamp {self.gemm1_clamp_limit}"
        elif impl.w13_bias is not None or impl.w2_bias is not None:
            why = "expert biases"
        if why is None:
            w = (impl.w13, impl.gemm1_wei_scales, impl.w2, impl.gemm2_wei_scales)
            if any(t is None or not t.is_contiguous() or t.device.type != "xpu" for t in w):
                why = "weights not contiguous on the device"
            elif w[1].dtype != torch.bfloat16 or w[3].dtype != torch.bfloat16:
                why = f"scales {w[1].dtype}"
            elif w[0].element_size() != 1 or w[0].dim() != 3:
                why = f"w13 {w[0].dtype} {tuple(w[0].shape)}"
            elif (w[1].shape[-1] * 128 != w[0].shape[-1] * 2
                    or w[3].shape[-1] * 128 != w[2].shape[-1] * 2):
                why = "quant group size is not 128"
        if why is None:
            n_local = impl.num_experts
            hot_map = impl.expert_map
            if hot_map is None:
                hot_map = torch.arange(n_local, dtype=torch.int32, device=impl.w13.device)
            hot_map = hot_map.to(torch.int32).contiguous()
            if expert_map is not None and (expert_map.numel() != hot_map.numel() or not torch.equal(
                    expert_map.to(device=hot_map.device, dtype=torch.int32), hot_map)):
                why = "vLLM's expert_map differs from the stock kernel's"
            else:
                local = hot_map[hot_map >= 0]
                if local.numel() != n_local or not torch.equal(
                        torch.sort(local).values.cpu(), torch.arange(n_local, dtype=torch.int32)):
                    why = f"expert map names {local.numel()} local slots, kernel has {n_local}"
        if why is not None:
            if not getattr(XPUExperts, "_b70_resident_warned", False):
                XPUExperts._b70_resident_warned = True
                logger.warning("B70-0054 resident decode op: %s; stock path", why)
            return False
        cold_map = torch.full_like(hot_map, -1)
        if not getattr(XPUExperts, "_b70_resident_logged", False):
            XPUExperts._b70_resident_logged = True
            logger.info("B70-0054 resident decode op: on: %d local of %d experts, batches of <= %d tokens",
                        n_local, hot_map.numel(), _op.decode_op_max_rows())
        return (hot_map, cold_map, *w, *w)


class XPUExpertsFp8(XPUExperts):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config,
            quant_config,
            max_num_tokens,
            num_dispatchers,
        )

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kFp8StaticTensorSym, None),
            (kFp8StaticTensorSym, kFp8DynamicTensorSym),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A


class XPUExpertsMxFp8(XPUExpertsFp8):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config,
            quant_config,
            max_num_tokens,
            num_dispatchers,
        )
        assert quant_config.quant_dtype == "mxfp8"

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kMxfp8Static, None),
            (kMxfp8Static, kMxfp8Dynamic),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A


class XPUExpertsBlockFp8(XPUExperts):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config,
            quant_config,
            max_num_tokens,
            num_dispatchers,
        )

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kFp8Static128BlockSym, kFp8Dynamic128Sym),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A


class XPUExpertsWNA16(XPUExperts):
    """W4A16 INT4-symmetric MoE backed by `xpu_fused_moe(is_int4=True)`.

    Weight layout when `is_int4=True` (per `xpu_fused_moe` docstring):
        w13: [num_experts, 2*inter_size, hidden_size]   contiguous int4-packed
        w13_scales: [num_experts, 2*inter_size, hidden_size // group_size]
        w2:  [num_experts, hidden_size, inter_size]     contiguous int4-packed
        w2_scales:  [num_experts, hidden_size, inter_size // group_size]

    Pairs with `INCXPULinearMethod` for the linear layers; together they
    cover full-attn + MoE on Intel XPU end-to-end without IPEX.
    """

    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config,
            quant_config,
            max_num_tokens,
            num_dispatchers,
        )

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        return (weight_key, activation_key) in (
            (kInt4Static, None),
            (kInt4Static32, None),
        )


    def b70_apply_hotcold(
        self,
        output: torch.Tensor,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        activation: MoEActivation,
    ) -> None:
        """B70 0040: VRAM experts + pinned-host experts, summed (b70_hotcold.py)."""
        hot, cold = self.b70_hotcold.impls(
            w1, self.w1_scale, w2, self.w2_scale, topk_ids.size(-1),
            activation.value, {"gemm1_clamp_limit": self.gemm1_clamp_limit},
        )
        if self.moe_config.ep_size > 1:
            # 0.32: the stock remap of topk_ids == -1 (b70_hotcold.remap_unrouted).
            from vllm.model_executor.layers.fused_moe.b70_hotcold import remap_unrouted

            topk_ids = remap_unrouted(topk_ids, self.b70_hotcold.num_global, self.moe_config.ep_rank)
        if not (activation.value == "silu" and self.gemm1_clamp_limit is None
                and self.b70_hotcold.apply_decode_op(output, hidden_states, topk_weights, topk_ids)):
            from vllm.model_executor.layers.fused_moe import b70_moe_w4a8 as _w4a8

            if not _w4a8.maybe_apply(hot, output, hidden_states, topk_weights, topk_ids):
                hot.apply(output=output, hidden_states=hidden_states,
                          topk_weights=topk_weights, topk_ids=topk_ids)
            cold_out = torch.empty_like(output)
            self.b70_hotcold.apply_cold(cold, cold_out, hidden_states, topk_weights, topk_ids)
            output.add_(cold_out)
        counts = self.b70_hotcold.route_counts
        if counts is not None:
            # row 0: eager batches (prefill, mixed); row 1: captured graphs (decode)
            from vllm.model_executor.layers.fused_moe import b70_hotcold as _hc

            graph = torch.xpu.is_current_stream_capturing()
            # eager counting starts with the first real batch: the memory-profiling run must not pay for it
            if (graph and self.b70_hotcold.count_graphs) or (not graph and _hc._LIVE["on"]):
                row = counts[1 if graph else 0]
                # two kernels (inside the decode graphs too): negative (padding) ids land in the last
                # bucket via remainder, and the ones come from a shared static buffer
                ids = topk_ids.reshape(-1).remainder(row.shape[0])
                row.index_add_(0, ids, _hc.route_ones(ids.numel(), row.device))


class XPUExpertsMxFp4(XPUExperts):
    def __init__(
        self,
        moe_config: FusedMoEConfig,
        quant_config: FusedMoEQuantConfig,
        max_num_tokens: int | None = None,
        num_dispatchers: int | None = None,
    ):
        super().__init__(
            moe_config,
            quant_config,
            max_num_tokens,
            num_dispatchers,
        )

    def workspace_shapes(
        self,
        M: int,
        N: int,
        K: int,
        topk: int,
        global_num_experts: int,
        local_num_experts: int,
        expert_tokens_meta: mk.ExpertTokensMetadata | None,
        activation: MoEActivation,
    ) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
        # K = a1q.size(-1). When activations are pre-quantized packed mxfp4,
        # K is the packed hidden_size (= logical / 2); the kernel output is at
        # logical hidden_size (2 * K). When unquantized (bf16), K is already
        # the logical size.
        logical_K = K if self.expects_unquantized_inputs else 2 * K
        return (0,), (0,), (M, logical_K)

    @staticmethod
    def _supports_quant_scheme(
        weight_key: QuantKey | None,
        activation_key: QuantKey | None,
    ) -> bool:
        SUPPORTED_W_A = [
            (kMxfp4Static, None),
            (kMxfp4Static, kMxfp4Dynamic),
        ]
        return (weight_key, activation_key) in SUPPORTED_W_A
