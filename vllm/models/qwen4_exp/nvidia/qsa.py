# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""NVIDIA QSA owner with Triton kernels."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from typing import ClassVar, cast

import torch
from torch import nn
from transformers import Qwen4ExpTextConfig

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import VllmConfig
from vllm.config.cache import CacheConfig, CacheDType
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.attention import (
    set_default_quant_scales,
)
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import MRotaryEmbedding, get_rope
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention
from vllm.model_executor.models.qwen3_vl import mrope_positions_factor
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    WeightsMapper,
    extract_layer_index,
)
from vllm.platforms import current_platform
from vllm.platforms.interface import DeviceCapability
from vllm.utils.torch_utils import (
    kv_cache_dtype_str_to_dtype,
)
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionCGSupport,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.fa_utils import is_flash_attn_varlen_func_available
from vllm.v1.attention.backends.flash_attn import (
    FlashAttentionBackend,
    FlashAttentionImpl,
    FlashAttentionMetadata,
    FlashAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    FullAttentionSpec,
    KVCacheSpec,
    KVQuantMode,
    get_kv_quant_mode,
)

from ..common.qsa_cache import QSAForwardMetadata
from . import model
from .indexer_qsa import QSAIndexer

logger = init_logger(__name__)

# B70-IP-K6: int8 K/V with one fp32 scale per (token, kv head), stored inline
# after each head (TRITON_ATTN's int8_per_token_head layout).
QSA_INT8_PTH = "int8_per_token_head"
_QSA_CACHE_DTYPES = ("auto", "bfloat16", "fp8", "fp8_e4m3", QSA_INT8_PTH)


class Qwen4ExpQSAQKVIndexerLinear(MergedColumnParallelLinear):
    """Pack sharded Q/gate, K, V and replicated indexer Q/K in one GEMM."""

    def __init__(
        self,
        qkv_proj: QKVParallelLinear,
        index_size: int,
        quant_config: QuantizationConfig | None,
    ) -> None:
        self.num_kv_head_replicas = qkv_proj.num_kv_head_replicas
        super().__init__(
            input_size=qkv_proj.input_size,
            output_sizes=[*qkv_proj.output_sizes, index_size * qkv_proj.tp_size],
            bias=False,
            params_dtype=qkv_proj.params_dtype,
            quant_config=quant_config,
            prefix=qkv_proj.prefix,
        )

    def _load_shard(
        self,
        loader: Callable[..., None],
        param: nn.Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: int | tuple[int, ...] | None,
    ) -> None:
        tp_rank = self.tp_rank
        param_tp_rank = getattr(param, "tp_rank", None)
        if loaded_shard_id == 3:
            shard_rank = 0
        elif loaded_shard_id in (1, 2):
            shard_rank = tp_rank // self.num_kv_head_replicas
        else:
            shard_rank = tp_rank
        self.tp_rank = shard_rank
        if param_tp_rank is not None:
            param.tp_rank = shard_rank
        try:
            loader(param, loaded_weight, loaded_shard_id)
        finally:
            self.tp_rank = tp_rank
            if param_tp_rank is not None:
                param.tp_rank = param_tp_rank

    def weight_loader(self, param, loaded_weight, loaded_shard_id=None) -> None:
        self._load_shard(super().weight_loader, param, loaded_weight, loaded_shard_id)

    def weight_loader_v2(self, param, loaded_weight, loaded_shard_id=None) -> None:
        self._load_shard(
            super().weight_loader_v2, param, loaded_weight, loaded_shard_id
        )

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> Iterable[str]:
        def remap_shards():
            for name, weight in weights:
                shard_id = getattr(weight, "shard_id", None)
                if isinstance(shard_id, str):
                    weight = weight.detach()
                    weight.shard_id = {"q": 0, "k": 1, "v": 2}[shard_id]
                yield name, weight

        return super().load_weights(remap_shards())


def qsa_kv_cache_dtype(cache_config: CacheConfig, prefix: str) -> CacheDType:
    """The layer's KV cache dtype, honoring ``--kv-cache-dtype-skip-layers``."""
    if str(extract_layer_index(prefix)) in cache_config.kv_cache_dtype_skip_layers:
        return "auto"
    return cache_config.cache_dtype


class Qwen4ExpQSAMetadataBuilder(FlashAttentionMetadataBuilder):
    """Flash metadata supporting uniform decode and target-verify graphs."""

    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.UNIFORM_BATCH


class Qwen4ExpQSAFlashAttentionBackend(FlashAttentionBackend):
    """FullAttentionSpec backend used by the merged QSA owner."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    # fp8/fp8_e4m3: e4m3 bytes in a uint8 cache, written by reshape_and_cache
    # with the layer's per-tensor scales and dequantized on load inside the QSA
    # Triton kernel. flash-attn never runs over this cache, so its fp8 probe
    # does not apply (see supports_kv_cache_dtype and the impl constructor).
    # int8_per_token_head (B70-IP-K6): int8 K/V with inline fp32 per-(token,
    # head) scales, written by the fused prepare or the Triton per-token-head
    # writer and dequantized per token inside the QSA kernels.
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "fp8",
        "fp8_e4m3",
        "int8_per_token_head",
    ]

    @classmethod
    def customize_spec(cls, spec: AttentionSpec) -> AttentionSpec:
        """B70-IP-K6: the per-token-head packing of TRITON_ATTN (inline fp32
        scale after each K and V head), so page sizes, the hybrid page unify
        and the drafter's TRITON_ATTN pages agree. Other dtypes unchanged."""
        from vllm.v1.attention.backends.triton_attn import TritonAttentionBackend

        return TritonAttentionBackend.customize_spec(spec)

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        return kv_cache_dtype is None or kv_cache_dtype in cls.supported_kv_cache_dtypes

    @classmethod
    def supports_combination(
        cls,
        head_size: int,
        dtype: torch.dtype,
        kv_cache_dtype: CacheDType | None,
        block_size: int | None,
        use_mla: bool,
        has_sink: bool,
        use_sparse: bool,
        use_mm_prefix: bool,
        device_capability: DeviceCapability,
    ) -> str | None:
        # QSA dequantizes the fp8 KV in its own Triton kernel and never runs
        # flash-attn over the quantized cache, so the parent's fp8-KV rejection
        # does not apply and every combination it is handed is accepted here.
        return None

    @staticmethod
    def get_name() -> str:
        return "QWEN4_EXP_QSA_TRITON"

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None) -> list[int | MultipleOf]:
        # QSA consumes manager pages directly and does not use FA4 paged attention.
        return [MultipleOf(16)]

    @staticmethod
    def get_impl_cls() -> type[Qwen4ExpQSAFlashAttentionImpl]:
        return Qwen4ExpQSAFlashAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[Qwen4ExpQSAMetadataBuilder]:
        return Qwen4ExpQSAMetadataBuilder

    @classmethod
    def is_sparse(cls) -> bool:
        return True

    @classmethod
    def supports_kv_connector(cls) -> bool:
        return False


class Qwen4ExpQSAFlashAttentionImpl(FlashAttentionImpl):
    """Run paged sparse GQA with the QSA Triton kernel."""

    supports_dcp: bool = False
    supports_pcp: bool = False

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        # The parent constructor probes flash-attn for quantized-KV support and
        # raises where it is unavailable (sm120), but QSA dequantizes fp8 inside
        # its own Triton kernel and never runs flash-attn over the cache. Hand
        # the parent "auto" for that probe and restore the real dtype afterwards:
        # the parent only uses it there, and do_kv_cache_update reads the
        # attribute at call time.
        real_kv_cache_dtype = kv_cache_dtype
        if kv_cache_dtype in ("fp8", "fp8_e4m3", QSA_INT8_PTH):
            kv_cache_dtype = "auto"
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            sinks,
        )
        self.kv_cache_dtype = real_kv_cache_dtype
        if not is_flash_attn_varlen_func_available():
            raise NotImplementedError("Qwen4Exp QSA requires FlashAttention")
        if self.dcp_world_size != 1:
            raise NotImplementedError(
                "Qwen4Exp QSA does not support decode context parallelism"
            )
        if self.kv_cache_dtype not in _QSA_CACHE_DTYPES:
            raise NotImplementedError(
                "Qwen4Exp QSA requires a BF16, FP8-e4m3 or int8 per-token-head "
                "main KV cache"
            )
        self.supports_quant_query_input = False
        self._kv_quant_mode = get_kv_quant_mode(self.kv_cache_dtype)
        # B70-IP-K6: scale views over the inline scale bytes, rebuilt when the
        # bound cache changes; the runner drops them after profiling (it
        # clears ``_k_scale_cache``/``_v_scale_cache`` on every layer impl).
        self._k_scale_cache: torch.Tensor | None = None
        self._v_scale_cache: torch.Tensor | None = None
        self._pth_views: tuple | None = None
        if self._kv_quant_mode == KVQuantMode.INT8_PER_TOKEN_HEAD:
            logger.info_once(
                "B70-IP-K6: Qwen4Exp QSA main KV is int8 per-token-head "
                "(inline fp32 scales; int8->bf16 in the QSA kernels, K scale on "
                "scores, V scale on P)"
            )

    @property
    def is_int8_pth(self) -> bool:
        return self._kv_quant_mode == KVQuantMode.INT8_PER_TOKEN_HEAD

    def pth_kv_views(
        self, kv_cache: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """B70-IP-K6: (key, value, k_scale, v_scale) views of the int8 cache.

        Pure views (no fill: hybrid pages are shared with GDN state, B70-IP-K2);
        cached per bound cache tensor.
        """
        from .ops.qsa import qsa_pth_kv_views

        key = (kv_cache.data_ptr(), tuple(kv_cache.shape), tuple(kv_cache.stride()))
        if (
            self._pth_views is None
            or self._k_scale_cache is None
            or self._pth_views[0] != key
        ):
            views = qsa_pth_kv_views(kv_cache, self.head_size)
            self._k_scale_cache, self._v_scale_cache = views[2], views[3]
            self._pth_views = (key, views[0], views[1])
        return (
            self._pth_views[1],
            self._pth_views[2],
            self._k_scale_cache,
            self._v_scale_cache,
        )

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if not self.is_int8_pth:
            return super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
        # B70-IP-K6: unfused writer = TRITON_ATTN's per-token-head writer over
        # the same inline layout.
        from vllm.v1.attention.ops.triton_reshape_and_cache_flash import (
            triton_reshape_and_cache_flash_per_token_head_quant,
        )

        key_cache, value_cache, k_scale_cache, v_scale_cache = self.pth_kv_views(
            kv_cache
        )
        # key/value may be padded past slot_mapping (the C++ writer sizes its
        # grid by slot_mapping; the Triton one by key), so cut them here.
        num_slots = slot_mapping.shape[0]
        triton_reshape_and_cache_flash_per_token_head_quant(
            key[:num_slots],
            value[:num_slots],
            key_cache,
            value_cache,
            k_scale_cache,
            v_scale_cache,
            slot_mapping,
            kv_quant_mode=self._kv_quant_mode,
        )

    def forward_qsa(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: torch.Tensor,
        token_to_req: torch.Tensor,
        use_prefill_config: bool,
        output_gate: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del key, value
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("QSA does not support fused output quantization")
        if self.alibi_slopes is not None or self.sinks is not None:
            raise NotImplementedError("QSA does not support ALiBi or attention sinks")
        if self.sliding_window != (-1, -1):
            raise NotImplementedError("QSA does not support sliding-window attention")

        num_tokens = attn_metadata.num_actual_tokens
        output.zero_()
        if num_tokens == 0:
            return output

        topk_buffer = getattr(layer, "topk_indices_buffer", None)
        if topk_buffer is None:
            raise RuntimeError("QSA owner did not provide its top-k buffer")
        logical_indices = topk_buffer[:num_tokens]
        token_to_req = token_to_req[:num_tokens]
        k_scale = v_scale = None
        k_scale_cache = v_scale_cache = None
        if self.is_int8_pth:
            key_cache, value_cache, k_scale_cache, v_scale_cache = self.pth_kv_views(
                kv_cache
            )
        else:
            key_cache, value_cache = kv_cache.transpose(1, 2).split(
                self.head_size, dim=-1
            )
        if self.kv_cache_dtype in ("fp8", "fp8_e4m3"):
            # The cache is allocated as uint8; on sm_89+ reinterpret the e4m3
            # bytes (same itemsize, so shape and strides are preserved). Below
            # sm_89 the kernel byte-decodes the raw uint8 storage instead.
            if current_platform.has_device_capability(89):
                key_cache = key_cache.view(torch.float8_e4m3fn)
                value_cache = value_cache.view(torch.float8_e4m3fn)
            # Host-side per-tensor dequant scales (Python floats), as used by
            # other host-scale backends; folded into the kernel's scales.
            k_scale = layer._k_scale_float
            v_scale = layer._v_scale_float
        if query.dtype != torch.bfloat16 or key_cache.dtype not in (
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.uint8,
            torch.int8,
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA requires BF16 Q and BF16, FP8-e4m3 or int8 "
                "per-token-head K/V"
            )

        from .ops.qsa import qsa_sparse_paged_attention

        qsa_sparse_paged_attention(
            query[:num_tokens],
            key_cache,
            value_cache,
            logical_indices,
            attn_metadata.block_table,
            token_to_req,
            use_prefill_config,
            output[:num_tokens],
            k_scale=k_scale,
            v_scale=v_scale,
            output_gate=output_gate[:num_tokens],
            **(
                {"k_scale_cache": k_scale_cache, "v_scale_cache": v_scale_cache}
                if k_scale_cache is not None
                else {}
            ),
        )
        return output


def _qsa_eager_break(fn):
    """B70-QSA-T04: ``@eager_break_during_capture`` on ``_run_qsa``, unless
    B70_QSA_EAGER_BREAK=0.

    vLLM auto-enables breakable CUDA graphs for Qwen4Exp (vllm.py
    DEFAULT_BREAKABLE_CUDAGRAPH_ARCHITECTURES; the engine journal shows "Auto-
    enabling VLLM_USE_BREAKABLE_CUDAGRAPH=1"). The decorator then ends the graph
    segment at every QSA layer and replays the whole QSA body eagerly from
    Python on every step: 12 layers x (metadata reads, asserts, ~7-20 kernel
    launches incl. top-k). The QSA metadata builder declares UNIFORM_BATCH, so
    a capture of the body is supported in principle; B70_QSA_EAGER_BREAK=0 is
    the measurement arm that captures it into the decode segments instead.
    Decided at class definition (import time), like the decorator itself.
    """
    from .qsa_optimer import wrap_run_qsa

    fn = wrap_run_qsa(fn)  # B70-QSA-P: no-op unless B70_QSA_OPTIMER=1
    if os.environ.get("B70_QSA_EAGER_BREAK", "1") == "0":
        logger.info_once("B70-QSA: _run_qsa is captured into graph segments")
        return fn
    return eager_break_during_capture(fn)


class Qwen4ExpQSAAttention(Qwen3NextAttention, AttentionLayerBase):
    """Merged Qwen full-attention owner with a QSA index side branch."""

    supports_dcp = False

    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        config: Qwen4ExpTextConfig,
        layer_id: int,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        prefix: str = "",
    ) -> None:
        nn.Module.__init__(self)
        cache_config = vllm_config.cache_config
        model_config = vllm_config.model_config
        if cache_config is None:
            raise ValueError("Qwen4Exp QSA requires a paged KV cache")
        if model_config.dtype != torch.bfloat16:
            raise NotImplementedError("Qwen4Exp QSA currently requires BF16")
        if cache_config.cache_dtype not in _QSA_CACHE_DTYPES:
            raise NotImplementedError(
                "Qwen4Exp QSA requires a BF16, FP8-e4m3 or int8 per-token-head "
                "main KV cache"
            )
        if getattr(quant_config, "kv_cache_scheme", None) is not None:
            raise NotImplementedError("Qwen4Exp QSA does not support KV quantization")
        parallel_config = vllm_config.parallel_config
        if (
            parallel_config.prefill_context_parallel_size > 1
            or parallel_config.decode_context_parallel_size > 1
        ):
            raise NotImplementedError(
                "Qwen4Exp QSA does not support context parallelism"
            )
        if not getattr(config, "is_causal", True):
            raise NotImplementedError("Qwen4Exp QSA requires causal decoder attention")

        self.config = config
        self.hidden_size = int(config.hidden_size)
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = int(config.num_attention_heads)
        if self.total_num_heads % tp_size:
            raise ValueError("QSA attention heads must be divisible by TP size")
        self.num_heads = self.total_num_heads // tp_size
        # Decode/verify batches have at most 1 + num_spec query tokens per
        # request; use_prefill_config (max_query_len > this) steers the
        # config table. Shorter batches take the decode profile — harmless,
        # the difference is tile-shape tuning, not correctness.
        self._max_decode_query_len = 1 + vllm_config.num_speculative_tokens
        self.total_num_kv_heads = int(config.num_key_value_heads)
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size:
                raise ValueError("QSA KV heads must be divisible by TP size")
        elif tp_size % self.total_num_kv_heads:
            raise ValueError("TP size must be divisible by replicated QSA KV heads")
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = int(config.head_dim or self.hidden_size // self.total_num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        if self.dual_chunk_attention_config is not None:
            raise NotImplementedError("Qwen4Exp QSA does not support dual-chunk RoPE")
        # Qwen4Exp full-attention checkpoints always pack a sigmoid output
        # gate next to Q, even when an inherited config default says otherwise.
        self.attn_output_gate = True

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=False,
            quant_config=model.without_modelopt_fp4(quant_config),
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            reduce_results=reduce_results,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )
        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            max_position=config.max_position_embeddings,
            rope_parameters=config.rope_parameters,
            mrope_positions_factor=mrope_positions_factor(vllm_config),
        )
        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        mm_config = model_config.multimodal_config
        text_only = mm_config is None or mm_config.language_model_only
        mrope_section = getattr(self.rotary_emb, "mrope_section", None)
        supports_mrope = bool(
            type(self.rotary_emb) is MRotaryEmbedding
            and mrope_section
            and len(mrope_section) == 3
            and sum(mrope_section) == self.rotary_emb.rotary_dim // 2
            and getattr(self.rotary_emb, "mrope_interleaved", False)
        )
        supports_dtype = getattr(self.rotary_emb, "dtype", None) in (
            torch.float16,
            torch.bfloat16,
        )
        # B70-QSA: XPU runs the same Triton fused_qk_rmsnorm_rope_gate that
        # Qwen3NextAttention already enables there (qwen3_next.py).
        self.use_fused_qk_norm_rope_gate = (
            self.attn_output_gate
            and getattr(self.rotary_emb, "is_neox_style", False)
            and (current_platform.is_cuda() or current_platform.is_xpu())
            and supports_dtype
            and (text_only or supports_mrope)
        )

        self.layer_name = f"{prefix}.attn"
        self.attn_type = AttentionType.DECODER
        self.kv_cache_dtype = qsa_kv_cache_dtype(cache_config, prefix)
        self.kv_cache_torch_dtype = kv_cache_dtype_str_to_dtype(
            self.kv_cache_dtype, model_config
        )
        if self.kv_cache_torch_dtype not in (torch.bfloat16, torch.uint8, torch.int8):
            raise NotImplementedError(
                "Qwen4Exp QSA requires BF16, FP8-e4m3 (uint8) or int8 "
                "per-token-head cache storage"
            )
        self.kv_sharing_target_layer_name = None
        self.kv_cache = torch.tensor([])
        set_default_quant_scales(self, register_buffer=True)

        self.attn_backend = Qwen4ExpQSAFlashAttentionBackend
        self.impl = Qwen4ExpQSAFlashAttentionImpl(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
            None,
            None,
            self.kv_cache_dtype,
            None,
            AttentionType.DECODER,
            None,
        )
        self.indexer = QSAIndexer(
            vllm_config=vllm_config,
            config=config,
            layer_id=layer_id,
            rotary_emb=self.rotary_emb,
            quant_config=quant_config,
            prefix=f"{prefix}.indexer",
        )
        # One launch does the indexer prepare, the main QK-norm/RoPE/gate and
        # the main K/V cache write (see QSAIndexer.forward); otherwise all of
        # them take the separate kernels. The fused writer receives an fp8
        # pointer, which Triton cannot lower below sm_89.
        self.use_fused_qsa_prepare = (
            self.use_fused_qk_norm_rope_gate
            and self.indexer.use_fused_pre_indexer
            and (
                self.kv_cache_torch_dtype != torch.uint8
                or current_platform.has_device_capability(89)
            )
            # B70-QSA: B70_QSA_FUSED_PREPARE=0 forces the unfused kernels
            # (A/B and fallback if the fused launch misbehaves on a backend).
            and os.environ.get("B70_QSA_FUSED_PREPARE", "1") != "0"
        )
        logger.info_once(
            "B70-QSA: Qwen4Exp QSA attention on %s: fused_qk_norm_rope_gate=%s "
            "fused_qsa_prepare=%s kv_cache_dtype=%s",
            current_platform.device_name,
            self.use_fused_qk_norm_rope_gate,
            self.use_fused_qsa_prepare,
            self.kv_cache_dtype,
        )
        self.fuse_indexer_projection = vllm_config.lora_config is None
        if self.fuse_indexer_projection:
            self.index_qk_size = self.indexer.index_qk_proj.output_size
            self.qkv_proj = Qwen4ExpQSAQKVIndexerLinear(
                self.qkv_proj,
                self.index_qk_size,
                model.without_modelopt_fp4(quant_config),
            )
            del self.indexer.index_qk_proj

        max_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        # PACKED selection buffer: the trailing column holds each row's
        # valid-entry count (written by the expand kernel) — never a token
        # index; the sparse attention kernel reads it as its loop bound.
        # MTP skip_topk steps reuse rows frozen from step 0; the count is
        # a row column, so compaction/reuse keep it paired with the content.
        self.register_buffer(
            "topk_indices_buffer",
            torch.empty(
                max_tokens,
                self.indexer.packed_output_width,
                dtype=torch.int32,
            ),
            persistent=False,
        )

        static_context = vllm_config.compilation_config.static_forward_context
        if self.layer_name in static_context:
            raise ValueError(f"Duplicate layer name: {self.layer_name}")
        static_context[self.layer_name] = self

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        mapper = None
        if self.fuse_indexer_projection:
            mapper = WeightsMapper(
                orig_to_new_stacked={
                    "indexer.index_qk_proj.": ("qkv_proj.", 3),
                }
            )
        return AutoWeightsLoader(self).load_weights(weights, mapper=mapper)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return self.attn_backend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        # The backend owns its packing (B70-IP-K6: inline per-token-head
        # scales; bf16/fp8 specs come back unchanged).
        return self.attn_backend.customize_spec(
            FullAttentionSpec(
                block_size=vllm_config.cache_config.block_size,
                num_kv_heads=self.num_kv_heads,
                head_size=self.head_dim,
                head_size_v=self.head_dim,
                dtype=self.kv_cache_torch_dtype,
                kv_quant_mode=get_kv_quant_mode(self.kv_cache_dtype),
            )
        )

    @_qsa_eager_break
    def _run_qsa(
        self,
        projected_qk: torch.Tensor,
        positions: torch.Tensor,
        query: torch.Tensor | None,
        key: torch.Tensor | None,
        value: torch.Tensor | None,
        output: torch.Tensor,
        output_gate: torch.Tensor | None,
        qkv: torch.Tensor,
    ) -> None:
        # query/key/value/output_gate are None when the fused prepare runs
        # inside the indexer launch.
        metadata = get_forward_context().attn_metadata
        if isinstance(metadata, list):
            metadata = metadata[0]
        if not isinstance(metadata, dict):
            output.zero_()
            return
        main_metadata = cast(FlashAttentionMetadata, metadata[self.layer_name])
        if self.kv_cache.numel() == 0:
            raise RuntimeError("QSA main K/V cache is not bound")

        num_tokens = main_metadata.num_actual_tokens
        side_metadata = cast(
            QSAForwardMetadata,
            metadata[self.indexer.raw_key_cache.prefix],
        )
        if side_metadata.num_actual_tokens != num_tokens:
            raise RuntimeError("QSA main and side metadata token counts disagree")
        selected, main_outputs = self.indexer(
            projected_qk,
            positions,
            self.topk_indices_buffer[:num_tokens],
            attn=self,
            qkv=qkv,
            slot_mapping=main_metadata.slot_mapping,
        )
        if selected.shape != (num_tokens, self.indexer.packed_output_width):
            raise RuntimeError("QSA indexer returned an invalid selection shape")
        impl = cast(Qwen4ExpQSAFlashAttentionImpl, self.impl)
        if main_outputs is None:
            assert key is not None and value is not None
            impl.do_kv_cache_update(
                self,
                key,
                value,
                self.kv_cache,
                main_metadata.slot_mapping,
            )
        else:
            query, output_gate = main_outputs
        assert query is not None and output_gate is not None
        impl.forward_qsa(
            self,
            query,
            key,
            value,
            self.kv_cache,
            main_metadata,
            output,
            token_to_req=side_metadata.token_to_req,
            use_prefill_config=main_metadata.max_query_len > self._max_decode_query_len,
            output_gate=output_gate,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        if self.fuse_indexer_projection:
            qkv, projected_qk = qkv.split(
                [2 * self.q_size + 2 * self.kv_size, self.index_qk_size], dim=-1
            )
        else:
            projected_qk, _ = self.indexer.index_qk_proj(hidden_states)
        num_tokens = hidden_states.shape[0]
        if not self.use_fused_qsa_prepare:
            q, k, v, gate = self._project_qkv_gate(qkv, positions)
            assert gate is not None
            query = q.view(num_tokens, self.num_heads, self.head_dim)
            key = k.view(num_tokens, self.num_kv_heads, self.head_dim)
            value = v.view(num_tokens, self.num_kv_heads, self.head_dim)
        else:
            # Norm/RoPE/gate and the K/V cache write happen inside _run_qsa.
            query = key = value = gate = None
        attn_output = qkv.new_empty(num_tokens, self.num_heads, self.head_dim)
        self._run_qsa(
            projected_qk,
            positions,
            query,
            key,
            value,
            attn_output,
            gate,
            qkv,
        )
        flat_output = attn_output.view(num_tokens, -1)
        output, _ = self.o_proj(flat_output)
        return output


__all__ = [
    "QSAIndexer",
    "Qwen4ExpQSAAttention",
    "Qwen4ExpQSAFlashAttentionBackend",
    "Qwen4ExpQSAFlashAttentionImpl",
]
