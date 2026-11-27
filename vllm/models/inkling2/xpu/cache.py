# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-visit cache owners of Inkling2 (XPU port).

Every (layer, loop pass) visit owns two caches, both registered as attention
layers so vLLM's KV cache manager allocates, prefix-caches and offloads them:

* `Inkling2KVCache`: the visit's K/V (`SlidingWindowSpec` for the local
  layers, `FullAttentionSpec` for the global ones); metadata from
  `FlashAttentionBackend`'s builder.
* `Inkling2ConvState`: the short-conv inputs of the visit's four conv streams
  (K, V, attention output, MLP output) as a per-sequence state of the last
  `sconv_kernel_size - 1` tokens (a `MambaSpec`, so vLLM's hybrid allocator,
  prefix caching in align mode and the KV offload treat it like any recurrent
  state), the streams packed per KV head (`ref.ConvLayout`); metadata from the
  short-conv builder plus per-token request indices.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mamba.abstract import MambaBase
from vllm.models.inkling.amd.ops.sconv import sconv_seq_metadata
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.attention.backend import AttentionBackend, CommonAttentionMetadata
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.attention.backends.short_conv_attn import (
    ShortConvAttentionBackend,
    ShortConvAttentionMetadata,
    ShortConvAttentionMetadataBuilder,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    SlidingWindowSpec,
)

from .ops.ref import ConvLayout

_FP8_VIEW = {"fp8": torch.float8_e4m3fn, "fp8_e4m3": torch.float8_e4m3fn,
             "fp8_e5m2": torch.float8_e5m2}


def _register(prefix: str, layer: nn.Module) -> None:
    ctx = get_current_vllm_config().compilation_config.static_forward_context
    if prefix in ctx:
        raise ValueError(f"Duplicate layer name: {prefix}")
    ctx[prefix] = layer


def layer_metadata(prefix: str):
    """This step's metadata of cache owner `prefix` (None in a dummy run)."""
    md = get_forward_context().attn_metadata
    if not isinstance(md, dict):
        return None
    return md.get(prefix)


class Inkling2KVCache(nn.Module, AttentionLayerBase):
    def __init__(self, prefix: str, num_heads: int, num_kv_heads: int, head_dim: int,
                 sliding_window: int | None) -> None:
        super().__init__()
        self.prefix = prefix
        self.num_heads = num_heads  # read by the metadata builder
        self.head_size = head_dim
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        vllm_config = get_current_vllm_config()
        self.kv_cache_dtype = vllm_config.cache_config.cache_dtype
        self.dtype = kv_cache_dtype_str_to_dtype(self.kv_cache_dtype, vllm_config.model_config)
        # fp8 KV (`--kv-cache-dtype fp8` / `fp8_e4m3` / `fp8_e5m2`): vLLM allocates uint8 pages;
        # the conv kernel stores into, and the SYCL attention reads, the fp8 view. Per-tensor
        # scale 1.0: K is L2-clipped (<= qk_l2_clip_threshold, 22.6) and V RMS-normalized, so
        # both sit well inside e4m3's range.
        self.fp8_dtype = _FP8_VIEW.get(self.kv_cache_dtype)
        if self.kv_cache_dtype.startswith("fp8") and self.fp8_dtype is None:
            raise ValueError(f"Inkling2: unsupported KV cache dtype {self.kv_cache_dtype!r}")
        if self.fp8_dtype is not None:
            block = vllm_config.cache_config.block_size
            if block != 64:
                # The hybrid allocator raises the attention block until a page holds the
                # largest conv state (one hidden-size stream of 3 + num_speculative_tokens
                # bf16 rows); the SYCL kernels serve 64-token pages only.
                raise ValueError(
                    f"Inkling2 fp8 KV needs 64-token KV pages, but the conv state page sets "
                    f"{block} at this TP size and speculative length (an fp8 page of 64 tokens "
                    f"is smaller than the state); use bf16 KV here, a smaller TP size, or "
                    f"fewer speculative tokens")
        self.register_buffer("kv_scale", torch.ones(1, dtype=torch.float32), persistent=False)
        self.kv_cache = torch.tensor([])
        _register(prefix, self)

    def forward(self): ...

    def kv(self) -> torch.Tensor:
        """The cache the kernels read and write: the fp8 view of the uint8 pages under fp8 KV."""
        return self.kv_cache if self.fp8_dtype is None else self.kv_cache.view(self.fp8_dtype)

    def scales(self) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """(k_scale, v_scale) for the SYCL attention: the 1.0 tensors under fp8 KV, else None."""
        return (None, None) if self.fp8_dtype is None else (self.kv_scale, self.kv_scale)

    def get_attn_backend(self) -> type[AttentionBackend]:
        return FlashAttentionBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        block_size = vllm_config.cache_config.block_size
        if self.sliding_window is not None:
            return SlidingWindowSpec(block_size=block_size, num_kv_heads=self.num_kv_heads,
                                     head_size=self.head_dim, dtype=self.dtype,
                                     sliding_window=self.sliding_window)
        return FullAttentionSpec(block_size=block_size, num_kv_heads=self.num_kv_heads,
                                 head_size=self.head_dim, dtype=self.dtype)

    def view(self) -> torch.Tensor:
        """The visit's cache as [num_blocks, H, block_size, 2 * D]."""
        return self.kv()


@dataclass
class Inkling2ConvMetadata(ShortConvAttentionMetadata):
    """Short-conv metadata plus what the Inkling2 conv kernel reads, all in
    persistent buffers (the kernel is captured in graphs): per token its request
    and first row, and the group's block table, from which the kernel takes each
    request's state slot (column 0, or in align mode the block of its last
    token, which the runner has already seeded with the previous state)."""

    seq_idx: torch.Tensor | None = None
    query_start: torch.Tensor | None = None
    query_start_loc: torch.Tensor | None = None
    block_table: torch.Tensor | None = None
    align: bool = False
    state_block_size: int = 1


class Inkling2ConvMetadataBuilder(ShortConvAttentionMetadataBuilder):
    metadata_cls = Inkling2ConvMetadata
    needs_causal_conv1d_metadata = False

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        n_tok = vllm_config.scheduler_config.max_num_batched_tokens
        self.seq_idx_buf = torch.zeros(n_tok, dtype=torch.int32, device=device)
        self.query_start_buf = torch.zeros(n_tok, dtype=torch.int32, device=device)
        self.align = vllm_config.cache_config.mamba_cache_mode == "align"

    def build(self, common_prefix_len: int, common_attn_metadata: CommonAttentionMetadata,
              fast_build: bool = False, **kw):
        m = common_attn_metadata
        md = super().build(common_prefix_len, m, fast_build, **kw)
        n_act = int(m.query_start_loc_cpu[-1])
        n_pad = m.slot_mapping.shape[0]
        sconv_seq_metadata(m.query_start_loc, m.num_reqs, n_act, self.seq_idx_buf,
                           self.query_start_buf, n_pad)
        return replace(md, seq_idx=self.seq_idx_buf[:n_pad],
                       query_start=self.query_start_buf[:n_pad],
                       query_start_loc=m.query_start_loc, block_table=m.block_table_tensor,
                       align=self.align, state_block_size=self.kv_cache_spec.block_size)

    def build_for_cudagraph_capture(self, common_attn_metadata: CommonAttentionMetadata):
        """Decode-only capture, as the base, but for any verify length up to 1 + spec (a
        dynamic draft-length schedule captures shorter decode tiers); the conv kernel takes
        each request's length from query_start_loc."""
        m = common_attn_metadata
        assert m.max_query_len <= 1 + self.num_spec_tokens, m.max_query_len
        assert m.num_reqs <= self.decode_cudagraph_max_bs, m.num_reqs
        num_accepted = torch.diff(m.query_start_loc) if self.num_spec_tokens > 0 else None
        return self.build(0, m, num_accepted_tokens=num_accepted)

    def update_block_table(self, metadata, blk_table, slot_mapping):
        return replace(super().update_block_table(metadata, blk_table, slot_mapping),
                       block_table=blk_table)


def spec_tokens(vllm_config) -> int:
    sc = getattr(vllm_config, "speculative_config", None)
    return 0 if sc is None else int(sc.num_speculative_tokens or 0)


class Inkling2ConvBackend(ShortConvAttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "INKLING2_CONV_STATE"

    @staticmethod
    def get_builder_cls() -> type[Inkling2ConvMetadataBuilder]:
        return Inkling2ConvMetadataBuilder


class Inkling2ConvState(nn.Module, MambaBase):
    def __init__(self, prefix: str, layout: ConvLayout, kernel_size: int) -> None:
        super().__init__()
        self.prefix = prefix
        self.layout = layout
        self.kernel_size = kernel_size
        vllm_config = get_current_vllm_config()
        self.spec = spec_tokens(vllm_config)
        self.dtype = vllm_config.model_config.dtype
        assert self.dtype == torch.bfloat16, "Inkling2 conv state is bf16"
        self.kv_cache = (torch.tensor([]),)
        _register(prefix, self)

    def forward(self): ...

    @staticmethod
    def state_shape(layout: ConvLayout, kernel_size: int, spec: int = 0) -> tuple[int, int]:
        """(kernel_size - 1 + spec, H * W): the last inputs, oldest first (with speculative
        decoding, vLLM's rolling conv state of spec extra rows)."""
        return (kernel_size - 1 + spec, layout.num_heads * layout.width)

    def get_state_shape(self):
        return (self.state_shape(self.layout, self.kernel_size, self.spec),)

    def get_state_dtype(self) -> tuple[torch.dtype, ...]:
        return (self.dtype,)

    @property
    def mamba_type(self) -> MambaAttentionBackendEnum:
        return MambaAttentionBackendEnum.SHORT_CONV

    @property
    def is_kv_cache_tp_replicated(self) -> bool:
        """The attention / MLP stream states hold the full hidden size on every rank;
        the K/V state holds this rank's KV heads."""
        return self.layout.k_off < 0

    def get_attn_backend(self) -> type[AttentionBackend]:
        return Inkling2ConvBackend

    @property
    def state(self) -> torch.Tensor:
        """[num_slots, kernel_size - 1, H * W]."""
        return self.kv_cache[0]
