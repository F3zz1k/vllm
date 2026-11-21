# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-visit cache owners of Inkling2 (XPU port).

Every (layer, loop pass) visit owns two caches, both registered as attention
layers so vLLM's KV cache manager allocates, prefix-caches and offloads them:

* `Inkling2KVCache`: the visit's K/V (`SlidingWindowSpec` for the local
  layers, `FullAttentionSpec` for the global ones); metadata from
  `FlashAttentionBackend`'s builder.
* `Inkling2ConvState`: the short-conv inputs of the visit's four conv streams
  (K, V, attention output, MLP output) as a sliding-window cache of
  `sconv_kernel_size` tokens, one block of `kernel_size` tokens per page, the
  streams packed per KV head (`ref.ConvLayout`); metadata from the Inkling
  sconv backend (per-token request index and conv slots).
"""

from __future__ import annotations

import torch
from torch import nn

from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.models.inkling.amd.sconv_swa_attn import InklingSconvBackend
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.flash_attn import FlashAttentionBackend
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    SlidingWindowSpec,
)

from .ops.ref import ConvLayout


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
    def __init__(self, prefix: str, num_kv_heads: int, head_dim: int,
                 sliding_window: int | None) -> None:
        super().__init__()
        self.prefix = prefix
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.sliding_window = sliding_window
        vllm_config = get_current_vllm_config()
        self.kv_cache_dtype = vllm_config.cache_config.cache_dtype
        self.dtype = kv_cache_dtype_str_to_dtype(self.kv_cache_dtype, vllm_config.model_config)
        self.kv_cache = torch.tensor([])
        _register(prefix, self)

    def forward(self): ...

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
        return self.kv_cache


class Inkling2ConvState(nn.Module, AttentionLayerBase):
    def __init__(self, prefix: str, layout: ConvLayout, kernel_size: int) -> None:
        super().__init__()
        self.prefix = prefix
        self.layout = layout
        self.kernel_size = kernel_size
        vllm_config = get_current_vllm_config()
        self.dtype = vllm_config.model_config.dtype
        assert self.dtype == torch.bfloat16, "Inkling2 conv state is bf16"
        self.kv_cache = torch.tensor([])
        _register(prefix, self)

    def forward(self): ...

    def get_attn_backend(self) -> type[AttentionBackend]:
        return InklingSconvBackend

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        return SlidingWindowSpec(block_size=self.kernel_size,
                                 num_kv_heads=self.layout.num_heads,
                                 head_size=self.layout.width, head_size_v=0,
                                 dtype=self.dtype, sliding_window=self.kernel_size)
