# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os

import torch.nn as nn

from vllm.config import VllmConfig, replace
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model
from vllm.model_executor.model_loader.utils import get_draft_load_config
from vllm.v1.worker.gpu.spec_decode.eagle.utils import (
    _should_share,
    get_target_lm_head,
    maybe_share_target_embed,
)
from vllm.v1.worker.gpu.spec_decode.utils import get_pp_safe_draft_load_config

logger = init_logger(__name__)


def _b70_draft_attention_backend(vllm_config: VllmConfig):
    """B70-V7-DFA (``B70_DRAFT_FA_INT8=1``): run the drafter's attention on
    the native INT8 FLASH_ATTN decode (B70-IP-D2) instead of the configured
    backend when the draft KV is int8_per_token_head and B70_PTH_FA_DECODE=1.
    FLASH_ATTN keeps TRITON_ATTN's page bytes for this dtype, so KV sizing,
    hybrid unify and offload see the same pages."""
    spec = vllm_config.speculative_config
    configured = spec.attention_backend
    if os.environ.get("B70_DRAFT_FA_INT8", "0") != "1":
        return configured
    from vllm.v1.attention.backends import b70_pth_fa
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    dtype = spec.kv_cache_dtype or vllm_config.cache_config.cache_dtype
    if dtype != "int8_per_token_head" or not b70_pth_fa.fa_decode_enabled():
        logger.warning_once(
            "B70-V7-DFA drafter FLASH_ATTN int8 off: draft KV dtype %s, "
            "B70_PTH_FA_DECODE active %s (needs int8_per_token_head and 1)",
            dtype,
            b70_pth_fa.fa_decode_enabled(),
        )
        return configured
    hf = spec.draft_model_config.hf_config
    head_dim = getattr(hf, "head_dim", None) or (
        hf.hidden_size // hf.num_attention_heads
    )
    if head_dim % 64 != 0:
        logger.warning_once(
            "B70-V7-DFA drafter FLASH_ATTN int8 off: head_dim %d is not a "
            "multiple of 64 (INT8 decode surfaces need 64-B alignment)",
            head_dim,
        )
        return configured
    logger.info(
        "B70-V7-DFA drafter attention: FLASH_ATTN native INT8 per-token-head "
        "decode, head_dim %d (was %s)",
        head_dim,
        getattr(configured, "name", configured),
    )
    return AttentionBackendEnum.FLASH_ATTN


def load_dflash_model(target_model: nn.Module, vllm_config: VllmConfig) -> nn.Module:
    from vllm.compilation.backends import set_model_tag
    from vllm.model_executor.models.qwen3_dflash import (
        dflash_has_any_non_causal,
    )

    speculative_config = vllm_config.speculative_config
    assert speculative_config is not None
    draft_model_config = speculative_config.draft_model_config
    # Select an attention backend that supports the drafter's attention: mixing
    # a non-causal layer onto a causal-only backend would fail.
    draft_vllm_config = replace(
        vllm_config,
        attention_config=replace(
            vllm_config.attention_config,
            use_non_causal=dflash_has_any_non_causal(draft_model_config.hf_config),
            backend=_b70_draft_attention_backend(vllm_config),
        ),
        cache_config=(
            replace(
                vllm_config.cache_config,
                cache_dtype=speculative_config.kv_cache_dtype,
            )
            if speculative_config.kv_cache_dtype is not None
            else vllm_config.cache_config
        ),
        load_config=get_pp_safe_draft_load_config(get_draft_load_config(vllm_config)),
    )
    with set_model_tag("dflash_head"):
        dflash_model = get_model(
            vllm_config=draft_vllm_config, model_config=draft_model_config
        )

    target_language_model = (
        target_model.get_language_model()
        if hasattr(target_model, "get_language_model")
        else target_model
    )
    # MuseGlimmerForCausalLM marks its inner MuseGlimmerModel as the language
    # model, so get_language_model() already returns the inner module and has
    # no .model of its own.
    target_inner = getattr(target_language_model, "model", target_language_model)
    draft_inner = dflash_model.model
    # B70-V7-KV8: needs the quant methods' processed weights, so after get_model.
    if hasattr(draft_inner, "build_int8_context_kv"):
        draft_inner.build_int8_context_kv()

    maybe_share_target_embed(dflash_model, draft_inner, target_inner)

    target_lm_head = get_target_lm_head(target_model, target_language_model)
    draft_lm_head = getattr(dflash_model, "lm_head", None)
    if (
        target_lm_head is not None
        and not getattr(dflash_model.config, "has_own_lm_head", False)
        and _should_share(
            dflash_model, "has_own_lm_head", draft_lm_head, target_lm_head
        )
    ):
        if draft_lm_head is not None:
            del dflash_model.lm_head
        dflash_model.lm_head = target_lm_head

    return dflash_model
