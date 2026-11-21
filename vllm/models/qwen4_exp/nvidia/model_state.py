# Modified by Wulf for B70 Flash-Next Disk; upstream terms are retained.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Model-runner state for Qwen4Exp PLE inputs."""

from typing import Any

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.mamba_hybrid import MambaHybridModelState
from vllm.v1.worker.gpu.states import RequestState


import os

_B70_HOST_IDS = os.environ.get("B70_PLE_HOST_IDS", "0") == "1"


class Qwen4ExpModelState(MambaHybridModelState):
    """Add rollback-safe PLE n-gram context to the model inputs."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        config = self.model_config.hf_text_config
        from .ngram_embedding import Qwen4ExpNGramEmbedding

        self._disk_ple_modules = tuple(
            module
            for module in model.modules()
            if isinstance(module, Qwen4ExpNGramEmbedding)
            and getattr(module.ngram_embedding, "host_prepared", False)
        )
        if self._disk_ple_modules and len(self._disk_ple_modules) != 1:
            raise RuntimeError("Disk BF16 PLE requires exactly one target PLE module")
        self.uses_ngram_embedding = bool(config.ple_layer_ids)
        if not self.uses_ngram_embedding:
            self.ngram_context_len = 0
            self.ngram_eos_token_id = 0
            return

        if vllm_config.parallel_config.pipeline_parallel_size > 1:
            raise RuntimeError(
                "N-gram PLE embedding currently requires "
                "pipeline_parallel_size=1 because non-first pipeline ranks do "
                "not receive the raw input_ids required by PLE. Please run "
                "with PP=1."
            )

        self.ngram_context_len = int(config.ngram_size) - 1
        if self.ngram_context_len <= 0:
            raise ValueError("N-gram embedding requires context length >= 1.")
        self.ngram_eos_token_id = int(config.eos_token_id)
        # PLE runs inside captured regions, so these buffers keep a fixed shape
        # and address as the active request count changes between replays.
        self.ngram_context = torch.full(
            (self.max_num_reqs, self.ngram_context_len),
            self.ngram_eos_token_id,
            dtype=torch.int32,
            device=self.device,
        )
        self.ngram_context_offsets = torch.arange(
            -self.ngram_context_len,
            0,
            dtype=torch.int64,
            device=self.device,
        )
        self.ple_query_start_loc = torch.zeros(
            self.max_num_reqs + 1,
            dtype=torch.int32,
            device=self.device,
        )

    def _prepare_ngram_context(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> torch.Tensor:
        num_reqs = input_batch.num_reqs
        context = self.ngram_context
        context.fill_(self.ngram_eos_token_id)
        if num_reqs == 0:
            return context

        request_indices = input_batch.idx_mapping[:num_reqs].long()
        context_end = req_states.num_computed_tokens.gpu[request_indices].long()
        token_indices = context_end.unsqueeze(1) + self.ngram_context_offsets
        valid_tokens = token_indices >= 0
        token_indices.clamp_min_(0)
        context_tokens = req_states.all_token_ids.gpu[
            request_indices.unsqueeze(1), token_indices
        ]
        context[:num_reqs].copy_(
            torch.where(
                valid_tokens,
                context_tokens,
                context_tokens.new_full((), self.ngram_eos_token_id),
            )
        )
        return context

    def prepare_inputs(
        self,
        input_batch: InputBatch,
        req_states: RequestState,
    ) -> dict[str, Any]:
        model_inputs = super().prepare_inputs(input_batch, req_states)
        # B70 0041 (0.32 port): hot/cold route counts, read on the host every B70_MOE_ROUTE_COUNT_S.
        # Returns at once when no layer counts; its .cpu() syncs once per period (0.30 called it
        # right after the NVMe PLE host sync, which 0.32 does not have in that form).
        from vllm.model_executor.layers.fused_moe import b70_hotcold

        b70_hotcold.maybe_dump_routes()
        # B70 0046: diagnostic; one .cpu() per step while B70_MOE_ROUTE_TRACE is set.
        b70_hotcold.maybe_trace_routes()
        # B70 0043: every B70_MOE_HOTCOLD_SWAP_S; its first .cpu() waits for the previous step.
        b70_hotcold.maybe_swap()
        if not self.uses_ngram_embedding:
            return model_inputs

        num_reqs_padded = input_batch.num_reqs_after_padding
        query_start_loc = self.ple_query_start_loc
        query_start_loc[: num_reqs_padded + 1].copy_(input_batch.query_start_loc)
        # Represent unused capacity as trailing zero-length requests.
        query_start_loc[num_reqs_padded + 1 :].copy_(input_batch.query_start_loc[-1])
        model_inputs.update(
            query_start_loc=query_start_loc,
            ngram_context=self._prepare_ngram_context(input_batch, req_states),
        )
        for module in self._disk_ple_modules:
            num_tokens_padded = input_batch.num_tokens_after_padding
            module.ngram_embedding._check_capacity(num_tokens_padded)
            if input_batch.input_ids.shape[0] != num_tokens_padded:
                raise ValueError("Disk BF16 PLE input length differs from dispatch")
            if num_tokens_padded == 0:
                continue
            emb = module.ngram_embedding
            if getattr(emb, "_devgather", None) is not None and emb._pleg_checked:
                # B70-V7-PLEG: the forward hashes and gathers on the device.
                emb.prepare_device(num_tokens_padded)
                continue
            if _B70_HOST_IDS and hasattr(module, "compute_ngram_ids_host"):
                # B70-V32-03: one small D2H of the tokens + context, ids hashed on the host: the
                # step's only host sync no longer waits behind ~50 eager hash kernels.
                from vllm.v1.worker.gpu import b70_phase as _b70p

                _b70p.mark("ple-d2h")
                num_reqs = input_batch.num_reqs
                num_tokens = input_batch.num_tokens
                ctx = model_inputs["ngram_context"][:num_reqs]
                flat = torch.cat([input_batch.input_ids[:num_tokens].long(), ctx.reshape(-1).long()])
                flat = flat.to("cpu", non_blocking=False).numpy()
                _b70p.mark("ple-hash")
                ids = module.compute_ngram_ids_host(
                    flat[:num_tokens],
                    input_batch.query_start_loc_np[: num_reqs + 1],
                    flat[num_tokens:].reshape(num_reqs, -1),
                )
                _b70p.mark("ple-prep")
                module.ngram_embedding.prepare(None, num_tokens, num_tokens_padded, host_ids=ids)
                if getattr(emb, "_devgather", None) is not None and num_tokens > 0:
                    # B70-V7-PLEG: first real step, the host path is the reference.
                    emb._wait_for_stage()
                    emb.check_device_gather(
                        module.device_ngram_ids(
                            input_batch.input_ids, query_start_loc, model_inputs["ngram_context"]),
                        query_start_loc, num_tokens_padded)
                continue
            # Existing GPU IDs use the authoritative committed context above.
            # This producer runs before FULL replay as well as ordinary forward.
            ids = module.compute_ngram_ids(
                input_batch.input_ids,
                query_start_loc,
                model_inputs["ngram_context"],
            )
            module.ngram_embedding.prepare(
                ids,
                input_batch.num_tokens,
                num_tokens_padded,
            )
            if getattr(emb, "_devgather", None) is not None and input_batch.num_tokens > 0:
                emb._wait_for_stage()
                emb.check_device_gather(
                    module.device_ngram_ids(
                        input_batch.input_ids, query_start_loc, model_inputs["ngram_context"]),
                    query_start_loc, num_tokens_padded)
        return model_inputs

    def prepare_dummy_inputs(
        self,
        num_reqs: int,
        num_tokens: int,
    ) -> dict[str, Any]:
        model_inputs = super().prepare_dummy_inputs(num_reqs, num_tokens)
        if not self.uses_ngram_embedding:
            return model_inputs

        query_start_loc = self.ple_query_start_loc
        query_start_loc[0] = 0
        tokens_per_req, num_extra_tokens = divmod(num_tokens, num_reqs)
        query_lens = torch.full(
            (num_reqs,),
            tokens_per_req,
            dtype=query_start_loc.dtype,
            device=query_start_loc.device,
        )
        if num_extra_tokens > 0:
            query_lens[-num_extra_tokens:] += 1
        torch.cumsum(query_lens, dim=0, out=query_start_loc[1 : num_reqs + 1])
        query_start_loc[num_reqs + 1 :].fill_(num_tokens)

        ngram_context = self.ngram_context
        ngram_context.fill_(self.ngram_eos_token_id)
        model_inputs.update(
            query_start_loc=query_start_loc,
            ngram_context=ngram_context,
        )
        for module in self._disk_ple_modules:
            module.ngram_embedding.prepare(None, 0, num_tokens)
        return model_inputs


__all__ = ["Qwen4ExpModelState"]
