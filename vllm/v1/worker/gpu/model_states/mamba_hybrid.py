# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.config.compilation import CUDAGraphMode
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateCopyFuncsByType
from vllm.triton_utils import tl, triton
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.mamba2_attn import Mamba2AttentionMetadataBuilder
from vllm.v1.attention.backends.short_conv_attn import (
    PleShortConvAttentionMetadataBuilder,
    ShortConvAttentionMetadataBuilder,
)
from vllm.v1.attention.backends.utils import NULL_BLOCK_ID
from vllm.v1.core.kv_cache_utils import (
    MAX_FINISH_STOP_TOKEN_IDS,
    finish_stop_token_ids,
)
from vllm.v1.core.sched.output import NewRequestData
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.gpu.attn_utils import build_attn_metadata
from vllm.v1.worker.gpu.block_table import BlockTables
from vllm.v1.worker.gpu.buffer_utils import StagedWriteTensor, UvaBackedTensor
from vllm.v1.worker.gpu.input_batch import InputBatch
from vllm.v1.worker.gpu.mm.encoder_cache import EncoderCache
from vllm.v1.worker.gpu.model_states.default import DefaultModelState
from vllm.v1.worker.gpu.model_states.interface import ModelSpecificAttnMetadata
from vllm.v1.worker.gpu.model_states.recoverssm import RecoverSSMState
from vllm.v1.worker.mamba_utils import (
    MambaSpecDecodeGPUContext,
    get_mamba_group_ids,
    get_mamba_groups,
    preprocess_mamba_align_fused_kernel,
    validate_mamba_state_copy_funcs,
)
from vllm.v1.worker.utils import AttentionGroup


@dataclass
class MambaHybridAttnMetadata(ModelSpecificAttnMetadata):
    is_prefilling: torch.Tensor
    num_accepted_tokens: torch.Tensor | None = None
    num_decode_draft_tokens_cpu: torch.Tensor | None = None

    def get_extra_common_attn_kwargs(
        self, kv_cache_group_id: int, num_reqs: int
    ) -> dict[str, Any]:
        return {"is_prefilling": self.is_prefilling[:num_reqs]}

    def get_extra_attn_kwargs(
        self, attn_metadata_builder: Any, num_reqs: int
    ) -> dict[str, Any]:
        if not isinstance(
            attn_metadata_builder,
            (
                Mamba2AttentionMetadataBuilder,
                GDNAttentionMetadataBuilder,
                ShortConvAttentionMetadataBuilder,
                PleShortConvAttentionMetadataBuilder,
            ),
        ):
            return {}
        return {
            "num_accepted_tokens": None
            if self.num_accepted_tokens is None
            else self.num_accepted_tokens[:num_reqs],
            "num_decode_draft_tokens_cpu": None
            if self.num_decode_draft_tokens_cpu is None
            else self.num_decode_draft_tokens_cpu[:num_reqs],
        }


class MambaHybridModelState(DefaultModelState):
    """Model state for hybrid attention + Mamba / linear-attention models."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        encoder_cache: EncoderCache | None,
        device: torch.device,
    ) -> None:
        super().__init__(vllm_config, model, encoder_cache, device)
        self.cache_config = vllm_config.cache_config
        self.num_accepted_tokens_gpu = torch.ones(
            self.max_num_reqs, dtype=torch.int32, device=self.device
        )
        # Pre-copy "align" prefix-cache state (V2). The migration of each
        # request's mamba state across block boundaries runs as a fused GPU
        # kernel reusing the postprocess copy machinery, so the per-step src
        # columns and the running state_idx are kept GPU-resident.
        self._align_mode = self.cache_config.mamba_cache_mode == "align"
        self.recoverssm = (
            RecoverSSMState() if self.cache_config.use_kda_recoverssm else None
        )
        if self._align_mode:
            self._mamba_state_idx_gpu = torch.zeros(
                self.max_num_reqs, dtype=torch.int32, device=self.device
            )
            self._mamba_src_col_gpu = torch.full(
                (self.max_num_reqs,), -1, dtype=torch.int32, device=self.device
            )
            self._mamba_src_off_gpu = torch.zeros(
                self.max_num_reqs, dtype=torch.int32, device=self.device
            )
            self._mamba_ctx: MambaSpecDecodeGPUContext | None = None
            self._mamba_group_ids: list[int] = []
            self._mamba_spec: MambaSpec | None = None
            self._mamba_state_copy_funcs: MambaStateCopyFuncsByType | None = None
        # Stop tables for the finish checkpoint, see _detect_finish_kernel.
        self._finish_checkpoint = (
            self._align_mode and self.cache_config.enable_mamba_finish_checkpoint
        )
        if self._finish_checkpoint:
            self._finish_max_len = UvaBackedTensor(self.max_num_reqs, dtype=torch.int32)
            self._num_finish_stop_ids = UvaBackedTensor(
                self.max_num_reqs, dtype=torch.int32
            )
            self._finish_stop_ids = StagedWriteTensor(
                (self.max_num_reqs, MAX_FINISH_STOP_TOKEN_IDS),
                dtype=torch.int32,
                device=self.device,
            )
            self._finish_stopped_gpu = torch.zeros(
                self.max_num_reqs, dtype=torch.int32, device=self.device
            )
            self._finish_num_computed_gpu = torch.zeros(
                self.max_num_reqs, dtype=torch.int32, device=self.device
            )
            self._finish_num_accepted_gpu = torch.zeros(
                self.max_num_reqs, dtype=torch.int32, device=self.device
            )
            self._finish_idx_mapping_gpu = torch.zeros(
                self.max_num_reqs, dtype=torch.int32, device=self.device
            )

    def add_request(self, req_index: int, new_req_data: NewRequestData) -> None:
        super().add_request(req_index, new_req_data)
        # Must reset the speculative acceptance count in this idx which could be stale.
        self.num_accepted_tokens_gpu[req_index].fill_(1)
        if self._align_mode:
            # Seed the running state block from the resumed/prefilled position.
            # The column is in mamba blocks. cache_config.block_size is the
            # min over prefix-cacheable groups, so a drafter group with its own
            # smaller block (e.g. DFlash) pulls it below mamba_block_size.
            mamba_block_size = self.cache_config.mamba_block_size
            assert mamba_block_size is not None
            self._mamba_state_idx_gpu[req_index].fill_(
                (new_req_data.num_computed_tokens - 1) // mamba_block_size
            )
        if self._finish_checkpoint:
            self._finish_stopped_gpu[req_index].fill_(0)
            sampling_params = new_req_data.sampling_params
            if sampling_params is None:
                return
            stop_token_ids = finish_stop_token_ids(sampling_params)
            assert sampling_params.max_tokens is not None
            max_len = min(
                new_req_data.prompt_len + sampling_params.max_tokens,
                self.max_model_len,
            )
            if stop_token_ids is None:
                stop_token_ids = []
                max_len = torch.iinfo(torch.int32).max
            self._finish_max_len.np[req_index] = max_len
            self._num_finish_stop_ids.np[req_index] = len(stop_token_ids)
            self._finish_stop_ids.stage_write(req_index, 0, stop_token_ids)

    def apply_staged_writes(self) -> None:
        super().apply_staged_writes()
        if self._finish_checkpoint:
            self._finish_max_len.copy_to_uva()
            self._num_finish_stop_ids.copy_to_uva()
            self._finish_stop_ids.apply_write()

    def _get_mamba_group_info(
        self, kv_cache_config: KVCacheConfig
    ) -> tuple[list[int], MambaSpec]:
        if self._mamba_spec is None:
            mamba_groups = get_mamba_groups(kv_cache_config)
            mamba_spec = next(iter(mamba_groups))
            assert all(
                spec.block_size == mamba_spec.block_size
                and spec.num_speculative_blocks == mamba_spec.num_speculative_blocks
                and spec.mamba_cache_mode == mamba_spec.mamba_cache_mode
                for spec in mamba_groups
            ), "all mamba groups must share cache scheduling parameters"
            self._mamba_group_ids = get_mamba_group_ids(mamba_groups)
            self._mamba_spec = mamba_spec
        return self._mamba_group_ids, self._mamba_spec

    def initialize_kv_cache(
        self, kv_cache_config: KVCacheConfig, block_tables: BlockTables
    ) -> None:
        if not self._align_mode:
            return
        mamba_group_ids, _ = self._get_mamba_group_info(kv_cache_config)
        if self._mamba_state_copy_funcs is None:
            mamba_groups = get_mamba_groups(kv_cache_config)
            mamba_types = {spec.mamba_type for spec in mamba_groups}
            copy_funcs = self.model.get_mamba_state_copy_funcs(mamba_types)
            validate_mamba_state_copy_funcs(mamba_groups, copy_funcs)
            self._mamba_state_copy_funcs = copy_funcs
        # Both SD and DS conv layouts support a >0 spec-decode shift: the
        # fused pre-copy kernel (``_copy_mamba_state_block``) applies the
        # ``token_bias = num_accepted - 1`` window shift per conv layout
        # (SD: contiguous slice; DS: per-dim-row strided slice), matching
        # the V1 ``get_conv_copy_spec`` semantics.
        self._mamba_ctx = MambaSpecDecodeGPUContext.create(
            max_num_reqs=self.max_num_reqs,
            kv_cache_config=kv_cache_config,
            copy_funcs=self._mamba_state_copy_funcs,
            device=self.device,
            make_buffer=lambda n, dtype: CpuGpuBuffer(
                n, dtype=dtype, device=self.device
            ),
        )
        # Under PP a step's postprocess runs after later steps re-gathered the
        # batch-ordered input tables, so state copies read the per-request-slot
        # tables. Aligned state indices are for the current batch only.
        self._mamba_ctx.initialize_from_forward_context(
            kv_cache_config,
            self.vllm_config.compilation_config.static_forward_context,
            self._mamba_state_copy_funcs,
            [block_tables.block_tables[gid].gpu for gid in mamba_group_ids],
            [block_tables.input_block_tables[gid] for gid in mamba_group_ids],
        )

    def preprocess_state(
        self,
        input_batch: InputBatch,
        block_tables: tuple[torch.Tensor, ...],
        kv_cache_config: KVCacheConfig,
        num_computed_tokens: torch.Tensor,
    ) -> None:
        """Migrate each request's mamba state across block boundaries before the
        forward (V1 align semantics, done on GPU). Runs on real batches only
        (dummy DP/profiling runs skip preprocess_state), and before
        ``prepare_attn`` gathers ``num_accepted_tokens``, so the boundary reset
        is visible to the forward kernels.
        """
        if not self._align_mode:
            return
        num_reqs = input_batch.num_reqs
        if num_reqs == 0:
            return
        mamba_group_ids, mamba_spec = self._get_mamba_group_info(kv_cache_config)
        ctx = self._mamba_ctx
        assert ctx is not None

        # The state-advance + pre-copy kernels run every step; they fast-exit per
        # request when src_col < 0 or src_col == dst_col, so no copy happens on
        # steps that don't cross a block boundary. (Skipping the launch entirely
        # would need a V1-style async-D2H of the actual num_computed, since
        # num_computed_tokens_np is an optimistic mirror under async scheduling;
        # the launch cost is ~0.3% of TPOT, so the GPU fast-exit suffices.)
        block = 256
        grid = (triton.cdiv(num_reqs, block),)
        preprocess_mamba_align_fused_kernel[grid](
            input_batch.idx_mapping,
            self._mamba_state_idx_gpu,
            num_computed_tokens,
            input_batch.query_start_loc,
            self.num_accepted_tokens_gpu,
            self._mamba_src_col_gpu,
            self._mamba_src_off_gpu,
            num_reqs,
            BLOCK_SIZE=block,
            MAMBA_BLOCK_SIZE=mamba_spec.block_size,
        )
        if self._finish_checkpoint:
            # A request that stopped in an earlier step can still be in this
            # batch under async scheduling. Point its Mamba rows at the null
            # block, which the kernels skip, and skip its pre-copy.
            stopped = self._finish_stopped_gpu[input_batch.idx_mapping] != 0
            for group_id in mamba_group_ids:
                block_tables[group_id][:num_reqs].masked_fill_(
                    stopped[:, None], NULL_BLOCK_ID
                )
            self._mamba_src_col_gpu.masked_fill_(self._finish_stopped_gpu != 0, -1)
        ctx.run_fused_precopy(
            num_reqs,
            self._mamba_state_idx_gpu,
            self._mamba_src_col_gpu,
            self._mamba_src_off_gpu,
            input_batch.idx_mapping,
        )

    def prepare_attn(
        self,
        input_batch: InputBatch,
        cudagraph_mode: CUDAGraphMode,
        block_tables: tuple[torch.Tensor, ...],
        slot_mappings: torch.Tensor,
        attn_groups: list[list[AttentionGroup]],
        kv_cache_config: KVCacheConfig,
        for_capture: bool = False,
        ubatch_idx: int = 0,
        model_specific_attn_metadata: ModelSpecificAttnMetadata | None = None,
    ) -> dict[str, Any]:
        assert ubatch_idx == 0, "DBO is not supported"
        assert model_specific_attn_metadata is None
        if cudagraph_mode == CUDAGraphMode.FULL:
            num_reqs = input_batch.num_reqs_after_padding
            num_tokens = input_batch.num_tokens_after_padding
        else:
            num_reqs = input_batch.num_reqs
            num_tokens = input_batch.num_tokens
        query_start_loc_cpu = torch.from_numpy(input_batch.query_start_loc_np)
        # Prefer the promised bound: a capture dummy's measured max is its even
        # split, not the length the graph must replay.
        max_query_len = input_batch.max_query_len
        if max_query_len is None:
            max_query_len = input_batch.num_scheduled_tokens.max().item()
        seq_lens_cpu_upper_bound = input_batch.seq_lens_cpu_upper_bound
        if for_capture:
            # Capture with worst-case max_seq_len so the graph is valid at any replay.
            max_seq_len = self.max_model_len
        else:
            max_seq_len = seq_lens_cpu_upper_bound[:num_reqs].max().item()

        is_prefilling_np = input_batch.is_prefilling_np
        if input_batch.prefill_runs_as_decode_np is not None:
            # A prompt tail the scheduler padded with placeholder drafts must run
            # as a spec-decode row: the prefill kernels can't roll them back.
            is_prefilling_np = is_prefilling_np & ~input_batch.prefill_runs_as_decode_np
        is_prefilling = torch.zeros(num_reqs, dtype=torch.bool, device="cpu")
        is_prefilling[: input_batch.num_reqs] = torch.from_numpy(is_prefilling_np)
        # During CUDAGraph capture, num_decode_draft_tokens_cpu and num_accepted_tokens
        # are created by attn_metadata_builder.build_for_cudagraph_capture, so we only
        # compute them during actual (non-capture) forward execution.
        num_accepted_tokens = None
        num_decode_draft_tokens_cpu = None
        if not for_capture and self.vllm_config.num_speculative_tokens > 0:
            num_accepted_tokens = self.num_accepted_tokens_gpu.new_ones(num_reqs)
            num_accepted_tokens[: input_batch.num_reqs] = self.num_accepted_tokens_gpu[
                input_batch.idx_mapping
            ]

            # GDN uses >= 0 to select spec-decode rows, so non-decode rows
            # need the -1 sentinel. Decode rows without drafts stay spec rows:
            # only that path applies the previous step's accepted offset.
            num_decode_draft_tokens_np = np.full(num_reqs, -1, dtype=np.int32)
            num_draft_tokens_per_req = input_batch.num_draft_tokens_per_req
            if num_draft_tokens_per_req is None:
                num_draft_tokens_per_req = np.zeros(input_batch.num_reqs, np.int32)
            # Test request state, not num_scheduled_tokens == draft_count+1:
            # adaptive rewrites num_scheduled_tokens to an even split, so that
            # equality rarely holds and would demote every verify row to decode.
            is_decode = (
                ~is_prefilling_np
                & (input_batch.num_scheduled_tokens > 0)
                & (
                    input_batch.num_scheduled_tokens
                    <= self.vllm_config.num_speculative_tokens + 1
                )
            )
            num_decode_draft_tokens_np[: input_batch.num_reqs] = np.where(
                is_decode, num_draft_tokens_per_req, -1
            )
            num_decode_draft_tokens_cpu = torch.from_numpy(num_decode_draft_tokens_np)

        if self._align_mode:
            mamba_group_ids, _ = self._get_mamba_group_info(kv_cache_config)
            aligned_index_builders = []
            for group_idx, group_id in enumerate(mamba_group_ids):
                for group in attn_groups[group_id]:
                    builder = group.get_metadata_builder(0)
                    if hasattr(builder, "mamba_aligned_state_indices"):
                        aligned_index_builders.append((group_idx, builder))
            if aligned_index_builders:
                assert self._mamba_ctx is not None
                all_group_indices = self._mamba_ctx.compute_aligned_state_indices(
                    input_batch.seq_lens, num_reqs
                )
                for group_idx, builder in aligned_index_builders:
                    builder.mamba_aligned_state_indices = all_group_indices[group_idx]

        mamba_attn_metadata = MambaHybridAttnMetadata(
            is_prefilling=is_prefilling,
            num_accepted_tokens=num_accepted_tokens,
            num_decode_draft_tokens_cpu=num_decode_draft_tokens_cpu,
        )
        attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            query_start_loc_gpu=input_batch.query_start_loc,
            query_start_loc_cpu=query_start_loc_cpu,
            max_query_len=max_query_len,
            seq_lens=input_batch.seq_lens,
            max_seq_len=max_seq_len,
            block_tables=block_tables,
            slot_mappings=slot_mappings,
            kv_cache_config=kv_cache_config,
            seq_lens_cpu_upper_bound=seq_lens_cpu_upper_bound,
            dcp_local_seq_lens=input_batch.dcp_local_seq_lens,
            positions=input_batch.positions,
            model_specific_attn_metadata=mamba_attn_metadata,
            for_cudagraph_capture=for_capture,
            rswa_prefix_lens=input_batch.prompt_lens,
        )
        if self.recoverssm is not None:
            self.recoverssm.record_step(
                attn_metadata, attn_groups, for_capture=for_capture
            )
        return attn_metadata

    def postprocess_state(
        self,
        idx_mapping: torch.Tensor,
        num_sampled: torch.Tensor | int,
        num_computed_tokens: torch.Tensor | None = None,
        sampled_tokens: torch.Tensor | None = None,
    ) -> None:
        # Chunked prefill does not sample a token, so num_sampled can be 0.
        # Mamba treats num_accepted_tokens=1 as the neutral non-spec value.
        num_reqs = idx_mapping.shape[0]
        if num_reqs:
            if not isinstance(num_sampled, int):
                # idx_mapping may contain -1 sentinels (filtered rows) under PP; the
                # kernel skips them rather than scattering with a host-side gather.
                _scatter_num_accepted_kernel[(num_reqs,)](
                    idx_mapping,
                    num_sampled,
                    self.num_accepted_tokens_gpu,
                )
            else:
                # Fill with single value.
                _fill_num_accepted_kernel[(num_reqs,)](
                    idx_mapping,
                    self.num_accepted_tokens_gpu,
                    max(num_sampled, 1),
                )

        if self.recoverssm is not None:
            self.recoverssm.commit_step(
                num_sampled,
                idx_mapping,
                state_indices=(self._mamba_state_idx_gpu if self._align_mode else None),
                num_accepted_tokens=self.num_accepted_tokens_gpu,
            )

        if not num_reqs:
            return

        # Align: save the running state to the block-aligned position when
        # spec-decode acceptance leaves the sequence non-block-aligned (mirrors
        # the V1 align postprocess). num_computed_tokens already holds the
        # post-step advanced count.
        if (
            self._align_mode
            and num_computed_tokens is not None
            and self._mamba_ctx is not None
        ):
            if self._finish_checkpoint:
                assert sampled_tokens is not None
                assert not isinstance(num_sampled, int)
                postprocess_idx_mapping = self._finish_idx_mapping_gpu[:num_reqs]
                _detect_finish_kernel[(num_reqs,)](
                    idx_mapping,
                    sampled_tokens,
                    sampled_tokens.stride(0),
                    num_sampled,
                    num_computed_tokens,
                    self._finish_stop_ids.gpu,
                    self._finish_stop_ids.gpu.stride(0),
                    self._num_finish_stop_ids.gpu,
                    self._finish_max_len.gpu,
                    self._finish_stopped_gpu,
                    self.num_accepted_tokens_gpu,
                    self._finish_num_computed_gpu,
                    self._finish_num_accepted_gpu,
                    postprocess_idx_mapping,
                    BLOCK_SIZE=triton.next_power_of_2(sampled_tokens.shape[1]),
                )
                idx_mapping = postprocess_idx_mapping
                num_computed_tokens = self._finish_num_computed_gpu
            self._mamba_ctx.run_fused_postprocess_align(
                num_reqs,
                self.num_accepted_tokens_gpu,
                self._mamba_state_idx_gpu,
                num_computed_tokens,
                idx_mapping,
            )
            if self._finish_checkpoint:
                self._mamba_ctx.run_fused_postprocess_align(
                    num_reqs,
                    self._finish_num_accepted_gpu,
                    self._mamba_state_idx_gpu,
                    num_computed_tokens,
                    idx_mapping,
                    finalize=True,
                )


@triton.jit
def _detect_finish_kernel(
    idx_mapping_ptr,  # [num_reqs] batch_idx -> req_state_idx (-1 to skip)
    sampled_tokens_ptr,  # [num_reqs, num_speculative_steps + 1]
    sampled_tokens_stride,
    num_sampled_ptr,  # [num_reqs]
    num_computed_ptr,  # [max_num_reqs] post-step
    stop_ids_ptr,  # [max_num_reqs, MAX_FINISH_STOP_TOKEN_IDS]
    stop_ids_stride,
    num_stop_ids_ptr,  # [max_num_reqs]
    max_len_ptr,  # [max_num_reqs]
    stopped_ptr,  # [max_num_reqs]
    num_accepted_ptr,  # [max_num_reqs]
    finish_num_computed_ptr,  # [max_num_reqs] out
    finish_num_accepted_ptr,  # [max_num_reqs] out, 0 if not finishing
    postprocess_idx_mapping_ptr,  # [num_reqs] out
    BLOCK_SIZE: tl.constexpr,
):
    """Find the first sampled token that ends the request, as check_stop does,
    and truncate the step there: the stop token is never fed through the
    model, so the state to keep is the one before it. Requests that stopped in
    an earlier step are masked out of the postprocess.
    """
    row = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + row)
    stopped = tl.load(stopped_ptr + req_state_idx, mask=req_state_idx >= 0, other=1)
    if stopped != 0:
        tl.store(postprocess_idx_mapping_ptr + row, -1)
        return
    tl.store(postprocess_idx_mapping_ptr + row, req_state_idx)

    num_computed = tl.load(num_computed_ptr + req_state_idx)
    num_sampled = tl.load(num_sampled_ptr + row)
    # Sequence length before the first sampled token.
    seq_len = num_computed - num_sampled + 1
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_sampled
    tokens = tl.load(
        sampled_tokens_ptr + row * sampled_tokens_stride + offsets, mask=mask
    )
    max_len = tl.load(max_len_ptr + req_state_idx)
    is_stop = seq_len + offsets + 1 >= max_len
    num_stop_ids = tl.load(num_stop_ids_ptr + req_state_idx)
    for i in range(num_stop_ids):
        stop_id = tl.load(stop_ids_ptr + req_state_idx * stop_ids_stride + i)
        is_stop = is_stop | (tokens == stop_id)
    stop_idx = tl.min(tl.where(is_stop & mask, offsets, BLOCK_SIZE))
    finishes = stop_idx < BLOCK_SIZE
    if finishes:
        tl.store(stopped_ptr + req_state_idx, 1)
        tl.store(num_accepted_ptr + req_state_idx, stop_idx + 1)
        num_computed = seq_len + stop_idx
    tl.store(finish_num_computed_ptr + req_state_idx, num_computed)
    tl.store(
        finish_num_accepted_ptr + req_state_idx, tl.where(finishes, stop_idx + 1, 0)
    )


@triton.jit
def _scatter_num_accepted_kernel(
    idx_mapping_ptr,  # [num_reqs] batch_idx -> req_state_idx (-1 to skip)
    num_sampled_ptr,  # [num_reqs]
    num_accepted_ptr,  # [max_num_reqs]
):
    row = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + row)
    if req_state_idx < 0:
        return
    num_sampled = tl.load(num_sampled_ptr + row)
    tl.store(num_accepted_ptr + req_state_idx, tl.maximum(num_sampled, 1))


@triton.jit
def _fill_num_accepted_kernel(
    idx_mapping_ptr,  # [num_reqs] batch_idx -> req_state_idx (-1 to skip)
    num_accepted_ptr,  # [max_num_reqs]
    num_sampled,
):
    row = tl.program_id(0)
    req_state_idx = tl.load(idx_mapping_ptr + row)
    if req_state_idx < 0:
        return
    tl.store(num_accepted_ptr + req_state_idx, num_sampled)
