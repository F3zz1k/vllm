# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
import time
from collections import Counter, OrderedDict
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field, replace
from itertools import chain, islice
from types import SimpleNamespace
from typing import Any, NamedTuple

from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVCacheEvent
from vllm.distributed.kv_transfer.kv_connector.utils import yield_req_data
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    OffloadingWorkerMetadata,
    ReqId,
    TransferJob,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading import (
    b70_offload as _b70,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.events import (
    OffloadingEventGroupSpec,
    OffloadingEventsTracker,
    get_offloading_event_group_spec,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
    _ConnectorMetricName,
    _TransferMetricName,
)
from vllm.logger import init_logger
from vllm.utils.math_utils import cdiv, round_down
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import BlockHash
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
)
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    Locality,
    LookupResult,
    Medium,
    OffloadingManager,
    OffloadingSpec,
    OffloadKey,
    OffloadPolicy,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
    TierFilter,
    TierMatcher,
    get_offload_group_idx,
    make_offload_key,
)
from vllm.v1.kv_offload.flush_on_stop import flush_on_stop_sec
from vllm.v1.metrics.cache_hit_source import CacheHitSource
from vllm.v1.outputs import KVConnectorOutput
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)

# Exclusive RAM tier (2026-10-23). Write-back band: each step, the hashed GPU
# blocks near the head of the free queue (the next to be evicted) that the pool
# does not hold yet are held, copied to the pool and released back to the queue
# head, so the pool receives what the GPU is about to drop. While the pool has
# fewer than WRITE_THROUGH_MIN_FREE of its chunks admissible (free, dead, or
# another class's idle runs), compute-time stores stop: new content reaches the
# pool only through the band, so the pool converges on holding what the GPU
# does not.
WRITE_BACK = os.environ.get("VLLM_KV_WRITE_BACK", "0") == "1"
# The band's lead (free-queue positions scanned) is WRITE_BACK_SCAN plus the
# blocks the GPU reuses in WRITE_BACK_LEAD_FACTOR measured copy latencies,
# capped at WRITE_BACK_SCAN_MAX.
WRITE_BACK_SCAN = int(os.environ.get("VLLM_KV_WRITE_BACK_SCAN", "256"))
WRITE_BACK_SCAN_MAX = int(os.environ.get("VLLM_KV_WRITE_BACK_SCAN_MAX", "4096"))
WRITE_BACK_LEAD_FACTOR = float(os.environ.get("VLLM_KV_WRITE_BACK_LEAD_FACTOR", "2.0"))
# Blocks held by copies in flight, also capped at WRITE_BACK_FREE_SHARE of the
# free blocks, so the band never starves allocation.
WRITE_BACK_MAX_HELD = int(os.environ.get("VLLM_KV_WRITE_BACK_MAX_HELD", "1024"))
WRITE_BACK_FREE_SHARE = float(os.environ.get("VLLM_KV_WRITE_BACK_FREE_SHARE", "0.5"))
# A run the pool refused is not offered again for this long.
WRITE_BACK_REFUSED_BACKOFF_S = float(
    os.environ.get("VLLM_KV_WRITE_BACK_REFUSED_BACKOFF_S", "5.0")
)
# Bounded-wait variant (off by default): up to this many run members the GPU
# reuses in a step before the band copied them are copied out, and the step's
# forward waits for that copy instead of losing them.
WRITE_BACK_RESCUE = int(os.environ.get("VLLM_KV_WRITE_BACK_RESCUE", "0"))
# With recurrent state, end a prefill chunk at the last offload-chunk boundary
# inside the prompt (when no junction is set), so its state is handed off and a
# restore resumes there rather than up to a chunk earlier.
WRITE_BACK_PROMPT_BOUNDARY = (
    os.environ.get("VLLM_KV_WRITE_BACK_PROMPT_BOUNDARY", "1") == "1"
)
# A finished run's pins are dropped if it is not recorded within this long.
WRITE_BACK_PIN_S = 5.0
# Bounds of the band's bookkeeping maps.
_STATE_SETS_MAX = 1 << 16
_POOL_HAS_MAX = 1 << 18
# Conversations ("runs") remembered for whole-run write-back.
WRITE_BACK_MAX_RUNS = 4096
WRITE_THROUGH_MIN_FREE = float(os.environ.get("VLLM_KV_WRITE_THROUGH_MIN_FREE", "0.02"))
# With the band on, compute-time stores are off (2026-11-03, owner's decision:
# the RAM tier extends VRAM, it does not mirror it). Only the aligned boundary
# hand-offs stay at compute time. VLLM_KV_WRITE_BACK_MIRROR=1 restores the
# gate: mirror while the pool has WRITE_THROUGH_MIN_FREE admissible.
WRITE_BACK_MIRROR = os.environ.get("VLLM_KV_WRITE_BACK_MIRROR", "0") == "1"
# Aligned boundary hand-offs (a state, plus the attention block there, stored
# at compute time). With the band they duplicate what the GPU still holds,
# and the band can take the state from the GPU when the boundary is cached
# there; VLLM_KV_WRITE_BACK_HANDOFF=0 skips them (the junctions stay).
WRITE_BACK_HANDOFF = os.environ.get("VLLM_KV_WRITE_BACK_HANDOFF", "1") == "1"
_WRITE_BACK_REQ = "__kv_write_back__"
# Flush on stop (see vllm/v1/kv_offload/flush_on_stop.py): on a graceful stop
# every conversation still cached on the GPU is written to the pool as a whole
# restorable run, newest first, within this many seconds (0: off).
FLUSH_ON_STOP_SEC = flush_on_stop_sec()

KV_LOAD_TIERS_KEY = "kv_load_tiers"
MATCHER_MEDIUM_KEY = "medium"
MATCHER_LOCALITY_KEY = "locality"


@dataclass(slots=True)
class TransferJobStatus:
    """Tracks scheduler-side state for a single transfer job."""

    req_id: ReqId
    # Number of workers still pending. Starts at num_workers,
    # decremented as each worker reports completion. Job is done at 0.
    pending_count: int
    # Offload keys this job covers; passed to manager.complete_*().
    keys: set[OffloadKey]
    is_store: bool
    # Store source blocks fenced after the request finishes.
    deferred_fence_block_ids: list[int] | None = None
    # Store source blocks fenced when the transfer is created.
    fenced_block_ids: list[int] | None = None


class GroupOffloadConfig(NamedTuple):
    group_idx: int
    tokens_per_block: int
    tokens_per_chunk: int
    hashes_per_chunk: int
    # KV cache spec metadata propagated onto emitted BlockStored events so
    # KV-aware consumers can classify and filter the group.
    kv_event_group_spec: OffloadingEventGroupSpec
    # None below means full attention
    sliding_window_size_in_chunks: int | None
    kv_cache_spec: KVCacheSpec
    # Cached manager class for this group's KV cache spec, resolved at init time
    manager_cls: type[SingleTypeKVCacheManager]
    # Partial-tail data for this group comes from the scheduler's CoW hand-off
    # rather than the request block table.
    requires_cow_source: bool = False
    # True for EAGLE/MTP draft-model attention groups. The trailing chunk
    # of these groups is volatile and lacks a stable hash, so it must
    # be excluded from store and load scheduling.
    is_eagle_group: bool = False

    def load_window_size_in_chunks(self, num_tokens: int) -> int | None:
        window = self.sliding_window_size_in_chunks
        if isinstance(self.kv_cache_spec, SlidingWindowSpec):
            assert window is not None
            # Lookup rounds the right edge up, but allocation retains the
            # window ending at the actual hit boundary, possibly one chunk left.
            right_padding = -num_tokens % self.tokens_per_chunk
            window = max(
                window,
                cdiv(
                    self.kv_cache_spec.sliding_window
                    - 1
                    + self.kv_cache_spec.extra_retained_tokens
                    + right_padding,
                    self.tokens_per_chunk,
                ),
            )
        return window


def get_sliding_window_size_in_chunks(
    kv_cache_spec: KVCacheSpec, tokens_per_chunk: int
) -> int | None:
    if isinstance(kv_cache_spec, SlidingWindowSpec):
        assert kv_cache_spec.sliding_window > 0
        return max(
            cdiv(kv_cache_spec.sliding_window, tokens_per_chunk),
            cdiv(
                kv_cache_spec.sliding_window - 1 + kv_cache_spec.extra_retained_tokens,
                tokens_per_chunk,
            ),
        )

    if isinstance(kv_cache_spec, ChunkedLocalAttentionSpec):
        # Attention never reaches back past one chunk
        assert kv_cache_spec.attention_chunk_size > 0
        return cdiv(kv_cache_spec.attention_chunk_size, tokens_per_chunk)

    if isinstance(kv_cache_spec, MambaSpec):
        # Mamba depends on a single state
        return 1

    assert isinstance(kv_cache_spec, FullAttentionSpec)
    return None


def resolve_mamba_align_size(
    spec: "OffloadingSpec", kv_cache_config: KVCacheConfig
) -> int | None:
    """Scan all KV cache groups in *spec* and return the single mamba alignment
    size, or None if no group requires mamba alignment.

    For MambaSpec groups in "align" cache mode the hit window must be
    rounded down to a multiple of the offloaded chunk size. Asserts that all
    such groups agree on the same value.
    """
    mamba_align_size: int | None = None
    for group in spec.config.groups:
        tokens_per_block = group.tokens_per_block
        kv_spec = kv_cache_config.kv_cache_groups[group.group_id].kv_cache_spec
        if isinstance(kv_spec, MambaSpec) and kv_spec.mamba_cache_mode == "align":
            tokens_per_chunk = tokens_per_block * spec.blocks_per_chunk
            assert mamba_align_size is None or mamba_align_size == tokens_per_chunk
            mamba_align_size = tokens_per_chunk
    return mamba_align_size


class SchedulerOffloadConfig(NamedTuple):
    kv_group_configs: tuple[GroupOffloadConfig, ...]
    blocks_per_chunk: int
    tokens_per_hash: int
    num_workers: int
    offload_prompt_only: bool
    supports_partial_tail: bool
    alignment_tokens: int | None = None
    retention_interval: int | None = None
    dcp_world_size: int = 1

    @classmethod
    def from_spec(
        cls,
        spec: OffloadingSpec,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ) -> "SchedulerOffloadConfig":
        # Determine the alignment token count from the full-attention group(s).
        # This is the tokens_per_chunk of the full-attention group; load
        # hits are always aligned to this boundary, so SWA blocks earlier in
        # each segment can never serve a load hit. Relevant for hybrid
        # architectures like DeepSeek V4 (MLA + SWA groups).
        full_attn_tokens_per_chunk: set[int] = set()
        for group in spec.config.groups:
            tokens_per_block = group.tokens_per_block
            kv_spec = kv_cache_config.kv_cache_groups[group.group_id].kv_cache_spec
            sw = get_sliding_window_size_in_chunks(
                kv_spec, tokens_per_block * spec.blocks_per_chunk
            )
            if sw is None:
                full_attn_tokens_per_chunk.add(tokens_per_block * spec.blocks_per_chunk)

        # Only apply the optimization if there's a single consistent
        # full-attention alignment size.  For pure SWA/Mamba models (no
        # full-attention groups), fall back to the largest SWA/Mamba
        # tokens_per_chunk so retention_interval has a boundary
        # granularity to work with.
        alignment_tokens: int | None = None
        if len(full_attn_tokens_per_chunk) == 1:
            alignment_tokens = full_attn_tokens_per_chunk.pop()
        elif not full_attn_tokens_per_chunk:
            all_tokens_per_chunk = [
                tpb * spec.blocks_per_chunk for tpb in spec.tokens_per_block
            ]
            if all_tokens_per_chunk:
                alignment_tokens = max(all_tokens_per_chunk)

        retention_interval = vllm_config.cache_config.prefix_cache_retention_interval

        eagle_groups = {
            group.group_id
            for group in spec.config.groups
            if kv_cache_config.kv_cache_groups[group.group_id].is_eagle_group
        }

        use_eagle_block_drop = (
            vllm_config.speculative_config is not None
            and vllm_config.speculative_config.use_eagle_block_drop()
        )
        if use_eagle_block_drop and not eagle_groups:
            # No group is annotated as holding drafter layers. This happens
            # for shared-group MTP models (e.g. Qwen3.5-style), whose drafter
            # is a regular decoder layer merged into the target's
            # full-attention group, and for separate-draft models that miss
            # annotation. Treating every group as a drafter group here makes
            # the volatile-tail pop apply to ALL groups, which can zero the
            # whole request's offload hit whenever any group's servable
            # window is a single chunk (issue #52735). Drafter KV served one
            # chunk stale can only lower speculative acceptance rates — the
            # target model verifies every draft — so fail toward serving.
            logger.info_once(
                "KV offloading: speculative decoding is enabled but no "
                "KV-cache group is annotated as a drafter group; treating "
                "all groups as non-draft for offloading."
            )

        if eagle_groups:
            logger.info(
                "KV offloading: EAGLE/MTP draft attention groups %s "
                "detected. The trailing chunk of these groups will be "
                "excluded from offloading due to volatility.",
                sorted(eagle_groups),
            )

        kv_group_configs_list: list[GroupOffloadConfig] = []
        for group in spec.config.groups:
            group_id = group.group_id
            tokens_per_block = group.tokens_per_block
            kv_cache_group = kv_cache_config.kv_cache_groups[group_id]
            kv_spec = kv_cache_group.kv_cache_spec
            sw = get_sliding_window_size_in_chunks(
                kv_spec, tokens_per_block * spec.blocks_per_chunk
            )
            manager_cls = KVCacheSpecRegistry.get_manager_class(kv_spec)
            assert manager_cls is not None, (
                f"No manager found for KV cache spec {type(kv_spec).__name__}"
            )
            kv_group_configs_list.append(
                GroupOffloadConfig(
                    group_idx=group_id,
                    tokens_per_block=tokens_per_block,
                    tokens_per_chunk=tokens_per_block * spec.blocks_per_chunk,
                    hashes_per_chunk=(
                        (tokens_per_block * spec.blocks_per_chunk)
                        // spec.tokens_per_hash
                    ),
                    sliding_window_size_in_chunks=sw,
                    kv_cache_spec=kv_spec,
                    manager_cls=manager_cls,
                    kv_event_group_spec=get_offloading_event_group_spec(kv_cache_group),
                    is_eagle_group=group_id in eagle_groups,
                    requires_cow_source=(
                        isinstance(kv_spec, MambaSpec)
                        and kv_spec.mamba_cache_mode == "align"
                    ),
                )
            )
        kv_group_configs = tuple(kv_group_configs_list)
        group_block_sizes = {config.tokens_per_block for config in kv_group_configs}
        has_partial_recurrent_group = any(
            config.requires_cow_source
            and config.tokens_per_block > spec.tokens_per_hash
            for config in kv_group_configs
        )
        # Partial tails currently require one physical block per offload chunk
        # and uniform, non-windowed groups so one boundary identifies every
        # group's source. EAGLE and DCP need additional hand-off semantics.
        supports_partial_tail = (
            spec.blocks_per_chunk == 1
            and len(group_block_sizes) == 1
            and has_partial_recurrent_group
            and all(
                config.sliding_window_size_in_chunks is None
                or config.requires_cow_source
                for config in kv_group_configs
            )
            and not any(config.is_eagle_group for config in kv_group_configs)
            and vllm_config.parallel_config.decode_context_parallel_size == 1
        )

        if retention_interval is not None:
            if retention_interval < 0:
                raise ValueError(
                    f"prefix_cache_retention_interval "
                    f"({retention_interval}) must be non-negative."
                )
            for config in kv_group_configs:
                if (
                    config.sliding_window_size_in_chunks is not None
                    and retention_interval % config.tokens_per_chunk != 0
                ):
                    raise ValueError(
                        f"prefix_cache_retention_interval "
                        f"({retention_interval}) must be a multiple of "
                        f"tokens_per_chunk ({config.tokens_per_chunk})."
                    )

        return cls(
            num_workers=vllm_config.parallel_config.world_size,
            kv_group_configs=kv_group_configs,
            blocks_per_chunk=spec.blocks_per_chunk,
            tokens_per_hash=spec.tokens_per_hash,
            offload_prompt_only=spec.offload_prompt_only,
            supports_partial_tail=supports_partial_tail,
            alignment_tokens=alignment_tokens,
            retention_interval=retention_interval,
            dcp_world_size=vllm_config.parallel_config.decode_context_parallel_size,
        )


@dataclass
class RequestGroupState:
    offload_keys: list[OffloadKey] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)
    # Index of the next chunk to offload.
    next_stored_chunk_idx: int = 0
    # Number of offloaded chunks hit (including GPU prefix cache)
    # when the request first started
    num_hit_chunks: int = 0


@dataclass(slots=True)
class RequestOffloadState:
    config: SchedulerOffloadConfig
    req: Request
    req_context: ReqContext
    offloading_context: RequestOffloadingContext
    group_states: tuple[RequestGroupState, ...] = field(init=False)
    # upper bound on tokens to load for this request; None means no cap
    max_load_tokens: int | None = None
    # upper bound on tokens to offload for this request; None means no cap
    max_offload_tokens: int | None = None
    # Store at compute time even while the pool is full (the write-through
    # gate is closed), e.g. for a prefix that is primed in order to be pinned.
    write_through: bool = False
    # number of hits in the GPU cache
    num_locally_computed_tokens: int = 0
    # In-flight job IDs. Per the connector's invariant, at any given time
    # this contains either a single load job, or one or more store jobs.
    transfer_jobs: set[int] = field(default_factory=set)
    # time.monotonic() of this request's first deferred offload lookup;
    # None once consumed (observed) or while no lookup is pending.
    deferred_lookup_start_time: float | None = None
    # Fine-grained token boundary selected beyond the last complete offload
    # chunk. It is consumed when the corresponding load is scheduled.
    partial_tail_boundary: int | None = None
    # Loaded attention keys cover their token positions. Recurrent state
    # supports the whole reused external prefix.
    load_key_ranges: list[tuple[int, int, OffloadKey]] = field(default_factory=list)
    # Hit boundary proven by the full-attention groups in the latest lookup,
    # before any sliding-window/Mamba group narrowed it; 0 if not reached.
    full_attention_hit_boundary: int = 0
    # True once on_request_finished has been signaled to the manager.
    finished_signaled: bool = False

    def __post_init__(self) -> None:
        self.group_states = tuple(
            RequestGroupState() for _ in self.config.kv_group_configs
        )
        params = self.req.kv_transfer_params

        # NOTE: This field is experimental and subject to change in the future.
        raw = params.get("max_load_tokens") if params else None
        if type(raw) is int and raw >= 0:
            self.max_load_tokens = raw
            logger.debug(
                "Request %s: max_load_tokens set to %d",
                self.req.request_id,
                raw,
            )
        elif raw is not None:
            logger.warning(
                "max_load_tokens must be a non-negative int, got %r; ignoring", raw
            )

        # NOTE: This field is experimental and subject to change in the future.
        raw = params.get("max_offload_tokens") if params else None
        if type(raw) is int and raw >= 0:
            self.max_offload_tokens = raw
            logger.debug(
                "Request %s: max_offload_tokens set to %d",
                self.req.request_id,
                raw,
            )
        elif raw is not None:
            logger.warning(
                "max_offload_tokens must be a non-negative int, got %r; ignoring", raw
            )

        self.write_through = bool(params) and params.get("write_through") is True

    def update_offload_keys(self) -> None:
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            for req_block_hash in islice(
                self.req.block_hashes,
                group_config.hashes_per_chunk * len(group_state.offload_keys)
                + group_config.hashes_per_chunk
                - 1,
                None,
                group_config.hashes_per_chunk,
            ):
                key = make_offload_key(req_block_hash, group_config.group_idx)
                group_state.offload_keys.append(key)
                self.req_context.set_offload_key_position(
                    key,
                    len(group_state.offload_keys) * group_config.tokens_per_chunk,
                )

    def update_block_id_groups(
        self, new_block_id_groups: tuple[list[int], ...] | None
    ) -> None:
        if new_block_id_groups is None:
            return

        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            group_state.block_ids.extend(new_block_id_groups[group_config.group_idx])

    def storable_chunks(
        self,
        group_config: "GroupOffloadConfig",
        group_state: RequestGroupState,
        num_offloadable_tokens: int,
    ) -> int:
        """Number of allocated and keyed leading chunks eligible for store.

        For eagle/MTP groups the volatile trailing chunk of the offloadable
        range is excluded while decoding: the draft-layer KV of the last
        accepted position may be rewritten after spec-token rejection. During
        prefill the trailing chunk is stable (the draft input for a chunk's
        last position is the next prompt token), so it is stored immediately.
        Once the request has finished, no further spec-token rejection can
        rewrite the tail, so the exclusion is lifted and the final chunk
        becomes storable (issue #52735). The exclusion must be applied
        consistently everywhere ``next_stored_chunk_idx`` is derived:
        otherwise the trailing chunk of each step is skipped on collection but
        jumped over by ``next_stored_chunk_idx``, so it is never re-considered
        and a permanent hole breaks prefix-reuse lookup. ``is_finished`` is
        monotonic, so the finish-time calls all see the lifted exclusion.
        """
        num_chunks = num_offloadable_tokens // group_config.tokens_per_chunk
        is_decoding = num_offloadable_tokens > self.req.num_prompt_tokens
        # Finished requests have no pending speculation.
        if group_config.is_eagle_group and is_decoding and not self.req.is_finished():
            num_chunks = max(0, num_chunks - 1)
        num_allocated_chunks = (
            len(group_state.block_ids) // self.config.blocks_per_chunk
        )
        num_keyed_chunks = len(group_state.offload_keys)
        return min(num_chunks, num_allocated_chunks, num_keyed_chunks)

    def advance_stored_idx(self, num_offloadable_tokens: int) -> None:
        # max(): at the prefill->decode transition of a chunk-aligned prompt,
        # storable_chunks drops by one (the eagle exclusion kicks in), and the
        # index must not move backwards past already-stored chunks.
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            group_state.next_stored_chunk_idx = max(
                group_state.next_stored_chunk_idx,
                self.storable_chunks(group_config, group_state, num_offloadable_tokens),
            )

    def update_num_hit_chunks(self, num_cached_tokens: int) -> None:
        for group_config, group_state in zip(
            self.config.kv_group_configs, self.group_states
        ):
            group_state.num_hit_chunks = (
                num_cached_tokens // group_config.tokens_per_chunk
            )


def _parse_tier_filter(raw: Any) -> TierFilter:
    """Parse raw kv_transfer_params tier matchers into a TierFilter."""
    if not isinstance(raw, list):
        logger.warning(
            "_parse_tier_filter: expected list, got %s; ignoring",
            type(raw).__name__,
        )
        return TierFilter.ALL
    matchers: list[TierMatcher] = []
    for entry in raw:
        if not isinstance(entry, dict):
            logger.warning("_parse_tier_filter: entry is not a dict; skipping")
            continue
        medium: Medium | None = None
        locality: Locality | None = None
        raw_medium = entry.get(MATCHER_MEDIUM_KEY)
        if raw_medium is not None:
            try:
                medium = Medium(raw_medium.upper())
            except (ValueError, AttributeError):
                logger.warning(
                    "_parse_tier_filter: unknown medium %r; skipping entry",
                    raw_medium,
                )
                continue
        raw_locality = entry.get(MATCHER_LOCALITY_KEY)
        if raw_locality is not None:
            try:
                locality = Locality(raw_locality.upper())
            except (ValueError, AttributeError):
                logger.warning(
                    "_parse_tier_filter: unknown locality %r; skipping entry",
                    raw_locality,
                )
                continue
        matchers.append(TierMatcher(medium=medium, locality=locality))
    if not matchers:
        if not raw:  # input was [] — user explicitly wants nothing
            return TierFilter(matchers=())
        # all entries were invalid — fall back to ALL
        return TierFilter.ALL
    return TierFilter(matchers=tuple(matchers))


def _create_req_context(req: Request) -> ReqContext:
    params = req.kv_transfer_params
    load_filter = TierFilter.ALL
    if params:
        raw = params.get(KV_LOAD_TIERS_KEY)
        if raw is not None:
            load_filter = _parse_tier_filter(raw)
    return ReqContext(
        req_id=req.request_id,
        kv_transfer_params=params,
        kv_hints=req.kv_hints,
        load_tier_filter=load_filter,
    )


class OffloadingConnectorScheduler:
    """Implementation of Scheduler side methods"""

    # B70 0014 class-level defaults (all off), so instances built without
    # __init__ (tests, subclasses) see upstream behaviour.
    _b70_trace: bool = False
    _b70_junction: bool = False
    _b70_guard: bool = False
    _b70_counts: Counter | None = None
    _b70_ctx: dict | None = None

    def __init__(
        self,
        spec: OffloadingSpec,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ):
        self.config = SchedulerOffloadConfig.from_spec(
            spec, vllm_config, kv_cache_config
        )
        self.manager: OffloadingManager = spec.get_manager()
        self._connector_stats = OffloadingConnectorStats()

        full_attention_groups: list[int] = []
        sliding_window_groups: list[int] = []
        for config_idx, group_config in enumerate(self.config.kv_group_configs):
            if group_config.sliding_window_size_in_chunks is None:
                full_attention_groups.append(config_idx)
            else:
                sliding_window_groups.append(config_idx)

        # sort sliding window groups by window size in decreasing order
        def _sliding_window_sort_key(i: int) -> int:
            val = self.config.kv_group_configs[i].sliding_window_size_in_chunks
            assert val is not None
            return val

        sliding_window_groups.sort(key=_sliding_window_sort_key, reverse=True)

        # used by _lookup
        self._full_attention_groups: tuple[int, ...] = tuple(full_attention_groups)
        self._sliding_window_groups: tuple[int, ...] = tuple(sliding_window_groups)
        self._lookup_groups = self._full_attention_groups + self._sliding_window_groups
        self._mamba_align_size: int | None = resolve_mamba_align_size(
            spec, kv_cache_config
        )
        self._partial_tail_block_size = (
            self.config.kv_group_configs[0].tokens_per_block
            if self.config.supports_partial_tail
            else 0
        )
        self._cow_source_groups = frozenset(
            config.group_idx
            for config in self.config.kv_group_configs
            if config.requires_cow_source
        )

        self._req_status: dict[ReqId, RequestOffloadState] = {}
        self._current_batch_load_jobs: dict[int, TransferJob] = {}
        self._current_batch_jobs_to_flush: set[int] = set()
        # GPU block IDs allocated in the current engine step
        self._current_batch_allocated_block_ids: set[int] = set()
        # if GPU prefix caching is enabled,
        # Track loaded chunks to avoid redundant loads.
        self._chunks_being_loaded: set[OffloadKey] | None = (
            set() if vllm_config.cache_config.enable_prefix_caching else None
        )

        # Job ID counter shared by loads and stores.
        self._job_counter: int = 0
        # Threshold value for stale jobs. All job ids >= _stale_job_threshold are
        # active jobs.
        self._stale_job_threshold: int = 0
        self._jobs: dict[int, TransferJobStatus] = {}

        # block_id -> pending store job_ids. Used to track jobs that needs
        # flushing in case a block is re-allocated by the KV cache manager.
        # Populated only for finished requests (running-request blocks are
        # protected by their ref_cnt) and for sliding window blocks (which can
        # be freed before a request finishes).
        self._block_id_to_pending_jobs: dict[int, set[int]] = {}

        # Write-back band (see WRITE_BACK): the GPU block pool, the blocks each
        # write-back job holds, and the key each block was last written under.
        self._gpu_block_pool: Any = None
        self._wb_held: dict[int, list[Any]] = {}
        self._wb_written: dict[int, tuple[OffloadKey, float]] = {}
        # Finished requests' key chains: run tag (the request's last attention
        # key) -> keys root first, and key -> its newest run.
        self._run_keys: "OrderedDict[OffloadKey, list[OffloadKey]]" = OrderedDict()
        # The request's offload keys per group, kept per run so a run whose
        # chain broke (a member left the GPU uncopied) can be re-trimmed.
        self._run_by_group: dict[OffloadKey, list[list[OffloadKey]]] = {}
        self._key_run: dict[OffloadKey, OffloadKey] = {}
        self._write_through = True
        # A class with recurrent state restores only from a run holding every
        # state group at some boundary; a lone block can never be one.
        self._state_group_ids = frozenset(
            config.group_idx
            for config in self.config.kv_group_configs
            if isinstance(config.kv_cache_spec, MambaSpec)
        )
        self._has_state_groups = bool(self._state_group_ids)
        # The full-attention group whose keys define a run's boundaries.
        self._chain_ref_group: int | None = next(
            (
                idx
                for idx, config in enumerate(self.config.kv_group_configs)
                if config.sliding_window_size_in_chunks is None
                and not isinstance(config.kv_cache_spec, MambaSpec)
            ),
            None,
        )
        self._config_idx_by_group = {
            config.group_idx: idx
            for idx, config in enumerate(self.config.kv_group_configs)
        }
        # Boundary block hash -> state groups the pool holds there.
        self._state_sets: "OrderedDict[bytes, set[int]]" = OrderedDict()
        # Keys known to have reached the pool (bounded; for accounting).
        self._pool_has: "OrderedDict[OffloadKey, None]" = OrderedDict()
        self._wb_inflight: set[OffloadKey] = set()
        self._wb_refused: dict[OffloadKey, float] = {}
        self._wb_rescue: list[tuple[OffloadKey, OffloadKey, int, int]] = []
        # Run tag -> one of its keys, for runs whose GPU copy left this step:
        # touched in the pool once per step, so they become newest there.
        self._wb_left: dict[OffloadKey, OffloadKey] = {}
        # Finished runs pinned until the band copies them (see _pin_finished):
        # tag -> (pin time, [(key, block)]).
        self._wb_pinned: dict[OffloadKey, tuple[float, list[tuple[OffloadKey, Any]]]] = {}
        self._wb_pinned_ids: set[int] = set()
        self._wb_pinned_n = 0
        # job id -> (submit time, blocks); copy latency EWMA (s) and max.
        self._wb_submitted: dict[int, tuple[float, int]] = {}
        self._wb_latency = 0.2
        self._wb_latency_max = 0.0
        # GPU block reuse rate (blocks/s) from BlockPool.num_popped.
        self._wb_rate = 0.0
        self._wb_popped = 0
        self._wb_rate_at = time.monotonic()
        self._wb_lead = WRITE_BACK_SCAN
        self._wb_touching = False
        self._wb_counts: Counter = Counter()
        # Flush on stop: None -> "armed" (stop requested; this step records
        # the runs of requests the stop finished) -> "ready" -> "running"
        # (jobs submitted) -> "done" or "expired" (out of time: the engine
        # exits and the pool drops the unfinished stores).
        self._flush_state: str | None = None
        self._flush_t0 = 0.0
        self._flush_jobs: dict[int, tuple[int, int]] = {}  # job -> (run, blocks)
        self._flush_counts: Counter = Counter()

        self._events_tracker = OffloadingEventsTracker(spec.kv_events_config)

        # B70 0014 (env-gated; every flag off = upstream behaviour).
        self._b70_trace = _b70.trace_enabled()
        self._b70_junction = _b70.junction_enabled()
        self._b70_guard = _b70.guard_enabled()
        self._b70_guard_logged: dict[ReqId, tuple[int, ...]] = {}
        self._b70_counts: Counter | None = None
        self._b70_ctx: dict | None = None
        # req_id -> [stores submitted, stores completed, loads submitted,
        #            loads completed]
        self._b70_req_jobs: dict[ReqId, list[int]] = {}
        self._b70_state_logger: _b70.PeriodicStateLogger | None = None
        self._b70_kind_by_group: dict[int, str] = {
            config.group_idx: (
                "mamba"
                if isinstance(config.kv_cache_spec, MambaSpec)
                else "full"
                if config.sliding_window_size_in_chunks is None
                else "swa"
            )
            for config in self.config.kv_group_configs
        }
        if self._b70_trace:
            slot_bytes = int(getattr(spec, "kv_bytes_per_chunk", 0) or 0)
            _b70.log(
                "enabled scheduler trace: groups=%s retention_interval=%s "
                "alignment_tokens=%s partial_tail=%s slot_bytes=%d period_s=%.0f "
                "junction=%s empty_advance_guard=%s",
                dict(Counter(self._b70_kind_by_group.values())),
                self.config.retention_interval,
                self.config.alignment_tokens,
                self.config.supports_partial_tail,
                slot_bytes,
                _b70.trace_period_s(),
                self._b70_junction,
                self._b70_guard,
            )
            self._b70_state_logger = _b70.PeriodicStateLogger(
                _b70.trace_period_s(), lambda: self._b70_state_line(slot_bytes)
            )

    # --- B70 0014 helpers (only called with a flag on) ---

    def _b70_kind(self, group_idx: int) -> str:
        return self._b70_kind_by_group.get(group_idx, "?")

    def _b70_state_line(self, slot_bytes: int) -> str:
        jobs = list(self._jobs.values())
        n_store = sum(1 for j in jobs if j.is_store)
        return (
            _b70.cpu_tier_snapshot(self.manager, self._b70_kind, slot_bytes)
            + f" inflight_store_jobs={n_store}"
            f" write_back={dict(self._wb_counts)}"
            f" band_lead={self._wb_lead} band_rate={self._wb_rate:.1f}"
            f" band_latency={self._wb_latency:.3f}/{self._wb_latency_max:.3f}"
            f" band_held={sum(len(b) for b in self._wb_held.values())}"
            f" inflight_load_jobs={len(jobs) - n_store}"
            f" tracked_reqs={len(self._req_status)}"
        )

    def _b70_job_submitted(self, req_id: ReqId, is_store: bool) -> None:
        counts = self._b70_req_jobs.setdefault(req_id, [0, 0, 0, 0])
        counts[0 if is_store else 2] += 1

    def _b70_job_completed(self, job_status: TransferJobStatus) -> None:
        counts = self._b70_req_jobs.setdefault(job_status.req_id, [0, 0, 0, 0])
        counts[1 if job_status.is_store else 3] += 1

    def _b70_req_done(self, req_id: ReqId) -> None:
        self._b70_guard_logged.pop(req_id, None)
        counts = self._b70_req_jobs.pop(req_id, [0, 0, 0, 0])
        _b70.log(
            "req-done req=%s store_jobs submitted=%d completed=%d "
            "load_jobs submitted=%d completed=%d",
            req_id,
            *counts,
        )

    def _b70_maybe_set_junction(
        self, req_status: RequestOffloadState, result: int, ctx: dict
    ) -> None:
        """0014b: pin a shared-prefix junction where the offload hit was cut.

        The full-attention groups are looked up first; the first Mamba group
        then starts from the boundary they confirmed. If that sparse group has
        no stored state at or below it, the whole request misses (or hits
        shorter), and without a junction the recompute stores its only Mamba
        checkpoint at ``aligned(num_prompt_tokens - 1)`` again -- inside the
        part that varies -- so the prefix misses forever. Setting
        ``request.shared_prefix_boundary`` makes the core scheduler stop a
        prefill chunk there, the Mamba manager retain (hash) that state, and
        the aligned hand-off store it, so the next revisit hits.
        """
        boundary = None
        fa_seen = False
        for gidx, _start, _n_keys, n_hit, _counts, max_before in ctx["groups"]:
            kind = self._b70_kind(gidx)
            if kind == "full":
                if n_hit is None:
                    return  # deferred: the full-attention prefix is unconfirmed
                fa_seen = True
                continue
            if kind == "mamba" and fa_seen:
                boundary = max_before
            break
        if boundary is None:
            return
        req = req_status.req
        if req_status.num_locally_computed_tokens + result >= boundary:
            return  # the sparse group did not cut the hit
        if boundary <= req.shared_prefix_boundary:
            return
        req.shared_prefix_boundary = boundary
        ctx["junction"] = boundary

    def _b70_pending_skipped(
        self,
        req_status: RequestOffloadState,
        offered: Iterable[OffloadKey],
        stored: Iterable[OffloadKey],
    ) -> set[OffloadKey]:
        """0014d: offered keys prepare_store skipped as present that are still
        write-pending (a store in flight, possibly under another request)."""
        stored = set(stored)
        return {
            key
            for key in offered
            if key not in stored
            and self.manager.lookup(key, req_status.req_context)
            is LookupResult.HIT_PENDING
        }

    def _b70_guard_hold(
        self,
        req_status: RequestOffloadState,
        group_store_ranges: list[tuple[int, int]],
        pending: set[OffloadKey],
    ) -> None:
        """0014d: keep next_stored_chunk_idx at the first write-pending chunk
        so the range is offered again until that store lands (vllm#56795)."""
        held = []
        for group_config, group_state, (start, end) in zip(
            self.config.kv_group_configs, req_status.group_states, group_store_ranges
        ):
            for idx in range(start, min(end, len(group_state.offload_keys))):
                if group_state.offload_keys[idx] in pending:
                    if idx < group_state.next_stored_chunk_idx:
                        group_state.next_stored_chunk_idx = idx
                    held.append(idx)
                    break
        if self._b70_trace:
            sig = (len(pending), *sorted(set(held)))
            req_id = req_status.req.request_id
            if self._b70_guard_logged.get(req_id) != sig:
                self._b70_guard_logged[req_id] = sig
                _b70.log(
                    "guard-hold req=%s pending_keys=%d held_at_chunks=%s",
                    req_id,
                    len(pending),
                    sorted(set(held)),
                )

    def _b70_log_lookup(
        self, req_status: RequestOffloadState, result: int | None, ctx: dict
    ) -> None:
        req = req_status.req
        by_kind: dict[str, list] = {}
        for gidx, start, n_keys, n_hit, counts, max_before in ctx["groups"]:
            by_kind.setdefault(self._b70_kind(gidx), []).append((n_hit, counts))
        parts = []
        for kind, entries in sorted(by_kind.items()):
            hits = [h for h, _ in entries if h is not None]
            tot: Counter = Counter()
            for _, c in entries:
                tot.update(c)
            parts.append(
                f"{kind}[lookups={len(entries)} "
                f"hit_chunks={min(hits) if hits else '-'}.."
                f"{max(hits) if hits else '-'} deferred={len(entries) - len(hits)} "
                f"HIT={tot[LookupResult.HIT]} PENDING={tot[LookupResult.HIT_PENDING]} "
                f"RETRY={tot[LookupResult.RETRY]} MISS={tot[LookupResult.MISS]}]"
            )
        zeroed = "-"
        if result == 0 and ctx["groups"] and ctx["groups"][-1][3] == 0:
            gidx, start, n_keys, n_hit, counts, max_before = ctx["groups"][-1]
            zeroed = f"g{gidx}({self._b70_kind(gidx)})@{max_before}"
        _b70.log(
            "lookup req=%s prompt=%d local=%d hit=%s %s zeroed_by=%s "
            "shared_prefix_boundary=%s junction_set=%s",
            req.request_id,
            req.num_prompt_tokens,
            req_status.num_locally_computed_tokens,
            result,
            " ".join(parts),
            zeroed,
            getattr(req, "shared_prefix_boundary", None),
            ctx.get("junction"),
        )

    def _maybe_observe_lookup_async_delay(
        self, req_status: RequestOffloadState
    ) -> None:
        start_time = req_status.deferred_lookup_start_time
        if start_time is None:
            return
        req_status.deferred_lookup_start_time = None
        self._connector_stats.observe_histogram(
            _ConnectorMetricName.LOOKUP_ASYNC_DELAY,
            time.monotonic() - start_time,
        )

    def bind_gpu_block_pool(self, gpu_block_pool: Any) -> None:
        self._gpu_block_pool = gpu_block_pool
        if WRITE_BACK and hasattr(gpu_block_pool, "evict_listener"):
            gpu_block_pool.evict_listener = self._on_gpu_evict
            gpu_block_pool.revive_listener = self._on_gpu_revive

    def _write_through_enabled(self) -> bool:
        """Compute-time stores run unless the band is on and the pool is full
        (its class has under WRITE_THROUGH_MIN_FREE of its rows free)."""
        if not WRITE_BACK or self._gpu_block_pool is None:
            return True
        if not WRITE_BACK_MIRROR:
            return False
        free_fraction = getattr(self.manager, "pool_free_fraction", None)
        fraction = free_fraction() if free_fraction is not None else None
        return fraction is None or fraction >= WRITE_THROUGH_MIN_FREE

    def _run_tag(self, by_group: list[list[OffloadKey]]) -> OffloadKey | None:
        """A run's tag: the request's last attention key (its conversation)."""
        attn = next(
            (
                keys
                for idx, keys in enumerate(by_group)
                if keys
                and idx in self._full_attention_groups
                and not isinstance(
                    self.config.kv_group_configs[idx].kv_cache_spec, MambaSpec
                )
            ),
            None,
        )
        return attn[-1] if attn else None

    def _pin_finished(self, req_status: "RequestOffloadState") -> None:
        """Pin (one extra reference) the GPU blocks of a finishing request's
        run that the pool lacks, before the scheduler frees them. The run is
        recorded only at the end of the next schedule step, after that step's
        allocations, which under pressure reuse the freed blocks first: the
        run then broke and its conversation was lost (2026-11-04). The band
        copies a pinned run in its next step, whatever its budget."""
        by_group = [gs.offload_keys for gs in req_status.group_states]
        tag = self._run_tag(by_group)
        if tag is None or tag in self._wb_pinned:
            return
        pool = self._gpu_block_pool
        chain = self._restore_chain(
            by_group, have=lambda k: k in self._pool_has or self._gpu_has(k)
        )
        pins = []  # (key, block): a lookup-only alias block may carry no hash
        for key in chain:
            if key in self._pool_has or key in self._wb_inflight:
                continue
            found = pool.get_cached_block(
                BlockHash(bytes(key)[:-4]), [get_offload_group_idx(key)]
            )
            if found:
                pins.append((key, found[0]))
        if not pins:
            return
        if self._wb_pinned_n + len(pins) > WRITE_BACK_MAX_HELD:
            self._wb_counts["pin_over_cap"] += 1
            return
        pool.touch([block for _, block in pins])
        self._wb_pinned[tag] = (time.monotonic(), pins)
        self._wb_pinned_n += len(pins)
        self._wb_pinned_ids.update(block.block_id for _, block in pins)
        self._wb_counts["runs_pinned"] += 1

    def _unpin(self, tag: OffloadKey) -> None:
        """Drop a run's pins. A block left unreferenced returns to the free
        queue: at the head if the pool holds it, else at the tail."""
        entry = self._wb_pinned.pop(tag, None)
        if entry is None:
            return
        queue = self._gpu_block_pool.free_block_queue
        self._wb_pinned_n -= len(entry[1])
        for key, block in entry[1]:
            self._wb_pinned_ids.discard(block.block_id)
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                if key in self._pool_has:
                    queue.prepend_n([block])
                else:
                    queue.append_n([block])

    def _record_run(self, req_status: "RequestOffloadState") -> None:
        """Name a finished request's run after its conversation (the request's
        last attention key): a manager that tags runs moves what the request
        stored into it; with the band on, remember what a restore of it reads
        (see _restore_chain) so the band can write exactly that, whole."""
        by_group = [gs.offload_keys for gs in req_status.group_states]
        tag = self._run_tag(by_group)
        if tag is None:
            return
        merge_tag = getattr(self.manager, "merge_tag", None)
        if merge_tag is not None:
            merge_tag(req_status.req_context, tag)
        if not WRITE_BACK:
            return
        chain = self._restore_chain(
            by_group, have=lambda k: k in self._pool_has or self._gpu_has(k)
        )
        self._drop_run(tag)
        if not chain:
            # Nothing a restore could read (no boundary with every state
            # group stored): never offered, so never refused every step.
            self._wb_counts["run_unrestorable_dropped"] += 1
            return
        self._run_keys[tag] = chain
        self._run_by_group[tag] = [list(keys) for keys in by_group]
        for key in chain:
            self._key_run[key] = tag
        while len(self._run_keys) > WRITE_BACK_MAX_RUNS:
            self._drop_run(next(iter(self._run_keys)))

    def _drop_run(self, tag: OffloadKey) -> None:
        old = self._run_keys.pop(tag, None)
        self._run_by_group.pop(tag, None)
        self._wb_refused.pop(tag, None)
        if old is not None:
            for key in old:
                if self._key_run.get(key) == tag:
                    del self._key_run[key]

    def _run_members(self, tag: OffloadKey, config_idx_by_group: dict[int, int]):
        """The run's members the band can copy now: (key, block, config_idx,
        num_tokens) for every free GPU block of its chain not yet in the pool.
        A chain member that is neither on the GPU nor known in the pool broke
        the chain; the run is then re-trimmed to the deepest boundary below
        the break (what a restore can still read), or dropped when none is
        left, so the band never writes a chain no restore can use."""
        for _attempt in range(2):
            members, missing = [], []
            for run_key in self._run_keys[tag]:
                if run_key in self._wb_inflight:
                    continue
                found = self._storable_free_block(run_key, config_idx_by_group)
                if found is not None:
                    members.append((run_key, *found))
                elif run_key not in self._pool_has and not self._gpu_has(run_key):
                    missing.append(run_key)  # in use on the GPU is not missing
            if not missing:
                return members
            by_group = self._run_by_group.get(tag)
            if by_group is None:
                break
            configs = self.config.kv_group_configs
            limit = None
            missing_set = set(missing)
            for idx, keys in enumerate(by_group):
                tpc = configs[idx].tokens_per_chunk
                for i, key in enumerate(keys):
                    if key in missing_set:
                        end = (i + 1) * tpc - 1  # usable depths end before it
                        limit = end if limit is None else min(limit, end)
                        break
            chain = self._restore_chain(
                by_group,
                limit=limit,
                have=lambda k: k in self._pool_has
                or self._storable_free_block(k, config_idx_by_group) is not None,
            )
            if not chain:
                break
            self._wb_counts["run_broken_trimmed"] += 1
            for key in self._run_keys[tag]:
                if self._key_run.get(key) == tag:
                    del self._key_run[key]
            self._run_keys[tag] = chain
            for key in chain:
                self._key_run[key] = tag
        self._wb_counts["run_broken_dropped"] += 1
        self._drop_run(tag)
        return []

    def _note_state_stored(self, key: OffloadKey) -> None:
        """A recurrent-state key at a boundary is in the pool (stored or found
        present by a hand-off): remember its group under the boundary hash."""
        h = bytes(key)[:-4]
        groups = self._state_sets.get(h)
        if groups is None:
            groups = self._state_sets[h] = set()
            while len(self._state_sets) > _STATE_SETS_MAX:
                self._state_sets.popitem(last=False)
        else:
            self._state_sets.move_to_end(h)
        groups.add(get_offload_group_idx(key))
        self._note_in_pool([key])

    def _note_states(self, keys: Iterable[OffloadKey]) -> None:
        for key in keys:
            if get_offload_group_idx(key) in self._state_group_ids:
                self._note_state_stored(key)

    def _restore_chain(
        self,
        by_group: list[list[OffloadKey]],
        limit: int | None = None,
        have: Callable[[OffloadKey], bool] | None = None,
    ) -> list[OffloadKey]:
        """The keys a restore of this conversation reads, root first.

        Without recurrent state that is every key. With it, a restore resumes
        only at a boundary whose state keys are all in the pool (they reach it
        only through the aligned hand-offs), so the chain is trimmed to the
        deepest such boundary d: full-attention keys up to d, the sliding
        window ending at d and the state keys at d. Empty when no boundary
        has its state stored. ``have`` also accepts a boundary whose state keys
        it reports present (on the GPU or in the pool): the GPU keeps the state
        of every cached boundary, so the band can copy it from there."""
        configs = self.config.kv_group_configs
        if not self._state_group_ids:
            if limit is not None:
                by_group = [
                    keys[: (limit + 1) // configs[idx].tokens_per_chunk]
                    for idx, keys in enumerate(by_group)
                ]
            n = max((len(k) for k in by_group), default=0)
            return [keys[i] for i in range(n) for keys in by_group if i < len(keys)]
        ref = self._chain_ref_group
        if ref is None or not by_group[ref]:
            return []
        ref_tpc = configs[ref].tokens_per_chunk
        # A lookup drops the trailing chunk of an EAGLE/MTP drafter group (its
        # KV there is volatile), so a restore at boundary d reads that group
        # one chunk past d and the state at d: the chain carries it too.
        eagle = [
            idx
            for idx, config in enumerate(configs)
            if config.is_eagle_group and not isinstance(config.kv_cache_spec, MambaSpec)
        ]

        def eagle_fits(boundary: int) -> bool:
            for idx in eagle:
                tpc = configs[idx].tokens_per_chunk
                end = boundary + tpc
                if boundary % tpc or len(by_group[idx]) < end // tpc:
                    return False
                if limit is not None and end > limit:
                    return False
            return True

        depth = 0
        for i in range(len(by_group[ref]) - 1, -1, -1):
            if limit is not None and (i + 1) * ref_tpc > limit:
                continue
            boundary = (i + 1) * ref_tpc
            if not eagle_fits(boundary):
                continue
            groups = self._state_sets.get(bytes(by_group[ref][i])[:-4])
            if groups is not None and self._state_group_ids <= groups:
                depth = boundary
                break
            if have is not None and self._states_present(by_group, boundary, have):
                depth = boundary
                break
        if not depth:
            return []
        per_group: list[tuple[int, int, int]] = []  # (first, end) chunk per group
        for idx, (config, keys) in enumerate(zip(configs, by_group)):
            end = min(len(keys), depth // config.tokens_per_chunk)
            if end <= 0 or depth % config.tokens_per_chunk:
                per_group.append((0, 0, idx))
                continue
            if idx in eagle:
                end = min(len(keys), end + 1)
            if isinstance(config.kv_cache_spec, MambaSpec):
                first = end - 1
            elif config.sliding_window_size_in_chunks is not None:
                window = config.load_window_size_in_chunks(depth) or end
                first = max(0, end - window)
            else:
                first = 0
            per_group.append((first, end, idx))
        n = max(end for _, end, _ in per_group)
        return [
            by_group[idx][i]
            for i in range(n)
            for first, end, idx in per_group
            if first <= i < end
        ]

    def _states_present(
        self, by_group: list[list[OffloadKey]], boundary: int, have
    ) -> bool:
        """Whether every state group's key at ``boundary`` passes ``have``."""
        configs = self.config.kv_group_configs
        found = False
        for idx, config in enumerate(configs):
            if not isinstance(config.kv_cache_spec, MambaSpec):
                continue
            tpc = config.tokens_per_chunk
            if boundary % tpc:
                return False
            j = boundary // tpc - 1
            if not 0 <= j < len(by_group[idx]) or not have(by_group[idx][j]):
                return False
            found = True
        return found

    def _gpu_has(self, key: OffloadKey) -> bool:
        """The key's block is cached on the GPU (any reference count)."""
        pool = self._gpu_block_pool
        if pool is None:
            return False
        group = int.from_bytes(bytes(key)[-4:], "big")
        return bool(pool.get_cached_block(BlockHash(bytes(key)[:-4]), [group]))

    def _storable_free_block(self, key: OffloadKey, config_idx_by_group: dict[int, int]):
        """(block, config_idx, num_tokens) for key if its GPU block is free (settled,
        the next eviction candidates, or held only by a pin) and chunk-aligned;
        else None."""
        group = int.from_bytes(bytes(key)[-4:], "big")
        config_idx = config_idx_by_group.get(group)
        if config_idx is None:
            return None
        found = self._gpu_block_pool.get_cached_block(BlockHash(bytes(key)[:-4]), [group])
        if not found:
            return None
        block = found[0]
        num_tokens = block.block_hash_num_tokens
        free = block.ref_cnt == 0 or (
            block.ref_cnt == 1 and block.block_id in self._wb_pinned_ids
        )
        if (not free or not num_tokens
                or num_tokens % self.config.kv_group_configs[config_idx].tokens_per_chunk):
            return None
        return block, config_idx, num_tokens

    # --- band pacing and accounting ---

    def _note_in_pool(self, keys: Iterable[OffloadKey]) -> None:
        has = self._pool_has
        for key in keys:
            has[key] = None
            has.move_to_end(key)
        while len(has) > _POOL_HAS_MAX:
            has.popitem(last=False)

    def _forget_in_pool(self, keys: Iterable[OffloadKey]) -> int:
        """Keys the pool no longer holds: drop them (and, for a state key, its
        boundary's state set) from the band's view. Returns how many it knew."""
        n = 0
        for key in keys:
            if self._pool_has.pop(key, 0) is None:
                self._state_sets.pop(bytes(key)[:-4], None)
                n += 1
        return n

    def _prepare_store(self, keys: Any, req_context: Any) -> Any:
        out = self.manager.prepare_store(keys, req_context)
        self._note_evicted(out)
        return out

    def _prepare_store_run(self, keys: Any, tag: OffloadKey, whole: bool = False) -> Any:
        out = self.manager.prepare_store_run(keys, tag, whole=whole)
        self._note_evicted(out)
        return out

    def _note_evicted(self, store_output: Any) -> None:
        """A store's own evictions leave the pool: without this the band kept
        counting them as present, never rewrote them when the GPU dropped its
        copy, and counted that loss as evict_copied (H1, 2026-11-03)."""
        evicted = getattr(store_output, "evicted_keys", None) if store_output else None
        if evicted and WRITE_BACK:
            self._wb_counts["pool_evicted_forgot"] += self._forget_in_pool(evicted)

    def _on_gpu_evict(self, block: Any) -> None:
        """BlockPool hook: a hashed block is being reused. A member of a known
        run that the pool does not hold is lost (the band never waits for it);
        with WRITE_BACK_RESCUE set, up to that many per step are copied out
        instead, and the step's forward waits for the copy (bounded wait). A
        member the pool holds makes its run the pool's only copy: it is
        touched there (newest), so the pool evicts runs the GPU still holds
        first."""
        key = OffloadKey(bytes(block.block_hash))
        tag = self._key_run.get(key)
        if tag is None:
            self._wb_counts["evict_other"] += 1
            return
        if key in self._pool_has or key in self._wb_inflight:
            self._wb_counts["evict_copied"] += 1
            if key in self._pool_has:
                self._wb_left.setdefault(tag, key)
            return
        if len(self._wb_rescue) < WRITE_BACK_RESCUE:
            config_idx = self._config_idx_by_group.get(get_offload_group_idx(key))
            num_tokens = block.block_hash_num_tokens
            if config_idx is not None and num_tokens and not (
                num_tokens % self.config.kv_group_configs[config_idx].tokens_per_chunk
            ):
                self._wb_rescue.append((tag, key, block.block_id, config_idx))
                self._wb_counts["evict_rescued"] += 1
                return
        self._wb_counts["evict_uncopied"] += 1

    def _on_gpu_revive(self, block: Any) -> None:
        """BlockPool hook: a free hashed block is hit again on the GPU. If the
        band already copied it, that copy is (for now) a duplicate."""
        if self._wb_touching:
            return  # the band's own hold
        if self._wb_written.get(block.block_id, (None,))[0] == OffloadKey(
            bytes(block.block_hash)
        ):
            self._wb_counts["revived_copied"] += 1

    def _band_lead(self, now: float) -> int:
        """Free-queue positions the band scans: the blocks the GPU will reuse
        within LEAD_FACTOR copy latencies at the measured reuse rate."""
        pool = self._gpu_block_pool
        popped = getattr(pool, "num_popped", None)
        if popped is not None:
            dt = now - self._wb_rate_at
            if dt >= 0.5:
                rate = (popped - self._wb_popped) / dt
                self._wb_rate = 0.7 * self._wb_rate + 0.3 * rate
                self._wb_popped, self._wb_rate_at = popped, now
        latency = max(self._wb_latency, 0.05)
        lead = int(self._wb_rate * latency * WRITE_BACK_LEAD_FACTOR) + WRITE_BACK_SCAN
        return min(lead, WRITE_BACK_SCAN_MAX)

    def _build_write_back_jobs(self) -> dict[int, TransferJob]:
        """Write whole conversations that are about to leave the GPU, root first.

        A hashed block among the first ``_band_lead`` free-queue positions
        triggers its run: every free, chunk-aligned block of what a restore of
        that run reads is offered to the pool in one tagged, all-or-nothing
        store, copied as one job while held, and released to the queue head.
        The band holds at most WRITE_BACK_FREE_SHARE of the free blocks, so allocation is
        never starved by it; a block the GPU reuses before its copy started is
        counted as lost (see _on_gpu_evict), never waited for. Blocks of no
        known run are offered alone on classes without recurrent state."""
        pool = self._gpu_block_pool
        if not WRITE_BACK or pool is None or self.config.blocks_per_chunk != 1:
            return {}
        now = time.monotonic()
        rescue_jobs = self._build_rescue_jobs(now)
        held = sum(len(b) for b in self._wb_held.values())
        free = pool.get_num_free_blocks()
        budget = min(WRITE_BACK_MAX_HELD, int((free + held) * WRITE_BACK_FREE_SHARE)) - held
        lead = self._band_lead(now)
        self._wb_lead = lead
        if self._wb_left:
            self.manager.touch(list(self._wb_left.values()), None)
            self._wb_left.clear()
        config_idx_by_group = self._config_idx_by_group
        units: list[tuple[OffloadKey | None, list]] = []
        seen_runs: set[OffloadKey] = set()
        # Pinned finished runs first, outside the budget: their blocks are out
        # of the free queue already. Pins end after this step's copy jobs.
        pinned = []
        for tag, (pinned_at, _) in list(self._wb_pinned.items()):
            if tag not in self._run_keys:
                if now - pinned_at > WRITE_BACK_PIN_S:
                    self._unpin(tag)  # never recorded, or dropped
                continue
            pinned.append(tag)
            seen_runs.add(tag)
            members = self._run_members(tag, config_idx_by_group)
            if any(m[0] not in self._pool_has for m in members):
                units.append((tag, members))
        if budget <= 0 and self._wb_held:
            self._wb_counts["step_budget_full"] += 1
        queue = pool.free_block_queue
        node = queue.fake_free_list_head.next_free_block
        taken = 0
        for _ in range(lead):
            if node is None or node is queue.fake_free_list_tail or taken >= budget:
                break
            block, node = node, node.next_free_block
            if block.block_hash is None:
                continue
            key = OffloadKey(bytes(block.block_hash))
            written = self._wb_written.get(block.block_id)
            if written is not None and written[0] == key and now - written[1] < 10.0:
                continue  # confirmed in the pool recently
            tag = self._key_run.get(key)
            if tag is not None:
                if tag in seen_runs:
                    continue
                seen_runs.add(tag)
                refused = self._wb_refused.get(tag)
                if refused is not None and now - refused < WRITE_BACK_REFUSED_BACKOFF_S:
                    continue
                members = self._run_members(tag, config_idx_by_group)
                # Only members the pool lacks are held and copied. With none,
                # skip: a re-offer would make a run the GPU still holds newest
                # in the pool, ahead of runs only the pool holds.
                need = sum(1 for m in members if m[0] not in self._pool_has)
                if not need:
                    continue
                if taken + need > budget:
                    # A run larger than the whole budget is written alone
                    # once nothing else is held, rather than never.
                    if taken or self._wb_held:
                        self._wb_counts["run_deferred_budget"] += 1
                        continue
                    self._wb_counts["run_oversize"] += 1
                self._wb_counts["run_offered"] += 1
            else:
                if self._has_state_groups:
                    continue  # a lone block can never be restored
                found = self._storable_free_block(key, config_idx_by_group)
                if found is None:
                    continue
                members = [(key, *found)]
                need = 1
            units.append((tag, members))
            taken += need

        store_jobs = rescue_jobs
        for tag, members in units:
            keys = self._offer_keys(tag, members)
            if hasattr(self.manager, "prepare_store_run"):
                # A known run is a whole conversation; a block of no known
                # run is stored alone, as a run of its own.
                store_output = self._prepare_store_run(
                    keys, tag if tag is not None else keys[0], whole=tag is not None
                )
            else:
                store_output = self._prepare_store(keys, None)
            if store_output is None and tag is not None:
                # The pool found the run unrestorable without the keys the
                # band left out as present (another engine's store evicted
                # them, or a load demoted their run and it went first):
                # forget them and re-trim now, to what the GPU still holds.
                # Waiting out the refusal backoff lost the run: the GPU
                # reused its blocks meanwhile (H1, 2026-11-03).
                offered = {m[0] for m in members}
                forgot = self._forget_in_pool(
                    k for k in self._run_keys.get(tag, ()) if k not in offered
                )
                if forgot and tag in self._run_keys:
                    self._wb_counts["store_refused_retrimmed"] += 1
                    retry = self._run_members(tag, config_idx_by_group)
                    if retry:
                        members = retry
                        keys = self._offer_keys(tag, members)
                        store_output = self._prepare_store_run(keys, tag, whole=True)
            if store_output is None:
                self._wb_counts["store_refused"] += 1
                if tag is not None:
                    self._wb_refused[tag] = now
                self._connector_stats.increase_counter(_ConnectorMetricName.ALLOCATION_FAILURE)
                continue
            chunk_by_key = dict(
                zip(store_output.keys_to_store, store_output.store_spec.block_ids.tolist())
            )
            new = []
            for key, block, config_idx, _num_tokens in members:
                self._wb_written[block.block_id] = (key, now)
                chunk = chunk_by_key.get(key)
                if chunk is None:
                    self._note_in_pool([key])  # already there (joined the run)
                    continue
                new.append((key, block, config_idx, chunk))
            if not new:
                continue
            self._wb_touching = True
            try:
                pool.touch([m[1] for m in new])
            finally:
                self._wb_touching = False
            job_id, job = self._band_job(new, store_output, now)
            self._wb_held[job_id] = [m[1] for m in new]
            store_jobs[job_id] = job
        for tag in pinned:
            self._unpin(tag)  # its copy jobs hold what they copy
        return store_jobs

    def _offer_keys(self, tag: OffloadKey | None, members: list) -> list[OffloadKey]:
        """The keys a band store offers: the members it copies and, for a run,
        the chain keys the pool already holds, so they join the run. Left out,
        they stayed only in the conversation's older runs and left the pool
        with the oldest of them (2026-11-04)."""
        keys = [m[0] for m in members]
        if tag is None:
            return keys
        copying = set(keys)
        return [k for k in self._run_keys.get(tag, ()) if k in copying or k in self._pool_has]

    def _band_job(
        self, members: list[tuple[OffloadKey, Any, int, int]], store_output: Any, now: float
    ) -> tuple[int, TransferJob]:
        """One store job copying ``members`` (key, block, config_idx, chunk),
        ordered by group as GPULoadStoreSpec requires."""
        members = sorted(members, key=lambda m: m[2])
        num_groups = len(self.config.kv_group_configs)
        group_sizes = [0] * num_groups
        for m in members:
            group_sizes[m[2]] += 1
        keys = {m[0] for m in members}
        job_id = self._generate_job_id()
        self._jobs[job_id] = TransferJobStatus(
            req_id=_WRITE_BACK_REQ,
            pending_count=self.config.num_workers,
            keys=keys,
            is_store=True,
        )
        self._wb_inflight.update(keys)
        self._wb_submitted[job_id] = (now, len(members))
        self._wb_counts["blocks_submitted"] += len(members)
        # blocks_per_chunk == 1: block_indices only offsets a partial first
        # chunk, which a whole-block copy never has.
        job = TransferJob(
            req_id=_WRITE_BACK_REQ,
            # Free, held blocks: no forward writes them, so the copy needs no
            # wait for the compute stream (its own lane in the worker).
            src_spec=GPULoadStoreSpec(
                [m[1].block_id for m in members],
                group_sizes=group_sizes,
                block_indices=[0] * num_groups,
                settled=True,
            ),
            dst_spec=type(store_output.store_spec)([m[3] for m in members]),
        )
        return job_id, job

    def _build_rescue_jobs(self, now: float) -> dict[int, TransferJob]:
        """WRITE_BACK_RESCUE: copy out run members the GPU reused this step
        before the band reached them. Their blocks now belong to a new request
        whose forward overwrites them, so each job is flushed: the worker waits
        for it before that forward (the bounded wait of the A/B)."""
        if not self._wb_rescue:
            return {}
        jobs: dict[int, TransferJob] = {}
        by_tag: dict[OffloadKey, list] = {}
        for tag, key, block_id, config_idx in self._wb_rescue:
            by_tag.setdefault(tag, []).append((key, block_id, config_idx))
        self._wb_rescue.clear()
        for tag, items in by_tag.items():
            store_output = self._prepare_store_run(
                [k for k, _, _ in items], tag, whole=False
            ) if hasattr(self.manager, "prepare_store_run") else None
            if store_output is None:
                self._wb_counts["rescue_refused"] += len(items)
                continue
            chunk_by_key = dict(
                zip(store_output.keys_to_store, store_output.store_spec.block_ids.tolist())
            )
            new = [
                (key, SimpleNamespace(block_id=block_id), config_idx, chunk_by_key[key])
                for key, block_id, config_idx in items
                if key in chunk_by_key
            ]
            if not new:
                continue
            job_id, job = self._band_job(new, store_output, now)
            self._wb_held[job_id] = []
            self._current_batch_jobs_to_flush.add(job_id)
            jobs[job_id] = job
        return jobs

    def _complete_write_back(self, job_id: int, keys: set[OffloadKey]) -> None:
        submitted = self._wb_submitted.pop(job_id, None)
        if submitted is not None:
            latency = time.monotonic() - submitted[0]
            self._wb_latency = 0.8 * self._wb_latency + 0.2 * latency
            self._wb_latency_max = max(self._wb_latency_max, latency)
            self._wb_counts["blocks_copied"] += submitted[1]
        self._wb_inflight.difference_update(keys)
        self._note_in_pool(keys)
        self._release_write_back(job_id)

    def _release_write_back(self, job_id: int) -> None:
        """Return a write-back job's blocks to the queue HEAD (still hashed), so
        the hold did not change their eviction order."""
        pool = self._gpu_block_pool
        for block in self._wb_held.pop(job_id, ()):
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                pool.free_block_queue.prepend_n([block])

    def _generate_job_id(self) -> int:
        job_id = self._job_counter
        self._job_counter += 1
        return job_id

    def _remove_pending_job(self, job_id: int, block_ids: list[int] | None) -> None:
        for bid in block_ids or ():
            pending = self._block_id_to_pending_jobs[bid]
            pending.remove(job_id)
            if not pending:
                del self._block_id_to_pending_jobs[bid]

    def _calc_num_offloadable_tokens(
        self, req_status: RequestOffloadState, num_computed_tokens: int
    ) -> int:
        num = min(num_computed_tokens, req_status.req.num_tokens)
        max_offload_tokens = req_status.max_offload_tokens
        if max_offload_tokens is not None:
            num = min(num, max_offload_tokens)
        if self.config.offload_prompt_only:
            num = min(num, req_status.req.num_prompt_tokens)
        return num

    def _lookup_results(
        self,
        keys: Iterable[OffloadKey],
        req_context: ReqContext,
        stop_at_miss: bool,
    ) -> Iterator[tuple[OffloadKey, LookupResult]]:
        """Pair each key with its lookup result. A manager that answers many
        keys in one call (``lookup_batch``, e.g. a remote shared pool) gets one
        call per scan instead of one per key; with ``stop_at_miss`` it may stop
        after the first MISS, as a prefix scan would."""
        lookup_batch = getattr(self.manager, "lookup_batch", None)
        if lookup_batch is None:
            for key in keys:
                yield key, self.manager.lookup(key, req_context)
            return
        keys = list(keys)
        yield from zip(keys, lookup_batch(keys, req_context, stop_at_miss))

    def _maximal_prefix_lookup(
        self,
        keys: Iterable[OffloadKey],
        req_context: ReqContext,
        req: Request,
        group_config: GroupOffloadConfig,
        start_chunk_idx: int,
    ) -> int | None:
        """Return the number of consecutive offloaded chunks from the start,
        or None if the backend deferred a lookup."""
        hit_count = 0
        defer_lookup = False
        for local_idx, (key, result) in enumerate(
            self._lookup_results(keys, req_context, stop_at_miss=True)
        ):
            if self._b70_counts is not None:
                self._b70_counts[result] += 1
            match result:
                case LookupResult.HIT:
                    self._events_tracker.record_lookup(
                        req,
                        group_config,
                        start_chunk_idx + local_idx,
                        key,
                    )
                    hit_count += 1
                case LookupResult.HIT_PENDING:
                    defer_lookup = True
                    hit_count += 1
                case LookupResult.RETRY:
                    # Don't break: keep scanning to let manager kick off
                    # async lookups (until a miss is detected).
                    defer_lookup = True
                case LookupResult.MISS:
                    break
        return hit_count if not defer_lookup else None

    def _sliding_window_lookup(
        self,
        keys: Sequence[OffloadKey],
        sliding_window_size: int,
        req_context: ReqContext,
        initial_window_size: int | None = None,
    ) -> int | None:
        """Return the end index (in `keys`) of the last run of
        `sliding_window_size` consecutive hits, scanning from the end.
        The first run may need a larger window for a partial rightmost chunk.
        Returns 0 on miss, None if the backend deferred a lookup."""
        defer_lookup = False
        pending_in_window = False
        consecutive_hits = 0
        required_window = initial_window_size or sliding_window_size
        # A batching manager answers the whole span in one call; any other
        # keeps the lazy per-key lookups (a lookup may start promotions, so a
        # scan that stops early must not look up keys it never reached).
        lookup_batch = getattr(self.manager, "lookup_batch", None)
        batch = (
            lookup_batch(list(keys), req_context, False)
            if lookup_batch is not None
            else None
        )
        for idx in range(len(keys) - 1, -1, -1):
            result = (
                batch[idx]
                if batch is not None
                else self.manager.lookup(keys[idx], req_context)
            )
            if self._b70_counts is not None:
                self._b70_counts[result] += 1
            match result:
                case LookupResult.HIT:
                    consecutive_hits += 1
                case LookupResult.HIT_PENDING:
                    # Block is in cache, just not readable yet — counts
                    # as hit for the consecutive streak. Don't break:
                    # keep scanning to let manager kick off async lookups.
                    pending_in_window = True
                    consecutive_hits += 1
                case LookupResult.RETRY:
                    # Block location uncertain — does not count as hit.
                    # Don't break: keep scanning to let manager kick off
                    # async lookups.
                    defer_lookup = True
                    consecutive_hits = 0
                    pending_in_window = False
                    required_window = sliding_window_size
                case LookupResult.MISS:
                    consecutive_hits = 0
                    # This gap rules out the incomplete window to its right.
                    pending_in_window = False
                    required_window = sliding_window_size
            if consecutive_hits == required_window:
                return (
                    None if defer_lookup or pending_in_window else idx + required_window
                )
        return None if defer_lookup or pending_in_window else consecutive_hits

    def _lookup_complete_chunks(
        self,
        req_status: RequestOffloadState,
        max_num_new_tokens: int | None = None,
    ) -> int | None:
        if not (self._b70_trace or self._b70_junction):
            return self._lookup_complete_chunks_impl(req_status, max_num_new_tokens)
        ctx: dict = {"groups": [], "junction": None}
        self._b70_ctx = ctx
        try:
            result = self._lookup_complete_chunks_impl(req_status, max_num_new_tokens)
        finally:
            self._b70_ctx = None
            self._b70_counts = None
        if self._b70_junction and result is not None:
            self._b70_maybe_set_junction(req_status, result, ctx)
        if self._b70_trace:
            self._b70_log_lookup(req_status, result, ctx)
        return result

    def _lookup_complete_chunks_impl(
        self,
        req_status: RequestOffloadState,
        max_num_new_tokens: int | None = None,
    ) -> int | None:
        """
        Find how many tokens beyond num_locally_computed_tokens can be loaded.

        Iterates full-attention groups first (prefix lookup), then sliding-window
        groups (suffix lookup). Each group may tighten max_hit_size_tokens, which
        can invalidate an earlier group's result, so the loop re-runs when that
        happens until num_hit_tokens converges.
        """
        num_computed_tokens = req_status.num_locally_computed_tokens
        req_status.full_attention_hit_boundary = 0
        max_hit_size_tokens: int = req_status.req.num_tokens
        if max_num_new_tokens is not None:
            max_hit_size_tokens = min(
                max_hit_size_tokens, num_computed_tokens + max_num_new_tokens
            )
        if req_status.max_load_tokens is not None:
            max_hit_size_tokens = min(
                max_hit_size_tokens,
                num_computed_tokens + req_status.max_load_tokens,
            )
        if self._sliding_window_groups:
            # the last prompt token has to be recomputed to get the logprobs
            # for sliding window attention, we must reduce by 1 to make sure
            # we still have a hit after reduction
            max_hit_size_tokens -= 1
            if self._mamba_align_size is not None:
                # Constrain hit-window to the mamba block size.
                max_hit_size_tokens = round_down(
                    max_hit_size_tokens, self._mamba_align_size
                )

        num_hit_tokens: int = 0
        defer_lookup = False
        lookup_groups = self._lookup_groups

        # Tracks which eagle groups have already popped their volatile trailing chunk
        # in the current convergence iteration. Reset when a non-eagle group
        # tightens the hit boundary, requiring a fresh pop.
        eagle_verified: set[int] = set()
        full_attention_confirmed = False
        full_attention_boundary_recorded = False
        while lookup_groups:
            looked_up_sliding_window: bool = False
            groups_iter = iter(lookup_groups)
            lookup_groups = ()
            for group_idx in groups_iter:
                group_config: GroupOffloadConfig = self.config.kv_group_configs[
                    group_idx
                ]
                group_state: RequestGroupState = req_status.group_states[group_idx]
                tokens_per_chunk = group_config.tokens_per_chunk
                offload_keys = group_state.offload_keys

                assert (
                    len(offload_keys) >= req_status.req.num_tokens // tokens_per_chunk
                )

                is_eagle_unverified = (
                    group_config.is_eagle_group and group_idx not in eagle_verified
                )

                # Constrain to a chunk-aligned boundary for this group.
                max_hit_size_tokens = min(
                    max_hit_size_tokens, len(offload_keys) * tokens_per_chunk
                )
                if req_status.max_load_tokens is not None:
                    max_hit_size_tokens = round_down(
                        max_hit_size_tokens, tokens_per_chunk
                    )
                if max_hit_size_tokens - num_computed_tokens < tokens_per_chunk:
                    # We can only load less than a chunk, so skip.
                    return 0

                sliding_window_size_in_chunks = (
                    group_config.sliding_window_size_in_chunks
                )
                if (
                    sliding_window_size_in_chunks is not None
                    and not full_attention_boundary_recorded
                ):
                    # Groups before the first sparse group are full attention.
                    full_attention_boundary_recorded = True
                    if full_attention_confirmed and not defer_lookup:
                        req_status.full_attention_hit_boundary = max_hit_size_tokens

                # For eagle groups, query one extra chunk that will be popped.
                # Widening applies to every group type: without it, the pop
                # below shrinks max_hit_size_tokens past what was queried,
                # which can push the confirmed boundary under a coarser
                # sibling group's chunk granularity and zero the whole
                # request's hit (issue #52735).
                query_max = max_hit_size_tokens
                if is_eagle_unverified:
                    query_max = min(
                        max_hit_size_tokens + tokens_per_chunk,
                        len(offload_keys) * tokens_per_chunk,
                    )

                num_chunks = min(cdiv(query_max, tokens_per_chunk), len(offload_keys))
                start_chunk_idx = num_computed_tokens // tokens_per_chunk
                offload_keys = offload_keys[start_chunk_idx:num_chunks]

                if self._b70_ctx is not None:
                    self._b70_counts = Counter()
                    b70_max_before = max_hit_size_tokens
                # end index (in the sliced offload_keys) up to which we
                # have backend-confirmed hits
                num_hit_chunks: int | None
                if sliding_window_size_in_chunks is None:
                    num_hit_chunks = self._maximal_prefix_lookup(
                        offload_keys,
                        req_status.req_context,
                        req_status.req,
                        group_config,
                        start_chunk_idx,
                    )
                else:
                    required_window = sliding_window_size_in_chunks
                    if is_eagle_unverified:
                        required_window += 1
                    candidate_end = min(
                        max_hit_size_tokens,
                        (num_chunks - int(is_eagle_unverified)) * tokens_per_chunk,
                    )
                    initial_window = group_config.load_window_size_in_chunks(
                        candidate_end
                    )
                    assert initial_window is not None
                    num_hit_chunks = self._sliding_window_lookup(
                        offload_keys,
                        required_window,
                        req_status.req_context,
                        initial_window + int(is_eagle_unverified),
                    )
                if self._b70_ctx is not None:
                    self._b70_ctx["groups"].append(
                        (
                            group_config.group_idx,
                            start_chunk_idx,
                            len(offload_keys),
                            num_hit_chunks,
                            self._b70_counts,
                            b70_max_before,
                        )
                    )
                if num_hit_chunks == 0:
                    return 0

                if num_hit_chunks is None:
                    defer_lookup = True
                else:
                    if is_eagle_unverified:
                        num_hit_chunks -= 1
                        eagle_verified.add(group_idx)

                    max_hit_size_tokens = min(
                        max_hit_size_tokens,
                        tokens_per_chunk * (start_chunk_idx + num_hit_chunks),
                    )
                    if sliding_window_size_in_chunks is None:
                        full_attention_confirmed = True

                new_num_hit_tokens = max_hit_size_tokens - num_computed_tokens
                if new_num_hit_tokens < tokens_per_chunk:
                    # We can only load less than a chunk, so skip.
                    return 0

                if new_num_hit_tokens < num_hit_tokens:
                    if not group_config.is_eagle_group:
                        eagle_verified.clear()
                    if defer_lookup:
                        # make another iteration on all groups to check
                        # if we still need to defer lookup
                        defer_lookup = False
                        lookup_groups = self._lookup_groups
                    elif looked_up_sliding_window and not lookup_groups:
                        # we need another iteration to confirm previously looked up
                        # sliding window works with the new_num_hit_tokens
                        lookup_groups = self._sliding_window_groups

                looked_up_sliding_window |= sliding_window_size_in_chunks is not None
                num_hit_tokens = new_num_hit_tokens

        if defer_lookup:
            logger.debug(
                "Offloading manager delayed request %s as backend requested",
                req_status.req.request_id,
            )
            return None

        # Possibly delay the request if any hit chunk is already being loaded.
        if self._chunks_being_loaded:
            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                tokens_per_chunk = group_config.tokens_per_chunk
                sliding_window_size_in_chunks = group_config.load_window_size_in_chunks(
                    num_computed_tokens + num_hit_tokens
                )
                offload_keys = group_state.offload_keys
                num_chunks = cdiv(
                    num_computed_tokens + num_hit_tokens, tokens_per_chunk
                )
                start_chunk_idx = num_computed_tokens // tokens_per_chunk
                offload_keys = offload_keys[start_chunk_idx:num_chunks]
                if sliding_window_size_in_chunks is not None:
                    offload_keys = offload_keys[-sliding_window_size_in_chunks:]
                if any(key in self._chunks_being_loaded for key in offload_keys):
                    # Hit chunks are being loaded, so delay the request.
                    logger.debug(
                        "Delaying request %s since some of its"
                        " chunks are already being loaded",
                        req_status.req.request_id,
                    )
                    return None

        logger.debug(
            "Request %s hit %s offloaded tokens after %s GPU hit tokens",
            req_status.req.request_id,
            num_hit_tokens,
            num_computed_tokens,
        )

        return num_hit_tokens

    def _make_boundary_key(
        self,
        request: Request,
        group_idx: int,
        boundary_tokens: int,
        req_context: ReqContext,
    ) -> OffloadKey:
        hash_idx = boundary_tokens // self.config.tokens_per_hash - 1
        key = make_offload_key(request.block_hashes[hash_idx], group_idx)
        req_context.set_offload_key_position(key, boundary_tokens)
        return key

    def _lookup(
        self,
        req_status: RequestOffloadState,
        max_num_new_tokens: int | None = None,
    ) -> int | None:
        complete_hit = self._lookup_complete_chunks(req_status, max_num_new_tokens)
        req_status.partial_tail_boundary = None
        if complete_hit is None or not self.config.supports_partial_tail:
            return complete_hit

        local_tokens = req_status.num_locally_computed_tokens
        complete_boundary = local_tokens + complete_hit
        tokens_per_hash = self.config.tokens_per_hash
        block_end = complete_boundary + self._partial_tail_block_size
        max_boundary = min(req_status.req.num_prompt_tokens - 1, block_end - 1)
        if max_num_new_tokens is not None:
            max_boundary = min(max_boundary, local_tokens + max_num_new_tokens)
        if req_status.max_load_tokens is not None:
            max_boundary = min(
                max_boundary,
                local_tokens + req_status.max_load_tokens,
            )
        max_boundary = round_down(max_boundary, tokens_per_hash)
        if max_boundary <= complete_boundary:
            return complete_hit

        pending = False
        for boundary in range(max_boundary, complete_boundary, -tokens_per_hash):
            boundary_pending = False
            boundary_missed = False
            boundary_keys = []
            for group_config in self.config.kv_group_configs:
                key = self._make_boundary_key(
                    req_status.req,
                    group_config.group_idx,
                    boundary,
                    req_status.req_context,
                )
                boundary_keys.append(key)
                result = self.manager.lookup(key, req_status.req_context)
                if result is LookupResult.MISS:
                    boundary_missed = True
                    break
                if result in (LookupResult.HIT_PENDING, LookupResult.RETRY):
                    boundary_pending = True

            pending |= boundary_pending
            if not boundary_missed and not boundary_pending:
                for group_config, key in zip(
                    self.config.kv_group_configs, boundary_keys
                ):
                    self._events_tracker.record_partial_lookup(
                        req_status.req, group_config, boundary, key
                    )
                req_status.partial_tail_boundary = boundary
                return boundary - local_tokens

        if pending and complete_hit == 0:
            return None
        return complete_hit

    def _maybe_register_shared_prefix_junction(
        self, req_status: RequestOffloadState, num_hit_tokens: int
    ) -> None:
        """Pin a shared-prefix junction where a sparse group cut the hit.

        Offload-tier counterpart of the junction ``get_computed_blocks`` pins
        for a local hit: under sparse retention the recompute then retains
        (and hands off for offload) the sliding-window/Mamba state at the
        boundary the full-attention groups proved, instead of storing it
        again at its replay boundary inside the part that differs.
        """
        if self.config.retention_interval is None:
            return
        boundary = req_status.full_attention_hit_boundary
        if boundary <= req_status.num_locally_computed_tokens + num_hit_tokens:
            return
        request = req_status.req
        # Never lower a junction the local prefix cache already set.
        if boundary > request.shared_prefix_boundary:
            request.shared_prefix_boundary = boundary

    def _maybe_set_prompt_boundary(
        self, req_status: RequestOffloadState, start: int
    ) -> None:
        """WRITE_BACK_PROMPT_BOUNDARY: make the last offload-chunk boundary
        strictly inside the prompt a junction, so the scheduler ends a chunk
        there and its recurrent state is handed off to the pool. Never moves a
        junction that is already set past the resume point."""
        if not (WRITE_BACK and WRITE_BACK_PROMPT_BOUNDARY and self._state_group_ids):
            return
        if self._chain_ref_group is None or self.config.retention_interval is None:
            return
        request = req_status.req
        tpc = self.config.kv_group_configs[self._chain_ref_group].tokens_per_chunk
        target = (request.num_prompt_tokens - 1) // tpc * tpc
        if any(c.is_eagle_group for c in self.config.kv_group_configs):
            # A restore at d also reads a drafter group's chunk after d (its
            # trailing chunk is dropped at lookup): end a chunk one chunk
            # earlier, so the state there has that chunk inside the prompt.
            target -= tpc
        if target <= start or request.shared_prefix_boundary > start:
            return
        request.shared_prefix_boundary = target
        self._wb_counts["prompt_boundary_set"] += 1

    def on_new_request(self, request: Request) -> None:
        """Called when a new request is added to the scheduler."""
        req_context = _create_req_context(request)
        offloading_context = self.manager.on_new_request(req_context)
        req_status = RequestOffloadState(
            config=self.config,
            req=request,
            req_context=req_context,
            offloading_context=offloading_context,
        )
        self._req_status[request.request_id] = req_status

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
        max_num_new_tokens: int | None = None,
    ) -> tuple[int | None, bool]:
        """Get number of new tokens that can be loaded beyond the
        num_computed_tokens.

        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally
                computed tokens for this request
            max_num_new_tokens (int | None): cap on the number of tokens that
                may be loaded beyond `num_computed_tokens`, if any.

        Returns:
            A tuple with the following elements:
                - The number of tokens that can be loaded beyond what is
                  already computed.
                  If None, it means that the connector needs more time to
                  determine the number of matched tokens, and the scheduler
                  should query for this request again later.
                - `True` if tokens will be loaded asynchronously
                  (between scheduler steps).

        """
        req_status = self._req_status[request.request_id]
        for group_state in req_status.group_states:
            group_state.block_ids.clear()

        if req_status.transfer_jobs:
            logger.debug(
                "Delaying request %s since it still has in-flight transfers",
                request.request_id,
            )
            return None, False

        req_status.update_offload_keys()
        req_status.num_locally_computed_tokens = num_computed_tokens

        num_hit_tokens: int | None
        if request.skip_reading_prefix_cache:
            num_hit_tokens = 0
        else:
            lookup_start = time.monotonic()
            num_hit_tokens = self._lookup(req_status, max_num_new_tokens)
            self._connector_stats.observe_histogram(
                _ConnectorMetricName.LOOKUP_SYNC_DELAY,
                time.monotonic() - lookup_start,
            )
            if num_hit_tokens is None:
                if req_status.deferred_lookup_start_time is None:
                    req_status.deferred_lookup_start_time = lookup_start
            else:
                self._maybe_observe_lookup_async_delay(req_status)
                self._maybe_register_shared_prefix_junction(req_status, num_hit_tokens)
                self._maybe_set_prompt_boundary(
                    req_status, num_computed_tokens + num_hit_tokens
                )
        req_status.update_num_hit_chunks(num_computed_tokens + (num_hit_tokens or 0))

        return num_hit_tokens, bool(num_hit_tokens)

    def get_external_cache_hit_sources(
        self,
        request: Request,
        num_external_tokens: int,
    ) -> dict[CacheHitSource, int]:
        """Split the accepted external hit by the tier each loaded key came from.

        Attention keys cover only their loaded token ranges; recurrent state
        covers the whole reused prefix. Overlaps report the outermost tier.
        Uncovered tokens, including omitted sliding-window history, are
        ``external_unspecified``.
        """
        sources: dict[CacheHitSource, int] = {}
        if num_external_tokens == 0:
            return sources
        req_status = self._req_status[request.request_id]
        start = req_status.num_locally_computed_tokens
        end = start + num_external_tokens

        events: list[tuple[int, int, CacheHitSource]] = [
            (end, 0, CacheHitSource.EXTERNAL_UNSPECIFIED)
        ]
        for lo, hi, key in req_status.load_key_ranges:
            lo, hi = max(start, lo), min(end, hi)
            if lo >= hi:
                continue
            source = CacheHitSource(
                self.manager.get_load_source(key, req_status.req_context)
            )
            events.extend(((lo, 1, source), (hi, -1, source)))

        # Sweep endpoints instead of rescanning every range for each interval.
        active: Counter[CacheHitSource] = Counter()
        position = start
        for boundary, delta, source in sorted(events):
            if boundary > position:
                tiers = [tier for tier, count in active.items() if count > 0]
                tier = (
                    CacheHitSource.outermost(tiers)
                    if tiers
                    else CacheHitSource.EXTERNAL_UNSPECIFIED
                )
                sources[tier] = sources.get(tier, 0) + boundary - position
            active[source] += delta
            position = boundary
        return sources

    def update_state_after_alloc(
        self, request: Request, blocks: KVCacheBlocks, num_external_tokens: int
    ):
        if num_external_tokens == 0:
            return

        req_status = self._req_status[request.request_id]

        num_locally_computed_tokens = req_status.num_locally_computed_tokens
        num_cached_tokens = num_locally_computed_tokens + num_external_tokens
        partial_tail_boundary = req_status.partial_tail_boundary
        if partial_tail_boundary is not None:
            assert partial_tail_boundary == num_cached_tokens

        keys_to_load: list[OffloadKey] = []
        req_status.load_key_ranges.clear()
        dst_block_ids: list[int] = []
        # per group
        group_sizes: list[int] = []
        block_indices: list[int] = []
        for group_config, group_state in zip(
            self.config.kv_group_configs,
            req_status.group_states,
        ):
            group_blocks = blocks.blocks[group_config.group_idx]
            self._current_batch_allocated_block_ids.update(
                block.block_id for block in group_blocks if block.block_id != 0
            )

            tokens_per_block = group_config.tokens_per_block
            tokens_per_chunk = group_config.tokens_per_chunk
            offload_keys = group_state.offload_keys
            num_gpu_blocks = cdiv(num_cached_tokens, tokens_per_block)

            assert len(group_blocks) >= num_gpu_blocks
            # ``load_start_gpu_block_idx``: the index in ``group_blocks`` where the
            # load region begins -- a slice bound, not a count of computed blocks.
            # Scan from the computed boundary, not 0: sparse groups (Mamba,
            # SWA) legitimately hold non-null unhashed blocks below it.
            # Skip nulls (sentinel / out-of-retention padding).
            first_fresh_gpu_block_idx = cdiv(
                num_locally_computed_tokens, tokens_per_block
            )
            load_start_gpu_block_idx = num_gpu_blocks
            for i in range(first_fresh_gpu_block_idx, num_gpu_blocks):
                block = group_blocks[i]
                if not block.is_null and block.block_hash is None:
                    load_start_gpu_block_idx = i
                    break

            assert num_locally_computed_tokens % tokens_per_block == 0
            num_pending_gpu_blocks = num_gpu_blocks - load_start_gpu_block_idx

            if group_config.sliding_window_size_in_chunks is not None:
                assert (
                    num_pending_gpu_blocks
                    <= group_config.sliding_window_size_in_chunks
                    * self.config.blocks_per_chunk
                    + 1
                )

            num_chunks = cdiv(num_cached_tokens, tokens_per_chunk)
            if num_pending_gpu_blocks:
                start_chunk_idx = (
                    load_start_gpu_block_idx // self.config.blocks_per_chunk
                )
                end_chunk_idx = num_chunks - (partial_tail_boundary is not None)
                assert len(offload_keys) >= end_chunk_idx
                group_keys_to_load = offload_keys[start_chunk_idx:end_chunk_idx]
                if partial_tail_boundary is not None:
                    group_keys_to_load.append(
                        self._make_boundary_key(
                            request,
                            group_config.group_idx,
                            partial_tail_boundary,
                            req_status.req_context,
                        )
                    )
                keys_to_load.extend(group_keys_to_load)
                if isinstance(group_config.kv_cache_spec, MambaSpec):
                    req_status.load_key_ranges.extend(
                        (num_locally_computed_tokens, num_cached_tokens, key)
                        for key in group_keys_to_load
                    )
                else:
                    for chunk_idx, key in enumerate(
                        group_keys_to_load, start_chunk_idx
                    ):
                        range_start = max(
                            load_start_gpu_block_idx * tokens_per_block,
                            chunk_idx * tokens_per_chunk,
                        )
                        range_end = min(
                            num_cached_tokens, (chunk_idx + 1) * tokens_per_chunk
                        )
                        req_status.load_key_ranges.append((range_start, range_end, key))

            dst_block_ids.extend(
                block.block_id
                for block in group_blocks[load_start_gpu_block_idx:num_gpu_blocks]
            )
            group_sizes.append(num_pending_gpu_blocks)
            block_indices.append(load_start_gpu_block_idx)

            # Skip prefix-hit chunks for block-level policy; for
            # request-level, next_stored_chunk_idx stays at 0 so all
            # chunks (including hits) are offloaded.
            if req_status.offloading_context.policy == OffloadPolicy.CHUNK_LEVEL:
                group_state.next_stored_chunk_idx = num_chunks

        src_spec = self.manager.prepare_load(keys_to_load, req_status.req_context)
        dst_spec = GPULoadStoreSpec(
            dst_block_ids, group_sizes=group_sizes, block_indices=block_indices
        )

        load_job_id = self._generate_job_id()
        self._current_batch_load_jobs[load_job_id] = TransferJob(
            req_id=request.request_id,
            src_spec=src_spec,
            dst_spec=dst_spec,
        )
        # a load can only be issued when no other jobs are pending.
        assert not req_status.transfer_jobs
        req_status.transfer_jobs.add(load_job_id)
        self._jobs[load_job_id] = TransferJobStatus(
            req_id=request.request_id,
            pending_count=self.config.num_workers,
            keys=set(keys_to_load),
            is_store=False,
        )
        if self._b70_trace:
            self._b70_job_submitted(request.request_id, False)
            _b70.log(
                "load req=%s local=%d external=%d keys=%s job=%d",
                request.request_id,
                num_locally_computed_tokens,
                num_external_tokens,
                dict(
                    Counter(
                        self._b70_kind(get_offload_group_idx(k)) for k in keys_to_load
                    )
                ),
                load_job_id,
            )

        if self._chunks_being_loaded is not None:
            self._chunks_being_loaded.update(keys_to_load)
        req_status.partial_tail_boundary = None

    def _update_req_states(self, scheduler_output: SchedulerOutput) -> None:
        """Update request states from the Scheduler's output."""
        # new_block_ids_end[req_id][i] = end of pre-existing block_ids for
        # the i-th sliding window group (before this step's extend).
        # Used to detect sliding window blocks that got re-allocated.
        new_block_ids_end: dict[str, tuple[int, ...]] = {}

        for req_id, new_block_id_groups, preempted in yield_req_data(scheduler_output):
            req_status = self._req_status[req_id]
            req_status.update_offload_keys()

            if preempted:
                for group_state in req_status.group_states:
                    group_state.block_ids.clear()

            if new_block_id_groups:
                if self._sliding_window_groups:
                    new_block_ids_end[req_id] = tuple(
                        len(req_status.group_states[grp_idx].block_ids)
                        for grp_idx in self._sliding_window_groups
                    )
                req_status.update_block_id_groups(new_block_id_groups)
                for group_config in self.config.kv_group_configs:
                    new_blocks = new_block_id_groups[group_config.group_idx]
                    for bid in new_blocks:
                        if bid != 0:
                            self._current_batch_allocated_block_ids.add(bid)

        for copy in scheduler_output.kv_cache_block_copies or ():
            self._current_batch_allocated_block_ids.add(copy.dst_block_id)

        # Zero out stale block_ids in sliding window groups' pending-store
        # positions. Only sliding window groups can have stale entries (blocks
        # freed by remove_skipped_blocks then reallocated). Only positions in
        # [next_stored_chunk_idx * bsf, end) need checking where end is the
        # pre-extend length: earlier positions were already offloaded, later
        # ones are fresh allocations from this step.
        if self._sliding_window_groups and self._current_batch_allocated_block_ids:
            blocks_per_chunk = self.config.blocks_per_chunk
            for req_id, req_status in self._req_status.items():
                ends = new_block_ids_end.get(req_id)
                for i, grp_idx in enumerate(self._sliding_window_groups):
                    group_state = req_status.group_states[grp_idx]
                    start = group_state.next_stored_chunk_idx * blocks_per_chunk
                    end = ends[i] if ends is not None else len(group_state.block_ids)
                    for j in range(start, end):
                        if (
                            group_state.block_ids[j]
                            in self._current_batch_allocated_block_ids
                        ):
                            group_state.block_ids[j] = 0

    def _build_aligned_boundary_store_jobs(
        self, handoffs: dict[str, list[tuple[int, int, int]]]
    ) -> dict[int, TransferJob]:
        store_jobs: dict[int, TransferJob] = {}
        if WRITE_BACK and not WRITE_BACK_MIRROR and not WRITE_BACK_HANDOFF:
            self._wb_counts["handoff_skipped"] += sum(len(e) for e in handoffs.values())
            return store_jobs
        num_groups = len(self.config.kv_group_configs)
        config_idx_by_group = {
            config.group_idx: idx
            for idx, config in enumerate(self.config.kv_group_configs)
        }
        for req_id, entries in handoffs.items():
            req_status = self._req_status.get(req_id)
            if req_status is None:
                continue
            req = req_status.req
            max_boundary = self._calc_num_offloadable_tokens(req_status, req.num_tokens)
            b70_stored: list[tuple[int, int, int]] = []
            handed: set[int] = set()
            for group_idx, block_id, boundary in entries:
                config_idx = config_idx_by_group.get(group_idx)
                if config_idx is None:
                    continue
                group_config = self.config.kv_group_configs[config_idx]
                if (
                    block_id == 0
                    or boundary > max_boundary
                    or boundary % group_config.tokens_per_chunk != 0
                ):
                    continue

                # No write-through gate: a boundary state is stored only
                # here (interior states carry no GPU hash under sparse
                # retention, so the band cannot offer them).
                handed.add(boundary)
                key = self._make_boundary_key(
                    req, group_idx, boundary, req_status.req_context
                )
                store_output = self._prepare_store([key], req_status.req_context)
                if store_output is None:
                    self._connector_stats.increase_counter(
                        _ConnectorMetricName.ALLOCATION_FAILURE
                    )
                    continue
                if WRITE_BACK and group_idx in self._state_group_ids:
                    self._note_state_stored(key)
                if not store_output.keys_to_store:
                    continue

                job_id = self._generate_job_id()
                req_status.transfer_jobs.add(job_id)
                self._block_id_to_pending_jobs.setdefault(block_id, set()).add(job_id)
                self._jobs[job_id] = TransferJobStatus(
                    req_id=req_id,
                    pending_count=self.config.num_workers,
                    keys={key},
                    is_store=True,
                    fenced_block_ids=[block_id],
                )
                if self._b70_trace:
                    self._b70_job_submitted(req_id, True)
                    b70_stored.append((group_idx, boundary, job_id))
                group_sizes = [0] * num_groups
                group_sizes[config_idx] = 1
                block_indices = [0] * num_groups
                block_indices[config_idx] = (
                    boundary // group_config.tokens_per_block - 1
                )
                store_jobs[job_id] = TransferJob(
                    req_id=req_id,
                    src_spec=GPULoadStoreSpec(
                        [block_id],
                        group_sizes=group_sizes,
                        block_indices=block_indices,
                    ),
                    dst_spec=store_output.store_spec,
                )
                self._events_tracker.record_store(
                    req,
                    group_config,
                    boundary // group_config.tokens_per_chunk - 1,
                    key,
                )
            if handed and not (self._write_through or req_status.write_through):
                store_jobs.update(
                    self._build_boundary_attention_jobs(req_status, handed, b70_stored)
                )
            if self._b70_trace and entries:
                b70_offered = Counter(
                    (self._b70_kind(g), b) for g, _, b in entries
                )
                b70_done = Counter((self._b70_kind(g), b) for g, b, _ in b70_stored)
                _b70.log(
                    "handoff req=%s prompt=%d max_boundary=%d junction=%s "
                    "offered=%s stored=%s jobs=%s",
                    req_id,
                    req.num_prompt_tokens,
                    max_boundary,
                    req.shared_prefix_boundary,
                    {f"{k}@{b}": n for (k, b), n in sorted(b70_offered.items())},
                    {f"{k}@{b}": n for (k, b), n in sorted(b70_done.items())},
                    [j for _, _, j in b70_stored],
                )
        return store_jobs

    def _build_boundary_attention_jobs(
        self,
        req_status: "RequestOffloadState",
        boundaries: set[int],
        stored: list[tuple[int, int, int]],
    ) -> dict[int, TransferJob]:
        """With compute-time stores off, store the attention blocks at each
        boundary whose state was just handed off, so the run holds a complete
        boundary at once (a state-only run is unrestorable and is the first
        thing evicted) and the band adds the rest of the prefix later."""
        store_jobs: dict[int, TransferJob] = {}
        num_groups = len(self.config.kv_group_configs)
        req = req_status.req
        for boundary in sorted(boundaries):
            members = []
            for config_idx, (group_config, group_state) in enumerate(
                zip(self.config.kv_group_configs, req_status.group_states)
            ):
                if isinstance(group_config.kv_cache_spec, MambaSpec):
                    continue
                tpc = group_config.tokens_per_chunk
                if boundary % tpc != 0:
                    continue
                # A sliding-window group stores the whole window a restore at
                # the boundary reads: its earlier chunks leave the GPU once the
                # decode passes them, before the band can copy the run.
                window = group_config.load_window_size_in_chunks(boundary) or 1
                for end in range(boundary, max(0, boundary - window * tpc), -tpc):
                    block_idx = end // group_config.tokens_per_block - 1
                    if not 0 <= block_idx < len(group_state.block_ids):
                        continue
                    block_id = group_state.block_ids[block_idx]
                    if block_id == 0:
                        continue
                    key = self._make_boundary_key(
                        req, group_config.group_idx, end, req_status.req_context
                    )
                    members.append((config_idx, group_config, block_id, block_idx, key))
            if not members:
                continue
            store_output = self._prepare_store(
                [m[4] for m in members], req_status.req_context
            )
            if store_output is None:
                self._wb_counts["boundary_set_refused"] += 1
                self._connector_stats.increase_counter(
                    _ConnectorMetricName.ALLOCATION_FAILURE
                )
                continue
            self._wb_counts["boundary_set_stored"] += 1
            chunk_by_key = dict(
                zip(store_output.keys_to_store, store_output.store_spec.block_ids.tolist())
            )
            for config_idx, group_config, block_id, block_idx, key in members:
                chunk = chunk_by_key.get(key)
                if chunk is None:
                    continue
                job_id = self._generate_job_id()
                req_status.transfer_jobs.add(job_id)
                self._block_id_to_pending_jobs.setdefault(block_id, set()).add(job_id)
                self._jobs[job_id] = TransferJobStatus(
                    req_id=req.request_id,
                    pending_count=self.config.num_workers,
                    keys={key},
                    is_store=True,
                    fenced_block_ids=[block_id],
                )
                if self._b70_trace:
                    self._b70_job_submitted(req.request_id, True)
                    stored.append((group_config.group_idx, boundary, job_id))
                group_sizes = [0] * num_groups
                group_sizes[config_idx] = 1
                block_indices = [0] * num_groups
                block_indices[config_idx] = block_idx
                store_jobs[job_id] = TransferJob(
                    req_id=req.request_id,
                    src_spec=GPULoadStoreSpec(
                        [block_id], group_sizes=group_sizes, block_indices=block_indices
                    ),
                    dst_spec=type(store_output.store_spec)([chunk]),
                )
                self._events_tracker.record_store(
                    req,
                    group_config,
                    (block_idx + 1) * group_config.tokens_per_block
                    // group_config.tokens_per_chunk - 1,
                    key,
                )
        return store_jobs

    def _build_partial_tail_store_jobs(
        self, scheduler_output: SchedulerOutput
    ) -> dict[int, TransferJob]:
        block_state = scheduler_output.kv_connector_block_state
        handoffs = block_state.boundary_state_offloads if block_state else None
        if not handoffs:
            return {}

        store_jobs = self._build_aligned_boundary_store_jobs(handoffs)
        if not self.config.supports_partial_tail:
            return store_jobs

        for req_id, entries in handoffs.items():
            entries = [
                entry
                for entry in entries
                if entry[2] % self._partial_tail_block_size != 0
            ]
            if not entries:
                continue
            req_status = self._req_status.get(req_id)
            assert req_status is not None
            boundaries = {boundary for _, _, boundary in entries}
            assert len(boundaries) == 1
            boundary = boundaries.pop()
            req = req_status.req
            group_states = {
                group.group_idx: state
                for group, state in zip(
                    self.config.kv_group_configs, req_status.group_states
                )
            }
            max_boundary = min(
                req.num_prompt_tokens,
                req_status.max_offload_tokens or req.num_prompt_tokens,
            )
            assert boundary > 0
            assert boundary % self.config.tokens_per_hash == 0
            assert boundary <= max_boundary

            cow_blocks = {group_idx: block_id for group_idx, block_id, _ in entries}
            assert self._cow_source_groups.issubset(cow_blocks)

            assert boundary % self._partial_tail_block_size != 0
            block_idx = boundary // self._partial_tail_block_size
            if any(
                group.group_idx not in self._cow_source_groups
                and block_idx >= len(group_states[group.group_idx].block_ids)
                for group in self.config.kv_group_configs
            ):
                continue
            keys = [
                self._make_boundary_key(
                    req, group.group_idx, boundary, req_status.req_context
                )
                for group in self.config.kv_group_configs
            ]
            block_ids = [
                cow_blocks[group.group_idx]
                if group.group_idx in self._cow_source_groups
                else group_states[group.group_idx].block_ids[block_idx]
                for group in self.config.kv_group_configs
            ]
            assert all(block_id != 0 for block_id in block_ids)

            # No write-through gate here: a partial-tail state exists only in
            # this hand-off (its key names a sub-block hash, which the band,
            # keyed by GPU block hashes, never offers), so it is stored now or
            # never; it costs one chunk per group.
            store_output = self._prepare_store(keys, req_status.req_context)
            if store_output is None:
                self._connector_stats.increase_counter(
                    _ConnectorMetricName.ALLOCATION_FAILURE
                )
                continue
            if not store_output.keys_to_store:
                continue

            for group_config, key in zip(self.config.kv_group_configs, keys):
                if key in store_output.keys_to_store:
                    self._events_tracker.record_partial_store(
                        req, group_config, boundary, key
                    )

            group_by_key = {key: idx for idx, key in enumerate(keys)}
            accepted_groups = [group_by_key[key] for key in store_output.keys_to_store]
            group_sizes = [0] * len(self.config.kv_group_configs)
            block_indices = [0] * len(self.config.kv_group_configs)
            for group_idx in accepted_groups:
                group_sizes[group_idx] = 1
                block_indices[group_idx] = block_idx
            source_blocks = [block_ids[group_idx] for group_idx in accepted_groups]

            job_id = self._generate_job_id()
            req_status.transfer_jobs.add(job_id)
            for block_id in source_blocks:
                self._block_id_to_pending_jobs.setdefault(block_id, set()).add(job_id)
            self._jobs[job_id] = TransferJobStatus(
                req_id=req_id,
                pending_count=self.config.num_workers,
                keys=set(store_output.keys_to_store),
                is_store=True,
                fenced_block_ids=source_blocks,
            )
            if self._b70_trace:
                self._b70_job_submitted(req_id, True)
                _b70.log(
                    "partial-tail-store req=%s boundary=%d keys=%d job=%d",
                    req_id,
                    boundary,
                    len(store_output.keys_to_store),
                    job_id,
                )
            store_jobs[job_id] = TransferJob(
                req_id=req_id,
                src_spec=GPULoadStoreSpec(
                    source_blocks,
                    group_sizes=group_sizes,
                    block_indices=block_indices,
                ),
                dst_spec=store_output.store_spec,
            )

        return store_jobs

    def _reachable_store_block_mask(
        self,
        group_config: GroupOffloadConfig,
        start_chunk_idx: int,
        end_chunk_idx: int,
        final_segment_end_chunk_idx: int | None,
        reachable_boundaries: tuple[int, ...],
    ) -> list[bool] | None:
        """Build the block mask for a range of candidate offload chunks."""
        blocks_per_chunk = self.config.blocks_per_chunk
        kv_cache_spec = group_config.kv_cache_spec
        if isinstance(kv_cache_spec, SlidingWindowSpec) and (
            kv_cache_spec.extra_retained_tokens
        ):
            # Offload restores all allocated history, including MTP re-prefill
            # tokens. Widen only the store mask, not the model's attention window.
            kv_cache_spec = replace(
                kv_cache_spec,
                sliding_window=(
                    kv_cache_spec.sliding_window + kv_cache_spec.extra_retained_tokens
                ),
                extra_retained_tokens=0,
            )
        return group_config.manager_cls.reachable_block_mask(
            start_block=start_chunk_idx * blocks_per_chunk,
            end_block=end_chunk_idx * blocks_per_chunk,
            alignment_tokens=self.config.alignment_tokens,
            kv_cache_spec=kv_cache_spec,
            use_eagle=group_config.is_eagle_group,
            retention_interval=self.config.retention_interval,
            reachable_boundaries=reachable_boundaries,
            dcp_world_size=self.config.dcp_world_size,
            final_segment_end_block=(
                final_segment_end_chunk_idx * blocks_per_chunk
                if final_segment_end_chunk_idx is not None
                else None
            ),
        )

    def _final_swa_alignment_blocks(
        self, group_config: GroupOffloadConfig
    ) -> int | None:
        """Return representable final-tail alignment for a non-EAGLE SWA group."""
        alignment_tokens = self.config.alignment_tokens
        if (
            alignment_tokens is None
            or group_config.is_eagle_group
            or not isinstance(group_config.kv_cache_spec, SlidingWindowSpec)
            or alignment_tokens % group_config.tokens_per_block != 0
        ):
            return None
        return alignment_tokens // group_config.tokens_per_block

    def _build_store_jobs(
        self,
        scheduler_output: SchedulerOutput,
    ) -> dict[int, TransferJob]:
        blocks_per_chunk = self.config.blocks_per_chunk
        store_jobs: dict[int, TransferJob] = {}
        for req_id in chain(
            scheduler_output.num_scheduled_tokens,
            scheduler_output.finished_req_ids or (),
        ):
            req_status = self._req_status.get(req_id)
            if req_status is None:
                continue
            req = req_status.req

            if req.status is RequestStatus.FINISHED_ABORTED:
                num_tokens_after_batch = req.num_computed_tokens
            elif req.is_finished():
                # Clamp to the GPU prefix cache's commit point. The final sampled
                # token's slot is never committed (under spec decode it holds a
                # rejected draft's KV), so a block ending there must not be stored.
                num_tokens_after_batch = max(req.num_prompt_tokens, req.num_tokens - 1)
            else:
                num_scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
                num_tokens_after_batch = req.num_computed_tokens + num_scheduled_tokens

            num_offloadable_tokens = self._calc_num_offloadable_tokens(
                req_status, num_tokens_after_batch
            )
            prompt_offloadable_tokens = self._calc_num_offloadable_tokens(
                req_status, req.num_prompt_tokens
            )

            # Filter out chunks skipped due to sliding window attention / SSM
            # or unreachable by the load path's alignment constraints.
            new_offload_keys: list[OffloadKey] = []
            group_store_ranges: list[tuple[int, int]] = []

            reachable_boundaries: tuple[int, ...] = ()
            if self.config.retention_interval is not None:
                reachable_boundaries = (req.num_prompt_tokens - 1,)
                if req.shared_prefix_boundary:
                    reachable_boundaries += (req.shared_prefix_boundary,)

            for group_config, group_state in zip(
                self.config.kv_group_configs, req_status.group_states
            ):
                num_chunks = req_status.storable_chunks(
                    group_config, group_state, num_offloadable_tokens
                )

                start_chunk_idx = group_state.next_stored_chunk_idx
                prompt_horizon_chunks = (
                    prompt_offloadable_tokens // group_config.tokens_per_chunk
                )
                final_swa_alignment_blocks = self._final_swa_alignment_blocks(
                    group_config
                )
                reconsider_final_swa_tail = (
                    req.is_finished()
                    and num_chunks != prompt_horizon_chunks
                    and self.config.retention_interval is None
                    and final_swa_alignment_blocks is not None
                )
                if (
                    self.config.retention_interval is not None
                    or group_config.is_eagle_group
                ):
                    store_horizon_chunks = None
                elif req.is_finished():
                    store_horizon_chunks = num_chunks
                elif num_chunks <= prompt_horizon_chunks:
                    store_horizon_chunks = prompt_horizon_chunks
                else:
                    # An active decode frontier is not a final request
                    # boundary. Only fixed alignment tails are reachable.
                    store_horizon_chunks = None
                if reconsider_final_swa_tail:
                    assert final_swa_alignment_blocks is not None
                    horizon_blocks = num_chunks * blocks_per_chunk
                    partial_segment_start_block = (
                        horizon_blocks - horizon_blocks % final_swa_alignment_blocks
                    )
                    partial_segment_start_chunk = (
                        partial_segment_start_block // blocks_per_chunk
                    )
                    start_chunk_idx = min(start_chunk_idx, partial_segment_start_chunk)
                group_store_ranges.append((start_chunk_idx, num_chunks))

                if group_config.requires_cow_source:
                    continue

                if num_chunks <= start_chunk_idx:
                    continue
                offload_keys = group_state.offload_keys[start_chunk_idx:num_chunks]
                # For each chunk, take the last corresponding GPU block. For
                # blocks_per_chunk=3 and GPU block IDs 1 5 6 7 2 4 9 3 8,
                # this selects GPU blocks 6 4 8.
                # A block_id of 0 means either a sliding window / SSM skip
                # or a stale entry that was zeroed out — skip it either way.
                offload_block_ids = group_state.block_ids[
                    start_chunk_idx * blocks_per_chunk
                    + blocks_per_chunk
                    - 1 : num_chunks * blocks_per_chunk : blocks_per_chunk
                ]
                assert len(offload_keys) == len(offload_block_ids)

                # Use reachable_block_mask to filter unreachable chunks
                # (SWA/Mamba sparsity + retention interval).
                # reachable_block_mask operates in KV-block coordinates,
                # so convert chunk indices to block indices.
                block_mask = self._reachable_store_block_mask(
                    group_config=group_config,
                    start_chunk_idx=start_chunk_idx,
                    end_chunk_idx=num_chunks,
                    final_segment_end_chunk_idx=store_horizon_chunks,
                    reachable_boundaries=reachable_boundaries,
                )

                prompt_horizon_block_mask: list[bool] | None = None
                if reconsider_final_swa_tail:
                    prompt_horizon_block_mask = self._reachable_store_block_mask(
                        group_config=group_config,
                        start_chunk_idx=start_chunk_idx,
                        end_chunk_idx=num_chunks,
                        final_segment_end_chunk_idx=prompt_horizon_chunks,
                        reachable_boundaries=reachable_boundaries,
                    )

                for key_idx, (offload_key, block_id) in enumerate(
                    zip(offload_keys, offload_block_ids)
                ):
                    if block_id == 0:
                        continue
                    abs_chunk_idx = start_chunk_idx + key_idx
                    if (
                        reconsider_final_swa_tail
                        and abs_chunk_idx < group_state.next_stored_chunk_idx
                        and (
                            prompt_horizon_block_mask is None
                            or any(
                                prompt_horizon_block_mask[
                                    key_idx * blocks_per_chunk + block_idx
                                ]
                                for block_idx in range(blocks_per_chunk)
                            )
                        )
                    ):
                        continue
                    # A chunk is reachable if any of its constituent
                    # blocks is reachable.
                    if block_mask is not None and not any(
                        block_mask[key_idx * blocks_per_chunk + b]
                        for b in range(blocks_per_chunk)
                    ):
                        continue
                    new_offload_keys.append(offload_key)

            if not new_offload_keys:
                req_status.advance_stored_idx(num_offloadable_tokens)
                continue

            if not (self._write_through or req_status.write_through):
                # Pool full: the band writes these when the GPU is about to
                # drop them.
                req_status.advance_stored_idx(num_offloadable_tokens)
                continue
            store_output = self._prepare_store(
                new_offload_keys, req_status.req_context
            )
            if store_output is None:
                self._connector_stats.increase_counter(
                    _ConnectorMetricName.ALLOCATION_FAILURE
                )
                logger.warning("Request %s: cannot store chunks", req_id)
                continue

            if not store_output.keys_to_store:
                if self._b70_guard:
                    pending = self._b70_pending_skipped(
                        req_status, new_offload_keys, ()
                    )
                    if pending:
                        self._b70_guard_hold(req_status, group_store_ranges, pending)
                        continue
                req_status.advance_stored_idx(num_offloadable_tokens)
                continue

            keys_to_store = set(store_output.keys_to_store)

            group_sizes: list[int] = []
            block_indices: list[int] = []
            src_block_ids: list[int] = []
            fenced_block_ids: list[int] = []
            deferred_fence_block_ids: list[int] = []
            for group_config, group_state, store_range in zip(
                self.config.kv_group_configs,
                req_status.group_states,
                group_store_ranges,
            ):
                is_sliding_window = (
                    group_config.sliding_window_size_in_chunks is not None
                )
                start_chunk_idx, num_chunks = store_range
                block_ids = group_state.block_ids
                num_group_blocks = 0
                start_gpu_block_idx: int | None = None
                for idx, offload_key in enumerate(
                    group_state.offload_keys[start_chunk_idx:num_chunks]
                ):
                    if offload_key not in keys_to_store:
                        continue

                    chunk_idx = start_chunk_idx + idx

                    self._events_tracker.record_store(
                        req, group_config, chunk_idx, offload_key
                    )

                    gpu_block_idx = chunk_idx * blocks_per_chunk
                    for i in range(blocks_per_chunk):
                        block_id = block_ids[gpu_block_idx + i]
                        if block_id == 0:
                            continue
                        if start_gpu_block_idx is None:
                            start_gpu_block_idx = gpu_block_idx + i
                        src_block_ids.append(block_id)
                        num_group_blocks += 1
                        if is_sliding_window:
                            fenced_block_ids.append(block_id)
                        else:
                            deferred_fence_block_ids.append(block_id)

                group_sizes.append(num_group_blocks)
                block_indices.append(start_gpu_block_idx or 0)
                group_state.next_stored_chunk_idx = max(
                    group_state.next_stored_chunk_idx, num_chunks
                )
            if self._b70_guard:
                pending = self._b70_pending_skipped(
                    req_status, new_offload_keys, keys_to_store
                )
                if pending:
                    self._b70_guard_hold(req_status, group_store_ranges, pending)

            src_spec = GPULoadStoreSpec(
                src_block_ids, group_sizes=group_sizes, block_indices=block_indices
            )
            dst_spec = store_output.store_spec

            job_id = self._generate_job_id()
            # a store can only be issued when no load is pending.
            if req_status.transfer_jobs:
                any_jid = next(iter(req_status.transfer_jobs))
                assert self._jobs[any_jid].is_store
            req_status.transfer_jobs.add(job_id)

            # Watch sliding window blocks as they may get evicted
            # before the request finishes
            for bid in fenced_block_ids:
                self._block_id_to_pending_jobs.setdefault(bid, set()).add(job_id)

            # the non-sliding window blocks will be watched only
            # when the request finishes
            self._jobs[job_id] = TransferJobStatus(
                req_id=req_id,
                pending_count=self.config.num_workers,
                keys=set(keys_to_store),
                is_store=True,
                deferred_fence_block_ids=deferred_fence_block_ids,
                fenced_block_ids=fenced_block_ids or None,
            )
            if self._b70_trace:
                self._b70_job_submitted(req_id, True)
                b70_kinds = Counter(
                    self._b70_kind(get_offload_group_idx(k)) for k in keys_to_store
                )
                _b70.log(
                    "store req=%s upto=%d prompt=%d reachable_boundaries=%s "
                    "offered=%d stored=%s job=%d finished=%s",
                    req_id,
                    num_offloadable_tokens,
                    req.num_prompt_tokens,
                    reachable_boundaries,
                    len(new_offload_keys),
                    dict(b70_kinds),
                    job_id,
                    req.is_finished(),
                )

            store_jobs[job_id] = TransferJob(
                req_id=req_id, src_spec=src_spec, dst_spec=dst_spec
            )

            logger.debug(
                "Request %s offloading %s chunks upto %d tokens (job %d)",
                req_id,
                len(keys_to_store),
                num_offloadable_tokens,
                job_id,
            )

            if req.is_finished():
                # Register non-sliding-window blocks for flush detection.
                for bid in deferred_fence_block_ids:
                    self._block_id_to_pending_jobs.setdefault(bid, set()).add(job_id)
                    if bid in self._current_batch_allocated_block_ids:
                        self._current_batch_jobs_to_flush.add(job_id)

        return store_jobs

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        self._update_req_states(scheduler_output)
        schedule_end_context = ScheduleEndContext(
            new_req_ids=[req.req_id for req in scheduler_output.scheduled_new_reqs],
            preempted_req_ids=scheduler_output.preempted_req_ids or (),
        )
        self.manager.on_schedule_end(schedule_end_context)

        # Flush jobs for preempted requests.
        for req_id in scheduler_output.preempted_req_ids or ():
            req_status = self._req_status.get(req_id)
            if req_status is None or not req_status.transfer_jobs:
                continue
            any_jid = next(iter(req_status.transfer_jobs))
            assert self._jobs[any_jid].is_store
            self._current_batch_jobs_to_flush.update(req_status.transfer_jobs)

        # Flush jobs that contain re-allocated blocks.
        if (
            self._block_id_to_pending_jobs
            and not self._block_id_to_pending_jobs.keys().isdisjoint(
                self._current_batch_allocated_block_ids
            )
        ):
            self._current_batch_jobs_to_flush.update(
                jid
                for bid in self._current_batch_allocated_block_ids
                if bid in self._block_id_to_pending_jobs
                for jid in self._block_id_to_pending_jobs[bid]
            )

        self._write_through = self._write_through_enabled()
        partial_store_jobs = self._build_partial_tail_store_jobs(scheduler_output)
        normal_store_jobs = self._build_store_jobs(scheduler_output)
        if self._flush_state == "ready":
            write_back_jobs = self._build_flush_jobs()
        elif self._flush_state is None or self._flush_state == "armed":
            write_back_jobs = self._build_write_back_jobs()
        else:
            write_back_jobs = {}
        meta = OffloadingConnectorMetadata(
            load_jobs=self._current_batch_load_jobs,
            store_jobs=partial_store_jobs | normal_store_jobs | write_back_jobs,
            jobs_to_flush=self._current_batch_jobs_to_flush,
        )

        # All prepare_store calls for finished requests have been issued.
        # Signal on_request_finished and clean up state where possible.
        for req_id in scheduler_output.finished_req_ids or ():
            req_status = self._req_status.get(req_id)
            if req_status is None:
                continue
            req_status.finished_signaled = True
            self._record_run(req_status)
            self.manager.on_request_finished(req_status.req_context)
            if not req_status.transfer_jobs:
                del self._req_status[req_id]
                if self._b70_trace:
                    self._b70_req_done(req_id)
        self._current_batch_load_jobs = {}
        self._current_batch_jobs_to_flush = set()
        self._current_batch_allocated_block_ids = set()
        if self._flush_state == "armed":
            self._flush_state = "ready"
        return meta

    def has_pending_push_work(self) -> bool:
        """Whether the engine must keep stepping.

        While True, build_connector_meta() and update_connector_output()
        continue to be called even when no requests are scheduled.
        """
        state = self._flush_state
        if state is not None:
            if state in ("armed", "ready"):
                return True
            if state == "running":
                if not self._flush_jobs:
                    self._flush_end("done")
                elif time.monotonic() - self._flush_t0 >= FLUSH_ON_STOP_SEC:
                    self._flush_end("expired")
                    return False
                else:
                    return True
            if self._flush_state == "expired":
                return False
        return bool(self._jobs) or self.manager.has_pending_work()

    # --- flush on stop ---

    def request_flush_on_stop(self) -> None:
        """The engine is stopping (SIGTERM): from the next step, write every
        conversation cached on the GPU to the pool, then let the engine exit."""
        if self._flush_state is not None:
            return
        if (
            not WRITE_BACK
            or FLUSH_ON_STOP_SEC <= 0
            or self._gpu_block_pool is None
            or self.config.blocks_per_chunk != 1
            or not getattr(self.manager, "available", True)
        ):
            return
        self._flush_state = "armed"
        self._flush_t0 = time.monotonic()
        logger.info(
            "KV flush on stop: writing the GPU's cached conversations to the "
            "pool (budget %.0fs, %d runs known)",
            FLUSH_ON_STOP_SEC,
            len(self._run_keys),
        )

    def _build_flush_jobs(self) -> dict[int, TransferJob]:
        """Every known run with free GPU blocks the pool lacks, as one whole
        tagged store each (what a restore of it reads, see _restore_chain).

        Runs are chosen newest first while their new chunks fit the rows the
        pool can admit without evicting a live run of this class (an
        exclusive pool evicts its oldest run, which would be the newest one
        flushed here). The chosen runs are stored oldest first, so the pool's
        recency order matches the GPU's, and copied newest first, so the
        newest are in the pool if the budget runs out."""
        self._flush_state = "running"
        now = time.monotonic()
        config_idx_by_group = self._config_idx_by_group
        admit_fn = getattr(self.manager, "pool_admit_rows", None)
        admit = admit_fn() if admit_fn is not None else None
        if not isinstance(admit, int):
            admit = None
        counts = self._flush_counts
        chosen: list[tuple[OffloadKey, list]] = []
        seen: set[OffloadKey] = set()
        rows = 0
        for tag in reversed(list(self._run_keys)):
            if tag not in self._run_keys:
                continue  # dropped by an earlier run's re-trim
            members = self._run_members(tag, config_idx_by_group)
            if not members:
                counts["runs_nothing_to_write"] += 1
                continue
            new_rows = sum(1 for m in members if m[0] not in seen)
            if admit is not None and rows + new_rows > admit:
                counts["runs_pool_full"] += 1
                continue
            rows += new_rows
            seen.update(m[0] for m in members)
            chosen.append((tag, members))
        jobs: list[tuple[int, TransferJob]] = []
        pool = self._gpu_block_pool
        for order, (tag, members) in enumerate(reversed(chosen)):
            keys = [m[0] for m in members]
            if hasattr(self.manager, "prepare_store_run"):
                store_output = self._prepare_store_run(keys, tag, whole=True)
            else:
                store_output = self._prepare_store(keys, None)
            if store_output is None:
                counts["runs_refused"] += 1
                continue
            chunk_by_key = dict(
                zip(store_output.keys_to_store, store_output.store_spec.block_ids.tolist())
            )
            new = []
            for key, block, config_idx, _num_tokens in members:
                chunk = chunk_by_key.get(key)
                if chunk is None:
                    self._note_in_pool([key])  # already there (joined the run)
                    continue
                new.append((key, block, config_idx, chunk))
            if not new:
                counts["runs_already_in_pool"] += 1
                continue
            self._wb_touching = True
            try:
                pool.touch([m[1] for m in new])
            finally:
                self._wb_touching = False
            job_id, job = self._band_job(new, store_output, now)
            self._wb_held[job_id] = [m[1] for m in new]
            self._flush_jobs[job_id] = (order, len(new))
            counts["runs_submitted"] += 1
            counts["blocks_submitted"] += len(new)
            counts["tokens_submitted"] += max(m[3] for m in members)
            jobs.append((job_id, job))
        # Newest first: the worker submits store jobs in this order.
        jobs.reverse()
        logger.info(
            "KV flush on stop: %d runs (%d blocks) submitted in %.2fs; admit "
            "rows %s, %d runs left out (pool full), %d refused",
            counts["runs_submitted"],
            counts["blocks_submitted"],
            time.monotonic() - now,
            admit,
            counts["runs_pool_full"],
            counts["runs_refused"],
        )
        return dict(jobs)

    def _flush_job_done(self, job_id: int) -> None:
        done = self._flush_jobs.pop(job_id, None)
        if done is not None:
            self._flush_counts["runs_written"] += 1
            self._flush_counts["blocks_written"] += done[1]

    def _flush_end(self, outcome: str) -> None:
        self._flush_state = outcome
        counts = self._flush_counts
        logger.info(
            "KV flush on stop %s after %.2fs: %d of %d runs written (%d of %d "
            "blocks, ~%d tokens submitted); %d runs left out (pool full), %d "
            "refused, %d with nothing to write",
            outcome,
            time.monotonic() - self._flush_t0,
            counts["runs_written"],
            counts["runs_submitted"],
            counts["blocks_written"],
            counts["blocks_submitted"],
            counts["tokens_submitted"],
            counts["runs_pool_full"],
            counts["runs_refused"],
            counts["runs_nothing_to_write"],
        )

    def update_connector_output(self, connector_output: KVConnectorOutput):
        """Update KVConnector state from worker-side connectors output.

        Args:
            connector_output (KVConnectorOutput): the worker-side
                connectors output.

        """
        meta = connector_output.kv_connector_worker_meta
        if not isinstance(meta, OffloadingWorkerMetadata):
            assert meta is None
            meta = OffloadingWorkerMetadata()
        if not meta.transfer_stats.is_empty():
            transfer_stats = OffloadingConnectorStats()
            if not meta.transfer_stats.load.is_empty():
                transfer_stats.increase_counter(
                    _TransferMetricName.LOAD_BYTES,
                    meta.transfer_stats.load.bytes,
                )
                transfer_stats.increase_counter(
                    _TransferMetricName.LOAD_TIME,
                    meta.transfer_stats.load.time,
                )
                for size in meta.transfer_stats.load.sizes:
                    transfer_stats.observe_histogram(
                        _TransferMetricName.LOAD_SIZE, size
                    )
            if not meta.transfer_stats.store.is_empty():
                transfer_stats.increase_counter(
                    _TransferMetricName.STORE_BYTES,
                    meta.transfer_stats.store.bytes,
                )
                transfer_stats.increase_counter(
                    _TransferMetricName.STORE_TIME,
                    meta.transfer_stats.store.time,
                )
                for size in meta.transfer_stats.store.sizes:
                    transfer_stats.observe_histogram(
                        _TransferMetricName.STORE_SIZE, size
                    )
            self._connector_stats.aggregate(transfer_stats)

        for job_id, count in meta.completed_jobs.items():
            assert count > 0
            if job_id < self._stale_job_threshold:
                logger.debug(
                    "Skipping stale completed job %d (pre-reset counter: %d)",
                    job_id,
                    self._stale_job_threshold,
                )
                continue
            job_status = self._jobs[job_id]
            job_status.pending_count -= count
            if job_status.pending_count > 0:
                continue
            assert job_status.pending_count == 0
            if self._b70_trace:
                self._b70_job_completed(job_status)

            if job_status.req_id == _WRITE_BACK_REQ:
                self.manager.complete_store(job_status.keys, None)
                self._complete_write_back(job_id, job_status.keys)
                if self._flush_jobs:
                    self._flush_job_done(job_id)
                del self._jobs[job_id]
                continue

            req_status = self._req_status[job_status.req_id]
            if job_status.is_store:
                self.manager.complete_store(job_status.keys, req_status.req_context)
                if WRITE_BACK:
                    self._note_in_pool(job_status.keys)
                    self._note_states(job_status.keys)
            else:
                self.manager.complete_load(job_status.keys, req_status.req_context)
                if WRITE_BACK:
                    # Exclusive pool: the run is now oldest there, but the
                    # states it holds are what a restore at that boundary needs.
                    self._note_states(job_status.keys)
                if self._chunks_being_loaded:
                    self._chunks_being_loaded.difference_update(job_status.keys)
            if self._block_id_to_pending_jobs:
                # Sliding window blocks are tracked from store creation
                # and must be cleaned up unconditionally.
                self._remove_pending_job(job_id, job_status.fenced_block_ids)
                # Non-sliding-window blocks are only tracked after
                # request_finished, so only clean up for finished requests.
                if req_status.req.is_finished():
                    self._remove_pending_job(
                        job_id, job_status.deferred_fence_block_ids
                    )

            del self._jobs[job_id]
            req_status.transfer_jobs.remove(job_id)
            if req_status.finished_signaled and not req_status.transfer_jobs:
                del self._req_status[job_status.req_id]
                if self._b70_trace:
                    self._b70_req_done(job_status.req_id)

    def get_stats(self) -> OffloadingConnectorStats | None:
        stats: OffloadingConnectorStats | None = None
        if not self._connector_stats.is_empty():
            stats = self._connector_stats
            self._connector_stats = OffloadingConnectorStats()

        manager_stats = self.manager.get_stats()
        if manager_stats is not None:
            if stats is None:
                stats = manager_stats
            else:
                stats.aggregate(manager_stats)

        return stats

    def request_finished(
        self,
        request: Request,
    ) -> tuple[bool, dict[str, Any] | None]:
        """Called when a request has finished, before its blocks are freed.

        Returns:
            True if the request is being saved/sent asynchronously and blocks
            should not be freed until the request_id is returned from
            get_finished().
            Optional KVTransferParams to be included in the request outputs
            returned by the engine.

        """
        req_status = self._req_status.get(request.request_id)

        if req_status is None:
            # Untracked request (offloading never started): no in-flight jobs,
            # nothing was deferred, so finalize immediately.
            req_context = _create_req_context(request)
            self.manager.on_new_request(req_context)
            self.manager.on_request_finished(req_context)
            return False, None

        self._maybe_observe_lookup_async_delay(req_status)

        # Update offload keys with final block hash so _build_store_jobs can
        # create store jobs for the last block(s) on the next schedule step.
        req_status.update_offload_keys()
        if WRITE_BACK and self._gpu_block_pool is not None:
            self._pin_finished(req_status)

        # Keep req_status alive: _build_store_jobs will process finished_req_ids
        # on the next step and handle cleanup after creating store jobs.
        # Register deferred fences so future block reuse triggers a flush via
        # _block_id_to_pending_jobs.
        for job_id in req_status.transfer_jobs:
            job_status = self._jobs[job_id]
            for bid in job_status.deferred_fence_block_ids or ():
                self._block_id_to_pending_jobs.setdefault(bid, set()).add(job_id)

        return False, None

    def take_events(self) -> Iterable[KVCacheEvent]:
        """Drain pending KV cache events.

        Complete metadata is available only when self-describing KV events
        are enabled, and only for full-attention groups. Other shapes retain
        the previous placeholder payload so consumers can ignore them.

        Yields:
            ``BlockStored`` or ``BlockRemoved`` events corresponding to
            the underlying :class:`OffloadingEvent` stream.

        """
        yield from self._events_tracker.take_events(self.manager.take_events())

    def reset_cache(self) -> None:
        """Reset the offloading manager cache, evicting all stored chunks."""
        # reset_cache cannot be called in the middle of a schedule step
        assert not self._current_batch_load_jobs
        assert not self._current_batch_jobs_to_flush
        assert not self._current_batch_allocated_block_ids

        # Flush all in-flight jobs
        self._current_batch_jobs_to_flush.update(self._jobs.keys())

        for req_id, status in list(self._req_status.items()):
            if status.req.is_finished():
                if not status.finished_signaled:
                    self._record_run(status)
                    self.manager.on_request_finished(status.req_context)
                del self._req_status[req_id]

        # Reset offloading manager cache
        self.manager.reset_cache()

        # Reset store progress so active requests re-offload from chunk 0.
        for status in self._req_status.values():
            for group_state in status.group_states:
                group_state.next_stored_chunk_idx = 0
            status.transfer_jobs.clear()
            status.partial_tail_boundary = None

        # Discard jobs and save job_counter to be able to discard worker responses
        self._stale_job_threshold = self._job_counter
        for job_id in list(self._wb_held):
            self._release_write_back(job_id)
        for tag in list(self._wb_pinned):
            self._unpin(tag)
        self._wb_written.clear()
        self._wb_inflight.clear()
        self._wb_submitted.clear()
        self._wb_rescue.clear()
        self._flush_jobs.clear()
        self._pool_has.clear()
        self._state_sets.clear()
        self._jobs.clear()
        self._block_id_to_pending_jobs.clear()

        # The manager pool is empty; pending event payloads and announced
        # reference counts are stale.
        self._events_tracker.reset()

        # Note: _current_batch_jobs_to_flush is intentionally NOT cleared.
        # The load flush IDs collected above must be delivered to workers.
        if self._chunks_being_loaded is not None:
            self._chunks_being_loaded.clear()

    def shutdown(self) -> None:
        if self._b70_state_logger is not None:
            self._b70_state_logger.stop()
        self.manager.shutdown()
