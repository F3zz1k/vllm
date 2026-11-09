# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The answer tail: the end of a finished turn, kept reachable so the next turn
of the same conversation resumes there instead of re-prefilling it. Owned by
``--enable-mamba-finish-checkpoint``: the tail key, the registry that keeps the
finished request's tail blocks, the hit probe, the per-group assembly, the
expiry and the config validation.

Registry entries are lookup-only. A tail key is never a block's primary hash,
so the entries are neither offloaded nor published as KV events."""

from collections import Counter
from collections.abc import Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from vllm.logger import init_logger
from vllm.utils.math_utils import round_down
from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashWithGroupId,
    KVCacheBlock,
    generate_block_hash_extra_keys,
    hash_block_tokens,
    make_block_hash_with_group_id,
)
from vllm.v1.core.single_type_kv_cache_manager import (
    CircularBufferManager,
    FullAttentionManager,
    MambaManager,
    SingleTypeKVCacheManager,
    SlidingWindowManager,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
from vllm.v1.kv_offload.answer_tail_store import (
    AnswerTailHandoff,
    AnswerTailStore,
    NullAnswerTailStore,
    tail_rows_for_group,
)
from vllm.v1.request import Request, RequestStatus

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams
    from vllm.v1.core.block_pool import BlockPool

logger = init_logger(__name__)

# Marks a finished request's tail key so that it never equals a full-unit key.
FINISH_TAIL_EXTRA_KEY = "finish_tail"

# Per-request stop token table of the Mamba finish checkpoint in the worker.
MAX_FINISH_STOP_TOKEN_IDS = 128


def finish_stop_token_ids(params: "SamplingParams") -> list[int] | None:
    """The token ids check_stop ends a request on (eos_token_id is None under
    ignore_eos), or None if they do not fit the worker's table, in which case
    the request's final state is neither kept nor registered."""
    stop_token_ids = [] if params.eos_token_id is None else [params.eos_token_id]
    stop_token_ids.extend(params.stop_token_ids or ())
    if len(stop_token_ids) > MAX_FINISH_STOP_TOKEN_IDS:
        return None
    return stop_token_ids


class AnswerTailRegistry:
    """The lookup-only index of finished requests' tail blocks.

    A block is reachable from more than one key: its own prefix-cache hash plus
    the tail keys registered here. ``lengths_by_parent`` indexes the registered
    tails by the hash their key is chained on, so a hit probe only walks the
    tails that exist."""

    def __init__(self) -> None:
        self.hashes_by_block: dict[
            int, dict[BlockHashWithGroupId, tuple[BlockHash | None, int] | None]
        ] = {}
        # Registered tail lengths by the hash their key is chained on.
        self.lengths_by_parent: dict[BlockHash | None, Counter[int]] = {}
        # Optional hook for the tier a tail was offered to, called with each
        # tail key this registry drops. A queued store reads the block the key
        # named, and a block that loses its hashes can be handed to a new owner
        # and written, so the offer has to go with the key. ``AnswerTail``
        # installs it, because that is where the store lives.
        self.forget_hook: Callable[[BlockHashWithGroupId], None] | None = None
        # Optional hook for the tier a tail was offered to, called when the
        # whole prefix cache goes. A tier keyed on that cache cannot outlive it
        # knowing anything about it, and a reset does not run the per-key hook:
        # it drops the index in one step, so the tier has to be told once, in
        # the same step, or it keeps bookkeeping for a cache it no longer
        # shares a key space with.
        self.reset_hook: Callable[[], None] | None = None

    def register(
        self,
        pool: "BlockPool",
        block_hash_with_group_id: BlockHashWithGroupId,
        block: KVCacheBlock,
        tail: tuple[BlockHash | None, int] | None = None,
    ) -> None:
        """Make ``block`` reachable from a finished request's tail key.

        ``tail`` is the parent hash and length of a tail key, indexed so that
        lookups probe only the registered tails.
        """
        assert not block.is_null
        if block.block_hash == block_hash_with_group_id or (
            pool.cached_block_hash_to_block.contain(
                block_hash_with_group_id, block.block_id
            )
        ):
            return
        tails = self.hashes_by_block.setdefault(block.block_id, {})
        tails[block_hash_with_group_id] = tail
        if tail is not None:
            parent, length = tail
            lengths = self.lengths_by_parent.setdefault(parent, Counter())
            lengths[length] += 1
        pool.cached_block_hash_to_block.insert(block_hash_with_group_id, block)

    def forget(self, block: KVCacheBlock) -> list[BlockHashWithGroupId]:
        """Drop a block's tail keys as it loses its hashes, and return them for
        the caller to remove from the prefix cache.

        Only a key registered as a tail's own reaches ``forget_hook``: the
        boundary and in-window entries this registry also holds are not tails,
        and no tier was offered one under those keys.
        """
        tails = self.hashes_by_block.pop(block.block_id, None)
        if tails is None:
            return []
        hook = self.forget_hook
        for key, tail in tails.items():
            if tail is None:
                continue
            parent, length = tail
            lengths = self.lengths_by_parent[parent]
            lengths[length] -= 1
            if not lengths[length]:
                del lengths[length]
                if not lengths:
                    del self.lengths_by_parent[parent]
            if hook is not None:
                hook(key)
        return list(tails)

    def has(self, block: KVCacheBlock) -> bool:
        """Whether a block holds a tail key, whatever else it is cached under."""
        return block.block_id in self.hashes_by_block

    def clear(self) -> None:
        """Drop the registry together with the prefix cache it belongs to, and
        tell the tier the tails were offered to that its key space is gone."""
        self.hashes_by_block.clear()
        self.lengths_by_parent.clear()
        hook = self.reset_hook
        if hook is not None:
            hook()


class AnswerTail:
    """The finished turns' tails of one KV cache manager: their keys, the
    registry entries that keep their blocks reachable, the hit probe and the
    per-group assembly."""

    def __init__(
        self,
        coordinator: HybridKVCacheCoordinator,
        block_pool: "BlockPool",
        caching_hash_fn: Callable[[Any], bytes],
        num_kv_cache_groups: int,
        store: AnswerTailStore | None = None,
    ) -> None:
        self.coordinator = coordinator
        self.block_pool = block_pool
        self.caching_hash_fn = caching_hash_fn
        self.num_kv_cache_groups = num_kv_cache_groups
        # The RAM tier the tails are offered to. Nothing reads it back until a
        # real store exists, so the null one keeps the feature GPU-only.
        self.store: AnswerTailStore = (
            NullAnswerTailStore() if store is None else store
        )
        self.validate_groups(coordinator)
        # A tail the GPU forgets has to leave the tier it was offered to, and
        # this is the only object that knows both: the hook reads ``self.store``
        # when it runs, so the connector installing its own store later is
        # seen. The null store's ``forget_tail`` does nothing.
        block_pool.answer_tail.forget_hook = self._forget_from_store
        # A reset of the prefix cache drops the registry in one step, so the
        # per-key hook never runs: the tier is told once, here, and the only
        # thing it can honestly do is forget everything keyed on this cache.
        block_pool.answer_tail.reset_hook = self._reset_store

    def _forget_from_store(self, tail_key: BlockHashWithGroupId) -> None:
        """Tell the store that the GPU no longer holds a tail's key."""
        self.store.forget_tail(tail_key)

    def _reset_store(self) -> None:
        """Tell the store that the prefix cache it was keyed on is gone."""
        self.store.reset()

    @staticmethod
    def validate_groups(coordinator: HybridKVCacheCoordinator) -> None:
        """Reject the group sets whose tail cannot be assembled on a hit.

        A hit ends inside a block whatever the prefix match unit: its shared
        tail block is privatised by the partial-hit copy-on-write, which keys on
        the block offset, so partial hash hits are not needed. A circular buffer
        (the QSA ring) is copied from the finished request's ring on a hit
        (CircularBufferManager).
        """
        if not (
            isinstance(coordinator, HybridKVCacheCoordinator)
            and coordinator.full_attention_group_id is not None
            and all(
                type(manager)
                in (FullAttentionManager, MambaManager, SlidingWindowManager)
                if manager.kv_cache_spec.prefix_cacheable
                else type(manager) is CircularBufferManager
                for manager in coordinator.single_type_managers
            )
        ):
            raise ValueError(
                "--enable-mamba-finish-checkpoint supports only models whose "
                "KV cache groups are full attention, sliding window, Mamba "
                "and circular buffers."
            )

    @staticmethod
    def should_cache_finished(
        request: Request,
        new_token_ids: list[int],
        status_before_stop: RequestStatus,
        output_is_stale: bool,
    ) -> bool:
        """Whether this stop is one the worker also detects, and so registers
        the request's final state."""
        return bool(
            new_token_ids
            and status_before_stop == RequestStatus.RUNNING
            and not output_is_stale
            and not request.resumable
            and request.status
            in (
                RequestStatus.FINISHED_STOPPED,
                RequestStatus.FINISHED_LENGTH_CAPPED,
            )
        )

    def apply_hit(
        self,
        request: Request,
        computed_blocks: tuple[list[KVCacheBlock], ...],
        num_computed_tokens: int,
        num_uncached: int,
        max_cache_hit_length: int,
    ) -> tuple[tuple[list[KVCacheBlock], ...], int, int]:
        """Take a finished request's tail that ``request`` repeats, if the
        prefix-cache lookup stops short of it, and return the blocks to use,
        the tokens they cover and the updated uncached shared prefix."""
        tail_hit = self._find_finish_tail_hit(
            request, max_cache_hit_length, num_computed_tokens
        )
        if tail_hit is None:
            return computed_blocks, num_computed_tokens, num_uncached
        num_longest_hit = num_computed_tokens + num_uncached
        computed_blocks, num_new_computed_tokens = tail_hit
        num_uncached = max(0, num_longest_hit - num_new_computed_tokens)
        return computed_blocks, num_new_computed_tokens, num_uncached

    def cache_finished(self, request: Request) -> None:
        """Register the final state of a request that finished on its EOS, a
        stop token or its length limit.

        The worker keeps the state before the last token in block
        ``(num_tokens - 2) // block_size`` of every Mamba group. That block and
        the attention block holding the same position get a tail key covering
        the tokens after the last full prefix match unit, the last token
        included, and full attention also registers that unit as a lookup-only
        entry. A sliding-window group also registers the in-window blocks
        before the tail under their own keys, and a circular buffer (the ring of
        the open compression group) is kept under the tail key whole. The tier
        is offered the group's state at the tail, which is the window's worth of
        blocks for a windowed group and one block for the rest. Must run before
        the request's blocks are freed.
        """
        params = request.sampling_params
        if params is None or finish_stop_token_ids(params) is None:
            # The worker does not keep the final state of this request.
            return
        num_tokens = request.num_tokens
        pool = self.block_pool
        registry = pool.answer_tail
        hash_block_size = pool.hash_block_size
        start = round_down(num_tokens - 1, hash_block_size)
        tail_blocks: list[tuple[SingleTypeKVCacheManager, list[KVCacheBlock]]] = []
        for manager in self.coordinator.single_type_managers:
            blocks = manager.req_to_blocks.get(request.request_id, [])
            if isinstance(manager, CircularBufferManager):
                if len(blocks) != 1:
                    return
                tail_blocks.append((manager, blocks))
                continue
            if not manager.kv_cache_spec.prefix_cacheable:
                continue
            block_idx = (num_tokens - 2) // manager.block_size
            first = self._finish_window_start(manager, num_tokens - 1)
            if block_idx >= len(blocks) or any(
                block.is_null for block in blocks[first : block_idx + 1]
            ):
                return
            tail_blocks.append((manager, blocks))

        tail_hash = self._finish_tail_hash(request, start, num_tokens)
        tail = (self._finish_tail_parent(request, start), num_tokens - start)
        handoff = AnswerTailHandoff(
            request_id=request.request_id,
            tail_hash=tail_hash,
            parent=tail[0],
            length=tail[1],
            resume_token=num_tokens - 1,
        )
        for manager, blocks in tail_blocks:
            if isinstance(manager, CircularBufferManager):
                key = make_block_hash_with_group_id(
                    tail_hash, manager.kv_cache_group_id
                )
                registry.register(pool, key, blocks[0], tail)
                # The GPU-side registration stands on its own: a tier that
                # refuses the tail costs nothing here.
                self.store.store_tail(
                    handoff, manager.kv_cache_group_id, [blocks[0].block_id]
                )
                continue
            if isinstance(manager, SlidingWindowManager):
                block_idx = (num_tokens - 2) // manager.block_size
                for j in range(self._finish_window_start(manager, num_tokens - 1), block_idx):
                    registry.register(
                        pool,
                        make_block_hash_with_group_id(
                            request.block_hashes[
                                (j + 1) * manager.block_size // hash_block_size - 1
                            ],
                            manager.kv_cache_group_id,
                        ),
                        blocks[j],
                    )
            if isinstance(manager, FullAttentionManager) and start:
                registry.register(
                    pool,
                    make_block_hash_with_group_id(
                        request.block_hashes[start // hash_block_size - 1],
                        manager.kv_cache_group_id,
                    ),
                    blocks[(start - 1) // manager.block_size],
                )
            key = make_block_hash_with_group_id(tail_hash, manager.kv_cache_group_id)
            tail_idx = (num_tokens - 2) // manager.block_size
            tail_block = blocks[tail_idx]
            registry.register(pool, key, tail_block, tail)
            self.store.store_tail(
                handoff,
                manager.kv_cache_group_id,
                self._store_row_blocks(manager, blocks, tail_idx),
            )
        logger.debug(
            "Registered the final state of request %s at %d tokens.",
            request.request_id,
            num_tokens - 1,
        )

    def _finish_tail_parent(self, request: Request, start: int) -> BlockHash | None:
        """The hash a tail starting at ``start`` is chained on."""
        if not start:
            return None
        return request.block_hashes[start // self.block_pool.hash_block_size - 1]

    def _finish_tail_hash(self, request: Request, start: int, end: int) -> BlockHash:
        """Hash a finished request's tail ``all_token_ids[start:end]``."""
        extra_keys, _ = generate_block_hash_extra_keys(request, start, end, 0)
        return hash_block_tokens(
            self.caching_hash_fn,
            self._finish_tail_parent(request, start),
            request.all_token_ids[start:end],
            (*(extra_keys or ()), FINISH_TAIL_EXTRA_KEY),
        )

    def _find_finish_tail_hit(
        self, request: Request, max_cache_hit_length: int, num_computed_tokens: int
    ) -> tuple[tuple[list[KVCacheBlock], ...], int] | None:
        """Find a finished request's tail that ``request`` repeats.

        Probes the tails registered by ``cache_finished`` from the deepest
        prefix match unit full attention holds down to the normal hit, longest
        first, so a hit is always longer than the normal one. A hit needs every
        group and ends one token before the finished request's last token, which
        the key covers, so it skips the EAGLE drop.
        """
        coordinator = self.coordinator
        assert isinstance(coordinator, HybridKVCacheCoordinator)
        # A record of a tail is made against the cache that was here when it was
        # probed, and this probe runs again whenever the scheduler looks at the
        # request, so a record never outlives the step that made it: a reset in
        # between leaves the request holding a name for rows nothing can reach.
        request.tail_resume = None
        lengths_by_parent = self.block_pool.answer_tail.lengths_by_parent
        if not lengths_by_parent:
            return None
        hash_block_size = self.block_pool.hash_block_size
        anchor_blocks: list[list[KVCacheBlock]] = [
            [] for _ in range(self.num_kv_cache_groups)
        ]
        start = max_cache_hit_length
        for spec, group_ids, manager_cls, _ in coordinator.attention_groups:
            if not isinstance(spec, FullAttentionSpec):
                continue
            blocks, num_hit_tokens = manager_cls.find_longest_cache_hit(
                block_hashes=request.block_hashes,
                max_length=max_cache_hit_length,
                kv_cache_group_ids=group_ids,
                block_pool=self.block_pool,
                kv_cache_spec=spec,
                drop_eagle_block=False,
                alignment_tokens=hash_block_size,
            )
            start = min(start, num_hit_tokens)
            for group_id, group_blocks in zip(group_ids, blocks):
                anchor_blocks[group_id] = group_blocks

        # The tail covers the hit plus the finished request's last token.
        tails = sorted(
            (tail_start + length, tail_start)
            for tail_start in range(
                start,
                round_down(num_computed_tokens, hash_block_size) - 1,
                -hash_block_size,
            )
            for length in lengths_by_parent.get(
                self._finish_tail_parent(request, tail_start), ()
            )
            if num_computed_tokens + 1 < tail_start + length <= request.num_tokens
        )
        for end, tail_start in reversed(tails):
            tail_hash = self._finish_tail_hash(request, tail_start, end)
            hit_blocks = self._finish_tail_blocks(
                request, tail_hash, end, anchor_blocks
            )
            if hit_blocks is not None:
                logger.debug(
                    "Request %s hit a finished request's tail at %d tokens.",
                    request.request_id,
                    end - 1,
                )
                return hit_blocks, end - 1
            if request.tail_resume is None:
                # The GPU registry has come up short, so ask the tier the
                # connector keeps. Deepest first, and the first answer is the
                # one worth taking: a shallower tail the tier holds is one the
                # next probe reaches again once this request is retried.
                self._probe_tier(request, tail_hash, end)
        return None

    def _probe_tier(
        self, request: Request, tail_hash: BlockHash, end: int
    ) -> None:
        """Leave a tier's answer on the request for the connector to fence.

        The tier answers only "these rows are here": the position a resume
        restarts from is this side's to know, because this side is the one that
        repeated the turn. So the record carries the position and the tier's own
        cache number, and it grants nothing. Whether the rows can actually be
        read back is decided by the connector, which is the only one that can
        see them, and it decides before the scheduler allocates.
        """
        hit = self.store.lookup_tail(tail_hash, range(self.num_kv_cache_groups))
        if hit is None:
            return
        request.tail_resume = replace(hit, resume_token=end - 1, num_tokens=end)

    def _finish_tail_blocks(
        self,
        request: Request,
        tail_hash: BlockHash,
        end: int,
        anchor_blocks: list[list[KVCacheBlock]],
    ) -> tuple[list[KVCacheBlock], ...] | None:
        """Every group's blocks for a resume at ``end - 1`` from the tail
        ``tail_hash``, or None if any is gone: Mamba nulls then the tail, full
        attention its anchor then the tail, a sliding window nulls, the window
        before the tail and the tail, a circular buffer its kept ring."""
        coordinator = self.coordinator
        pool = self.block_pool
        hash_block_size = pool.hash_block_size
        hit_blocks: list[list[KVCacheBlock]] = [[] for _ in range(self.num_kv_cache_groups)]
        for spec, group_ids, _, _ in coordinator.attention_groups:
            tail_blocks = pool.get_cached_block(tail_hash, group_ids)
            if tail_blocks is None:
                return None
            for group_id, tail_block in zip(group_ids, tail_blocks):
                manager = coordinator.single_type_managers[group_id]
                num_blocks = (end - 2) // manager.block_size
                if isinstance(spec, MambaSpec):
                    prefix = [pool.null_block] * num_blocks
                elif isinstance(manager, SlidingWindowManager):
                    first = self._finish_window_start(manager, end - 1)
                    prefix = [pool.null_block] * first
                    for j in range(first, num_blocks):
                        found = pool.get_cached_block(
                            request.block_hashes[(j + 1) * manager.block_size // hash_block_size - 1],
                            [group_id],
                        )
                        if found is None:
                            return None
                        prefix.append(found[0])
                else:
                    prefix = anchor_blocks[group_id][:num_blocks]
                hit_blocks[group_id] = [*prefix, tail_block]
        for group_id, manager in enumerate(coordinator.single_type_managers):
            if isinstance(manager, CircularBufferManager):
                ring = pool.get_cached_block(tail_hash, [group_id])
                if ring is None:
                    return None
                hit_blocks[group_id] = list(ring)
        return tuple(hit_blocks)

    @staticmethod
    def _store_row_blocks(
        manager: SingleTypeKVCacheManager,
        blocks: list[KVCacheBlock],
        tail_idx: int,
    ) -> list[int]:
        """The blocks a group's tail rows read, oldest first: the window's
        worth of blocks for a windowed group, the tail block for the rest. A
        table shorter than the row count is offered as it stands, and the
        store refuses a tail that cannot name all of its rows rather than
        storing half of one."""
        rows = tail_rows_for_group(manager.kv_cache_spec, manager.block_size)
        first = tail_idx if rows is None else max(0, tail_idx - rows + 1)
        return [block.block_id for block in blocks[first : tail_idx + 1]]

    @staticmethod
    def _finish_window_start(manager: SingleTypeKVCacheManager, num_computed: int) -> int:
        """First block a resume at ``num_computed`` tokens reads besides the
        tail block: the window start for a sliding window, else the tail."""
        if isinstance(manager, SlidingWindowManager):
            return manager.get_num_skipped_tokens(num_computed) // manager.block_size
        return (num_computed - 1) // manager.block_size


def validate_mamba_finish_checkpoint(vllm_config: "VllmConfig") -> None:
    """Validate `enable_mamba_finish_checkpoint` against the rest of the config."""
    if not vllm_config.cache_config.enable_mamba_finish_checkpoint:
        return

    blockers: list[str] = []
    if not (
        vllm_config.cache_config.enable_prefix_caching
        and vllm_config.cache_config.mamba_cache_mode == "align"
    ):
        blockers.append("it needs prefix caching with mamba_cache_mode='align'")
    if vllm_config.model_config is not None and not vllm_config.model_config.is_hybrid:
        blockers.append("it needs a hybrid attention and Mamba model")
    if vllm_config.parallel_config.pipeline_parallel_size > 1:
        blockers.append("pipeline parallelism is not supported")
    if (
        vllm_config.parallel_config.decode_context_parallel_size > 1
        or vllm_config.parallel_config.prefill_context_parallel_size > 1
    ):
        blockers.append("context parallelism is not supported")
    if vllm_config.kv_transfer_config is not None and (
        vllm_config.kv_transfer_config.kv_connector
        not in (None, "OffloadingConnector", "SimpleCPUOffloadConnector")
    ):
        blockers.append("only the offloading KV connectors are supported")
    spec = vllm_config.speculative_config
    if (
        spec is not None
        and (spec.use_eagle() or spec.uses_draft_model())
        and (
            spec.draft_model_config.is_hybrid
            or spec.draft_model_config.is_attention_free
        )
    ):
        blockers.append("a draft model with Mamba layers is not supported")
    if vllm_config.cache_config.use_replayssm:
        blockers.append("ReplaySSM defers the Mamba state writes")
    if vllm_config.num_prefill_lookahead_tokens > 1:
        blockers.append(
            "multi-module MTP draft KV depends on tokens past the last one"
        )
    if (
        vllm_config.kv_events_config is not None
        and vllm_config.kv_events_config.enable_kv_cache_events
    ):
        blockers.append("its cache entries are not published as KV events")

    if blockers:
        raise ValueError(
            "--enable-mamba-finish-checkpoint is not supported in this "
            f"configuration: {'; '.join(blockers)}."
        )
