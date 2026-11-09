# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The finish-tail registry of --enable-mamba-finish-checkpoint: the lookup-only
index that keeps a finished request's tail blocks reachable from the tail key
after the request itself is gone (2026-11-09).

Written before the extraction, so the same file documents today's behaviour at
base 539da2bff6 and the behaviour after the move to
vllm/v1/core/answer_tail.py. The two names the move renames are resolved at run
time by _tail_maps and the FINISH_TAIL_EXTRA_KEY import; everything else is
reached the way the scheduler reaches it, through the manager and the pool."""

from collections import Counter
from collections.abc import Sequence

import pytest
import torch

from tests.v1.core.test_prefix_caching import make_kv_cache_manager, make_request
from vllm.utils.hashing import sha256
from vllm.utils.math_utils import round_down
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    BlockHashWithGroupId,
    generate_block_hash_extra_keys,
    hash_block_tokens,
    init_none_hash,
    make_block_hash_with_group_id,
)
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
)
try:  # The tail store is new; base 539da2bff6 has no answer_tail_store module.
    from vllm.v1.kv_offload.answer_tail_store import (
        AnswerTailHandoff,
        AnswerTailHit,
        NullAnswerTailStore,
    )
except ImportError:  # base 539da2bff6
    pytest.skip("answer_tail_store absent", allow_module_level=True)

try:  # The extraction moves the tail-key marker to vllm.v1.core.answer_tail.
    from vllm.v1.core.answer_tail import FINISH_TAIL_EXTRA_KEY
except ImportError:  # base 539da2bff6
    from vllm.v1.core.kv_cache_manager import (
        _FINISH_TAIL_EXTRA_KEY as FINISH_TAIL_EXTRA_KEY,
    )

BLOCK = 16
WINDOW = 20
NUM_BLOCKS = 64
FULL, MAMBA, SWA, RING = 0, 1, 2, 3
PROMPT = list(range(1, 41))
OUTPUT = list(range(1000, 1008))
# A turn finished by another engine: the prompt and answer a later request
# replays, and the follow-up that makes the replay longer than the turn that
# produced the tail.
REPLAYED = PROMPT + OUTPUT
FOLLOWUP = list(range(2000, 2008))
# 40 prompt + 8 answer tokens. The answer ends on a prefix match unit boundary,
# so the tail spans exactly one full unit and its untagged hash is that unit's
# key, which is what the tail-key marker has to keep apart. Only 47 of the 48
# tokens are ever committed, so the last block of each group stays uncached and
# is reachable from the tail key alone.


@pytest.fixture(autouse=True)
def _hash_seed():
    init_none_hash(sha256)


def _manager():
    """The Flash-Next group set: full attention, Mamba align, a sliding-window
    drafter and the QSA ring, with the prefix match unit equal to the block."""
    return make_kv_cache_manager(
        kv_cache_config=KVCacheConfig(
            num_blocks=NUM_BLOCKS,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    ["full"],
                    FullAttentionSpec(
                        block_size=BLOCK,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.float32,
                    ),
                ),
                KVCacheGroupSpec(
                    ["mamba"],
                    MambaSpec(
                        block_size=BLOCK,
                        shapes=((1, 1),),
                        dtypes=(torch.float32,),
                        mamba_cache_mode="align",
                    ),
                ),
                KVCacheGroupSpec(
                    ["swa"],
                    SlidingWindowSpec(
                        block_size=BLOCK,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.float32,
                        sliding_window=WINDOW,
                    ),
                ),
                KVCacheGroupSpec(
                    ["ring"],
                    CircularBufferSpec(
                        block_size=BLOCK,
                        num_kv_heads=1,
                        head_size=1,
                        dtype=torch.float32,
                    ),
                ),
            ],
        ),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=BLOCK,
        enable_mamba_finish_checkpoint=True,
    )


def _run_to_finish(manager, request, output, free_request=True):
    """Prefill, decode the answer, register the tail and free the request, as
    the scheduler does on a stop. Returns the blocks the request held, per
    group."""
    blocks, num_computed, _ = manager.get_computed_blocks(request)
    assert manager.allocate_slots(
        request, request.num_tokens - num_computed, num_computed, blocks
    ) is not None
    request.num_computed_tokens = request.num_tokens
    request.append_output_token_ids(output[0])
    for token in output[1:]:
        assert manager.allocate_slots(request, 1) is not None
        request.num_computed_tokens += 1
        request.append_output_token_ids(token)
    _, retained = manager.take_kv_cache_block_copies()
    manager.block_pool.free_blocks(retained)
    held = tuple(list(g) for g in manager.get_blocks(request.request_id).blocks)
    manager.cache_finished_request(request)
    if free_request:
        manager.free(request)
        manager.new_step_starts()
    return held


def _tail_maps(pool):
    """The registry's two maps: tail keys by block, and registered tail lengths
    by the parent hash a tail is chained on."""
    registry = getattr(pool, "answer_tail", None)
    if registry is not None:  # extracted, vllm.v1.core.answer_tail
        return registry.hashes_by_block, registry.lengths_by_parent
    return pool.finish_tail_hashes_by_block, pool.finish_tail_lengths_by_parent


def _tail_key(request):
    """The finished request's tail hash, its parent and its length, recomputed
    from the request's own tokens."""
    num_tokens = request.num_tokens
    start = round_down(num_tokens - 1, BLOCK)
    parent = None if not start else request.block_hashes[start // BLOCK - 1]
    extra_keys, _ = generate_block_hash_extra_keys(request, start, num_tokens, 0)
    tail = hash_block_tokens(
        sha256,
        parent,
        request.all_token_ids[start:num_tokens],
        (*(extra_keys or ()), FINISH_TAIL_EXTRA_KEY),
    )
    return tail, parent, num_tokens - start


def _registered_lengths(hashes):
    """The lengths the registry should be counting, aggregated from the tail
    entries it holds, so a missed or a double decrement shows up here."""
    lengths: dict[BlockHash | None, Counter] = {}
    for entries in hashes.values():
        for tail in entries.values():
            if tail is not None:
                lengths.setdefault(tail[0], Counter())[tail[1]] += 1
    return lengths


def test_cache_finished_request_registers_the_tail_key_of_every_group():
    manager = _manager()
    finished = make_request("a", PROMPT, BLOCK, sha256)
    held = _run_to_finish(manager, finished, OUTPUT)
    pool = manager.block_pool
    hashes, lengths = _tail_maps(pool)

    tail, parent, length = _tail_key(finished)
    assert length == BLOCK and parent == finished.block_hashes[1]
    tail_block_idx = (finished.num_tokens - 2) // BLOCK
    for group_id, group in enumerate(held):
        block = group[0] if group_id == RING else group[tail_block_idx]
        assert pool.get_cached_block(tail, [group_id]) == [block]
        assert make_block_hash_with_group_id(tail, group_id) in hashes[block.block_id]

    # One tail entry per group, counted once, and the counters match the
    # entries: the in-window and boundary registrations the same blocks get are
    # no-ops where the block already carries that key as its primary hash.
    assert len(hashes) == len(held)
    assert lengths[parent][length] == len(held)
    assert dict(lengths) == _registered_lengths(hashes)


def test_a_tail_key_can_never_equal_a_full_unit_key():
    manager = _manager()
    finished = make_request("a", PROMPT, BLOCK, sha256)
    held = _run_to_finish(manager, finished, OUTPUT)
    pool = manager.block_pool

    # The tail spans exactly one unit, so the untagged hash is that unit's key:
    # the marker is all that keeps a tail entry from being read as a full-unit
    # cache entry.
    unit_key = finished.block_hashes[finished.num_tokens // BLOCK - 1]
    tail, parent, _ = _tail_key(finished)
    start = round_down(finished.num_tokens - 1, BLOCK)
    untagged = hash_block_tokens(
        sha256, parent, finished.all_token_ids[start : finished.num_tokens], None
    )
    assert untagged == unit_key
    assert tail != unit_key

    # The tail key is lookup-only: it is never the block's primary hash, and the
    # unit key it would have collided with stays a miss.
    tail_block = held[FULL][-1]
    assert tail_block.block_hash is None
    assert pool.get_cached_block(tail, [FULL]) == [tail_block]
    assert pool.get_cached_block(unit_key, [FULL]) is None


def test_forgetting_a_tail_block_drops_its_entries_and_the_count_once():
    manager = _manager()
    finished = make_request("a", PROMPT, BLOCK, sha256)
    held = _run_to_finish(manager, finished, OUTPUT)
    pool = manager.block_pool
    hashes, lengths = _tail_maps(pool)
    tail, parent, length = _tail_key(finished)
    tail_block_idx = (finished.num_tokens - 2) // BLOCK
    assert lengths[parent][length] == len(held)

    ring = held[RING][0]
    pool.evict_blocks({ring.block_id})  # the eviction a connector reports
    assert ring.block_id not in hashes
    assert pool.get_cached_block(tail, [RING]) is None
    assert dict(lengths) == _registered_lengths(hashes)

    pool._maybe_evict_cached_block(held[MAMBA][tail_block_idx])  # get_new_blocks
    assert held[MAMBA][tail_block_idx].block_id not in hashes
    assert pool.get_cached_block(tail, [MAMBA]) is None
    assert dict(lengths) == _registered_lengths(hashes)

    # A copy-on-write move counts the moved entry once: the source holds only
    # the tail key, and the destination takes it over.
    src = held[FULL][tail_block_idx]
    dst = pool.get_new_blocks(1)[0]
    assert dst.block_hash is None
    pool.move_block_hashes(src, dst)
    assert dict(lengths) == _registered_lengths(hashes)
    assert pool.get_cached_block(tail, [FULL]) == [dst]


def test_a_freed_block_holding_only_a_tail_key_is_reused_last():
    manager = _manager()
    finished = make_request("a", PROMPT, BLOCK, sha256)
    held = _run_to_finish(manager, finished, OUTPUT, free_request=False)
    pool = manager.block_pool
    hashes, _ = _tail_maps(pool)

    # The ring group is not prefix cacheable: its block is reachable from the
    # tail key alone, which is what keeps it out of the front of the free queue.
    ring = held[RING][0]
    assert ring.block_hash is None and ring.block_id in hashes
    pool.free_blocks([ring])
    queue = pool.free_block_queue
    assert queue.fake_free_list_head.next_free_block is not ring
    assert ring.next_free_block is queue.fake_free_list_tail

    # A block with neither a hash nor a tail key still goes first (LIFO).
    plain = pool.get_new_blocks(1)[0]
    pool.free_blocks([plain])
    assert queue.fake_free_list_head.next_free_block is plain


def test_reset_prefix_cache_clears_the_tail_registry():
    manager = _manager()
    finished = make_request("a", PROMPT, BLOCK, sha256)
    held = _run_to_finish(manager, finished, OUTPUT)
    pool = manager.block_pool
    tail, _, _ = _tail_key(finished)
    assert pool.get_num_free_blocks() == pool.num_gpu_blocks - 1  # only the null

    assert pool.reset_prefix_cache() is True
    hashes, lengths = _tail_maps(pool)
    assert not hashes and not lengths
    for group_id in range(len(held)):
        assert pool.get_cached_block(tail, [group_id]) is None


def test_a_prefix_cache_reset_tells_the_tier_keyed_on_it():
    """A tier that keeps tails by key cannot outlive the cache those keys come
    from, and a reset drops the registry in one step rather than key by key, so
    the per-key forget never runs. Without one reset call the tier keeps
    bookkeeping for a cache it no longer shares a key space with: a queue of
    offers naming blocks the core has forgotten, and beliefs about rows whose
    keys nothing can name again. The hole is only visible through a bound tier,
    because the core's own store is the one that is told."""
    manager = _manager()
    store = _RecordingTailStore()
    manager.answer_tail.store = store
    finished = make_request("a", PROMPT, BLOCK, sha256)
    _run_to_finish(manager, finished, OUTPUT)
    assert store.resets == 0

    assert manager.block_pool.reset_prefix_cache() is True

    assert store.resets == 1


class _RecordingTailStore(NullAnswerTailStore):
    """The tier a finished turn's tails are offered to, keeping the keys it was
    told the GPU forgot, the rows it was offered per group, whether it was told
    the prefix cache it was keyed on is gone, the tails it holds by parent, and
    the answer it gives a probe."""

    def __init__(self) -> None:
        self.forgotten: list[BlockHashWithGroupId] = []
        self.offered: list[tuple[int, list[int]]] = []
        self.resets = 0
        self.probed: list[tuple[BlockHash, tuple[int, ...]]] = []
        self.asked: list[tuple[BlockHash | None, ...]] = []
        self.lengths: dict[BlockHash | None, tuple[int, ...]] = {}
        self.answer = False
        self.can_answer = False

    def answers_lookups(self) -> bool:
        return self.can_answer

    def tail_lengths(
        self, parents: Sequence[BlockHash | None]
    ) -> dict[BlockHash | None, tuple[int, ...]]:
        self.asked.append(tuple(parents))
        return {p: self.lengths[p] for p in parents if p in self.lengths}

    def store_tail(
        self, handoff: AnswerTailHandoff, group_id: int, block_ids: list[int]
    ) -> bool:
        self.offered.append((group_id, list(block_ids)))
        return False

    def forget_tail(self, tail_key: BlockHashWithGroupId) -> None:
        self.forgotten.append(tail_key)

    def lookup_tail(
        self, tail_key: BlockHash, group_ids: Sequence[int]
    ) -> AnswerTailHit | None:
        self.probed.append((tail_key, tuple(group_ids)))
        if not self.answer:
            return None
        return AnswerTailHit(
            tail_key=tail_key,
            resume_token=0,
            group_block_ids={},
            num_tokens=0,
            generation=7,
        )

    def reset(self) -> None:
        self.resets += 1


def test_a_tail_the_gpu_forgot_reaches_the_store_it_was_offered_to():
    """The store a tail was offered to reads the block the tail key named, and
    that block goes back to the free queue with the request that finished. A
    registry that forgot the key quietly would leave a queued store copying
    whoever holds the block next, so the forget reaches the tier the tails were
    offered to, and reaches it with tail keys only: the boundary and in-window
    entries are not what any tier was offered."""
    manager = _manager()
    store = _RecordingTailStore()
    manager.answer_tail.store = store
    finished = make_request("a", PROMPT, BLOCK, sha256)
    held = _run_to_finish(manager, finished, OUTPUT)
    pool = manager.block_pool
    tail, _, _ = _tail_key(finished)
    tail_block_idx = (finished.num_tokens - 2) // BLOCK

    pool._maybe_evict_cached_block(held[MAMBA][tail_block_idx])  # get_new_blocks

    assert pool.get_cached_block(tail, [MAMBA]) is None
    assert store.forgotten == [make_block_hash_with_group_id(tail, MAMBA)]


def test_a_tail_the_tier_holds_is_recorded_with_the_position_the_core_knows():
    """The tier answers only that rows are here; where a resume restarts from is
    the core's to know, because the core is the one that repeated the turn. So
    the record the request is left carrying has the core's position and the
    tier's cache number, and grants nothing: the connector that can see the rows
    is the one that decides, and it decides before anything is allocated.
    """
    manager = _manager()
    store = _RecordingTailStore()
    store.answer = True
    manager.answer_tail.store = store
    finished = make_request("a", PROMPT, BLOCK, sha256)
    tail, _, _ = _tail_key(finished)
    resuming = make_request("b", PROMPT, BLOCK, sha256)

    manager.answer_tail._probe_tier(resuming, tail, finished.num_tokens)

    assert store.probed == [(tail, tuple(range(manager.num_kv_cache_groups)))]
    record = resuming.tail_resume
    assert record is not None
    assert record.tail_key == tail
    assert record.resume_token == finished.num_tokens - 1
    assert record.num_tokens == finished.num_tokens
    assert record.generation == 7


def test_a_tail_the_tier_has_not_is_recorded_as_nothing():
    """A miss leaves no record, so the request resumes from the hit it already
    has rather than from a position nothing agreed to fill."""
    manager = _manager()
    store = _RecordingTailStore()
    manager.answer_tail.store = store
    finished = make_request("a", PROMPT, BLOCK, sha256)
    tail, _, _ = _tail_key(finished)
    resuming = make_request("b", PROMPT, BLOCK, sha256)

    manager.answer_tail._probe_tier(resuming, tail, finished.num_tokens)

    assert store.probed == [(tail, tuple(range(manager.num_kv_cache_groups)))]
    assert resuming.tail_resume is None


def test_a_record_of_a_tail_is_made_afresh_on_every_probe():
    """A lookup can be retried a step later, and a prefix cache reset in between
    takes the keys the record names with it. The probe runs again on every
    attempt, so clearing what it left the previous time is what stops a request
    resuming from rows that belonged to the cache that is gone."""
    manager = _manager()
    store = _RecordingTailStore()
    store.answer = True
    manager.answer_tail.store = store
    finished = make_request("a", PROMPT, BLOCK, sha256)
    tail, _, _ = _tail_key(finished)
    resuming = make_request("b", PROMPT, BLOCK, sha256)
    manager.answer_tail._probe_tier(resuming, tail, finished.num_tokens)
    assert resuming.tail_resume is not None

    store.answer = False
    manager.answer_tail._find_finish_tail_hit(resuming, 0, 0)

    assert resuming.tail_resume is None


def test_a_fresh_engine_asks_the_tier_for_the_tails_its_registry_cannot_know():
    """A brand-new engine has finished nothing, so the registry the probe walks
    is empty and the turn replayed from before the restart is never asked
    about. Its candidates are the resuming request's own hash boundaries, and
    only the tier can say which of them it holds a tail under."""
    manager = _manager()
    store = _RecordingTailStore()
    store.answer = True
    store.can_answer = True
    manager.answer_tail.store = store
    finished = make_request("a", REPLAYED, BLOCK, sha256)
    tail, parent, length = _tail_key(finished)
    store.lengths = {parent: (length,)}
    resuming = make_request("b", REPLAYED + FOLLOWUP, BLOCK, sha256)
    assert not _tail_maps(manager.block_pool)[1]

    manager.answer_tail._find_finish_tail_hit(resuming, resuming.num_tokens - 1, 0)

    assert store.probed == [(tail, tuple(range(manager.num_kv_cache_groups)))]
    assert store.asked and parent in store.asked[0]
    record = resuming.tail_resume
    assert record is not None
    assert record.tail_key == tail
    assert record.resume_token == finished.num_tokens - 1
    assert record.num_tokens == finished.num_tokens


def test_a_tier_that_cannot_answer_is_never_asked_about_those_candidates():
    """The empty registry is also the fast path that keeps a tierless
    deployment, and every deployment before any tail finishes, off the
    per-group cache-hit walk on every request: the candidates are only worth
    enumerating once a tier exists that could answer for them. A registry that
    does know a tail changes nothing: enumerating the boundaries a request
    repeats is a question for the tier, whatever this process finished."""
    manager = _manager()
    store = _RecordingTailStore()
    manager.answer_tail.store = store
    finished = make_request("a", REPLAYED, BLOCK, sha256)
    _, parent, length = _tail_key(finished)
    store.lengths = {parent: (length,)}
    resuming = make_request("b", REPLAYED + FOLLOWUP, BLOCK, sha256)

    manager.answer_tail._find_finish_tail_hit(resuming, resuming.num_tokens - 1, 0)

    assert store.asked == []
    assert store.probed == []
    assert resuming.tail_resume is None

    # Now the registry knows a tail, so the walk runs and the tier is asked
    # about what this process finished. The candidates stay the tier's to answer
    # and it still cannot, so the boundaries are never enumerated.
    _run_to_finish(manager, finished, OUTPUT)
    assert _tail_maps(manager.block_pool)[1]

    manager.answer_tail._find_finish_tail_hit(resuming, resuming.num_tokens - 1, 0)

    assert store.asked == []
    assert store.probed
    assert resuming.tail_resume is None


def test_the_store_is_offered_the_window_a_windowed_group_state_sits_in():
    """A tier keeps the rows it is handed rather than blocks it can reach on the
    GPU, so a windowed group is offered the blocks its window spans (ceil(20 /
    16) = 2 here) and every other group the one block its state at the tail is."""
    manager = _manager()
    store = _RecordingTailStore()
    manager.answer_tail.store = store
    finished = make_request("a", PROMPT, BLOCK, sha256)
    held = _run_to_finish(manager, finished, OUTPUT)
    tail_block_idx = (finished.num_tokens - 2) // BLOCK
    offered = dict(store.offered)

    assert offered[FULL] == [held[FULL][tail_block_idx].block_id]
    assert offered[MAMBA] == [held[MAMBA][tail_block_idx].block_id]
    assert offered[RING] == [held[RING][0].block_id]
    assert len(offered[SWA]) == (WINDOW + BLOCK - 1) // BLOCK
    first = tail_block_idx - len(offered[SWA]) + 1
    assert offered[SWA] == [
        block.block_id for block in held[SWA][first : tail_block_idx + 1]
    ]
