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
    from vllm.v1.kv_offload.answer_tail_store import NullAnswerTailStore
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


class _RecordingTailStore(NullAnswerTailStore):
    """The tier a finished turn's tails are offered to, keeping the keys it was
    told the GPU forgot. The store side of the seam is tested with the store
    that actually serves it; here only the reach of the registry's forget is."""

    def __init__(self) -> None:
        self.forgotten: list[BlockHashWithGroupId] = []

    def forget_tail(self, tail_key: BlockHashWithGroupId) -> None:
        self.forgotten.append(tail_key)


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
