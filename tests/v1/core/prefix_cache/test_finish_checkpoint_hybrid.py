# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""--enable-mamba-finish-checkpoint on hybrid layouts beyond full attention and
Mamba: a sliding-window group (a drafter) and a circular buffer (the QSA ring
holding the open compression group), with the prefix match unit equal to the
Mamba block, as Qwen3.8-Flash-Next runs (2026-11-05)."""

import pytest
import torch

from tests.v1.core.test_prefix_caching import make_kv_cache_manager, make_request
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.single_type_kv_cache_manager import CircularBufferManager
from vllm.v1.kv_cache_interface import (
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
)

BLOCK = 16
NUM_BLOCKS = 64
WINDOW = 20


@pytest.fixture(autouse=True)
def _hash_seed():
    init_none_hash(sha256)


def _groups(sliding: bool, ring: bool) -> list[KVCacheGroupSpec]:
    groups = [
        KVCacheGroupSpec(["full"], FullAttentionSpec(
            block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32)),
        KVCacheGroupSpec(["mamba"], MambaSpec(
            block_size=BLOCK, shapes=((1, 1),), dtypes=(torch.float32,),
            mamba_cache_mode="align")),
    ]
    if sliding:
        groups.append(KVCacheGroupSpec(["swa"], SlidingWindowSpec(
            block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32,
            sliding_window=WINDOW)))
    if ring:
        groups.append(KVCacheGroupSpec(["ring"], CircularBufferSpec(
            block_size=BLOCK, num_kv_heads=1, head_size=1, dtype=torch.float32)))
    return groups


def _manager(hash_block_size: int, sliding: bool = True, ring: bool = True):
    return make_kv_cache_manager(
        kv_cache_config=KVCacheConfig(num_blocks=NUM_BLOCKS, kv_cache_tensors=[],
                                      kv_cache_groups=_groups(sliding, ring)),
        max_model_len=8192,
        enable_caching=True,
        hash_block_size=hash_block_size,
        enable_mamba_finish_checkpoint=True,
    )


def _run_to_finish(manager, request, output):
    """Prefill, decode to the last output token, register and free (as the
    scheduler does on a stop). Returns the request's blocks per group."""
    blocks, num_computed, _ = manager.get_computed_blocks(request)
    assert manager.allocate_slots(
        request, request.num_tokens - num_computed, num_computed, blocks) is not None
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
    manager.free(request)
    manager.new_step_starts()
    return held


def test_full_and_mamba_with_a_finer_unit_still_hit():
    # Control: the layout the 27B runs (unit finer than the Mamba block).
    manager = _manager(4, sliding=False, ring=False)
    prompt, output = list(range(1, 22)), list(range(1000, 1006))
    finished = make_request("a", prompt, 4, sha256)
    held = _run_to_finish(manager, finished, output)
    follow = make_request("b", prompt + output + [7] * 9, 4, sha256)
    blocks, num_computed, _ = manager.get_computed_blocks(follow)
    assert num_computed == finished.num_tokens - 1
    assert blocks.blocks[1][-1] is held[1][(finished.num_tokens - 2) // BLOCK]


@pytest.mark.parametrize("num_output", [6, 3, 9])
def test_flash_next_layout_hits_at_the_last_token_in_every_group(num_output):
    manager = _manager(BLOCK)
    prompt, output = list(range(1, 41)), list(range(1000, 1000 + num_output))
    finished = make_request("a", prompt, BLOCK, sha256)
    held = _run_to_finish(manager, finished, output)
    L = finished.num_tokens
    tail = (L - 2) // BLOCK
    full, mamba, swa, ring = held

    follow = make_request("b", prompt + output + [7] * 9, BLOCK, sha256)
    blocks, num_computed, _ = manager.get_computed_blocks(follow)
    assert num_computed == L - 1
    hit_full, hit_mamba, hit_swa, hit_ring = blocks.blocks
    assert hit_full == full[:tail + 1]
    assert hit_mamba[-1] is mamba[tail] and all(b.is_null for b in hit_mamba[:-1])
    # The window ending at L - 1, nulls before it.
    first = max(0, L - 1 - WINDOW + 1) // BLOCK
    assert all(b.is_null for b in hit_swa[:first])
    assert hit_swa[first:] == swa[first:tail + 1]
    # Mid-block, the open compression group comes from the finished ring; on
    # a block boundary (a multiple of the compression ratio) the plain prefix
    # hit serves it, with an empty group and a fresh ring.
    mid_block = (L - 1) % BLOCK != 0
    assert hit_ring == (ring if mid_block else [])

    assert manager.allocate_slots(
        follow, follow.num_tokens - num_computed, num_computed, blocks) is not None
    ring_mgr = manager.coordinator.single_type_managers[3]
    assert isinstance(ring_mgr, CircularBufferManager)
    new_ring = manager.get_blocks("b").blocks[3]
    assert len(new_ring) == 1 and new_ring[0] is not ring[0]
    copies, retained = manager.take_kv_cache_block_copies()
    copied = (ring[0].block_id, new_ring[0].block_id) in {
        (c.src_block_id, c.dst_block_id) for c in copies}
    assert copied == mid_block
    # A copied ring must not be zeroed afterwards (a fresh one is, when the
    # worker zeroes new blocks).
    zeroed = new_ring[0].block_id in ring_mgr.take_new_block_ids()
    assert zeroed == (not mid_block and ring_mgr.records_new_block_ids)
    manager.block_pool.free_blocks(retained)


@pytest.mark.parametrize("evict", ["ring", "swa"])
def test_a_missing_ring_or_window_block_falls_back(evict):
    manager = _manager(BLOCK)
    prompt, output = list(range(1, 41)), list(range(1000, 1006))
    finished = make_request("a", prompt, BLOCK, sha256)
    held = _run_to_finish(manager, finished, output)
    L = finished.num_tokens
    victim = held[3][0] if evict == "ring" else held[2][(L - 2) // BLOCK - 1]
    pool = manager.block_pool
    pool.free_block_queue.remove(victim)
    pool._maybe_evict_cached_block(victim)
    pool.free_block_queue.append(victim)
    follow = make_request("b", prompt + output + [7] * 9, BLOCK, sha256)
    _, num_computed, _ = manager.get_computed_blocks(follow)
    assert num_computed < L - 1                    # no stale ring, no hole
