# SPDX-License-Identifier: Apache-2.0
"""Write-back band of the offloading scheduler (exclusive RAM tier, 2026-10-23).

The band holds the hashed, chunk-aligned GPU blocks nearest the free-queue head,
stores the ones the pool lacks, and on completion returns them to the queue
HEAD with their hash, so the hold does not change eviction order.
"""

from unittest.mock import MagicMock

import pytest

import vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler as sched_mod
from tests.v1.kv_connector.unit.offloading_connector.test_scheduler import (
    _make_partial_tail_scheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    make_block_hash_with_group_id,
)
from vllm.v1.kv_offload.base import OffloadKey, PrepareStoreOutput
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.outputs import KVConnectorOutput


class FakeGPUPool:
    """The two BlockPool members the band uses, with BlockPool.touch semantics."""

    def __init__(self, blocks: list[KVCacheBlock]) -> None:
        self.free_block_queue = FreeKVCacheBlockQueue(blocks)

    def get_cached_block(self, block_hash, group_ids):
        node = self.free_block_queue.fake_free_list_head.next_free_block
        while node is not self.free_block_queue.fake_free_list_tail:
            if node.block_hash is not None and bytes(node.block_hash) == bytes(
                make_block_hash_with_group_id(block_hash, group_ids[0])
            ):
                return [node]
            node = node.next_free_block
        return None

    def touch(self, blocks):
        for block in blocks:
            if block.ref_cnt == 0:
                self.free_block_queue.remove(block)
            block.ref_cnt += 1


def _queue_ids(pool: FakeGPUPool) -> list[int]:
    out, node = [], pool.free_block_queue.fake_free_list_head.next_free_block
    while node is not pool.free_block_queue.fake_free_list_tail:
        out.append(node.block_id)
        node = node.next_free_block
    return out


@pytest.fixture
def band(monkeypatch):
    monkeypatch.setattr(sched_mod, "WRITE_BACK", True)
    monkeypatch.setattr(sched_mod, "WRITE_BACK_SCAN", 8)
    monkeypatch.setattr(sched_mod, "WRITE_BACK_MAX_HELD", 4)
    scheduler = _make_partial_tail_scheduler()
    chunk = scheduler.config.kv_group_configs[0].tokens_per_chunk
    group = scheduler.config.kv_group_configs[0].group_idx
    blocks = [KVCacheBlock(i) for i in range(1, 7)]
    # 1: unhashed, 2..4: hashed and chunk-aligned, 5: hashed but partial, 6: aligned
    for b, n in ((blocks[1], 1), (blocks[2], 2), (blocks[3], 3), (blocks[5], 4)):
        b.set_block_hash(make_block_hash_with_group_id(BlockHash(b"k%d" % b.block_id), group),
                         n * chunk)
    blocks[4].set_block_hash(make_block_hash_with_group_id(BlockHash(b"p5"), group), chunk + 1)
    pool = FakeGPUPool(blocks)
    scheduler.bind_gpu_block_pool(pool)
    scheduler.manager.pool_stats = MagicMock(return_value={"num_chunks": 100, "free": 0})
    present = {OffloadKey(bytes(blocks[3].block_hash))}

    def prepare_store(keys, req_context):
        new = [k for k in keys if k not in present]
        return PrepareStoreOutput(keys_to_store=new,
                                  store_spec=CPULoadStoreSpec(list(range(len(new)))),
                                  evicted_keys=[])

    scheduler.manager.prepare_store.side_effect = prepare_store
    return scheduler, pool, blocks


def test_band_holds_and_stores_only_what_the_pool_lacks(band):
    scheduler, pool, blocks = band
    jobs = scheduler._build_write_back_jobs()
    stored_blocks = sorted(j.src_spec.block_ids.tolist()[0] for j in jobs.values())
    assert stored_blocks == [2, 3, 6]          # 1 unhashed, 4 in pool, 5 partial
    for b in (blocks[1], blocks[2], blocks[5]):
        assert b.ref_cnt == 1                   # held: cannot be reallocated
    assert _queue_ids(pool) == [1, 4, 5]
    # Confirmed and in-flight blocks are not offered again.
    assert scheduler._build_write_back_jobs() == {}


def test_completion_releases_to_queue_head_with_hash(band):
    scheduler, pool, blocks = band
    jobs = scheduler._build_write_back_jobs()
    meta = OffloadingWorkerMetadata()
    for job_id in jobs:
        for _ in range(scheduler.config.num_workers):
            meta.mark_completed(job_id)
    scheduler._update_connector_output(KVConnectorOutput(kv_connector_worker_meta=meta))
    assert scheduler.manager.complete_store.call_count == 3
    assert not scheduler._wb_held and not scheduler._jobs
    assert all(b.ref_cnt == 0 for b in blocks)
    ids = _queue_ids(pool)
    assert set(ids[:3]) == {2, 3, 6} and ids[3:] == [1, 4, 5]
    assert blocks[1].block_hash is not None     # still a prefix-cache hit


def test_held_budget_caps_the_band(band, monkeypatch):
    scheduler, pool, blocks = band
    monkeypatch.setattr(sched_mod, "WRITE_BACK_MAX_HELD", 1)
    assert len(scheduler._build_write_back_jobs()) == 1
    assert scheduler._build_write_back_jobs() == {}


def test_write_through_stops_only_when_pool_is_full(band):
    scheduler, _, _ = band
    assert not scheduler._write_through_enabled()
    scheduler.manager.pool_stats.return_value = {"num_chunks": 100, "free": 50}
    assert scheduler._write_through_enabled()


def test_reset_releases_held_blocks(band):
    scheduler, pool, blocks = band
    assert scheduler._build_write_back_jobs()
    scheduler.reset_cache()
    assert not scheduler._wb_held
    assert all(b.ref_cnt == 0 for b in blocks)


def test_whole_run_is_written_root_first_under_its_tag(band):
    scheduler, pool, blocks = band
    # Blocks 2, 3, 6 form one finished conversation: root (2), then 3, tip (6).
    chain = [OffloadKey(bytes(blocks[i].block_hash)) for i in (1, 2, 5)]
    tag = chain[-1]
    scheduler._run_keys[tag] = chain
    for k in chain:
        scheduler._key_run[k] = tag
    calls = []
    scheduler.manager.prepare_store_run = MagicMock(side_effect=lambda keys, t: (
        calls.append((list(keys), t)) or PrepareStoreOutput(
            keys_to_store=list(keys), store_spec=CPULoadStoreSpec(list(range(len(keys)))),
            evicted_keys=[])))
    jobs = scheduler._build_write_back_jobs()
    # One tagged store for the whole run, root first, even though the queue
    # holds the tip nearest eviction.
    assert calls == [(chain, tag)]
    assert [j.src_spec.block_ids.tolist()[0] for j in jobs.values()] == [2, 3, 6]
