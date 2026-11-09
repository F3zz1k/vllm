# SPDX-License-Identifier: Apache-2.0
"""Write-back band of the offloading scheduler (exclusive RAM tier, 2026-10-23).

The band holds the hashed, chunk-aligned GPU blocks nearest the free-queue head,
stores the ones the pool lacks, and on completion returns them to the queue
HEAD with their hash, so the hold does not change eviction order.

The last section is the answer tail's marker positive control: one payload per
group, written through the store job the connector builds and read back through
the restore job it builds, so a group whose bytes land in another group's row
is caught by the bytes and not by a count.
"""

import mmap
import os
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler as sched_mod
from tests.v1.kv_connector.unit.offloading_connector.test_scheduler import (
    _assert_nothing_half_built,
    _make_drafter_partial_tail_scheduler,
    _make_partial_tail_request,
    _make_partial_tail_scheduler,
    _make_windowed_tail_scheduler,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
    OffloadingConnectorScheduler,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    FreeKVCacheBlockQueue,
    KVCacheBlock,
    make_block_hash_with_group_id,
)
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.kv_offload.base import (
    GPULoadStoreSpec,
    LookupResult,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    get_offload_group_idx,
    make_offload_key,
)
from vllm.v1.kv_offload.answer_tail_store import (
    AnswerTailHandoff,
    AnswerTailHit,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.shared_pool import server as pool_server
from vllm.v1.outputs import KVConnectorOutput


class FakeGPUPool:
    """The two BlockPool members the band uses, with BlockPool.touch semantics."""

    def __init__(self, blocks: list[KVCacheBlock]) -> None:
        self.blocks = blocks
        self.free_block_queue = FreeKVCacheBlockQueue(blocks)

    def get_cached_block(self, block_hash, group_ids):
        # Like BlockPool: any hashed block, in use or free.
        want = bytes(make_block_hash_with_group_id(block_hash, group_ids[0]))
        for block in self.blocks:
            if block.block_hash is not None and bytes(block.block_hash) == want:
                return [block]
        return None

    def get_num_free_blocks(self):
        return self.free_block_queue.num_free_blocks

    def touch(self, blocks):
        for block in blocks:
            if block.ref_cnt == 0:
                self.free_block_queue.remove(block)
            block.ref_cnt += 1


def _src_blocks(jobs) -> list[int]:
    return [b for j in jobs.values() for b in j.src_spec.block_ids.tolist()]


def _queue_ids(pool: FakeGPUPool) -> list[int]:
    out, node = [], pool.free_block_queue.fake_free_list_head.next_free_block
    while node is not pool.free_block_queue.fake_free_list_tail:
        out.append(node.block_id)
        node = node.next_free_block
    return out


@pytest.fixture
def band(monkeypatch):
    return _make_band(monkeypatch)


@pytest.fixture
def band_with_tails(monkeypatch):
    """The band with `enable_partial_tail_offload` on. A tail key reaches the
    band's view only through the tail store route, which the flag gates, so a
    test that counts what the band does with a tail key has to run here: on the
    flag-off band that state cannot exist, and the flag's one counting predicate
    reports nothing, as it must."""
    return _make_band(monkeypatch, partial_tail_offload=True)


def _make_band(monkeypatch, partial_tail_offload: bool = False):
    monkeypatch.setattr(sched_mod, "WRITE_BACK", True)
    monkeypatch.setattr(sched_mod, "WRITE_BACK_SCAN", 8)
    monkeypatch.setattr(sched_mod, "WRITE_BACK_MAX_HELD", 4)
    monkeypatch.setattr(sched_mod, "WRITE_BACK_FREE_SHARE", 1.0)
    scheduler = _make_partial_tail_scheduler(
        partial_tail_offload=partial_tail_offload
    )
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
    # Band mechanics on lone blocks; the state-class rule has its own test.
    scheduler._has_state_groups = False
    scheduler.manager.pool_free_fraction = MagicMock(return_value=0.0)
    scheduler.manager.pool_evicted = MagicMock(return_value=([], False))
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
    stored_blocks = sorted(_src_blocks(jobs))
    assert stored_blocks == [2, 3, 6]          # 1 unhashed, 4 in pool, 5 partial
    for b in (blocks[1], blocks[2], blocks[5]):
        assert b.ref_cnt == 1                   # held: cannot be reallocated
    assert _queue_ids(pool) == [1, 4, 5]
    # Confirmed and in-flight blocks are not offered again.
    assert scheduler._build_write_back_jobs() == {}


def test_completion_returns_blocks_where_they_were_with_hash(band):
    scheduler, pool, blocks = band
    jobs = scheduler._build_write_back_jobs()
    meta = OffloadingWorkerMetadata()
    for job_id in jobs:
        for _ in range(scheduler.config.num_workers):
            meta.mark_completed(job_id)
    scheduler.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=meta))
    assert scheduler.manager.complete_store.call_count == 3
    assert not scheduler._wb_held and not scheduler._jobs
    assert all(b.ref_cnt == 0 for b in blocks)
    # Where they were: copying to RAM must not change what VRAM evicts (they
    # used to go to the head, so VRAM dropped sessions that fit in it).
    assert _queue_ids(pool) == [1, 2, 3, 4, 5, 6]
    assert blocks[1].block_hash is not None     # still a prefix-cache hit


def _complete_all(scheduler, jobs):
    meta = OffloadingWorkerMetadata()
    for job_id in jobs:
        for _ in range(scheduler.config.num_workers):
            meta.mark_completed(job_id)
    scheduler.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=meta))


def test_a_block_whose_predecessor_was_reused_returns_at_the_head(band):
    scheduler, pool, blocks = band
    jobs = scheduler._build_write_back_jobs()     # takes 2, 3, 6; queue 1 4 5
    pool.touch([blocks[0]])                       # block 1 is allocated meanwhile
    _complete_all(scheduler, jobs)
    assert _queue_ids(pool) == [2, 3, 4, 5, 6]    # 3 still after 2, 6 after 5


@pytest.mark.parametrize("has_state", [True, False])
def test_band_stores_are_held_only_without_recurrent_state(band, has_state):
    # With recurrent state the GPU frees a boundary's state right after writing
    # it, so the pool's copy is not a duplicate and must not be evicted first.
    scheduler, pool, blocks = band
    scheduler._has_state_groups = True
    _register_run(scheduler, blocks, (1, 2, 5))
    scheduler._has_state_groups = has_state
    scheduler.manager.prepare_store_run = MagicMock(
        side_effect=lambda keys, t, whole, held=False: _cpu_store_output(keys))
    scheduler._build_write_back_jobs()
    assert scheduler.manager.prepare_store_run.call_args.kwargs.get("held", False) is (not has_state)


def test_pool_evictions_reported_by_the_server_are_forgotten(band):
    # The band skips a run whose chunks it believes the pool holds. Another
    # engine's store evicted them: without the feed the GPU later dropped the
    # only copy, counted as copied.
    scheduler, pool, blocks = band
    scheduler._has_state_groups = True
    chain = _register_run(scheduler, blocks, (1, 2, 5))
    scheduler._note_in_pool(chain)
    scheduler.manager.prepare_store_run = MagicMock(
        side_effect=lambda keys, t, whole, held=False: _cpu_store_output(keys))
    assert scheduler._build_write_back_jobs() == {}
    scheduler.manager.pool_evicted = MagicMock(return_value=(chain, False))
    jobs = scheduler._build_write_back_jobs()
    assert not set(chain) & set(scheduler._pool_has)
    assert scheduler._wb_counts["pool_evicted_seen"] == 3
    assert sorted(_src_blocks(jobs)) == [2, 3, 6]


def test_an_eviction_feed_overflow_forgets_the_whole_view(band):
    scheduler, pool, blocks = band
    scheduler._note_in_pool([OffloadKey(bytes(blocks[3].block_hash))])
    scheduler.manager.pool_evicted = MagicMock(return_value=([], True))
    scheduler._build_write_back_jobs()
    assert scheduler._wb_counts["pool_view_reset"] == 1


def test_held_budget_caps_the_band(band, monkeypatch):
    scheduler, pool, blocks = band
    monkeypatch.setattr(sched_mod, "WRITE_BACK_MAX_HELD", 1)
    assert len(scheduler._build_write_back_jobs()) == 1
    assert scheduler._build_write_back_jobs() == {}


def test_band_alone_stores_nothing_at_compute_time(band):
    scheduler, _, _ = band
    scheduler.manager.pool_free_fraction.return_value = 0.5
    assert not scheduler._write_through_enabled()


def test_write_through_stops_only_when_pool_is_full(band, monkeypatch):
    monkeypatch.setattr(sched_mod, "WRITE_BACK_MIRROR", True)
    scheduler, _, _ = band
    assert not scheduler._write_through_enabled()
    scheduler.manager.pool_free_fraction.return_value = 0.5
    assert scheduler._write_through_enabled()
    # Unknown (pool unreachable): stores run and fail softly.
    scheduler.manager.pool_free_fraction.return_value = None
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
    scheduler.manager.prepare_store_run = MagicMock(side_effect=lambda keys, t, whole, held=False: (
        calls.append((list(keys), t, whole)) or PrepareStoreOutput(
            keys_to_store=list(keys), store_spec=CPULoadStoreSpec(list(range(len(keys)))),
            evicted_keys=[])))
    jobs = scheduler._build_write_back_jobs()
    # One tagged store for the whole run, root first, even though the queue
    # holds the tip nearest eviction; only a known run is marked whole.
    lone = OffloadKey(bytes(blocks[3].block_hash))
    assert calls == [(chain, tag, True), ([lone], lone, False)]
    # One copy job per run.
    assert [j.src_spec.block_ids.tolist() for j in jobs.values()] == [[2, 3, 6], [4]]


def test_a_block_of_no_known_run_is_a_run_of_its_own(band):
    scheduler, pool, blocks = band
    calls = []
    scheduler.manager.prepare_store_run = MagicMock(side_effect=lambda keys, t, whole, held=False: (
        calls.append((list(keys), t, whole)) or PrepareStoreOutput(
            keys_to_store=list(keys), store_spec=CPULoadStoreSpec([0]),
            evicted_keys=[])))
    scheduler._build_write_back_jobs()
    assert calls and all(keys == [tag] and not whole for keys, tag, whole in calls)


def _finished(scheduler, keys_by_group):
    return SimpleNamespace(
        group_states=[SimpleNamespace(offload_keys=k) for k in keys_by_group],
        req_context=ReqContext(req_id="req"),
    )


@pytest.mark.parametrize("write_back", [False, True])
def test_a_finished_request_merges_into_its_conversation(band, monkeypatch, write_back):
    scheduler, _, _ = band
    monkeypatch.setattr(sched_mod, "WRITE_BACK", write_back)
    num_groups = len(scheduler.config.kv_group_configs)
    attn_idx = next(
        idx for idx, cfg in enumerate(scheduler.config.kv_group_configs)
        if not isinstance(cfg.kv_cache_spec, MambaSpec)
    )
    keys_by_group = [
        [OffloadKey(b"g%d-%d" % (g, i)) for i in range(3)] for g in range(num_groups)
    ]
    scheduler.manager.merge_tag = MagicMock()
    # Every state group is in the pool at the last boundary.
    for g, cfg in enumerate(scheduler.config.kv_group_configs):
        if isinstance(cfg.kv_cache_spec, MambaSpec):
            scheduler._note_state_stored(
                OffloadKey(bytes(keys_by_group[attn_idx][-1])[:-4] + cfg.group_idx.to_bytes(4, "big")))
    scheduler._record_run(_finished(scheduler, keys_by_group))
    # The run is named after the conversation's last attention key, whatever
    # the band does.
    (ctx, tag), _ = scheduler.manager.merge_tag.call_args
    assert ctx.req_id == "req" and tag == keys_by_group[attn_idx][-1]
    assert (tag in scheduler._run_keys) == write_back


@pytest.mark.parametrize(
    "params, forced",
    [(None, False), ({}, False), ({"write_through": "yes"}, False),
     ({"write_through": True}, True)],
)
def test_write_through_param_is_read_per_request(params, forced):
    scheduler = _make_partial_tail_scheduler()
    _make_partial_tail_request(scheduler, params)
    assert scheduler._req_status["req"].write_through is forced


def _cpu_store_output(keys, req_context=None):
    keys = list(keys)
    return PrepareStoreOutput(keys_to_store=keys,
                              store_spec=CPULoadStoreSpec(list(range(len(keys)))),
                              evicted_keys=[])


def _takes_every_key(keys, tag, whole=False, held=False, tails=None):
    """A pool that allocates a row for every key the store offered, whatever
    else the call named (see ``prepare_store_run``)."""
    return _cpu_store_output(keys)


@pytest.mark.parametrize("forced", [False, True])
def test_a_boundary_state_is_stored_while_the_gate_is_closed(forced):
    # A boundary state exists only in its hand-off (the band cannot offer
    # interior states), so the gate never drops it; with the gate closed the
    # attention blocks at that boundary go with it, making the run restorable
    # at once instead of a state-only run that is evicted first.
    scheduler = _make_partial_tail_scheduler()
    _make_partial_tail_request(scheduler, {"write_through": True} if forced else None)
    status = scheduler._req_status["req"]
    attn_idx, attn_cfg = next(
        (idx, cfg) for idx, cfg in enumerate(scheduler.config.kv_group_configs)
        if not isinstance(cfg.kv_cache_spec, MambaSpec)
    )
    status.group_states[attn_idx].block_ids[:] = list(
        range(41, 41 + 16 // attn_cfg.tokens_per_block)
    )
    scheduler.manager.prepare_store.side_effect = _cpu_store_output
    scheduler._write_through = False
    jobs = scheduler._build_aligned_boundary_store_jobs({"req": [(1, 98, 16)]})
    srcs = sorted(j.src_spec.block_ids.tolist()[0] for j in jobs.values())
    if forced:
        assert srcs == [98]       # write-through request: normal stores carry attention
    else:
        assert srcs == sorted([98, 41 + 16 // attn_cfg.tokens_per_block - 1])
    assert all(j in status.transfer_jobs for j in jobs)


def test_a_lone_block_is_not_stored_for_a_class_with_state(band):
    scheduler, pool, blocks = band
    scheduler._has_state_groups = True  # the fixture's config is hybrid
    scheduler.manager.prepare_store_run = MagicMock()
    assert scheduler._build_write_back_jobs() == {}
    scheduler.manager.prepare_store_run.assert_not_called()
    assert all(b.ref_cnt == 0 for b in blocks)


def _register_run(scheduler, blocks, idxs):
    chain = [OffloadKey(bytes(blocks[i].block_hash)) for i in idxs]
    scheduler._run_keys[chain[-1]] = chain
    for k in chain:
        scheduler._key_run[k] = chain[-1]
    return chain


def test_a_run_larger_than_the_budget_is_written_once_nothing_is_held(band, monkeypatch):
    scheduler, pool, blocks = band
    monkeypatch.setattr(sched_mod, "WRITE_BACK_MAX_HELD", 2)
    chain = _register_run(scheduler, blocks, (1, 2, 5))  # 3 storable members > 2
    scheduler.manager.prepare_store_run = MagicMock(
        side_effect=lambda keys, t, whole, held=False: _cpu_store_output(keys))
    scheduler._has_state_groups = True  # no lone blocks: only the run
    jobs = scheduler._build_write_back_jobs()
    (keys, tag), kwargs = scheduler.manager.prepare_store_run.call_args
    assert list(keys) == chain and tag == chain[-1] and kwargs["whole"]
    assert sorted(_src_blocks(jobs)) == [2, 3, 6]
    assert scheduler._wb_counts["run_oversize"] == 1


def test_an_oversize_run_waits_while_other_blocks_are_held(band, monkeypatch):
    scheduler, pool, blocks = band
    monkeypatch.setattr(sched_mod, "WRITE_BACK_MAX_HELD", 2)
    _register_run(scheduler, blocks, (1, 2, 5))
    scheduler._wb_held[999] = [KVCacheBlock(99)]
    scheduler.manager.prepare_store_run = MagicMock()
    assert scheduler._build_write_back_jobs() == {}
    assert scheduler._wb_counts["run_deferred_budget"] == 1


def test_reset_cache_merges_a_finished_unsignalled_request(band):
    scheduler, _, _ = band
    request = _make_partial_tail_request(scheduler)
    request.is_finished.return_value = True
    status = scheduler._req_status["req"]
    attn_idx = next(
        idx for idx, cfg in enumerate(scheduler.config.kv_group_configs)
        if not isinstance(cfg.kv_cache_spec, MambaSpec)
    )
    status.group_states[attn_idx].offload_keys.append(OffloadKey(b"tip"))
    scheduler.manager.merge_tag = MagicMock()
    scheduler.reset_cache()
    (ctx, tag), _ = scheduler.manager.merge_tag.call_args
    assert ctx.req_id == "req" and tag == OffloadKey(b"tip")
    assert "req" not in scheduler._req_status


# --- 2026-11-03: the band writes what a restore reads, paced and accounted ---


def _hkey(i: int, group: int) -> OffloadKey:
    return OffloadKey(i.to_bytes(32, "big") + group.to_bytes(4, "big"))


def _groups(scheduler):
    configs = scheduler.config.kv_group_configs
    attn = next(i for i, c in enumerate(configs) if not isinstance(c.kv_cache_spec, MambaSpec))
    state = [i for i, c in enumerate(configs) if isinstance(c.kv_cache_spec, MambaSpec)]
    return configs, attn, state


def test_restore_chain_is_trimmed_to_the_deepest_stored_state(band):
    scheduler, _, _ = band
    configs, attn, state = _groups(scheduler)
    assert state and len({c.tokens_per_chunk for c in configs}) == 1
    by_group = [[_hkey(i, c.group_idx) for i in range(4)] for c in configs]
    assert scheduler._restore_chain(by_group) == []          # no state stored
    for s in state:                                           # state at chunk 1
        scheduler._note_state_stored(by_group[s][1])
    chain = scheduler._restore_chain(by_group)
    expect = [by_group[attn][0], by_group[attn][1]] + [by_group[s][1] for s in state]
    assert sorted(chain) == sorted(expect)
    assert chain.index(by_group[attn][0]) < chain.index(by_group[attn][1])
    # Interior states and attention past the boundary are never offered.
    assert by_group[attn][2] not in chain and by_group[state[0]][0] not in chain


def test_a_run_with_no_stored_state_is_not_remembered(band):
    scheduler, _, _ = band
    configs, attn, _ = _groups(scheduler)
    by_group = [[_hkey(i, c.group_idx) for i in range(3)] for c in configs]
    scheduler.manager.merge_tag = MagicMock()
    scheduler._record_run(_finished(scheduler, by_group))
    assert by_group[attn][-1] not in scheduler._run_keys
    assert scheduler._wb_counts["run_unrestorable_dropped"] == 1


def test_a_refused_run_is_offered_again_the_next_step(band):
    # No backoff: the GPU reuses a refused run's blocks while it waits, so a
    # wait lost the run (2026-11-05). The next step offers it again.
    scheduler, pool, blocks = band
    scheduler._has_state_groups = True
    _register_run(scheduler, blocks, (1, 2, 5))
    scheduler.manager.prepare_store_run = MagicMock(return_value=None)
    assert scheduler._build_write_back_jobs() == {}
    assert scheduler._wb_counts["store_refused"] == 1
    scheduler._build_write_back_jobs()
    assert scheduler.manager.prepare_store_run.call_count == 2
    assert all(b.ref_cnt == 0 for b in blocks)


def test_gpu_eviction_is_classified_by_copy_state(band):
    scheduler, pool, blocks = band
    _register_run(scheduler, blocks, (1, 2))
    scheduler._on_gpu_evict(blocks[1])                      # run member, no copy
    scheduler._note_in_pool([OffloadKey(bytes(blocks[2].block_hash))])
    scheduler._on_gpu_evict(blocks[2])                      # member, in the pool
    scheduler._on_gpu_evict(blocks[5])                      # in no run
    c = scheduler._wb_counts
    assert (c["evict_uncopied"], c["evict_copied"], c["evict_other"]) == (1, 1, 1)


def test_rescue_copies_an_evicted_member_and_flushes_it(band, monkeypatch):
    scheduler, pool, blocks = band
    monkeypatch.setattr(sched_mod, "WRITE_BACK_RESCUE", 1)
    scheduler._has_state_groups = True
    chain = _register_run(scheduler, blocks, (1, 2))
    scheduler.manager.prepare_store_run = MagicMock(
        side_effect=lambda keys, t, whole, held=False: _cpu_store_output(keys))
    scheduler._on_gpu_evict(blocks[1])
    scheduler._on_gpu_evict(blocks[2])                      # over the per-step cap
    assert scheduler._wb_counts["evict_rescued"] == 1
    assert scheduler._wb_counts["evict_uncopied"] == 1
    pool.free_block_queue.remove(blocks[1])                 # now another request's
    blocks[1].ref_cnt = 1
    jobs = scheduler._build_write_back_jobs()
    rescue = [j for j, job in jobs.items() if job.src_spec.block_ids.tolist() == [2]]
    assert rescue and rescue[0] in scheduler._current_batch_jobs_to_flush
    first = scheduler.manager.prepare_store_run.call_args_list[0]
    assert list(first.args[0]) == [chain[0]] and first.kwargs["whole"] is False
    # Completing it releases nothing it does not hold.
    meta = OffloadingWorkerMetadata()
    for _ in range(scheduler.config.num_workers):
        meta.mark_completed(rescue[0])
    scheduler.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=meta))
    assert blocks[1].ref_cnt == 1 and chain[0] in scheduler._pool_has


def test_band_lead_follows_the_reuse_rate(band, monkeypatch):
    scheduler, pool, _ = band
    monkeypatch.setattr(sched_mod, "WRITE_BACK_SCAN_MAX", 10000)
    clock = [0.0]
    monkeypatch.setattr(sched_mod.time, "monotonic", lambda: clock[0])
    pool.num_popped = 0
    scheduler._wb_rate_at, scheduler._wb_popped, scheduler._wb_rate = 0.0, 0, 0.0
    scheduler._wb_latency = 1.0
    assert scheduler._band_lead(0.0) == sched_mod.WRITE_BACK_SCAN
    for _ in range(20):                                   # 100 blocks/s for 20 s
        clock[0] += 1.0
        pool.num_popped += 100
        lead = scheduler._band_lead(clock[0])
    assert lead == pytest.approx(100 * 1.0 * sched_mod.WRITE_BACK_LEAD_FACTOR
                                 + sched_mod.WRITE_BACK_SCAN, rel=0.01)


def test_the_band_holds_at_most_its_share_of_free_blocks(band, monkeypatch):
    scheduler, pool, blocks = band
    monkeypatch.setattr(sched_mod, "WRITE_BACK_FREE_SHARE", 0.25)   # 6 free -> 1
    jobs = scheduler._build_write_back_jobs()
    assert len(_src_blocks(jobs)) == 1


def test_a_revived_copied_block_counts_as_duplicate(band):
    scheduler, pool, blocks = band
    jobs = scheduler._build_write_back_jobs()
    assert scheduler._wb_counts["revived_copied"] == 0      # the hold is not a hit
    meta = OffloadingWorkerMetadata()
    for job_id in jobs:
        for _ in range(scheduler.config.num_workers):
            meta.mark_completed(job_id)
    scheduler.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=meta))
    scheduler._on_gpu_revive(blocks[1])
    assert scheduler._wb_counts["revived_copied"] == 1


def test_prompt_boundary_becomes_the_junction_when_none_is_set(band):
    scheduler, _, _ = band
    request = _make_partial_tail_request(scheduler)
    request.shared_prefix_boundary = 0
    status = scheduler._req_status["req"]
    scheduler.config = scheduler.config._replace(retention_interval=0)  # sparse
    tpc = scheduler.config.kv_group_configs[scheduler._chain_ref_group].tokens_per_chunk
    scheduler._maybe_set_prompt_boundary(status, 0)
    assert request.shared_prefix_boundary == (30 - 1) // tpc * tpc
    # An existing junction past the resume point is kept.
    request.shared_prefix_boundary = 4
    scheduler._maybe_set_prompt_boundary(status, 0)
    assert request.shared_prefix_boundary == 4


def test_restore_chain_limit_picks_a_shallower_stored_state(band):
    scheduler, _, _ = band
    configs, attn, state = _groups(scheduler)
    by_group = [[_hkey(i, c.group_idx) for i in range(4)] for c in configs]
    for s in state:
        scheduler._note_state_stored(by_group[s][1])
        scheduler._note_state_stored(by_group[s][3])
    tpc = configs[attn].tokens_per_chunk
    assert by_group[attn][3] in scheduler._restore_chain(by_group)
    shallow = scheduler._restore_chain(by_group, limit=3 * tpc - 1)
    assert by_group[attn][1] in shallow and by_group[attn][2] not in shallow
    assert all(by_group[s][1] in shallow and by_group[s][3] not in shallow for s in state)


def test_a_broken_run_is_retrimmed_then_dropped(band):
    """A chain member neither on the GPU nor in the pool breaks the run: it is
    re-trimmed to the deepest stored boundary below the break, and dropped
    when no boundary is left, instead of writing a chain no restore can use."""
    scheduler, _, _ = band
    scheduler._has_state_groups = True
    configs, attn, state = _groups(scheduler)
    by_group = [[_hkey(i, c.group_idx) for i in range(4)] for c in configs]
    for s in state:
        scheduler._note_state_stored(by_group[s][1])
        scheduler._note_state_stored(by_group[s][3])
    scheduler.manager.merge_tag = MagicMock()
    scheduler._record_run(_finished(scheduler, by_group))
    tag = by_group[attn][-1]
    assert by_group[attn][3] in scheduler._run_keys[tag]
    # The shallow boundary's chain is in the pool; attention chunk 2 is lost.
    scheduler._note_in_pool([by_group[attn][0], by_group[attn][1]]
                            + [by_group[s][1] for s in state])
    scheduler._note_in_pool([by_group[s][3] for s in state] + [by_group[attn][3]])
    assert scheduler._run_members(tag, scheduler._config_idx_by_group) == []
    assert tag in scheduler._run_keys
    assert by_group[attn][2] not in scheduler._run_keys[tag]
    assert scheduler._wb_counts["run_broken_trimmed"] == 1
    # Now the shallow root is gone too: nothing restorable is left.
    scheduler._pool_has.pop(by_group[attn][0])
    assert scheduler._run_members(tag, scheduler._config_idx_by_group) == []
    assert tag not in scheduler._run_keys
    assert scheduler._wb_counts["run_broken_dropped"] == 1


def _two_point_run(scheduler):
    """A recorded run with stored states at chunks 1 and 3 (boundaries 2 and 4 chunks)."""
    configs, attn, state = _groups(scheduler)
    by_group = [[_hkey(i, c.group_idx) for i in range(4)] for c in configs]
    for s in state:
        scheduler._note_state_stored(by_group[s][1])
        scheduler._note_state_stored(by_group[s][3])
    scheduler.manager.merge_tag = MagicMock()
    scheduler._record_run(_finished(scheduler, by_group))
    return by_group, attn, state, by_group[attn][-1]


def test_a_run_keeps_an_older_restore_point(band):
    """A branch or fan-out from earlier in the conversation, and a break below
    the deepest point, need an older boundary's state: the run keeps it."""
    scheduler, _, _ = band
    scheduler._has_state_groups = True
    by_group, attn, state, tag = _two_point_run(scheduler)
    keys = scheduler._run_keys[tag]
    assert all(by_group[s][3] in keys and by_group[s][1] in keys for s in state)
    assert scheduler._run_extras[tag] == {by_group[s][1] for s in state}
    assert all(by_group[s][0] not in keys and by_group[s][2] not in keys for s in state)


def test_keep_points_zero_keeps_only_the_deepest(band, monkeypatch):
    monkeypatch.setattr(sched_mod, "WRITE_BACK_KEEP_POINTS", 0)
    scheduler, _, _ = band
    scheduler._has_state_groups = True
    by_group, attn, state, tag = _two_point_run(scheduler)
    assert all(by_group[s][1] not in scheduler._run_keys[tag] for s in state)


def test_a_lost_older_point_is_forgotten_without_trimming(band):
    scheduler, _, _ = band
    scheduler._has_state_groups = True
    by_group, attn, state, tag = _two_point_run(scheduler)
    old_point = {by_group[s][1] for s in state}
    scheduler._note_in_pool([k for k in scheduler._run_keys[tag] if k not in old_point])
    scheduler._forget_in_pool(old_point)  # the pool evicted the older point
    assert scheduler._run_members(tag, scheduler._config_idx_by_group) == []
    keys = scheduler._run_keys[tag]
    assert by_group[attn][3] in keys and not (old_point & set(keys))
    assert scheduler._wb_counts["run_point_lost"] == 1
    assert scheduler._wb_counts["run_broken_trimmed"] == 0
    assert scheduler._wb_counts["run_broken_dropped"] == 0


def test_restore_chain_carries_the_eagle_chunk_past_the_state(band):
    """A lookup drops a drafter group's trailing chunk, so a restore at the
    state boundary d reads that group to d + one chunk: the chain holds it."""
    scheduler, _, _ = band
    configs, attn, state = _groups(scheduler)
    configs = list(configs)
    configs[attn] = configs[attn]._replace(is_eagle_group=True)
    scheduler.config = SimpleNamespace(**{
        **{k: getattr(scheduler.config, k) for k in dir(scheduler.config)
           if not k.startswith("_")},
        "kv_group_configs": configs,
    })
    by_group = [[_hkey(i, c.group_idx) for i in range(4)] for c in configs]
    for s in state:
        scheduler._note_state_stored(by_group[s][1])
        scheduler._note_state_stored(by_group[s][3])
    chain = scheduler._restore_chain(by_group)
    # state at chunk 3 has no eagle chunk past it; state at chunk 1 does
    assert by_group[attn][2] in chain and by_group[attn][3] not in chain
    assert all(by_group[s][1] in chain and by_group[s][3] not in chain for s in state)


def test_prompt_boundary_is_a_chunk_earlier_with_a_drafter_group(band):
    scheduler, _, _ = band
    request = _make_partial_tail_request(scheduler)
    request.shared_prefix_boundary = 0
    status = scheduler._req_status["req"]
    configs = [c._replace(is_eagle_group=True) if not isinstance(c.kv_cache_spec, MambaSpec)
               else c for c in scheduler.config.kv_group_configs]
    scheduler.config = scheduler.config._replace(retention_interval=0, kv_group_configs=configs)
    tpc = configs[scheduler._chain_ref_group].tokens_per_chunk
    scheduler._maybe_set_prompt_boundary(status, 0)
    assert request.shared_prefix_boundary == (30 - 1) // tpc * tpc - tpc


def test_keys_a_store_evicted_leave_the_band_view(band):
    """H1: the pool's own evictions (returned by a store) are forgotten, so the
    band rewrites those blocks when the GPU drops them instead of assuming
    the pool still has them."""
    scheduler, pool, blocks = band
    scheduler._has_state_groups = True
    _register_run(scheduler, blocks, (1, 2))
    gone = _hkey(7, 0)
    scheduler._note_in_pool([gone])
    scheduler.manager.prepare_store_run = MagicMock(
        side_effect=lambda keys, t, whole, held=False: PrepareStoreOutput(
            keys_to_store=list(keys), store_spec=CPULoadStoreSpec(list(range(len(keys)))),
            evicted_keys=[gone]))
    assert scheduler._build_write_back_jobs()
    assert gone not in scheduler._pool_has
    assert scheduler._wb_counts["pool_evicted_forgot"] == 1


def test_a_refused_run_with_stale_pool_keys_is_retrimmed_at_once(band, monkeypatch):
    """H1: a refusal caused by keys the band wrongly assumed in the pool is
    retried in the same step with the re-trimmed run, not after the backoff."""
    scheduler, pool, blocks = band
    scheduler._has_state_groups = True
    chain = _register_run(scheduler, blocks, (1, 2, 5))
    stale = _hkey(9, 0)
    scheduler._run_keys[chain[-1]] = [stale] + chain
    scheduler._key_run[stale] = chain[-1]
    scheduler._note_in_pool([stale])
    calls = []

    def prepare(keys, t, whole, held=False):
        calls.append(list(keys))
        if len(calls) == 1:
            return None
        return _cpu_store_output(keys)

    scheduler.manager.prepare_store_run = MagicMock(side_effect=prepare)
    scheduler.manager.lookup_batch = MagicMock(side_effect=lambda keys, ctx: [
        LookupResult.MISS if k == stale else LookupResult.HIT for k in keys])
    real = scheduler._run_members
    retried = []

    def run_members(tag, by_group):
        out = real(tag, by_group)
        if stale not in scheduler._pool_has:
            retried.append(tag)
            scheduler._run_keys[tag] = chain     # what a re-trim would leave
            out = real(tag, by_group)
        return out

    monkeypatch.setattr(scheduler, "_run_members", run_members)
    jobs = scheduler._build_write_back_jobs()
    assert len(calls) == 2 and retried
    assert sorted(_src_blocks(jobs)) == [2, 3, 6]
    assert scheduler._wb_counts["store_refused_retrimmed"] == 1
    assert scheduler._wb_counts["store_refused"] == 0


def test_a_refusal_for_room_forgets_nothing_the_pool_still_holds(band):
    """A refused run whose keys the pool still holds (no room, not stale) keeps
    the band's view: forgetting them made the run look broken and dropped it."""
    scheduler, pool, blocks = band
    scheduler._has_state_groups = True
    chain = _register_run(scheduler, blocks, (1, 2, 5))
    held = _hkey(9, 0)
    scheduler._run_keys[chain[-1]] = [held] + chain
    scheduler._key_run[held] = chain[-1]
    scheduler._note_in_pool([held])
    scheduler.manager.prepare_store_run = MagicMock(return_value=None)
    scheduler.manager.lookup_batch = MagicMock(side_effect=lambda keys, ctx: [LookupResult.HIT] * len(keys))
    assert scheduler._build_write_back_jobs() == {}
    assert held in scheduler._pool_has
    assert scheduler._wb_counts["store_refused"] == 1
    assert scheduler._wb_counts["store_refused_retrimmed"] == 0
    assert chain[-1] in scheduler._run_keys


# --- 2026-11-04: the pool evicts runs the GPU still holds first ---


def test_a_run_the_pool_holds_whole_is_not_offered_again(band):
    scheduler, pool, blocks = band
    scheduler._has_state_groups = True
    chain = _register_run(scheduler, blocks, (1, 2, 5))
    scheduler._note_in_pool(chain)
    scheduler.manager.prepare_store_run = MagicMock()
    # A re-offer would only make the run newest in the pool.
    assert scheduler._build_write_back_jobs() == {}
    scheduler.manager.prepare_store_run.assert_not_called()


def test_the_budget_counts_only_members_the_pool_lacks(band, monkeypatch):
    scheduler, pool, blocks = band
    monkeypatch.setattr(sched_mod, "WRITE_BACK_MAX_HELD", 2)
    scheduler._has_state_groups = True
    chain = _register_run(scheduler, blocks, (1, 2, 5))  # 3 members, the root in the pool
    scheduler._note_in_pool(chain[:1])
    scheduler.manager.prepare_store_run = MagicMock(
        side_effect=lambda keys, t, whole, held=False: _cpu_store_output([k for k in keys if k != chain[0]]))
    jobs = scheduler._build_write_back_jobs()
    assert sorted(_src_blocks(jobs)) == [3, 6]
    assert scheduler._wb_counts["run_oversize"] == 0


def test_a_member_leaving_the_gpu_makes_its_run_newest_in_the_pool(band):
    scheduler, pool, blocks = band
    chain = _register_run(scheduler, blocks, (1, 2))
    scheduler._note_in_pool(chain[1:])
    scheduler._on_gpu_evict(blocks[1])      # not in the pool: lost, nothing to touch
    scheduler._on_gpu_evict(blocks[2])      # in the pool: now its only copy
    scheduler.manager.touch.reset_mock()
    scheduler._build_write_back_jobs()
    scheduler.manager.touch.assert_called_once_with([chain[1]], None)
    scheduler._build_write_back_jobs()
    scheduler.manager.touch.assert_called_once()


# --- 2026-11-04: a finished run is pinned until the band copies it ---


def _pin_chain(scheduler, monkeypatch, chain):
    monkeypatch.setattr(scheduler, "_run_tag", lambda by_group: chain[-1])
    monkeypatch.setattr(scheduler, "_restore_chain", lambda by_group, **kw: chain)
    scheduler._pin_finished(SimpleNamespace(group_states=[]))


def test_a_finished_run_is_pinned_until_the_band_copies_it(band, monkeypatch):
    scheduler, pool, blocks = band
    scheduler._has_state_groups = True
    run = [blocks[i] for i in (1, 2, 5)]
    pool.touch(run)                              # the finishing request holds them
    chain = [OffloadKey(bytes(b.block_hash)) for b in run]
    _pin_chain(scheduler, monkeypatch, chain)
    for b in run:                                # the scheduler frees them
        b.ref_cnt -= 1
    # Pinned: out of the free queue, so this step's allocations cannot reuse them.
    assert all(b.ref_cnt == 1 for b in run) and not {2, 3, 6} & set(_queue_ids(pool))
    _register_run(scheduler, blocks, (1, 2, 5))  # recorded at the end of the next step
    scheduler.manager.prepare_store_run = MagicMock(
        side_effect=lambda keys, t, whole, held=False: _cpu_store_output(keys))
    jobs = scheduler._build_write_back_jobs()
    assert sorted(_src_blocks(jobs)) == [2, 3, 6]
    # The copy job holds them now; the pins are gone.
    assert not scheduler._wb_pinned and all(b.ref_cnt == 1 for b in run)


def test_a_refused_pinned_run_stays_pinned_until_stored(band, monkeypatch):
    # Unpinned after one refusal, the GPU reused the run's blocks (tail first)
    # and lost them. Now the pins stay, the run is offered again each step, and
    # the pins go once its copy jobs hold the blocks, or after PIN_HOLD_S.
    scheduler, pool, blocks = band
    scheduler._has_state_groups = True
    run = [blocks[i] for i in (1, 2, 5)]
    pool.touch(run)
    chain = [OffloadKey(bytes(b.block_hash)) for b in run]
    _pin_chain(scheduler, monkeypatch, chain)
    for b in run:
        b.ref_cnt -= 1
    _register_run(scheduler, blocks, (1, 2, 5))
    clock = [100.0]
    monkeypatch.setattr(sched_mod.time, "monotonic", lambda: clock[0])
    scheduler._wb_pinned = {t: (clock[0], bs) for t, (_, bs) in scheduler._wb_pinned.items()}
    scheduler.manager.prepare_store_run = MagicMock(return_value=None)
    assert scheduler._build_write_back_jobs() == {}
    assert chain[-1] in scheduler._wb_pinned and all(b.ref_cnt == 1 for b in run)
    assert scheduler._wb_counts["pin_kept_refused"] == 1
    scheduler.manager.prepare_store_run = MagicMock(
        side_effect=lambda keys, t, whole, held=False: _cpu_store_output(keys))
    clock[0] += 1.0
    jobs = scheduler._build_write_back_jobs()
    assert sorted(_src_blocks(jobs)) == [2, 3, 6]
    assert not scheduler._wb_pinned and all(b.ref_cnt == 1 for b in run)


def test_a_refused_pin_expires_after_the_hold(band, monkeypatch):
    scheduler, pool, blocks = band
    scheduler._has_state_groups = True
    run = [blocks[1], blocks[2]]
    pool.touch(run)
    chain = [OffloadKey(bytes(b.block_hash)) for b in run]
    _pin_chain(scheduler, monkeypatch, chain)
    for b in run:
        b.ref_cnt -= 1
    _register_run(scheduler, blocks, (1, 2))
    clock = [100.0]
    monkeypatch.setattr(sched_mod.time, "monotonic", lambda: clock[0])
    scheduler._wb_pinned = {t: (clock[0], bs) for t, (_, bs) in scheduler._wb_pinned.items()}
    scheduler.manager.prepare_store_run = MagicMock(return_value=None)
    scheduler._build_write_back_jobs()
    assert scheduler._wb_pinned
    clock[0] += sched_mod.WRITE_BACK_PIN_HOLD_S + 1
    scheduler._build_write_back_jobs()
    assert not scheduler._wb_pinned and all(b.ref_cnt == 0 for b in run)
    assert scheduler._wb_counts["pin_expired"] == 1


def test_band_counters_reach_the_stats_once(band):
    scheduler, _, _ = band
    scheduler.manager.get_stats = MagicMock(return_value=None)
    scheduler._wb_counts["run_offered"] += 3
    stats = scheduler.get_stats()
    name = sched_mod._ConnectorMetricName.WRITE_BACK_EVENTS
    assert stats.data["data"][name] == {("run_offered",): 3}
    scheduler._wb_counts["run_offered"] += 1
    stats = scheduler.get_stats()
    assert stats.data["data"][name] == {("run_offered",): 1}


def test_a_pinned_run_never_recorded_is_released(band, monkeypatch):
    scheduler, pool, blocks = band
    scheduler._has_state_groups = True               # no lone-block stores
    run = [blocks[1], blocks[2]]
    pool.touch(run)
    _pin_chain(scheduler, monkeypatch, [OffloadKey(bytes(b.block_hash)) for b in run])
    for b in run:
        b.ref_cnt -= 1
    clock = [100.0]
    monkeypatch.setattr(sched_mod.time, "monotonic", lambda: clock[0])
    scheduler._wb_pinned = {t: (clock[0], bs) for t, (_, bs) in scheduler._wb_pinned.items()}
    scheduler._build_write_back_jobs()
    assert scheduler._wb_pinned                  # waits for the run to be recorded
    clock[0] += sched_mod.WRITE_BACK_PIN_S + 1
    scheduler._build_write_back_jobs()
    assert not scheduler._wb_pinned and all(b.ref_cnt == 0 for b in run)
    assert _queue_ids(pool)[-2:] == [3, 2]       # the tail, end first, as vLLM frees


def test_a_run_member_in_use_on_the_gpu_is_not_missing(band):
    scheduler, pool, blocks = band
    scheduler._has_state_groups = True
    chain = _register_run(scheduler, blocks, (1, 2, 5))
    pool.touch([blocks[2]])                      # a running request reads it
    members = scheduler._run_members(chain[-1], scheduler._config_idx_by_group)
    assert sorted(m[1].block_id for m in members) == [2, 6]
    assert chain[-1] in scheduler._run_keys and not scheduler._wb_counts["run_broken_trimmed"]


def test_a_boundary_hand_off_stores_the_whole_sliding_window(monkeypatch):
    # A restore at the boundary reads the window ending there; its earlier
    # chunks leave the GPU as the decode passes them, so the hand-off stores
    # all of them, not only the chunk at the boundary (2026-11-04).
    monkeypatch.setattr(sched_mod.GroupOffloadConfig, "load_window_size_in_chunks",
                        lambda self, num_tokens: 2)
    scheduler = _make_partial_tail_scheduler()
    request = _make_partial_tail_request(scheduler)
    status = scheduler._req_status["req"]
    attn_idx, attn_cfg = next(
        (idx, cfg) for idx, cfg in enumerate(scheduler.config.kv_group_configs)
        if not isinstance(cfg.kv_cache_spec, MambaSpec)
    )
    tpb, tpc = attn_cfg.tokens_per_block, attn_cfg.tokens_per_chunk
    boundary = 2 * tpc
    request.num_tokens = request.num_prompt_tokens = boundary + 2
    request.all_token_ids = list(range(boundary + 2))
    request.block_hashes = [BlockHash(f"h{i}".encode()) for i in range(boundary)]
    status.group_states[attn_idx].block_ids[:] = list(range(41, 41 + boundary // tpb))
    scheduler.manager.prepare_store.side_effect = _cpu_store_output
    scheduler._write_through = False
    jobs = scheduler._build_aligned_boundary_store_jobs({"req": [(1, 98, boundary)]})
    srcs = sorted(j.src_spec.block_ids.tolist()[0] for j in jobs.values())
    assert srcs == sorted([98] + [41 + e // tpb - 1 for e in (boundary, tpc)])


def test_a_run_store_joins_the_chain_keys_the_pool_already_holds(band):
    # A chain key the GPU no longer holds but the pool does is offered too,
    # so it joins the run instead of living only in an older run (2026-11-04).
    scheduler, pool, blocks = band
    scheduler._has_state_groups = True
    only_in_pool = _hkey(7, scheduler.config.kv_group_configs[0].group_idx)
    chain = [only_in_pool] + [OffloadKey(bytes(blocks[i].block_hash)) for i in (2, 5)]
    scheduler._run_keys[chain[-1]] = chain
    for k in chain:
        scheduler._key_run[k] = chain[-1]
    scheduler._note_in_pool([only_in_pool])
    scheduler.manager.prepare_store_run = MagicMock(
        side_effect=lambda keys, t, whole, held=False: _cpu_store_output([k for k in keys if k != only_in_pool]))
    jobs = scheduler._build_write_back_jobs()
    (keys, tag), kwargs = scheduler.manager.prepare_store_run.call_args
    assert list(keys) == chain and tag == chain[-1] and kwargs["whole"]
    assert sorted(_src_blocks(jobs)) == [3, 6]       # only what the pool lacks is copied


def test_a_pinned_alias_block_without_its_own_hash_is_released(band, monkeypatch):
    # The pool's map also holds lookup-only aliases (a finish-checkpoint anchor or
    # tail): such a block can carry no hash of its own (2026-11-05 crash).
    scheduler, pool, blocks = band
    scheduler._has_state_groups = True
    alias = KVCacheBlock(9)
    key = OffloadKey(bytes(blocks[1].block_hash))
    monkeypatch.setattr(pool, "get_cached_block", lambda h, g: [alias])
    alias.ref_cnt = 1                            # the finishing request holds it
    _pin_chain(scheduler, monkeypatch, [key])
    alias.ref_cnt -= 1                           # the scheduler frees it
    assert alias.block_hash is None and alias.ref_cnt == 1
    scheduler._unpin(key)
    assert alias.ref_cnt == 0 and not scheduler._wb_pinned


# --- a tail key in the band's view ---


def _assert_one_row_per_key(scheduler, jobs, load: bool = False):
    """The invariant the transfer layer pairs on: the per-group split of a
    store's keys, the GPU blocks it reads, and the keys themselves are the same
    size. A store whose split does not add up to its keys mis-pairs rows with
    GPU blocks, the phase-1 P0 class. A restore pairs on the same invariant, and
    there the GPU side is the destination, so ``load`` names which spec carries
    the split.
    """
    for job_id, job in jobs.items():
        keys = scheduler._jobs[job_id].keys
        gpu_spec = job.dst_spec if load else job.src_spec
        assert sum(gpu_spec.group_sizes) == len(gpu_spec.block_ids) == len(keys)


def _stored_tail():
    """A finished turn's tail stored for real, through the route that stores
    tails, and the band's state after it: the scheduler, the request's
    bookkeeping, the tail's keys in group order, and the run the tail was
    recorded against."""
    scheduler, handoff, blocks = _answer_tail_offer()
    jobs = scheduler._build_answer_tail_store_jobs()
    [job_id] = jobs
    _assert_one_row_per_key(scheduler, jobs)
    keys = sorted(scheduler._jobs[job_id].keys)
    req_status = scheduler._req_status["req"]
    run = scheduler._run_tag([gs.offload_keys for gs in req_status.group_states])
    assert run is not None and keys == sorted(_tail_keys(handoff, blocks))
    return scheduler, req_status, keys, run


def test_a_tail_key_is_never_a_band_member(band):
    # A tail key names a position between two chunk boundaries, but the GPU
    # block it points at is hashed and chunk-aligned, so the band's membership
    # test would take it for a whole chunk and copy that block a second time
    # under the tail's own key.
    scheduler, pool, blocks = band
    scheduler._tail_run[OffloadKey(bytes(blocks[1].block_hash))] = None
    jobs = scheduler._build_write_back_jobs()
    _assert_one_row_per_key(scheduler, jobs)
    assert sorted(_src_blocks(jobs)) == [3, 6]   # block 2 is a tail, not a chunk
    assert _queue_ids(pool) == [1, 2, 4, 5]      # and it was never taken from it


def test_a_tail_key_is_never_offered_as_a_whole_run_state(band):
    # A whole-run store asks the pool to admit a run on the strength of the keys
    # offered with it, and a tail's pseudo-boundary can complete a boundary the
    # conversation's own keys do not. The band also has no whole-chunk source to
    # copy a tail from, so a tail is never part of this offer.
    scheduler, pool, blocks = band
    scheduler._has_state_groups = True
    tail = _hkey(7, scheduler.config.kv_group_configs[1].group_idx)
    chain = [OffloadKey(bytes(blocks[i].block_hash)) for i in (1, 2, 5)]
    scheduler._run_keys[chain[-1]] = chain + [tail]
    for k in chain:
        scheduler._key_run[k] = chain[-1]
    scheduler._tail_run[tail] = chain[-1]
    scheduler._note_in_pool([tail])
    scheduler.manager.prepare_store_run = MagicMock(
        side_effect=lambda keys, t, whole, held=False: _cpu_store_output(
            [k for k in keys if k != tail]
        )
    )
    jobs = scheduler._build_write_back_jobs()
    _assert_one_row_per_key(scheduler, jobs)
    (keys, tag), kwargs = scheduler.manager.prepare_store_run.call_args
    assert list(keys) == chain and tag == chain[-1] and kwargs["whole"]
    assert sorted(_src_blocks(jobs)) == [2, 3, 6]


def test_a_tail_leaves_when_its_run_leaves():
    # A tail belongs to a run, not to a session: the band forgets it the step
    # the run is dropped, so no later run picks up a tail it cannot write.
    scheduler, _, keys, run = _stored_tail()
    assert run is not None and scheduler._run_tails[run] == keys
    scheduler._drop_run(run)
    assert run not in scheduler._run_tails
    assert not [k for k in keys if k in scheduler._tail_run]


def test_a_tail_the_pool_evicted_is_counted_and_forgotten(band_with_tails):
    # A tail has no GPU hash to be re-offered from, so the pool's eviction feed
    # is the only way the band learns the arena dropped it.
    scheduler, pool, blocks = band_with_tails
    tag = OffloadKey(bytes(blocks[5].block_hash))
    tail = _hkey(7, 1)
    scheduler._tail_run[tail] = tag
    scheduler._run_tails[tag] = [tail]
    scheduler._note_in_pool([tail])
    scheduler.manager.pool_evicted = MagicMock(return_value=([tail], False))
    jobs = scheduler._build_write_back_jobs()
    _assert_one_row_per_key(scheduler, jobs)
    assert scheduler._tail_counts["tail_evicted"] == 1
    assert tail not in scheduler._tail_run and tag not in scheduler._run_tails


def test_a_tail_key_never_marks_a_chunk_boundary_state():
    # The pool holds a tail's rows, so they belong in the pool view; they are
    # not the state of a chunk boundary, which is what a restore chain reads.
    scheduler, _, keys, run = _stored_tail()
    state_key = next(
        k for k in keys if get_offload_group_idx(k) in scheduler._state_group_ids
    )
    scheduler._note_states(keys)               # what a completed store does
    assert state_key in scheduler._pool_has
    assert bytes(state_key)[:-4] not in scheduler._state_sets


def test_a_tail_moves_to_the_run_its_request_last_named(monkeypatch):
    # A tail stored mid-decode is recorded against the attention key of its own
    # step. The finish names the run after the request's last one and moves the
    # rows there, so the band's record moves with them: otherwise the tail is
    # forgotten while the pool still holds it, and read as a chunk key again.
    # Only the band keeps run records, so only the band re-tags.
    scheduler, req_status, keys, early = _stored_tail()
    assert scheduler._tail_run[keys[0]] == early
    req_status.req.block_hashes.append(BlockHash(b"h7"))
    req_status.update_offload_keys()           # the next step keyed a deeper chunk
    monkeypatch.setattr(sched_mod, "WRITE_BACK", True)
    scheduler._record_run(req_status)
    last = req_status.group_states[0].offload_keys[-1]
    assert last != early
    assert all(scheduler._tail_run[k] == last for k in keys)
    assert scheduler._run_tails[last] == keys
    assert early not in scheduler._run_tails


def test_a_tail_is_recorded_against_the_run_that_owns_it():
    # A tail belongs to a run, not to a session: the run named after the
    # request's last attention key is what the tail leaves with, and the run a
    # later store of the conversation takes it along into.
    scheduler, _, keys, run = _stored_tail()
    assert run is not None
    assert scheduler._run_tails[run] == keys
    assert all(scheduler._tail_run[key] == run for key in keys)
    assert scheduler._tail_counts["tail_run_missing"] == 0


# --- the answer tail's store route: the core offers, the connector stores ---


def _answer_tail_offer(
    keyed_chunk: bool = True,
    pool_refused: bool = False,
    groups: tuple[int, ...] | None = None,
    null_group: int | None = None,
    scheduler: OffloadingConnectorScheduler | None = None,
    rows_override: dict[int, int] | None = None,
):
    """A finished turn's tail as the engine core offers it, and the connector
    that could store it: the scheduler, the hand-off and the GPU blocks the
    offer named per group, oldest row first. Nothing is stored here; the test
    drains the queue.
    """
    if scheduler is None:
        scheduler = _make_drafter_partial_tail_scheduler(True)
        prompt_tokens = 30
    else:
        # Long enough that a windowed group's rows all sit inside its table.
        prompt_tokens = 12_000
    request = _make_partial_tail_request(scheduler, num_prompt_tokens=prompt_tokens)
    req_status = scheduler._req_status["req"]
    if keyed_chunk:
        req_status.update_offload_keys()       # the conversation keyed a chunk
    tail_configs = scheduler.config.partial_tail_groups
    wanted = tuple(g.group_idx for g in tail_configs) if groups is None else groups
    handoff = AnswerTailHandoff(
        request_id="req",
        tail_hash=BlockHash(b"tail"),
        parent=None,
        length=12,
        resume_token=request.num_tokens - 1,
    )
    rows = {g.group_idx: g.tail_rows for g in tail_configs}
    if rows_override:
        rows.update(rows_override)
    blocks = {
        group_idx: [
            0
            if group_idx == null_group and row == 0
            else 100 + 10 * group_idx + row
            for row in range(rows[group_idx])
        ]
        for group_idx in wanted
    }
    for group_idx, block_ids in blocks.items():
        assert scheduler.answer_tail_store.store_tail(handoff, group_idx, block_ids)
    scheduler.manager.prepare_store_run = MagicMock(
        side_effect=_takes_every_key
    )
    if pool_refused:
        # No side effect: a mock's side effect wins over its return value, and
        # the refusal this test is about is the pool answering None.
        scheduler.manager.prepare_store_run = MagicMock(return_value=None)
    scheduler.manager.pin = MagicMock(return_value=1)
    scheduler.manager.unpin = MagicMock(return_value=1)
    return scheduler, handoff, blocks


def _tail_keys(handoff, groups) -> list[OffloadKey]:
    return [make_offload_key(bytes(handoff.tail_hash), group) for group in groups]


def _tail_run_tag(scheduler) -> OffloadKey:
    """The run the request names: its last full-attention chunk key."""
    req_status = scheduler._req_status["req"]
    return scheduler._run_tag([gs.offload_keys for gs in req_status.group_states])


def test_an_offered_answer_tail_becomes_one_store_in_its_run():
    # The core can name the tail's key but can allocate no row and take no lane
    # time, so the offer becomes the connector's store: one job over the tail's
    # groups, in the run its request names, so the pool's run and the band's
    # record agree at store time rather than at the request's release.
    scheduler, handoff, blocks = _answer_tail_offer()
    jobs = scheduler._build_answer_tail_store_jobs()

    [job_id] = jobs
    _assert_one_row_per_key(scheduler, jobs)
    tail_keys = _tail_keys(handoff, sorted(blocks))
    assert scheduler._jobs[job_id].keys == set(tail_keys)
    (offered, tag), kwargs = scheduler.manager.prepare_store_run.call_args
    assert list(offered) == tail_keys
    assert tag == _tail_run_tag(scheduler)
    assert not kwargs["whole"]
    assert jobs[job_id].src_spec.block_ids.tolist() == [
        block for group in sorted(blocks) for block in blocks[group]
    ]
    assert scheduler._tail_counts["tail_offered"] == 1
    assert scheduler._tail_counts["tail_stored"] == 1
    assert all(scheduler._tail_run[key] == tag for key in tail_keys)


def test_a_tail_store_names_its_parent_and_length_with_its_rows():
    """The pool keeps rows, not turns: a tail key hashes the finished turn's
    tokens, so neither the parent it is chained on nor the length it covers can
    be read back off the row it was stored under. The store names both, because
    an engine that starts after this one has nothing else to ask with."""
    scheduler, handoff, blocks = _answer_tail_offer()

    scheduler._build_answer_tail_store_jobs()

    (_, _), kwargs = scheduler.manager.prepare_store_run.call_args
    assert kwargs["tails"] == {
        bytes(key): (b"", handoff.length) for key in _tail_keys(handoff, sorted(blocks))
    }


def test_a_tail_store_waits_for_the_step_that_may_still_own_its_blocks():
    # The finished request's blocks are back in the free queue by now, so the
    # job is listed ahead of the deferred stores and waited on: that is what
    # orders the copy before the block's new owner writes it. The band's second
    # lane stays out of it, because settled would also drop the wait for the
    # writes of the step that last held the block.
    scheduler, _, _ = _answer_tail_offer()
    jobs = scheduler._build_answer_tail_store_jobs()

    [job_id] = jobs
    _assert_one_row_per_key(scheduler, jobs)
    assert scheduler._current_batch_jobs_to_flush == {job_id}
    assert jobs[job_id].src_spec.settled is False


def test_a_tail_store_is_offered_again_until_its_retries_run_out():
    # A full pool is not a lost tail: the state exists only in the hand-off
    # that named it, so a retry can wait for room and nothing else. After the
    # budget the rows are given up on and counted, because a tail nobody stored
    # is invisible to every other counter.
    scheduler, _, _ = _answer_tail_offer(pool_refused=True)
    steps = 1 + sched_mod._TAIL_STORE_RETRIES

    for _ in range(steps):
        assert not scheduler._build_answer_tail_store_jobs()

    assert scheduler.manager.prepare_store_run.call_count == steps
    assert scheduler._tail_counts["tail_offered"] == 1
    assert scheduler._tail_counts["tail_retried"] == sched_mod._TAIL_STORE_RETRIES
    assert scheduler._tail_counts["store_pool_refused"] == 1
    assert scheduler._tail_counts["tail_stored"] == 0
    _assert_nothing_half_built(scheduler)


def _tail_gpu_blocks(handoff, blocks: dict[int, list[int]]) -> dict[int, KVCacheBlock]:
    """The GPU as a retried offer sees a one-row-per-group layout: one block
    per group, cached under that group's tail key, which is what the retry is
    checked against."""
    tail_blocks = {
        group_idx: KVCacheBlock(block_ids[0]) for group_idx, block_ids in blocks.items()
    }
    for group_idx, block in tail_blocks.items():
        block.set_block_hash(
            make_block_hash_with_group_id(handoff.tail_hash, group_idx), 12
        )
    return tail_blocks


def test_a_retry_of_a_tail_the_gpu_forgot_is_refused_as_stale():
    # The reviewer's shape: the pool is full at N, so the tail waits for room,
    # and the blocks it named went back to the free queue with the finished
    # turn. A new owner takes one and writes it, and the store reads blocks and
    # not keys, so the copy would land those bytes in the tail's rows and a
    # resume would read them as the state at the tail. A retried offer is
    # checked against the GPU before anything is reserved, and a tail the GPU
    # has forgotten is given up on rather than offered a fourth time.
    scheduler, handoff, blocks = _answer_tail_offer(pool_refused=True)
    tail_blocks = _tail_gpu_blocks(handoff, blocks)
    scheduler.bind_gpu_block_pool(FakeGPUPool(list(tail_blocks.values())))

    # Step N: the pool is full, and the tail is queued for a retry.
    assert not scheduler._build_answer_tail_store_jobs()
    assert scheduler._tail_counts["tail_retried"] == 1

    # The block is freed, reallocated and written: BlockPool forgets a reused
    # block's hashes, and the tail key goes with them.
    for group_idx, block in tail_blocks.items():
        block.reset_hash()
        block.set_block_hash(
            make_block_hash_with_group_id(BlockHash(b"owner"), group_idx), 12
        )

    # Step N+2: the retry refuses, stores nothing, and asks the pool for
    # nothing, because a retry cannot bring the tail's bytes back.
    assert not scheduler._build_answer_tail_store_jobs()
    assert scheduler._tail_counts["stale_offer"] == 1
    assert scheduler._tail_counts["tail_stored"] == 0
    assert scheduler._tail_counts["store_pool_refused"] == 0
    assert scheduler.manager.prepare_store_run.call_count == 1
    assert scheduler.answer_tail_store.pending == {}
    _assert_nothing_half_built(scheduler)


def test_a_tail_the_gpu_forgot_is_not_offered_again():
    # The fence proper: the core drops the offer the step the GPU forgets the
    # tail's key, so the store route is never handed a tail whose blocks are
    # gone. The re-check above is the second line, for a release that lands
    # while the offer is already in the connector's hand.
    scheduler, handoff, blocks = _answer_tail_offer()
    assert scheduler.answer_tail_store.has_offer_for("req")

    scheduler.answer_tail_store.forget_tail(
        make_block_hash_with_group_id(handoff.tail_hash, min(blocks))
    )

    assert not scheduler.answer_tail_store.has_offer_for("req")
    assert not scheduler._build_answer_tail_store_jobs()
    scheduler.manager.prepare_store_run.assert_not_called()
    assert dict(scheduler._tail_counts) == {}
    _assert_nothing_half_built(scheduler)


def _finished_step(req_ids: set[str]) -> SchedulerOutput:
    """The step that reports these requests finished, with nothing else
    scheduled: the shape the answer-tail store route is built inside, which is
    also the step that releases the request's offload state."""
    output = SchedulerOutput.make_empty()
    output.finished_req_ids = set(req_ids)
    return output


def _quiet_whole_chunk_store(req_status) -> None:
    """Cap a request's offloadable tokens at none, so the whole-chunk store
    path has nothing to do and this test sees only the tail's own store."""
    req_status.req.is_finished.return_value = True
    req_status.max_offload_tokens = 0


def test_a_queued_tail_keeps_its_request_alive_until_the_pool_makes_room():
    # A finished request with no store in flight has its offload state dropped
    # the step it finishes, which is the production case: the band stores under
    # a tag of its own and write-through is off. A tail the pool refused needs
    # that state for its retry, because the retry stores into the run the state
    # names, so releasing it anyway leaves every retry answering that it does
    # not know the request and no tail ever stored.
    scheduler, _, _ = _answer_tail_offer(pool_refused=True)
    req_status = scheduler._req_status["req"]
    _quiet_whole_chunk_store(req_status)
    assert not req_status.transfer_jobs

    assert not scheduler.build_connector_meta(_finished_step({"req"})).store_jobs
    assert "req" in scheduler._req_status
    assert scheduler._tail_counts["tail_retried"] == 1
    assert scheduler._tail_counts["untracked_request"] == 0

    # Step N+1: the pool has room, and the tail's own retry stores into it.
    scheduler.manager.prepare_store_run.side_effect = (
        lambda keys, tag, whole=False, held=False, tails=None: _cpu_store_output(keys)
    )
    meta = scheduler.build_connector_meta(_finished_step(set()))
    _assert_one_row_per_key(scheduler, meta.store_jobs)
    assert len(meta.store_jobs) == 1
    assert scheduler._tail_counts["tail_stored"] == 1
    assert scheduler._tail_counts["store_pool_refused"] == 0


def test_a_tail_the_pool_never_made_room_for_is_counted_once_and_released():
    # The retry budget is the tail's whole patience: when it ends the store is
    # given up on and counted, and the request state the retry was holding goes
    # with it. The request finished at the first of these steps, so nothing
    # else would ever have taken that state back.
    scheduler, _, _ = _answer_tail_offer(pool_refused=True)
    _quiet_whole_chunk_store(scheduler._req_status["req"])

    for step in range(1 + sched_mod._TAIL_STORE_RETRIES):
        meta = scheduler.build_connector_meta(
            _finished_step({"req"} if step == 0 else set())
        )
        assert not meta.store_jobs

    assert scheduler._tail_counts["store_pool_refused"] == 1
    assert scheduler._tail_counts["tail_retried"] == sched_mod._TAIL_STORE_RETRIES
    assert "req" not in scheduler._req_status
    assert scheduler._tail_offer_holds == set()
    _assert_nothing_half_built(scheduler)


def test_a_tail_whose_request_named_no_run_never_tags_a_run_of_its_own():
    # A request that keyed no full-attention chunk names no run. Storing its
    # tail under a tag of its own would leave a run of nothing but tail rows:
    # unrestorable, so the arena takes it first, and immortal, because nothing
    # can complete a tail's pseudo-boundary from inside.
    scheduler, _, _ = _answer_tail_offer(keyed_chunk=False)

    assert not scheduler._build_answer_tail_store_jobs()
    scheduler.manager.prepare_store_run.assert_not_called()
    assert scheduler._tail_counts == {"tail_offered": 1, "tail_run_missing": 1}
    assert scheduler._tail_run == {}
    _assert_nothing_half_built(scheduler)


def test_an_answer_tail_missing_a_group_is_refused_whole():
    # A tail that names no block for one of its groups cannot be restored, and
    # a restore missing a recurrent state is read silently rather than
    # reported, so half a tail is never stored.
    scheduler, _, _ = _answer_tail_offer(groups=(1,))

    assert not scheduler._build_answer_tail_store_jobs()
    scheduler.manager.prepare_store_run.assert_not_called()
    assert scheduler._tail_counts["hand_off_incomplete"] == 1
    _assert_nothing_half_built(scheduler)


def test_an_answer_tail_with_a_null_source_block_is_refused():
    # A null block carries no bytes: storing one writes residue under a key a
    # resume will read back as the state at the tail.
    scheduler, _, _ = _answer_tail_offer(null_group=1)

    assert not scheduler._build_answer_tail_store_jobs()
    scheduler.manager.prepare_store_run.assert_not_called()
    assert scheduler._tail_counts["null_source_block"] == 1
    _assert_nothing_half_built(scheduler)


def test_a_tail_store_holds_its_rows_only_for_the_copy():
    # The pool's pin is the only hold it offers a client and it outlives the
    # client, so a tail pins its rows for one copy and releases them when the
    # job ends. A row left pinned is skipped by the eviction order for ever.
    scheduler, handoff, blocks = _answer_tail_offer()
    jobs = scheduler._build_answer_tail_store_jobs()
    [job_id] = jobs
    _assert_one_row_per_key(scheduler, jobs)
    pinned = _tail_keys(handoff, sorted(blocks))
    scheduler.manager.pin.assert_called_once_with(pinned)
    assert scheduler._tail_counts["tail_pinned"] == len(pinned)

    scheduler.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                completed_jobs={job_id: 1}
            )
        )
    )

    scheduler.manager.unpin.assert_called_once_with(pinned)
    assert scheduler._tail_counts["tail_unpinned"] == len(pinned)
    assert "tail_pin_leaked" not in scheduler._tail_counts


def test_a_local_cache_reset_takes_the_tail_bookkeeping_with_it():
    """``/reset_prefix_cache`` without ``reset_connector`` wipes the core's
    block hashes and never asks the connector, and every piece of tail
    bookkeeping is named after those hashes: a queued offer names blocks the
    core has forgotten, and the run a tail belongs to, the keys the arena holds
    and the finished request an offer was keeping alive are all named after keys
    that no longer exist. The connector's own reset cleared all of it; the
    point here is that the *local* reset reaches the same surface, because the
    core tells the tier it offered its tails to."""
    scheduler, handoff, blocks = _answer_tail_offer()
    req_status = scheduler._req_status["req"]
    keys = _tail_keys(handoff, sorted(blocks))
    scheduler._note_tail_keys(req_status, keys)
    scheduler._note_in_pool(keys)
    req_status.finished_signaled = True
    scheduler._release_finished_request("req")
    assert scheduler.answer_tail_store.pending and scheduler._tail_offer_holds
    assert scheduler._tail_run and scheduler._pool_has

    # What the core's registry reset reaches: the tier the tails were offered
    # to is told once, and the connector's state goes with it.
    scheduler.answer_tail_store.reset()

    assert not scheduler.answer_tail_store.pending
    assert not scheduler._tail_offer_holds
    assert "req" not in scheduler._req_status
    assert not scheduler._tail_run and not scheduler._run_tails
    assert not scheduler._pool_has and not scheduler._state_sets


def test_a_local_cache_reset_leaves_a_running_request_alone():
    """A prefix-cache reset cannot free a block a running request holds, so the
    keys its chunks are named after survive it. Dropping that state would make
    the request re-offer chunks the arena already holds, and the reset is not
    allowed to cost that."""
    scheduler, handoff, blocks = _answer_tail_offer()
    req_status = scheduler._req_status["req"]
    keys = _tail_keys(handoff, sorted(blocks))
    scheduler._note_in_pool(keys)

    scheduler.answer_tail_store.reset()

    assert scheduler._req_status["req"] is req_status


def test_a_tail_pin_left_by_a_shutdown_is_released():
    # No completion arrives for a job the engine never finished, and the pool
    # does not release a client's pins when the client goes away: releasing here
    # is the last chance to leave no row the eviction order has to skip.
    scheduler, handoff, blocks = _answer_tail_offer()
    jobs = scheduler._build_answer_tail_store_jobs()
    _assert_one_row_per_key(scheduler, jobs)

    scheduler.shutdown()

    scheduler.manager.unpin.assert_called_once_with(_tail_keys(handoff, sorted(blocks)))
    assert scheduler._tail_pinned == {}
    assert scheduler._tail_counts["tail_unpinned"] == len(blocks)
    assert scheduler._tail_counts.get("tail_pin_leaked", 0) == 0


# --- a windowed group's rows: the store shape of a Flash-Next tail ---


def _windowed_tail_offer(
    sliding_window: int = 8192,
    block_size: int = 1600,
    **offer_kwargs,
):
    """A Flash-Next tail offered to a store: two full-attention groups and one
    sliding-window group, so the windowed group costs the blocks its window
    spans and the other two one block each.
    """
    return _answer_tail_offer(
        scheduler=_make_windowed_tail_scheduler(
            sliding_window=sliding_window, block_size=block_size
        ),
        **offer_kwargs,
    )


def _window_row_key(scheduler, block_idx: int, group_idx: int = 2) -> OffloadKey:
    """The key a windowed group's row is stored under: the key the block at
    ``block_idx`` owns at the position that block ends at, which is the key an
    aligned store of the same block would use."""
    group = next(
        g for g in scheduler.config.kv_group_configs if g.group_idx == group_idx
    )
    boundary = (block_idx + 1) * group.tokens_per_block
    hashes = scheduler._req_status["req"].req.block_hashes
    return make_offload_key(
        hashes[boundary // scheduler.config.tokens_per_hash - 1], group_idx
    )


def test_a_windowed_group_costs_its_window_of_rows_in_the_store():
    # A resume at the tail reads the whole window of a windowed group, so the
    # store takes a row for every block that window spans and pairs each with
    # the block its key names: here 1 + 1 + ceil(8192 / 1600) rows.
    scheduler, handoff, blocks = _windowed_tail_offer()
    jobs = scheduler._build_answer_tail_store_jobs()

    [job_id] = jobs
    _assert_one_row_per_key(scheduler, jobs)
    src = jobs[job_id].src_spec
    assert list(src.group_sizes) == [1, 1, 6]
    assert src.block_ids.tolist() == [
        block for group in sorted(blocks) for block in blocks[group]
    ]
    # The window's rows are the six blocks ending at the tail block, so the
    # group's transfer starts at its third block and runs six on.
    assert src.block_indices[2] == 2
    # The pool numbers its rows in the order it was offered its keys, so the
    # rows the job hands the worker pair with those blocks one for one.
    assert jobs[job_id].dst_spec.block_ids.tolist() == list(range(8))
    (offered, tag), _ = scheduler.manager.prepare_store_run.call_args
    assert tag == _tail_run_tag(scheduler)
    tail_keys = _tail_keys(handoff, sorted(blocks))
    assert sorted(k for k in offered if k in tail_keys) == sorted(tail_keys)
    assert len(offered) - len(tail_keys) == 5  # the window's other rows
    assert all(scheduler._tail_run[k] == tag for k in tail_keys)


def test_a_window_row_the_pool_already_holds_is_not_stored_twice():
    # A window row carries the key its block owns, the same key an aligned
    # store of that block uses, so the pool's own dedup is the tail's: the row
    # is taken once, the rows that are left stay paired with the blocks their
    # keys name, and a chunk key is never recorded as a tail.
    scheduler, _, blocks = _windowed_tail_offer()
    # The window is the six blocks ending at the tail block, so its third row
    # is the block three places before that one.
    resident = _window_row_key(scheduler, 4)
    scheduler.manager.prepare_store_run = MagicMock(
        side_effect=lambda keys, tag, whole=False, held=False, tails=None: (
            _cpu_store_output([key for key in keys if key != resident])
        )
    )
    jobs = scheduler._build_answer_tail_store_jobs()

    [job_id] = jobs
    _assert_one_row_per_key(scheduler, jobs)
    src = jobs[job_id].src_spec
    assert list(src.group_sizes) == [1, 1, 5]
    kept = blocks[2][:2] + blocks[2][3:]
    assert src.block_ids.tolist() == blocks[0] + blocks[1] + kept
    assert jobs[job_id].dst_spec.block_ids.tolist() == list(range(7))
    assert resident not in scheduler._jobs[job_id].keys
    assert resident not in scheduler._tail_run


def test_a_window_that_does_not_divide_the_block_rounds_the_rows_up():
    # 5000 tokens of window at 1600 tokens per block is 3.125 blocks, and the
    # store pays for the fourth because a resume reads it whole.
    scheduler, _, _ = _windowed_tail_offer(sliding_window=5000)
    assert scheduler.config.kv_group_configs[2].tail_rows == 4

    jobs = scheduler._build_answer_tail_store_jobs()

    [job_id] = jobs
    _assert_one_row_per_key(scheduler, jobs)
    assert list(jobs[job_id].src_spec.group_sizes) == [1, 1, 4]
    assert len(scheduler._jobs[job_id].keys) == 6


def test_a_tail_that_names_fewer_rows_than_a_group_costs_is_refused_whole():
    # A tail naming three of the four rows its windowed group costs is half a
    # tail, and a restore missing a window row reads the wrong KV and reports
    # nothing. The pool is asked for nothing and nothing is left half-built.
    scheduler, _, _ = _windowed_tail_offer(sliding_window=5000, rows_override={2: 3})

    assert not scheduler._build_answer_tail_store_jobs()
    scheduler.manager.prepare_store_run.assert_not_called()
    assert scheduler._tail_counts["hand_off_incomplete"] == 1
    _assert_nothing_half_built(scheduler)


def test_a_group_the_store_cannot_size_refuses_the_tail_before_asking_the_pool():
    # A row count the specs do not give is not guessed at: a tail stored at a
    # shape borrowed from another layout costs rows no restore can read back,
    # which is the defect this row count exists to close.
    scheduler, _, _ = _windowed_tail_offer()
    groups = list(scheduler.config.partial_tail_groups)
    groups[2] = groups[2]._replace(tail_rows=None)
    scheduler.config = scheduler.config._replace(partial_tail_groups=tuple(groups))

    assert not scheduler._build_answer_tail_store_jobs()
    scheduler.manager.prepare_store_run.assert_not_called()
    assert scheduler._tail_counts["window_rows_unresolved"] == 1
    _assert_nothing_half_built(scheduler)


def test_a_retry_revalidates_every_row_the_offer_named():
    # The blocks a retried offer named went back to the free queue with the
    # finished turn, and a window row is keyed by the key its own block owns,
    # so the tail keys are not the only ones at risk. A row whose block moved
    # holds someone else's bytes, and the tail is given up on whole.
    scheduler, _, _ = _windowed_tail_offer(pool_refused=True)
    [offer] = scheduler.answer_tail_store.take_offers()
    rows = scheduler._tail_store_rows(offer, scheduler._req_status["req"])
    assert scheduler.answer_tail_store.requeue(offer)
    gpu = {}
    for row in rows:
        block = KVCacheBlock(row.block_id)
        block.set_block_hash(
            make_block_hash_with_group_id(
                BlockHash(bytes(row.key)[:-4]), row.group.group_idx
            ),
            12,
        )
        gpu[row.block_id] = block
    scheduler.bind_gpu_block_pool(FakeGPUPool(list(gpu.values())))

    # Step N: the pool is full, and every row is still where the offer named it.
    assert not scheduler._build_answer_tail_store_jobs()
    assert scheduler._tail_counts["tail_retried"] == 1

    # A new owner takes the block a window row named, and its key with it.
    moved = next(row for row in rows if not row.is_tail)
    gpu[moved.block_id].reset_hash()
    gpu[moved.block_id].set_block_hash(
        make_block_hash_with_group_id(BlockHash(b"owner"), moved.group.group_idx), 12
    )

    # Step N+2: the retry sees the moved row and gives the tail up.
    assert not scheduler._build_answer_tail_store_jobs()
    assert scheduler._tail_counts["stale_offer"] == 1
    assert scheduler._tail_counts["tail_stored"] == 0
    assert scheduler.manager.prepare_store_run.call_count == 1
    assert scheduler.answer_tail_store.pending == {}
    _assert_nothing_half_built(scheduler)


def test_the_restore_of_a_tail_is_refused_by_name():
    # A tail that comes back missing a group, or carrying a state that is not
    # the one the resume position needs, is read silently. So nothing is copied
    # until every row is proven held, which leaves no half tail to reach a
    # request, and a restore that was not earned is counted rather than hidden.
    scheduler, handoff, _ = _answer_tail_offer()
    hit = AnswerTailHit(
        tail_key=handoff.tail_hash,
        resume_token=handoff.resume_token,
        group_block_ids={},
        num_tokens=30,
    )

    assert scheduler.answer_tail_store.lookup_tail(handoff.tail_hash, [0, 1]) is None
    # No destination block is no restore: the caller fenced a resume it could not
    # fill, and counting one would report a tail restored that no request got.
    assert scheduler.answer_tail_store.start_load(hit, {}) is False
    assert scheduler.answer_tail_store.wait(hit) is False
    assert scheduler._tail_counts["tail_restore_refused"] == 0


# --- the restore route: the core records a tail, the connector fences it ---


def _tail_resume_layout(
    resume_token: int = 29,
    rows: tuple[OffloadKey, ...] | None = None,
    generation: int | None = None,
):
    """A request whose own chunks stop at the last proved boundary, and the tail
    the tier answered for the token beyond it: the scheduler, its request state,
    the record the core left, and the rows a restore has to read.

    The arena answers HIT for every key, and the rows are marked in the band's
    mirror as a deployment that mirrors the pool would have them. A test that
    wants one without the other says so itself.
    """
    scheduler = _make_drafter_partial_tail_scheduler(True)
    request = _make_partial_tail_request(scheduler, num_prompt_tokens=32)
    req_status = scheduler._req_status["req"]
    req_status.num_locally_computed_tokens = 0
    req_status.update_offload_keys()
    scheduler.manager.lookup.return_value = LookupResult.HIT
    if rows is None:
        rows = tuple(
            make_offload_key(b"tail", group.group_idx)
            for group in scheduler.config.partial_tail_groups
        )
    scheduler._tail_row_keys[b"tail"] = rows
    scheduler._note_in_pool(rows)
    record = AnswerTailHit(
        tail_key=BlockHash(b"tail"),
        resume_token=resume_token,
        group_block_ids={},
        num_tokens=30,
        generation=(
            scheduler._cache_generation if generation is None else generation
        ),
    )
    request.tail_resume = record
    return scheduler, req_status, record


def test_a_stored_tail_records_the_rows_a_restore_has_to_read():
    # The tail key names only a windowed group's last row, so the store writes
    # down every row it wrote. Without that record a restore would be granted
    # over the rows the tail key happens to name and read the rest as whatever
    # the arena holds under some other key.
    scheduler, handoff, _ = _answer_tail_offer()
    scheduler._build_answer_tail_store_jobs()

    rows = scheduler._tail_row_keys[bytes(handoff.tail_hash)]
    tail_groups = [group.group_idx for group in scheduler.config.partial_tail_groups]
    assert list(rows) == _tail_keys(handoff, sorted(tail_groups))


def test_a_tail_the_arena_holds_is_answered_for_every_group_it_carries():
    # The tier answers "these rows are here" and nothing else; a probe that
    # leaves out a group the tail carries would resume without that group's
    # state, so the answer is no.
    scheduler, _, record = _tail_resume_layout()
    groups = [group.group_idx for group in scheduler.config.partial_tail_groups]

    hit = scheduler.answer_tail_store.lookup_tail(record.tail_key, groups)
    assert hit is not None
    assert hit.generation == scheduler._cache_generation
    assert scheduler.answer_tail_store.lookup_tail(record.tail_key, groups[:1]) is None


def test_a_resume_the_fence_grants_adds_its_token_to_the_hit():
    # A tail resumes one token past the last boundary the request's own chunks
    # prove, which is a token the request would otherwise recompute. The grant
    # is the deeper of the two routes, so the boundary the connector records is
    # the resume's, not the aligned one's.
    scheduler, req_status, record = _tail_resume_layout()

    assert scheduler._lookup(req_status) == record.resume_token
    assert req_status.partial_tail_boundary == record.resume_token
    assert req_status.tail_resume is record
    assert scheduler._tail_counts["tail_restore_granted"] == 1
    assert scheduler._tail_counts["tail_restore_refused"] == 0


def test_a_granted_resume_is_loaded_under_the_tail_key_into_its_own_block():
    # The row at a resume is the tail's own, keyed by a hash no block of the
    # resuming request owns, so the load asks for the tail key where an aligned
    # boundary would ask for the block's. One key, one destination block, in the
    # same group order: the pairing the transfer layer depends on is the test.
    scheduler, req_status, record = _tail_resume_layout()
    request = req_status.req
    drafter_group_idx = 2
    mamba_group_idx = 1
    assert scheduler._lookup(req_status) == record.resume_token

    scheduler.update_state_after_alloc(
        request,
        KVCacheBlocks(
            (
                [KVCacheBlock(31), KVCacheBlock(32)],
                [KVCacheBlock(0, is_null=True), KVCacheBlock(41)],
                [KVCacheBlock(51), KVCacheBlock(52)],
            )
        ),
        num_external_tokens=record.resume_token,
    )

    [load_job] = scheduler._current_batch_load_jobs.values()
    dst_spec = load_job.dst_spec
    assert isinstance(dst_spec, GPULoadStoreSpec)
    [job] = scheduler._jobs.values()
    assert dst_spec.group_sizes == [2, 1, 1]
    assert sum(dst_spec.group_sizes) == len(dst_spec.block_ids) == len(job.keys)
    tail_keys = [
        make_offload_key(b"tail", group.group_idx)
        for group in scheduler.config.partial_tail_groups
    ]
    assert all(key in job.keys for key in tail_keys)
    # The boundary key belongs to the aligned route only: asking for both would
    # load two rows into the block at the resume position.
    boundary = make_offload_key(BlockHash(b"h6"), 0)
    assert boundary not in job.keys
    assert scheduler._tail_counts["tail_restored"] == 1
    assert scheduler._tail_counts["tail_restore_refused"] == 0
    # Consumed: a second allocation for the same request cannot load it again.
    assert req_status.tail_resume is None
    assert req_status.partial_tail_boundary is None


def test_a_resume_that_fills_no_destination_row_is_not_counted_as_a_restore():
    # A tail group whose block at the resume position the GPU already holds
    # fills no destination row, so that group's row never reaches the load. The
    # tier says so, and the count says it for a load that filled some rows but
    # not every group's: a restore counted here would report a tail handed to a
    # request that was handed part of one.
    scheduler, req_status, record = _tail_resume_layout()
    assert scheduler._lookup(req_status) == record.resume_token

    scheduler.update_state_after_alloc(
        req_status.req,
        KVCacheBlocks(
            (
                [KVCacheBlock(31), KVCacheBlock(32)],
                # The recurrent group's blocks are both null: nothing here to
                # copy a row into.
                [KVCacheBlock(0, is_null=True), KVCacheBlock(0, is_null=True)],
                [KVCacheBlock(51), KVCacheBlock(52)],
            )
        ),
        num_external_tokens=record.resume_token,
    )

    assert scheduler._tail_counts["tail_restored"] == 0
    assert scheduler._tail_counts["tail_restore_refused"] == 1
    assert req_status.req.tail_resume is None


def test_a_resume_retried_over_the_same_blocks_is_one_restore():
    # A deferred lookup is retried, and the retry builds the same load into the
    # same blocks. The rows are on their way already, so the retry joins them:
    # counting twice would report a tail restored that was restored once.
    scheduler, _, record = _tail_resume_layout()
    dests = {0: [32], 1: [41]}

    assert scheduler.answer_tail_store.start_load(record, dests) is True
    assert scheduler.answer_tail_store.start_load(record, dests) is True
    assert scheduler._tail_counts["tail_restored"] == 1
    # Other blocks are other rows: a second request resuming the same tail needs
    # its own copy, because the copy lands in the blocks it was allocated.
    assert scheduler.answer_tail_store.start_load(record, {0: [99]}) is True
    assert scheduler._tail_counts["tail_restored"] == 2


def test_a_resume_the_arena_cannot_fill_is_refused_and_claims_nothing():
    # The fence is all or nothing, and it is decided before the scheduler
    # allocates: a resume that loaded some rows would hand the request a state
    # that is not the one at the position it restarts from, and nothing reports
    # that. So a row the arena lost costs the request its recompute and nothing
    # else. It is the arena that has to be short of the row, not the band's
    # mirror, which decides nothing here.
    scheduler, req_status, record = _tail_resume_layout()
    lost = scheduler._tail_row_keys[b"tail"][-1]
    scheduler.manager.lookup.side_effect = lambda key, ctx: (
        LookupResult.MISS if key == lost else LookupResult.HIT
    )

    assert scheduler._lookup(req_status) == 28
    assert scheduler._tail_counts["tail_restore_refused"] == 1
    assert scheduler._tail_counts["tail_restore_granted"] == 0
    # The aligned boundary the request proved for itself stands; only the
    # resume's extra token is gone.
    assert req_status.tail_resume is None
    assert req_status.partial_tail_boundary != record.resume_token
    assert req_status.req.tail_resume is None
    assert not scheduler._current_batch_load_jobs


def test_the_fence_asks_the_arena_before_it_grants_a_resume():
    # The band's view of the pool is drained once per step, inside the band
    # builder, which runs after this lookup: on its own it can still believe in
    # a row the arena dropped, and a row that is not ready by the time the load
    # asks for it ends the engine. So the arena is the last word, and it is
    # asked for every row -- which is also the hold that keeps those rows for
    # the step this decision is made in.
    scheduler, req_status, _ = _tail_resume_layout()
    rows = scheduler._tail_row_keys[b"tail"]
    lost = rows[-1]
    scheduler.manager.lookup.side_effect = lambda key, ctx: (
        LookupResult.MISS if key == lost else LookupResult.HIT
    )

    # The aligned boundary the request proved for itself stands; only the
    # resume's extra token goes.
    assert scheduler._lookup(req_status) == 28
    assert scheduler._tail_counts["tail_restore_refused"] == 1
    assert scheduler._tail_counts["tail_restore_granted"] == 0
    assert req_status.tail_resume is None
    assert not scheduler._current_batch_load_jobs
    looked_up = [call.args[0] for call in scheduler.manager.lookup.call_args_list]
    assert all(row in looked_up for row in rows)


def test_an_empty_pool_view_refuses_no_resume_the_arena_holds():
    # The band's mirror of the pool is filled by the completion path only where
    # a deployment writes back, so the default deployment has an empty one and
    # pre-filtered every resume away on it, whatever the arena would have
    # answered. The arena is the last word, and it is asked for every row.
    scheduler, req_status, record = _tail_resume_layout()
    scheduler._pool_has.clear()
    scheduler, req_status, record = _tail_resume_layout()
    scheduler._pool_has.clear()

    assert scheduler._lookup(req_status) == record.resume_token
    assert scheduler._tail_counts["tail_restore_granted"] == 1
    assert scheduler._tail_counts["tail_restore_refused"] == 0


def test_a_resume_record_of_a_cache_that_is_gone_is_refused():
    # A reset wipes the prefix cache without asking the tier, and every key the
    # record names went with it. Honouring one would read rows under keys that
    # now belong to a different conversation.
    scheduler, req_status, record = _tail_resume_layout()
    scheduler._forget_local_cache()

    assert scheduler._lookup(req_status) == 28
    assert scheduler._tail_counts["tail_restore_refused"] == 1
    assert req_status.tail_resume is None


def test_a_resume_past_what_the_load_can_reach_is_refused():
    # A resume the chunks below cannot reach leaves a hole the request believes
    # computed. The reach is the one the aligned boundary shares, so a tail that
    # resumes past the last token the request can be given is not this load's.
    scheduler, req_status, _ = _tail_resume_layout(resume_token=32)

    assert scheduler._lookup(req_status) == 28
    assert scheduler._tail_counts["tail_restore_refused"] == 1
    assert req_status.tail_resume is None


def test_a_tail_whose_window_needs_more_rows_than_a_resume_fills_is_refused():
    # A windowed group's tail costs the window's worth of rows, and a resume has
    # one destination block for a group. Half a window is the wrong state to
    # hand back, so that layout is refused by name until a restore can size it.
    scheduler, req_status, _ = _tail_resume_layout(
        rows=(make_offload_key(b"tail", 0), make_offload_key(b"other", 0))
    )
    scheduler._forget_in_pool([make_offload_key(b"other", 0)])
    scheduler._note_in_pool([make_offload_key(b"other", 0)])

    assert scheduler._lookup(req_status) == 28
    assert scheduler._tail_counts["tail_restore_refused"] == 1


def test_an_aligned_boundary_deeper_than_a_resume_is_kept_and_counts_nothing():
    # A tail is not a claim to be the deepest hit: where the request's own
    # chunks prove a boundary as deep, that route is taken and the record costs
    # nothing, so the counters say no restore happened.
    scheduler, req_status, record = _tail_resume_layout(resume_token=28)

    assert scheduler._lookup(req_status) == 28
    assert req_status.tail_resume is None
    assert scheduler._tail_counts["tail_restore_granted"] == 0
    assert scheduler._tail_counts["tail_restore_refused"] == 0


def test_a_resume_is_refused_for_a_request_that_is_gone():
    # The record is made while the request is being scheduled and the fence runs
    # whenever the lookup does; a request that has since finished has no
    # destination for the rows and no one to read them.
    scheduler, req_status, _ = _tail_resume_layout()
    req_status.req.is_finished.return_value = True

    assert scheduler._lookup(req_status) == 28
    assert scheduler._tail_counts["tail_restore_refused"] == 1
    assert req_status.tail_resume is None


def test_a_tail_another_engine_stored_is_granted_over_the_rows_its_key_names():
    """The rows a tail was written as were recorded by the store that wrote
    them, so a brand-new engine has no record of them and refused every resume
    for a conversation its predecessor saved. The tail key names one row per
    group wherever they were written, so the fence derives them and asks the
    arena, which is the same proof the store's own record would have given."""
    scheduler, req_status, record = _tail_resume_layout()
    scheduler._tail_row_keys.clear()
    scheduler._pool_has.clear()
    scheduler._pool_tail_keys = {b"tail"}
    rows = tuple(
        make_offload_key(b"tail", group.group_idx)
        for group in scheduler.config.partial_tail_groups
    )

    assert scheduler._lookup(req_status) == record.resume_token

    assert scheduler._tail_counts["tail_restore_granted"] == 1
    assert scheduler._tail_counts["tail_restore_refused"] == 0
    # The arena was asked, not assumed, and what it confirmed is now on file.
    looked_up = [call.args[0] for call in scheduler.manager.lookup.call_args_list]
    assert all(row in looked_up for row in rows)
    assert scheduler._tail_row_keys[b"tail"] == rows
    # The pool view is marked as well, on the arena's own answer. Without it the
    # next request of the same conversation is answered from a record of rows
    # this process never stored, which is to say: not at all.
    assert all(row in scheduler._pool_has for row in rows)


def test_a_tail_another_engine_named_is_refused_when_the_arena_lacks_a_row():
    """Deriving the rows is a guess about the layout, not a claim about the
    pool: one row the arena will not confirm and the resume is refused whole,
    because a tail that comes back missing a group reads a wrong state and
    reports nothing. Nothing is adopted from a refused confirmation either."""
    scheduler, req_status, record = _tail_resume_layout()
    scheduler._tail_row_keys.clear()
    scheduler._pool_has.clear()
    scheduler._pool_tail_keys = {b"tail"}
    lost = make_offload_key(
        b"tail", scheduler.config.partial_tail_groups[-1].group_idx
    )
    scheduler.manager.lookup.side_effect = lambda key, ctx: (
        LookupResult.MISS if key == lost else LookupResult.HIT
    )

    assert scheduler._lookup(req_status) < record.resume_token

    assert scheduler._tail_counts["tail_restore_refused"] == 1
    assert scheduler._tail_counts["tail_restore_granted"] == 0
    assert b"tail" not in scheduler._tail_row_keys
    assert not scheduler._pool_has


def test_the_tier_answers_for_a_tail_this_process_never_stored():
    """The core asks whether the rows are here before it leaves a resume on the
    request, and a fresh connector has never stored them: the pool's own record
    of its tails is all there is to answer from, and a probe that leaves out a
    group the tail carries is still no."""
    scheduler, _, record = _tail_resume_layout()
    scheduler._tail_row_keys.clear()
    groups = [group.group_idx for group in scheduler.config.partial_tail_groups]

    assert scheduler.answer_tail_store.lookup_tail(record.tail_key, groups) is None

    scheduler._pool_tail_keys = {b"tail"}
    hit = scheduler.answer_tail_store.lookup_tail(record.tail_key, groups)
    assert hit is not None
    assert hit.generation == scheduler._cache_generation
    assert scheduler.answer_tail_store.lookup_tail(record.tail_key, groups[:1]) is None


def test_the_tail_candidates_come_from_the_pool_and_are_asked_for_once_a_refresh():
    """A probe runs on every lookup of every waiting request, so the pool is
    asked at most once per refresh and every probe in between reads the copy:
    the round trip is what must not sit on the scheduler's critical path. The
    empty parent is the wire form of a tail that starts the conversation."""
    scheduler = _make_drafter_partial_tail_scheduler(True)
    parent = BlockHash(b"parent")
    scheduler.manager.tail_index = MagicMock(
        return_value={b"parent": ((12, b"tail"),), b"": ((7, b"root"),)}
    )

    found = scheduler._pool_tail_lengths([None, parent, BlockHash(b"other")])

    assert found == {None: (7,), parent: (12,)}
    assert scheduler._pool_tail_keys == {b"tail", b"root"}
    assert scheduler.manager.tail_index.call_count == 1
    assert scheduler._pool_tail_lengths([parent]) == {parent: (12,)}
    assert scheduler.manager.tail_index.call_count == 1
    assert scheduler._tail_counts["tail_index_unsupported"] == 0


def test_a_pool_that_keeps_no_tail_record_leaves_the_route_as_it_was():
    """Production keeps the server that predates the record until the roll, so
    the new question has to be answered by a pool that never heard of it. The
    connector then knows only the tails it stored itself, exactly as before, and
    says so once rather than once per probe."""
    scheduler = _make_drafter_partial_tail_scheduler(True)
    scheduler.manager.tail_index = MagicMock(return_value=None)

    assert scheduler._pool_tail_lengths([BlockHash(b"parent")]) == {}
    assert scheduler._pool_tail_lengths([BlockHash(b"parent")]) == {}

    assert scheduler._tail_counts["tail_index_unsupported"] == 1
    assert scheduler._pool_tail_keys == set()


def test_a_replayed_turn_relearns_a_tail_the_arena_still_holds(monkeypatch):
    """The bench's own sequence, end to end: store a tail, a bare
    ``/reset_prefix_cache``, the turn replayed and its tail offered again, then
    a later resume. The arena already holds the rows, so the re-offer takes none
    of them back; unless that counts as a store the band records, the reset
    dropped the rows a restore is granted over and the tail can never be
    resumed again for as long as the arena keeps holding it."""
    # The completion path fills the band's mirror of the pool only under
    # write-back, so the run is the one a rig does.
    monkeypatch.setattr(sched_mod, "WRITE_BACK", True)
    scheduler = _make_drafter_partial_tail_scheduler(True)
    request = _make_partial_tail_request(scheduler, num_prompt_tokens=32)
    req_status = scheduler._req_status["req"]
    req_status.num_locally_computed_tokens = 0
    req_status.update_offload_keys()
    handoff = AnswerTailHandoff(
        request_id="req",
        tail_hash=BlockHash(b"tail"),
        parent=None,
        length=12,
        resume_token=29,
    )
    blocks = {
        group.group_idx: [
            100 + 10 * group.group_idx + row
            for row in range(group.tail_rows)
        ]
        for group in scheduler.config.partial_tail_groups
    }

    def offer() -> None:
        for group_idx, block_ids in blocks.items():
            assert scheduler.answer_tail_store.store_tail(
                handoff, group_idx, block_ids
            )

    offer()
    scheduler.manager.prepare_store_run = MagicMock(
        side_effect=_takes_every_key
    )
    jobs = scheduler._build_answer_tail_store_jobs()
    _assert_one_row_per_key(scheduler, jobs)
    [job_id] = jobs
    scheduler.update_connector_output(
        KVConnectorOutput(
            kv_connector_worker_meta=OffloadingWorkerMetadata(
                completed_jobs={job_id: 1}
            )
        )
    )
    rows = scheduler._tail_row_keys[bytes(handoff.tail_hash)]
    assert rows and all(key in scheduler._pool_has for key in rows)
    assert scheduler._tail_counts["tail_stored"] == 1

    # The bare reset: the core's hashes go, and the connector's record of which
    # rows the arena holds is named after them.
    scheduler.answer_tail_store.reset()
    assert not scheduler._tail_row_keys and not scheduler._pool_has

    # The replayed turn finishes at the same tail and offers it again. The arena
    # holds every row and says so, and its answer names no row to write.
    scheduler.manager.lookup.return_value = LookupResult.HIT
    offer()
    scheduler.manager.prepare_store_run = MagicMock(
        return_value=PrepareStoreOutput(
            keys_to_store=[], store_spec=CPULoadStoreSpec([]), evicted_keys=[]
        )
    )
    assert not scheduler._build_answer_tail_store_jobs()
    assert scheduler._tail_row_keys[bytes(handoff.tail_hash)] == rows
    assert all(key in scheduler._pool_has for key in rows)
    assert scheduler._tail_counts["tail_stored"] == 2

    # A later resume of the same conversation is granted again.
    request.tail_resume = AnswerTailHit(
        tail_key=BlockHash(b"tail"),
        resume_token=29,
        group_block_ids={},
        num_tokens=30,
        generation=scheduler._cache_generation,
    )
    assert scheduler._lookup(req_status) == 29
    assert scheduler._tail_counts["tail_restore_granted"] == 1

    # The heal reaches as far as the arena's answer and no further. Another reset
    # drops the record, and this time one row is not there to be found: the tail
    # is recorded again, because which rows a tail is made of is a fact about the
    # offer, while the pool view is only ever the arena's. A resume is then
    # refused rather than granted on a belief nobody confirmed.
    scheduler.answer_tail_store.reset()
    lost = rows[-1]
    scheduler.manager.lookup.side_effect = lambda key, ctx: (
        LookupResult.MISS if key == lost else LookupResult.HIT
    )
    offer()
    assert not scheduler._build_answer_tail_store_jobs()
    assert scheduler._tail_row_keys[bytes(handoff.tail_hash)] == rows
    assert not scheduler._pool_has
    # And it is not counted either: the mirror was left unmarked, so no restore
    # can be granted over this tail, and a store counted here would be a tail
    # reported as held that the arena refused to confirm.
    assert scheduler._tail_counts["tail_stored"] == 2
    request.tail_resume = AnswerTailHit(
        tail_key=BlockHash(b"tail"),
        resume_token=29,
        group_block_ids={},
        num_tokens=30,
        generation=scheduler._cache_generation,
    )
    assert scheduler._lookup(req_status) == 28
    assert scheduler._tail_counts["tail_restore_granted"] == 1
    assert scheduler._tail_counts["tail_restore_refused"] == 1


def test_the_answer_tail_store_is_installed_only_with_the_flag():
    # The core offers a tail to whatever store its answer tail holds, so the
    # install is the seam a flag-off deployment must not cross.
    for enabled in (False, True):
        scheduler = _make_drafter_partial_tail_scheduler(enabled)
        answer_tail = SimpleNamespace(store=None)
        scheduler.bind_answer_tail_store(SimpleNamespace(answer_tail=answer_tail))
        assert (answer_tail.store is scheduler.answer_tail_store) is enabled


def test_an_offered_answer_tail_is_inert_without_the_flag():
    # Flag off, no core brings a tail here; should one arrive anyway, nothing is
    # stored, nothing is counted and nothing is left queued for a later step to
    # act on. The load side is asked through the same reporter the store side
    # uses, so a refused restore is silent here too: this is the third sighting
    # of the class, after the layout refusals and the assert-to-refuse helper
    # each leaked a series into a flag-off engine.
    scheduler = _make_drafter_partial_tail_scheduler(False)
    scheduler.manager.prepare_store_run = MagicMock()
    handoff = AnswerTailHandoff(
        request_id="req",
        tail_hash=BlockHash(b"tail"),
        parent=None,
        length=12,
        resume_token=29,
    )
    scheduler.answer_tail_store.store_tail(handoff, 0, [100])

    assert not scheduler._build_answer_tail_store_jobs()
    scheduler.manager.prepare_store_run.assert_not_called()
    assert dict(scheduler._tail_counts) == {}
    assert scheduler.answer_tail_store.pending == {}
    # The record a re-offer of a resident tail writes is written nowhere here.
    assert not scheduler._tail_row_keys and not scheduler._pool_has

    hit = AnswerTailHit(
        tail_key=handoff.tail_hash,
        resume_token=handoff.resume_token,
        group_block_ids={},
        num_tokens=30,
    )
    assert scheduler.answer_tail_store.lookup_tail(handoff.tail_hash, [0, 1]) is None
    assert scheduler.answer_tail_store.start_load(hit, {}) is False
    assert scheduler.answer_tail_store.wait(hit) is False
    assert dict(scheduler._tail_counts) == {}
    stats = scheduler.get_stats()
    series = stats.data["data"] if stats is not None else {}
    assert sched_mod._ConnectorMetricName.PARTIAL_TAIL_EVENTS not in series


# --- the marker positive control: a group's bytes, in the row its key names ---
#
# What this section proves is not that a row round-trips, which the pool's own
# tests do, but that the connector paired the right source block with the right
# destination row for every group. A permutation applied to the store and to
# the restore alike leaves every key, every count and the one-row-per-key
# invariant intact and hands the resuming request another group's recurrent
# state, because a recurrent state carries no group label. That is the residual
# byte equality never closed (gpu-gate-1.md 7.2, key-contract.md 8).
#
# Three things here stand in for something else, and the claim stops at each.
# The copy is the CPU transcription of the worker's per-group walk
# (``SingleDirectionOffloadingHandler.transfer_async``), so what is under test
# is the pairing the job's two specs carry, not the CUDA copy engine or the
# per-layer page layout below it. And the rows a restore reads that are not the
# tail's own, the conversation's chunk rows, are seeded by the harness because
# no producer ran here; every tail row is the store's, end to end. And the
# consumer restores over the producer's store record, which only the engine
# that stored the tail ever writes, so that one record stands in for a store
# this consumer did not run.
#
# What the control cannot see, by construction, is disclosed rather than
# claimed: every group's row is the same harness-chosen ROW_BYTES, so a wrong
# per-group row size is invisible here, and a ``group_sizes`` that preserves the
# flat order, [2, 0] where [1, 1] is right, pairs the same rows in this copy and
# in the worker because both partition by that same field. In production such a
# value lands in the wrong byte plan; here the two sides move together, so no
# fault of that class can show up as a mismatch.

ROW_BYTES = 1024
ARENA_SUPERPAGES = 16
# Group indices a marker is defined for: the offloaded groups of the layout
# under test, with room for one the control never sees.
MARKER_GROUPS = 4


def _group_marker(group_idx: int) -> bytes:
    """What one group's row must carry: its own index in the first four bytes
    and a fill byte no other group uses, so a row that came back with another
    group's state, or with nobody's, says which it is."""
    fill = bytes([0x60 + group_idx])
    return group_idx.to_bytes(4, "big") + fill * (ROW_BYTES - 4)


def _row_payload(key: OffloadKey) -> bytes:
    """The bytes of a row the conversation's own store wrote: named after the
    row's key, and never a group's marker."""
    filler = b"row:" + bytes(key)
    return (filler * (ROW_BYTES // len(filler) + 1))[:ROW_BYTES]


def _marker_owner(payload: bytes) -> int | None:
    """Which group's marker a row carries, or None when it carries nobody's."""
    for group_idx in range(MARKER_GROUPS):
        if payload == _group_marker(group_idx):
            return group_idx
    return None


class _StandInBlocks:
    """A byte array standing in for GPU memory, one ROW_BYTES span per block
    id. The real copy reads a KV tensor per layer; all the pairing under test
    can reach is the ROW_BYTES span an arena row is paired with, so that span
    is all this holds."""

    def __init__(self) -> None:
        self.landed: dict[int, bytes] = {}

    def seed(self, block_id: int, payload: bytes) -> None:
        self.landed[block_id] = payload

    def read(self, block_id: int) -> bytes:
        return self.landed[block_id]

    def write(self, block_id: int, payload: bytes) -> None:
        self.landed[block_id] = payload


class MarkerPool:
    """A real arena over a real ``mmap``, answering the connector as a shared
    pool would.

    The connector keeps its mock manager and this replaces the answers a pool
    gives it, so every row, hold and completion below is the arena's own answer
    to the connector's own call. Rows are read and written through the mapping,
    which is what makes where a byte landed a fact about the arena rather than a
    number in a table the test wrote.
    """

    def __init__(self, tmp_path, state_groups: tuple[int, ...]) -> None:
        path = str(tmp_path / "arena.mmap")
        size = ARENA_SUPERPAGES * ROW_BYTES
        fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
        os.ftruncate(fd, size)
        self.arena = pool_server.Arena(path, size, ROW_BYTES, quarantine_sec=0.0)
        self.arena.open_file(fd, mmap.mmap(fd, size))
        self.cls = self.arena.class_for(
            "marker", {}, "L-marker", ROW_BYTES, state_groups
        )
        self.rows = _ArenaRows(self)
        self.clients: list = []

    def close(self) -> None:
        self.arena.close()

    def bind(self, scheduler) -> None:
        """Attach one more engine: the connector's pool calls answer from here."""
        client = pool_server.ClientState(
            client_id=len(self.clients), name=f"engine{len(self.clients)}", cls=self.cls
        )
        self.clients.append(client)
        self.cls.clients += 1
        manager = scheduler.manager
        manager.prepare_store = lambda keys, ctx: self._prepare_store(
            keys, client, None
        )
        manager.prepare_store_run = (
            lambda keys, tag, whole=False, held=False, tails=None: self._prepare_store(
                keys, client, bytes(tag), whole, held, tails
            )
        )
        manager.complete_store = lambda keys, ctx, success=True: (
            self.arena.complete_store(client, [bytes(k) for k in keys], success)
        )
        manager.lookup = lambda key, ctx: self._lookup(key, client)
        manager.prepare_load = lambda keys, ctx: self._prepare_load(keys, client)
        manager.complete_load = lambda keys, ctx: self.arena.complete_load(
            client, [bytes(k) for k in keys]
        )
        manager.pin = lambda keys: self.arena.pin(self.cls, [bytes(k) for k in keys])
        manager.unpin = lambda keys: self.arena.unpin(
            self.cls, [bytes(k) for k in keys]
        )
        manager.on_schedule_end = lambda ctx: self.arena.end_session(client)

    def read_row(self, row: int) -> bytes:
        start = self.cls.offset(row)
        return bytes(self.arena.mm[start : start + ROW_BYTES])

    def write_row(self, row: int, payload: bytes) -> None:
        start = self.cls.offset(row)
        self.arena.mm[start : start + ROW_BYTES] = payload

    def row_of(self, key: OffloadKey) -> int:
        return self.cls.index[bytes(key)].row

    def _prepare_store(self, keys, client, tag, whole=False, held=False, tails=None):
        out = self.arena.prepare_store(
            client,
            [bytes(k) for k in keys],
            tag if tag is not None else client.name,
            whole=whole,
            held=held,
            tails=tails,
        )
        if out is None:
            return None
        new, rows, evicted = out
        # The control sizes its arena so nothing is reclaimed under it: an
        # evicted row would take a marker with it and read as a pairing fault.
        assert not evicted, "the arena evicted a row mid-control"
        return PrepareStoreOutput(
            keys_to_store=[OffloadKey(k) for k in new],
            store_spec=CPULoadStoreSpec(list(rows)),
            evicted_keys=[OffloadKey(k) for k in evicted],
        )

    def _lookup(self, key: OffloadKey, client) -> LookupResult:
        result = self.arena.lookup(client, [bytes(key)])[0]
        assert result != pool_server.P.HIT_PENDING, "a row left unwritten"
        return LookupResult.HIT if result == pool_server.P.HIT else LookupResult.MISS

    def seed_rows(self, keys) -> None:
        """Store a row for each key, the way the conversation's own stores
        would have, from the engine that stored the tail. A resume is reachable
        only over the chunk rows the arena holds, so a control that wants a
        restore has to leave those rows behind first. They carry their key's
        payload and never a group's marker, so a chunk row cannot be mistaken
        for one of the tail's.
        """
        producer = self.clients[0]
        for key in keys:
            raw = bytes(key)
            if raw in self.cls.index:
                continue
            out = self.arena.prepare_store(producer, [raw], raw)
            assert out is not None and out[0] == [raw], "no room for a chunk row"
            self.write_row(out[1][0], _row_payload(OffloadKey(raw)))
            self.arena.complete_store(producer, [raw], True)

    def _prepare_load(self, keys, client) -> CPULoadStoreSpec:
        """The rows a restore reads, which is the arena's own answer and nothing
        else: a key it does not hold ready raises, as it does in production."""
        return CPULoadStoreSpec(
            self.arena.prepare_load(client, [bytes(k) for k in keys])
        )


class _ArenaRows:
    """The pool side of a copy: row N is the ROW_BYTES span the arena file
    holds at ``N * ROW_BYTES``, read and written for real."""

    def __init__(self, pool: "MarkerPool") -> None:
        self.pool = pool

    def read(self, row: int) -> bytes:
        return self.pool.read_row(row)

    def write(self, row: int, payload: bytes) -> None:
        self.pool.write_row(row, payload)


def _group_runs(block_ids, group_sizes) -> list[list[int]]:
    """A spec's flat block list as one run per KV group, the way the copy
    walks it."""
    runs: list[list[int]] = []
    offset = 0
    for size in group_sizes:
        runs.append(list(block_ids[offset : offset + size]))
        offset += size
    assert offset == len(block_ids)
    return runs


def _copy_as_the_worker_would(
    job,
    src,
    dst,
    gpu_to_cpu: bool,
    swap_groups: tuple[int, int] | None = None,
    skip_groups: tuple[int, ...] = (),
    dup_row_into: tuple[int, int] | None = None,
) -> None:
    """Move the bytes one transfer job names, reading nothing but the job's own
    two specs.

    The worker walks the GPU spec's per-group sizes and takes that many blocks
    from each side in order; with one GPU block per chunk, which is what the
    tail gate requires, that is one arena row per block. This is the CPU
    stand-in for that walk: it pairs by position alone, so rows that sit in the
    wrong group's run move the wrong bytes here exactly as they would there.
    ``swap_groups``, ``skip_groups`` and ``dup_row_into`` are the deliberate
    faults the mutation tests inject, and none of them moves a count.
    ``dup_row_into=(group, position)`` repeats a group's last source row at an
    earlier position of its own run, which for a tail group is the row its tail
    key names: the destination block at that position is a chunk block, so the
    fault parks a tail row where a chunk row belongs.
    """
    gpu_spec = job.src_spec if gpu_to_cpu else job.dst_spec
    cpu_spec = job.dst_spec if gpu_to_cpu else job.src_spec
    gpu_ids = [int(b) for b in gpu_spec.block_ids]
    runs = _group_runs(cpu_spec.block_ids, gpu_spec.group_sizes)
    if swap_groups is not None:
        first, second = swap_groups
        assert len(runs[first]) == len(runs[second]), "not a like-for-like swap"
        runs[first], runs[second] = runs[second], runs[first]
    if dup_row_into is not None:
        group_idx, position = dup_row_into
        run = runs[group_idx]
        assert 0 <= position < len(run) - 1, (
            f"position {position} is not a chunk position in group {group_idx}'s run"
        )
        assert run[position] != run[-1], "the fault repeats a row already there"
        run[position] = run[-1]
    cpu_ids = [block for run in runs for block in run]
    assert len(cpu_ids) == len(gpu_ids)
    offset = 0
    for group_idx, size in enumerate(gpu_spec.group_sizes):
        for position in range(size):
            gpu_id = gpu_ids[offset + position]
            cpu_id = cpu_ids[offset + position]
            if group_idx not in skip_groups:
                if gpu_to_cpu:
                    dst.write(cpu_id, src.read(gpu_id))
                else:
                    dst.write(gpu_id, src.read(cpu_id))
        offset += size
    assert offset == len(gpu_ids)


@pytest.fixture
def marker_arena(tmp_path):
    """The arena the control runs over, closed whatever the test does next. The
    state groups come from the same layout the connector is built from, because
    they are what the server reads to decide which runs restore nothing."""
    layout = _make_drafter_partial_tail_scheduler(True)
    pool = MarkerPool(
        tmp_path,
        tuple(
            group.group_idx
            for group in layout.config.kv_group_configs
            if isinstance(group.kv_cache_spec, MambaSpec)
        ),
    )
    yield pool
    pool.close()


def _marker_round_trip(
    pool: "MarkerPool",
    store_swap: tuple[int, int] | None = None,
    store_skip: tuple[int, ...] = (),
    load_swap: tuple[int, int] | None = None,
    load_dup: tuple[int, int] | None = None,
) -> tuple[dict[int, bytes], dict[int, bytes]]:
    """One tail stored into a real arena by one engine and restored out of it by
    the next, and what each group's row carries at the end.

    Returns the bytes of each group's arena row, by group, and the bytes of
    every destination block the restore filled, by block id. The last four
    arguments are the injected faults: a swap or a skip applied to the store's
    rows, or a swap or a duplicated row applied to the restore's.

    The two engines are not symmetric. Only the engine that stores a tail
    records the rows it stored, so the consumer's record is the producer's,
    transplanted below: the restore is the consumer's own job over a store it
    did not run.
    """
    # The producer: a finished turn offers its tail and the store route takes
    # it, so the keys, the rows and the group order are all the connector's.
    producer, handoff, blocks = _answer_tail_offer()
    # The copy below pairs one GPU block with one arena row, which is what the
    # worker does only while a chunk is one GPU block. The fixture sits at 1; pin
    # it here, because a fixture that moved the chunk size would decouple this
    # transcription from the walk it stands in for while every assertion kept
    # passing. Both schedulers come from this fixture, so the producer's answer
    # is the consumer's too.
    assert producer.config.blocks_per_chunk == 1, (
        f"the fixture runs at blocks_per_chunk={producer.config.blocks_per_chunk}, "
        "where one GPU block is no longer one arena row"
    )
    pool.bind(producer)
    gpu = _StandInBlocks()
    for group_idx, block_ids in blocks.items():
        for block_id in block_ids:
            gpu.seed(block_id, _group_marker(group_idx))
    store_jobs = producer._build_answer_tail_store_jobs()
    [store_id] = store_jobs
    _assert_one_row_per_key(producer, store_jobs)
    _copy_as_the_worker_would(
        store_jobs[store_id],
        gpu,
        pool.rows,
        gpu_to_cpu=True,
        swap_groups=store_swap,
        skip_groups=store_skip,
    )
    _complete_all(producer, store_jobs)
    tail_keys = producer._tail_row_keys[bytes(handoff.tail_hash)]
    stored_by_group = {
        get_offload_group_idx(key): pool.read_row(pool.row_of(key)) for key in tail_keys
    }

    # The consumer: the turn replayed against the same arena by an engine that
    # stored nothing, fenced by the arena's own answer and loaded by the
    # connector's own restore route.
    consumer = _make_drafter_partial_tail_scheduler(True)
    request = _make_partial_tail_request(consumer, num_prompt_tokens=32)
    req_status = consumer._req_status["req"]
    req_status.num_locally_computed_tokens = 0
    req_status.update_offload_keys()
    pool.bind(consumer)
    # The conversation's own rows, stored before the resume is decided: the
    # aligned route can only reach a boundary over chunk rows the arena holds,
    # and a resume is granted one token past the deepest boundary it reaches.
    pool.seed_rows(key for gs in req_status.group_states for key in gs.offload_keys)
    # What a restore has to read is the store's own record, not a list here.
    # ``_tail_row_keys`` is written by the engine that stores a tail and the
    # fence refuses a resume without it, so an engine that stored nothing never
    # holds this record on its own: the transplant stands in for the store the
    # consumer did not run, and the job built from it is still the connector's.
    consumer._tail_row_keys[bytes(handoff.tail_hash)] = tail_keys
    consumer._note_in_pool(tail_keys)
    request.tail_resume = AnswerTailHit(
        tail_key=handoff.tail_hash,
        resume_token=handoff.resume_token,
        group_block_ids={},
        num_tokens=30,
        generation=consumer._cache_generation,
    )
    assert consumer._lookup(req_status) == handoff.resume_token

    # The destination blocks this control allocates, per KV group. The tail
    # fills the last block of its group's range, the one at the resume
    # position; the drafter group is not a tail group and fills one block with
    # its own chunk row.
    destination_blocks = (
        [KVCacheBlock(31), KVCacheBlock(32)],
        [KVCacheBlock(0, is_null=True), KVCacheBlock(41)],
        [KVCacheBlock(51), KVCacheBlock(52)],
    )
    dest = _StandInBlocks()
    consumer.update_state_after_alloc(
        request,
        KVCacheBlocks(destination_blocks),
        num_external_tokens=handoff.resume_token,
    )
    [load_job] = consumer._current_batch_load_jobs.values()
    _assert_one_row_per_key(consumer, consumer._current_batch_load_jobs, load=True)
    _copy_as_the_worker_would(
        load_job,
        pool.rows,
        dest,
        gpu_to_cpu=False,
        swap_groups=load_swap,
        dup_row_into=load_dup,
    )
    assert set(dest.landed) == {int(b) for b in load_job.dst_spec.block_ids}
    return stored_by_group, dest.landed


def _assert_markers_landed(
    stored_by_group: dict[int, bytes], landed: dict[int, bytes]
) -> None:
    """The control's assertion, one per group: the row the arena holds under
    group g's tail key carries group g's marker, and the destination block that
    row is restored into carries it too. Every other block the restore filled
    must carry nobody's marker, because a tail state parked in a chunk row is
    the same fault wearing a different hat. A group whose bytes sit in another
    group's row, or in nobody's, fails here with its own index in the message.
    """
    # Where each group's row has to land, read off the destination table the
    # control writes: the last block of the group's own allocated range, the
    # one at the resume position.
    tail_destinations = {0: 32, 1: 41}
    assert set(stored_by_group) == set(tail_destinations)
    for group_idx, payload in stored_by_group.items():
        assert payload == _group_marker(group_idx), (
            f"group {group_idx}'s row holds "
            f"{_describe_marker(payload)}: the store paired the wrong block "
            "with the row this group's key names"
        )
    for group_idx, block_id in tail_destinations.items():
        payload = landed[block_id]
        assert payload == _group_marker(group_idx), (
            f"group {group_idx}'s marker did not reach block {block_id}, which "
            f"holds {_describe_marker(payload)}: the restore paired the wrong "
            "row with this group's destination block"
        )
    for block_id, payload in landed.items():
        if block_id in tail_destinations.values():
            continue
        assert _marker_owner(payload) is None, (
            f"block {block_id} holds {_describe_marker(payload)}, but no tail "
            "row was meant to land there"
        )


def _describe_marker(payload: bytes) -> str:
    owner = _marker_owner(payload)
    if owner is not None:
        return f"group {owner}'s marker"
    if not any(payload):
        return "nothing but zeros"
    return "a row that is no group's marker"


def test_a_tail_marker_lands_in_the_row_its_group_names(marker_arena):
    """The positive control: each group's marker is written into the GPU block
    the offer names, stored through the store job the connector built, and read
    back through the restore job it built. Every count can be right and every
    key present while group 1's state sits in group 2's row, and this is the
    assertion that says otherwise."""
    stored_by_group, landed = _marker_round_trip(marker_arena)

    _assert_markers_landed(stored_by_group, landed)


def test_the_marker_control_fails_when_the_store_swaps_two_groups_rows(marker_arena):
    # Permutation is the failure the control exists for, applied identically by
    # every path is what a key-level assertion cannot see. Here the store writes
    # each group's bytes into the other group's row: the arena's index is
    # untouched, so every key and count still agrees, and a resume would hand
    # back another group's state.
    with pytest.raises(AssertionError, match="group 0's row holds group 1's"):
        _assert_markers_landed(*_marker_round_trip(marker_arena, store_swap=(0, 1)))


def test_the_marker_control_fails_when_a_restore_reads_another_groups_row(
    marker_arena,
):
    # The other half, and the one the store-side assertions cannot reach: the
    # rows are all correct, and the restore reads group 1's row into the
    # drafter group's block and back. A control that only read the arena back
    # would call this clean.
    with pytest.raises(AssertionError, match="group 1's marker did not reach"):
        _assert_markers_landed(*_marker_round_trip(marker_arena, load_swap=(1, 2)))


def test_the_marker_control_fails_when_a_restore_copies_a_tail_row_into_a_chunk(
    marker_arena,
):
    # The negative half, which no other fault reaches: every tail row lands where
    # its own key says, so each positive check passes, and group 0's tail row is
    # read a second time into the chunk block that sits below it. A control that
    # only followed the tail's own destinations would call this clean, and a
    # resume would later read that chunk row as the conversation's state.
    with pytest.raises(AssertionError, match="block 31 holds group 0's marker"):
        _assert_markers_landed(*_marker_round_trip(marker_arena, load_dup=(0, 0)))


def test_the_marker_control_fails_when_a_groups_row_is_left_zero(marker_arena):
    # A row nobody wrote is the third shape: the tail comes back whole, with one
    # group's state read as zeros, which is a state like any other as far as
    # every count is concerned.
    with pytest.raises(AssertionError, match="group 1's row holds nothing"):
        _assert_markers_landed(*_marker_round_trip(marker_arena, store_skip=(1,)))
