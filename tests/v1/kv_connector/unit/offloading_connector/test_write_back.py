# SPDX-License-Identifier: Apache-2.0
"""Write-back band of the offloading scheduler (exclusive RAM tier, 2026-10-23).

The band holds the hashed, chunk-aligned GPU blocks nearest the free-queue head,
stores the ones the pool lacks, and on completion returns them to the queue
HEAD with their hash, so the hold does not change eviction order.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

import vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler as sched_mod
from tests.v1.kv_connector.unit.offloading_connector.test_scheduler import (
    _assert_nothing_half_built,
    _make_drafter_partial_tail_scheduler,
    _make_partial_tail_request,
    _make_partial_tail_scheduler,
)
from tests.v1.kv_connector.unit.offloading_connector.utils import (
    generate_store_output,
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
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.kv_offload.base import (
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
    monkeypatch.setattr(sched_mod, "WRITE_BACK", True)
    monkeypatch.setattr(sched_mod, "WRITE_BACK_SCAN", 8)
    monkeypatch.setattr(sched_mod, "WRITE_BACK_MAX_HELD", 4)
    monkeypatch.setattr(sched_mod, "WRITE_BACK_FREE_SHARE", 1.0)
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


def _assert_one_row_per_key(scheduler, jobs):
    """The invariant the transfer layer pairs on: the per-group split of a
    store's keys, the GPU blocks it reads, and the keys themselves are the same
    size. A store whose split does not add up to its keys mis-pairs rows with
    GPU blocks, the phase-1 P0 class."""
    for job_id, job in jobs.items():
        keys = scheduler._jobs[job_id].keys
        assert sum(job.src_spec.group_sizes) == len(job.src_spec.block_ids) == len(keys)


def _finish_tail_store(keyed_chunk: bool = True):
    """A finished request's tail stored for real, and the band's state after
    it: the scheduler, the request's bookkeeping, the tail's keys in group
    order, and the run the tail was recorded against (None when the request
    keyed no full-attention chunk)."""
    scheduler = _make_drafter_partial_tail_scheduler(True)
    tokens = 30 if keyed_chunk else 12
    request = _make_partial_tail_request(scheduler, num_prompt_tokens=tokens)
    req_status = scheduler._req_status["req"]
    if keyed_chunk:
        req_status.update_offload_keys()       # the conversation keyed a chunk
    req_status.group_states[0].block_ids[:] = [11, 12] if keyed_chunk else [11]
    scheduler.manager.prepare_store.side_effect = (
        lambda keys, ctx: generate_store_output(keys)
    )
    tables = ([31, 32], [41], [51, 52]) if keyed_chunk else ([31], [41], [51])
    boundary = 28 if keyed_chunk else 12
    assert (
        scheduler.register_finished_partial_tail(request, tables, [(1, 99, boundary)])
        is False
    )
    [job_id] = scheduler._finished_partial_tail_jobs
    _assert_one_row_per_key(
        scheduler, {job_id: scheduler._finished_partial_tail_jobs[job_id]}
    )
    keys = sorted(scheduler._jobs[job_id].keys)
    run = scheduler._run_tag([gs.offload_keys for gs in req_status.group_states])
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
    scheduler, _, keys, run = _finish_tail_store()
    assert run is not None and scheduler._run_tails[run] == keys
    scheduler._drop_run(run)
    assert run not in scheduler._run_tails
    assert not [k for k in keys if k in scheduler._tail_run]


def test_a_tail_the_pool_evicted_is_counted_and_forgotten(band):
    # A tail has no GPU hash to be re-offered from, so the pool's eviction feed
    # is the only way the band learns the arena dropped it.
    scheduler, pool, blocks = band
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
    scheduler, _, keys, run = _finish_tail_store()
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
    scheduler, req_status, keys, early = _finish_tail_store()
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


def test_a_finish_tail_is_recorded_against_the_run_that_owns_it():
    # A tail belongs to a run, not to a session: the run named after the
    # request's last attention key is what the tail leaves with, and the run a
    # later store of the conversation takes it along into.
    scheduler, _, keys, run = _finish_tail_store()
    assert run is not None
    assert scheduler._run_tails[run] == keys
    assert all(scheduler._tail_run[key] == run for key in keys)
    assert scheduler._tail_counts["tail_run_missing"] == 0


def test_a_tail_of_a_request_that_keyed_no_chunk_has_no_run_and_is_counted():
    # A request that keyed no full-attention chunk names no run. Writing its
    # tail under a tag of its own would leave a run of nothing but tail rows,
    # which restores nothing and cannot become restorable on its own, so the
    # band records the tail without a run and counts it.
    scheduler, _, keys, run = _finish_tail_store(keyed_chunk=False)
    assert run is None
    assert scheduler._run_tails == {}
    assert all(scheduler._tail_run[key] is None for key in keys)
    assert scheduler._tail_counts["tail_run_missing"] == 1


# --- the answer tail's store route: the core offers, the connector stores ---


def _answer_tail_offer(
    keyed_chunk: bool = True,
    pool_refused: bool = False,
    groups: tuple[int, ...] | None = None,
    null_group: int | None = None,
):
    """A finished turn's tail as the engine core offers it, and the connector
    that could store it: the scheduler, the hand-off and the GPU block the
    offer named per group. Nothing is stored here; the test drains.
    """
    scheduler = _make_drafter_partial_tail_scheduler(True)
    request = _make_partial_tail_request(scheduler)
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
    blocks = {
        group_idx: (0 if group_idx == null_group else 100 + group_idx)
        for group_idx in wanted
    }
    for group_idx, block_id in blocks.items():
        assert scheduler.answer_tail_store.store_tail(handoff, group_idx, block_id)
    scheduler.manager.prepare_store_run = MagicMock(
        side_effect=lambda keys, tag, whole=False, held=False: _cpu_store_output(keys)
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
        blocks[group] for group in sorted(blocks)
    ]
    assert scheduler._tail_counts["tail_offered"] == 1
    assert scheduler._tail_counts["tail_stored"] == 1
    assert all(scheduler._tail_run[key] == tag for key in tail_keys)


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


def _tail_gpu_blocks(handoff, blocks: dict[int, int]) -> dict[int, KVCacheBlock]:
    """The GPU as a retried offer sees it: one block per group, each cached
    under that group's tail key, which is what the retry is checked against."""
    tail_blocks = {
        group_idx: KVCacheBlock(block_id) for group_idx, block_id in blocks.items()
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


def test_the_restore_of_a_tail_is_refused_by_name():
    # A tail that comes back missing a group, or carrying a state that is not
    # the one the resume position needs, is read silently. So the load side
    # refuses by name and copies nothing, which leaves no half tail to reach a
    # request: the byte-level gate is what opens it.
    scheduler, handoff, _ = _answer_tail_offer()
    hit = AnswerTailHit(
        tail_key=handoff.tail_hash,
        resume_token=handoff.resume_token,
        group_block_ids={},
        num_tokens=30,
    )

    assert scheduler.answer_tail_store.lookup_tail(handoff.tail_hash, [0, 1]) is None
    assert scheduler.answer_tail_store.start_load(hit, {}) is False
    assert scheduler.answer_tail_store.wait(hit) is False
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
    # act on.
    scheduler = _make_drafter_partial_tail_scheduler(False)
    scheduler.manager.prepare_store_run = MagicMock()
    handoff = AnswerTailHandoff(
        request_id="req",
        tail_hash=BlockHash(b"tail"),
        parent=None,
        length=12,
        resume_token=29,
    )
    scheduler.answer_tail_store.store_tail(handoff, 0, 100)

    assert not scheduler._build_answer_tail_store_jobs()
    scheduler.manager.prepare_store_run.assert_not_called()
    assert dict(scheduler._tail_counts) == {}
    assert scheduler.answer_tail_store.pending == {}
