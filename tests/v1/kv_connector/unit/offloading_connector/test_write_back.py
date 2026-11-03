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
from vllm.v1.kv_offload.base import OffloadKey, PrepareStoreOutput, ReqContext
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


def test_completion_releases_to_queue_head_with_hash(band):
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
    ids = _queue_ids(pool)
    assert set(ids[:3]) == {2, 3, 6} and ids[3:] == [1, 4, 5]
    assert blocks[1].block_hash is not None     # still a prefix-cache hit


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
    scheduler.manager.prepare_store_run = MagicMock(side_effect=lambda keys, t, whole: (
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
    scheduler.manager.prepare_store_run = MagicMock(side_effect=lambda keys, t, whole: (
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
        side_effect=lambda keys, t, whole: _cpu_store_output(keys))
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


def test_a_refused_run_is_not_offered_again_until_the_backoff(band, monkeypatch):
    scheduler, pool, blocks = band
    scheduler._has_state_groups = True
    chain = _register_run(scheduler, blocks, (1, 2, 5))
    scheduler.manager.prepare_store_run = MagicMock(return_value=None)
    clock = [100.0]
    monkeypatch.setattr(sched_mod.time, "monotonic", lambda: clock[0])
    assert scheduler._build_write_back_jobs() == {}
    assert scheduler._wb_counts["store_refused"] == 1
    scheduler._build_write_back_jobs()
    assert scheduler.manager.prepare_store_run.call_count == 1
    clock[0] += sched_mod.WRITE_BACK_REFUSED_BACKOFF_S + 1
    scheduler._build_write_back_jobs()
    assert scheduler.manager.prepare_store_run.call_count == 2
    assert chain[-1] in scheduler._wb_refused
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
        side_effect=lambda keys, t, whole: _cpu_store_output(keys))
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
