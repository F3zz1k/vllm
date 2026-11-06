# SPDX-License-Identifier: Apache-2.0
"""Flush on stop: a graceful stop writes the GPU's cached conversations to the
pool as whole runs, newest first, within VLLM_KV_FLUSH_ON_STOP_SEC."""

from unittest.mock import MagicMock

import pytest

import vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler as sched_mod
from tests.v1.kv_connector.unit.offloading_connector.test_write_back import (  # noqa: F401
    _register_run,
    band,
)
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingWorkerMetadata,
)
from vllm.v1.kv_offload import flush_on_stop
from vllm.v1.kv_offload.base import OffloadKey, PrepareStoreOutput
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.outputs import KVConnectorOutput


@pytest.fixture
def flush(band, monkeypatch):  # noqa: F811
    monkeypatch.setattr(sched_mod, "FLUSH_ON_STOP_SEC", 20.0)
    scheduler, pool, blocks = band
    calls = []
    scheduler.pool_present = set()  # keys the fake pool holds

    def prepare_store_run(keys, tag, whole=False, held=False):
        calls.append((list(keys), tag, whole))
        new = [k for k in keys if k not in scheduler.pool_present]
        return PrepareStoreOutput(
            keys_to_store=new,
            store_spec=CPULoadStoreSpec(list(range(len(new)))),
            evicted_keys=[],
        )

    scheduler.manager.prepare_store_run = MagicMock(side_effect=prepare_store_run)
    scheduler.manager.pool_flush_rows = MagicMock(return_value=None)
    scheduler.manager.has_pending_work = MagicMock(return_value=False)
    scheduler._has_state_groups = True  # runs only, as on the hybrid models
    old = _register_run(scheduler, blocks, (1,))  # block 2
    new = _register_run(scheduler, blocks, (2, 5))  # blocks 3, 6 (newest)
    return scheduler, blocks, calls, old, new


def _complete(scheduler, job_ids):
    meta = OffloadingWorkerMetadata()
    for job_id in job_ids:
        for _ in range(scheduler.config.num_workers):
            meta.mark_completed(job_id)
    scheduler.update_connector_output(KVConnectorOutput(kv_connector_worker_meta=meta))


def test_every_run_is_stored_oldest_first_and_copied_newest_first(flush):
    scheduler, blocks, calls, old, new = flush
    scheduler.request_flush_on_stop()
    assert scheduler._flush_state == "armed" and scheduler.has_pending_push_work()
    scheduler._flush_state = "ready"
    jobs = scheduler._build_flush_jobs()
    # Pool recency follows the GPU's: the older run is stored first.
    assert [(c[1], c[2]) for c in calls] == [(old[-1], True), (new[-1], True)]
    # The worker copies in dict order: the newest run first.
    assert [j.src_spec.block_ids.tolist() for j in jobs.values()] == [[3, 6], [2]]
    for b in (blocks[1], blocks[2], blocks[5]):
        assert b.ref_cnt == 1  # held until copied
    assert scheduler.has_pending_push_work()
    _complete(scheduler, jobs)
    assert not scheduler.has_pending_push_work()
    assert scheduler._flush_state == "done"
    assert scheduler._flush_counts["runs_written"] == 2
    assert all(b.ref_cnt == 0 for b in blocks)


def test_runs_beyond_the_pool_capacity_are_left_out_newest_kept(flush):
    scheduler, _, calls, old, new = flush
    scheduler.manager.pool_flush_rows.return_value = 2  # the newest run only
    scheduler.request_flush_on_stop()
    scheduler._flush_state = "ready"
    jobs = scheduler._build_flush_jobs()
    assert [c[1] for c in calls] == [new[-1]]
    assert len(jobs) == 1
    assert scheduler._flush_counts["runs_pool_full"] == 1


def _hold_in_pool(scheduler, blocks, i):
    """Block i's chunk is in the pool and its GPU block is in use, so the
    flush has nothing to copy for it; a free block would be copied again."""
    blocks[i].ref_cnt = 1
    key = OffloadKey(bytes(blocks[i].block_hash))
    scheduler.pool_present.add(key)
    scheduler._note_in_pool([key])
    return key


def test_a_full_pool_still_takes_the_flush_and_held_chunks_join_it(flush):
    # The pool's own free rows no longer limit the flush: it displaces older
    # runs. The newest run's chunk already in the pool is touched before any
    # store (a load demoted it to the eviction head) and offered with the run.
    scheduler, blocks, calls, old, new = flush
    held = _hold_in_pool(scheduler, blocks, 3)
    chain = _register_run(scheduler, blocks, (2, 3, 5))  # same tag as `new`: replaces it
    scheduler.request_flush_on_stop()
    scheduler._flush_state = "ready"
    jobs = scheduler._build_flush_jobs()
    scheduler.manager.touch.assert_called_once_with([held], None)
    assert [(c[0], c[1]) for c in calls] == [(old, old[-1]), (chain, chain[-1])]
    assert sorted(sorted(j.src_spec.block_ids.tolist()) for j in jobs.values()) == [[2], [3, 6]]
    assert scheduler._flush_counts["runs_pool_full"] == 0


def test_a_restored_conversation_still_on_the_gpu_is_touched_not_stored(flush):
    # Loaded from the pool and unchanged: nothing to write, but the load
    # demoted its run to the eviction head and the GPU copy is about to go, so
    # it is touched with the others. A run off the GPU keeps its place.
    scheduler, blocks, calls, old, new = flush
    restored = _register_run(scheduler, blocks, (3,))  # newest
    _hold_in_pool(scheduler, blocks, 3)
    gone = OffloadKey(b"gone" + bytes(restored[0])[-4:])
    scheduler._note_in_pool([gone])
    scheduler._run_keys[gone] = [gone]  # newer still, but not on the GPU
    scheduler._key_run[gone] = gone
    scheduler.request_flush_on_stop()
    scheduler._flush_state = "ready"
    jobs = scheduler._build_flush_jobs()
    scheduler.manager.touch.assert_called_once_with(restored, None)
    assert restored[-1] not in [c[1] for c in calls] and gone not in [c[1] for c in calls]
    assert len(jobs) == 2
    assert scheduler._flush_counts["runs_already_in_pool"] == 1
    assert scheduler._flush_counts["runs_nothing_to_write"] == 1


def test_held_chunks_count_against_the_capacity(flush):
    scheduler, blocks, calls, old, new = flush
    _hold_in_pool(scheduler, blocks, 3)
    chain = _register_run(scheduler, blocks, (2, 3, 5))
    scheduler.manager.pool_flush_rows.return_value = 3  # the newest chain only
    scheduler.request_flush_on_stop()
    scheduler._flush_state = "ready"
    scheduler._build_flush_jobs()
    assert [c[1] for c in calls] == [chain[-1]]
    assert scheduler._flush_counts["runs_pool_full"] == 1


def test_an_offered_chunk_the_pool_evicted_is_released_not_left_pending(flush):
    # The band believed the chunk present; another engine's store evicted it,
    # so the store allocates it. Nothing writes it: the row is released.
    scheduler, blocks, calls, old, new = flush
    held = _hold_in_pool(scheduler, blocks, 3)
    scheduler.pool_present.discard(held)  # evicted meanwhile
    chain = _register_run(scheduler, blocks, (2, 3, 5))
    scheduler.request_flush_on_stop()
    scheduler._flush_state = "ready"
    jobs = scheduler._build_flush_jobs()
    scheduler.manager.complete_store.assert_called_once_with([held], None, success=False)
    assert held not in scheduler._pool_has
    assert scheduler._wb_counts["offered_evicted"] == 1
    assert sorted(sorted(j.src_spec.block_ids.tolist()) for j in jobs.values()) == [[2], [3, 6]]
    assert chain[-1] in [c[1] for c in calls]


def test_the_band_waits_one_step_for_the_runs_the_stop_finished(flush):
    scheduler, _, _, _, _ = flush
    scheduler.request_flush_on_stop()
    scheduler.build_connector_meta(MagicMock(
        scheduled_new_reqs=[], preempted_req_ids=(), finished_req_ids=(),
        scheduled_cached_reqs=MagicMock(req_ids=[]), num_scheduled_tokens={},
    ))
    assert scheduler._flush_state == "ready"


def test_out_of_time_lets_the_engine_exit(flush, monkeypatch):
    scheduler, _, _, _, _ = flush
    scheduler.request_flush_on_stop()
    scheduler._flush_state = "ready"
    assert scheduler._build_flush_jobs()
    monkeypatch.setattr(sched_mod, "FLUSH_ON_STOP_SEC", 0.0)
    assert not scheduler.has_pending_push_work()
    assert scheduler._flush_state == "expired"
    # Still out, although the jobs never completed.
    assert scheduler._jobs and not scheduler.has_pending_push_work()


@pytest.mark.parametrize("write_back, budget", [(False, 20.0), (True, 0.0)])
def test_off_without_the_band_or_a_budget(flush, monkeypatch, write_back, budget):
    scheduler, _, _, _, _ = flush
    monkeypatch.setattr(sched_mod, "WRITE_BACK", write_back)
    monkeypatch.setattr(sched_mod, "FLUSH_ON_STOP_SEC", budget)
    scheduler.request_flush_on_stop()
    assert scheduler._flush_state is None
    assert not scheduler.has_pending_push_work()


def test_a_reset_during_the_flush_ends_it(flush):
    scheduler, _, _, _, _ = flush
    scheduler.request_flush_on_stop()
    scheduler._flush_state = "ready"
    scheduler._build_flush_jobs()
    scheduler.reset_cache()
    assert not scheduler.has_pending_push_work()
    assert scheduler._flush_state == "done"


def test_process_manager_grace(monkeypatch):
    cfg = MagicMock(kv_transfer_config=object())
    monkeypatch.setenv("VLLM_KV_WRITE_BACK", "1")
    monkeypatch.setenv("VLLM_KV_FLUSH_ON_STOP_SEC", "20")
    assert flush_on_stop.engine_stop_grace_sec(cfg) == 20 + flush_on_stop.TEARDOWN_GRACE_SEC
    monkeypatch.setenv("VLLM_KV_FLUSH_ON_STOP_SEC", "0")
    assert flush_on_stop.engine_stop_grace_sec(cfg) == 0
    monkeypatch.setenv("VLLM_KV_FLUSH_ON_STOP_SEC", "20")
    monkeypatch.setenv("VLLM_KV_WRITE_BACK", "0")
    assert flush_on_stop.engine_stop_grace_sec(cfg) == 0
    monkeypatch.setenv("VLLM_KV_WRITE_BACK", "1")
    assert flush_on_stop.engine_stop_grace_sec(MagicMock(kv_transfer_config=None)) == 0
