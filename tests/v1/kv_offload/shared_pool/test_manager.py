# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the shared KV pool spec and scheduler-side manager, against a
real pool server running in a thread."""

import os
import threading
import time
from typing import Any

import pytest
import torch

from vllm.v1.kv_offload.base import LookupResult, ReqContext, make_offload_key
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingGroupConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from vllm.v1.kv_offload.cpu.common import CPUOffloadingMetrics
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.factory import OffloadingSpecFactory
from vllm.v1.kv_offload.shared_pool.client import pool_info
from vllm.v1.kv_offload.shared_pool.server import PoolServer
from vllm.v1.kv_offload.shared_pool.spec import SharedPoolOffloadingSpec

PAGE = 4096
CTX = ReqContext(req_id="r")


def key(i: int):
    return make_offload_key(i.to_bytes(32, "big"), 0)


@pytest.fixture
def server(tmp_path):
    sock = str(tmp_path / "kvpool.sock")
    srv = PoolServer(sock, {"p": {"bytes": 2 * PAGE}, "big": {"bytes": 8 * PAGE}},
                     shm_dir=str(tmp_path), quarantine_sec=0.0, populate=False)
    thread = threading.Thread(target=srv.serve_forever, kwargs={"log_interval": 3600},
                              daemon=True)
    thread.start()
    for _ in range(200):
        if os.path.exists(sock):
            break
        time.sleep(0.01)
    yield srv, sock
    srv.stop()
    thread.join(timeout=5)


def make_config(
    sock: str,
    pool: str = "p",
    rank: int = 0,
    world_size: int = 1,
    dtype: str = "float16",
    engine_id: str = "engine-a",
    worker_kv_bytes_per_block: int = 8,
) -> OffloadingConfig:
    extra: dict[str, Any] = {
        "spec_name": "SharedPoolOffloadingSpec",
        "shared_pool": {"socket": sock, "pool": pool, "timeout_sec": 5},
    }
    return OffloadingConfig(
        groups=(OffloadingGroupConfig(16, ("layer",), 0),),
        worker_kv_bytes_per_block=worker_kv_bytes_per_block,
        enable_kv_cache_events=True,
        extra_config=extra,
        engine_id=engine_id,
        model=OffloadingModelConfig(name="test-model", dtype=dtype),
        cache=OffloadingCacheConfig(tokens_per_hash=16, blocks_per_chunk=1),
        parallel=OffloadingParallelConfig(
            rank=rank,
            world_size=world_size,
            tp_size=world_size,
            pp_size=1,
            pcp_size=1,
            dcp_size=1,
            data_parallel_index=0,
            data_parallel_size=1,
            data_parallel_rank_local=None,
            is_parallelism_agnostic=False,
        ),
    )


def make_spec(sock: str, **kw) -> SharedPoolOffloadingSpec:
    spec = OffloadingSpecFactory.create_spec(make_config(sock, **kw))
    assert isinstance(spec, SharedPoolOffloadingSpec)
    return spec


def store(manager, keys):
    with manager.lock:
        out = manager.prepare_store(keys, CTX)
    assert out is not None
    with manager.lock:
        manager.complete_store(out.keys_to_store, CTX)
    return out


def test_spec_needs_a_pool_name(server):
    _, sock = server
    config = make_config(sock)
    config.extra_config["shared_pool"].pop("pool")
    with pytest.raises(ValueError, match="shared_pool.pool"):
        OffloadingSpecFactory.create_spec(config)


def test_layout_key_ignores_rank_and_engine_but_not_layout(server):
    _, sock = server
    base = make_spec(sock).layout_key
    assert make_spec(sock, rank=0, world_size=2).layout_key != base
    assert (
        make_spec(sock, rank=1, world_size=2, engine_id="b").layout_key
        == make_spec(sock, rank=0, world_size=2).layout_key
    )
    assert make_spec(sock, dtype="fp8").layout_key != base
    assert make_spec(sock, worker_kv_bytes_per_block=8192).layout_key != base


def test_two_engines_share_one_chunk(server):
    _, sock = server
    a = make_spec(sock, engine_id="a").get_manager()
    b = make_spec(sock, engine_id="b").get_manager()
    out = store(a, [key(1)])
    with b.lock:
        assert b.lookup(key(1), CTX) == LookupResult.HIT
        assert b.lookup(key(2), CTX) == LookupResult.MISS
        spec = b.prepare_load([key(1)], CTX)
    assert list(spec.chunk_ids) == list(out.store_spec.chunk_ids)
    with b.lock:
        b.complete_load([key(1)], CTX)
    # B storing the same key is deduplicated.
    with b.lock:
        again = b.prepare_store([key(1)], CTX)
    assert again is not None and again.keys_to_store == []


def test_lock_region_holds_a_hit_against_other_engines(server):
    _, sock = server
    a = make_spec(sock, engine_id="a").get_manager()
    b = make_spec(sock, engine_id="b").get_manager()
    store(a, [key(1), key(2)])  # the pool holds 2 chunks
    with b.lock:
        assert b.lookup(key(1), CTX) == LookupResult.HIT
        assert b.lookup(key(2), CTX) == LookupResult.HIT
        with a.lock:
            assert a.prepare_store([key(3)], CTX) is None
    with a.lock:
        out = a.prepare_store([key(3)], CTX)
    assert out is not None and len(out.evicted_keys) == 1


def test_reset_cache_drops_only_this_engine(server):
    _, sock = server
    a = make_spec(sock, engine_id="a").get_manager()
    b = make_spec(sock, engine_id="b").get_manager()
    store(a, [key(1), key(2)])
    with b.lock:
        b.lookup(key(1), CTX)
        b.prepare_load([key(1)], CTX)
    with b.lock:
        b.reset_cache()
    with a.lock:
        out = a.prepare_store([key(3), key(4)], CTX)
    assert out is not None and sorted(out.evicted_keys) == sorted([key(1), key(2)])


def test_server_loss_degrades_to_misses(server):
    srv, sock = server
    a = make_spec(sock, engine_id="a").get_manager()
    store(a, [key(1)])
    srv.stop()
    time.sleep(1.5)
    with a.lock:
        assert a.lookup(key(1), CTX) == LookupResult.MISS
        assert a.prepare_store([key(2)], CTX) is None
        a.complete_store([key(2)], CTX)
        a.touch([key(1)], CTX)
    assert not a.available


def test_events_and_stats(server):
    _, sock = server
    a = make_spec(sock, engine_id="a").get_manager()
    store(a, [key(1)])
    with a.lock:
        a.lookup(key(1), CTX)
        a.prepare_load([key(1)], CTX)
        stats = a.get_stats()
    assert stats is not None
    events = list(a.take_events())
    assert len(events) == 1 and not events[0].removed
    assert a.pool_stats(max_age=0)["held"] == 1
    # One of the pool's two chunks is held by the load.
    usage = stats._values[CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC]
    assert list(usage.values()) == [0.5]


def test_external_region_maps_the_server_file(server):
    _, sock = server
    spec = make_spec(sock, pool="big")
    spec._ensure_pool()
    info = pool_info(sock, spec._pool_args())
    region = SharedOffloadRegion(
        engine_id="x",
        num_chunks=spec.num_chunks,
        rank=0,
        kv_bytes_per_chunk=spec.kv_bytes_per_chunk,
        cpu_page_size=spec.cpu_page_size_per_worker,
        external_path=info["path"],
    )
    try:
        assert spec.num_chunks == 8
        region.base_tensor[PAGE : 2 * PAGE].fill_(7)
        with open(info["path"], "rb") as f:
            f.seek(PAGE)
            assert f.read(PAGE) == bytes([7]) * PAGE
        view = region.create_next_worker_view(spec.cpu_page_size_per_worker)
        assert view.shape[0] == 8 and torch.all(view[1] == 7)
    finally:
        region.cleanup()
    assert os.path.exists(info["path"])  # never unlinked by a client


def test_lookup_batch_holds_hits_and_stops_at_miss(server):
    _, sock = server
    a = make_spec(sock, engine_id="a").get_manager()
    b = make_spec(sock, engine_id="b").get_manager()
    store(a, [key(1), key(2)])
    with b.lock:
        res = b.lookup_batch([key(1), key(2), key(3), key(1)], CTX, True)
        assert res == [LookupResult.HIT, LookupResult.HIT, LookupResult.MISS]
        full = b.lookup_batch([key(3), key(2)], CTX, False)
        assert full == [LookupResult.MISS, LookupResult.HIT]
        # Held for this step: a cannot evict them to make room.
        with a.lock:
            assert a.prepare_store([key(4), key(5)], CTX) is None
    with a.lock:
        assert a.prepare_store([key(4), key(5)], CTX) is not None
