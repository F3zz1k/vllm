# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the shared KV pool server.

The server is standard library only and runs outside the vLLM environment, so
these tests load it by path and need neither torch nor the vllm package.
"""

import importlib.util
import os
import socket
import sys
import threading
import time
from pathlib import Path

import pytest

_PKG = Path(__file__).resolve().parents[4] / "vllm/v1/kv_offload/shared_pool"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_kvpool_{name}", _PKG / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


sys.path.insert(0, str(_PKG))
P = _load("protocol")
S = _load("server")


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def key(i: int) -> bytes:
    return i.to_bytes(32, "big") + (0).to_bytes(4, "big")


def make_pool(num_chunks: int = 4, quarantine_sec: float = 30.0, clock=None) -> "S.Pool":
    return S.Pool("p", "layout", 4096, num_chunks, path=None,
                  quarantine_sec=quarantine_sec, clock=clock or Clock())


def client(i: int) -> "S.ClientState":
    return S.ClientState(client_id=i, name=f"c{i}")


def store(pool, c, keys, success=True):
    res = pool.prepare_store(c, keys)
    assert res is not None
    pool.complete_store(c, res[0], success)
    return res


def test_store_then_other_client_hits_and_loads_same_chunk():
    pool, a, b = make_pool(), client(1), client(2)
    new, chunks, evicted = store(pool, a, [key(1), key(2)])
    assert new == [key(1), key(2)] and evicted == []
    assert pool.lookup(b, [key(1), key(2), key(3)]) == [P.HIT, P.HIT, P.MISS]
    assert pool.prepare_load(b, [key(1), key(2)]) == chunks
    pool.end_session(b)
    pool.complete_load(b, [key(1), key(2)])
    assert pool.stats()["evictable"] == 2


def test_store_deduplicates_across_clients():
    pool, a, b = make_pool(), client(1), client(2)
    store(pool, a, [key(1)])
    new, chunks, _ = pool.prepare_store(b, [key(1), key(2), key(2)])
    assert new == [key(2)] and len(chunks) == 1
    assert pool.counters["store_dedup"] == 2


def test_pending_write_is_hit_pending_and_not_rewritten():
    pool, a, b = make_pool(), client(1), client(2)
    pool.prepare_store(a, [key(1)])
    assert pool.lookup(b, [key(1)]) == [P.HIT_PENDING]
    assert pool.prepare_store(b, [key(1)]) == ([], [], [])
    # Only the writer can complete it.
    pool.complete_store(b, [key(1)], True)
    assert pool.lookup(b, [key(1)]) == [P.HIT_PENDING]
    pool.complete_store(a, [key(1)], True)
    assert pool.lookup(b, [key(1)]) == [P.HIT]


def test_lru_evicts_oldest_and_touch_refreshes():
    pool, a = make_pool(num_chunks=3), client(1)
    store(pool, a, [key(1)])
    store(pool, a, [key(2)])
    store(pool, a, [key(3)])
    pool.touch([key(1)])
    _, _, evicted = store(pool, a, [key(4)])
    assert evicted == [key(2)]
    assert pool.lookup(a, [key(2)]) == [P.MISS]


def test_lookup_hold_blocks_eviction_by_another_client_until_session_ends():
    pool, a, b = make_pool(num_chunks=2), client(1), client(2)
    store(pool, a, [key(1), key(2)])
    assert pool.lookup(b, [key(1), key(2)]) == [P.HIT, P.HIT]
    # a needs a chunk: everything is held by b's session.
    assert pool.prepare_store(a, [key(3)]) is None
    assert pool.counters["store_rejected"] == 1
    pool.end_session(b)
    # Released tail first: the deeper chunk goes before the head.
    _, _, evicted = pool.prepare_store(a, [key(3)])
    assert evicted == [key(2)]


def test_load_reference_outlives_session():
    pool, a, b = make_pool(num_chunks=1), client(1), client(2)
    store(pool, a, [key(1)])
    pool.lookup(b, [key(1)])
    pool.prepare_load(b, [key(1)])
    pool.end_session(b)
    assert pool.prepare_store(a, [key(2)]) is None
    pool.complete_load(b, [key(1)])
    assert pool.prepare_store(a, [key(2)])[2] == [key(1)]


def test_prepare_store_is_all_or_nothing():
    pool, a, b = make_pool(num_chunks=3), client(1), client(2)
    store(pool, a, [key(1), key(2)])
    pool.lookup(b, [key(1)])
    before = pool.stats()
    # Needs 3 chunks: 1 free + 1 evictable is not enough.
    assert pool.prepare_store(a, [key(5), key(6), key(7)]) is None
    after = pool.stats()
    assert {k: v for k, v in after.items() if k != "counters"} == {
        k: v for k, v in before.items() if k != "counters"
    }
    assert pool.lookup(a, [key(2)]) == [P.HIT]


def test_prepare_load_without_ready_entry_raises():
    pool, a = make_pool(), client(1)
    with pytest.raises(KeyError):
        pool.prepare_load(a, [key(9)])
    pool.prepare_store(a, [key(1)])
    with pytest.raises(KeyError):
        pool.prepare_load(a, [key(1)])


def test_complete_load_without_prepare_raises():
    pool, a = make_pool(), client(1)
    store(pool, a, [key(1)])
    with pytest.raises(KeyError):
        pool.complete_load(a, [key(1)])


def test_failed_store_frees_chunk_after_quarantine():
    clock = Clock()
    pool, a = make_pool(num_chunks=1, quarantine_sec=30.0, clock=clock), client(1)
    pool.prepare_store(a, [key(1)])
    pool.complete_store(a, [key(1)], False)
    assert pool.lookup(a, [key(1)]) == [P.MISS]
    assert pool.prepare_store(a, [key(2)]) is None
    clock.t += 31
    assert pool.prepare_store(a, [key(2)])[0] == [key(2)]


def test_release_client_drops_holds_loads_and_unfinished_writes():
    clock = Clock()
    pool, a, b = make_pool(num_chunks=3, clock=clock), client(1), client(2)
    store(pool, a, [key(1)])
    pool.lookup(b, [key(1)])
    pool.prepare_load(b, [key(1)])
    pool.prepare_store(b, [key(2)])
    pool.release_client(b)
    stats = pool.stats()
    # b's load of key(1) may still be copying: held back until the quarantine
    # ends. Its unfinished write of key(2) is gone, its chunk quarantined.
    assert stats["held"] == 1 and stats["write_pending"] == 0
    assert stats["evictable"] == 0 and stats["quarantined"] == 1
    assert pool.lookup(a, [key(2)]) == [P.MISS]
    aborted_chunk = 1  # key(2) took the second fresh chunk
    _, chunks3, _ = pool.prepare_store(a, [key(3)])
    assert chunks3 == [2]
    # Nothing reusable yet: chunk 1 quarantined, key(1) held back.
    assert pool.prepare_store(a, [key(4)]) is None
    clock.t += 31
    pool.complete_store(a, [key(3)], True)
    _, chunks4, evicted = pool.prepare_store(a, [key(4)])
    assert chunks4 == [aborted_chunk] and evicted == []
    assert pool.stats()["held"] == 0 and key(1) in pool.lru


def test_store_never_evicts_keys_of_the_offered_prefix():
    pool, a = make_pool(num_chunks=3), client(1)
    store(pool, a, [key(1), key(2)])
    # The scheduler re-offers the whole prefix with two new chunks: one free
    # chunk, and the only evictable ones are this prefix's own. Reject, never
    # punch a hole in it.
    assert pool.prepare_store(a, [key(1), key(2), key(3), key(4)]) is None
    assert pool.lookup(a, [key(1), key(2)]) == [P.HIT, P.HIT]
    pool.end_session(a)


def test_store_evicts_other_chunks_and_refreshes_offered_prefix():
    pool, a = make_pool(num_chunks=4), client(1)
    store(pool, a, [key(1), key(2)])
    store(pool, a, [key(9)])                 # newer than key(1), key(2)
    new, _, evicted = pool.prepare_store(a, [key(1), key(2), key(3), key(4)])
    assert new == [key(3), key(4)] and evicted == [key(9)]
    assert key(1) in pool.index and key(2) in pool.index
    # The re-offered prefix is now most recent, head newest.
    assert list(pool.lru)[-2:] == [key(2), key(1)]


def test_eviction_takes_conversation_tail_before_head():
    pool, a = make_pool(num_chunks=4), client(1)
    store(pool, a, [key(1), key(2), key(3)])
    _, _, evicted = store(pool, a, [key(4), key(5)])
    assert evicted == [key(3)]
    assert pool.lookup(a, [key(1), key(2)]) == [P.HIT, P.HIT]


def test_refcounts_balance_after_mixed_traffic():
    pool = make_pool(num_chunks=8)
    clients = [client(i) for i in range(3)]
    for step in range(200):
        c = clients[step % 3]
        ks = [key((step * 7 + j) % 12) for j in range(3)]
        hits = [k for k, r in zip(ks, pool.lookup(c, ks)) if r == P.HIT]
        if hits:
            pool.prepare_load(c, hits)
        res = pool.prepare_store(c, ks)
        pool.end_session(c)
        if hits:
            pool.complete_load(c, hits)
        if res is not None:
            pool.complete_store(c, res[0], step % 11 != 0)
    for c in clients:
        pool.release_client(c)
    assert all(e.refs == 0 for e in pool.index.values())
    stats = pool.stats()
    assert stats["held"] == 0 and stats["write_pending"] == 0
    assert stats["evictable"] == stats["ready"]


# --- socket server ---


@pytest.fixture
def server(tmp_path):
    sock = str(tmp_path / "kvpool.sock")
    srv = S.PoolServer(sock, {"p": {"bytes": 8 * 4096}, "q": {"bytes": 4096}},
                       shm_dir=str(tmp_path), quarantine_sec=0.0)
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


def call(conn, op, args=None):
    P.send_message(conn, (op, args))
    return P.recv_message(conn)


def connect(sock):
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.settimeout(5)
    conn.connect(sock)
    return conn


def attach_args(**kw):
    args = {"pool": "p", "layout_key": "L", "row_stride": 4096,
            "version": P.PROTOCOL_VERSION, "name": "t"}
    args.update(kw)
    return args


def test_socket_attach_store_and_cross_client_load(server):
    srv, sock = server
    a, b = connect(sock), connect(sock)
    ok, info = call(a, P.OP_ATTACH, attach_args())
    assert ok and info["num_chunks"] == 8
    assert os.path.getsize(info["path"]) == 8 * 4096
    ok, info_b = call(b, P.OP_ATTACH, attach_args(name="b"))
    assert ok and info_b["path"] == info["path"]
    ok, (new, chunks, evicted) = call(a, P.OP_PREPARE_STORE, [key(1)])
    assert ok and new == [key(1)]
    assert call(a, P.OP_COMPLETE_STORE, ([key(1)], True)) == (True, None)
    assert call(b, P.OP_LOOKUP, [key(1)]) == (True, [P.HIT])
    assert call(b, P.OP_PREPARE_LOAD, [key(1)]) == (True, chunks)
    assert call(b, P.OP_END_SESSION) == (True, None)
    assert call(b, P.OP_COMPLETE_LOAD, [key(1)]) == (True, None)
    ok, stats = call(b, P.OP_STATS)
    assert ok and stats["ready"] == 1 and stats["evictable"] == 1
    a.close()
    b.close()


def test_socket_layout_mismatch_is_refused(server):
    _, sock = server
    a, b = connect(sock), connect(sock)
    assert call(a, P.OP_ATTACH, attach_args())[0]
    ok, err = call(b, P.OP_ATTACH, attach_args(layout_key="other"))
    assert not ok and "layout" in err
    ok, err = call(b, P.OP_ATTACH, attach_args(pool="nope"))
    assert not ok and "unknown pool" in err
    a.close()
    b.close()


def test_socket_info_creates_region_without_registering(server):
    srv, sock = server
    a = connect(sock)
    ok, info = call(a, P.OP_INFO, {"pool": "q", "layout_key": "L", "row_stride": 4096})
    assert ok and info["num_chunks"] == 1 and os.path.exists(info["path"])
    ok, err = call(a, P.OP_LOOKUP, [key(1)])
    assert not ok and "not attached" in err
    a.close()


def test_socket_disconnect_releases_client_state(server):
    srv, sock = server
    a, b = connect(sock), connect(sock)
    call(a, P.OP_ATTACH, attach_args())
    call(b, P.OP_ATTACH, attach_args(name="b"))
    call(a, P.OP_PREPARE_STORE, [key(1), key(2)])
    call(a, P.OP_COMPLETE_STORE, ([key(1)], True))
    call(b, P.OP_LOOKUP, [key(1)])
    call(b, P.OP_PREPARE_LOAD, [key(1)])
    b.close()
    a2 = connect(sock)
    call(a2, P.OP_ATTACH, attach_args(name="a2"))
    # b's load reference is held back until the (zero) quarantine is applied,
    # which happens on the next allocation.
    for _ in range(100):
        stats = call(a2, P.OP_STATS)[1]
        if stats["counters"].get("loads_held_back"):
            break
        time.sleep(0.01)
    assert stats["write_pending"] == 1 and stats["counters"]["loads_held_back"] == 1
    call(a2, P.OP_PREPARE_STORE, [key(9)])
    call(a2, P.OP_COMPLETE_STORE, ([key(9)], True))
    assert call(a2, P.OP_STATS)[1]["held"] == 0
    a.close()
    for _ in range(100):
        stats = call(a2, P.OP_STATS)[1]
        if stats["write_pending"] == 0:
            break
        time.sleep(0.01)
    assert stats["write_pending"] == 0 and stats["ready"] == 2   # key(1), key(9)
    assert call(a2, P.OP_LOOKUP, [key(2)]) == (True, [P.MISS])
    a2.close()


def test_socket_pipelined_requests(server):
    _, sock = server
    a = connect(sock)
    payload = b"".join(P.encode((P.OP_ATTACH, attach_args())) if i == 0
                       else P.encode((P.OP_LOOKUP, [key(i)])) for i in range(50))
    a.sendall(payload)
    replies = [P.recv_message(a) for _ in range(50)]
    assert replies[0][0] and all(r == (True, [P.MISS]) for r in replies[1:])
    a.close()


# --- cold tier ---

ROW = 4096


def cold_pool(tmp_path, num_chunks=3, clock=None, **cold_cfg):
    clock = clock or Clock()
    pool = S.Pool("p", "L", ROW, num_chunks, path=str(tmp_path / "region.mmap"),
                  quarantine_sec=0.0, clock=clock)
    pool.create_region(populate=False)
    cfg = {"dir": str(tmp_path / "cold"), "max_bytes": 100 * ROW, "min_loads": 1,
           "write_policy": "reused",
           "write_bytes_per_hour": 3600 * 100 * ROW, "write_burst_bytes": 100 * ROW}
    cfg.update(cold_cfg)
    pool.attach_cold(S.ColdTier(pool, cfg, sync=True))
    return pool


def fill(pool, key_, byte):
    chunk = pool.index[key_].chunk
    pool.mm[chunk * ROW:(chunk + 1) * ROW] = bytes([byte]) * ROW


def stored(pool, c, k, byte):
    res = pool.prepare_store(c, [k])
    fill(pool, k, byte)
    pool.complete_store(c, [k], True)
    return res


def load(pool, c, k):
    assert pool.lookup(c, [k]) == [P.HIT]
    chunk = pool.prepare_load(c, [k])[0]
    pool.end_session(c)
    data = bytes(pool.mm[chunk * ROW:(chunk + 1) * ROW])
    pool.complete_load(c, [k])
    return data


def test_cold_nothing_written_on_store(tmp_path):
    pool, a = cold_pool(tmp_path), client(1)
    stored(pool, a, key(1), 7)
    assert pool.cold.stats()["chunks"] == 0
    assert pool.counters["cold_written"] == 0


def test_cold_demotes_reused_chunk_on_eviction_and_promotes_on_miss(tmp_path):
    pool, a = cold_pool(tmp_path, num_chunks=1), client(1)
    stored(pool, a, key(1), 7)
    assert load(pool, a, key(1)) == bytes([7]) * ROW
    stored(pool, a, key(2), 9)  # evicts key(1), which was loaded once
    assert pool.counters["cold_written"] == 1
    assert os.path.getsize(pool.cold.path(key(1))) == ROW
    # Miss in RAM, hit on disk: read back into a chunk (evicting key(2)).
    assert pool.lookup(a, [key(1)]) == [P.HIT_PENDING]
    assert pool.counters["cold_promoted"] == 1
    assert load(pool, a, key(1)) == bytes([7]) * ROW


def test_cold_skips_chunks_never_reused(tmp_path):
    pool, a = cold_pool(tmp_path, num_chunks=1, min_loads=1), client(1)
    stored(pool, a, key(1), 7)
    stored(pool, a, key(2), 9)
    assert pool.counters["cold_skip_low_value"] == 1
    assert pool.cold.stats()["chunks"] == 0
    assert pool.lookup(a, [key(1)]) == [P.MISS]


def test_cold_write_budget_limits_demotions(tmp_path):
    clock = Clock()
    pool, a = cold_pool(tmp_path, num_chunks=1, clock=clock,
                        write_bytes_per_hour=3600 * ROW, write_burst_bytes=ROW), client(1)
    for i in range(1, 4):
        stored(pool, a, key(i), i)
        load(pool, a, key(i))
    stored(pool, a, key(9), 9)
    # Burst of one row: the first eviction is written, the next two are not.
    assert pool.counters["cold_written"] == 1
    assert pool.counters["cold_skip_budget"] == 2
    clock.t += 1.5  # refills one row per second
    load(pool, a, key(9))
    stored(pool, a, key(10), 10)
    assert pool.counters["cold_written"] == 2


def test_cold_each_key_written_once(tmp_path):
    pool, a = cold_pool(tmp_path, num_chunks=1), client(1)
    stored(pool, a, key(1), 7)
    load(pool, a, key(1))
    stored(pool, a, key(2), 9)
    pool.lookup(a, [key(1)])
    load(pool, a, key(1))
    stored(pool, a, key(3), 3)  # evicts the promoted key(1) again
    assert pool.counters["cold_written"] == 1


def test_cold_cap_trims_least_recent(tmp_path):
    pool, a = cold_pool(tmp_path, num_chunks=1, max_bytes=2 * ROW), client(1)
    for i in range(1, 5):
        stored(pool, a, key(i), i)
        load(pool, a, key(i))
    stored(pool, a, key(9), 9)
    assert pool.cold.stats()["chunks"] == 2
    assert key(3) in pool.cold.disk and key(4) in pool.cold.disk
    assert not os.path.exists(pool.cold.path(key(1)))


def test_cold_corrupt_file_is_dropped(tmp_path):
    pool, a = cold_pool(tmp_path, num_chunks=1), client(1)
    stored(pool, a, key(1), 7)
    load(pool, a, key(1))
    stored(pool, a, key(2), 9)
    with open(pool.cold.path(key(1)), "r+b") as f:
        f.truncate(ROW // 2)
    assert pool.lookup(a, [key(1)]) == [P.HIT_PENDING]
    assert pool.counters["cold_read_failed"] == 1
    assert key(1) not in pool.index and key(1) not in pool.cold.disk
    assert pool.lookup(a, [key(1)]) == [P.MISS]
    assert pool.stats()["free"] == 1


def test_pin_never_evicted_persisted_and_restored_after_restart(tmp_path):
    pool, a = cold_pool(tmp_path, num_chunks=2), client(1)
    stored(pool, a, key(1), 5)
    assert pool.pin([key(1)]) == 1
    assert os.path.getsize(pool.cold.path(key(1))) == ROW
    stored(pool, a, key(2), 6)
    stored(pool, a, key(3), 7)  # must evict key(2), not the pinned key(1)
    assert pool.lookup(a, [key(1), key(2)]) == [P.HIT, P.MISS]
    pool.end_session(a)
    pool.close_region()
    # A new server: the pin list and the file bring key(1) back into RAM.
    pool2 = cold_pool(tmp_path, num_chunks=2)
    assert pool2.stats()["pinned"] == 1
    assert load(pool2, client(2), key(1)) == bytes([5]) * ROW
    assert pool2.unpin(None) == 1
    assert pool2.stats()["pinned"] == 0 and not pool2.cold.pinned


def test_pin_recent_pins_chunks_used_in_window(tmp_path):
    clock = Clock()
    pool, a = cold_pool(tmp_path, num_chunks=4, clock=clock), client(1)
    stored(pool, a, key(1), 1)
    clock.t += 100
    stored(pool, a, key(2), 2)
    load(pool, a, key(2))
    assert pool.pin_recent(10) == 1
    assert pool.index[key(2)].pinned and not pool.index[key(1)].pinned


def test_pin_recent_window_excludes_later_traffic_and_covers_pending(tmp_path):
    clock = Clock()
    pool, a = cold_pool(tmp_path, num_chunks=4, clock=clock), client(1)
    stored(pool, a, key(1), 1)          # the prime, at t
    pool.prepare_store(a, [key(2)])     # still being written at pin time
    clock.t += 30
    stored(pool, a, key(3), 3)          # unrelated traffic after the prime
    assert pool.pin_recent(31, until_ago=20) == 1
    assert pool.index[key(1)].pinned and not pool.index[key(3)].pinned
    pool.complete_store(a, [key(2)], True)
    assert pool.index[key(2)].pinned


def test_pin_before_store_applies_when_stored(tmp_path):
    pool, a = cold_pool(tmp_path, num_chunks=2), client(1)
    assert pool.pin([key(1)]) == 0
    stored(pool, a, key(1), 4)
    assert pool.index[key(1)].pinned
    assert os.path.exists(pool.cold.path(key(1)))


def test_socket_cold_tier_threads_promote_asynchronously(tmp_path):
    sock = str(tmp_path / "kvpool.sock")
    cold = {"dir": str(tmp_path / "cold"), "max_bytes": 100 * ROW, "min_loads": 1,
            "write_bytes_per_hour": 3600 * 100 * ROW, "write_burst_bytes": 100 * ROW,
            "threads": 2, "write_policy": "reused"}
    srv = S.PoolServer(sock, {"p": {"bytes": ROW, "cold": cold}},
                       shm_dir=str(tmp_path), quarantine_sec=0.0, populate=False)
    thread = threading.Thread(target=srv.serve_forever, kwargs={"log_interval": 3600},
                              daemon=True)
    thread.start()
    for _ in range(200):
        if os.path.exists(sock):
            break
        time.sleep(0.01)
    try:
        a = connect(sock)
        ok, info = call(a, P.OP_ATTACH, attach_args(row_stride=ROW))
        assert ok and info["num_chunks"] == 1
        region = open(info["path"], "r+b")
        _, (_, (c1,), _) = call(a, P.OP_PREPARE_STORE, [key(1)])
        region.seek(c1 * ROW)
        region.write(b"\x2a" * ROW)
        region.flush()
        call(a, P.OP_COMPLETE_STORE, ([key(1)], True))
        call(a, P.OP_LOOKUP, [key(1)])
        call(a, P.OP_PREPARE_LOAD, [key(1)])
        call(a, P.OP_END_SESSION)
        call(a, P.OP_COMPLETE_LOAD, [key(1)])
        call(a, P.OP_PREPARE_STORE, [key(2)])  # evicts key(1) to disk
        call(a, P.OP_COMPLETE_STORE, ([key(2)], True))
        for _ in range(500):
            if call(a, P.OP_STATS)[1]["cold"]["chunks"] == 1:
                break
            time.sleep(0.01)
        results = []
        for _ in range(500):
            (_, (r,)) = call(a, P.OP_LOOKUP, [key(1)])
            results.append(r)
            if r == P.HIT:
                break
            time.sleep(0.01)
        assert results[0] == P.HIT_PENDING and results[-1] == P.HIT
        (_, (c,)) = call(a, P.OP_PREPARE_LOAD, [key(1)])
        region.seek(c * ROW)
        assert region.read(ROW) == b"\x2a" * ROW
        region.close()
        a.close()
    finally:
        srv.stop()
        thread.join(timeout=5)


def test_cold_pins_policy_never_writes_evictions_but_reads_pins(tmp_path):
    pool, a = cold_pool(tmp_path, num_chunks=1, write_policy="pins"), client(1)
    stored(pool, a, key(1), 7)
    load(pool, a, key(1))
    load(pool, a, key(1))
    stored(pool, a, key(2), 9)          # evicts a much-reused chunk
    assert pool.counters["cold_written"] == 0
    assert pool.cold.stats()["chunks"] == 0
    assert pool.pin([key(2)]) == 1      # on demand: written at once
    assert pool.counters["cold_written"] == 1
    pool.unpin([key(2)])
    stored(pool, a, key(3), 3)          # evicts key(2), already on disk
    assert pool.lookup(a, [key(2)]) == [P.HIT_PENDING]
    assert load(pool, a, key(2)) == bytes([9]) * ROW


def test_server_removes_stale_regions_on_start(tmp_path):
    stale = tmp_path / "vllm_kvpool_p_deadbeef.mmap"
    stale.write_bytes(b"x" * ROW)
    live = tmp_path / "vllm_kvpool_p.mmap"
    live.write_bytes(b"z" * ROW)
    other = tmp_path / "vllm_offload_engine.mmap"
    other.write_bytes(b"y")
    srv = S.PoolServer(str(tmp_path / "s.sock"), {"p": {"bytes": ROW}}, shm_dir=str(tmp_path))
    assert srv.remove_stale_regions() == 1
    assert not stale.exists() and live.exists() and other.exists()


def test_restarted_pool_adopts_region_and_quarantines_every_chunk(tmp_path):
    clock = Clock()
    path = str(tmp_path / "vllm_kvpool_p.mmap")
    pool = S.Pool("p", "L", ROW, 2, path=path, quarantine_sec=30.0, clock=clock)
    pool.create_region(populate=False)
    a = client(1)
    stored(pool, a, key(1), 7)
    ino = os.stat(path).st_ino
    pool.close_region()
    assert os.path.exists(path)
    # A new server on the same path: same inode, empty index, nothing reusable yet.
    pool2 = S.Pool("p", "L", ROW, 2, path=path, quarantine_sec=30.0, clock=clock)
    pool2.create_region(populate=False)
    assert os.stat(path).st_ino == ino
    assert pool2.lookup(a, [key(1)]) == [P.MISS]
    assert pool2.prepare_store(a, [key(2)]) is None
    clock.t += 31
    assert pool2.prepare_store(a, [key(2)])[0] == [key(2)]
    pool2.close_region(unlink=True)
    assert not os.path.exists(path)


def test_adopting_a_region_of_another_size_fails_and_keeps_it(tmp_path):
    path = tmp_path / "vllm_kvpool_p.mmap"
    path.write_bytes(b"x" * ROW)
    pool = S.Pool("p", "L", ROW, 2, path=str(path), quarantine_sec=0.0, clock=Clock())
    with pytest.raises(RuntimeError, match="needs"):
        pool.create_region(populate=False)
    assert path.stat().st_size == ROW


def test_lookup_prefix_stops_after_first_miss_and_holds_only_hits():
    pool, a, b = make_pool(num_chunks=4), client(1), client(2)
    store(pool, a, [key(1), key(2), key(4)])
    assert pool.lookup(b, [key(1), key(2), key(3), key(4)], stop_at_miss=True) == [
        P.HIT, P.HIT, P.MISS]
    assert set(b.holds) == {key(1), key(2)}
    assert pool.lookup(b, [key(3), key(4)]) == [P.MISS, P.HIT]


def test_stale_pending_write_of_another_client_is_a_miss():
    clock = Clock()
    pool, a, b = make_pool(num_chunks=4, clock=clock), client(1), client(2)
    pool.prepare_store(a, [key(1)])
    assert pool.lookup(b, [key(1)]) == [P.HIT_PENDING]
    clock.t += 2.5
    # Too old for b to wait on; a still sees its own write as pending.
    assert pool.lookup(b, [key(1)]) == [P.MISS]
    assert pool.lookup(a, [key(1)]) == [P.HIT_PENDING]
    assert pool.counters["lookup_pending_as_miss"] == 1
    # b's recompute does not store a second copy, and a's write still lands.
    assert pool.prepare_store(b, [key(1)]) == ([], [], [])
    pool.complete_store(a, [key(1)], True)
    assert pool.lookup(b, [key(1)]) == [P.HIT]


def make_exclusive_pool(num_chunks: int = 3) -> "S.Pool":
    return S.Pool("p", "layout", 4096, num_chunks, path=None, clock=Clock(), exclusive=True)


def test_exclusive_lookup_without_load_does_not_refresh():
    pool, a, b = make_exclusive_pool(), client(1), client(2)
    store(pool, a, [key(1)])
    store(pool, a, [key(2)])
    store(pool, a, [key(3)])
    assert pool.lookup(b, [key(1)]) == [P.HIT]
    pool.end_session(b)
    _, _, evicted = store(pool, a, [key(4)])
    assert evicted == [key(1)]


def test_exclusive_held_key_keeps_its_place_and_is_skipped():
    pool, a, b = make_exclusive_pool(), client(1), client(2)
    store(pool, a, [key(1)])
    store(pool, a, [key(2)])
    store(pool, a, [key(3)])
    assert pool.lookup(b, [key(1)]) == [P.HIT]
    assert pool.stats()["evictable"] == 2 and pool.stats()["held"] == 1
    _, _, evicted = store(pool, a, [key(4)])
    assert evicted == [key(2)]
    pool.end_session(b)
    _, _, evicted = store(pool, a, [key(5)])
    assert evicted == [key(1)]


def test_exclusive_completed_load_demotes_to_eviction_end():
    pool, a, b = make_exclusive_pool(), client(1), client(2)
    store(pool, a, [key(1)])
    store(pool, a, [key(2)])
    store(pool, a, [key(3)])
    assert pool.lookup(b, [key(3)]) == [P.HIT]
    pool.prepare_load(b, [key(3)])
    pool.end_session(b)
    pool.complete_load(b, [key(3)])
    assert pool.counters["demoted_on_load"] == 1
    _, _, evicted = store(pool, a, [key(4)])
    assert evicted == [key(3)]


def test_exclusive_restore_of_existing_key_refreshes_it():
    pool, a = make_exclusive_pool(), client(1)
    store(pool, a, [key(1)])
    store(pool, a, [key(2)])
    store(pool, a, [key(3)])
    # A write-back of a chunk the pool already holds (the GPU is evicting it
    # again) moves it to the newest end.
    store(pool, a, [key(1)])
    _, _, evicted = store(pool, a, [key(4)])
    assert evicted == [key(2)]


def test_exclusive_store_rejected_when_everything_is_held():
    pool, a, b = make_exclusive_pool(num_chunks=2), client(1), client(2)
    store(pool, a, [key(1), key(2)])
    assert pool.lookup(b, [key(1), key(2)]) == [P.HIT, P.HIT]
    assert pool.prepare_store(a, [key(3)]) is None
    pool.end_session(b)


def test_exclusive_evicts_whole_runs_oldest_first():
    pool, a = make_exclusive_pool(num_chunks=5), client(1)
    res = pool.prepare_store(a, [key(1), key(2)], key(2))      # conversation X, root first
    pool.complete_store(a, res[0], True)
    res = pool.prepare_store(a, [key(3), key(4)], key(4))      # conversation Y
    pool.complete_store(a, res[0], True)
    res = pool.prepare_store(a, [key(5), key(6)], key(6))      # needs 1 more chunk: all of X goes
    assert res is not None and sorted(res[2]) == [key(1), key(2)]
    assert pool.counters["runs_evicted"] == 1


def test_exclusive_shared_prefix_moves_to_newest_run():
    pool, a = make_exclusive_pool(num_chunks=4), client(1)
    res = pool.prepare_store(a, [key(1), key(2)], key(2))      # X = root + its turn
    pool.complete_store(a, res[0], True)
    res = pool.prepare_store(a, [key(1), key(3)], key(3))      # Y shares the root
    pool.complete_store(a, res[0], True)
    res = pool.prepare_store(a, [key(7), key(8)], key(8))      # evicts X's remainder only
    assert res is not None and res[2] == [key(2)]
    assert pool.lookup(a, [key(1), key(3)]) == [P.HIT, P.HIT]
    pool.end_session(a)


def test_exclusive_load_demotes_its_whole_run():
    pool, a, b = make_exclusive_pool(num_chunks=4), client(1), client(2)
    res = pool.prepare_store(a, [key(1), key(2)], key(2))
    pool.complete_store(a, res[0], True)
    res = pool.prepare_store(a, [key(3), key(4)], key(4))
    pool.complete_store(a, res[0], True)
    assert pool.lookup(b, [key(3), key(4)]) == [P.HIT, P.HIT]
    pool.prepare_load(b, [key(3), key(4)])
    pool.end_session(b)
    pool.complete_load(b, [key(3), key(4)])
    res = pool.prepare_store(a, [key(5)], key(5))
    assert res is not None and sorted(res[2]) == [key(3), key(4)]


def test_exclusive_offered_run_is_never_evicted_for_itself():
    pool, a = make_exclusive_pool(num_chunks=2), client(1)
    res = pool.prepare_store(a, [key(1), key(2)], key(2))
    pool.complete_store(a, res[0], True)
    assert pool.prepare_store(a, [key(1), key(2), key(3)], key(2)) is None
    assert pool.lookup(a, [key(1), key(2)]) == [P.HIT, P.HIT]
    pool.end_session(a)
