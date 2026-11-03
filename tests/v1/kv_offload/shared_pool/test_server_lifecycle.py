# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The arena server over its socket: deferred attach, prepare and release,
spill and restore, and what it does at start and stop."""

import json
import mmap
import os
import socket
import subprocess
import sys
import threading
import time

import pytest

from .test_server import ROW, P, S, key

SUPERPAGES = 8


def arena_cfg(tmp_path, **kw):
    cfg = {"path": str(tmp_path / "vllm_kvpool_joint.mmap"), "bytes": SUPERPAGES * ROW,
           "superpage_bytes": ROW, "spill_dir": str(tmp_path / "spill"),
           "spill_max_bytes": 1e9, "nvme_free_floor": 0}
    cfg.update(kw)
    return cfg


class Server:
    def __init__(self, tmp_path, pools=None, mem=1 << 40, mappers=(), quarantine_sec=0.0,
                 **cfg):
        self.sock = str(tmp_path / "kvpool.sock")
        self.mappers = list(mappers)
        self.mem = mem
        self.srv = S.PoolServer(self.sock, pools or {"p": {}, "q": {}},
                                arena_cfg(tmp_path, **cfg), shm_dir=str(tmp_path),
                                quarantine_sec=quarantine_sec, populate=False,
                                mem_available=lambda: self.mem,
                                find_mappers=lambda path: self.mappers)
        self.thread = threading.Thread(target=self.srv.serve_forever,
                                       kwargs={"log_interval": 3600}, daemon=True)
        if os.path.exists(self.sock):
            os.unlink(self.sock)
        self.thread.start()
        for _ in range(500):
            if os.path.exists(self.sock):
                break
            time.sleep(0.01)

    def stop(self):
        self.srv.stop()
        self.thread.join(timeout=10)
        assert not self.thread.is_alive()


@pytest.fixture
def server(tmp_path):
    srv = Server(tmp_path)
    yield srv
    srv.stop()


def connect(sock):
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    conn.settimeout(10)
    conn.connect(sock)
    return conn


def call(conn, op, args=None):
    P.send_message(conn, (op, args))
    return P.recv_message(conn)


def attach_args(**kw):
    args = {"pool": "p", "layout_key": "L", "row_stride": ROW,
            "version": P.PROTOCOL_VERSION, "name": "t"}
    args.update(kw)
    return args


def wait_state(sock, want, timeout=10.0):
    conn = connect(sock)
    try:
        deadline = time.monotonic() + timeout
        while True:
            ok, st = call(conn, P.OP_STATE)
            assert ok
            if st["state"] == want:
                return st
            assert time.monotonic() < deadline, st
            time.sleep(0.01)
    finally:
        conn.close()


def put(conn, path, keys, tag, byte):
    """Store keys through the protocol, writing their rows like a worker."""
    ok, (new, rows, _) = call(conn, P.OP_PREPARE_STORE_TAGGED, [keys, tag])
    assert ok
    with open(path, "r+b") as f:
        for row in rows:
            f.seek(row * ROW)
            f.write(bytes([byte]) * ROW)
    assert call(conn, P.OP_COMPLETE_STORE, (new, True)) == (True, None)


def get(conn, path, keys):
    ok, res = call(conn, P.OP_LOOKUP, keys)
    assert ok and res == [P.HIT] * len(keys), res
    ok, rows = call(conn, P.OP_PREPARE_LOAD, keys)
    call(conn, P.OP_END_SESSION)
    with open(path, "rb") as f:
        out = []
        for row in rows:
            f.seek(row * ROW)
            out.append(f.read(ROW)[0])
    call(conn, P.OP_COMPLETE_LOAD, keys)
    return out


def test_attach_waits_for_the_arena_and_info_shares_it(server):
    a = connect(server.sock)
    assert call(a, P.OP_STATE)[1]["state"] == P.ABSENT
    ok, info = call(a, P.OP_ATTACH, attach_args())
    assert ok and info["arena"] and info["num_chunks"] == SUPERPAGES
    assert os.path.getsize(info["path"]) == SUPERPAGES * ROW
    assert info["class_free_fraction"] == 1.0
    b = connect(server.sock)
    ok, info_b = call(b, P.OP_INFO, {"pool": "p", "layout_key": "L", "row_stride": ROW})
    assert ok and info_b["path"] == info["path"]
    ok, err = call(b, P.OP_LOOKUP, [key(1)])
    assert not ok and "not attached" in err
    ok, err = call(b, P.OP_ATTACH, attach_args(layout_key="other"))
    assert not ok and "layout" in err
    ok, err = call(b, P.OP_ATTACH, attach_args(version=1))
    assert not ok and "protocol" in err
    st = call(b, P.OP_STATE)[1]
    assert st["state"] == P.READY and st["clients"] == 1 and "p" in st["classes"]
    a.close()
    b.close()


def test_requests_after_a_waiting_attach_keep_their_order(server):
    a = connect(server.sock)
    frames = [P.encode((P.OP_STATE, None)), P.encode((P.OP_ATTACH, attach_args()))]
    frames += [P.encode((P.OP_LOOKUP, [key(i)])) for i in range(20)]
    a.sendall(b"".join(frames))
    replies = [P.recv_message(a) for _ in range(22)]
    assert replies[0][1]["state"] == P.ABSENT
    assert replies[1][0] and replies[1][1]["arena"]
    assert all(r == (True, [P.MISS]) for r in replies[2:])
    a.close()


def test_prepare_refused_under_the_memory_floor_answers_waiting_attaches(tmp_path):
    srv = Server(tmp_path, mem=SUPERPAGES * ROW, populate_floor_bytes=1 << 30)
    try:
        a = connect(srv.sock)
        a.sendall(P.encode((P.OP_ATTACH, attach_args())) + P.encode((P.OP_STATE, None)))
        ok, err = P.recv_message(a)
        assert not ok and "not ready" in err and "floor" in err
        ok, st = P.recv_message(a)
        assert st["state"] == P.ABSENT and st["last_prepare"]["ok"] is False
        srv.mem = 1 << 40
        assert call(a, P.OP_ATTACH, attach_args())[0]
        a.close()
    finally:
        srv.stop()


def test_release_refusals(tmp_path):
    srv = Server(tmp_path, quarantine_sec=30.0)
    try:
        a = connect(srv.sock)
        call(a, P.OP_ATTACH, attach_args())
        res = call(a, P.OP_RELEASE, {"reason": "t"})[1]
        assert not res["ok"] and "clients attached" in res["reason"]
        call(a, P.OP_PREPARE_STORE_TAGGED, [[key(1)], b"t"])
        a.close()  # its unfinished write is quarantined
        b = connect(srv.sock)
        time.sleep(0.05)
        res = call(b, P.OP_RELEASE, {"reason": "t"})[1]
        assert not res["ok"] and "quarantine" in res["reason"]
        srv.srv.quarantine_sec = 0.0
        srv.srv.arena._quarantine.clear()
        call(b, P.OP_SPAWN_OPEN, {"pool": "p", "token": "s1", "ttl_sec": 600})
        res = call(b, P.OP_RELEASE, {"reason": "t"})[1]
        assert not res["ok"] and "spawn" in res["reason"]
        call(b, P.OP_SPAWN_CLOSE, {"token": "s1"})
        srv.mappers = [4242]
        res = call(b, P.OP_RELEASE, {"reason": "t"})[1]
        assert not res["ok"] and "4242" in res["reason"]
        assert call(b, P.OP_STATE)[1]["last_release"]["ok"] is False
        srv.mappers = []
        assert call(b, P.OP_RELEASE, {"reason": "t"})[1]["ok"]
        wait_state(srv.sock, P.ABSENT)
        b.close()
    finally:
        srv.stop()


def test_spawn_tokens_expire(tmp_path):
    srv = Server(tmp_path)
    try:
        now = [1000.0]
        srv.srv.clock = lambda: now[0]
        a = connect(srv.sock)
        st = call(a, P.OP_SPAWN_OPEN, {"pool": "p", "token": "t", "ttl_sec": 5})[1]
        assert st["spawn_tokens"] == {"t": "p"}
        wait_state(srv.sock, P.READY)
        now[0] += 6
        assert call(a, P.OP_STATE)[1]["spawn_tokens"] == {}
        a.close()
    finally:
        srv.stop()


def test_spill_and_restore_round_trip_across_a_restart(tmp_path):
    srv = Server(tmp_path)
    a = connect(srv.sock)
    path = call(a, P.OP_ATTACH, attach_args())[1]["path"]
    put(a, path, [key(1), key(2)], b"X", 7)
    # Engines tag a request's own run with a str ("<engine>/<request>"); one
    # never merged into a conversation (an aborted request) still spills.
    put(a, path, [key(3)], "e1/r9", 9)
    a.close()
    b = connect(srv.sock)
    assert call(b, P.OP_RELEASE, {"reason": "idle"})[1]["ok"]
    st = wait_state(srv.sock, P.ABSENT)
    assert not os.path.exists(path)
    assert st["spill"]["runs"] == 2 and st["spill"]["files"] == 3
    assert st["spill"]["bytes"] == 3 * ROW
    # Attaching again re-creates the arena and reads the runs back first.
    ok, info = call(b, P.OP_ATTACH, attach_args())
    assert ok
    assert get(b, path, [key(1), key(2), key(3)]) == [7, 7, 9]
    b.close()
    srv.stop()
    # A new server with the manifest on disk stays ABSENT until asked.
    srv2 = Server(tmp_path)
    try:
        c = connect(srv2.sock)
        assert call(c, P.OP_STATE)[1]["state"] == P.ABSENT
        call(c, P.OP_SPAWN_OPEN, {"pool": "p", "token": "t1", "ttl_sec": 60})
        wait_state(srv2.sock, P.READY)
        call(c, P.OP_ATTACH, attach_args())
        assert [r.tag for r in srv2.srv.arena.order] == [b"X", b"e1/r9"]  # the old order
        assert get(c, path, [key(3)]) == [9]
        assert [r.tag for r in srv2.srv.arena.order] == [b"e1/r9", b"X"]  # loaded: oldest
        c.close()
    finally:
        srv2.stop()


def spill_files(tmp_path):
    return {p.relative_to(tmp_path / "spill").as_posix(): p.stat().st_mtime_ns
            for p in (tmp_path / "spill").rglob("*.bin")}


def manifest_of(tmp_path):
    return json.loads((tmp_path / "spill" / "manifest.json").read_text())


def test_a_second_spill_keeps_files_still_in_the_arena_and_drops_the_rest(tmp_path):
    srv = Server(tmp_path)
    try:
        a = connect(srv.sock)
        path = call(a, P.OP_ATTACH, attach_args())[1]["path"]
        put(a, path, [key(1)], b"X", 1)
        put(a, path, [key(2)], b"Y", 2)
        a.close()
        b = connect(srv.sock)
        call(b, P.OP_RELEASE, {})
        wait_state(srv.sock, P.ABSENT)
        first = spill_files(tmp_path)
        assert len(first) == 2  # one file per superpage
        by_key = {k: rel for rel, f in manifest_of(tmp_path)["files"].items() for k in f["keys"]}
        call(b, P.OP_ATTACH, attach_args())
        cls = srv.srv.arena.classes["p"]
        srv.srv.arena.evict_run(cls.runs[b"X"])
        put(b, path, [key(3)], b"Z", 3)
        b.close()
        c = connect(srv.sock)
        call(c, P.OP_RELEASE, {})
        wait_state(srv.sock, P.ABSENT)
        second = spill_files(tmp_path)
        k2 = by_key[key(2).hex()]
        assert k2 in second and second[k2] == first[k2]  # kept, not written again
        assert by_key[key(1).hex()] not in second  # its chunk left the arena
        assert len(second) == 2
        manifest = manifest_of(tmp_path)
        assert [r["tag"] for r in manifest["runs"]] == [b"Y".hex(), b"Z".hex()]
        call(c, P.OP_ATTACH, attach_args())
        assert get(c, path, [key(2), key(3)]) == [2, 3]
        c.close()
    finally:
        srv.stop()


def test_a_spill_file_that_fails_to_restore_is_written_again(tmp_path):
    srv = Server(tmp_path)
    try:
        a = connect(srv.sock)
        path = call(a, P.OP_ATTACH, attach_args())[1]["path"]
        put(a, path, [key(1)], b"X", 1)
        a.close()
        b = connect(srv.sock)
        call(b, P.OP_RELEASE, {})
        wait_state(srv.sock, P.ABSENT)
        (chunk,) = (tmp_path / "spill").rglob("*.bin")
        chunk.write_bytes(b"short")  # unreadable as a row
        call(b, P.OP_ATTACH, attach_args())
        cls = srv.srv.arena.classes["p"]
        assert key(1) not in cls.index
        assert cls.counters["spill_restore_failed"] == 1
        assert not chunk.exists()
        put(b, path, [key(1)], b"X", 4)
        b.close()
        c = connect(srv.sock)
        call(c, P.OP_RELEASE, {})
        wait_state(srv.sock, P.ABSENT)
        (fresh,) = (tmp_path / "spill").rglob("*.bin")
        assert fresh.read_bytes() == bytes([4]) * ROW  # a fresh copy, not the bad file
        c.close()
    finally:
        srv.stop()


def test_spill_cap_keeps_the_newest_whole_runs(tmp_path):
    srv = Server(tmp_path, spill_max_bytes=2 * ROW)
    try:
        a = connect(srv.sock)
        path = call(a, P.OP_ATTACH, attach_args())[1]["path"]
        put(a, path, [key(1), key(2)], b"old", 1)
        put(a, path, [key(3), key(4), key(5)], b"big", 2)
        put(a, path, [key(6), key(7)], b"new", 3)
        a.close()
        b = connect(srv.sock)
        call(b, P.OP_RELEASE, {})
        st = wait_state(srv.sock, P.ABSENT)
        assert st["spill"]["runs"] == 1 and st["spill"]["bytes"] == 2 * ROW
        call(b, P.OP_ATTACH, attach_args())
        assert get(b, path, [key(6), key(7)]) == [3, 3]
        assert call(b, P.OP_LOOKUP, [key(1)])[1] == [P.MISS]
        b.close()
    finally:
        srv.stop()


def gate_writes(monkeypatch, block_at):
    """Make the spill's file writes wait on a gate from the block_at-th on."""
    gate, entered = threading.Event(), threading.Event()
    real = S._write_spill_file
    n = [0]

    def slow(*args):
        n[0] += 1
        if n[0] >= block_at:
            entered.set()
            gate.wait(10)
        real(*args)

    monkeypatch.setattr(S, "_write_spill_file", slow)
    return gate, entered


def arena_bytes_on_disk(path):
    return os.stat(path).st_blocks * 512


def test_release_frees_what_it_will_not_write_at_once(tmp_path, monkeypatch):
    srv = Server(tmp_path, spill_max_bytes=ROW)
    try:
        a = connect(srv.sock)
        path = call(a, P.OP_ATTACH, attach_args())[1]["path"]
        for i in range(3):
            put(a, path, [key(i)], f"r{i}".encode(), 10 + i)
        a.close()
        assert arena_bytes_on_disk(path) >= 3 * ROW
        gate, entered = gate_writes(monkeypatch, 1)
        b = connect(srv.sock)
        res = call(b, P.OP_RELEASE, {})[1]
        assert res["ok"] and res["free_now_bytes"] == 7 * ROW and res["write_bytes"] == ROW
        assert entered.wait(5)
        st = call(b, P.OP_STATE)[1]
        assert st["state"] == P.SPILLING
        assert st["freed_bytes"] == 7 * ROW and st["remaining_bytes"] == ROW
        assert st["free_soon_bytes"] == 0
        # Only the newest run's row still holds pages; the others read as zeros.
        assert arena_bytes_on_disk(path) == ROW
        cls = srv.srv.arena.classes["p"]
        with open(path, "rb") as f:
            for i in range(3):
                f.seek(cls.index[key(i)].row * ROW)
                assert f.read(ROW)[0] == (12 if i == 2 else 0)
        gate.set()
        st = wait_state(srv.sock, P.ABSENT)
        assert st["last_release"]["spill"]["freed_bytes"] == 8 * ROW
        assert st["spill"]["runs"] == 1
        b.close()
    finally:
        srv.stop()


def test_each_superpage_is_freed_once_its_file_is_durable(tmp_path, monkeypatch):
    srv = Server(tmp_path, spill_threads=1)
    try:
        a = connect(srv.sock)
        path = call(a, P.OP_ATTACH, attach_args())[1]["path"]
        for i in range(3):
            put(a, path, [key(i)], f"r{i}".encode(), 10 + i)
        a.close()
        gate, entered = gate_writes(monkeypatch, 3)
        b = connect(srv.sock)
        assert call(b, P.OP_RELEASE, {})[1]["ok"]
        assert entered.wait(5)
        st = call(b, P.OP_STATE)[1]["spill_progress"]
        assert st["files_done"] == 2 and st["freed_bytes"] == 7 * ROW
        assert arena_bytes_on_disk(path) == ROW
        gate.set()
        wait_state(srv.sock, P.ABSENT)
        b.close()
    finally:
        srv.stop()


def test_attach_during_a_spill_refills_the_freed_superpages(tmp_path, monkeypatch):
    srv = Server(tmp_path, spill_threads=1)
    try:
        a = connect(srv.sock)
        path = call(a, P.OP_ATTACH, attach_args())[1]["path"]
        for i in range(3):
            put(a, path, [key(i)], f"r{i}".encode(), 5 + i)
        a.close()
        inode = os.stat(path).st_ino
        gate, entered = gate_writes(monkeypatch, 2)
        b = connect(srv.sock)
        assert call(b, P.OP_RELEASE, {})[1]["ok"]
        assert entered.wait(5)  # the first file is durable and its superpage freed
        c = connect(srv.sock)
        P.send_message(c, (P.OP_ATTACH, attach_args()))
        time.sleep(0.05)
        gate.set()  # the second file completes; the third is never started
        ok, _ = P.recv_message(c)
        assert ok and os.stat(path).st_ino == inode  # the same arena, not a new one
        st = call(b, P.OP_STATE)[1]
        assert st["state"] == P.READY and st["last_release"]["reason"] == "aborted"
        assert st["last_prepare"]["ok"] and st["last_prepare"]["restored"] == 2
        cls = srv.srv.arena.classes["p"]
        assert cls.counters["chunks_dropped_by_spill_abort"] == 2
        assert get(c, path, [key(0), key(1), key(2)]) == [5, 6, 7]
        assert len(manifest_of(tmp_path)["runs"]) == 2  # the runs whole on disk
        b.close()
        c.close()
    finally:
        srv.stop()


def test_attach_during_a_spill_without_room_to_refill_waits_for_it(tmp_path, monkeypatch):
    srv = Server(tmp_path, populate_floor_bytes=1 << 30)
    try:
        a = connect(srv.sock)
        path = call(a, P.OP_ATTACH, attach_args())[1]["path"]
        put(a, path, [key(1)], b"X", 5)
        a.close()
        gate, entered = gate_writes(monkeypatch, 1)
        b = connect(srv.sock)
        assert call(b, P.OP_RELEASE, {})[1]["ok"]
        assert entered.wait(5)
        srv.mem = (1 << 30) + ROW  # too little to populate the 7 freed superpages again
        c = connect(srv.sock)
        P.send_message(c, (P.OP_ATTACH, attach_args()))
        time.sleep(0.05)
        assert not srv.srv._spill_abort.is_set()
        gate.set()
        ok, err = P.recv_message(c)  # released, then the new arena is refused
        assert not ok and "floor" in err
        st = call(b, P.OP_STATE)[1]
        assert st["state"] == P.ABSENT and st["last_release"]["ok"]
        srv.mem = 1 << 40
        assert call(c, P.OP_ATTACH, attach_args())[0]
        assert get(c, path, [key(1)]) == [5]
        b.close()
        c.close()
    finally:
        srv.stop()


def test_pressure_release_spills_the_newest_runs_that_fit_the_budget(tmp_path):
    srv = Server(tmp_path, spill_rate_bytes_per_sec=ROW)
    try:
        a = connect(srv.sock)
        path = call(a, P.OP_ATTACH, attach_args())[1]["path"]
        put(a, path, [key(1), key(2)], b"old", 1)
        put(a, path, [key(3)], b"mid", 2)
        put(a, path, [key(4), key(5)], b"big", 3)
        put(a, path, [key(6)], b"new", 4)
        a.close()
        b = connect(srv.sock)
        res = call(b, P.OP_RELEASE, {"reason": "pressure", "detail": "ram gate: x/y",
                                     "target_bytes": 5 * ROW, "budget_sec": 2.0})[1]
        assert res["ok"] and res["write_bytes"] == 2 * ROW and res["free_now_bytes"] == 6 * ROW
        st = wait_state(srv.sock, P.ABSENT)
        spill = st["last_release"]["spill"]
        assert spill["reason"] == "pressure: ram gate: x/y" and spill["target_met_sec"] is not None
        assert [r["tag"] for r in manifest_of(tmp_path)["runs"]] == [b"mid".hex(), b"new".hex()]
        call(b, P.OP_ATTACH, attach_args())
        assert get(b, path, [key(3)]) == [2] and get(b, path, [key(6)]) == [4]
        assert call(b, P.OP_LOOKUP, [key(4)])[1] == [P.MISS]
        b.close()
    finally:
        srv.stop()


def test_clean_stop_without_clients_spills(tmp_path):
    srv = Server(tmp_path)
    a = connect(srv.sock)
    path = call(a, P.OP_ATTACH, attach_args())[1]["path"]
    put(a, path, [key(1)], b"X", 4)
    a.close()
    time.sleep(0.05)
    srv.stop()
    assert not os.path.exists(path)
    assert (tmp_path / "spill" / "manifest.json").exists()


def test_stop_with_a_client_attached_keeps_the_arena(tmp_path):
    srv = Server(tmp_path)
    a = connect(srv.sock)
    path = call(a, P.OP_ATTACH, attach_args())[1]["path"]
    srv.stop()
    assert os.path.exists(path)
    a.close()


def test_start_adopts_a_mapped_arena_and_removes_an_unmapped_one(tmp_path):
    path = tmp_path / "vllm_kvpool_joint.mmap"
    path.write_bytes(b"\0" * SUPERPAGES * ROW)
    srv = Server(tmp_path, mappers=[99], quarantine_sec=30.0)
    try:
        st = wait_state(srv.sock, P.READY)
        assert st["state"] == P.READY and path.exists()
        a = connect(srv.sock)
        call(a, P.OP_ATTACH, attach_args())
        ok, res = call(a, P.OP_PREPARE_STORE_TAGGED, [[key(1)], b"t"])
        assert ok and res is None  # every superpage is quarantined
        a.close()
    finally:
        srv.stop()
    srv = Server(tmp_path)
    try:
        assert wait_state(srv.sock, P.ABSENT) and not path.exists()
    finally:
        srv.stop()


def test_server_removes_stale_regions_but_not_the_arena(tmp_path):
    stale = tmp_path / "vllm_kvpool_qwen3.8-27b.mmap"
    stale.write_bytes(b"x" * ROW)
    other = tmp_path / "vllm_offload_engine.mmap"
    other.write_bytes(b"y")
    srv = S.PoolServer(str(tmp_path / "s.sock"), {"p": {}}, arena_cfg(tmp_path),
                       shm_dir=str(tmp_path))
    arena = tmp_path / "vllm_kvpool_joint.mmap"
    arena.write_bytes(b"z")
    assert srv.remove_stale_regions() == 1
    assert not stale.exists() and arena.exists() and other.exists()


def test_spill_load_removes_files_the_manifest_does_not_name(tmp_path):
    spill = tmp_path / "spill"
    (spill / "p_L" / "abc").mkdir(parents=True)
    (spill / "p_L" / "abc" / "dead.bin").write_bytes(b"x")  # the per-chunk format
    (spill / "p_L" / "abc" / "half.bin.tmp").write_bytes(b"x")
    (spill / "p_L" / "1-000001.bin").write_bytes(b"y")
    store = S.SpillStore(str(spill), 10**9, 0)
    store.load()
    assert not list(spill.rglob("*.bin")) and not list(spill.rglob("*.tmp"))
    assert not (spill / "p_L").exists()


def test_other_mappers_finds_a_process_mapping_the_file(tmp_path):
    path = tmp_path / "arena.mmap"
    path.write_bytes(b"\0" * mmap.PAGESIZE)
    code = ("import mmap,sys;f=open(sys.argv[1],'r+b');m=mmap.mmap(f.fileno(),0);"
            "print('ok',flush=True);sys.stdin.read()")
    proc = subprocess.Popen([sys.executable, "-c", code, str(path)],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    try:
        assert proc.stdout.readline().strip() == "ok"
        assert proc.pid in S.other_mappers(str(path))
    finally:
        proc.stdin.close()
        proc.wait(timeout=10)
    assert proc.pid not in S.other_mappers(str(path))
