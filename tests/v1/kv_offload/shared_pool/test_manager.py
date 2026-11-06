# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the shared KV pool spec and scheduler-side manager (protocol
contract v2), against a scripted pool server running in a thread. The real
server has its own tests; these check what the engine sends and how it
behaves when the pool is slow, absent or replaced."""

import dataclasses
import os
import socket
import threading
import time
from collections.abc import Callable
from typing import Any

import pytest
import torch

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.canonical_mapping import (
    _opaque_fallback_mapping,
)
from vllm.v1.kv_offload.base import (
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalPageMapping,
    CopyRun,
    LookupResult,
    OffloadPolicy,
    ReqContext,
    ScheduleEndContext,
    make_offload_key,
)
from vllm.v1.kv_offload.config import (
    OffloadingCacheConfig,
    OffloadingConfig,
    OffloadingGroupConfig,
    OffloadingModelConfig,
    OffloadingParallelConfig,
)
from vllm.v1.kv_offload.factory import OffloadingSpecFactory
from vllm.v1.kv_offload.shared_pool import client as C
from vllm.v1.kv_offload.shared_pool import protocol as P
from vllm.v1.kv_offload.shared_pool.manager import SharedPoolOffloadingManager
from vllm.v1.kv_offload.shared_pool.spec import SharedPoolOffloadingSpec

PAGE = 4096
ROWS = 8
# Room for ROWS rows at the largest stride any test maps (TP=2: two pages).
ARENA_BYTES = 2 * ROWS * PAGE
CTX = ReqContext(req_id="r")
END = ScheduleEndContext(new_req_ids=(), preempted_req_ids=())


def key(i: int):
    return make_offload_key(i.to_bytes(32, "big"), 0)


class Refuse(Exception):
    """Raised by a handler: the server answers with an error."""


NO_REPLY = object()


class ScriptedServer:
    """Speaks the pool framing on a Unix socket. Each op answers from
    ``handlers`` (op -> callable(args) -> value); every request is recorded.
    A handler returning NO_REPLY leaves the client waiting; one raising Refuse
    sends an error reply."""

    def __init__(self, sock_path: str, arena_path: str) -> None:
        self.sock_path = sock_path
        self.arena_path = arena_path
        self.calls: list[tuple[str, Any]] = []
        self.free_fraction = 0.5
        self.handlers: dict[str, Callable[[Any], Any]] = {
            P.OP_ATTACH: lambda a: self.info(),
            P.OP_INFO: lambda a: self.info(),
            P.OP_STATE: lambda a: {
                "state": "READY",
                "classes": {"p": {"free_fraction": self.free_fraction}},
            },
            P.OP_LOOKUP: lambda keys: [P.HIT] * len(keys),
            P.OP_LOOKUP_PREFIX: lambda keys: [P.HIT] * len(keys),
            P.OP_PREPARE_STORE_TAGGED: lambda a: (a[0], list(range(len(a[0]))), []),
        }
        self._stopped = False
        self._srv = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self._srv.bind(sock_path)
        self._srv.listen(8)
        self._conns: list[socket.socket] = []
        threading.Thread(target=self._accept, daemon=True).start()

    def info(self) -> dict[str, Any]:
        return {
            "path": self.arena_path,
            "num_chunks": ROWS,
            "row_stride": PAGE,
            "generation": 1,
            "client_id": 7,
            "arena": True,
            "class_free_fraction": self.free_fraction,
        }

    def ops(self) -> list[str]:
        return [op for op, _ in self.calls]

    def _accept(self) -> None:
        while True:
            try:
                conn, _ = self._srv.accept()
            except OSError:
                return
            self._conns.append(conn)
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        while True:
            try:
                op, args = P.recv_message(conn)  # type: ignore[misc]
            except OSError:
                return
            self.calls.append((op, args))
            try:
                reply = (True, self.handlers.get(op, lambda a: None)(args))
            except Refuse as exc:
                reply = (False, str(exc))
            if reply[1] is NO_REPLY:
                continue
            try:
                P.send_message(conn, reply)
            except OSError:
                return

    def stop(self) -> None:
        if self._stopped:
            return
        self._stopped = True
        self._srv.close()
        for conn in self._conns:
            conn.close()
        os.unlink(self.sock_path)


@pytest.fixture
def server(tmp_path):
    arena = tmp_path / "arena.mmap"
    with open(arena, "wb") as f:
        f.truncate(ARENA_BYTES)
    srv = ScriptedServer(str(tmp_path / "kvpool.sock"), str(arena))
    yield srv
    srv.stop()


def make_config(
    sock: str,
    pool: str = "p",
    rank: int = 0,
    world_size: int = 1,
    dtype: str = "float16",
    engine_id: str = "engine-a",
    worker_kv_bytes_per_block: int = PAGE,
    canonical: bool = False,
    portable: bool = False,
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
            is_parallelism_agnostic=portable,
        ),
        canonical_layout=canonical,
        kv_cache_layout="LBHNC",
    )


def make_spec(sock: str, **kw) -> SharedPoolOffloadingSpec:
    spec = OffloadingSpecFactory.create_spec(make_config(sock, **kw))
    assert isinstance(spec, SharedPoolOffloadingSpec)
    return spec


def manager_of(spec: SharedPoolOffloadingSpec) -> SharedPoolOffloadingManager:
    manager = spec.get_manager()
    assert isinstance(manager, SharedPoolOffloadingManager)
    return manager


# --- layout key ---


def test_spec_needs_a_pool_name(tmp_path):
    config = make_config(str(tmp_path / "s"))
    config.extra_config["shared_pool"].pop("pool")
    with pytest.raises(ValueError, match="shared_pool.pool"):
        OffloadingSpecFactory.create_spec(config)


def test_layout_key_ignores_rank_and_engine_but_not_layout(tmp_path):
    sock = str(tmp_path / "s")
    base = make_spec(sock).layout_key
    assert make_spec(sock, rank=0, world_size=2).layout_key != base
    assert (
        make_spec(sock, rank=1, world_size=2, engine_id="b").layout_key
        == make_spec(sock, rank=0, world_size=2).layout_key
    )
    assert make_spec(sock, dtype="fp8").layout_key != base
    assert make_spec(sock, worker_kv_bytes_per_block=2 * PAGE).layout_key != base


def test_portable_canonical_layout_key_spans_tp_sizes(tmp_path):
    sock = str(tmp_path / "s")

    def key_at(tp: int, rank: int = 0, **kw) -> str:
        # The same 4 pages per block, split over tp workers
        kw = {"canonical": True, "portable": True, **kw}
        return make_spec(
            sock,
            rank=rank,
            world_size=tp,
            worker_kv_bytes_per_block=4 * PAGE // tp,
            **kw,
        ).layout_key

    assert key_at(2) == key_at(4) == key_at(4, rank=3)
    # Canonical pages not certified portable stay one class per TP size
    assert key_at(2, portable=False) != key_at(4, portable=False)
    # Canonical rows never share a class with direct-layout rows
    assert key_at(2) != key_at(2, canonical=False, portable=False)


def test_canonical_worker_refuses_layers_the_key_does_not_promise(tmp_path):
    spec = make_spec(str(tmp_path / "s"), world_size=2, canonical=True, portable=True)

    def caches(mapping: CanonicalPageMapping | None) -> CanonicalKVCaches:
        return CanonicalKVCaches([], [[CanonicalKVCacheRef(0, 8, mapping)]])

    portable = CanonicalPageMapping(16, 8, (CopyRun(0, 0, 8, 1, 8, 8),), 1, 0, True)
    spec._check_canonical_refs(caches(portable))
    with pytest.raises(RuntimeError, match="no canonical mapping"):
        spec._check_canonical_refs(caches(None))
    with pytest.raises(RuntimeError, match="not portable"):
        spec._check_canonical_refs(caches(_opaque_fallback_mapping(8, 2, 0)))


# --- attach and run tags ---


def test_attach_sends_the_protocol_version_and_sizes_from_the_reply(server):
    spec = make_spec(server.sock_path)
    manager = manager_of(spec)
    assert manager.available and spec.num_chunks == ROWS
    op, args = server.calls[0]
    assert op == P.OP_ATTACH
    assert args["version"] == P.PROTOCOL_VERSION == 3 and args["pool"] == "p"
    assert args["row_stride"] == PAGE and args["name"] == "engine-a"
    assert args["state_groups"] == []


def test_attach_and_info_name_the_recurrent_groups(server):
    config = make_config(server.sock_path)
    config = dataclasses.replace(
        config,
        groups=(
            OffloadingGroupConfig(16, ("attn",), 0),
            OffloadingGroupConfig(16, ("gdn",), 1, holds_state=True),
        ),
    )
    spec = OffloadingSpecFactory.create_spec(config)
    assert isinstance(spec, SharedPoolOffloadingSpec)
    manager_of(spec)
    assert spec._pool_args()["state_groups"] == [1]
    assert server.calls[0][1]["state_groups"] == [1]


def test_every_store_names_a_run(server):
    manager = manager_of(make_spec(server.sock_path, engine_id="e1"))
    out = manager.prepare_store([key(1), key(2)], CTX)
    assert out is not None and out.keys_to_store == [key(1), key(2)]
    manager.prepare_store_run([key(3)], key(9))
    manager.prepare_store_run([key(4)], key(9), whole=True)
    stores = [a for op, a in server.calls if op == P.OP_PREPARE_STORE_TAGGED]
    # A compute-time store is in its request's run, a write-back store in its
    # conversation's (marked whole when it is one); untagged stores are gone.
    assert stores == [
        [[bytes(key(1)), bytes(key(2))], "e1/r"],
        [[bytes(key(3))], bytes(key(9))],
        [[bytes(key(4))], bytes(key(9)), True],
    ]
    assert "prepare_store" not in server.ops()


def test_merge_tag_moves_the_request_run_into_its_conversation(server):
    manager = manager_of(make_spec(server.sock_path, engine_id="e1"))
    manager.merge_tag(CTX, key(5))
    assert server.calls[-1] == (P.OP_MERGE_TAG, ("e1/r", bytes(key(5))))


def test_requests_offer_loaded_chunks_so_runs_start_at_the_root(server):
    manager = manager_of(make_spec(server.sock_path))
    assert manager.on_new_request(CTX).policy == OffloadPolicy.REQUEST_LEVEL


def test_holds_end_at_the_end_of_the_step_that_made_them(server):
    manager = manager_of(make_spec(server.sock_path))
    manager.on_schedule_end(END)
    assert P.OP_END_SESSION not in server.ops()  # nothing held
    assert manager.lookup(key(1), CTX) == LookupResult.HIT
    manager.on_schedule_end(END)
    manager.on_schedule_end(END)
    assert server.ops().count(P.OP_END_SESSION) == 1


def test_free_fraction_comes_from_state_at_most_once_a_second(server):
    manager = manager_of(make_spec(server.sock_path))
    assert manager.pool_free_fraction() == 0.5  # from the attach reply
    server.free_fraction = 0.01
    assert manager.pool_free_fraction(max_age=0) == 0.01
    server.free_fraction = 0.9
    assert manager.pool_free_fraction() == 0.01
    assert server.ops().count(P.OP_STATE) == 1


# --- running without the pool ---


def test_no_server_means_an_engine_without_offloading(tmp_path):
    spec = make_spec(str(tmp_path / "absent.sock"))
    manager = manager_of(spec)
    assert not manager.available and spec.num_chunks == 0
    assert manager.lookup(key(1), CTX) == LookupResult.MISS
    assert manager.prepare_store([key(1)], CTX) is None
    assert manager.pool_free_fraction() is None
    assert spec._map_arena() is None
    assert spec.get_worker_handshake() == {"arena": None}


def test_attach_past_the_deadline_gives_up(server):
    server.handlers[P.OP_ATTACH] = lambda a: NO_REPLY
    spec = make_spec(server.sock_path)
    start = time.monotonic()
    assert spec._attach(timeout=0.2) is None
    assert time.monotonic() - start < 5


def test_a_refused_or_malformed_attach_is_no_pool(server):
    def refuse(args):
        raise Refuse("layout key differs from the class's")

    server.handlers[P.OP_ATTACH] = refuse
    assert not manager_of(make_spec(server.sock_path)).available
    server.handlers[P.OP_ATTACH] = lambda a: None
    assert not manager_of(make_spec(server.sock_path)).available


def test_losing_the_server_degrades_to_misses(server):
    manager = manager_of(make_spec(server.sock_path))
    server.stop()
    assert manager.lookup(key(1), CTX) == LookupResult.MISS
    assert manager.prepare_store([key(2)], CTX) is None
    manager.complete_store([key(2)], CTX)
    manager.touch([key(1)], CTX)
    assert not manager.available


# --- workers and scheduler agree on the arena ---


def mapped(spec: SharedPoolOffloadingSpec) -> dict[str, Any]:
    region = spec._map_arena()
    assert region is not None
    region.cleanup()
    return spec.get_worker_handshake()


def test_workers_on_the_attached_arena_keep_the_pool(server):
    sched = make_spec(server.sock_path, world_size=2)
    handshakes = {
        r: mapped(make_spec(server.sock_path, rank=r, world_size=2)) for r in (0, 1)
    }
    manager = manager_of(sched)
    sched.set_worker_handshakes(handshakes)
    assert manager.available and manager.reconnect is not None


@pytest.mark.parametrize("case", ["worker_without_arena", "worker_missing"])
def test_a_worker_without_the_arena_disables_the_pool(server, case):
    sched = make_spec(server.sock_path, world_size=2)
    handshakes = {0: mapped(make_spec(server.sock_path, rank=0, world_size=2))}
    if case == "worker_without_arena":
        handshakes[1] = {"arena": None}
    manager = manager_of(sched)
    sched.set_worker_handshakes(handshakes)
    assert not manager.available and manager.reconnect is None


def test_workers_on_another_arena_disable_the_pool(server):
    sched = make_spec(server.sock_path)
    manager = manager_of(sched)
    sched.set_worker_handshakes({0: {"arena": ((1, 2), ROWS)}})
    assert not manager.available


def test_reattach_only_to_the_arena_the_workers_mapped(server, tmp_path):
    sched = make_spec(server.sock_path)
    manager = manager_of(sched)
    sched.set_worker_handshakes({0: mapped(make_spec(server.sock_path))})
    manager.disable("test")
    client = sched._reattach()  # same file
    assert client is not None
    client.close()
    other = tmp_path / "other.mmap"
    with open(other, "wb") as f:
        f.truncate(ARENA_BYTES)
    server.arena_path = str(other)
    assert sched._reattach() is None  # another arena: never again
    server.arena_path = str(tmp_path / "arena.mmap")
    assert sched._reattach() is None


def test_a_degraded_manager_reconnects_at_a_step_end(server, monkeypatch):
    import vllm.v1.kv_offload.shared_pool.manager as manager_mod

    monkeypatch.setattr(manager_mod, "_RECONNECT_INTERVAL_SEC", 0.0)
    sched = make_spec(server.sock_path)
    manager = manager_of(sched)
    sched.set_worker_handshakes({0: mapped(make_spec(server.sock_path))})
    manager.disable("test")
    assert manager.lookup(key(1), CTX) == LookupResult.MISS
    manager.on_schedule_end(END)
    assert manager.available
    assert manager.lookup(key(1), CTX) == LookupResult.HIT


# --- the worker's view of the arena ---


def test_external_region_maps_the_server_file_whole(server):
    spec = make_spec(server.sock_path)
    region = spec._map_arena()
    assert region is not None
    try:
        assert spec.num_chunks == ROWS and region.external
        region.base_tensor[PAGE : 2 * PAGE].fill_(7)
        with open(server.arena_path, "rb") as f:
            f.seek(PAGE)
            assert f.read(PAGE) == bytes([7]) * PAGE
        view = region.create_next_worker_view(spec.cpu_page_size_per_worker)
        assert view.shape[0] == ROWS and torch.all(view[1] == 7)
    finally:
        region.cleanup()
    assert os.path.exists(server.arena_path)  # never unlinked by a client
