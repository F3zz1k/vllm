# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the shared KV arena server: runs, eviction and the arena's
superpages and classes, driven through Arena directly.

The server is standard library only and runs outside the vLLM environment, so
these tests load it by path and need neither torch nor the vllm package.
"""

import collections
import hashlib
import importlib.util
import mmap
import os
import random
import sys
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

ROW = 4096


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def key(i: int, group: int = 0) -> bytes:
    return i.to_bytes(32, "big") + group.to_bytes(4, "big")


def make_arena(superpages=4, sp_bytes=ROW, pin_superpages=0, clock=None,
               quarantine_sec=30.0, path=None) -> "S.Arena":
    return S.Arena(path, superpages * sp_bytes, sp_bytes, pin_superpages * sp_bytes,
                   quarantine_sec=quarantine_sec, clock=clock or Clock())


def make_class(arena, name="p", stride=ROW, state_groups=(), **cfg) -> "S.LayoutClass":
    return arena.class_for(name, cfg, f"L-{name}", stride, state_groups)


def client(cls, i: int) -> "S.ClientState":
    c = S.ClientState(client_id=i, name=f"c{i}", cls=cls)
    cls.clients += 1
    return c


def single(rows=4, **kw):
    """One class whose rows are whole superpages, and two clients."""
    arena = make_arena(superpages=rows, **kw)
    cls = make_class(arena)
    return arena, cls, client(cls, 1), client(cls, 2)


def store(arena, c, keys, tag=None, success=True):
    res = arena.prepare_store(c, keys, tag if tag is not None else keys[-1])
    assert res is not None
    arena.complete_store(c, res[0], success)
    return res


def load(arena, c, keys):
    assert arena.lookup(c, keys) == [P.HIT] * len(keys)
    rows = arena.prepare_load(c, keys)
    arena.end_session(c)
    arena.complete_load(c, keys)
    return rows


def check(arena):
    """Invariants of the superpage and row accounting."""
    used = collections.Counter()
    for cls in {c for c in arena.sp_owner if c is not None} | set(arena.classes.values()):
        for k, e in cls.index.items():
            assert cls.row_key[e.row] == k
            assert not e.pinned or arena.sp_pin[cls.sp_of(e.row)]  # pins live in the zone
            used[(cls, e.row)] += 1
            if e.moving is not None:
                used[(cls, e.moving)] += 1
            for run in e.runs:
                assert k in run.keys and run.cls is cls
        assert len(cls.row_key) == len(cls.index)
        assert cls.n_pending == sum(1 for e in cls.index.values() if not e.ready)
        assert cls.n_refd == sum(1 for e in cls.index.values() if e.refs)
        assert cls.n_pinned == sum(1 for e in cls.index.values() if e.pinned)
        free_rows = 0
        for sp, free in cls.sp_free.items():
            assert arena.sp_owner[sp] is cls
            rows = set(cls.rows_of(sp))
            assert set(free) <= rows and len(set(free)) == len(free)
            if not arena.sp_pin[sp] and sp in cls.partial:
                free_rows += len(free)
            if free and sp not in cls.draining:
                assert sp in (cls.pin_partial if arena.sp_pin[sp] else cls.partial)
            assert arena.sp_used[sp] == len(rows) - len(free)
        assert free_rows == cls.free_rows
        for run in cls.runs.values():
            assert run in arena.order
            for k in run.keys:
                assert run in cls.index[k].runs
    for _, cls, row in arena._quarantine:
        used[(cls, row)] += 1
    for (cls, row), n in used.items():
        assert n == 1, f"row {row} of {cls.name} used {n} times"
        sp = cls.sp_of(row)
        assert arena.sp_owner[sp] is cls and row in cls.rows_of(sp)
        assert row not in cls.sp_free[sp]
    for sp in range(arena.num_sp):
        if arena.sp_owner[sp] is None:
            assert sp in arena.free_sp and arena.sp_used[sp] == 0
    assert len(set(arena.free_sp)) == len(arena.free_sp)
    assert arena.pin_sp == sum(1 for sp in range(arena.num_sp)
                               if arena.sp_owner[sp] is not None and arena.sp_pin[sp])
    # No byte of the arena belongs to two rows.
    spans = sorted((cls.offset(r), cls.offset(r) + cls.stride) for (cls, r) in used)
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert end <= start


# --- single class: stores, loads, holds ---


def test_store_then_other_client_hits_and_loads_same_row():
    arena, cls, a, b = single()
    new, rows, evicted = store(arena, a, [key(1), key(2)])
    assert new == [key(1), key(2)] and evicted == []
    assert arena.lookup(b, [key(1), key(2), key(3)]) == [P.HIT, P.HIT, P.MISS]
    assert arena.prepare_load(b, [key(1), key(2)]) == rows
    arena.end_session(b)
    arena.complete_load(b, [key(1), key(2)])
    assert cls.stats()["evictable"] == 2
    check(arena)


def test_store_deduplicates_across_clients():
    arena, cls, a, b = single()
    store(arena, a, [key(1)])
    new, rows, _ = arena.prepare_store(b, [key(1), key(2), key(2)], b"t")
    assert new == [key(2)] and len(rows) == 1
    assert cls.counters["store_dedup"] == 2


def test_pending_write_is_hit_pending_and_not_rewritten():
    arena, _, a, b = single()
    arena.prepare_store(a, [key(1)], b"t")
    assert arena.lookup(b, [key(1)]) == [P.HIT_PENDING]
    assert arena.prepare_store(b, [key(1)], b"u") == ([], [], [])
    arena.complete_store(b, [key(1)], True)  # only the writer completes it
    assert arena.lookup(b, [key(1)]) == [P.HIT_PENDING]
    arena.complete_store(a, [key(1)], True)
    assert arena.lookup(b, [key(1)]) == [P.HIT]


def test_lookup_hold_blocks_eviction_until_session_ends():
    arena, cls, a, b = single(rows=2)
    store(arena, a, [key(1), key(2)])
    assert arena.lookup(b, [key(1), key(2)]) == [P.HIT, P.HIT]
    assert arena.prepare_store(a, [key(3)], key(3)) is None
    assert cls.counters["store_rejected"] == 1
    arena.end_session(b)
    _, _, evicted = arena.prepare_store(a, [key(3)], key(3))
    assert sorted(evicted) == [key(1), key(2)]  # the whole run
    check(arena)


def test_load_reference_outlives_session():
    arena, _, a, b = single(rows=1)
    store(arena, a, [key(1)])
    arena.lookup(b, [key(1)])
    arena.prepare_load(b, [key(1)])
    arena.end_session(b)
    assert arena.prepare_store(a, [key(2)], key(2)) is None
    arena.complete_load(b, [key(1)])
    assert arena.prepare_store(a, [key(2)], key(2))[2] == [key(1)]


def test_store_is_all_or_nothing():
    arena, cls, a, b = single(rows=3)
    store(arena, a, [key(1)])
    store(arena, a, [key(2)])
    arena.lookup(b, [key(1)])
    before = (cls.stats(), list(arena.order))
    # Needs 3 rows: 1 free + 1 evictable run is not enough.
    assert arena.prepare_store(a, [key(5), key(6), key(7)], b"x") is None
    after = (cls.stats(), list(arena.order))

    def strip(state):
        return {k: v for k, v in state[0].items() if k != "counters"}, state[1]

    assert strip(after) == strip(before)
    assert arena.lookup(a, [key(2)]) == [P.HIT]
    check(arena)


def test_prepare_load_and_complete_load_need_their_counterparts():
    arena, _, a, _ = single()
    with pytest.raises(KeyError):
        arena.prepare_load(a, [key(9)])
    arena.prepare_store(a, [key(1)], b"t")
    with pytest.raises(KeyError):
        arena.prepare_load(a, [key(1)])
    arena.complete_store(a, [key(1)], True)
    with pytest.raises(KeyError):
        arena.complete_load(a, [key(1)])


def test_failed_store_frees_row_after_quarantine():
    clock = Clock()
    arena, _, a, _ = single(rows=1, clock=clock)
    arena.prepare_store(a, [key(1)], b"t")
    arena.complete_store(a, [key(1)], False)
    assert arena.lookup(a, [key(1)]) == [P.MISS]
    assert arena.prepare_store(a, [key(2)], b"u") is None
    clock.t += 31
    assert arena.prepare_store(a, [key(2)], b"u")[0] == [key(2)]
    check(arena)


def test_release_client_drops_holds_loads_and_unfinished_writes():
    clock = Clock()
    arena, cls, a, b = single(rows=3, clock=clock)
    store(arena, a, [key(1)])
    arena.lookup(b, [key(1)])
    arena.prepare_load(b, [key(1)])
    arena.prepare_store(b, [key(2)], b"w")
    arena.release_client(b)
    stats = cls.stats()
    # b's load of key(1) may still be copying: held back until the quarantine
    # ends. Its unfinished write of key(2) is gone, its row quarantined.
    assert stats["held"] == 1 and stats["write_pending"] == 0
    assert arena.lookup(a, [key(2)]) == [P.MISS]
    assert len(arena._quarantine) == 1 and arena.in_quarantine()
    store(arena, a, [key(3)])
    # Full: key(1) held back, key(2)'s row quarantined, so key(3) is the victim.
    assert store(arena, a, [key(4)])[2] == [key(3)]
    clock.t += 31
    arena.maintain()
    assert cls.stats()["held"] == 0 and not arena.in_quarantine()
    assert store(arena, a, [key(5)])[2] == []  # the quarantined row is free again
    check(arena)


def test_lookup_prefix_stops_after_first_miss_and_holds_only_hits():
    arena, _, a, b = single()
    store(arena, a, [key(1), key(2), key(4)])
    assert arena.lookup(b, [key(1), key(2), key(3), key(4)], stop_at_miss=True) == [
        P.HIT, P.HIT, P.MISS]
    assert set(b.holds) == {key(1), key(2)}


def test_stale_pending_write_of_another_client_is_a_miss():
    clock = Clock()
    arena, cls, a, b = single(clock=clock)
    arena.prepare_store(a, [key(1)], b"t")
    assert arena.lookup(b, [key(1)]) == [P.HIT_PENDING]
    clock.t += 2.5
    assert arena.lookup(b, [key(1)]) == [P.MISS]
    assert arena.lookup(a, [key(1)]) == [P.HIT_PENDING]
    assert cls.counters["lookup_pending_as_miss"] == 1
    assert arena.prepare_store(b, [key(1)], b"u") == ([], [], [])
    arena.complete_store(a, [key(1)], True)
    assert arena.lookup(b, [key(1)]) == [P.HIT]


# --- runs: order and eviction ---


def test_lookup_without_load_does_not_refresh():
    arena, _, a, b = single(rows=3)
    for i in (1, 2, 3):
        store(arena, a, [key(i)])
    assert arena.lookup(b, [key(1)]) == [P.HIT]
    arena.end_session(b)
    assert store(arena, a, [key(4)])[2] == [key(1)]


def test_held_run_keeps_its_place_and_is_skipped():
    arena, _, a, b = single(rows=3)
    for i in (1, 2, 3):
        store(arena, a, [key(i)])
    assert arena.lookup(b, [key(1)]) == [P.HIT]
    assert store(arena, a, [key(4)])[2] == [key(2)]
    arena.end_session(b)
    assert store(arena, a, [key(5)])[2] == [key(1)]


def test_completed_load_demotes_its_run():
    arena, cls, a, b = single(rows=3)
    for i in (1, 2, 3):
        store(arena, a, [key(i)])
    load(arena, b, [key(3)])
    assert cls.counters["demoted_on_load"] == 1
    assert store(arena, a, [key(4)])[2] == [key(3)]


def test_non_exclusive_class_does_not_demote_on_load():
    arena = make_arena(superpages=3)
    cls = make_class(arena, exclusive=False)
    a, b = client(cls, 1), client(cls, 2)
    for i in (1, 2, 3):
        store(arena, a, [key(i)])
    load(arena, b, [key(3)])
    assert store(arena, a, [key(4)])[2] == [key(1)]


def test_store_to_an_existing_run_refreshes_it():
    arena, _, a, _ = single(rows=3)
    for i in (1, 2, 3):
        store(arena, a, [key(i)])
    store(arena, a, [key(1)])  # written back again: newest
    assert store(arena, a, [key(4)])[2] == [key(2)]


def test_whole_runs_are_evicted_oldest_first():
    arena, cls, a, _ = single(rows=5)
    store(arena, a, [key(1), key(2)], key(2))
    store(arena, a, [key(3), key(4)], key(4))
    res = arena.prepare_store(a, [key(5), key(6)], key(6))
    assert res is not None and sorted(res[2]) == [key(1), key(2)]
    assert cls.counters["runs_evicted"] == 1


def test_a_touched_run_outlives_a_run_stored_after_it():
    # The band touches a run whose GPU copy left: the pool's only copy goes last.
    arena, _, a, _ = single(rows=5)
    store(arena, a, [key(1), key(2)], key(2))
    store(arena, a, [key(3), key(4)], key(4))
    arena.touch(a, [key(1)])
    res = arena.prepare_store(a, [key(5), key(6)], key(6))
    assert res is not None and sorted(res[2]) == [key(3), key(4)]


def test_offered_run_is_never_evicted_for_itself():
    arena, _, a, _ = single(rows=2)
    store(arena, a, [key(1), key(2)], key(2))
    assert arena.prepare_store(a, [key(1), key(2), key(3)], key(2)) is None
    assert arena.lookup(a, [key(1), key(2)]) == [P.HIT, P.HIT]


def test_offered_keys_of_another_run_survive_its_eviction():
    arena, _, a, _ = single(rows=3)
    store(arena, a, [key(1), key(2)], b"X")
    store(arena, a, [key(5)], b"W")
    # Y re-offers X's chunks with two new ones: W goes; X's chunks stay in Y.
    new, _, evicted = store(arena, a, [key(1), key(2), key(3)], b"Y")
    assert new == [key(3)] and evicted == [key(5)]
    assert arena.lookup(a, [key(1), key(2), key(3)]) == [P.HIT] * 3
    check(arena)


def test_shared_prefix_belongs_to_both_runs():
    arena, cls, a, _ = single(rows=4)
    store(arena, a, [key(1), key(2)], b"X")
    store(arena, a, [key(1), key(3)], b"Y")
    assert {r.tag for r in cls.index[key(1)].runs} == {b"X", b"Y"}
    # Evicting X frees only its own chunk: the root stays for Y.
    _, _, evicted = store(arena, a, [key(7), key(8)], b"Z")
    assert evicted == [key(2)]
    assert arena.lookup(a, [key(1), key(3)]) == [P.HIT, P.HIT]
    check(arena)


def test_merge_tag_moves_a_request_into_its_conversation():
    arena, cls, a, _ = single(rows=8)
    store(arena, a, [key(1), key(2)], b"conv")
    store(arena, a, [key(9)], b"other")
    store(arena, a, [key(3)], b"req")
    assert arena.merge_tag(a, b"req", b"conv") == 1
    assert b"req" not in cls.runs
    assert set(cls.runs[b"conv"].keys) == {key(1), key(2), key(3)}
    assert list(arena.order)[0].tag == b"conv"  # it kept its place
    # A merge into a new tag renames the run in place.
    store(arena, a, [key(4)], b"req2")
    assert arena.merge_tag(a, b"req2", b"conv2") == 1
    assert list(arena.order)[-1].tag == b"conv2"
    assert arena.merge_tag(a, b"nope", b"conv") == 0
    check(arena)


def test_release_promotes_runs_the_client_loaded():
    arena, cls, a, b = single(rows=4)
    store(arena, a, [key(1), key(2)], b"X")
    store(arena, a, [key(3), key(4)], b"Y")
    load(arena, b, [key(1), key(2)])
    assert list(arena.order)[0].tag == b"X"
    arena.release_client(b)
    assert list(arena.order)[-1].tag == b"X"
    assert cls.counters["promoted_on_release"] == 1


def test_a_store_ends_the_demotion_of_its_run():
    arena, _, a, b = single(rows=4)
    store(arena, a, [key(1), key(2)], b"X")
    store(arena, a, [key(3), key(4)], b"Y")
    load(arena, b, [key(1), key(2)])
    store(arena, a, [key(1)], b"X")  # written back again by a
    store(arena, a, [key(3)], b"Y")  # Y newest
    arena.release_client(b)          # X is not promoted again
    assert [r.tag for r in arena.order] == [b"X", b"Y"]


# --- restorable runs (hybrid models) ---

ATT, ST1, ST2 = 0, 1, 2


def hybrid(rows=8, clock=None):
    arena = make_arena(superpages=rows, clock=clock or Clock())
    cls = make_class(arena, state_groups=(ST1, ST2))
    return arena, cls, client(cls, 1)


def test_run_without_a_complete_boundary_is_not_restorable():
    arena, cls, a = hybrid()
    store(arena, a, [key(1, ATT), key(1, ST1)], b"X")
    assert not cls.runs[b"X"].restorable and cls.stats()["unrestorable_runs"] == 1
    store(arena, a, [key(1, ST2)], b"X")
    assert cls.runs[b"X"].restorable
    arena.prepare_store(a, [key(2, ATT), key(2, ST1), key(2, ST2)], b"Y")
    assert cls.runs[b"Y"].restorable


def test_dead_runs_are_evicted_first_once_past_the_grace():
    clock = Clock()
    arena, cls, a = hybrid(rows=4, clock=clock)
    store(arena, a, [key(1, ATT), key(1, ST1), key(1, ST2)], b"old")  # restorable, oldest
    store(arena, a, [key(2, ATT)], b"dead")                           # attention only
    # Within the grace the dead run is merely newer: the old run goes.
    new = [key(9, ATT), key(9, ST1), key(9, ST2)]
    assert sorted(store(arena, a, new, b"n")[2]) == [key(1, ATT), key(1, ST1), key(1, ST2)]
    clock.t += S.DEAD_RUN_GRACE_SEC + 1
    assert store(arena, a, [key(8, ATT)], b"m")[2] == [key(2, ATT)]
    assert cls.counters["unrestorable_runs_evicted"] == 1


def test_whole_store_of_an_unrestorable_run_is_rejected():
    arena, cls, a = hybrid()
    assert arena.prepare_store(a, [key(1, ATT), key(2, ATT)], b"X", whole=True) is None
    assert cls.counters["store_rejected_unrestorable"] == 1 and not cls.index
    res = arena.prepare_store(a, [key(1, ATT), key(1, ST1), key(1, ST2)], b"X", whole=True)
    assert res is not None and len(res[0]) == 3


# --- several classes in one arena ---

SP = 4 * ROW  # four small rows, or one big row, per superpage


def two_classes(superpages=4, pin_superpages=0, path=None, clock=None):
    arena = make_arena(superpages=superpages, sp_bytes=SP, pin_superpages=pin_superpages,
                       path=path, clock=clock)
    big = make_class(arena, "big", stride=SP)
    small = make_class(arena, "small", stride=ROW)
    return arena, big, small, client(big, 1), client(small, 2)


def test_classes_get_their_own_superpages_and_num_chunks_by_stride():
    arena, big, small, b, s = two_classes()
    store(arena, b, [key(1)])
    store(arena, s, [key(1), key(2)])  # the same key bytes in another class
    assert big.num_rows == 4 and small.num_rows == 16
    assert len(big.sp_free) == 1 and len(small.sp_free) == 1
    assert arena.lookup(b, [key(2)]) == [P.MISS]
    check(arena)


def test_rows_of_an_uneven_stride_stay_inside_their_superpage():
    arena = make_arena(superpages=5, sp_bytes=10 * ROW)
    odd = make_class(arena, "odd", stride=3 * ROW)  # 2 or 3 rows per superpage
    c = client(odd, 1)
    stored = 0
    while (res := arena.prepare_store(c, [key(stored)], key(stored))) is not None and not res[2]:
        arena.complete_store(c, res[0], True)
        stored += 1
    assert odd.min_rows == 2 and odd.max_rows == 3 and stored >= 10
    check(arena)


def test_stride_that_cannot_fit_every_superpage_is_refused():
    arena = make_arena(superpages=2, sp_bytes=10 * ROW)
    with pytest.raises(ValueError, match="superpage"):
        make_class(arena, "x", stride=7 * ROW)
    with pytest.raises(ValueError, match="superpage"):
        make_class(arena, "y", stride=11 * ROW)


def test_one_order_across_classes_evicts_the_oldest_run_of_any_class():
    arena, big, small, b, s = two_classes(superpages=2)
    store(arena, s, [key(1), key(2), key(3), key(4)], b"S")  # fills one superpage
    store(arena, b, [key(1)], b"B1")                          # the other
    res = arena.prepare_store(b, [key(2)], b"B2")             # S is oldest
    assert res is not None and res[2] == []                   # not this class's keys
    assert not small.index and arena.counters["cross_class_runs_evicted"] == 1
    check(arena)


def test_cross_class_store_is_all_or_nothing():
    arena, big, small, b, s = two_classes(superpages=2)
    store(arena, s, [key(1), key(2)], b"S1")
    store(arena, b, [key(1)], b"B1")
    store(arena, s, [key(3), key(4)], b"S2")
    assert arena.lookup(s, [key(3)]) == [P.HIT]  # S2 is held
    assert arena.lookup(b, [key(1)]) == [P.HIT]  # and so is B1
    before = sorted(small.index), sorted(big.index), [r.tag for r in arena.order]
    # Evicting S1 frees two rows of a superpage S2 still uses: no superpage.
    assert arena.prepare_store(b, [key(2)], b"B2") is None
    assert (sorted(small.index), sorted(big.index), [r.tag for r in arena.order]) == before
    check(arena)


def test_admit_fraction_counts_other_classes_idle_runs_only():
    arena, big, small, b, s = two_classes(superpages=2)
    big.cross_class_idle_sec = 60.0
    store(arena, s, [key(1), key(2), key(3), key(4)], b"S")  # fills one superpage
    store(arena, b, [key(1)], b"B1")                          # the other
    assert big.available() == 0
    # S is another class's run, but still active: nothing to admit.
    assert big.admit_fraction() == 0.0
    # Once S has been idle long enough it is space for the big class.
    arena.clock.t += 61.0
    assert big.admit_fraction() == pytest.approx(1 / big.num_rows)
    # The small class may take the big class's idle run (one superpage), but
    # never counts its own old run S: that is live RAM-only data.
    small.cross_class_idle_sec = 60.0
    assert small.admit_fraction() == pytest.approx(
        (small.available() + SP // ROW) / small.num_rows)
    # A held run frees nothing.
    assert arena.lookup(b, [key(1)]) == [P.HIT]
    small._admit_cache = None
    assert small.admit_fraction() == pytest.approx(small.available() / small.num_rows)
    arena.end_session(b)
    check(arena)


def test_admit_fraction_counts_this_class_dead_runs():
    arena = make_arena(superpages=4)
    cls = make_class(arena, state_groups=(1,))
    c = client(cls, 1)
    store(arena, c, [key(1, 0), key(2, 0)], b"dead")   # no state: unrestorable
    store(arena, c, [key(3, 0), key(3, 1)], b"live")   # a complete boundary
    assert cls.admit_fraction() == 0.0
    arena.clock.t += S.DEAD_RUN_GRACE_SEC + 1
    cls._admit_cache = None
    assert cls.admit_fraction() == pytest.approx(2 / 4)
    check(arena)


def test_compaction_frees_a_superpage_and_moving_rows_answer_pending():
    arena, big, small, b, s = two_classes(superpages=3)
    for i in range(8):  # two small superpages
        store(arena, s, [key(i)], f"S{i}".encode())
    store(arena, b, [key(100)], b"B0")
    for i in (1, 3, 5, 7):  # each half empty now
        arena.evict_run(small.runs[f"S{i}".encode()])
    assert len(small.sp_free) == 2 and small.free_rows == 4
    # A big store needs a superpage: compacting the small class makes one.
    assert arena.prepare_store(b, [key(101)], b"B1") is None
    assert big.counters["store_waits_compaction"] == 1
    assert small.counters["rows_moving"] == 2
    moving = [k for k, e in small.index.items() if e.moving is not None]
    assert arena.lookup(s, moving) == [P.HIT_PENDING] * 2
    arena.mover.run_pending()
    assert len(small.sp_free) == 1 and len(arena.free_sp) == 1
    assert arena.lookup(s, moving) == [P.HIT] * 2
    arena.end_session(s)
    assert store(arena, b, [key(101)], b"B1")[0] == [key(101)]
    check(arena)


def test_compaction_keeps_every_byte(tmp_path):
    path = str(tmp_path / "arena.mmap")
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    os.ftruncate(fd, 6 * SP)
    mm = mmap.mmap(fd, 6 * SP)
    arena, big, small, b, s = two_classes(superpages=6, path=path)
    arena.open_file(fd, mm)
    rng = random.Random(7)
    digests = {}
    for i in range(24):
        res = arena.prepare_store(s, [key(i)], f"S{i}".encode())
        row = res[1][0]
        data = rng.randbytes(ROW)
        mm[row * ROW:(row + 1) * ROW] = data
        digests[key(i)] = hashlib.sha256(data).hexdigest()
        arena.complete_store(s, res[0], True)
    for i in range(24):
        if i % 3:
            arena.evict_run(small.runs[f"S{i}".encode()])
            digests.pop(key(i))
    started = arena.compact(4, exclude=big)
    assert started >= 3
    arena.mover.run_pending()
    assert small.counters["rows_moved"] >= 3
    for k, digest in digests.items():
        row = small.index[k].row
        assert hashlib.sha256(mm[row * ROW:(row + 1) * ROW]).hexdigest() == digest
    assert len(small.sp_free) == 2  # 8 rows left, packed into two superpages
    check(arena)
    arena.close()


def test_a_failed_move_gives_up_draining_its_superpage():
    arena, big, small, b, s = two_classes(superpages=3)
    for i in range(8):  # two small superpages
        store(arena, s, [key(i)], f"S{i}".encode())
    store(arena, b, [key(100)], b"B0")
    for i in (1, 3, 5, 7):
        arena.evict_run(small.runs[f"S{i}".encode()])
    assert arena.prepare_store(b, [key(101)], b"B1") is None  # starts compaction
    assert len(small.draining) == 1

    def fail(job):
        raise OSError("injected")
    arena.mover._execute = fail
    arena.mover.run_pending()
    assert small.counters["move_failed"] == 2
    assert not small.draining  # back to allocatable, no longer counted as supply
    check(arena)
    # Compaction is planned again instead of waiting on a superpage that
    # never empties.
    arena.mover._execute = arena._move_execute
    assert arena.prepare_store(b, [key(101)], b"B1") is None
    assert len(small.draining) == 1
    arena.mover.run_pending()
    assert store(arena, b, [key(101)], b"B1")[0] == [key(101)]
    check(arena)


def test_held_rows_are_not_moved():
    arena, big, small, b, s = two_classes(superpages=3)
    for i in range(8):
        store(arena, s, [key(i)], f"S{i}".encode())
    for i in (1, 2, 3, 5, 6, 7):
        arena.evict_run(small.runs[f"S{i}".encode()])
    arena.lookup(s, [key(0), key(4)])  # one held row in each small superpage
    assert arena.compact(1, exclude=big) == 0
    arena.end_session(s)
    assert arena.compact(1, exclude=big) == 1


# --- pins ---


def test_pin_in_place_for_a_one_row_superpage_and_never_evicted():
    arena, big, small, b, s = two_classes(superpages=3, pin_superpages=1)
    store(arena, b, [key(1)])
    assert arena.pin(big, [key(1)]) == 1
    assert big.index[key(1)].pinned and arena.pin_sp == 1
    store(arena, b, [key(2)])
    store(arena, b, [key(3)])
    assert arena.prepare_store(b, [key(4)], key(4))[2] == [key(2)]
    assert arena.lookup(b, [key(1)]) == [P.HIT]
    check(arena)


def test_pin_of_a_small_row_moves_it_into_the_pin_zone():
    arena, big, small, b, s = two_classes(superpages=3, pin_superpages=1)
    store(arena, s, [key(1), key(2)])
    assert arena.pin(small, [key(1)]) == 1
    assert arena.lookup(s, [key(1)]) == [P.HIT_PENDING]
    arena.mover.run_pending()
    e = small.index[key(1)]
    assert e.pinned and arena.sp_pin[small.sp_of(e.row)]
    check(arena)


def test_pin_zone_is_capped():
    arena, big, small, b, s = two_classes(superpages=3, pin_superpages=1)
    store(arena, b, [key(1)])
    store(arena, b, [key(2)])
    assert arena.pin(big, [key(1), key(2)]) == 1
    assert big.counters["pin_zone_full"] == 1 and not big.index[key(2)].pinned


def test_pinned_chunk_does_not_keep_its_run_from_being_evicted():
    arena, cls, a, _ = single(rows=3, pin_superpages=1)
    store(arena, a, [key(1), key(2)], b"X")
    arena.pin(cls, [key(1)])
    store(arena, a, [key(3)], b"Y")
    res = arena.prepare_store(a, [key(4)], b"Z")
    assert res is not None and res[2] == [key(2)] and cls.index[key(1)].pinned
    check(arena)


def test_unpinned_chunk_in_no_run_goes_first():
    arena, cls, a, _ = single(rows=3, pin_superpages=1)
    store(arena, a, [key(1)], b"X")
    arena.pin(cls, [key(1)])
    arena.evict_run(cls.runs[b"X"])  # the run goes, the pinned chunk stays
    assert key(1) in cls.index and not cls.index[key(1)].runs
    store(arena, a, [key(2)], b"Y")
    assert arena.unpin(cls, None) == 1
    assert store(arena, a, [key(3), key(4)], b"Z")[2] == [key(1)]
    check(arena)


def cold_arena(tmp_path, superpages=3, pin_superpages=2, clock=None):
    path = str(tmp_path / "arena.mmap")
    size = superpages * ROW
    fd = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    os.ftruncate(fd, size)
    arena = make_arena(superpages=superpages, pin_superpages=pin_superpages,
                       quarantine_sec=0.0, path=path, clock=clock)
    arena.open_file(fd, mmap.mmap(fd, size))
    cls = make_class(arena)
    cls.cold = S.ColdTier(cls, {"dir": str(tmp_path / "cold"), "max_bytes": 100 * ROW,
                                "sync": True})
    cls.cold.scan()
    for k in sorted(cls.pins):
        cls.cold.start_promotion(k)
    return arena, cls


def stored(arena, c, k, byte):
    res = arena.prepare_store(c, [k], k)
    row = c.cls.index[k].row
    arena.mm[row * ROW:(row + 1) * ROW] = bytes([byte]) * ROW
    arena.complete_store(c, res[0], True)


def read(arena, c, k):
    row = load(arena, c, [k])[0]
    return bytes(arena.mm[row * ROW:(row + 1) * ROW])


def test_pin_is_persisted_and_restored_into_the_pin_zone_after_restart(tmp_path):
    arena, cls = cold_arena(tmp_path)
    a = client(cls, 1)
    stored(arena, a, key(1), 5)
    assert arena.pin(cls, [key(1)]) == 1
    assert os.path.getsize(cls.cold.path(key(1))) == ROW
    assert cls.counters["cold_written"] == 1
    arena.close()
    os.unlink(tmp_path / "arena.mmap")
    arena2, cls2 = cold_arena(tmp_path)
    e = cls2.index[key(1)]
    assert e.pinned and arena2.sp_pin[cls2.sp_of(e.row)]
    assert read(arena2, client(cls2, 2), key(1)) == bytes([5]) * ROW
    assert arena2.unpin(cls2, None) == 1 and not cls2.pins
    arena2.close()


def test_pin_before_store_applies_when_stored(tmp_path):
    arena, cls = cold_arena(tmp_path)
    a = client(cls, 1)
    assert arena.pin(cls, [key(1)]) == 0
    stored(arena, a, key(1), 4)
    e = cls.index[key(1)]
    assert e.pinned and arena.sp_pin[cls.sp_of(e.row)]  # allocated in the pin zone
    assert os.path.exists(cls.cold.path(key(1)))
    arena.close()


def test_pin_recent_pins_what_was_used_in_the_window(tmp_path):
    clock = Clock()
    arena, cls = cold_arena(tmp_path, superpages=4, pin_superpages=2, clock=clock)
    a = client(cls, 1)
    stored(arena, a, key(1), 1)                 # the prime, at t
    arena.prepare_store(a, [key(2)], key(2))    # still being written at pin time
    clock.t += 30
    stored(arena, a, key(3), 3)                 # unrelated traffic after the prime
    assert arena.pin_recent(cls, 31, until_ago=20) == 1
    assert cls.index[key(1)].pinned and not cls.index[key(3)].pinned
    arena.complete_store(a, [key(2)], True)
    assert cls.index[key(2)].pinned
    arena.close()


def test_pinned_miss_on_disk_is_read_back(tmp_path):
    arena, cls = cold_arena(tmp_path)
    a = client(cls, 1)
    stored(arena, a, key(1), 9)
    arena.pin(cls, [key(1)])
    arena.unpin(cls, [key(1)])
    arena.evict_run(next(iter(cls.index[key(1)].runs)))
    cls.pins.add(key(1))  # pinned again while not resident
    assert arena.lookup(a, [key(1)]) == [P.HIT_PENDING]  # read synchronously here
    assert arena.lookup(a, [key(1)]) == [P.HIT]
    arena.end_session(a)
    assert read(arena, a, key(1)) == bytes([9]) * ROW
    arena.close()


def test_cold_rejects_other_write_policies(tmp_path):
    arena = make_arena()
    cls = make_class(arena)
    with pytest.raises(ValueError, match="pins"):
        S.ColdTier(cls, {"dir": str(tmp_path), "max_bytes": 1, "write_policy": "reused"})


# --- classes and layouts ---


def test_layout_change_is_refused_while_attached_and_replaces_an_idle_class():
    clock = Clock()
    arena = make_arena(superpages=2, clock=clock)
    cls = make_class(arena)
    a = client(cls, 1)
    store(arena, a, [key(1)])
    with pytest.raises(ValueError, match="layout"):
        arena.class_for("p", {}, "other", ROW, ())
    cls.clients -= 1
    new = arena.class_for("p", {}, "other", ROW, ())
    assert new is not cls and not new.index
    # The old rows wait out the quarantine before their superpage is free.
    assert len(arena.free_sp) == 1
    clock.t += 31
    arena.maintain()
    assert len(arena.free_sp) == 2
    check(arena)


def test_adopted_arena_hands_out_nothing_until_the_quarantine_ends():
    clock = Clock()
    arena, cls, a, _ = single(rows=2, clock=clock)
    arena.adopt()
    assert arena.prepare_store(a, [key(1)], b"t") is None
    clock.t += 31
    assert arena.prepare_store(a, [key(1)], b"t")[0] == [key(1)]


# --- randomised traffic ---


def test_mixed_traffic_keeps_every_invariant():
    rng = random.Random(1)
    clock = Clock()
    arena = make_arena(superpages=12, sp_bytes=SP, pin_superpages=2, clock=clock,
                       quarantine_sec=5.0)
    big = make_class(arena, "big", stride=SP)
    small = make_class(arena, "small", stride=ROW, state_groups=(ST1,))
    clients = [client(big, 1), client(big, 2), client(small, 3), client(small, 4)]
    for step in range(3000):
        clock.t += 0.1
        c = rng.choice(clients)
        conv = rng.randrange(10)
        keys = [key(conv * 10 + j, rng.choice((ATT, ST1))) for j in range(rng.randrange(1, 4))]
        hits = [k for k, r in zip(keys, arena.lookup(c, keys)) if r == P.HIT]
        loaded = hits if hits and rng.random() < 0.7 else []
        if loaded:
            arena.prepare_load(c, loaded)
        res = arena.prepare_store(c, keys, f"r{step}".encode(), whole=rng.random() < 0.2)
        arena.end_session(c)
        if loaded:
            arena.complete_load(c, loaded)
        if res is not None:
            arena.complete_store(c, res[0], rng.random() > 0.05)
            if rng.random() < 0.5:
                arena.merge_tag(c, f"r{step}".encode(), f"conv{conv}".encode())
        if rng.random() < 0.02:
            arena.pin(c.cls, keys[:1])
        if rng.random() < 0.01:
            arena.unpin(c.cls, None)
        if rng.random() < 0.05:
            arena.mover.run_pending()
        if rng.random() < 0.005:
            arena.release_client(c)
        if step % 50 == 0:
            check(arena)
    arena.mover.run_pending()
    for c in clients:
        arena.release_client(c)
    clock.t += 10
    arena.maintain()
    arena.mover.run_pending()
    check(arena)
    for cls in (big, small):
        assert all(e.refs == (1 if e.pinned else 0) for e in cls.index.values())
    assert big.counters["chunks_stored"] and small.counters["chunks_stored"]
    assert arena.counters["cross_class_runs_evicted"]
