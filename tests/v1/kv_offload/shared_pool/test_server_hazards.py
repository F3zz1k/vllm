# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eviction hazards of the per-model exclusive pool (joint-kv-design.md
section 2, P0), each written as what a correct arena does. Their first version
(the commit that added this file) failed on the pre-arena server, which is
what reproduced each hazard there.
"""

from .test_server import P, check, client, key, load, make_arena, make_class


def arena4():
    arena = make_arena(superpages=4)
    cls = make_class(arena)
    return arena, cls, client(cls, 1), client(cls, 2)


def put(arena, c, keys, tag):
    res = arena.prepare_store(c, keys, tag)
    assert res is not None
    arena.complete_store(c, res[0], True)
    return res


def test_a_chain_is_never_evicted_head_first():
    arena, cls, a, _ = arena4()
    put(arena, a, [key(1), key(2), key(3)], b"chain")
    put(arena, a, [key(9)], b"other")
    put(arena, a, [key(10)], b"new")                 # needs one row
    present = [k in cls.index for k in (key(1), key(2), key(3))]
    assert present in ([True] * 3, [False] * 3)      # whole or gone, never headless
    check(arena)


def test_run_with_a_held_chunk_is_not_evicted_in_part():
    arena, cls, a, b = arena4()
    put(arena, a, [key(1), key(2)], b"X")
    put(arena, a, [key(3), key(4)], b"Y")
    assert arena.lookup(b, [key(1)]) == [P.HIT]      # b holds X's head
    put(arena, a, [key(5)], b"Z")
    # X may not lose its tail while its head survives: Y goes whole instead.
    assert key(2) in cls.index and key(3) not in cls.index and key(4) not in cls.index
    arena.end_session(b)
    check(arena)


def test_shared_head_is_not_stripped_from_the_older_run():
    arena, cls, a, _ = arena4()
    put(arena, a, [key(1), key(2)], b"X")            # X = root + turn
    put(arena, a, [key(1), key(3)], b"Y")            # Y shares the root
    put(arena, a, [key(2)], b"X")                    # X's turn written back: X newest
    put(arena, a, [key(5), key(6)], b"Z")            # needs one more row: Y is oldest
    # Evicting Y must leave X restorable, so the shared root stays.
    assert key(1) in cls.index and key(2) in cls.index
    check(arena)


def test_load_demotes_only_the_loading_run():
    arena, cls, a, b = arena4()
    put(arena, a, [key(1), key(2)], b"X")
    put(arena, a, [key(1), key(3)], b"Y")            # Y shares the root
    load(arena, b, [key(1), key(2)])                 # b continues X
    put(arena, a, [key(5), key(6)], b"Z")            # needs one more row
    # X is on b's GPU now; Y is not, so Y must survive.
    assert key(3) in cls.index and key(2) not in cls.index
    check(arena)


def test_released_loader_promotes_its_runs_back():
    arena, cls, a, b = arena4()
    put(arena, a, [key(1), key(2)], b"X")
    put(arena, a, [key(3), key(4)], b"Y")
    load(arena, b, [key(1), key(2)])                 # X demoted: b's GPU holds it
    arena.release_client(b)                          # b is gone, and its GPU copy too
    put(arena, a, [key(5), key(6)], b"Z")
    # The RAM copy of X is now the only one: Y is older and goes first.
    assert key(1) in cls.index and key(2) in cls.index
    check(arena)
