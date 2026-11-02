# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eviction hazards of the exclusive pool (joint-kv-design.md section 2, P0).

Each test states what a correct pool does; on the pre-arena server every one
of them fails, which is what reproduces the hazard.
"""

import pytest

from .test_server import P, S, Clock, client, key

HAZARD = pytest.mark.xfail(strict=True, reason="hazard reproduced on the pre-arena server")


def make_pool(num_chunks):
    return S.Pool("p", "layout", 4096, num_chunks, path=None, clock=Clock(), exclusive=True)


def put(pool, c, keys, tag=None):
    res = pool.prepare_store(c, keys, tag)
    assert res is not None
    pool.complete_store(c, res[0], True)
    return res


def load(pool, c, keys):
    assert pool.lookup(c, keys) == [P.HIT] * len(keys)
    pool.prepare_load(c, keys)
    pool.end_session(c)
    pool.complete_load(c, keys)


@HAZARD
def test_untagged_chain_is_not_evicted_head_first():
    pool, a = make_pool(4), client(1)
    put(pool, a, [key(1), key(2), key(3)])          # one chain, stored untagged
    put(pool, a, [key(9)])
    put(pool, a, [key(10)])                          # needs one chunk
    # Evicting the head makes the rest of the chain unreachable.
    assert key(1) in pool.index


@HAZARD
def test_run_with_a_held_chunk_is_not_evicted_in_part():
    pool, a, b = make_pool(4), client(1), client(2)
    put(pool, a, [key(1), key(2)], key(2))           # run X
    put(pool, a, [key(3), key(4)], key(4))           # run Y
    assert pool.lookup(b, [key(1)]) == [P.HIT]       # b holds X's head
    put(pool, a, [key(5)], key(5))
    # X may not lose its tail while its head survives: Y goes whole instead.
    assert key(2) in pool.index
    pool.end_session(b)


@HAZARD
def test_shared_head_is_not_stripped_from_the_older_run():
    pool, a = make_pool(4), client(1)
    put(pool, a, [key(1), key(2)], key(2))           # X = root + turn
    put(pool, a, [key(1), key(3)], key(3))           # Y shares the root
    put(pool, a, [key(2)], key(2))                   # X's turn written back: X newest
    put(pool, a, [key(5), key(6)], key(6))           # needs one more chunk: Y is oldest
    # Evicting Y must leave X restorable, so the shared root stays.
    assert key(1) in pool.index and key(2) in pool.index


@HAZARD
def test_load_demotes_only_the_loading_run():
    pool, a, b = make_pool(4), client(1), client(2)
    put(pool, a, [key(1), key(2)], key(2))           # X
    put(pool, a, [key(1), key(3)], key(3))           # Y shares the root
    load(pool, b, [key(1), key(2)])                  # b continues X
    put(pool, a, [key(5), key(6)], key(6))           # needs one more chunk
    # X is on b's GPU now; Y is not, so Y must survive.
    assert key(3) in pool.index


@HAZARD
def test_released_loader_promotes_its_runs_back():
    pool, a, b = make_pool(4), client(1), client(2)
    put(pool, a, [key(1), key(2)], key(2))           # X
    put(pool, a, [key(3), key(4)], key(4))           # Y
    load(pool, b, [key(1), key(2)])                  # X demoted: b's GPU holds it
    pool.release_client(b)                           # b is gone, and its GPU copy too
    put(pool, a, [key(5), key(6)], key(6))
    # The RAM copy of X is now the only one: Y is older and goes first.
    assert key(1) in pool.index and key(2) in pool.index
