# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Server micro-benchmark: operation latency on a full arena of the
production geometry (joint-kv-design.md P2 gate: p99 under 5 ms with 50 keys
per block). Index only: no arena file, row copies not timed.

Run with -s to see the figures."""

import random
import time

from .test_server import P, S, Clock, client, key

ARENA = 94_489_280_512
SUPERPAGE = 55_705_600         # one canonical 27B row
FN_ROW = 3_276_800             # Flash-Next row, 17 per superpage
FN_GROUPS = 50                 # keys per Flash-Next block
B27_GROUPS = 4
FN_STATE = tuple(range(1, FN_GROUPS))  # all but group 0 hold recurrent state
B27_STATE = (1, 2, 3)


def block(conv: int, i: int, groups: int) -> list[bytes]:
    return [key(conv * 1000 + i, g) for g in range(groups)]


def test_full_arena_op_latency():
    rng = random.Random(3)
    clock = Clock()
    arena = S.Arena(None, ARENA, SUPERPAGE, 4 * SUPERPAGE, clock=clock)
    fn = arena.class_for("qwen3.8-flash-next", {}, "fn", FN_ROW, FN_STATE)
    b27 = arena.class_for("qwen3.8-27b", {}, "27b", SUPERPAGE, B27_STATE)
    clients = {fn: [client(fn, 1)], b27: [client(b27, 2), client(b27, 3)]}
    convs = {fn: [], b27: []}
    groups = {fn: FN_GROUPS, b27: B27_GROUPS}
    next_conv = [0]

    def new_conv(cls, blocks):
        conv = next_conv[0]
        next_conv[0] += 1
        c = rng.choice(clients[cls])
        for i in range(blocks):
            res = arena.prepare_store(c, block(conv, i, groups[cls]), f"r{conv}".encode())
            if res is None:
                return False
            arena.complete_store(c, res[0], True)
        arena.merge_tag(c, f"r{conv}".encode(), f"c{conv}".encode())
        convs[cls].append((conv, blocks))
        return True

    # Fill: Flash-Next runs of 4 blocks over three quarters of the superpages
    # (the most rows, so the largest index), 27B runs of 10 blocks after.
    while arena.free_sp:
        if len(fn.sp_free) < 0.75 * arena.num_sp:
            new_conv(fn, 4)
        else:
            new_conv(b27, 10)
    arena.mover.run_pending()
    assert len(arena.free_sp) < 0.01 * arena.num_sp  # full (compactions may free a few)
    rows = len(fn.index) + len(b27.index)

    timings: dict[str, list[float]] = {}

    def timed(name, fn_, *args):
        t0 = time.perf_counter()
        out = fn_(*args)
        timings.setdefault(name, []).append(time.perf_counter() - t0)
        return out

    for step in range(3000):
        clock.t += 0.05
        cls = fn if rng.random() < 0.5 else b27
        c = rng.choice(clients[cls])
        g = groups[cls]
        r = rng.random()
        if r < 0.4 and convs[cls]:
            conv, blocks = rng.choice(convs[cls][-200:])
            keys = [k for i in range(blocks) for k in block(conv, i, g)][: 50 * 8]
            res = timed("lookup_prefix", arena.lookup, c, keys, True)
            hits = keys[: len(res)] if all(x == P.HIT for x in res) else []
            if hits:
                timed("prepare_load", arena.prepare_load, c, hits)
            timed("end_session", arena.end_session, c)
            if hits:
                timed("complete_load", arena.complete_load, c, hits)
        else:
            conv = next_conv[0]
            next_conv[0] += 1
            keys = block(conv, 0, g)
            res = timed("prepare_store", arena.prepare_store, c, keys, f"r{conv}".encode())
            if res is not None:
                timed("complete_store", arena.complete_store, c, res[0], True)
                timed("merge_tag", arena.merge_tag, c, f"r{conv}".encode(), f"c{conv}".encode())
                convs[cls].append((conv, 1))
        if step % 10 == 0:
            timed("drain", arena.mover.run_pending)
            timed("stats", cls.stats)  # each engine polls it once a second

    every = sorted(t for ts in timings.values() for t in ts)
    p99 = every[int(0.99 * len(every))]
    print(f"\nfull arena: {arena.num_sp} superpages, {rows} rows at fill, "
          f"{len(arena.order)} runs, {len(every)} ops")
    for name, ts in sorted(timings.items()):
        ts.sort()
        print(f"  {name:15s} n={len(ts):5d} p50={ts[len(ts) // 2] * 1e3:7.3f} ms "
              f"p99={ts[int(0.99 * len(ts))] * 1e3:7.3f} ms max={ts[-1] * 1e3:7.3f} ms")
    print(f"  all ops p99 {p99 * 1e3:.3f} ms; cross-class evictions "
          f"{arena.counters['cross_class_runs_evicted']}, superpages compacted "
          f"{arena.counters['superpages_compacted']}")
    assert p99 < 5e-3
