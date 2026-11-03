# SPDX-License-Identifier: Apache-2.0
"""BlockPool hooks used by the offloading write-back band (2026-11-03)."""

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id


def test_listeners_see_reused_and_revived_hashed_blocks():
    pool = BlockPool(num_gpu_blocks=5, enable_caching=True, hash_block_size=16)
    evicted, revived = [], []
    pool.evict_listener = lambda b: evicted.append((b.block_id, bytes(b.block_hash)))
    pool.revive_listener = lambda b: revived.append(b.block_id)
    a, b = pool.get_new_blocks(2)
    assert pool.num_popped == 2 and not evicted
    for blk, name in ((a, b"a"), (b, b"b")):
        pool._insert_block_hash(make_block_hash_with_group_id(BlockHash(name), 0), blk, 16)
    pool.free_blocks([a, b])
    pool.touch([a])                                   # a prefix hit takes a back
    assert revived == [a.block_id]
    got = pool.get_new_blocks(3)                      # 2 unhashed, then b
    assert b in got and [e[0] for e in evicted] == [b.block_id]
    assert evicted[0][1] == bytes(make_block_hash_with_group_id(BlockHash(b"b"), 0))
    assert b.block_hash is None                       # the hash went after the hook
