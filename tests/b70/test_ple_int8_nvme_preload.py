# SPDX-License-Identifier: Apache-2.0
"""B70-PLE-PRELOAD: the rank's owned INT8 PLE rows preloaded into the pinned row cache."""
import importlib.util
import json
import logging
import os
import struct
import sys
import tempfile

import numpy as np
import pytest

HERE = os.path.dirname(__file__)
spec = importlib.util.spec_from_file_location(
    "ple_nvme", os.path.join(HERE, "../../vllm/models/qwen4_exp/nvidia/ple_nvme.py"))
ple_nvme = importlib.util.module_from_spec(spec)
sys.modules["ple_nvme"] = ple_nvme
spec.loader.exec_module(ple_nvme)

DIM = 164
PAGE = ple_nvme.PAGE


def _table(tmp, rows, pad=0):
    """A safetensors INT8 table; ``pad`` header bytes move the data off page alignment."""
    rng = np.random.default_rng(11)
    data = rng.integers(0, 256, size=(rows, DIM), dtype=np.uint8)
    hdr = json.dumps({"table": {"dtype": "U8", "shape": [rows, DIM], "data_offsets": [0, data.nbytes]},
                      "__metadata__": {"format": "lumnus-ple-int8-rowscale/v1"}}).encode()
    hdr += b" " * (pad + (-(len(hdr) + pad) % 8))
    path = os.path.join(tmp, "t.safetensors")
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr)) + hdr + data.tobytes())
    return path, data


def _server(path, tp_start, tp_end, cap, *, stats_on=True):
    start, shape, _ = ple_nvme.safetensors_tensor_location(path, "table")
    store = ple_nvme.PleNvmeRowStore(path, start, DIM, shape[0], io_threads=2, reader="py")
    slab = np.zeros((cap, DIM), np.uint8)
    cache = ple_nvme.PleRowCache(tp_end - tp_start, cap, DIM, slab=slab)
    stats = ple_nvme.PleNvmeStats(0, 1e9, stats_on)
    srv = ple_nvme.PleNvmeServer(tp_start=tp_start, tp_end=tp_end, multipliers=np.ones(3, np.int64),
                                 sizes=np.array([3000, 3000]), offsets=np.array([0, 3000]),
                                 eos_token_id=0, heads_per_ngram=1, cache=cache, store=store,
                                 stats=stats)
    return srv, store, cache, slab, stats


@pytest.mark.parametrize("tp", [(0, 3000), (1234, 4321), (4321, 6000)])
@pytest.mark.parametrize("chunk,threads", [(PAGE, 1), (2 * PAGE, 3), (64 << 20, 4)])
def test_preload_all_rows_resident_zero_reads(tp, chunk, threads):
    tp_start, tp_end = tp
    owned = tp_end - tp_start
    with tempfile.TemporaryDirectory() as tmp:
        path, data = _table(tmp, 6000, pad=37)
        srv, store, cache, slab, stats = _server(path, tp_start, tp_end, cap=owned + 100)
        info = ple_nvme.preload_cache(cache, store, tp_start, threads=threads, chunk_bytes=chunk)
        assert info is not None and info["rows"] == owned and info["bytes"] == owned * DIM
        assert info["direct"]  # ext4 /tmp takes O_DIRECT; the bounce handles misalignment
        # Every owned row is resident, slot r = row r, bytes equal the file.
        assert cache.all_resident
        assert np.array_equal(cache.row2slot, np.arange(owned))
        assert np.array_equal(cache.slot2row[:owned], np.arange(owned))
        assert (cache.slot2row[owned:] == -1).all()
        assert np.array_equal(slab[:owned], data[tp_start:tp_end])
        assert not slab[owned:].any()
        # Lookups after preload: zero NVMe reads, zero misses, exact gathers.
        def boom(*a, **k):
            raise AssertionError("NVMe read after preload")
        store.read_rows = boom
        rng = np.random.default_rng(3)
        epoch_before = cache.epoch.copy()
        for _ in range(20):
            ids = rng.integers(0, 6000, size=(64, 2))
            slots, mask = srv.resolve_ids(ids)
            assert np.array_equal(mask, (ids >= tp_start) & (ids < tp_end))
            t, c = np.nonzero(mask)
            assert np.array_equal(slab[slots[t, c]], data[ids[t, c]])
        assert stats.total_reads == 0 and stats.reads == 0 and stats.misses == 0
        assert stats.hits == stats.lookups > 0
        assert np.array_equal(cache.epoch, epoch_before)  # pins survive lookups
        store.close()


def test_preload_miss_path_never_displaces_preloaded_rows():
    """If a miss ever reached install, it takes a free tail slot, never a pinned one."""
    with tempfile.TemporaryDirectory() as tmp:
        path, data = _table(tmp, 1000)
        srv, store, cache, slab, _ = _server(path, 0, 1000, cap=1010)
        assert ple_nvme.preload_cache(cache, store, 0, threads=2, chunk_bytes=PAGE) is not None
        cache.begin_step()
        # Forge a miss: drop row 5 from the map (its slot stays pinned).
        cache.row2slot[5] = -1
        slots = cache.lookup(np.array([5, 6]))
        assert slots[0] == -1 and slots[1] == 6
        new = cache.install(np.array([5]), data[5:6])
        assert 1000 <= int(new[0]) < 1010
        assert np.array_equal(slab[:1000][np.arange(1000) != 5], data[np.arange(1000) != 5])
        assert np.array_equal(slab[new[0]], data[5])
        # Pinned slots cannot be allocated: only the 9 remaining tail slots can.
        cache.begin_step()
        with pytest.raises(RuntimeError, match="evictable"):
            cache.allocate(11)
        store.close()


def test_preload_falls_back_when_capacity_too_small(caplog):
    with tempfile.TemporaryDirectory() as tmp:
        path, data = _table(tmp, 3000)
        srv, store, cache, slab, stats = _server(path, 0, 3000, cap=512)
        with caplog.at_level(logging.WARNING, logger=ple_nvme.logger.name):
            assert ple_nvme.preload_cache(cache, store, 0, rank=2) is None
        warnings = [r for r in caplog.records if "preload (rank 2) skipped" in r.getMessage()]
        assert len(warnings) == 1
        assert not cache.all_resident and (cache.row2slot == -1).all() and not slab.any()
        # Normal cache behaviour: misses read from the file, CLOCK evicts.
        rng = np.random.default_rng(5)
        for _ in range(6):
            ids = rng.integers(0, 3000, size=(40, 2))
            slots, mask = srv.resolve_ids(ids)
            t, c = np.nonzero(mask)
            assert np.array_equal(slab[slots[t, c]], data[ids[t, c]])
        assert stats.total_reads > 0
        store.close()


def test_preload_range_buffered_and_retry():
    with tempfile.TemporaryDirectory() as tmp:
        path, data = _table(tmp, 500, pad=5)
        start, _, _ = ple_nvme.safetensors_tensor_location(path, "table")
        out = np.empty((300, DIM), np.uint8)
        assert not ple_nvme.preload_range(path, start + 100 * DIM, out, direct=False,
                                          threads=2, chunk_bytes=PAGE)
        assert np.array_equal(out, data[100:400])
        # One transient error and one short read per chunk are retried and still exact.
        state = {"n": 0}

        def flaky(fd, bufs, pos):
            state["n"] += 1
            if state["n"] % 3 == 1:
                raise OSError(5, "EIO")
            if state["n"] % 3 == 2:
                view = bufs[0][:PAGE]
                return os.preadv(fd, [view], pos)
            return os.preadv(fd, bufs, pos)
        out2 = np.empty((300, DIM), np.uint8)
        ple_nvme.preload_range(path, start + 100 * DIM, out2, direct=True, threads=1,
                               chunk_bytes=4 * PAGE, preadv=flaky)
        assert np.array_equal(out2, data[100:400])

        def dead(fd, bufs, pos):
            raise OSError(5, "EIO")
        with pytest.raises(RuntimeError, match="preload read failed"):
            ple_nvme.preload_range(path, start, np.empty((10, DIM), np.uint8), preadv=dead)
