# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared host-RAM KV pool server.

One process per host owns, for each named pool, a ``/dev/shm`` region of
fixed-size chunks and the index over it. Every vLLM instance whose KV layout
matches the pool maps the same region, so a chunk stored by one instance can
be loaded by any other with a single host-to-device copy, and each chunk is
held in RAM once however many instances use it.

The server holds all index state: which key lives in which chunk, whether the
chunk is still being written, how many readers hold it, and LRU order over the
chunks nobody holds. Clients (the scheduler of each vLLM instance) call it over
a UNIX socket. The worker processes never talk to it except to learn the region
path; they copy to and from the chunk ids their scheduler hands them.

Eviction order. Keys arrive in prefix order (a request's first chunk first),
and a prefix lookup stops at the first miss, so evicting the head of a
conversation makes every later chunk of it unreachable. Chunks therefore enter
the LRU tail first: of the chunks one call releases, the deepest is the oldest
and is evicted first, and the head goes last.

Holds. A lookup HIT pins the chunk for the calling client until it ends its
session (the scheduler's manager lock region, one scheduler step at most), so
another instance cannot evict a chunk between this instance's lookup and the
prepare_load that pins it for the transfer.

Crash safety. When a client disconnects, its holds and read references are
dropped and the chunks it was still writing are removed. Their chunk ids wait
in quarantine before reuse, so a transfer still running in a dying instance
cannot overwrite data a new owner has written.

Cold tier (optional, per pool). Nothing is written to disk when a chunk is
stored. With ``write_policy: "pins"`` (the default) only pinned chunks are ever
written, so the disk changes only on demand. With ``"reused"``, a chunk about to
be evicted from RAM is also written if it has been loaded at least
``min_loads`` times, it is not on disk already, and the hourly write budget
allows it. A
lookup that misses in RAM but finds the key on disk reads the file straight
into a free chunk and answers HIT_PENDING until the read completes. Pinned
chunks are never evicted, are written to disk at once, and are read back into
RAM when the pool is created, so they survive a reboot. The cold directory is
capped and trimmed least recently used first, never deleting a pinned chunk.

Standard library only: run it with the host Python, for example
``python3 server.py --config pools.json``.
"""

import argparse
import collections
import json
import logging
import mmap
import os
import queue
import selectors
import signal
import socket
import sys
import threading
import time
from dataclasses import dataclass, field

try:
    from . import protocol as P
except ImportError:  # run as a script
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import protocol as P  # type: ignore[no-redef]

logger = logging.getLogger("kvpool")

_MADV_POPULATE_WRITE = getattr(mmap, "MADV_POPULATE_WRITE", 23)
_POPULATE_STEP = 1 << 30


@dataclass(slots=True)
class Entry:
    chunk: int
    ready: bool
    writer: int | None
    refs: int = 0
    loads: int = 0
    stored_at: float = 0.0
    last_use: float = 0.0
    pinned: bool = False
    promoting: bool = False
    # Exclusive mode: the conversation ("run") this chunk belongs to; None means
    # a run of its own.
    tag: bytes | None = None


@dataclass(slots=True)
class ClientState:
    client_id: int
    name: str
    pool: "Pool | None" = None
    # Insertion ordered (prefix order of the lookups), released tail first.
    holds: dict[bytes, None] = field(default_factory=dict)
    loads: collections.Counter = field(default_factory=collections.Counter)
    writes: set[bytes] = field(default_factory=set)


class TokenBucket:
    def __init__(self, rate: float, burst: float, clock=time.monotonic) -> None:
        self.rate = rate
        self.burst = burst
        self._clock = clock
        self.tokens = burst
        self._at = clock()

    def take(self, n: float) -> bool:
        now = self._clock()
        self.tokens = min(self.burst, self.tokens + (now - self._at) * self.rate)
        self._at = now
        if self.tokens < n:
            return False
        self.tokens -= n
        return True


class ColdTier:
    """Selective disk tier under one pool's RAM region."""

    def __init__(
        self,
        pool: "Pool",
        cfg: dict,
        notify=None,
        sync: bool = False,
    ) -> None:
        self.pool = pool
        self.dir = os.path.join(cfg["dir"], f"{pool.name}_{pool.layout_key}")
        # "pins": only pinned chunks are written (on demand); evictions never
        # touch the disk. "reused": evicted chunks loaded >= min_loads times
        # are written too, within the write budget.
        self.write_policy = str(cfg.get("write_policy", "pins"))
        if self.write_policy not in ("pins", "reused"):
            raise ValueError(f"cold write_policy {self.write_policy!r}: pins or reused")
        self.min_loads = int(cfg.get("min_loads", 2))
        self.max_bytes = int(cfg["max_bytes"])
        rate = float(cfg.get("write_bytes_per_hour", 20e9)) / 3600.0
        self.bucket = TokenBucket(
            rate, float(cfg.get("write_burst_bytes", rate * 600)), pool._clock
        )
        # On-disk keys, least recently used first, with their sizes.
        self.disk: collections.OrderedDict[bytes, int] = collections.OrderedDict()
        self.disk_bytes = 0
        self.pinned: set[bytes] = set()
        self._pins_path = os.path.join(self.dir, "pins.txt")
        self._writing: set[bytes] = set()
        self._reading: dict[bytes, int] = {}
        self._done: queue.SimpleQueue = queue.SimpleQueue()
        self._jobs: queue.SimpleQueue = queue.SimpleQueue()
        self._notify = notify
        self._sync = sync
        self._threads: list[threading.Thread] = []
        if not sync:
            for i in range(int(cfg.get("threads", 4))):
                t = threading.Thread(target=self._run, name=f"kvpool-cold-{i}", daemon=True)
                t.start()
                self._threads.append(t)

    # --- disk layout ---

    def path(self, key: bytes) -> str:
        h = key.hex()
        return os.path.join(self.dir, h[:3], f"{h}.bin")

    def scan(self) -> None:
        os.makedirs(self.dir, exist_ok=True)
        found = []
        for root, _, files in os.walk(self.dir):
            for name in files:
                if not name.endswith(".bin"):
                    continue
                full = os.path.join(root, name)
                try:
                    st = os.stat(full)
                    key = bytes.fromhex(name[:-4])
                except (OSError, ValueError):
                    continue
                if st.st_size != self.pool.row_stride:
                    os.unlink(full)
                    continue
                found.append((st.st_mtime, key, st.st_size))
        for _, key, size in sorted(found):
            self.disk[key] = size
            self.disk_bytes += size
        if os.path.exists(self._pins_path):
            with open(self._pins_path) as f:
                self.pinned = {bytes.fromhex(line.strip()) for line in f if line.strip()}
        logger.info("pool %s: cold tier %s holds %d chunks (%.2f GiB), %d pinned",
                    self.pool.name, self.dir, len(self.disk),
                    self.disk_bytes / (1 << 30), len(self.pinned))

    def save_pins(self) -> None:
        tmp = self._pins_path + ".tmp"
        with open(tmp, "w") as f:
            for key in sorted(self.pinned):
                f.write(key.hex() + "\n")
        os.replace(tmp, self._pins_path)

    # --- jobs ---

    def _submit(self, job: tuple) -> None:
        if self._sync:
            self._done.put(self._execute(job))
            self.drain()
        else:
            self._jobs.put(job)

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            self._done.put(self._execute(job))
            if self._notify is not None:
                self._notify()

    def _execute(self, job: tuple) -> tuple:
        kind, key = job[0], job[1]
        try:
            if kind == "w":
                data = job[2]
                path = self.path(key)
                os.makedirs(os.path.dirname(path), exist_ok=True)
                tmp = f"{path}.{threading.get_ident()}.tmp"
                with open(tmp, "wb") as f:
                    f.write(data)
                os.replace(tmp, path)
                return ("w", key, len(data), None)
            chunk = job[2]
            row = self.pool.row_stride
            assert self.pool.mm is not None
            view = memoryview(self.pool.mm)[chunk * row : (chunk + 1) * row]
            try:
                with open(self.path(key), "rb", buffering=0) as f:
                    n = f.readinto(view)
            finally:
                view.release()
            if n != row:
                raise OSError(f"short read: {n} of {row} bytes")
            os.utime(self.path(key))
            return ("r", key, chunk, None)
        except Exception as exc:  # reported through the completion queue
            return (kind, key, job[2] if kind == "r" else 0, f"{type(exc).__name__}: {exc}")

    def drain(self) -> None:
        """Apply finished disk jobs. Runs on the event loop thread only."""
        pool = self.pool
        while True:
            try:
                kind, key, value, err = self._done.get_nowait()
            except queue.Empty:
                return
            if kind == "w":
                self._writing.discard(key)
                if err is not None:
                    logger.warning("cold write of %s failed: %s", key.hex(), err)
                    pool.counters["cold_write_failed"] += 1
                    continue
                if key not in self.disk:
                    self.disk[key] = value
                    self.disk_bytes += value
                self.disk.move_to_end(key)
                pool.counters["cold_written"] += 1
                pool.counters["cold_written_bytes"] += value
                self._enforce_cap()
            else:
                chunk = self._reading.pop(key, value)
                entry = pool.index.get(key)
                if err is not None or entry is None or entry.chunk != chunk:
                    logger.warning("cold read of %s failed: %s", key.hex(), err)
                    pool.counters["cold_read_failed"] += 1
                    if entry is not None and entry.chunk == chunk:
                        pool._run_remove(key, entry)
                        del pool.index[key]
                    pool._free.append(chunk)
                    self._forget(key)
                    continue
                entry.ready = True
                entry.promoting = False
                if key in self.pinned and not entry.pinned:
                    entry.pinned = True
                    entry.refs += 1
                if pool.exclusive:
                    pool._run_add(key, entry, entry.tag or key)
                elif entry.refs == 0:
                    pool.lru[key] = None
                if key in self.disk:
                    self.disk.move_to_end(key)
                pool.counters["cold_promoted"] += 1

    def _forget(self, key: bytes) -> None:
        size = self.disk.pop(key, None)
        if size is not None:
            self.disk_bytes -= size
        try:
            os.unlink(self.path(key))
        except FileNotFoundError:
            pass

    def _enforce_cap(self) -> None:
        if self.disk_bytes <= self.max_bytes:
            return
        for key in list(self.disk):
            if self.disk_bytes <= self.max_bytes:
                break
            if key in self.pinned or key in self._reading:
                continue
            self._forget(key)
            self.pool.counters["cold_deleted"] += 1

    # --- policy ---

    def _copy_out(self, entry: Entry) -> bytes:
        row = self.pool.row_stride
        assert self.pool.mm is not None
        return self.pool.mm[entry.chunk * row : (entry.chunk + 1) * row]

    def on_evict(self, key: bytes, entry: Entry) -> None:
        """The chunk is still intact: copy it out now if it is worth keeping."""
        pool = self.pool
        if key in self.disk or key in self._writing:
            if key in self.disk:
                self.disk.move_to_end(key)
            return
        if self.write_policy == "pins":
            return
        if entry.loads < self.min_loads:
            pool.counters["cold_skip_low_value"] += 1
            return
        if not self.bucket.take(pool.row_stride):
            pool.counters["cold_skip_budget"] += 1
            return
        self._writing.add(key)
        self._submit(("w", key, self._copy_out(entry)))

    def persist(self, key: bytes, entry: Entry) -> None:
        """Write a pinned chunk regardless of value or budget."""
        if key in self.disk or key in self._writing:
            return
        self._writing.add(key)
        self._submit(("w", key, self._copy_out(entry)))

    def start_promotion(self, key: bytes) -> bool:
        if key not in self.disk or key in self._reading:
            return False
        chunk = self.pool._alloc_one()
        if chunk is None:
            self.pool.counters["cold_promote_no_space"] += 1
            return False
        now = self.pool._clock()
        self.pool.index[key] = Entry(
            chunk=chunk, ready=False, writer=None, promoting=True,
            stored_at=now, last_use=now,
        )
        self._reading[key] = chunk
        self._submit(("r", key, chunk))
        return True

    def close(self) -> None:
        for _ in self._threads:
            self._jobs.put(None)
        for t in self._threads:
            t.join(timeout=30)
        self.drain()

    def stats(self) -> dict:
        return {
            "dir": self.dir,
            "chunks": len(self.disk),
            "bytes": self.disk_bytes,
            "max_bytes": self.max_bytes,
            "pinned": len(self.pinned),
            "writing": len(self._writing),
            "reading": len(self._reading),
            "budget_bytes": int(self.bucket.tokens),
        }


class Pool:
    """Index and region of one pool. Not thread safe: the server is a single
    event loop, so every method runs to completion before the next request;
    disk threads only touch chunk bytes and report back through a queue."""

    def __init__(
        self,
        name: str,
        layout_key: str,
        row_stride: int,
        num_chunks: int,
        path: str | None,
        quarantine_sec: float = 30.0,
        clock=time.monotonic,
        pending_wait_sec: float = 2.0,
        exclusive: bool = False,
    ) -> None:
        self.name = name
        # Exclusive mode (2026-10-23): the pool extends the GPU prefix caches
        # instead of mirroring them, and orders and evicts whole conversations
        # ("runs", tagged by the engine), never single chunks: on a hybrid model
        # a chain without its head or its end-boundary state restores nothing.
        # A run becomes newest when stored to (the GPU is dropping it) and oldest
        # when loaded (the loading GPU holds it again; the engine writes it back
        # if that GPU evicts it). Lookups that do not load change nothing.
        self.exclusive = exclusive
        self.runs: collections.OrderedDict[bytes, set[bytes]] = collections.OrderedDict()
        # A chunk another client started writing longer ago than this is a
        # MISS to everyone else: they recompute instead of waiting on a slow
        # writer (under deep prefills a writer's step can take seconds).
        self.pending_wait_sec = pending_wait_sec
        self.layout_key = layout_key
        self.row_stride = row_stride
        self.num_chunks = num_chunks
        self.path = path
        self.quarantine_sec = quarantine_sec
        self._clock = clock
        self.index: dict[bytes, Entry] = {}
        # Evictable keys (ready, no refs), least recently used first.
        self.lru: collections.OrderedDict[bytes, None] = collections.OrderedDict()
        self._free: list[int] = []
        self._next_fresh = 0
        self._quarantine: collections.deque[tuple[float, int]] = collections.deque()
        # Keys whose read references were force-dropped (a client gone or reset
        # while loads may still be copying): evictable only after the quarantine.
        self._held_back: collections.deque[tuple[float, bytes]] = collections.deque()
        self.counters = collections.Counter()
        self.fd: int | None = None
        self.mm: mmap.mmap | None = None
        self.cold: ColdTier | None = None

    # --- region ---

    def create_region(self, populate: bool = True) -> None:
        """Create the region, or adopt the file a previous server left at the
        same path. Adopting keeps the inode, so an engine that mapped it under
        the old server still sees the same pages; the index starts empty and
        every chunk waits out the quarantine first, so a transfer still running
        in such an engine cannot land in a chunk handed out again."""
        assert self.path is not None
        size = self.num_chunks * self.row_stride
        adopted = False
        try:
            fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        except FileExistsError:
            fd = os.open(self.path, os.O_RDWR)
            adopted = True
        try:
            if adopted and os.fstat(fd).st_size != size:
                raise RuntimeError(
                    f"{self.path} has {os.fstat(fd).st_size} bytes, the pool needs {size}; "
                    "stop every engine using it and remove it"
                )
            os.ftruncate(fd, size)
            mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
            t0 = time.monotonic()
            if populate:
                for off in range(0, size, _POPULATE_STEP):
                    mm.madvise(_MADV_POPULATE_WRITE, off, min(_POPULATE_STEP, size - off))
            logger.info(
                "pool %s: region %s, %d chunks x %d bytes = %.2f GiB, populated in %.1f s",
                self.name, self.path, self.num_chunks, self.row_stride,
                size / (1 << 30), time.monotonic() - t0,
            )
        except BaseException:
            os.close(fd)
            if not adopted:
                os.unlink(self.path)
            raise
        self.fd, self.mm = fd, mm
        if adopted:
            until = self._clock() + self.quarantine_sec
            self._next_fresh = self.num_chunks
            self._quarantine.extend((until, c) for c in range(self.num_chunks))
            logger.info("pool %s: adopted the existing region; chunks reusable in %.0f s",
                        self.name, self.quarantine_sec)

    def attach_cold(self, cold: ColdTier) -> None:
        """Scan the cold directory and read every pinned chunk back into RAM."""
        self.cold = cold
        cold.scan()
        for key in sorted(cold.pinned):
            if key not in self.index and not cold.start_promotion(key):
                logger.warning("pool %s: pinned chunk %s is not on disk or has no room",
                               self.name, key.hex())

    def close_region(self, unlink: bool = False) -> None:
        """Release the region. The file stays by default: engines may still map
        it, and the next server adopts it."""
        if self.cold is not None:
            self.cold.close()
        if self.mm is not None:
            self.mm.close()
            self.mm = None
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None
        if unlink and self.path is not None:
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass

    # --- chunk pool ---

    def _release_quarantine(self) -> None:
        now = self._clock()
        while self._quarantine and self._quarantine[0][0] <= now:
            self._free.append(self._quarantine.popleft()[1])
        while self._held_back and self._held_back[0][0] <= now:
            key = self._held_back.popleft()[1]
            entry = self.index.get(key)
            if entry is not None:
                self._unref(key, entry)

    def _num_free(self) -> int:
        return len(self._free) + self.num_chunks - self._next_fresh

    def _alloc(self) -> int:
        if self._free:
            return self._free.pop()
        assert self._next_fresh < self.num_chunks
        self._next_fresh += 1
        return self._next_fresh - 1

    def _num_evictable(self) -> int:
        if not self.exclusive:
            return len(self.lru)
        return sum(1 for e in self.index.values() if e.ready and e.refs == 0)

    # --- runs (exclusive mode) ---

    def _run_add(self, key: bytes, entry: Entry, tag: bytes) -> None:
        """Put key in run tag (moving it from another run) and make tag newest."""
        if entry.tag is not None and entry.tag != tag:
            self._run_remove(key, entry)
        entry.tag = tag
        self.runs.setdefault(tag, set()).add(key)
        self.runs.move_to_end(tag)

    def _run_remove(self, key: bytes, entry: Entry) -> None:
        if entry.tag is None:
            return
        run = self.runs.get(entry.tag)
        if run is not None:
            run.discard(key)
            if not run:
                del self.runs[entry.tag]

    def _evict_run(self, protected: set[bytes]) -> list[bytes]:
        """Evict every evictable chunk of the oldest run not in protected."""
        for tag, run in self.runs.items():
            if tag in protected:
                continue
            keys = [k for k in run if self.index[k].ready and self.index[k].refs == 0]
            if keys:
                for key in keys:
                    self._evict_key(key)
                self.counters["runs_evicted"] += 1
                return keys
        return []

    def _evict_key(self, key: bytes) -> None:
        entry = self.index.pop(key)
        self._run_remove(key, entry)
        if self.cold is not None:
            self.cold.on_evict(key, entry)
        self._free.append(entry.chunk)
        self.counters["chunks_evicted"] += 1

    def _evict_one(self) -> bytes:
        key, _ = self.lru.popitem(last=False)
        entry = self.index.pop(key)
        if self.cold is not None:
            self.cold.on_evict(key, entry)
        self._free.append(entry.chunk)
        self.counters["chunks_evicted"] += 1
        return key

    def _alloc_one(self) -> int | None:
        """A chunk for a promotion, evicting one if needed; None if full."""
        self._release_quarantine()
        if self._num_free() == 0:
            if self.exclusive:
                if not self._evict_run(set()):
                    return None
            else:
                if not self.lru:
                    return None
                self._evict_one()
        return self._alloc()

    def _quarantine_chunk(self, chunk: int) -> None:
        self._quarantine.append((self._clock() + self.quarantine_sec, chunk))

    def _ref(self, key: bytes, entry: Entry) -> None:
        if entry.refs == 0 and entry.ready and not self.exclusive:
            self.lru.pop(key, None)
        entry.refs += 1

    def _unref(self, key: bytes, entry: Entry) -> None:
        assert entry.refs > 0, f"refcount underflow on {key.hex()}"
        entry.refs -= 1
        if entry.refs == 0 and entry.ready and not self.exclusive:
            self.lru[key] = None

    # --- operations ---

    def lookup(
        self, client: ClientState, keys: list[bytes], stop_at_miss: bool = False
    ) -> list[int]:
        out = []
        now = self._clock()
        for key in keys:
            entry = self.index.get(key)
            if entry is None:
                if self.cold is not None and self.cold.start_promotion(key):
                    out.append(P.HIT_PENDING)
                else:
                    out.append(P.MISS)
                    self.counters["lookup_miss"] += 1
                    if stop_at_miss:
                        break
            elif not entry.ready:
                if (
                    entry.writer is not None
                    and entry.writer != client.client_id
                    and now - entry.stored_at > self.pending_wait_sec
                ):
                    out.append(P.MISS)
                    self.counters["lookup_pending_as_miss"] += 1
                    if stop_at_miss:
                        break
                else:
                    out.append(P.HIT_PENDING)
            else:
                if key not in client.holds:
                    client.holds[key] = None
                    self._ref(key, entry)
                entry.last_use = now
                out.append(P.HIT)
                self.counters["lookup_hit"] += 1
        return out

    def end_session(self, client: ClientState) -> None:
        for key in reversed(client.holds):
            entry = self.index.get(key)
            if entry is not None:
                self._unref(key, entry)
        client.holds.clear()

    def prepare_load(self, client: ClientState, keys: list[bytes]) -> list[int]:
        entries = []
        for key in keys:
            entry = self.index.get(key)
            if entry is None or not entry.ready:
                raise KeyError(
                    f"prepare_load of {key.hex()} which is not ready; a lookup "
                    "HIT must be held by the session that loads it"
                )
            entries.append(entry)
        now = self._clock()
        for key, entry in zip(keys, entries):
            self._ref(key, entry)
            client.loads[key] += 1
            entry.loads += 1
            entry.last_use = now
        self.counters["chunks_loaded"] += len(keys)
        return [e.chunk for e in entries]

    def complete_load(self, client: ClientState, keys: list[bytes]) -> None:
        for key in reversed(keys):
            if client.loads[key] <= 0:
                raise KeyError(f"complete_load of {key.hex()} without prepare_load")
            client.loads[key] -= 1
            if client.loads[key] == 0:
                del client.loads[key]
            entry = self.index.get(key)
            if entry is not None:
                self._unref(key, entry)
                if self.exclusive and entry.tag in self.runs:
                    self.runs.move_to_end(entry.tag, last=False)
                    self.counters["demoted_on_load"] += 1

    def prepare_store(
        self, client: ClientState, keys: list[bytes], tag: bytes | None = None
    ) -> tuple[list[bytes], list[int], list[bytes]] | None:
        """Allocate chunks for the keys the pool does not hold yet, evicting
        least recently used chunks if needed. All or nothing: returns None and
        changes nothing when there is not enough evictable space."""
        if self.exclusive:
            return self._prepare_store_runs(client, keys, tag)
        self._release_quarantine()
        new_keys = []
        seen = set()
        for key in keys:
            if key not in self.index and key not in seen:
                new_keys.append(key)
                seen.add(key)
        self.counters["store_dedup"] += len(keys) - len(new_keys)
        if not new_keys:
            return [], [], []
        # Offered keys the pool already holds belong to the same prefix: they
        # must survive this call (evicting one would leave a hole the scheduler
        # treats as stored). Refresh them to most recent, deepest first so the
        # head ends newest, which also keeps them out of the eviction below.
        protected = [k for k in dict.fromkeys(keys) if k in self.lru]
        for key in reversed(protected):
            self.lru.move_to_end(key)
        need = len(new_keys) - self._num_free()
        evicted: list[bytes] = []
        if need > 0:
            if need > len(self.lru) - len(protected):
                self.counters["store_rejected"] += 1
                return None
            evicted = [self._evict_one() for _ in range(need)]
        now = self._clock()
        chunks = []
        for key in new_keys:
            chunk = self._alloc()
            self.index[key] = Entry(
                chunk=chunk, ready=False, writer=client.client_id,
                stored_at=now, last_use=now,
            )
            client.writes.add(key)
            chunks.append(chunk)
        self.counters["chunks_stored"] += len(new_keys)
        return new_keys, chunks, evicted

    def _prepare_store_runs(
        self, client: ClientState, keys: list[bytes], tag: bytes | None
    ) -> tuple[list[bytes], list[int], list[bytes]] | None:
        """Exclusive-mode prepare_store: every offered key joins run tag (or a
        run of its own), the run becomes newest, and space comes from evicting
        whole older runs. All or nothing."""
        self._release_quarantine()
        offered = list(dict.fromkeys(keys))
        new_keys = [k for k in offered if k not in self.index]
        self.counters["store_dedup"] += len(keys) - len(new_keys)
        protected = {tag} if tag is not None else set()
        protected.update(self.index[k].tag or k for k in offered if k in self.index)
        need = len(new_keys) - self._num_free()
        if need > 0:
            avail = sum(
                1 for t, run in self.runs.items() if t not in protected
                for k in run if self.index[k].ready and self.index[k].refs == 0
            )
            if need > avail:
                self.counters["store_rejected"] += 1
                return None
        for key in offered:
            entry = self.index.get(key)
            if entry is not None:
                self._run_add(key, entry, tag if tag is not None else (entry.tag or key))
        evicted: list[bytes] = []
        while len(new_keys) > self._num_free():
            got = self._evict_run(protected)
            assert got, "evictable count and runs disagree"
            evicted += got
        now = self._clock()
        chunks = []
        for key in new_keys:
            entry = Entry(chunk=self._alloc(), ready=False, writer=client.client_id,
                          stored_at=now, last_use=now)
            self.index[key] = entry
            self._run_add(key, entry, tag if tag is not None else key)
            client.writes.add(key)
            chunks.append(entry.chunk)
        self.counters["chunks_stored"] += len(new_keys)
        return new_keys, chunks, evicted

    def complete_store(self, client: ClientState, keys: list[bytes], success: bool) -> None:
        for key in reversed(keys):
            if key not in client.writes:
                continue
            client.writes.discard(key)
            entry = self.index.get(key)
            if entry is None or entry.ready or entry.writer != client.client_id:
                continue
            if success:
                entry.ready = True
                entry.writer = None
                if self.cold is not None and key in self.cold.pinned:
                    self._pin_entry(key, entry)
                elif entry.refs == 0 and not self.exclusive:
                    self.lru[key] = None
            else:
                self._run_remove(key, entry)
                del self.index[key]
                self._quarantine_chunk(entry.chunk)
                self.counters["store_failed"] += 1

    def touch(self, keys: list[bytes]) -> None:
        now = self._clock()
        for key in reversed(keys):
            entry = self.index.get(key)
            if entry is None:
                continue
            entry.last_use = now
            if key in self.lru:
                self.lru.move_to_end(key)
            if self.exclusive and entry.tag in self.runs:
                self.runs.move_to_end(entry.tag)

    def release_client(self, client: ClientState) -> None:
        """Drop everything a client holds: on disconnect or a cache reset."""
        self.end_session(client)
        until = self._clock() + self.quarantine_sec
        for key, n in client.loads.items():
            entry = self.index.get(key)
            if entry is not None:
                # Keep one reference until the quarantine ends: a worker of the
                # departed client may still be copying this chunk.
                for _ in range(n - 1):
                    self._unref(key, entry)
                self._held_back.append((until, key))
                self.counters["loads_held_back"] += 1
        client.loads.clear()
        for key in client.writes:
            entry = self.index.get(key)
            if entry is not None and not entry.ready and entry.writer == client.client_id:
                self._run_remove(key, entry)
                del self.index[key]
                self._quarantine_chunk(entry.chunk)
                self.counters["store_aborted"] += 1
        client.writes.clear()

    # --- pins ---

    def _pin_entry(self, key: bytes, entry: Entry) -> None:
        if entry.pinned:
            return
        entry.pinned = True
        self._ref(key, entry)
        if self.cold is not None:
            self.cold.persist(key, entry)

    def pin(self, keys: list[bytes]) -> int:
        """Pin chunks: never evicted, kept on disk, restored after a restart.
        A key not resident is pinned when it is next stored or read back."""
        pinned = 0
        for key in keys:
            entry = self.index.get(key)
            if self.cold is not None:
                self.cold.pinned.add(key)
            if entry is not None and entry.ready:
                self._pin_entry(key, entry)
                pinned += 1
            elif entry is None and self.cold is not None:
                self.cold.start_promotion(key)
        if self.cold is not None:
            self.cold.save_pins()
        return pinned

    def pin_recent(self, seconds: float, until_ago: float = 0.0) -> int:
        """Pin every chunk used between ``seconds`` and ``until_ago`` seconds
        ago, for an operator that has just primed a prefix. Chunks still being
        written are pinned when their store completes."""
        now = self._clock()
        since, until = now - seconds, now - until_ago
        keys = [k for k, e in self.index.items() if since <= e.last_use <= until]
        if self.cold is None:
            # Without a cold tier there is no pin list to hold pending keys.
            keys = [k for k in keys if self.index[k].ready]
        return self.pin(keys)

    def unpin(self, keys: list[bytes] | None) -> int:
        """Unpin the given keys, or every pin when keys is None."""
        if keys is None:
            keys = [k for k, e in self.index.items() if e.pinned]
            if self.cold is not None:
                keys += [k for k in self.cold.pinned if k not in self.index]
        n = 0
        for key in keys:
            entry = self.index.get(key)
            if entry is not None and entry.pinned:
                entry.pinned = False
                self._unref(key, entry)
                n += 1
            if self.cold is not None:
                self.cold.pinned.discard(key)
        if self.cold is not None:
            self.cold.save_pins()
        return n

    def stats(self) -> dict:
        ready = sum(1 for e in self.index.values() if e.ready)
        out = {
            "pool": self.name,
            "num_chunks": self.num_chunks,
            "row_stride": self.row_stride,
            "used": len(self.index),
            "ready": ready,
            "write_pending": len(self.index) - ready,
            "evictable": self._num_evictable(),
            "held": ready - self._num_evictable(),
            "pinned": sum(1 for e in self.index.values() if e.pinned),
            "free": self._num_free(),
            "quarantined": len(self._quarantine),
            "counters": dict(self.counters),
        }
        if self.cold is not None:
            out["cold"] = self.cold.stats()
        return out


class PoolServer:
    def __init__(self, socket_path: str, pools_cfg: dict, shm_dir: str = "/dev/shm",
                 quarantine_sec: float = 30.0, populate: bool = True,
                 socket_group: str | None = None) -> None:
        self.socket_path = socket_path
        # Run as root (no tmpfs user quota on the region), let this group's
        # members (the router) reach the socket.
        self.socket_group = socket_group
        self.pools_cfg = pools_cfg
        self.shm_dir = shm_dir
        self.quarantine_sec = quarantine_sec
        self.populate = populate
        self.generation = f"{os.getpid():x}{int(time.time()):x}"
        self.pools: dict[str, Pool] = {}
        self._next_client = 1
        self._sel = selectors.DefaultSelector()
        self._conns: dict[socket.socket, tuple[bytearray, ClientState]] = {}
        self._running = False
        self._wake_r, self._wake_w = socket.socketpair()
        self._wake_r.setblocking(False)
        self._wake_w.setblocking(False)

    def _notify(self) -> None:
        try:
            self._wake_w.send(b"\0")
        except (BlockingIOError, OSError):
            pass

    def _on_wake(self, sock: socket.socket) -> None:
        try:
            while sock.recv(4096):
                pass
        except BlockingIOError:
            pass
        self._drain_cold()

    def _drain_cold(self) -> None:
        for pool in self.pools.values():
            if pool.cold is not None:
                pool.cold.drain()

    # --- request handling ---

    def _get_pool(self, args: dict, create: bool) -> Pool:
        name = args["pool"]
        cfg = self.pools_cfg.get(name)
        if cfg is None:
            raise ValueError(f"unknown pool {name!r}; configured: {sorted(self.pools_cfg)}")
        pool = self.pools.get(name)
        if pool is None:
            if not create:
                raise ValueError(f"pool {name!r} has no region yet")
            row_stride = int(args["row_stride"])
            num_chunks = int(cfg["bytes"]) // row_stride
            if num_chunks <= 0:
                raise ValueError(f"pool {name!r}: {cfg['bytes']} bytes hold no chunk")
            path = self.region_path(name)
            pool = Pool(name, args["layout_key"], row_stride, num_chunks, path,
                        quarantine_sec=self.quarantine_sec,
                        pending_wait_sec=float(cfg.get("pending_wait_sec", 2.0)),
                        exclusive=bool(cfg.get("exclusive", False)))
            pool.create_region(populate=self.populate)
            if cfg.get("cold"):
                pool.attach_cold(ColdTier(pool, cfg["cold"], notify=self._notify,
                                          sync=bool(cfg["cold"].get("sync", False))))
            self.pools[name] = pool
        if pool.layout_key != args["layout_key"] or pool.row_stride != int(args["row_stride"]):
            raise ValueError(
                f"pool {name!r} holds layout {pool.layout_key} with rows of "
                f"{pool.row_stride} bytes; this client has {args['layout_key']} "
                f"with rows of {args['row_stride']} bytes"
            )
        return pool

    def _pool_reply(self, pool: Pool) -> dict:
        return {
            "path": pool.path,
            "num_chunks": pool.num_chunks,
            "row_stride": pool.row_stride,
            "generation": self.generation,
        }

    def _named_pool(self, args: dict) -> Pool:
        pool = self.pools.get(args["pool"])
        if pool is None:
            raise ValueError(f"pool {args['pool']!r} has no region yet")
        return pool

    def handle(self, client: ClientState, op: str, args) -> object:
        if op == P.OP_ATTACH:
            if args.get("version") != P.PROTOCOL_VERSION:
                raise ValueError(f"protocol {args.get('version')} != {P.PROTOCOL_VERSION}")
            if client.pool is not None:
                raise ValueError("already attached")
            client.pool = self._get_pool(args, create=True)
            client.name = str(args.get("name", ""))
            logger.info("client %d (%s) attached to pool %s",
                        client.client_id, client.name, client.pool.name)
            return {**self._pool_reply(client.pool), "client_id": client.client_id}
        if op == P.OP_INFO:
            return self._pool_reply(self._get_pool(args, create=True))
        if op == P.OP_STATS:
            if client.pool is None:
                return {name: p.stats() for name, p in self.pools.items()}
            return client.pool.stats()
        # Pin operations come from operators (the router), not engines, so
        # they name the pool instead of attaching to it.
        if op == P.OP_PIN:
            return self._named_pool(args).pin(args["keys"])
        if op == P.OP_PIN_RECENT:
            return self._named_pool(args).pin_recent(
                float(args["seconds"]), float(args.get("until_ago", 0.0)))
        if op == P.OP_UNPIN:
            return self._named_pool(args).unpin(args.get("keys"))
        pool = client.pool
        if pool is None:
            raise ValueError("not attached")
        if op == P.OP_LOOKUP:
            return pool.lookup(client, args)
        if op == P.OP_LOOKUP_PREFIX:
            return pool.lookup(client, args, stop_at_miss=True)
        if op == P.OP_END_SESSION:
            pool.end_session(client)
            return None
        if op == P.OP_PREPARE_LOAD:
            return pool.prepare_load(client, args)
        if op == P.OP_COMPLETE_LOAD:
            pool.complete_load(client, args)
            return None
        if op == P.OP_PREPARE_STORE:
            res = pool.prepare_store(client, args)
            return None if res is None else list(res)
        if op == P.OP_PREPARE_STORE_TAGGED:
            keys, tag = args
            res = pool.prepare_store(client, keys, tag)
            return None if res is None else list(res)
        if op == P.OP_COMPLETE_STORE:
            keys, success = args
            pool.complete_store(client, keys, success)
            return None
        if op == P.OP_TOUCH:
            pool.touch(args)
            return None
        if op == P.OP_RELEASE_ALL:
            pool.release_client(client)
            return None
        raise ValueError(f"unknown op {op!r}")

    # --- event loop ---

    def _accept(self, lsock: socket.socket) -> None:
        conn, _ = lsock.accept()
        conn.setblocking(False)
        client = ClientState(client_id=self._next_client, name="")
        self._next_client += 1
        self._conns[conn] = (bytearray(), client)
        self._sel.register(conn, selectors.EVENT_READ, self._serve)

    def _drop(self, conn: socket.socket) -> None:
        self._sel.unregister(conn)
        _, client = self._conns.pop(conn)
        conn.close()
        if client.pool is not None:
            held = len(client.holds) + sum(client.loads.values()) + len(client.writes)
            client.pool.release_client(client)
            logger.info("client %d (%s) detached, released %d references",
                        client.client_id, client.name, held)

    def _serve(self, conn: socket.socket) -> None:
        buf, client = self._conns[conn]
        try:
            data = conn.recv(1 << 20)
        except ConnectionError:
            data = b""
        if not data:
            self._drop(conn)
            return
        buf += data
        try:
            requests = P.decode_frames(buf)
        except (P.ProtocolError, ValueError, EOFError) as exc:
            logger.warning("client %d sent a bad frame (%s); dropping it", client.client_id, exc)
            self._drop(conn)
            return
        # Finished disk reads become visible before this batch's lookups.
        self._drain_cold()
        out = bytearray()
        for req in requests:
            try:
                op, args = req
                reply = (True, self.handle(client, op, args))
            except Exception as exc:  # reported to the caller, not fatal here
                logger.warning("client %d %r failed: %s", client.client_id,
                               req[0] if isinstance(req, tuple) else req, exc)
                reply = (False, f"{type(exc).__name__}: {exc}")
            out += P.encode(reply)
        try:
            conn.setblocking(True)
            conn.sendall(out)
            conn.setblocking(False)
        except OSError:
            self._drop(conn)

    def region_path(self, pool: str) -> str:
        return os.path.join(self.shm_dir, f"vllm_kvpool_{pool}.mmap")

    def remove_stale_regions(self) -> int:
        """Delete region files no configured pool will adopt (a renamed pool,
        or the per-generation names of the first deployment). An engine still
        mapping one keeps its pages until it exits."""
        keep = {os.path.basename(self.region_path(p)) for p in self.pools_cfg}
        removed = 0
        for name in os.listdir(self.shm_dir):
            if name.startswith("vllm_kvpool_") and name.endswith(".mmap") and name not in keep:
                path = os.path.join(self.shm_dir, name)
                size = os.path.getsize(path)
                os.unlink(path)
                removed += 1
                logger.info("removed stale region %s (%.2f GiB)", path, size / (1 << 30))
        return removed

    def serve_forever(self, log_interval: float = 60.0) -> None:
        self.remove_stale_regions()
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass
        os.makedirs(os.path.dirname(self.socket_path), exist_ok=True)
        # Bind under a temporary name and rename once listening, so a client
        # never finds a socket path that refuses connections.
        tmp_path = f"{self.socket_path}.{os.getpid()}.tmp"
        lsock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        lsock.bind(tmp_path)
        if self.socket_group:
            import grp
            os.chown(tmp_path, -1, grp.getgrnam(self.socket_group).gr_gid)
            os.chmod(tmp_path, 0o660)
        else:
            os.chmod(tmp_path, 0o600)
        lsock.listen(64)
        os.rename(tmp_path, self.socket_path)
        lsock.setblocking(False)
        self._sel.register(lsock, selectors.EVENT_READ, self._accept)
        self._sel.register(self._wake_r, selectors.EVENT_READ, self._on_wake)
        self._running = True
        logger.info("kvpool generation %s listening on %s, pools %s",
                    self.generation, self.socket_path,
                    {k: v["bytes"] for k, v in self.pools_cfg.items()})
        next_log = time.monotonic() + log_interval
        try:
            while self._running:
                for key, _ in self._sel.select(timeout=1.0):
                    key.data(key.fileobj)
                if time.monotonic() >= next_log:
                    next_log = time.monotonic() + log_interval
                    for pool in self.pools.values():
                        logger.info("stats %s", json.dumps(pool.stats(), sort_keys=True))
        finally:
            for conn in list(self._conns):
                self._drop(conn)
            self._sel.close()
            lsock.close()
            try:
                os.unlink(self.socket_path)
            except FileNotFoundError:
                pass
            for pool in self.pools.values():
                pool.close_region()
            self._wake_r.close()
            self._wake_w.close()

    def stop(self, *_args) -> None:
        self._running = False


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True,
                    help='JSON: {"socket": path, "pools": {name: {"bytes": n, "cold": {...}}}}')
    args = ap.parse_args(argv)
    with open(args.config) as f:
        cfg = json.load(f)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    server = PoolServer(cfg["socket"], cfg["pools"], cfg.get("shm_dir", "/dev/shm"),
                        float(cfg.get("quarantine_sec", 30.0)),
                        socket_group=cfg.get("socket_group"))
    signal.signal(signal.SIGTERM, server.stop)
    signal.signal(signal.SIGINT, server.stop)
    server.serve_forever(float(cfg.get("log_interval_sec", 60.0)))


if __name__ == "__main__":
    main()
