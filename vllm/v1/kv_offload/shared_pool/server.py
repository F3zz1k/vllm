# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared host-RAM KV arena server.

One process per host owns one ``/dev/shm`` file, the arena, and the index over
it. Every vLLM engine using SharedPoolOffloadingSpec maps the whole arena at
its own row stride, so a row stored by one engine can be loaded by any other
engine of the same layout with one host-to-device copy, and models of
different row sizes share one byte budget.

Layout. The arena is cut into superpages. A superpage belongs to at most one
layout class (a model's KV layout, named in the config) at a time, and a row
of stride s is valid for its class only when [r*s, (r+1)*s) lies inside a
superpage that class owns, so the server never hands out a row that overlaps
another class's bytes. Pinned rows live in a pin zone of superpages capped by
``pin_max_bytes``. A class with several rows per superpage is compacted when
another class needs whole superpages: ready, unreferenced rows are copied into
fuller superpages of their own class, and a lookup of a row being moved
answers HIT_PENDING.

Runs. Every store names a run (a request id for compute-time stores, a
conversation for write-back; MERGE_TAG moves a request's keys into its
conversation). A chunk may belong to several runs, so a shared prefix survives
while any run that contains it does. One list orders the runs of every class,
oldest first. Eviction takes whole runs only, never one with a chunk held,
loading, moving or being written, and takes first the runs that can restore
nothing (on a hybrid model, a run with no boundary whose recurrent state keys
are all present). A store is all or nothing: the walk is simulated across all
classes first and committed only when it frees enough. A store makes its run
newest; a completed load makes the loading run oldest (its engine's GPU holds
it again) and a lookup changes nothing. When an engine goes away, the runs it
loaded and nobody stored again become newest: its GPU copy is gone.

Holds. A lookup HIT pins the chunk for the calling client until it ends its
session, so another engine cannot evict it between the lookup and the
prepare_load that pins it for the transfer.

Crash safety. When a client disconnects, its holds and read references are
dropped and the chunks it was still writing are removed. Their rows wait in
quarantine before reuse, so a transfer still running in a dying engine cannot
overwrite data a new owner has written.

Lifecycle. ABSENT (no arena) -> CREATING (a thread creates and populates the
file, then the spilled runs are read back) -> READY -> SPILLING (the newest
whole runs are written to NVMe, then the file is unlinked) -> ABSENT. PREPARE,
SPAWN_OPEN, ATTACH and INFO start a creation; ATTACH and INFO replies wait
until READY. RELEASE starts a spill only when nothing can still use the arena.
A clean stop with no client attached spills too.

Cold tier (optional, per class): pinned chunks are written to disk when
pinned and read back into the pin zone after a restart.

Standard library only: run it with the host Python, for example
``python3 server.py --config kvpool.json``.
"""

import argparse
import collections
import itertools
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
_IO_ALIGN = 4096
# A run that cannot restore anything is evicted before every other run once
# its last store is this old; a younger one is still being written.
DEAD_RUN_GRACE_SEC = 10.0
# Tag prefix of the run an unpinned chunk in no conversation gets.
_ORPHAN = b"orphan:"
MANIFEST = "manifest.json"
_MANIFEST_VERSION = 1


def _tag(tag: bytes | str) -> bytes:
    """A run tag as the index keeps it. Engines send a conversation's tag as
    its key (bytes) and a request's own run as "<engine>/<request>" (str)."""
    return tag.encode() if isinstance(tag, str) else bytes(tag)


def _group(key: bytes) -> int:
    """KV cache group of an offload key (block hash + 4-byte group index)."""
    return int.from_bytes(key[-4:], "big")


def mem_available() -> int:
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    raise RuntimeError("no MemAvailable in /proc/meminfo")


def other_mappers(path: str) -> list[int]:
    """Pids other than this one that map the file at ``path`` (same device
    and inode). Reads /proc/<pid>/maps, which names the device and inode of
    every mapping: one read per process instead of a stat per map_files
    entry. Needs root to see other users' processes."""
    try:
        st = os.stat(path)
    except FileNotFoundError:
        return []
    dev = f"{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}"
    ino = str(st.st_ino)
    me = os.getpid()
    out = []
    for name in os.listdir("/proc"):
        if not name.isdigit() or int(name) == me:
            continue
        try:
            with open(f"/proc/{name}/maps") as f:
                for line in f:
                    parts = line.split(None, 5)
                    if len(parts) >= 5 and parts[4] == ino and parts[3] == dev:
                        out.append(int(name))
                        break
        except OSError:
            continue
    return out


def _open_direct(path: str, flags: int, aligned: bool) -> int:
    """O_DIRECT when the transfer is aligned and the filesystem takes it."""
    if aligned:
        try:
            return os.open(path, flags | os.O_DIRECT, 0o600)
        except OSError:
            pass
    return os.open(path, flags, 0o600)


def _write_row_file(path: str, mm: mmap.mmap, off: int, length: int) -> None:
    aligned = off % _IO_ALIGN == 0 and length % _IO_ALIGN == 0
    fd = _open_direct(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, aligned)
    view = memoryview(mm)[off : off + length]
    try:
        pos = 0
        while pos < length:
            pos += os.pwrite(fd, view[pos:], pos)
        os.fdatasync(fd)
    finally:
        view.release()
        os.close(fd)


def _read_row_file(path: str, mm: mmap.mmap, off: int, length: int) -> None:
    aligned = off % _IO_ALIGN == 0 and length % _IO_ALIGN == 0
    fd = _open_direct(path, os.O_RDONLY, aligned)
    view = memoryview(mm)[off : off + length]
    try:
        size = os.fstat(fd).st_size
        if size != length:
            raise OSError(f"{path}: {size} bytes, expected {length}")
        pos = 0
        while pos < length:
            n = os.preadv(fd, [view[pos:]], pos)
            if n <= 0:
                raise OSError(f"{path}: short read at {pos}")
            pos += n
    finally:
        view.release()
        os.close(fd)


def _fsync_dir(path: str) -> None:
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@dataclass(slots=True, eq=False)
class Entry:
    row: int
    ready: bool
    writer: int | None
    refs: int = 0
    stored_at: float = 0.0
    last_use: float = 0.0
    pinned: bool = False
    promoting: bool = False  # being read from disk
    moving: int | None = None  # destination row of a copy in flight
    runs: set = field(default_factory=set)


class Run:
    """One run's chunks, evicted as a unit."""

    __slots__ = ("cls", "tag", "keys", "_groups", "complete", "stored_at", "demoted_by")

    def __init__(self, cls: "LayoutClass", tag: bytes, now: float) -> None:
        self.cls = cls
        self.tag = tag
        self.keys: dict[bytes, None] = {}
        # Block hash -> groups present, for classes with recurrent state.
        self._groups: dict[bytes, set[int]] = {}
        self.complete = 0  # boundaries with every state group present
        self.stored_at = now
        self.demoted_by: set = set()  # clients whose completed load demoted it

    @property
    def restorable(self) -> bool:
        return not self.cls.state_groups or self.complete > 0

    def add(self, key: bytes) -> None:
        if key in self.keys:
            return
        self.keys[key] = None
        if self.cls.state_groups:
            groups = self._groups.setdefault(key[:-4], set())
            was = self.cls.boundary_complete(groups)
            groups.add(_group(key))
            if not was and self.cls.boundary_complete(groups):
                self.complete += 1

    def discard(self, key: bytes) -> None:
        if key not in self.keys:
            return
        del self.keys[key]
        if self.cls.state_groups:
            h = key[:-4]
            groups = self._groups[h]
            was = self.cls.boundary_complete(groups)
            groups.discard(_group(key))
            if was and not self.cls.boundary_complete(groups):
                self.complete -= 1
            if not groups:
                del self._groups[h]

    def restorable_with(self, keys: list[bytes]) -> bool:
        if self.restorable:
            return True
        extra: dict[bytes, set[int]] = {}
        for key in keys:
            extra.setdefault(key[:-4], set()).add(_group(key))
        return any(
            self.cls.boundary_complete(groups | self._groups.get(h, set()))
            for h, groups in extra.items()
        )


@dataclass(slots=True, eq=False)
class ClientState:
    client_id: int
    name: str
    cls: "LayoutClass | None" = None
    # Insertion ordered (prefix order of the lookups), released tail first.
    holds: dict[bytes, None] = field(default_factory=dict)
    loads: collections.Counter = field(default_factory=collections.Counter)
    writes: set[bytes] = field(default_factory=set)
    # Runs this client's loads demoted and nobody has stored to since.
    demoted: dict[Run, None] = field(default_factory=dict)


@dataclass(slots=True, eq=False)
class Conn:
    client: ClientState
    buf: bytearray = field(default_factory=bytearray)
    # Requests waiting for the arena (an ATTACH or INFO and all after it).
    backlog: list = field(default_factory=list)


class Worker:
    """Background I/O for the event loop: jobs run on threads, or, with
    threads=0, only when run_pending() is called (tests); results are applied
    by drain(), on the event loop thread."""

    def __init__(self, name: str, threads: int, execute, apply, notify=None) -> None:
        self._execute = execute
        self._apply = apply
        self._notify = notify
        self._jobs: queue.SimpleQueue = queue.SimpleQueue()
        self._done: queue.SimpleQueue = queue.SimpleQueue()
        self._manual: list = []
        self._threads = [
            threading.Thread(target=self._run, name=f"kvpool-{name}-{i}", daemon=True)
            for i in range(threads)
        ]
        for t in self._threads:
            t.start()

    def submit(self, job: tuple) -> None:
        if self._threads:
            self._jobs.put(job)
        else:
            self._manual.append(job)

    def run_pending(self) -> None:
        jobs, self._manual = self._manual, []
        for job in jobs:
            self._done.put((job, self._safe(job)))
        self.drain()

    def _safe(self, job: tuple) -> str | None:
        try:
            self._execute(job)
            return None
        except Exception as exc:  # reported through drain()
            return f"{type(exc).__name__}: {exc}"

    def _run(self) -> None:
        while True:
            job = self._jobs.get()
            if job is None:
                return
            self._done.put((job, self._safe(job)))
            if self._notify is not None:
                self._notify()

    def drain(self) -> None:
        while True:
            try:
                job, err = self._done.get_nowait()
            except queue.Empty:
                return
            self._apply(job, err)

    def close(self) -> None:
        for _ in self._threads:
            self._jobs.put(None)
        for t in self._threads:
            t.join(timeout=30)
        self.drain()


class ColdTier:
    """Disk copies of one class's pinned chunks: written when pinned, read back
    into the pin zone after a restart or on a lookup miss."""

    def __init__(self, cls: "LayoutClass", cfg: dict, notify=None,
                 threads: int | None = None) -> None:
        if str(cfg.get("write_policy", "pins")) != "pins":
            raise ValueError("cold write_policy: only 'pins' is supported")
        self.cls = cls
        self.dir = os.path.join(cfg["dir"], f"{cls.name}_{cls.layout_key}")
        self.max_bytes = int(cfg["max_bytes"])
        # On-disk keys, least recently used first, with their sizes.
        self.disk: collections.OrderedDict[bytes, int] = collections.OrderedDict()
        self.disk_bytes = 0
        self._pins_path = os.path.join(self.dir, "pins.txt")
        self._writing: dict[bytes, Entry] = {}
        self._reading: dict[bytes, Entry] = {}
        n = 0 if cfg.get("sync") else (int(cfg.get("threads", 4)) if threads is None else threads)
        self._sync = n == 0
        self.worker = Worker(f"cold-{cls.name}", n, self._execute, self._apply, notify)

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
                if st.st_size != self.cls.stride:
                    os.unlink(full)
                    continue
                found.append((st.st_mtime, key, st.st_size))
        for _, key, size in sorted(found):
            self.disk[key] = size
            self.disk_bytes += size
        if os.path.exists(self._pins_path):
            with open(self._pins_path) as f:
                self.cls.pins |= {bytes.fromhex(line.strip()) for line in f if line.strip()}
        logger.info("class %s: cold tier %s holds %d chunks (%.2f GiB), %d pinned",
                    self.cls.name, self.dir, len(self.disk),
                    self.disk_bytes / (1 << 30), len(self.cls.pins))

    def save_pins(self) -> None:
        tmp = self._pins_path + ".tmp"
        with open(tmp, "w") as f:
            for key in sorted(self.cls.pins):
                f.write(key.hex() + "\n")
        os.replace(tmp, self._pins_path)

    def _execute(self, job: tuple) -> None:
        kind, key, row = job
        mm = self.cls.arena.mm
        assert mm is not None
        off, length = self.cls.offset(row), self.cls.stride
        path = self.path(key)
        if kind == "w":
            os.makedirs(os.path.dirname(path), exist_ok=True)
            tmp = f"{path}.{threading.get_ident()}.tmp"
            _write_row_file(tmp, mm, off, length)
            os.replace(tmp, path)
        else:
            _read_row_file(path, mm, off, length)
            os.utime(path)

    def _apply(self, job: tuple, err: str | None) -> None:
        kind, key, row = job
        cls = self.cls
        if kind == "w":
            entry = self._writing.pop(key)
            if cls.index.get(key) is entry:
                cls.unref(key, entry)
            if err is not None:
                logger.warning("cold write of %s failed: %s", key.hex(), err)
                cls.counters["cold_write_failed"] += 1
                return
            if key not in self.disk:
                self.disk[key] = cls.stride
                self.disk_bytes += cls.stride
            self.disk.move_to_end(key)
            cls.counters["cold_written"] += 1
            self._enforce_cap()
            return
        entry = self._reading.pop(key)
        if cls.index.get(key) is not entry:
            return  # the class was retired meanwhile; its rows are quarantined
        if err is not None:
            logger.warning("cold read of %s failed: %s", key.hex(), err)
            cls.counters["cold_read_failed"] += 1
            cls.drop_entry(key, entry, quarantine=False)
            self._forget(key)
            return
        cls.mark_ready(entry)
        entry.promoting = False
        if key in cls.pins:
            cls.pin_entry(key, entry)
        else:  # unpinned while it was read: a run of its own, evicted first
            cls.arena.orphan(cls, key, entry)
        if key in self.disk:
            self.disk.move_to_end(key)
        cls.counters["cold_promoted"] += 1

    def _forget(self, key: bytes) -> None:
        size = self.disk.pop(key, None)
        if size is not None:
            self.disk_bytes -= size
        try:
            os.unlink(self.path(key))
        except FileNotFoundError:
            pass

    def _enforce_cap(self) -> None:
        for key in list(self.disk):
            if self.disk_bytes <= self.max_bytes:
                break
            if key in self.cls.pins or key in self._reading:
                continue
            self._forget(key)
            self.cls.counters["cold_deleted"] += 1

    def persist(self, key: bytes, entry: Entry) -> None:
        """Write a pinned chunk. The write holds a reference, so the row
        neither moves nor goes while it is copied out."""
        if key in self.disk or key in self._writing:
            return
        self.cls.ref(entry)
        self._writing[key] = entry
        self.worker.submit(("w", key, entry.row))
        if self._sync:
            self.worker.run_pending()

    def start_promotion(self, key: bytes) -> bool:
        """Read a pinned chunk back into the pin zone (never evicting)."""
        cls = self.cls
        if key not in cls.pins or key not in self.disk or key in self._reading:
            return False
        row = cls.alloc(pin=True)
        if row is None:
            cls.counters["cold_promote_no_space"] += 1
            return False
        now = cls.arena.clock()
        entry = Entry(row=row, ready=False, writer=None, promoting=True,
                      stored_at=now, last_use=now)
        cls.add_entry(key, entry)
        self._reading[key] = entry
        self.worker.submit(("r", key, row))
        if self._sync:
            self.worker.run_pending()
        return True

    def reading(self, key: bytes) -> bool:
        return key in self._reading

    def close(self) -> None:
        self.worker.close()

    def stats(self) -> dict:
        return {
            "dir": self.dir,
            "chunks": len(self.disk),
            "bytes": self.disk_bytes,
            "max_bytes": self.max_bytes,
            "writing": len(self._writing),
            "reading": len(self._reading),
        }


class LayoutClass:
    """One model layout's rows, index, runs and pins inside the arena."""

    def __init__(self, arena: "Arena", name: str, cfg: dict, layout_key: str,
                 stride: int, state_groups) -> None:
        sp = arena.sp_bytes
        if stride <= 0 or stride > sp:
            raise ValueError(f"class {name!r}: a row of {stride} bytes does not fit a "
                             f"superpage of {sp} bytes; raise superpage_bytes")
        self.max_rows = sp // stride
        self.min_rows = self.max_rows if sp % stride == 0 else self.max_rows - 1
        if self.min_rows < 1:
            raise ValueError(f"class {name!r}: rows of {stride} bytes do not fit every "
                             f"superpage of {sp} bytes; use a superpage_bytes that is a "
                             "multiple of the row or at least twice its size")
        if sp % stride:
            logger.warning("class %s: rows of %d bytes leave up to %.1f%% of each "
                           "%d-byte superpage unused", name, stride,
                           100.0 * (sp - self.min_rows * stride) / sp, sp)
        self.arena = arena
        self.name = name
        self.cfg = cfg
        self.layout_key = layout_key
        self.stride = stride
        self.state_groups = frozenset(int(g) for g in state_groups or ())
        # Exclusive: the arena extends the GPU caches, so a completed load
        # makes its run oldest. Otherwise loads do not reorder.
        self.exclusive = bool(cfg.get("exclusive", True))
        # A chunk another client started writing longer ago than this is a
        # MISS to everyone else: they recompute instead of waiting on a slow
        # writer (under deep prefills a writer's step can take seconds).
        self.pending_wait_sec = float(cfg.get("pending_wait_sec", 2.0))
        self.num_rows = arena.size // stride
        self.index: dict[bytes, Entry] = {}
        self.row_key: dict[int, bytes] = {}
        self.runs: dict[bytes, Run] = {}
        # Owned superpage -> its free rows; partial: the superpages with a
        # free row, in allocation order (normal and pin zone apart).
        self.sp_free: dict[int, list[int]] = {}
        self.partial: dict[int, None] = {}
        self.pin_partial: dict[int, None] = {}
        self.draining: set[int] = set()
        self.free_rows = 0  # free rows of the normal superpages in partial
        self.pins: set[bytes] = set()
        self.pin_wanted: set[bytes] = set()
        self.clients = 0
        self.cold: ColdTier | None = None
        self.counters = collections.Counter()
        # Kept as entries change, so stats cost nothing however large the index.
        self.n_pending = 0  # not ready: being written or read
        self.n_refd = 0     # held by a reader, a pin or a disk write
        self.n_pinned = 0

    # --- geometry ---

    def rows_of(self, sp: int) -> range:
        s, b = self.stride, self.arena.sp_bytes
        return range(-(-sp * b // s), (sp + 1) * b // s)

    def sp_of(self, row: int) -> int:
        return row * self.stride // self.arena.sp_bytes

    def offset(self, row: int) -> int:
        return row * self.stride

    def boundary_complete(self, groups: set[int]) -> bool:
        return self.state_groups <= groups and len(groups) > len(self.state_groups)

    def available(self) -> int:
        """Rows this class can allocate now without evicting anything."""
        return self.free_rows + self.arena.claimable() * self.min_rows

    # --- rows ---

    def alloc(self, pin: bool = False, claim: bool = True) -> int | None:
        part = self.pin_partial if pin else self.partial
        if not part and not (claim and self.arena.claim_sp(self, pin)):
            return None
        sp = next(iter(part))
        rows = self.sp_free[sp]
        row = rows.pop()
        self.arena.sp_used[sp] += 1
        if not pin:
            self.free_rows -= 1
        if not rows:
            del part[sp]
        return row

    def give_back(self, row: int) -> None:
        """Return a row nobody uses any more; an emptied superpage goes back
        to the arena."""
        a = self.arena
        sp = self.sp_of(row)
        a.sp_used[sp] -= 1
        if a.sp_used[sp] == 0:
            a.release_sp(sp)
            return
        self.sp_free[sp].append(row)
        if sp in self.draining:
            return
        if a.sp_pin[sp]:
            self.pin_partial.setdefault(sp, None)
        else:
            if sp not in self.partial:
                self.partial[sp] = None
            self.free_rows += 1

    def undrain(self, sp: int) -> None:
        """Give up draining a superpage (a row move failed, so it can never
        empty): its free rows become allocatable again and compaction no
        longer counts it as supply."""
        self.draining.discard(sp)
        free = self.sp_free[sp]
        if free:
            self.partial[sp] = None
            self.free_rows += len(free)

    def add_entry(self, key: bytes, entry: Entry) -> None:
        self.index[key] = entry
        self.row_key[entry.row] = key
        self.n_pending += not entry.ready

    def mark_ready(self, entry: Entry) -> None:
        entry.ready = True
        self.n_pending -= 1

    def forget(self, key: bytes, entry: Entry) -> None:
        """Take a chunk out of the index (its row is the caller's to free)."""
        del self.index[key]
        del self.row_key[entry.row]
        self.n_pending -= not entry.ready
        self.n_refd -= entry.refs > 0
        self.n_pinned -= entry.pinned

    def drop_entry(self, key: bytes, entry: Entry, quarantine: bool) -> None:
        """Remove a chunk from the index and from every run it is in."""
        self.forget(key, entry)
        for run in list(entry.runs):
            self.arena.run_discard(run, key)
        entry.runs.clear()
        self.pin_wanted.discard(key)
        if quarantine:
            self.arena.quarantine(self, entry.row)
        else:
            self.give_back(entry.row)

    def ref(self, entry: Entry) -> None:
        self.n_refd += entry.refs == 0
        entry.refs += 1

    def unref(self, key: bytes, entry: Entry) -> None:
        assert entry.refs > 0, f"refcount underflow on {key.hex()}"
        entry.refs -= 1
        self.n_refd -= entry.refs == 0

    def movable(self, entry: Entry) -> bool:
        """Ready and used by nobody: free to evict or to move."""
        return (entry.ready and entry.moving is None and not entry.promoting
                and entry.refs == 0)

    def busy(self) -> bool:
        return any(not e.ready or e.moving is not None or e.promoting
                   or e.refs > (1 if e.pinned else 0) for e in self.index.values())

    def pin_entry(self, key: bytes, entry: Entry) -> None:
        if entry.pinned:
            return
        entry.pinned = True
        self.n_pinned += 1
        self.ref(entry)
        self.pin_wanted.discard(key)
        if self.cold is not None:
            self.cold.persist(key, entry)

    def unrestorable_runs(self) -> int:
        return sum(1 for r in self.arena.unrestorable if r.cls is self)

    def stats(self) -> dict:
        ready = len(self.index) - self.n_pending
        held = min(ready, self.n_refd + self.counters["rows_moving"])
        out = {
            "pool": self.name,
            "layout_key": self.layout_key,
            "num_chunks": self.num_rows,
            "row_stride": self.stride,
            "used": len(self.index),
            "ready": ready,
            "write_pending": len(self.index) - ready,
            "evictable": ready - held,
            "held": held,
            "pinned": self.n_pinned,
            "free": self.available(),
            "superpages": len(self.sp_free),
            "pin_superpages": sum(1 for sp in self.sp_free if self.arena.sp_pin[sp]),
            "runs": len(self.runs),
            "unrestorable_runs": self.unrestorable_runs(),
            "clients": self.clients,
            "counters": dict(self.counters),
        }
        if self.cold is not None:
            out["cold"] = self.cold.stats()
        return out


class Arena:
    """Superpages, classes and the global run order of one arena file. Not
    thread safe: the server is a single event loop, so every method runs to
    completion before the next request; I/O threads only copy row bytes and
    report back through a queue."""

    def __init__(self, path: str | None, size: int, superpage_bytes: int,
                 pin_max_bytes: int = 0, quarantine_sec: float = 30.0,
                 clock=time.monotonic, mover_threads: int = 0, notify=None) -> None:
        if superpage_bytes <= 0 or size < superpage_bytes:
            raise ValueError(f"an arena of {size} bytes holds no superpage of "
                             f"{superpage_bytes} bytes")
        self.path = path
        self.sp_bytes = superpage_bytes
        self.num_sp = size // superpage_bytes
        self.size = self.num_sp * superpage_bytes
        self.pin_sp_max = pin_max_bytes // superpage_bytes
        self.quarantine_sec = quarantine_sec
        self.clock = clock
        self.sp_owner: list[LayoutClass | None] = [None] * self.num_sp
        self.sp_pin = bytearray(self.num_sp)
        self.sp_used = [0] * self.num_sp
        self.free_sp = list(range(self.num_sp - 1, -1, -1))  # pop() = lowest
        self.pin_sp = 0
        self.classes: dict[str, LayoutClass] = {}
        self.order: collections.OrderedDict[Run, None] = collections.OrderedDict()
        self.unrestorable: dict[Run, None] = {}
        self._quarantine: collections.deque[tuple[float, LayoutClass, int]] = (
            collections.deque())
        # Read references force-dropped (a client gone while its loads may
        # still be copying): released only after the quarantine.
        self._held_back: collections.deque[tuple[float, LayoutClass, bytes, Entry]] = (
            collections.deque())
        # An adopted arena (a previous server's file, still mapped by engines)
        # hands out no superpage until this time.
        self.adopt_until = 0.0
        self.fd: int | None = None
        self.mm: mmap.mmap | None = None
        self.counters = collections.Counter()
        self.mover = Worker("mover", mover_threads, self._move_execute, self._move_apply,
                            notify)

    def open_file(self, fd: int, mm: mmap.mmap) -> None:
        self.fd, self.mm = fd, mm

    def adopt(self) -> None:
        self.adopt_until = self.clock() + self.quarantine_sec

    def close(self, unlink: bool = False) -> None:
        for cls in self.classes.values():
            if cls.cold is not None:
                cls.cold.close()
        self.mover.close()
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

    # --- classes ---

    def class_for(self, name: str, cfg: dict, layout_key: str, stride: int,
                  state_groups) -> LayoutClass:
        """The class for an engine of this layout. A class whose layout
        changed is replaced only when nothing uses it; its rows go through the
        quarantine, so a transfer of a departed engine cannot land in a reused
        row."""
        groups = frozenset(int(g) for g in state_groups or ())
        cls = self.classes.get(name)
        if cls is not None and (cls.layout_key, cls.stride, cls.state_groups) == (
                layout_key, stride, groups):
            return cls
        if cls is not None:
            if cls.clients or cls.busy():
                raise ValueError(
                    f"pool {name!r} holds layout {cls.layout_key} with rows of "
                    f"{cls.stride} bytes and state groups {sorted(cls.state_groups)}; "
                    f"this client has {layout_key} with rows of {stride} bytes and "
                    f"state groups {sorted(groups)}")
            logger.info("class %s: layout %s replaced by %s; dropping %d chunks",
                        name, cls.layout_key, layout_key, len(cls.index))
            self.retire(cls)
        cls = LayoutClass(self, name, cfg, layout_key, stride, groups)
        self.classes[name] = cls
        return cls

    def retire(self, cls: LayoutClass) -> None:
        for run in list(cls.runs.values()):
            self.remove_run(run)
        for key, entry in list(cls.index.items()):
            cls.drop_entry(key, entry, quarantine=True)
        if cls.cold is not None:
            cls.cold.close()
            cls.cold = None
        del self.classes[cls.name]

    # --- superpages ---

    def claimable(self) -> int:
        return len(self.free_sp) if self.clock() >= self.adopt_until else 0

    def claim_sp(self, cls: LayoutClass, pin: bool) -> bool:
        if not self.claimable() or (pin and self.pin_sp >= self.pin_sp_max):
            return False
        sp = self.free_sp.pop()
        self.sp_owner[sp] = cls
        self.sp_pin[sp] = pin
        rows = list(cls.rows_of(sp))
        rows.reverse()
        cls.sp_free[sp] = rows
        if pin:
            self.pin_sp += 1
            cls.pin_partial[sp] = None
        else:
            cls.partial[sp] = None
            cls.free_rows += len(rows)
        return True

    def release_sp(self, sp: int) -> None:
        cls = self.sp_owner[sp]
        assert cls is not None and self.sp_used[sp] == 0
        free = cls.sp_free.pop(sp)
        if self.sp_pin[sp]:
            self.pin_sp -= 1
            cls.pin_partial.pop(sp, None)
        elif sp in cls.draining:
            cls.draining.discard(sp)
        elif sp in cls.partial:
            del cls.partial[sp]
            cls.free_rows -= len(free)
        self.sp_owner[sp] = None
        self.sp_pin[sp] = 0
        self.free_sp.append(sp)

    def pin_in_place(self, cls: LayoutClass, entry: Entry) -> bool:
        """Turn the superpage of a chunk that is its superpage's only row into
        a pin superpage instead of copying the row."""
        sp = cls.sp_of(entry.row)
        if len(cls.rows_of(sp)) != 1 or self.pin_sp >= self.pin_sp_max:
            return False
        self.sp_pin[sp] = 1
        self.pin_sp += 1
        return True

    def owners(self) -> set[LayoutClass]:
        return {c for c in self.sp_owner if c is not None}

    # --- quarantine ---

    def quarantine(self, cls: LayoutClass, row: int) -> None:
        self._quarantine.append((self.clock() + self.quarantine_sec, cls, row))

    def hold_back(self, cls: LayoutClass, key: bytes, entry: Entry) -> None:
        self._held_back.append((self.clock() + self.quarantine_sec, cls, key, entry))

    def in_quarantine(self) -> bool:
        return bool(self._quarantine or self._held_back) or self.clock() < self.adopt_until

    def maintain(self) -> None:
        """Release what the quarantine is done with; retry wanted pins."""
        now = self.clock()
        while self._quarantine and self._quarantine[0][0] <= now:
            _, cls, row = self._quarantine.popleft()
            cls.give_back(row)
        while self._held_back and self._held_back[0][0] <= now:
            _, cls, key, entry = self._held_back.popleft()
            if cls.index.get(key) is entry:
                cls.unref(key, entry)
        for cls in self.classes.values():
            for key in list(cls.pin_wanted):
                entry = cls.index.get(key)
                if entry is None:
                    cls.pin_wanted.discard(key)  # pinned when stored again
                else:
                    self.request_pin(cls, key, entry)

    # --- runs ---

    def get_run(self, cls: LayoutClass, tag: bytes) -> Run:
        run = cls.runs.get(tag)
        if run is None:
            run = cls.runs[tag] = Run(cls, tag, self.clock())
            self.order[run] = None
            self._sync(run)
        return run

    def _sync(self, run: Run) -> None:
        if run.restorable:
            self.unrestorable.pop(run, None)
        else:
            self.unrestorable.setdefault(run, None)

    def join(self, run: Run, key: bytes, entry: Entry) -> None:
        if run not in entry.runs:
            entry.runs.add(run)
            run.add(key)
            self._sync(run)

    def run_discard(self, run: Run, key: bytes) -> None:
        run.discard(key)
        if not run.keys:
            self.remove_run(run)
        else:
            self._sync(run)

    def remove_run(self, run: Run) -> None:
        self.order.pop(run, None)
        self.unrestorable.pop(run, None)
        if run.cls.runs.get(run.tag) is run:
            del run.cls.runs[run.tag]
        for client in run.demoted_by:
            client.demoted.pop(run, None)
        run.demoted_by.clear()

    def refresh(self, run: Run) -> None:
        """A store to the run: newest, and no longer demoted by anyone."""
        self.order.move_to_end(run)
        run.stored_at = self.clock()
        for client in run.demoted_by:
            client.demoted.pop(run, None)
        run.demoted_by.clear()

    def orphan(self, cls: LayoutClass, key: bytes, entry: Entry) -> None:
        """An unpinned chunk in no run: a run of its own at the oldest end."""
        if entry.runs:
            return
        run = self.get_run(cls, _ORPHAN + key)
        self.join(run, key, entry)
        self.order.move_to_end(run, last=False)

    def evict_run(self, run: Run) -> list[bytes]:
        """Evict a run; free its chunks that no other run holds."""
        cls = run.cls
        self.remove_run(run)
        freed = []
        for key in run.keys:
            entry = cls.index[key]
            entry.runs.discard(run)
            if entry.runs or entry.pinned:
                continue
            assert cls.movable(entry), f"evicting a busy chunk {key.hex()}"
            cls.forget(key, entry)
            cls.give_back(entry.row)
            freed.append(key)
        cls.counters["runs_evicted"] += 1
        cls.counters["chunks_evicted"] += len(freed)
        if not run.restorable:
            cls.counters["unrestorable_runs_evicted"] += 1
        return freed

    def _candidates(self):
        """Runs that restore nothing first, then every run oldest first."""
        cutoff = self.clock() - DEAD_RUN_GRACE_SEC
        dead = [r for r in self.unrestorable if r.stored_at <= cutoff]
        return itertools.chain(dead, self.order)

    def plan(self, cls: LayoutClass, need: int, keep: set[bytes],
             protect: Run | None) -> tuple[list[Run], int] | None:
        """Simulate the eviction walk over every class until ``need`` rows of
        ``cls`` can be had. Returns (victims, superpages to compact), or None
        when even evicting every candidate cannot free enough. Freed rows of
        another class count only as whole superpages, emptied directly or by
        compacting a class with several rows per superpage, which is preferred
        to evicting more. ``keep`` (the offered keys) are never freed."""
        adopting = self.clock() < self.adopt_until
        direct = cls.available()
        others = [c for c in self.owners() if c is not cls and c.max_rows > 1]
        slack = {c: c.free_rows for c in others}
        draining = sum(len(c.draining) for c in others)  # compactions in flight

        def shortfall() -> int | None:
            """Superpages compaction must free, or None if not enough yet."""
            if direct >= need:
                return 0
            missing = -(-(need - direct) // cls.min_rows)
            have = 0 if adopting else draining + sum(s // c.max_rows for c, s in slack.items())
            return missing if have >= missing else None

        if (sps := shortfall()) is not None:
            return [], sps
        gone: set[Run] = set()
        victims: list[Run] = []
        freed_in_sp: collections.Counter = collections.Counter()
        for run in self._candidates():
            if run in gone or run is protect:
                continue
            c = run.cls
            frees = []
            for key in run.keys:
                if c is cls and key in keep:
                    continue
                entry = c.index[key]
                if entry.pinned or any(r is not run and r not in gone for r in entry.runs):
                    continue  # survives in another run
                if not c.movable(entry):
                    break
                frees.append(entry)
            else:
                gone.add(run)
                victims.append(run)
                for entry in frees:
                    sp = c.sp_of(entry.row)
                    normal = not self.sp_pin[sp] and sp not in c.draining
                    if c is cls and normal:
                        direct += 1
                        continue
                    freed_in_sp[sp] += 1
                    if c in slack and normal:
                        slack[c] += 1
                    if self.sp_used[sp] == freed_in_sp[sp]:  # the superpage empties
                        if not adopting:
                            direct += cls.min_rows
                        if c in slack and normal:
                            slack[c] -= len(c.sp_free[sp]) + freed_in_sp[sp]
                if (sps := shortfall()) is not None:
                    return victims, sps
        return None

    # --- compaction ---

    def compact(self, want: int, exclude: LayoutClass) -> int:
        """Move rows of other classes with several rows per superpage out of
        their emptiest superpages until ``want`` superpages will be freed.
        Rows in flight answer HIT_PENDING; nothing is evicted or reordered."""
        others = [c for c in self.owners() if c is not exclude and c.max_rows > 1]
        want -= sum(len(c.draining) for c in others)  # already on their way
        started = 0
        for cls in sorted(others, key=lambda c: -c.free_rows):
            while started < want:
                src = self._drain_source(cls)
                if src is None:
                    break
                self._drain_superpage(cls, src)
                started += 1
        self.counters["superpages_compacted"] += started
        return started

    def _drain_source(self, cls: LayoutClass) -> int | None:
        """The normal superpage with the fewest used rows, all movable, whose
        rows fit in the class's other free rows. Pin superpages are never
        drained, and pinned chunks live only there."""
        best, best_used = None, None
        for sp, free in cls.sp_free.items():
            if self.sp_pin[sp] or sp in cls.draining:
                continue
            used = self.sp_used[sp]
            if best_used is not None and used >= best_used:
                continue
            if used > cls.free_rows - len(free):
                continue
            keys = [cls.row_key.get(r) for r in cls.rows_of(sp) if r not in free]
            if any(k is None or not cls.movable(cls.index[k]) for k in keys):
                continue  # a quarantined or busy row
            best, best_used = sp, used
        return best

    def _drain_superpage(self, cls: LayoutClass, sp: int) -> None:
        free = cls.sp_free[sp]
        if sp in cls.partial:
            del cls.partial[sp]
            cls.free_rows -= len(free)
        cls.draining.add(sp)
        for row in cls.rows_of(sp):
            key = cls.row_key.get(row)
            if key is None:
                continue
            dst = cls.alloc(claim=False)
            assert dst is not None, "compaction planned more rows than are free"
            self.start_move(cls, key, cls.index[key], dst)

    def start_move(self, cls: LayoutClass, key: bytes, entry: Entry, dst: int) -> None:
        entry.moving = dst
        cls.counters["rows_moving"] += 1
        self.mover.submit((cls, key, entry, entry.row, dst))

    def _move_execute(self, job: tuple) -> None:
        cls, _, _, src, dst = job
        if self.mm is None or self.fd is None:
            return  # an index-only arena (tests)
        length = cls.stride
        view = memoryview(self.mm)[cls.offset(src) : cls.offset(src) + length]
        try:
            pos = 0
            while pos < length:  # pwrite releases the GIL, a slice copy would not
                pos += os.pwrite(self.fd, view[pos:], cls.offset(dst) + pos)
        finally:
            view.release()

    def _move_apply(self, job: tuple, err: str | None) -> None:
        cls, key, entry, src, dst = job
        cls.counters["rows_moving"] -= 1
        if cls.index.get(key) is not entry or entry.moving != dst:
            cls.give_back(dst)
            return
        entry.moving = None
        if err is not None:
            logger.warning("moving %s failed: %s", key.hex(), err)
            cls.counters["move_failed"] += 1
            cls.give_back(dst)
            if cls.sp_of(src) in cls.draining:
                cls.undrain(cls.sp_of(src))
            return
        del cls.row_key[src]
        cls.row_key[dst] = key
        entry.row = dst
        cls.give_back(src)
        cls.counters["rows_moved"] += 1
        if key in cls.pins and not entry.pinned and self.sp_pin[cls.sp_of(dst)]:
            cls.pin_entry(key, entry)

    # --- engine operations (on the client's class) ---

    def lookup(self, client: ClientState, keys: list[bytes],
               stop_at_miss: bool = False) -> list[int]:
        cls = client.cls
        assert cls is not None
        out = []
        now = self.clock()
        for key in keys:
            entry = cls.index.get(key)
            if entry is None:
                if cls.cold is not None and cls.cold.start_promotion(key):
                    out.append(P.HIT_PENDING)
                else:
                    out.append(P.MISS)
                    cls.counters["lookup_miss"] += 1
                    if stop_at_miss:
                        break
            elif not entry.ready:
                if (entry.writer is not None and entry.writer != client.client_id
                        and now - entry.stored_at > cls.pending_wait_sec):
                    out.append(P.MISS)
                    cls.counters["lookup_pending_as_miss"] += 1
                    if stop_at_miss:
                        break
                else:
                    out.append(P.HIT_PENDING)
            elif entry.moving is not None:
                out.append(P.HIT_PENDING)
                cls.counters["lookup_moving"] += 1
            else:
                if key not in client.holds:
                    client.holds[key] = None
                    cls.ref(entry)
                entry.last_use = now
                out.append(P.HIT)
                cls.counters["lookup_hit"] += 1
        return out

    def end_session(self, client: ClientState) -> None:
        cls = client.cls
        assert cls is not None
        for key in reversed(client.holds):
            entry = cls.index.get(key)
            if entry is not None:
                cls.unref(key, entry)
        client.holds.clear()

    def prepare_load(self, client: ClientState, keys: list[bytes]) -> list[int]:
        cls = client.cls
        assert cls is not None
        entries = []
        for key in keys:
            entry = cls.index.get(key)
            if entry is None or not entry.ready or entry.moving is not None:
                raise KeyError(
                    f"prepare_load of {key.hex()} which is not ready; a lookup "
                    "HIT must be held by the session that loads it")
            entries.append(entry)
        now = self.clock()
        for key, entry in zip(keys, entries):
            cls.ref(entry)
            client.loads[key] += 1
            entry.last_use = now
        cls.counters["chunks_loaded"] += len(keys)
        return [e.row for e in entries]

    def complete_load(self, client: ClientState, keys: list[bytes]) -> None:
        cls = client.cls
        assert cls is not None
        keys = list(keys)
        for key in keys:
            if client.loads[key] <= 0:
                raise KeyError(f"complete_load of {key.hex()} without prepare_load")
        entries = []
        for key in reversed(keys):
            client.loads[key] -= 1
            if client.loads[key] == 0:
                del client.loads[key]
            entry = cls.index.get(key)
            if entry is not None:
                cls.unref(key, entry)
                entries.append(entry)
        if not cls.exclusive or not entries:
            return
        # The loading engine's GPU holds this chain again: demote the runs
        # that contain all of it (the conversation it continues), not the
        # other conversations a shared head also belongs to.
        wanted = set(keys)
        for run in min((e.runs for e in entries), key=len):
            if wanted.issubset(run.keys):
                self.order.move_to_end(run, last=False)
                run.demoted_by.add(client)
                client.demoted[run] = None
                cls.counters["demoted_on_load"] += 1

    def prepare_store(self, client: ClientState, keys: list[bytes], tag: bytes,
                      whole: bool = False) -> tuple[list[bytes], list[int], list[bytes]] | None:
        """Allocate rows for the keys the class does not hold yet; every
        offered key joins run tag, which becomes newest. Space comes from
        evicting whole older runs of any class, or from compacting. All or
        nothing: returns None and changes nothing when the walk cannot free
        enough. A store that needs a compaction returns None once it has
        started it (with the evictions it needed); the next store succeeds."""
        cls = client.cls
        assert cls is not None
        self.maintain()
        offered = list(dict.fromkeys(keys))
        new_keys = [k for k in offered if k not in cls.index]
        cls.counters["store_dedup"] += len(keys) - len(new_keys)
        run = cls.runs.get(tag)
        if whole and not (run or Run(cls, tag, 0.0)).restorable_with(offered):
            cls.counters["store_rejected_unrestorable"] += 1
            return None
        victims: list[Run] = []
        compact = 0
        if new_keys:
            plan = self.plan(cls, len(new_keys), set(offered), run)
            if plan is None:
                cls.counters["store_rejected"] += 1
                return None
            victims, compact = plan
        if run is None:
            run = self.get_run(cls, tag)
        else:
            self.refresh(run)
        for key in offered:
            entry = cls.index.get(key)
            if entry is not None:
                self.join(run, key, entry)
        evicted: list[bytes] = []
        for victim in victims:
            freed = self.evict_run(victim)
            if victim.cls is cls:
                evicted += freed
            else:
                self.counters["cross_class_runs_evicted"] += 1
        if compact:
            self.compact(compact, exclude=cls)
        rows: list[int] = []
        for key in new_keys:
            row = cls.alloc(pin=True) if key in cls.pins else None
            row = row if row is not None else cls.alloc()
            if row is None:
                break
            rows.append(row)
        if len(rows) < len(new_keys):
            for row in rows:
                cls.give_back(row)
            if not run.keys:
                self.remove_run(run)
            cls.counters["store_waits_compaction" if compact else "store_rejected"] += 1
            return None
        now = self.clock()
        for key, row in zip(new_keys, rows):
            entry = Entry(row=row, ready=False, writer=client.client_id,
                          stored_at=now, last_use=now)
            cls.add_entry(key, entry)
            self.join(run, key, entry)
            client.writes.add(key)
        cls.counters["chunks_stored"] += len(new_keys)
        return new_keys, rows, evicted

    def merge_tag(self, client: ClientState, src: bytes, dst: bytes) -> int:
        """Move every key of run src into run dst (the conversation). An
        existing dst keeps its place in the order; a new one takes src's."""
        cls = client.cls
        assert cls is not None
        run = cls.runs.get(src)
        if run is None or src == dst:
            return 0
        n = len(run.keys)
        target = cls.runs.get(dst)
        if target is None:
            del cls.runs[src]
            run.tag = dst
            cls.runs[dst] = run
            return n
        for key in list(run.keys):
            entry = cls.index[key]
            entry.runs.discard(run)
            self.join(target, key, entry)
        for c in run.demoted_by:
            target.demoted_by.add(c)
            c.demoted[target] = None
        self.remove_run(run)
        cls.counters["tags_merged"] += 1
        return n

    def complete_store(self, client: ClientState, keys: list[bytes], success: bool) -> None:
        cls = client.cls
        assert cls is not None
        for key in reversed(keys):
            if key not in client.writes:
                continue
            client.writes.discard(key)
            entry = cls.index.get(key)
            if entry is None or entry.ready or entry.writer != client.client_id:
                continue
            if success:
                cls.mark_ready(entry)
                entry.writer = None
                if key in cls.pins:
                    self.request_pin(cls, key, entry)
            else:
                cls.drop_entry(key, entry, quarantine=True)
                cls.counters["store_failed"] += 1

    def touch(self, client: ClientState, keys: list[bytes]) -> None:
        cls = client.cls
        assert cls is not None
        now = self.clock()
        for key in reversed(keys):
            entry = cls.index.get(key)
            if entry is None:
                continue
            entry.last_use = now
            for run in entry.runs:
                self.order.move_to_end(run)

    def release_client(self, client: ClientState) -> None:
        """Drop everything a client holds: on disconnect or a cache reset."""
        cls = client.cls
        assert cls is not None
        self.end_session(client)
        for key, n in client.loads.items():
            entry = cls.index.get(key)
            if entry is not None:
                # Keep one reference until the quarantine ends: a worker of the
                # departed client may still be copying this chunk.
                for _ in range(n - 1):
                    cls.unref(key, entry)
                self.hold_back(cls, key, entry)
                cls.counters["loads_held_back"] += 1
        client.loads.clear()
        for key in client.writes:
            entry = cls.index.get(key)
            if entry is not None and not entry.ready and entry.writer == client.client_id:
                cls.drop_entry(key, entry, quarantine=True)
                cls.counters["store_aborted"] += 1
        client.writes.clear()
        # Its GPU copies are gone: the runs it loaded are the only copies now.
        for run in client.demoted:
            run.demoted_by.discard(client)
            if run in self.order:
                self.order.move_to_end(run)
                cls.counters["promoted_on_release"] += 1
        client.demoted.clear()

    # --- pins ---

    def request_pin(self, cls: LayoutClass, key: bytes, entry: Entry) -> None:
        """Pin a stored chunk: where it is if that is the pin zone or the
        chunk is its superpage's only row, else by moving it there; a chunk in
        use is retried by maintain()."""
        if entry.pinned:
            cls.pin_wanted.discard(key)
            return
        if entry.ready and entry.moving is None and not entry.promoting:
            if self.sp_pin[cls.sp_of(entry.row)] or self.pin_in_place(cls, entry):
                cls.pin_entry(key, entry)
                return
            if entry.refs == 0:
                cls.pin_wanted.discard(key)
                dst = cls.alloc(pin=True)
                if dst is None:
                    cls.counters["pin_zone_full"] += 1
                    return
                self.start_move(cls, key, entry, dst)
                return
        cls.pin_wanted.add(key)

    def pin(self, cls: LayoutClass, keys: list[bytes]) -> int:
        """Pin chunks: never evicted, kept on disk, restored after a restart.
        A key not resident is pinned when it is next stored or read back.
        Returns how many chunks are pinned or moving into the pin zone."""
        n = 0
        for key in keys:
            entry = cls.index.get(key)
            if cls.cold is None and (entry is None or not entry.ready):
                continue  # without a cold tier nothing would pin it later
            cls.pins.add(key)
            if entry is None:
                cls.cold.start_promotion(key)
            elif entry.ready:
                self.request_pin(cls, key, entry)
                n += entry.pinned or entry.moving is not None
        if cls.cold is not None:
            cls.cold.save_pins()
        return n

    def pin_recent(self, cls: LayoutClass, seconds: float, until_ago: float = 0.0) -> int:
        """Pin every chunk used between ``seconds`` and ``until_ago`` seconds
        ago, for an operator that has just primed a prefix. Chunks still being
        written are pinned when their store completes."""
        now = self.clock()
        since, until = now - seconds, now - until_ago
        return self.pin(cls, [k for k, e in cls.index.items() if since <= e.last_use <= until])

    def unpin(self, cls: LayoutClass, keys: list[bytes] | None) -> int:
        """Unpin the given keys, or every pin when keys is None."""
        n = 0
        for key in list(cls.pins) if keys is None else keys:
            cls.pins.discard(key)
            cls.pin_wanted.discard(key)
            entry = cls.index.get(key)
            if entry is not None and entry.pinned:
                entry.pinned = False
                cls.n_pinned -= 1
                cls.unref(key, entry)
                self.orphan(cls, key, entry)
                n += 1
        if cls.cold is not None:
            cls.cold.save_pins()
        return n


class SpillStore:
    """Whole runs written to NVMe when the arena is released, read back when
    it is created again. One file per chunk, named by class, layout and key, so
    a chunk still on disk from the last spill is not written again. The
    manifest (classes, runs oldest first) is replaced atomically; files it no
    longer names are deleted only once it is durable."""

    def __init__(self, directory: str | None, max_bytes: int, free_floor: int) -> None:
        self.dir = directory
        self.max_bytes = max_bytes if directory else 0
        self.free_floor = free_floor
        self.files: dict[str, int] = {}  # relative path -> bytes, as the manifest names
        self.manifest_time: float | None = None
        self.manifest_runs = 0

    def key_path(self, cls_name: str, layout_key: str, key: bytes) -> str:
        h = key.hex()
        return os.path.join(f"{cls_name}_{layout_key}", h[:3], f"{h}.bin")

    def read_manifest(self) -> dict | None:
        if not self.dir:
            return None
        try:
            with open(os.path.join(self.dir, MANIFEST)) as f:
                manifest = json.load(f)
        except FileNotFoundError:
            return None
        except (OSError, ValueError) as exc:
            logger.warning("spill manifest unreadable (%s); ignoring it", exc)
            return None
        if manifest.get("version") != _MANIFEST_VERSION:
            logger.warning("spill manifest version %s; ignoring it", manifest.get("version"))
            return None
        return manifest

    def _manifest_files(self, manifest: dict) -> dict[str, int]:
        out = {}
        for item in manifest["runs"]:
            info = manifest["classes"][item["cls"]]
            for k in item["keys"]:
                out[self.key_path(item["cls"], info["layout_key"], bytes.fromhex(k))] = (
                    int(info["row_stride"]))
        return out

    def load(self) -> None:
        """At start: remember the files the manifest names and delete every
        other chunk file (those of an aborted or interrupted spill)."""
        if not self.dir:
            return
        os.makedirs(self.dir, exist_ok=True)
        manifest = self.read_manifest()
        self.files = self._manifest_files(manifest) if manifest else {}
        if manifest:
            self.manifest_time = manifest.get("time")
            self.manifest_runs = len(manifest["runs"])
        removed = 0
        for root, _, names in os.walk(self.dir):
            for name in names:
                full = os.path.join(root, name)
                if os.path.relpath(full, self.dir) in self.files:
                    continue
                if name.endswith((".bin", ".tmp")):
                    os.unlink(full)
                    removed += 1
        logger.info("spill %s: the manifest names %d chunk files (%.2f GB); removed %d others",
                    self.dir, len(self.files), sum(self.files.values()) / 1e9, removed)

    def select(self, arena: Arena) -> tuple[dict, list[tuple[str, int, int]]]:
        """The newest whole restorable runs within max_bytes, writing no more
        new bytes than NVMe has free above the floor (a chunk file already on
        disk costs nothing)."""
        free = 0
        if self.max_bytes > 0:
            st = os.statvfs(self.dir)
            free = st.f_bavail * st.f_frsize - self.free_floor
        total = new = 0
        chosen: list[Run] = []
        picked: set[tuple[LayoutClass, bytes]] = set()
        writes: list[tuple[str, int, int]] = []
        for run in reversed(arena.order) if self.max_bytes > 0 else ():
            cls = run.cls
            if not run.restorable:
                continue
            entries = [(k, cls.index[k]) for k in run.keys if (cls, k) not in picked]
            if any(not e.ready or e.moving is not None or e.promoting for _, e in entries):
                continue
            paths = [self.key_path(cls.name, cls.layout_key, k) for k, _ in entries]
            add_total = len(entries) * cls.stride
            add_new = sum(cls.stride for p in paths if p not in self.files)
            if total + add_total > self.max_bytes or new + add_new > free:
                continue
            total += add_total
            new += add_new
            chosen.append(run)
            for (k, e), path in zip(entries, paths):
                picked.add((cls, k))
                if path not in self.files:
                    writes.append((path, cls.offset(e.row), cls.stride))
        classes = {r.cls for r in chosen}
        manifest = {
            "version": _MANIFEST_VERSION,
            "time": time.time(),
            "superpage_bytes": arena.sp_bytes,
            "classes": {c.name: {"layout_key": c.layout_key, "row_stride": c.stride,
                                 "state_groups": sorted(c.state_groups)} for c in classes},
            "runs": [{"cls": r.cls.name, "tag": r.tag.hex(), "keys": [k.hex() for k in r.keys]}
                     for r in reversed(chosen)],
        }
        return manifest, writes

    def write(self, mm: mmap.mmap, manifest: dict, writes: list, abort: threading.Event) -> bool:
        """Write the new chunk files, then the manifest. Returns True once
        durable, False when aborted (the new files are removed again)."""
        if not self.dir:
            return True
        t0 = time.monotonic()
        done: list[str] = []
        dirs: set[str] = set()
        try:
            for rel, off, length in writes:
                if abort.is_set():
                    raise InterruptedError
                full = os.path.join(self.dir, rel)
                os.makedirs(os.path.dirname(full), exist_ok=True)
                _write_row_file(full + ".tmp", mm, off, length)
                os.replace(full + ".tmp", full)
                done.append(rel)
                dirs.add(os.path.dirname(full))
            if abort.is_set():
                raise InterruptedError
            for d in dirs:
                _fsync_dir(d)
            path = os.path.join(self.dir, MANIFEST)
            with open(path + ".tmp", "w") as f:
                json.dump(manifest, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(path + ".tmp", path)
            _fsync_dir(self.dir)
        except InterruptedError:
            for rel in done:
                os.unlink(os.path.join(self.dir, rel))
            return False
        keep = self._manifest_files(manifest)
        for rel in self.files.keys() - keep.keys():
            try:
                os.unlink(os.path.join(self.dir, rel))
            except FileNotFoundError:
                pass
        self.files = keep
        self.manifest_time = manifest["time"]
        self.manifest_runs = len(manifest["runs"])
        logger.info("spilled %d runs (%d chunk files, %d new, %.2f GB written) in %.1f s",
                    len(manifest["runs"]), len(keep), len(writes),
                    sum(w[2] for w in writes) / 1e9, time.monotonic() - t0)
        return True

    def restore(self, mm: mmap.mmap, jobs: list[tuple]) -> list[tuple[str, bytes]]:
        """Thread: read spilled chunks into their rows; returns the failures."""
        failed = []
        t0 = time.monotonic()
        for cls, key, row, rel in jobs:
            try:
                _read_row_file(os.path.join(self.dir, rel), mm, cls.offset(row), cls.stride)
            except OSError as exc:
                logger.warning("spilled chunk %s not restored: %s", rel, exc)
                failed.append((cls.name, key))
                # Forget the file, so the next spill writes a fresh copy.
                self.files.pop(rel, None)
                try:
                    os.unlink(os.path.join(self.dir, rel))
                except OSError:
                    pass
        logger.info("restored %d spilled chunks in %.1f s (%d failed)",
                    len(jobs) - len(failed), time.monotonic() - t0, len(failed))
        return failed

    def stats(self) -> dict:
        return {"dir": self.dir, "files": len(self.files), "bytes": sum(self.files.values()),
                "runs": self.manifest_runs, "manifest_time": self.manifest_time,
                "max_bytes": self.max_bytes}


class PoolServer:
    def __init__(self, socket_path: str, pools_cfg: dict, arena_cfg: dict,
                 shm_dir: str = "/dev/shm", quarantine_sec: float = 30.0,
                 populate: bool = True, socket_group: str | None = None,
                 clock=time.monotonic, mem_available=mem_available,
                 find_mappers=other_mappers, io_threads: int = 1) -> None:
        self.socket_path = socket_path
        # Run as root (no tmpfs user quota on the arena), let this group's
        # members (the router) reach the socket.
        self.socket_group = socket_group
        self.pools_cfg = pools_cfg
        self.shm_dir = shm_dir
        self.quarantine_sec = quarantine_sec
        self.populate = populate
        self.clock = clock
        self._mem_available = mem_available
        self._find_mappers = find_mappers
        self._io_threads = io_threads
        self.arena_path = str(arena_cfg.get("path")
                              or os.path.join(shm_dir, "vllm_kvpool_joint.mmap"))
        sp = arena_cfg.get("superpage_bytes")
        if sp is None:
            hints = [int(c["row_stride"]) for c in pools_cfg.values() if "row_stride" in c]
            if not hints:
                raise ValueError("arena: set superpage_bytes, or row_stride on a pool")
            sp = max(hints)
        self.superpage_bytes = int(sp)
        self.arena_size = int(arena_cfg["bytes"]) // self.superpage_bytes * self.superpage_bytes
        self.pin_max_bytes = int(arena_cfg.get("pin_max_bytes", 0))
        self.populate_floor = int(arena_cfg.get("populate_floor_bytes", 8 << 30))
        self.spill = SpillStore(arena_cfg.get("spill_dir"),
                                int(float(arena_cfg.get("spill_max_bytes", 0))),
                                int(float(arena_cfg.get("nvme_free_floor", 10e9))))
        self._server_id = f"{os.getpid():x}{int(time.time()):x}"
        self._arenas = 0  # arenas created or adopted by this server
        self.state = P.ABSENT
        self.arena: Arena | None = None
        self.spawn_tokens: dict[str, tuple[str, float]] = {}
        self.last_release: dict | None = None
        self.last_prepare: dict | None = None
        self._spill_abort = threading.Event()
        self._populated = 0  # bytes populated so far while CREATING (the thread's)
        self._life_q: queue.SimpleQueue = queue.SimpleQueue()
        self._life_thread: threading.Thread | None = None
        self._next_client = 1
        self._sel = selectors.DefaultSelector()
        self._conns: dict[socket.socket, Conn] = {}
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
        self._drain()

    def _drain(self) -> None:
        if self.arena is not None:
            self.arena.mover.drain()
            for cls in list(self.arena.classes.values()):
                if cls.cold is not None:
                    cls.cold.worker.drain()
        while True:
            try:
                kind, result = self._life_q.get_nowait()
            except queue.Empty:
                break
            self._life_thread = None
            getattr(self, f"_on_{kind}")(result)

    @property
    def generation(self) -> str:
        """This server and arena incarnation: it changes when either does."""
        return f"{self._server_id}.{self._arenas}"

    def _attached(self) -> int:
        return sum(1 for c in self._conns.values() if c.client.cls is not None)

    # --- lifecycle ---

    def _new_arena(self) -> Arena:
        return Arena(self.arena_path, self.arena_size, self.superpage_bytes,
                     self.pin_max_bytes, self.quarantine_sec, self.clock,
                     mover_threads=self._io_threads, notify=self._notify)

    def _start(self, kind: str, target, *args) -> None:
        def run() -> None:
            try:
                result = target(*args)
            except Exception as exc:  # reported to the event loop
                logger.exception("%s failed", kind)
                result = exc
            self._life_q.put((kind, result))
            self._notify()

        self._life_thread = threading.Thread(target=run, name=f"kvpool-{kind}", daemon=True)
        self._life_thread.start()

    def adopt_or_clear(self) -> None:
        """At start: an arena file another process still maps is adopted (the
        same inode, an empty index, every superpage quarantined); one nobody
        maps holds nothing an index knows of and is removed."""
        if not os.path.exists(self.arena_path):
            return
        pids = self._find_mappers(self.arena_path)
        if not pids:
            os.unlink(self.arena_path)
            logger.info("removed the unmapped arena %s an earlier server left", self.arena_path)
            return
        size = os.path.getsize(self.arena_path)
        if size != self.arena_size:
            raise RuntimeError(
                f"{self.arena_path} has {size} bytes, the arena needs {self.arena_size}, and "
                f"pids {pids} still map it: stop them and remove it")
        arena = self._new_arena()
        fd = os.open(self.arena_path, os.O_RDWR)
        arena.open_file(fd, mmap.mmap(fd, size, mmap.MAP_SHARED,
                                      mmap.PROT_READ | mmap.PROT_WRITE))
        arena.adopt()
        self.arena = arena
        self._arenas += 1
        self.state = P.READY
        logger.info("adopted the arena %s (mapped by %s); rows reusable in %.0f s",
                    self.arena_path, pids, self.quarantine_sec)

    def prepare(self, reason: str) -> None:
        if self.state in (P.READY, P.CREATING):
            return
        if self.state == P.SPILLING:
            self._spill_abort.set()
            logger.info("spill aborted: %s", reason)
            return
        avail = self._mem_available()
        if avail - self.arena_size < self.populate_floor:
            self.last_prepare = {
                "time": time.time(), "ok": False,
                "reason": f"MemAvailable {avail} minus the arena's {self.arena_size} bytes "
                          f"is under the floor {self.populate_floor}"}
            logger.warning("prepare (%s) refused: %s", reason, self.last_prepare["reason"])
            self._flush_backlogs(force=True)
            return
        self.state = P.CREATING
        self._populated = 0
        self.last_prepare = {"time": time.time(), "ok": None, "reason": reason}
        logger.info("creating the arena %s (%.2f GiB): %s", self.arena_path,
                    self.arena_size / (1 << 30), reason)
        self._start("created", self._create_file)

    def _create_file(self):
        """Thread: create and populate the arena, then read the spill manifest."""
        t0 = time.monotonic()
        size = self.arena_size
        fd = os.open(self.arena_path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
        try:
            os.ftruncate(fd, size)
            mm = mmap.mmap(fd, size, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE)
            try:
                if self.populate:
                    for off in range(0, size, _POPULATE_STEP):
                        avail = self._mem_available()
                        if avail - (size - off) < self.populate_floor:
                            raise MemoryError(
                                f"MemAvailable {avail} minus the {size - off} bytes still to "
                                f"populate is under the floor {self.populate_floor}")
                        mm.madvise(_MADV_POPULATE_WRITE, off, min(_POPULATE_STEP, size - off))
                        self._populated = min(off + _POPULATE_STEP, size)
            except BaseException:
                mm.close()
                raise
        except BaseException:
            os.close(fd)
            os.unlink(self.arena_path)
            raise
        logger.info("arena %s: %.2f GiB populated in %.1f s", self.arena_path,
                    size / (1 << 30), time.monotonic() - t0)
        return fd, mm, self.spill.read_manifest()

    def _on_created(self, result) -> None:
        if isinstance(result, Exception):
            self.state = P.ABSENT
            self.last_prepare = {"time": time.time(), "ok": False,
                                 "reason": f"{type(result).__name__}: {result}"}
            self._flush_backlogs(force=True)
            return
        fd, mm, manifest = result
        arena = self._new_arena()
        arena.open_file(fd, mm)
        self.arena = arena
        self._arenas += 1
        jobs = self._plan_restore(arena, manifest) if manifest else []
        if jobs:
            self._start("restored", self.spill.restore, mm, jobs)
        else:
            self._on_restored([])

    def _plan_restore(self, arena: Arena, manifest: dict) -> list[tuple]:
        """Pins first (the cold tiers read them into the pin zone), then rows
        for the spilled runs, newest first while they fit, and the runs rebuilt
        in their old order. The chunks stay pending until the read thread has
        filled them."""
        if manifest.get("superpage_bytes") != arena.sp_bytes:
            logger.warning("the spill manifest is for superpages of %s bytes; ignoring it",
                           manifest.get("superpage_bytes"))
            return []
        classes = {}
        for name, info in manifest["classes"].items():
            cfg = self.pools_cfg.get(name)
            if cfg is None:
                continue
            try:
                cls = arena.class_for(name, cfg, info["layout_key"], int(info["row_stride"]),
                                      info.get("state_groups", ()))
            except ValueError as exc:
                logger.warning("spilled class %s not restored: %s", name, exc)
                continue
            self._attach_cold(cls)
            classes[name] = cls
        now = self.clock()
        jobs, chosen = [], []
        for item in reversed(manifest["runs"]):  # newest first
            cls = classes.get(item["cls"])
            if cls is None:
                continue
            keys = [bytes.fromhex(k) for k in item["keys"]]
            mine = [k for k in dict.fromkeys(keys) if k not in cls.index]
            rows = []
            for k in mine:
                row = cls.alloc(pin=True) if k in cls.pins else None
                row = row if row is not None else cls.alloc()
                if row is None:
                    break
                rows.append(row)
            if len(rows) < len(mine):
                for row in rows:
                    cls.give_back(row)
                break  # the arena is full
            for k, row in zip(mine, rows):
                cls.add_entry(k, Entry(row=row, ready=False, writer=None, promoting=True,
                                       stored_at=now, last_use=now))
                jobs.append((cls, k, row, self.spill.key_path(cls.name, cls.layout_key, k)))
            chosen.append((cls, bytes.fromhex(item["tag"]), keys))
        for cls, tag, keys in reversed(chosen):  # oldest first, so newest ends newest
            run = arena.get_run(cls, tag)
            for k in keys:
                arena.join(run, k, cls.index[k])
        logger.info("restoring %d runs, %d chunks from the spill", len(chosen), len(jobs))
        return jobs

    def _on_restored(self, result) -> None:
        arena = self.arena
        assert arena is not None
        failed = set(result) if isinstance(result, list) else None
        restored = 0
        for cls in arena.classes.values():
            for key, entry in list(cls.index.items()):
                if not entry.promoting or (cls.cold is not None and cls.cold.reading(key)):
                    continue
                if failed is None or (cls.name, key) in failed:
                    cls.drop_entry(key, entry, quarantine=False)
                    cls.counters["spill_restore_failed"] += 1
                    continue
                cls.mark_ready(entry)
                entry.promoting = False
                if key in cls.pins:
                    arena.request_pin(cls, key, entry)
                restored += 1
        self.state = P.READY
        self.last_prepare = {**(self.last_prepare or {}), "ok": True, "restored": restored}
        logger.info("arena ready, %d chunks restored", restored)
        self._flush_backlogs()

    def _release_refusal(self, stopping: bool = False) -> str | None:
        arena = self.arena
        assert arena is not None
        if clients := self._attached():
            return f"{clients} clients attached"
        if not stopping:
            if arena.in_quarantine():
                return "the quarantine has not elapsed"
            if tokens := self._live_tokens():
                return f"open spawn tokens {sorted(tokens)}"
        for cls in arena.classes.values():
            if cls.busy():
                return f"class {cls.name} has chunks pending, loading or moving"
        if pids := self._find_mappers(self.arena_path):
            return f"the arena is mapped by pid(s) {pids}"
        return None

    def release(self, reason: str) -> dict:
        if self.state in (P.ABSENT, P.SPILLING):
            return {"ok": True, "reason": self.state.lower(), "state": self.state_reply()}
        refusal = ("the arena is being created" if self.state == P.CREATING
                   else self._release_refusal())
        if refusal is not None:
            self.last_release = {"time": time.time(), "ok": False, "reason": refusal}
            logger.info("release (%s) refused: %s", reason, refusal)
            return {"ok": False, "reason": refusal, "state": self.state_reply()}
        assert self.arena is not None
        manifest, writes = self.spill.select(self.arena)
        self.state = P.SPILLING
        self._spill_abort.clear()
        logger.info("release (%s): spilling %d runs, %d new chunk files",
                    reason, len(manifest["runs"]), len(writes))
        self._start("spilled", self.spill.write, self.arena.mm, manifest, writes,
                    self._spill_abort)
        return {"ok": True, "reason": "spilling", "state": self.state_reply()}

    def _on_spilled(self, result) -> None:
        arena = self.arena
        assert arena is not None
        if result is True:
            arena.close(unlink=True)  # in the event loop, after the manifest is durable
            self.arena = None
            self.state = P.ABSENT
            self.last_release = {"time": time.time(), "ok": True, "reason": "released"}
            logger.info("arena released")
            waiting = any(c.backlog for c in self._conns.values()) or self._live_tokens()
            if waiting and self._running:
                self.prepare("requests arrived during the spill")
            return
        self.state = P.READY
        reason = "aborted" if result is False else f"spill failed: {result}"
        self.last_release = {"time": time.time(), "ok": False, "reason": reason}
        logger.info("release ended with the arena intact: %s", reason)
        self._flush_backlogs()

    def _live_tokens(self) -> dict:
        now = self.clock()
        for token, (_, until) in list(self.spawn_tokens.items()):
            if until <= now:
                del self.spawn_tokens[token]
                logger.info("spawn token %s expired", token)
        return self.spawn_tokens

    def state_reply(self) -> dict:
        arena = self.arena
        out = {
            "state": self.state,
            "arena_bytes": self.arena_size,
            "superpage_bytes": self.superpage_bytes,
            "populated_bytes": self._populated if self.state == P.CREATING else 0,
            "used_bytes": 0,
            "generation": self.generation,
            "clients": self._attached(),
            "spawn_tokens": {t: p for t, (p, _) in self._live_tokens().items()},
            "classes": {},
            "spill": self.spill.stats(),
            "last_release": self.last_release,
            "last_prepare": self.last_prepare,
        }
        if arena is not None and self.state in (P.READY, P.SPILLING):
            out["populated_bytes"] = arena.size
            out["used_bytes"] = (arena.num_sp - len(arena.free_sp)) * arena.sp_bytes
            out["free_superpages"] = len(arena.free_sp)
            out["pin_superpages"] = arena.pin_sp
            out["counters"] = dict(arena.counters)
            for name, cls in arena.classes.items():
                out["classes"][name] = {
                    "layout_key": cls.layout_key,
                    "row_stride": cls.stride,
                    "rows_total": cls.num_rows,
                    "rows_used": len(cls.index),
                    "free_fraction": cls.available() / cls.num_rows,
                    "pinned": cls.n_pinned,
                    "runs": len(cls.runs),
                    "unrestorable_runs": cls.unrestorable_runs(),
                }
        return out

    # --- request handling ---

    def _class_for(self, args: dict) -> LayoutClass:
        name = args["pool"]
        cfg = self.pools_cfg.get(name)
        if cfg is None:
            raise ValueError(f"unknown pool {name!r}; configured: {sorted(self.pools_cfg)}")
        assert self.arena is not None
        cls = self.arena.class_for(name, cfg, args["layout_key"], int(args["row_stride"]),
                                   args.get("state_groups", ()))
        self._attach_cold(cls)
        return cls

    def _class_reply(self, cls: LayoutClass) -> dict:
        return {
            "path": self.arena_path,
            "num_chunks": cls.num_rows,
            "row_stride": cls.stride,
            "generation": self.generation,
            "arena": True,
            "class_free_fraction": cls.available() / cls.num_rows,
        }

    def _attach_cold(self, cls: LayoutClass) -> None:
        cfg = cls.cfg.get("cold")
        if not cfg or cls.cold is not None:
            return
        cls.cold = ColdTier(cls, cfg, notify=self._notify,
                            threads=None if self._io_threads else 0)
        cls.cold.scan()
        for key in sorted(cls.pins):
            if key not in cls.index and not cls.cold.start_promotion(key):
                logger.warning("class %s: pinned chunk %s is not on disk or has no room",
                               cls.name, key.hex())

    def _named_class(self, args: dict) -> LayoutClass:
        self._require_ready()
        assert self.arena is not None
        cls = self.arena.classes.get(args["pool"])
        if cls is None:
            raise ValueError(f"pool {args['pool']!r} has no class in the arena yet")
        return cls

    def _require_ready(self) -> None:
        if self.state != P.READY:
            detail = (self.last_prepare or {}).get("reason", "")
            raise RuntimeError(f"the arena is not ready (state {self.state}; {detail})")

    def handle(self, client: ClientState, op: str, args) -> object:
        if op == P.OP_STATE:
            return self.state_reply()
        if op == P.OP_PREPARE:
            self.prepare(str((args or {}).get("reason", "operator")))
            return self.state_reply()
        if op == P.OP_RELEASE:
            return self.release(str((args or {}).get("reason", "operator")))
        if op == P.OP_SPAWN_OPEN:
            self.spawn_tokens[str(args["token"])] = (
                str(args.get("pool", "")), self.clock() + float(args.get("ttl_sec", 3600)))
            self.prepare(f"spawn of {args.get('pool', '')}")
            return self.state_reply()
        if op == P.OP_SPAWN_CLOSE:
            self.spawn_tokens.pop(str(args["token"]), None)
            return self.state_reply()
        if op == P.OP_ATTACH:
            if args.get("version") != P.PROTOCOL_VERSION:
                raise ValueError(f"protocol {args.get('version')} != {P.PROTOCOL_VERSION}")
            if client.cls is not None:
                raise ValueError("already attached")
            self._require_ready()
            cls = self._class_for(args)
            client.cls = cls
            cls.clients += 1
            client.name = str(args.get("name", ""))
            logger.info("client %d (%s) attached to class %s",
                        client.client_id, client.name, cls.name)
            return {**self._class_reply(cls), "client_id": client.client_id}
        if op == P.OP_INFO:
            self._require_ready()
            return self._class_reply(self._class_for(args))
        if op == P.OP_STATS:
            if client.cls is not None:
                return client.cls.stats()
            if self.arena is None:
                return {}
            return {name: c.stats() for name, c in self.arena.classes.items()}
        # Pin operations come from operators (the router), not engines, so
        # they name the class instead of attaching to it.
        if op == P.OP_PIN:
            return self.arena.pin(self._named_class(args), args["keys"])
        if op == P.OP_PIN_RECENT:
            return self.arena.pin_recent(self._named_class(args), float(args["seconds"]),
                                   float(args.get("until_ago", 0.0)))
        if op == P.OP_UNPIN:
            return self.arena.unpin(self._named_class(args), args.get("keys"))
        if client.cls is None:
            raise ValueError("not attached")
        if op == P.OP_LOOKUP:
            return self.arena.lookup(client, args)
        if op == P.OP_LOOKUP_PREFIX:
            return self.arena.lookup(client, args, stop_at_miss=True)
        if op == P.OP_END_SESSION:
            self.arena.end_session(client)
            return None
        if op == P.OP_PREPARE_LOAD:
            return self.arena.prepare_load(client, args)
        if op == P.OP_COMPLETE_LOAD:
            self.arena.complete_load(client, args)
            return None
        if op == P.OP_PREPARE_STORE_TAGGED:
            whole = bool(args[2]) if len(args) > 2 else False
            res = self.arena.prepare_store(client, args[0], _tag(args[1]), whole)
            return None if res is None else list(res)
        if op == P.OP_MERGE_TAG:
            return self.arena.merge_tag(client, _tag(args[0]), _tag(args[1]))
        if op == P.OP_COMPLETE_STORE:
            keys, success = args
            self.arena.complete_store(client, keys, success)
            return None
        if op == P.OP_TOUCH:
            self.arena.touch(client, args)
            return None
        if op == P.OP_RELEASE_ALL:
            self.arena.release_client(client)
            return None
        raise ValueError(f"unknown op {op!r}")

    # --- event loop ---

    def _accept(self, lsock: socket.socket) -> None:
        conn, _ = lsock.accept()
        conn.setblocking(False)
        client = ClientState(client_id=self._next_client, name="")
        self._next_client += 1
        self._conns[conn] = Conn(client)
        self._sel.register(conn, selectors.EVENT_READ, self._serve)

    def _drop(self, conn: socket.socket) -> None:
        self._sel.unregister(conn)
        client = self._conns.pop(conn).client
        conn.close()
        if client.cls is not None:
            held = len(client.holds) + sum(client.loads.values()) + len(client.writes)
            self.arena.release_client(client)
            client.cls.clients -= 1
            logger.info("client %d (%s) detached, released %d references",
                        client.client_id, client.name, held)

    def _reply(self, client: ClientState, req) -> bytes:
        try:
            op, args = req
            reply = (True, self.handle(client, op, args))
        except Exception as exc:  # reported to the caller, not fatal here
            logger.warning("client %d %r failed: %s", client.client_id,
                           req[0] if isinstance(req, tuple) else req, exc)
            reply = (False, f"{type(exc).__name__}: {exc}")
        return P.encode(reply)

    def _process(self, conn: socket.socket, requests: list, force: bool = False) -> None:
        """Answer requests in order. An ATTACH or INFO before the arena is
        READY waits, with everything after it on that connection; force
        answers them now (with an error), after a refused creation."""
        st = self._conns[conn]
        out = bytearray()
        for i, req in enumerate(requests):
            waits = (not force and isinstance(req, tuple)
                     and req[0] in (P.OP_ATTACH, P.OP_INFO) and self.state != P.READY)
            if st.backlog or waits:
                st.backlog.extend(requests[i:])
                self._send(conn, out)
                if self.state in (P.ABSENT, P.SPILLING):
                    self.prepare(f"{req[0]} from client {st.client.client_id}")
                return
            out += self._reply(st.client, req)
        self._send(conn, out)

    def _send(self, conn: socket.socket, out: bytes) -> None:
        if not out or conn not in self._conns:
            return
        try:
            conn.setblocking(True)
            conn.sendall(out)
            conn.setblocking(False)
        except OSError:
            self._drop(conn)

    def _flush_backlogs(self, force: bool = False) -> None:
        for conn in list(self._conns):
            st = self._conns.get(conn)
            if st is not None and st.backlog:
                pending, st.backlog = st.backlog, []
                self._process(conn, pending, force=force)

    def _serve(self, conn: socket.socket) -> None:
        st = self._conns[conn]
        try:
            data = conn.recv(1 << 20)
        except ConnectionError:
            data = b""
        if not data:
            self._drop(conn)
            return
        st.buf += data
        try:
            requests = P.decode_frames(st.buf)
        except (P.ProtocolError, ValueError, EOFError) as exc:
            logger.warning("client %d sent a bad frame (%s); dropping it",
                           st.client.client_id, exc)
            self._drop(conn)
            return
        # Finished disk reads and moves become visible before this batch.
        self._drain()
        self._process(conn, requests)

    def remove_stale_regions(self) -> int:
        """Delete the region files of the per-model pools the arena replaces
        (and of the first deployment's per-generation names). An engine still
        mapping one keeps its pages until it exits."""
        keep = os.path.basename(self.arena_path)
        removed = 0
        for name in os.listdir(self.shm_dir):
            if name.startswith("vllm_kvpool_") and name.endswith(".mmap") and name != keep:
                path = os.path.join(self.shm_dir, name)
                size = os.path.getsize(path)
                os.unlink(path)
                removed += 1
                logger.info("removed stale region %s (%.2f GiB)", path, size / (1 << 30))
        return removed

    def serve_forever(self, log_interval: float = 60.0) -> None:
        self.remove_stale_regions()
        self.spill.load()
        self.adopt_or_clear()
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
        logger.info("kvpool generation %s listening on %s; arena %s of %d bytes in "
                    "superpages of %d; classes %s; state %s", self.generation,
                    self.socket_path, self.arena_path, self.arena_size,
                    self.superpage_bytes, sorted(self.pools_cfg), self.state)
        next_log = time.monotonic() + log_interval
        try:
            while self._running:
                for key, _ in self._sel.select(timeout=1.0):
                    key.data(key.fileobj)
                self._drain()
                self._live_tokens()
                if self.arena is not None and self.state == P.READY:
                    self.arena.maintain()
                if time.monotonic() >= next_log:
                    next_log = time.monotonic() + log_interval
                    logger.info("state %s", json.dumps(self.state_reply(), sort_keys=True))
        finally:
            self._shutdown(lsock)

    def _shutdown(self, lsock: socket.socket) -> None:
        while self._life_thread is not None:
            self._life_thread.join()
            self._drain()
        if self.state == P.READY and self.arena is not None:
            refusal = self._release_refusal(stopping=True)
            if refusal is None:
                manifest, writes = self.spill.select(self.arena)
                logger.info("stopping: spilling %d runs", len(manifest["runs"]))
                if self.spill.write(self.arena.mm, manifest, writes, threading.Event()):
                    self.arena.close(unlink=True)
                    self.arena = None
            else:
                logger.info("stopping with the arena kept: %s", refusal)
        for conn in list(self._conns):
            self._drop(conn)
        self._sel.close()
        lsock.close()
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass
        if self.arena is not None:
            self.arena.close()  # the file stays: engines may map it, the next server adopts it
        self._wake_r.close()
        self._wake_w.close()

    def stop(self, *_args) -> None:
        self._running = False


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True,
                    help='JSON: {"socket": path, "arena": {...}, "pools": {name: {...}}}')
    args = ap.parse_args(argv)
    with open(args.config) as f:
        cfg = json.load(f)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    server = PoolServer(cfg["socket"], cfg["pools"], cfg["arena"],
                        shm_dir=cfg.get("shm_dir", "/dev/shm"),
                        quarantine_sec=float(cfg.get("quarantine_sec", 30.0)),
                        socket_group=cfg.get("socket_group"))
    signal.signal(signal.SIGTERM, server.stop)
    signal.signal(signal.SIGINT, server.stop)
    server.serve_forever(float(cfg.get("log_interval_sec", 60.0)))


if __name__ == "__main__":
    main()
