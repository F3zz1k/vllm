# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B70-PLE-RAM: the rank's INT8 row-scale PLE rows, resident in host RAM (host side).

Port of Lumnus 0008 (``B70_PLE_INT8``, v0.30) onto the 0.32 host-prepared PLE
path. Host-only (numpy / os / threads) and free of vLLM imports, so the tests
load it standalone, as ``ple_nvme``.

PLE is either RAM native or on SSD. This is the RAM-native side: the
rank's owned row range [tp_start, tp_end) of the table file is read once at
boot into pinned slabs and addressed directly by row id. No row cache, no
slot map, no reader on the step path.

Slabs: the torch caching host allocator rounds every pinned request up to the
next power of two (``CachingHostAllocator.h``, ``pinned_max_round_threshold``
defaults to SIZE_MAX), so one 12.2 GiB request would pin 16 GiB; and torch
2.13+xpu returned pageable memory above 2^34 B (Lumnus 0002). ``plan_slabs``
cuts the range into slabs that each fill a power of two to within one row
(at most ``max_bytes``, default 8 GiB), plus one tail slab under 1 MiB, so the
pinned footprint is the table's own size.
"""

from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np

try:
    from .ple_nvme import preload_range
except ImportError:  # loaded standalone by the tests
    from ple_nvme import preload_range

SCALE_BYTES = 4
MAX_SLAB_BYTES = 1 << 33
TAIL_BYTES = 1 << 20


def pow2_ceil(n: int) -> int:
    return 1 << max(0, int(n) - 1).bit_length() if n > 0 else 0


def plan_slabs(rows: int, row_bytes: int, max_bytes: int = MAX_SLAB_BYTES,
               tail_bytes: int = TAIL_BYTES) -> list[int]:
    """Rows per slab: each slab is the largest power of two <= min(left, max_bytes)
    worth of whole rows; whatever is left under ``tail_bytes`` is one last slab."""
    if rows < 0 or row_bytes <= 0 or max_bytes < row_bytes:
        raise ValueError(f"plan_slabs: rows={rows} row_bytes={row_bytes} max_bytes={max_bytes}")
    out: list[int] = []
    left = int(rows)
    while left > 0:
        nbytes = left * row_bytes
        if nbytes <= tail_bytes:
            out.append(left)
            break
        cap = 1 << (min(nbytes, max_bytes).bit_length() - 1)
        take = min(left, cap // row_bytes)
        out.append(take)
        left -= take
    return out


def pinned_bytes(slab_rows: list[int], row_bytes: int) -> int:
    """Bytes the power-of-two caching host allocator takes for these slabs."""
    return sum(pow2_ceil(r * row_bytes) for r in slab_rows)


class PleRamTable:
    """The rank's owned rows [tp_start, tp_end) of a packed INT8 table, in slabs.

    ``alloc(rows, row_bytes) -> uint8 ndarray [rows, row_bytes]`` makes one slab
    (the embedding passes pinned torch storage; tests pass numpy).
    """

    def __init__(self, tp_start: int, tp_end: int, row_bytes: int, *,
                 alloc: Callable[[int, int], np.ndarray] | None = None,
                 max_slab_bytes: int = MAX_SLAB_BYTES, tail_bytes: int = TAIL_BYTES) -> None:
        if not 0 <= tp_start <= tp_end:
            raise ValueError(f"PLE RAM: bad row range [{tp_start}, {tp_end})")
        self.tp_start = int(tp_start)
        self.tp_end = int(tp_end)
        self.row_bytes = int(row_bytes)
        self.num_rows = self.tp_end - self.tp_start
        self.slab_rows = plan_slabs(self.num_rows, self.row_bytes, max_slab_bytes, tail_bytes)
        alloc = alloc or (lambda r, b: np.empty((r, b), dtype=np.uint8))
        self.slabs: list[np.ndarray] = []
        for r in self.slab_rows:
            slab = alloc(r, self.row_bytes)
            if (slab.dtype != np.uint8 or slab.shape != (r, self.row_bytes)
                    or not slab.flags.c_contiguous):
                raise ValueError("PLE RAM: slab allocator returned the wrong array")
            self.slabs.append(slab)
        # Local row starts of each slab (+ the end), for the slab lookup.
        self.bounds = np.concatenate(
            [[0], np.cumsum(self.slab_rows, dtype=np.int64)]).astype(np.int64)
        self.loaded = False

    @property
    def nbytes(self) -> int:
        return self.num_rows * self.row_bytes

    def load(self, path: str, data_start: int, *, direct: bool = True, threads: int = 4,
             chunk_bytes: int = 64 << 20) -> dict:
        """Read file bytes of rows [tp_start, tp_end) into the slabs (sequential
        page-aligned chunks, O_DIRECT when the filesystem takes it)."""
        t0 = time.perf_counter()
        used_direct = True
        for i, slab in enumerate(self.slabs):
            first = self.tp_start + int(self.bounds[i])
            used_direct &= preload_range(path, int(data_start) + first * self.row_bytes, slab,
                                         direct=direct, threads=threads, chunk_bytes=chunk_bytes)
        seconds = time.perf_counter() - t0
        self.loaded = True
        return {"rows": self.num_rows, "bytes": self.nbytes, "seconds": seconds,
                "gbps": self.nbytes / max(seconds, 1e-9) / 1e9, "direct": bool(used_direct),
                "slabs": len(self.slabs)}

    def _locate(self, local: np.ndarray) -> np.ndarray:
        return np.searchsorted(self.bounds, local, side="right") - 1

    def rows(self, global_rows: np.ndarray) -> np.ndarray:
        """uint8 [n, row_bytes] copies of owned rows (global ids)."""
        g = np.asarray(global_rows, dtype=np.int64).reshape(-1)
        if g.size and (g.min() < self.tp_start or g.max() >= self.tp_end):
            raise IndexError(f"PLE RAM: rows outside [{self.tp_start}, {self.tp_end})")
        out = np.empty((g.size, self.row_bytes), dtype=np.uint8)
        self._copy(out, np.arange(g.size), g - self.tp_start)
        return out

    def _copy(self, out: np.ndarray, dst: np.ndarray | tuple, local: np.ndarray) -> None:
        if len(self.slabs) == 1:
            out[dst] = self.slabs[0][local]
            return
        which = self._locate(local)
        for s in np.unique(which):
            sel = which == s
            d = tuple(x[sel] for x in dst) if isinstance(dst, tuple) else dst[sel]
            out[d] = self.slabs[s][local[sel] - self.bounds[s]]

    def gather_into(self, stage: np.ndarray, ids: np.ndarray, heads: np.ndarray) -> int:
        """stage[t, heads[c]] = row(ids[t, c]) for every owned id; others untouched.

        ``stage`` uint8 [T', H, row_bytes]; ``ids`` int64 [T, len(heads)] global
        row ids of this rank's heads. Returns the number of rows written.
        """
        ids = np.asarray(ids, dtype=np.int64)
        mask = (ids >= self.tp_start) & (ids < self.tp_end)
        tok, col = np.nonzero(mask)
        if tok.size == 0:
            return 0
        local = ids[tok, col] - self.tp_start
        self._copy(stage, (tok, np.asarray(heads, dtype=np.int64)[col]), local)
        return int(tok.size)

    def scan_scales(self, logical_dim: int, step: int = 1 << 22) -> None:
        """Every owned row's float32 scale is finite and >= 0 (0008's full slab scan)."""
        for i, slab in enumerate(self.slabs):
            for start in range(0, slab.shape[0], step):
                part = slab[start:start + step]
                sc = np.ndarray((part.shape[0],), dtype="<f4", buffer=part,
                                offset=logical_dim, strides=(self.row_bytes,))
                if not bool(np.isfinite(sc).all()) or bool((sc < 0).any()):
                    bad = start + int(np.nonzero(~(np.isfinite(sc) & (sc >= 0)))[0][0])
                    raise ValueError(
                        f"INT8 PLE: non-finite or negative row scale at row "
                        f"{self.tp_start + int(self.bounds[i]) + bad}")
