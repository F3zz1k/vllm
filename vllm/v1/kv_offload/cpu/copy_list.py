# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV offload copies on a copy list of our own (VLLM_KV_OFFLOAD_COPY_LIST=1,
XPU, staged copies).

On a B70 the UR adapter sends every queue's copies in a process through one
in-order ring on the copy engine, behind the forward's per-step output copies,
which wait on compute: each copy a lane waits for moves once per forward step
(bench/copy-hol: 2.9 GB/s in 64 MB pieces; restores ran at 0.66 GB/s under
agent load). An immediate command list created directly in Level Zero, in the
engine's own context, has a ring of its own and copies at the engine's rate
while the forward runs (bench/copy-ipc copyhol2: 11.2 GB/s, compute unchanged).

Each lane (store, load, settled store) runs the worker's own staged copy code
(_staged_items, _staged_batches, _staged_cpu, _staged_dma) with its own pinned
slots and events; only the device copies go to the list. The worker keeps
every fence and the lane order: a lane's copy closure runs its job here."""

import ctypes
import os
import threading
from collections import deque

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

ENABLED = os.environ.get("VLLM_KV_OFFLOAD_COPY_LIST", "0") == "1"
# store, load, and the settled (band) store lane: the worker's three copy lanes.
LANES = ("store", "load", "band")
# Slots per lane: these copies are not paced by the forward, so a short ring
# keeps the copy engine busy (each slot is pinned host memory).
RING = int(os.environ.get("VLLM_KV_OFFLOAD_COPY_LIST_RING", "3"))


class CopyList:
    """The worker's three lanes on one list (appends under a lock, each lane
    waits on its own events)."""

    def __init__(self, gpu_tensors: list[torch.Tensor], cpu_tensors: list[torch.Tensor],
                 staged_refs: dict[bool, object]) -> None:
        from vllm.v1.kv_offload.cpu import l0copy
        from vllm.v1.kv_offload.cpu.gpu_worker import STAGED_SLOT_BYTES

        l0 = l0copy.List(gpu_tensors[0].data_ptr(), len(LANES) * RING)
        lock = threading.Lock()
        self._lanes = {
            name: _Lane(name != "load", staged_refs[name != "load"], gpu_tensors, cpu_tensors,
                        l0, lock, i * RING, RING, STAGED_SLOT_BYTES)
            for i, name in enumerate(LANES)
        }
        self.failed = False
        logger.info("KV offload copies on an own copy list (%d slots of %d MiB per lane)",
                    RING, STAGED_SLOT_BYTES >> 20)

    def run(self, gpu_to_cpu: bool, settled: bool, src_blocks: np.ndarray,
            dst_blocks: np.ndarray, group_sizes) -> None:
        lane = self._lanes["band" if settled else ("store" if gpu_to_cpu else "load")]
        assert lane.h.gpu_to_cpu == gpu_to_cpu
        lane.run(src_blocks, dst_blocks, group_sizes)


class _Lane:
    """One lane: a handler built like the staged-copy tests build theirs, its
    own slots and events, copies on the shared list under a lock. The device
    tensors are anything with data_ptr() and stride(0)."""

    def __init__(self, gpu_to_cpu: bool, refs, gpu, cpu, l0, lock, first_event: int,
                 ring: int, slot_bytes: int) -> None:
        from vllm.v1.kv_offload.cpu.gpu_worker import SingleDirectionOffloadingHandler

        h = object.__new__(SingleDirectionOffloadingHandler)
        h.gpu_to_cpu = gpu_to_cpu
        h._staged_refs = refs
        h.src_tensors, h.dst_tensors = (gpu, cpu) if gpu_to_cpu else (cpu, gpu)
        h._swap_blocks_batch = self._swap
        self.h, self.l0, self.lock = h, l0, lock
        self.events = list(range(first_event, first_event + ring))
        self.slot_bytes = slot_bytes
        self.slots: list[torch.Tensor | None] = [None] * ring
        self.armed = [False] * ring
        self._event = -1

    def _slot(self, idx: int, nbytes: int) -> torch.Tensor:
        slot = self.slots[idx]
        if slot is None or slot.numel() < nbytes:
            if self.armed[idx]:
                self._wait(idx)
            size = max(nbytes, self.slot_bytes)
            ptr = self.l0.host_alloc(size)  # a larger page than a slot is rare: the old slot is kept
            buf = (ctypes.c_int8 * size).from_address(ptr)
            slot = self.slots[idx] = torch.frombuffer(buf, dtype=torch.int8)
        return slot

    def _swap(self, src, dst, sizes, **_) -> None:
        with self.lock:
            self.l0.copy_async(dst.tolist(), src.tolist(), sizes.tolist(), self._event)

    def _dma(self, batch, idx: int, slot: torch.Tensor) -> None:
        self._event = self.events[idx]
        self.h._staged_dma(batch, slot)
        self.armed[idx] = True

    def _wait(self, idx: int) -> None:
        self.l0.wait(self.events[idx])
        self.armed[idx] = False

    def run(self, src_blocks, dst_blocks, group_sizes) -> None:
        from vllm.v1.kv_offload.cpu.gpu_worker import _batch_bytes

        h = self.h
        batches = h._staged_batches(h._staged_items(src_blocks, dst_blocks, group_sizes))
        ring = len(self.events)
        if h.gpu_to_cpu:
            pending: deque = deque()
            last = len(batches) - 1
            for k, batch in enumerate(batches):
                idx = k % ring
                slot = self._slot(idx, _batch_bytes(batch))
                self._dma(batch, idx, slot)
                pending.append((batch, idx, slot))
                while len(pending) >= ring or (pending and k == last):
                    done_batch, done_idx, done_slot = pending.popleft()
                    self._wait(done_idx)
                    h._staged_cpu(done_batch, done_slot)
        else:
            for k, batch in enumerate(batches):
                idx = k % ring
                slot = self._slot(idx, _batch_bytes(batch))
                if self.armed[idx]:
                    self._wait(idx)
                h._staged_cpu(batch, slot)
                self._dma(batch, idx, slot)
            for idx in range(ring):
                if self.armed[idx]:
                    self._wait(idx)
