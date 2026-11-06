# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""KV offload copy helper (VLLM_KV_OFFLOAD_COPY_HELPER=1, XPU, shared region).

On a B70 every queue of a process sends its copies through one in-order ring
on the one user copy engine, behind the forward's per-step output copies,
which wait on compute: each copy a lane waits for moves once per forward step
(bench/copy-hol: 2.9 GB/s in 64 MB pieces in process; restores ran at 0.66
GB/s under agent load). A helper process has its own context on the copy
engine and copies at its full rate (bench/copy-ipc: 13.8 GB/s into an engine's
tensor during per-step compute).

One helper per offloading worker (one GPU rank). It opens the worker's KV
cache through Level Zero IPC (l0ipc), maps the shared offload region, and runs
the worker's own staged copy code (_staged_items, _staged_batches,
_staged_cpu, _staged_dma) on its own pinned slots, one thread per lane. The
worker keeps every fence and the lane order: a lane's copy closure hands its
job to the helper and waits for the reply."""

import contextlib
import ctypes
import mmap
import os
import pickle
import socket
import struct
import subprocess
import sys
import threading
import time
from collections import deque
from typing import Any

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

ENABLED = os.environ.get("VLLM_KV_OFFLOAD_COPY_HELPER", "0") == "1"
# store, load, and the settled (band) store lane: the worker's three copy lanes.
LANES = ("store", "load", "band")
# Slots per lane in the helper: its copies are not paced by the forward, so a
# short ring keeps the copy engine busy (each slot is pinned host memory).
HELPER_RING = int(os.environ.get("VLLM_KV_OFFLOAD_HELPER_RING", "3"))
_HDR = struct.Struct("<Q")


@contextlib.contextmanager
def plain_kv_allocation():
    """Allocate the KV cache outside expandable segments when the helper is
    on: memory mapped into an expandable segment cannot be exported."""
    if not (ENABLED and torch.xpu.is_available()):
        yield
        return
    get = torch._C._accelerator_getAllocatorSettings
    set_ = torch._C._accelerator_setAllocatorSettings
    expandable = "expandable_segments:True" in get()
    if expandable:
        set_("expandable_segments:False")
    try:
        yield
    finally:
        if expandable:
            set_("expandable_segments:True")


def _send(sock: socket.socket, obj: Any, fds: list[int] | None = None) -> None:
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    if fds:
        socket.send_fds(sock, [_HDR.pack(len(data))], fds)
        sock.sendall(data)
    else:
        sock.sendall(_HDR.pack(len(data)) + data)


def _recv_exact(sock: socket.socket, n: int, head: bytes = b"") -> bytes:
    buf = bytearray(head)
    while len(buf) < n:
        part = sock.recv(n - len(buf))
        if not part:
            raise ConnectionError("copy helper connection closed")
        buf += part
    return bytes(buf)


def _recv(sock: socket.socket, max_fds: int = 0) -> tuple[Any, list[int]]:
    fds: list[int] = []
    if max_fds:
        head, fds, _, _ = socket.recv_fds(sock, _HDR.size, max_fds)
        if not head:
            raise ConnectionError("copy helper connection closed")
        head = _recv_exact(sock, _HDR.size, head)
    else:
        head = _recv_exact(sock, _HDR.size)
    (n,) = _HDR.unpack(head)
    return pickle.loads(_recv_exact(sock, n)), fds


class CopyHelper:
    """Worker side: starts the helper and runs jobs on it, one socket per lane
    (each lane's thread waits for its own replies)."""

    def __init__(
        self,
        device_index: int,
        gpu_tensors: list[torch.Tensor],
        cpu_tensors: list[torch.Tensor],
        region_path: str,
        region_ptr: int,
        region_size: int,
        staged_refs: dict[bool, Any],
    ) -> None:
        from vllm.v1.kv_offload.cpu import l0ipc

        storages = {t.untyped_storage().data_ptr() for t in gpu_tensors}
        exports = {p: l0ipc.export(p) for p in storages}
        gpu = []
        for t in gpu_tensors:
            e = exports[t.untyped_storage().data_ptr()]
            gpu.append((t.untyped_storage().data_ptr(), e.offset + t.data_ptr()
                        - t.untyped_storage().data_ptr(), t.stride(0)))
        cpu = [
            (t.untyped_storage().data_ptr() - region_ptr, t.storage_offset(),
             tuple(t.shape), tuple(t.stride()))
            for t in cpu_tensors
        ]
        self._socks: dict[str, socket.socket] = {}
        children = []
        for lane in LANES:
            a, b = socket.socketpair()
            self._socks[lane] = a
            children.append(b)
        env = {k: v for k, v in os.environ.items() if k != "L0IPC_ENGINE"}
        # Only this worker's card: a context on every card the worker sees
        # costs host memory (GTT) for nothing.
        mask = [m for m in os.environ.get("ZE_AFFINITY_MASK", "").split(",") if m]
        if mask:
            env["ZE_AFFINITY_MASK"] = mask[device_index]
            device_index = 0
        self._proc = subprocess.Popen(
            [sys.executable, "-m", "vllm.v1.kv_offload.cpu.copy_helper",
             *[str(b.fileno()) for b in children]],
            pass_fds=[b.fileno() for b in children],
            env=env,
        )
        for b in children:
            b.close()
        ptrs = list(exports)
        init = {
            "device": device_index,
            "region": (region_path, region_size),
            "allocs": [(exports[p].raw, exports[p].size) for p in ptrs],
            "gpu": [(ptrs.index(base), off, stride) for base, off, stride in gpu],
            "cpu": cpu,
            "staged_refs": staged_refs,
        }
        _send(self._socks[LANES[0]], init, [exports[p].fd for p in ptrs])
        for p in ptrs:
            os.close(exports[p].fd)
        reply, _ = _recv(self._socks[LANES[0]])
        if reply != "ready":
            raise RuntimeError(f"copy helper failed to start: {reply}")
        self.failed = False
        logger.info("KV offload copy helper %d started (ZE_AFFINITY_MASK=%s, %d KV allocation(s))",
                    self._proc.pid, env.get("ZE_AFFINITY_MASK", "unset"), len(ptrs))

    def run(self, gpu_to_cpu: bool, settled: bool, src_blocks: np.ndarray,
            dst_blocks: np.ndarray, group_sizes) -> None:
        lane = "band" if settled else ("store" if gpu_to_cpu else "load")
        sock = self._socks[lane]
        _send(sock, (gpu_to_cpu, np.asarray(src_blocks), np.asarray(dst_blocks), list(group_sizes)))
        reply, _ = _recv(sock)
        if reply != "ok":
            raise RuntimeError(f"copy helper job failed: {reply}")

    def close(self) -> None:
        for sock in self._socks.values():
            with contextlib.suppress(OSError):
                sock.close()
        with contextlib.suppress(Exception):
            self._proc.wait(timeout=10)
        if self._proc.poll() is None:
            self._proc.terminate()


# --- the helper process ---


class _Device:
    """A device tensor of the worker, as the staged copy code reads it."""

    def __init__(self, address: int, stride: int) -> None:
        self._address, self._stride = address, stride

    def data_ptr(self) -> int:
        return self._address

    def stride(self, dim: int) -> int:
        assert dim == 0
        return self._stride


class _Lane:
    """One lane: a handler built like the staged-copy tests build theirs, its
    own slots and events, copies on the shared list under a lock."""

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


def _serve(lane: _Lane, sock: socket.socket) -> None:
    while True:
        try:
            job, _ = _recv(sock)
        except ConnectionError:
            return
        try:
            gpu_to_cpu, src, dst, group_sizes = job
            assert gpu_to_cpu == lane.h.gpu_to_cpu
            lane.run(src, dst, group_sizes)
            _send(sock, "ok")
        except Exception as e:  # noqa: BLE001
            _send(sock, f"{type(e).__name__}: {e}")


def main(fds: list[int]) -> None:
    from vllm.v1.kv_offload.cpu import l0ipc
    from vllm.v1.kv_offload.cpu.gpu_worker import STAGED_SLOT_BYTES

    socks = [socket.socket(fileno=fd) for fd in fds]
    init, alloc_fds = _recv(socks[0], max_fds=16)
    try:
        l0 = l0ipc.Helper(init["device"], len(LANES) * HELPER_RING)
        bases = [l0.open(fd, raw, 0) for fd, (raw, _size) in zip(alloc_fds, init["allocs"])]
        gpu = [_Device(bases[i] + off, stride) for i, off, stride in init["gpu"]]
        path, size = init["region"]
        with open(path, "r+b") as f:
            region = mmap.mmap(f.fileno(), size)
        flat = torch.frombuffer(region, dtype=torch.int8)
        # The worker's views, rebuilt over this process's mapping of the region:
        # the same shape and strides, offset by where their storage starts.
        cpu = [flat.as_strided(shape, stride, base + storage_offset)
               for base, storage_offset, shape, stride in init["cpu"]]
        lock = threading.Lock()
        lanes = {
            name: _Lane(name != "load", init["staged_refs"][name != "load"], gpu, cpu, l0, lock,
                        i * HELPER_RING, HELPER_RING, STAGED_SLOT_BYTES)
            for i, name in enumerate(LANES)
        }
    except Exception as e:  # noqa: BLE001
        _send(socks[0], f"{type(e).__name__}: {e}")
        return
    _send(socks[0], "ready")
    threads = [threading.Thread(target=_serve, args=(lanes[name], sock), daemon=True)
               for name, sock in zip(LANES, socks)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()


if __name__ == "__main__":
    main([int(a) for a in sys.argv[1:]])
