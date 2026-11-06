# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ctypes wrapper of libl0ipc.so (l0ipc.c): Level Zero IPC for the KV offload
copy helper (copy_helper.py). The engine side exports a device allocation
(the library is preloaded with L0IPC_ENGINE=1 so a tracer can learn its
context); the helper opens it in its own context and copies to and from it on
an immediate command list of the copy engine, signalling events."""

import ctypes
import os
from typing import NamedTuple

LIB_PATH = os.environ.get("VLLM_L0IPC_LIB", "/opt/b70/libl0ipc.so")
HANDLE_BYTES = 64


class Export(NamedTuple):
    """An exported allocation: the descriptor to pass (SCM_RIGHTS), the raw
    Level Zero handle, the pointer's offset in the allocation and its size."""

    fd: int
    raw: bytes
    offset: int
    size: int


class L0IpcError(RuntimeError):
    pass


def _lib() -> ctypes.CDLL:
    lib = ctypes.CDLL(LIB_PATH)
    lib.l0ipc_host_alloc.restype = ctypes.c_void_p
    lib.l0ipc_host_alloc.argtypes = [ctypes.c_uint64]
    return lib


def export(ptr: int) -> Export:
    """Engine side: export the device allocation holding ``ptr``."""
    lib = _lib()
    fd, off, size = ctypes.c_int(), ctypes.c_uint64(), ctypes.c_uint64()
    raw = (ctypes.c_ubyte * HANDLE_BYTES)()
    r = lib.l0ipc_export(
        ctypes.c_void_p(ptr), ctypes.byref(fd), ctypes.byref(off), ctypes.byref(size), raw
    )
    if r:
        raise L0IpcError(
            f"l0ipc_export {r:#x}: is libl0ipc.so preloaded with L0IPC_ENGINE=1 and "
            "ZE_ENABLE_TRACING_LAYER=1, and the memory not in an expandable segment?"
        )
    return Export(fd.value, bytes(raw), off.value, size.value)


class Helper:
    """Helper side: one context on visible device ``index``, one immediate
    command list on the copy engine and ``num_events`` events."""

    def __init__(self, index: int, num_events: int) -> None:
        self.lib = _lib()
        if r := self.lib.l0ipc_open(index):
            raise L0IpcError(f"l0ipc_open {r:#x}")
        if r := self.lib.l0ipc_events(num_events):
            raise L0IpcError(f"l0ipc_events {r:#x}")

    def open(self, fd: int, raw: bytes, offset: int) -> int:
        """The device address, in this process, of the exported pointer."""
        out = ctypes.c_void_p()
        handle = (ctypes.c_ubyte * HANDLE_BYTES)(*raw)
        r = self.lib.l0ipc_import(2, fd, handle, ctypes.c_uint64(offset), ctypes.byref(out))
        if r:
            raise L0IpcError(f"l0ipc_import {r:#x}")
        return int(out.value)

    def host_alloc(self, size: int) -> int:
        ptr = self.lib.l0ipc_host_alloc(size)
        if not ptr:
            raise L0IpcError(f"l0ipc_host_alloc of {size} bytes")
        return int(ptr)

    def copy_async(self, dst, src, sizes, event: int) -> None:
        """Append copies (sequences of addresses and sizes) that signal
        ``event`` once all are done."""
        n = len(sizes)
        if not n:
            return
        arrays = (
            (ctypes.c_void_p * n)(*dst),
            (ctypes.c_void_p * n)(*src),
            (ctypes.c_uint64 * n)(*sizes),
        )
        if r := self.lib.l0ipc_copy_async(*arrays, n, event):
            raise L0IpcError(f"l0ipc_copy_async {r:#x}")

    def wait(self, event: int) -> None:
        if r := self.lib.l0ipc_event_wait(event):
            raise L0IpcError(f"l0ipc_event_wait {r:#x}")
