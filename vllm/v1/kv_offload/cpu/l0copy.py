# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""ctypes wrapper of libl0copy.so (l0copy.c): an immediate command list of our
own on the copy engine, in the engine's Level Zero context, for the KV offload
copy lanes (copy_list.py). The library must be preloaded with
ZE_ENABLE_TRACING_LAYER=1 so its tracer learns that context."""

import ctypes
import os

LIB_PATH = os.environ.get("VLLM_L0COPY_LIB", "/opt/b70/libl0copy.so")


class L0CopyError(RuntimeError):
    pass


class List:
    """The list on the copy engine of the device holding ``device_ptr``, with
    ``num_events`` events."""

    def __init__(self, device_ptr: int, num_events: int) -> None:
        self.lib = ctypes.CDLL(LIB_PATH)
        self.lib.l0copy_host_alloc.restype = ctypes.c_void_p
        self.lib.l0copy_host_alloc.argtypes = [ctypes.c_uint64]
        if r := self.lib.l0copy_open(ctypes.c_void_p(device_ptr), num_events):
            raise L0CopyError(
                f"l0copy_open {r:#x}: is libl0copy.so preloaded with ZE_ENABLE_TRACING_LAYER=1?"
            )

    def host_alloc(self, size: int) -> int:
        ptr = self.lib.l0copy_host_alloc(size)
        if not ptr:
            raise L0CopyError(f"l0copy_host_alloc of {size} bytes")
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
        if r := self.lib.l0copy_copy_async(*arrays, n, event):
            raise L0CopyError(f"l0copy_copy_async {r:#x}")

    def wait(self, event: int) -> None:
        if r := self.lib.l0copy_wait(event):
            raise L0CopyError(f"l0copy_wait {r:#x}")
