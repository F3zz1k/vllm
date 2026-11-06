# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The copy helper's lane (copy_helper._Lane), CPU only: Level Zero is a fake
whose copies are memmoves, the device side is a CPU tensor. A helper store and
load must move exactly what the staged in-process path moves (the oracle of
test_staged_copy), through its own slot ring and events."""

import ctypes
import threading

import numpy as np
import pytest
import torch

from tests.v1.kv_offload.cpu.test_staged_copy import _oracle, _shard
from vllm.v1.kv_offload.base import CanonicalKVCacheRef
from vllm.v1.kv_offload.cpu import copy_helper, gpu_worker
from vllm.v1.kv_offload.cpu.gpu_worker import _staged_ref

_HEADS, _HEAD_BYTES, _TOKENS = 4, 64, 4
_CANON = 2 * _TOKENS * _HEADS * _HEAD_BYTES


class _FakeL0:
    """Copies happen at once; events record what was signalled and waited."""

    def __init__(self) -> None:
        self.buffers = []
        self.signalled: list[int] = []
        self.waited: list[int] = []

    def host_alloc(self, size: int) -> int:
        buf = (ctypes.c_int8 * size)()
        self.buffers.append(buf)
        return ctypes.addressof(buf)

    def copy_async(self, dst, src, sizes, event: int) -> None:
        for d, s, n in zip(dst, src, sizes):
            ctypes.memmove(d, s, n)
        self.signalled.append(event)

    def wait(self, event: int) -> None:
        self.waited.append(event)


def _lane(gpu_to_cpu: bool, device: torch.Tensor, cpu: torch.Tensor, mapping, l0, ring=2):
    refs = [[_staged_ref(CanonicalKVCacheRef(0, mapping.local_page_size_bytes, mapping))]]
    dev = copy_helper._Device(device.data_ptr(), device.stride(0))
    return copy_helper._Lane(gpu_to_cpu, refs, [dev], [cpu], l0, threading.Lock(), 10, ring, 1500)


@pytest.mark.parametrize("tp,ring", [(2, 2), (4, 3), (2, 1)])
def test_helper_store_and_load_match_the_staged_oracle(tp, ring, monkeypatch):
    monkeypatch.setattr(gpu_worker, "STAGED_SLOT_BYTES", 1500)  # several batches per job
    torch.manual_seed(0)
    num_blocks, num_rows = 5, 8
    local_bytes = 2 * _TOKENS * _HEADS // tp * _HEAD_BYTES
    rows = [6, 1, 3, 0, 7]
    canon = torch.zeros(num_rows, _CANON, dtype=torch.int8)
    expected = np.zeros((num_rows, _CANON), dtype=np.int8)
    devices = []
    for rank in range(tp):
        device = torch.randint(-128, 128, (num_blocks + 2, local_bytes), dtype=torch.int8)
        devices.append(device)
        l0 = _FakeL0()
        _lane(True, device, canon, _shard(tp, rank), l0, ring).run(
            np.arange(2, 2 + num_blocks), np.asarray(rows), (num_blocks,)
        )
        # every signalled event is waited for before its slot is scattered
        assert sorted(l0.signalled) == sorted(l0.waited) and len(l0.signalled) > 1
        _oracle(device.numpy()[2:], _shard(tp, rank), expected, rows)
    assert np.array_equal(canon.numpy(), expected)
    for rank in range(tp):
        back = torch.zeros(num_blocks + 3, local_bytes, dtype=torch.int8)
        l0 = _FakeL0()
        _lane(False, back, canon, _shard(tp, rank), l0, ring).run(
            np.asarray(rows), np.arange(3, 8), (num_blocks,)
        )
        assert torch.equal(back[3:8], devices[rank][2:7])
        assert not back[:3].any()
        assert sorted(l0.signalled) == sorted(l0.waited)  # the job ends with nothing in flight


def test_plain_kv_allocation_is_a_no_op_when_off(monkeypatch):
    monkeypatch.setattr(copy_helper, "ENABLED", False)
    with copy_helper.plain_kv_allocation():
        t = torch.zeros(4)
    assert t.numel() == 4


def test_framing_round_trips_objects_and_descriptors():
    import os
    import socket

    a, b = socket.socketpair()
    r, w = os.pipe()
    try:
        copy_helper._send(a, {"x": np.arange(3)}, [r])
        obj, fds = copy_helper._recv(b, max_fds=4)
        assert np.array_equal(obj["x"], np.arange(3)) and len(fds) == 1
        copy_helper._send(a, ("ok", 1))
        assert copy_helper._recv(b)[0] == ("ok", 1)
        for fd in fds:
            os.close(fd)
    finally:
        os.close(r)
        os.close(w)
        a.close()
        b.close()
