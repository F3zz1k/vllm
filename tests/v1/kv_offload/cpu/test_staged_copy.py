# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Staged canonical copies (gpu_worker._transfer_staged), CPU only: the device
side of each page is a CPU tensor and the per-page DMA is a memmove, so the
staging, scatter, gather and writer rotation are checked against the same
numpy oracle the descriptor path is tested with."""

import ctypes

import numpy as np
import pytest
import torch

from vllm.v1.kv_offload.base import CanonicalKVCacheRef, CanonicalPageMapping, CopyRun
from vllm.v1.kv_offload.cpu import gpu_worker
from vllm.v1.kv_offload.cpu.gpu_worker import (
    SingleDirectionOffloadingHandler,
    _staged_ref,
)

_HEADS, _HEAD_BYTES, _TOKENS = 4, 64, 4
_CANON = 2 * _TOKENS * _HEADS * _HEAD_BYTES


def _shard(tp: int, rank: int) -> CanonicalPageMapping:
    local = _HEADS // tp * _HEAD_BYTES
    row = _HEADS * _HEAD_BYTES
    runs = (
        CopyRun(0, rank * local, local, _TOKENS, local, row),
        CopyRun(_TOKENS * local, _TOKENS * row + rank * local, local, _TOKENS, local, row),
    )
    return CanonicalPageMapping(_CANON, 2 * _TOKENS * local, runs, 1, 0, True)


def _memmove(src: torch.Tensor, dst: torch.Tensor, sizes: torch.Tensor, **_) -> None:
    for s, d, n in zip(src.tolist(), dst.tolist(), sizes.tolist()):
        ctypes.memmove(d, s, n)


def _handler(gpu, cpu, mappings, gpu_to_cpu):
    """A handler with only the staged-copy state (the constructor insists on
    a device tensor)."""
    h = object.__new__(SingleDirectionOffloadingHandler)
    refs = [
        [CanonicalKVCacheRef(tensor_idx=0, page_size_bytes=m.local_page_size_bytes, mapping=m)]
        for m in mappings
    ]
    h.layer_refs_per_group = refs
    h._staged_refs = [[_staged_ref(r) for r in g] for g in refs]
    h.gpu_to_cpu = gpu_to_cpu
    h.src_tensors, h.dst_tensors = ([gpu], [cpu]) if gpu_to_cpu else ([cpu], [gpu])
    h._swap_blocks_batch = _memmove
    h._slots, h._slot_events = [], [None, None]
    return h


def _run(h, gpu_ids, cpu_ids, group_sizes):
    src, dst = (gpu_ids, cpu_ids) if h.gpu_to_cpu else (cpu_ids, gpu_ids)
    items = h._staged_items(np.asarray(src), np.asarray(dst), group_sizes)
    for batch in h._staged_batches(items):
        slot = torch.zeros(gpu_worker._batch_bytes(batch), dtype=torch.int8)
        if h.gpu_to_cpu:
            h._staged_dma(batch, slot)
            h._staged_cpu(batch, slot)
        else:
            h._staged_cpu(batch, slot)
            h._staged_dma(batch, slot)
    return items


def _oracle(local: np.ndarray, mapping: CanonicalPageMapping, out: np.ndarray, rows):
    for b, row in enumerate(rows):
        for run in mapping.runs:
            for i in range(run.num_fragments):
                lo = run.local_offset + i * run.local_stride
                co = run.canonical_offset + i * run.canonical_stride
                out[row, co : co + run.fragment_size] = local[b, lo : lo + run.fragment_size]


def test_staged_ref_requires_exact_coverage():
    assert _staged_ref(CanonicalKVCacheRef(0, 1024, _shard(2, 0))).span == 1024
    gap = CanonicalPageMapping(_CANON, 1024, (CopyRun(0, 0, 128, 4, 256, 512),), 1, 0, True)
    assert _staged_ref(CanonicalKVCacheRef(0, 1024, gap)) is None


@pytest.mark.parametrize("tp,misalign", [(2, 0), (4, 0), (2, 40)])
def test_staged_store_and_load_match_descriptor_oracle(tp, misalign, monkeypatch):
    # tiny slots force several batches per transfer
    monkeypatch.setattr(gpu_worker, "STAGED_SLOT_BYTES", 1500)
    torch.manual_seed(0)
    num_blocks, num_rows = 5, 8
    local_bytes = 2 * _TOKENS * _HEADS // tp * _HEAD_BYTES
    rows = [6, 1, 3, 0, 7]  # arena chunks, deliberately unordered
    # misalign shifts the arena view off the fragment grid: run-by-run path
    backing = torch.zeros(num_rows, _CANON + 64, dtype=torch.int8)
    canon = backing[:, misalign : misalign + _CANON]
    expected = np.zeros((num_rows, _CANON), dtype=np.int8)
    gpus = []
    for rank in range(tp):
        gpu = torch.randint(-128, 128, (num_blocks + 2, local_bytes), dtype=torch.int8)
        gpus.append(gpu)
        ids = list(range(2, 2 + num_blocks))
        _run(_handler(gpu, canon, [_shard(tp, rank)], True), ids, rows, (num_blocks,))
        _oracle(gpu.numpy()[2:], _shard(tp, rank), expected, rows)
    assert np.array_equal(canon.numpy(), expected)
    assert not backing[:, :misalign].any() and not backing[:, misalign + _CANON :].any()
    # every rank reloads its shard bit-exact into other device blocks
    for rank in range(tp):
        back = torch.zeros(num_blocks + 3, local_bytes, dtype=torch.int8)
        _run(_handler(back, canon, [_shard(tp, rank)], False), [3, 4, 5, 6, 7], rows, (num_blocks,))
        assert torch.equal(back[3:8], gpus[rank][2:7])
        assert not back[:3].any()


def test_staged_store_keeps_writer_rotation_and_groups():
    # group 0: replicated whole pages (2 writers), group 1: a tp=2 shard
    whole = CanonicalPageMapping(256, 256, (CopyRun(0, 0, 256, 1, 256, 256),), 2, 1, True)
    gpu = torch.randint(-128, 128, (6, 1024), dtype=torch.int8)
    canon = torch.zeros(10, _CANON, dtype=torch.int8)
    h = _handler(gpu, canon, [whole, _shard(2, 0)], True)
    items = _run(h, [0, 1, 2, 3, 4], [4, 5, 6, 8, 9], (3, 2))
    # writer 1 of 2 writes the odd canonical chunks of group 0 only
    assert [c.tolist() for p, _, c in items if p.mapping is whole] == [[5]]
    assert torch.equal(canon[5, :256], gpu[1, :256])
    assert not canon[4].any() and not canon[6].any()
    expected = np.zeros((10, _CANON), dtype=np.int8)
    _oracle(gpu.numpy()[3:5], _shard(2, 0), expected, [8, 9])
    assert np.array_equal(canon.numpy()[8:10], expected[8:10])


def test_direct_layout_pages_stage_as_whole_slices(monkeypatch):
    """A direct-layout ref is one whole-page run: staged as slice copies."""
    monkeypatch.setattr(gpu_worker, "STAGED_SLOT_BYTES", 300 << 10)
    page = 128 << 10
    ref = gpu_worker._identity_ref(CanonicalKVCacheRef(0, page, None))
    gpu = torch.randint(-128, 128, (6, page), dtype=torch.int8)
    backing = torch.zeros(5, page + 4096, dtype=torch.int8)  # row stride != page
    cpu = backing[:, 100 : 100 + page]
    h = _handler(gpu, cpu, [ref.mapping], True)
    _run(h, [5, 1, 3], [0, 4, 2], (3,))
    assert torch.equal(cpu[0], gpu[5]) and torch.equal(cpu[4], gpu[1])
    assert torch.equal(cpu[2], gpu[3]) and not cpu[1].any()
    back = torch.zeros(6, page, dtype=torch.int8)
    _run(_handler(back, cpu, [ref.mapping], False), [0, 2, 4], [0, 4, 2], (3,))
    assert torch.equal(back[0], gpu[5]) and torch.equal(back[2], gpu[1])
    assert torch.equal(back[4], gpu[3])
