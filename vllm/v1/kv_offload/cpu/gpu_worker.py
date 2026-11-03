# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import functools
import os
import queue
import threading
import time
from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch

from vllm import _custom_ops as ops
from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON, triton
from vllm.utils.math_utils import cdiv
from vllm.utils.torch_utils import PIN_MEMORY
from vllm.v1.kv_offload.base import (
    BlockIDsLoadStoreSpec,
    CanonicalKVCacheRef,
    CanonicalKVCaches,
    CanonicalPageMapping,
    GPULoadStoreSpec,
    LoadStoreSpec,
    OffloadingWorker,
    TransferResult,
)
from vllm.v1.kv_offload.cpu.host_register import host_register, host_unregister
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.cpu.swap_blocks_triton import (
    THRESHOLD_BYTES,
    swap_blocks_batch,
)

logger = init_logger(__name__)


def _select_swap_blocks_fn(
    layer_refs_per_group: list[list[CanonicalKVCacheRef]],
    gpu_to_cpu: bool,
    host_memory_is_pinned: bool = True,
):
    """Resolve the swap_blocks function for a handler at init time."""
    # GPU->CPU is bandwidth-bound; the dedicated copy engine beats Triton.
    # The Triton kernel dereferences CPU pointers on the GPU, which is only
    # valid for pinned host memory.
    if gpu_to_cpu or not host_memory_is_pinned:
        return ops.swap_blocks_batch
    # Fall back to the C++ DMA path on platforms where Triton isn't usable
    # (e.g. ROCm host mappings) or where GPU kernels cannot directly
    # dereference CPU pointers (XPU lacks CUDA's unified virtual address space,
    # so the Triton kernel's tl.load(cpu_ptr) is invalid on XPU).
    if not HAS_TRITON or current_platform.is_xpu() or current_platform.is_rocm():
        return ops.swap_blocks_batch
    page_sizes = [r.page_size_bytes for g in layer_refs_per_group for r in g]
    # Triton wins only on small, 8-byte-aligned payloads.
    if (
        not page_sizes
        or max(page_sizes) >= THRESHOLD_BYTES
        or any(s % 8 for s in page_sizes)
    ):
        return ops.swap_blocks_batch
    chunk = min(triton.next_power_of_2(max(page_sizes)), 8192)
    return functools.partial(swap_blocks_batch, bytes_per_chunk=chunk)


@dataclass
class Transfer:
    job_id: int
    stream: torch.cuda.Stream
    start_event: torch.Event
    end_event: torch.Event
    num_bytes: int
    batch_src: torch.Tensor
    batch_dst: torch.Tensor
    batch_sizes: torch.Tensor
    # Set by the copy thread once it has recorded the transfer's events; None
    # when the transfer was submitted on the calling thread.
    submitted: threading.Event | None = None
    # A settled store (see GPULoadStoreSpec.settled): its own copy lane
    settled: bool = False


# Copy thread (2026-10-23): submit each transfer's copies from a background
# thread instead of the caller (the engine's forward thread). On XPU the pool's
# D2H path waits per descriptor and its H2D path does a CPU staging memcpy, both
# inside the copy call, so on the caller they block the forward for the whole
# copy. The caller still orders the copy after the compute stream
# (wait_stream), so the data copied is the same.
COPY_THREAD = os.environ.get("VLLM_KV_OFFLOAD_COPY_THREAD", "0") == "1"

# Debug (2026-10-31): check every canonical copy descriptor against the storage
# of the tensor it was computed from before submitting it, so an address the
# copy engine would fault on raises here with the offending descriptor instead.
CHECK_BOUNDS = os.environ.get("VLLM_KV_OFFLOAD_CHECK_BOUNDS", "0") == "1"

# Debug (2026-10-31): VLLM_KV_OFFLOAD_REGISTER=0 leaves the offload region
# unregistered with the device, so copies go through the driver's own staging;
# it tells a registration fault (copy engine CAT errors) from any other.
REGISTER_REGION = os.environ.get("VLLM_KV_OFFLOAD_REGISTER", "1") == "1"

# Staged canonical copies (2026-11-03). Under the canonical layout with TP > 1
# a rank's share of a canonical page is one fragment per token and head run
# (512 bytes per token at TP=2 on the 27B, 256 at TP=4), and the XPU batch copy
# issues one command per fragment: measured on a B70, 512-byte fragments move
# 0.06 GB/s device-to-registered-host (each waits, the registered arena reads
# as pageable) and 0.16 GB/s host-to-device. Staged, each local page moves as
# ONE contiguous command between the device and a pinned staging slot (6.6 GB/s
# D2H, 9.8 GB/s H2D measured) and the copy thread scatters or gathers the
# fragments between the slot and the arena with strided CPU copies (11 GB/s),
# double buffered so the CPU side of one batch overlaps the DMA of the next.
# Needs the copy thread (the CPU work runs there, never on the forward).
STAGED_COPY = os.environ.get("VLLM_KV_OFFLOAD_STAGED_COPY", "1") == "1"
STAGED_SLOT_BYTES = int(os.environ.get("VLLM_KV_OFFLOAD_STAGED_SLOT_MB", "128")) << 20
_STAGED_ALIGN = 4096


class _StagedRef(NamedTuple):
    """One data ref's local page span and its fragment runs, for staged copies.
    ``frags`` groups the fragments by size: (size, offsets in the staged page,
    offsets in the canonical page), so a batch moves each size class with one
    indexed copy (one GIL-free op instead of one per run: per-run copies on
    the copy thread queued behind the forward thread for the GIL)."""

    tensor_idx: int
    span_lo: int
    span: int
    runs: tuple
    total_bytes: int
    mapping: CanonicalPageMapping
    frags: tuple


def _rows_view(storage, dtype: torch.dtype, width: int) -> torch.Tensor:
    """The whole storage as rows of ``width`` elements of ``dtype``."""
    elt = torch.tensor([], dtype=dtype).element_size()
    rows = storage.nbytes() // (elt * width)
    return torch.tensor([], dtype=dtype).set_(storage, 0, (rows, width), (width, 1))


def _batch_bytes(batch) -> int:
    """Slot bytes a staged batch needs: (plan, gpu blocks, cpu chunks, offset)."""
    plan, g_ids, _, off = batch[-1]
    step = -(-plan.span // _STAGED_ALIGN) * _STAGED_ALIGN
    return max(STAGED_SLOT_BYTES, off + (len(g_ids) - 1) * step + plan.span)


def _staged_ref(ref: CanonicalKVCacheRef) -> "_StagedRef | None":
    """The ref's staged-copy plan: the local span its runs cover, which must be
    covered exactly (no gaps, no overlaps), else None (copy it unstaged)."""
    mapping = ref.mapping
    if mapping is None or not mapping.runs:
        return None
    intervals = []
    for run in mapping.runs:
        if run.local_stride < run.fragment_size and run.num_fragments > 1:
            return None
        for i in range(run.num_fragments):
            start = run.local_offset + i * run.local_stride
            intervals.append((start, start + run.fragment_size))
    intervals.sort()
    lo, hi = intervals[0][0], intervals[0][0]
    for start, end in intervals:
        if start != hi:
            return None  # a gap or an overlap: the span would copy foreign bytes
        hi = end
    total = sum(r.fragment_size * r.num_fragments for r in mapping.runs)
    by_size: dict[int, tuple[list, list]] = {}
    for run in mapping.runs:
        i = np.arange(run.num_fragments, dtype=np.int64)
        local, canon = by_size.setdefault(run.fragment_size, ([], []))
        local.append(run.local_offset - lo + i * run.local_stride)
        canon.append(run.canonical_offset + i * run.canonical_stride)
    frags = tuple(
        (size, np.concatenate(local), np.concatenate(canon))
        for size, (local, canon) in sorted(by_size.items())
    )
    return _StagedRef(
        ref.tensor_idx, lo, hi - lo, tuple(mapping.runs), total, mapping, frags
    )


def _check_descriptor_bounds(
    src: np.ndarray,
    dst: np.ndarray,
    sizes: np.ndarray,
    src_tensor: torch.Tensor,
    dst_tensor: torch.Tensor,
    what: str,
) -> None:
    for name, addrs, tensor in (("src", src, src_tensor), ("dst", dst, dst_tensor)):
        storage = tensor.untyped_storage()
        lo = storage.data_ptr()
        hi = lo + storage.nbytes()
        bad = (addrs < lo) | (addrs + sizes.astype(np.uint64) > hi)
        if bad.any():
            i = int(np.argmax(bad))
            raise RuntimeError(
                f"KV offload {name} descriptor {i} of {what} lies outside its "
                f"storage: {int(addrs[i]):#x} + {int(sizes[i])} not in "
                f"[{lo:#x}, {hi:#x}), {int(bad.sum())} of {len(addrs)} bad"
            )


class _CopyThread:
    """One FIFO thread running submitted copy closures in order."""

    def __init__(self, name: str, device: torch.device | None) -> None:
        self._q: queue.SimpleQueue = queue.SimpleQueue()
        self._device = device
        self.error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)
        self._thread.start()

    def submit(self, fn) -> threading.Event:
        done = threading.Event()
        self._q.put((fn, done))
        return done

    def _run(self) -> None:
        if self._device is not None and self._device.type == "xpu":
            torch.xpu.set_device(self._device)
        while True:
            item = self._q.get()
            if item is None:
                return
            fn, done = item
            try:
                if self.error is None:
                    fn()
            except BaseException as e:  # noqa: BLE001
                logger.exception("KV offload copy thread failed")
                self.error = e
            finally:
                done.set()

    def stop(self) -> None:
        self._q.put(None)
        self._thread.join(timeout=60)


def compute_sub_block_ptrs(
    block_ids: np.ndarray,
    blocks_per_chunk: int,
    output: np.ndarray,
    tensor: torch.Tensor,
    skip_count: int = 0,
):
    """Compute byte pointers for sub-blocks of the given block IDs.

    Each block in block_ids contains blocks_per_chunk sub-blocks.
    The pointer for sub-block j of block b is:
        base_ptr + b * row_stride + j * block_page_size

    where block_page_size = tensor.shape[1] // blocks_per_chunk (gpu page size).

    This handles tensors where row_stride != blocks_per_chunk * block_page_size
    (e.g. non-contiguous CPU tensors).

    Args:
        block_ids: array of block IDs at the tensor's native granularity.
        blocks_per_chunk: number of sub-blocks per block.
        output: pre-allocated pointer array to write pointers into.
        tensor: the source or destination tensor.
        skip_count: sub-blocks to skip in the first block.

    """
    assert skip_count < blocks_per_chunk

    num_sub_blocks = len(output)
    base_ptr = tensor.data_ptr()
    row_stride = tensor.stride(0)

    if blocks_per_chunk == 1:
        # Fast path: 1:1 mapping, no sub-block expansion needed.
        output[:] = base_ptr + block_ids.astype(np.uint64)[:num_sub_blocks] * row_stride
        return

    # Vectorized expansion for blocks_per_chunk > 1.
    assert tensor.shape[1] % blocks_per_chunk == 0
    block_page_size = tensor.shape[1] // blocks_per_chunk
    sub_offsets = np.arange(blocks_per_chunk, dtype=np.uint64) * block_page_size
    # (num_blocks, 1) + (1, blocks_per_chunk) -> (num_blocks, blocks_per_chunk)
    all_ptrs = (
        base_ptr + block_ids.astype(np.uint64)[:, np.newaxis] * row_stride
    ) + sub_offsets[np.newaxis, :]
    # Flatten and apply skip_count / truncation
    flat = all_ptrs.ravel()
    output[:] = flat[skip_count : skip_count + num_sub_blocks]


class CopyPlan(NamedTuple):
    """Precomputed fragment-copy template for one data ref under the canonical
    CPU layout, unrolled from the ref's mapped runs. Offsets are relative to
    the per-block base pointers on each side."""

    frag_offsets_src: np.ndarray
    frag_offsets_dst: np.ndarray
    frag_sizes: np.ndarray
    total_bytes: int

    @property
    def num_frags(self) -> int:
        return len(self.frag_sizes)


def _build_copy_plan(ref: CanonicalKVCacheRef, gpu_to_cpu: bool) -> CopyPlan:
    """Unroll one data ref's mapped runs into a per-fragment CopyPlan."""
    mapping = ref.mapping
    assert mapping is not None
    local: list[int] = []
    canonical: list[int] = []
    sizes: list[int] = []
    for run in mapping.runs:
        for i in range(run.num_fragments):
            local.append(run.local_offset + i * run.local_stride)
            canonical.append(run.canonical_offset + i * run.canonical_stride)
            sizes.append(run.fragment_size)
    src, dst = (local, canonical) if gpu_to_cpu else (canonical, local)
    return CopyPlan(
        frag_offsets_src=np.asarray(src, dtype=np.uint64),
        frag_offsets_dst=np.asarray(dst, dtype=np.uint64),
        frag_sizes=np.asarray(sizes, dtype=np.int64),
        total_bytes=sum(sizes),
    )


def _canonical_page_ids(
    block_ids: np.ndarray, blocks_per_chunk: int, count: int, skip_count: int
) -> np.ndarray:
    """Global canonical page ids matching compute_sub_block_ptrs' enumeration.
    These identify canonical pages consistently across ranks, so they key
    CanonicalPageMapping.is_writer rotation."""
    if blocks_per_chunk == 1:
        return block_ids[:count]
    flat = (
        block_ids[:, np.newaxis] * blocks_per_chunk + np.arange(blocks_per_chunk)
    ).ravel()
    return flat[skip_count : skip_count + count]


def _canonical_block_sizes(
    layer_refs_per_group: list[list[CanonicalKVCacheRef]], num_tensors: int
) -> list[int]:
    """Canonical CPU bytes per GPU block for each tensor, taken from the refs'
    mappings. Requires every ref to carry a mapping."""
    canonical_bytes_per_block = [0] * num_tensors
    for layer_refs in layer_refs_per_group:
        for ref in layer_refs:
            assert ref.mapping is not None
            canonical_bytes_per_block[ref.tensor_idx] = max(
                canonical_bytes_per_block[ref.tensor_idx],
                ref.mapping.canonical_page_size_bytes,
            )
    assert all(size > 0 for size in canonical_bytes_per_block)
    return canonical_bytes_per_block


# Bound registration size to avoid driver limits on large host allocations.
MAX_HOST_REGISTER_CHUNK_BYTES = 64 * 1024**3


def pin_mmap_region(region: SharedOffloadRegion) -> None:
    """Register row-aligned chunks, rolling back on failure."""
    rank = region.rank
    base_ptr = region._base.data_ptr()
    total_size = region.total_size_bytes
    # Chunks end on block-row boundaries, which are page aligned, so neither the
    # driver's page rounding nor any single block transfer straddles two
    # registrations.
    # A region shared by several engines (the shared KV pool) registers one
    # row per call. On XPU, registering a multi-GiB shared region in one call
    # made the copy engine fault intermittently (CAT errors) once several
    # engines shared it; with the region unregistered the faults stopped
    # (2026-10-31).
    max_chunk = (
        min(MAX_HOST_REGISTER_CHUNK_BYTES, region._row_stride)
        if region.external
        else MAX_HOST_REGISTER_CHUNK_BYTES
    )
    rows_per_chunk = max(max_chunk // region._row_stride, 1)
    chunk_size = rows_per_chunk * region._row_stride

    # Register, drain and roll back through the same helper, so a failed
    # chunk leaves neither a pending error nor a partly pinned region.
    addresses: list[int] = []
    for offset in range(0, total_size, chunk_size):
        address = base_ptr + offset
        size = min(chunk_size, total_size - offset)
        if host_register(address, size):
            addresses.append(address)
            continue
        logger.warning(
            "host_register failed for rank=%d at %.2f of %.2f GB; "
            "the offload region stays pageable",
            rank,
            offset / 1e9,
            total_size / 1e9,
        )
        for registered in reversed(addresses):
            host_unregister(registered)
        return

    region.pinned_addresses.extend(addresses)
    region.is_pinned = True
    logger.debug(
        "Host-registered mmap region rank=%d %.2f GB in %d chunk(s)",
        rank,
        total_size / 1e9,
        len(addresses),
    )


def _new_descriptor_buffers(
    num_copy_ops: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pin = PIN_MEMORY
    # CUDA cache_kernels.cu requires int64; XPU DMA engine requires uint64.
    ptr_dtype = torch.uint64 if current_platform.is_xpu() else torch.int64
    return (
        torch.empty(num_copy_ops, dtype=ptr_dtype, pin_memory=pin),
        torch.empty(num_copy_ops, dtype=ptr_dtype, pin_memory=pin),
        torch.empty(num_copy_ops, dtype=ptr_dtype, pin_memory=pin),
    )


class SingleDirectionOffloadingHandler:
    """Handles transfers for a single direction, either CPU->GPU or GPU->CPU.
    Transfers are guaranteed to be executed in order of their submission.
    Each transfer uses a unique CUDA stream, and its stream will start
    executing only after the streams of previous transfers have finished.
    """

    # Started on the first transfer when COPY_THREAD is set. Declared on the
    # class so a handler built without __init__ still reads None.
    _copy_thread: "_CopyThread | None" = None
    _band_thread: "_CopyThread | None" = None

    def __init__(
        self,
        gpu_tensors: list[torch.Tensor],
        cpu_tensors: list[torch.Tensor],
        blocks_per_chunk: int,
        layer_refs_per_group: list[list[CanonicalKVCacheRef]],
        gpu_to_cpu: bool,
        canonical_layout: bool = False,
        host_memory_is_pinned: bool = True,
    ):
        """Initialize a SingleDirectionOffloadingHandler.

        Args:
            gpu_tensors: list of GPU KV cache tensors.
                Each of shape (num_gpu_blocks, gpu_page_size_bytes) with dtype int8.
            cpu_tensors: list of CPU KV cache tensors.
                Each of shape (num_cpu_chunks, cpu_page_size_bytes) with dtype int8.
                Order should match gpu_tensors.
            blocks_per_chunk: number of blocks transferred per chunk.
            layer_refs_per_group: list of CanonicalKVCacheRef per group.
            gpu_to_cpu: if True, transfer from GPU to CPU; otherwise CPU to GPU.
            canonical_layout: if True, CPU pages use the canonical layout
                described by the refs' mappings.
            host_memory_is_pinned: whether the CPU tensors are pinned, so GPU
                kernels may dereference them directly.

        """
        assert len(gpu_tensors) == len(cpu_tensors)
        assert len(gpu_tensors) > 0

        canonical_bytes_per_block = (
            _canonical_block_sizes(layer_refs_per_group, len(gpu_tensors))
            if canonical_layout
            else None
        )

        # assert input tensors are as expected
        for t_idx, (gpu_tensor, cpu_tensor) in enumerate(zip(gpu_tensors, cpu_tensors)):
            assert gpu_tensor.dtype == torch.int8
            assert gpu_tensor.ndim == 2
            assert gpu_tensor.is_cuda or gpu_tensor.is_xpu
            assert cpu_tensor.dtype == torch.int8
            assert cpu_tensor.ndim == 2
            assert cpu_tensor.device.type == "cpu"
            _, gpu_page_size = gpu_tensor.shape
            _, cpu_page_size = cpu_tensor.shape
            if canonical_bytes_per_block is not None:
                assert (
                    cpu_page_size == canonical_bytes_per_block[t_idx] * blocks_per_chunk
                )
            else:
                assert cpu_page_size == gpu_page_size * blocks_per_chunk

        self.src_tensors: list[torch.Tensor] = (
            gpu_tensors if gpu_to_cpu else cpu_tensors
        )
        self.dst_tensors: list[torch.Tensor] = (
            cpu_tensors if gpu_to_cpu else gpu_tensors
        )
        self.gpu_to_cpu: bool = gpu_to_cpu
        self.layer_refs_per_group = layer_refs_per_group
        self._swap_blocks_batch = _select_swap_blocks_fn(
            layer_refs_per_group, gpu_to_cpu, host_memory_is_pinned
        )

        # GPU blocks may be smaller
        # cpu_page_size = gpu_page_size * blocks_per_chunk.
        self.src_blocks_per_chunk = 1 if self.gpu_to_cpu else blocks_per_chunk
        self.dst_blocks_per_chunk = blocks_per_chunk if self.gpu_to_cpu else 1

        # Per (group, ref) static copy plans for the canonical layout
        self._canonical_copy_plans: list[list[CopyPlan]] | None = (
            [
                [_build_copy_plan(ref, gpu_to_cpu) for ref in layer_refs]
                for layer_refs in layer_refs_per_group
            ]
            if canonical_layout
            else None
        )
        self._fill_group_ops = (
            self._fill_canonical_ops if canonical_layout else self._fill_direct_ops
        )
        # Reusable per-block base-pointer scratch for the canonical fill,
        # sized to the largest possible group (grown on demand)
        num_scratch_blocks = gpu_tensors[0].shape[0] if canonical_layout else 0
        self._scratch_bases_src = np.empty(num_scratch_blocks, dtype=np.uint64)
        self._scratch_bases_dst = np.empty(num_scratch_blocks, dtype=np.uint64)

        # Staged canonical copies: per (group, ref) plans, or None when any ref
        # cannot be staged (then every transfer takes the descriptor path)
        self._staged_refs: list[list[_StagedRef | None]] | None = None
        if (
            canonical_layout
            and STAGED_COPY
            and COPY_THREAD
            and blocks_per_chunk == 1
            and current_platform.is_xpu()
        ):
            plans = [[_staged_ref(ref) for ref in refs] for refs in layer_refs_per_group]
            stageable = all(
                p is not None or ref.mapping is None or not ref.mapping.runs
                for refs, ps in zip(layer_refs_per_group, plans)
                for ref, p in zip(refs, ps)
            )
            if stageable:
                self._staged_refs = plans
            logger.info(
                "KV offload %s: staged canonical copies %s",
                "store" if gpu_to_cpu else "load",
                "on" if stageable else "off (a ref has a non-contiguous local span)",
            )
        # staging slots and their last events, per lane (ordered, settled)
        self._lane_slots: tuple[list[torch.Tensor], list[torch.Tensor], ] = ([], [])
        self._lane_slot_events: tuple[list, list] = ([None, None], [None, None])
        # end event of the last transfer of each lane (chaining within a lane)
        self._lane_last: list[torch.Event | None] = [None, None]

        # job_id -> event
        self._transfer_events: dict[int, torch.Event] = {}
        # queue of transfers (job_id, stream, event)
        self._transfers: deque[Transfer] = deque()
        # list of CUDA streams available for re-use
        self._stream_pool: list[torch.cuda.Stream] = []
        # list of CUDA events available for re-use
        self._event_pool: list[torch.Event] = []
        # list of pinned descriptor buffer sets available for re-use
        self._buffer_pool: list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = []

    def _estimate_max_copy_ops(self, group_sizes: Sequence[int]) -> int:
        """Upper bound on the number of copy descriptors for a transfer.

        Exact for the direct layout. The canonical path may fill fewer:
        writer rotation later drops the blocks this rank does not write."""
        num_copy_ops = 0
        for g_idx, (group_size, layer_refs) in enumerate(
            zip(group_sizes, self.layer_refs_per_group)
        ):
            if self._canonical_copy_plans is None:
                num_copy_ops += group_size * len(layer_refs)
            else:
                num_copy_ops += group_size * sum(
                    plan.num_frags for plan in self._canonical_copy_plans[g_idx]
                )
        return num_copy_ops

    def _fill_direct_ops(
        self,
        g_idx: int,
        group_src: np.ndarray,
        group_dst: np.ndarray,
        group_size: int,
        src_skip_count: int,
        dst_skip_count: int,
        all_src: np.ndarray,
        all_dst: np.ndarray,
        all_sizes: np.ndarray,
        op_idx: int,
    ) -> tuple[int, int]:
        """Fill one group's copy descriptors for the direct (worker-private)
        layout: one whole-page copy per (block, ref).

        Returns (op_idx past the filled descriptors, bytes added)."""
        num_bytes = 0
        for data_ref in self.layer_refs_per_group[g_idx]:
            t_idx = data_ref.tensor_idx
            end_idx = op_idx + group_size

            compute_sub_block_ptrs(
                group_src,
                self.src_blocks_per_chunk,
                all_src[op_idx:end_idx],
                self.src_tensors[t_idx],
                skip_count=src_skip_count,
            )
            compute_sub_block_ptrs(
                group_dst,
                self.dst_blocks_per_chunk,
                all_dst[op_idx:end_idx],
                self.dst_tensors[t_idx],
                skip_count=dst_skip_count,
            )

            all_sizes[op_idx:end_idx] = data_ref.page_size_bytes
            num_bytes += group_size * data_ref.page_size_bytes
            op_idx = end_idx
        return op_idx, num_bytes

    def _fill_canonical_ops(
        self,
        g_idx: int,
        group_src: np.ndarray,
        group_dst: np.ndarray,
        group_size: int,
        src_skip_count: int,
        dst_skip_count: int,
        all_src: np.ndarray,
        all_dst: np.ndarray,
        all_sizes: np.ndarray,
        op_idx: int,
    ) -> tuple[int, int]:
        """Fill one group's copy descriptors for the canonical layout:
        scatter each block through the ref's precomputed CopyPlan, keeping
        only the blocks this rank writes.

        Returns (op_idx past the filled descriptors, bytes added)."""
        assert self._canonical_copy_plans is not None
        # Zero-copy reinterpretation for pointer arithmetic: uint64 and the
        # buffers' int64 are bit-equivalent for addresses
        all_src_u64 = all_src.view(np.uint64)
        all_dst_u64 = all_dst.view(np.uint64)
        if group_size > len(self._scratch_bases_src):
            self._scratch_bases_src = np.empty(group_size, dtype=np.uint64)
            self._scratch_bases_dst = np.empty(group_size, dtype=np.uint64)

        num_bytes = 0
        for plan, data_ref in zip(
            self._canonical_copy_plans[g_idx], self.layer_refs_per_group[g_idx]
        ):
            if plan.num_frags == 0:
                continue
            t_idx = data_ref.tensor_idx

            # 1. Base byte pointer of every block on each side
            block_bases_src = self._scratch_bases_src[:group_size]
            block_bases_dst = self._scratch_bases_dst[:group_size]
            compute_sub_block_ptrs(
                group_src,
                self.src_blocks_per_chunk,
                block_bases_src,
                self.src_tensors[t_idx],
                skip_count=src_skip_count,
            )
            compute_sub_block_ptrs(
                group_dst,
                self.dst_blocks_per_chunk,
                block_bases_dst,
                self.dst_tensors[t_idx],
                skip_count=dst_skip_count,
            )

            # 2. On store, keep only the blocks this rank is elected to write
            mapping = data_ref.mapping
            assert mapping is not None
            if self.gpu_to_cpu and mapping.num_writers > 1:
                block_bases_src, block_bases_dst = self._filter_writer_blocks(
                    block_bases_src,
                    block_bases_dst,
                    mapping,
                    group_dst,
                    group_size,
                    dst_skip_count,
                )
            num_active_blocks = len(block_bases_src)

            # 3. Expand (block base + fragment offset) into one descriptor
            #    per (block, fragment), writing straight into the descriptor
            #    buffers: reshaping a contiguous 1D slice is a view, so the
            #    broadcasts below allocate nothing
            end_idx = op_idx + num_active_blocks * plan.num_frags
            np.add(
                block_bases_src[:, None],
                plan.frag_offsets_src[None, :],
                out=all_src_u64[op_idx:end_idx].reshape(
                    num_active_blocks, plan.num_frags
                ),
            )
            np.add(
                block_bases_dst[:, None],
                plan.frag_offsets_dst[None, :],
                out=all_dst_u64[op_idx:end_idx].reshape(
                    num_active_blocks, plan.num_frags
                ),
            )
            all_sizes[op_idx:end_idx].reshape(num_active_blocks, plan.num_frags)[:] = (
                plan.frag_sizes
            )
            if CHECK_BOUNDS:
                _check_descriptor_bounds(
                    all_src_u64[op_idx:end_idx],
                    all_dst_u64[op_idx:end_idx],
                    all_sizes[op_idx:end_idx],
                    self.src_tensors[t_idx],
                    self.dst_tensors[t_idx],
                    f"group {g_idx} tensor {t_idx}",
                )
            num_bytes += num_active_blocks * plan.total_bytes
            op_idx = end_idx
        return op_idx, num_bytes

    def _filter_writer_blocks(
        self,
        block_bases_src: np.ndarray,
        block_bases_dst: np.ndarray,
        mapping: CanonicalPageMapping,
        group_dst: np.ndarray,
        group_size: int,
        dst_skip_count: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Keep only the blocks this rank writes: replicated ranks take turns
        writing shared canonical pages, keyed by the rank-consistent CPU-side
        canonical page id."""
        cpu_page_ids = _canonical_page_ids(
            group_dst,
            self.dst_blocks_per_chunk,
            group_size,
            dst_skip_count,
        )
        writer_mask = cpu_page_ids % mapping.num_writers == mapping.writer_index
        return block_bases_src[writer_mask], block_bases_dst[writer_mask]

    def transfer_async(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        assert isinstance(src_spec, BlockIDsLoadStoreSpec)
        assert isinstance(dst_spec, BlockIDsLoadStoreSpec)

        src_blocks = src_spec.block_ids
        dst_blocks = dst_spec.block_ids
        assert src_blocks.ndim == 1
        assert dst_blocks.ndim == 1

        num_src_blocks = len(src_blocks)
        num_dst_blocks = len(dst_blocks)

        # There are 2 types of transfers:
        # 1. GPU -> CPU
        # 2. CPU -> GPU
        #
        # transfers are also to CPU chunks, EXCEPT MAYBE for the first and last chunk.
        # i.e. the first and last CPU chunks in src_blocks can match against
        # a smaller (byte-wise) set of GPU blocks in dst_blocks.
        # In such cases, we may need to skip some gpu-sized sub-blocks,
        # and start reading/writing from the middle of the first CPU chunk.
        # If we have multiple KV cache groups (when using HMA with hybrid models),
        # we may have a partial first/last CPU chunk per each group.
        # The group_sizes parameter encodes the size of each group of blocks
        # in the GPU dst_blocks.
        # If group_sizes is None, we assume all blocks belong to a single group.
        # The logical_offset parameter maps each group of blocks to its logical
        # offset inside the request, counting in GPU blocks.
        # This allows us to find the correct starting position
        # in the matching first CPU chunk.

        # extract group_sizes from the GPU spec
        gpu_spec = src_spec if self.gpu_to_cpu else dst_spec
        assert isinstance(gpu_spec, GPULoadStoreSpec)
        group_sizes = gpu_spec.group_sizes
        assert len(group_sizes) == len(self.layer_refs_per_group)

        # extract block indices from the GPU spec
        block_indices = gpu_spec.block_indices
        assert len(block_indices) == len(self.layer_refs_per_group)

        if self._staged_refs is not None:
            return self._transfer_staged(
                job_id,
                src_blocks,
                dst_blocks,
                group_sizes,
                block_indices,
                settled=bool(getattr(gpu_spec, "settled", False)),
            )

        num_copy_ops = self._estimate_max_copy_ops(group_sizes)

        # reuse a pooled buffer set, growing it if this transfer needs more room
        batch_src, batch_dst, batch_sizes = (
            self._buffer_pool.pop()
            if self._buffer_pool
            else _new_descriptor_buffers(num_copy_ops)
        )
        if batch_src.numel() < num_copy_ops:
            batch_src, batch_dst, batch_sizes = _new_descriptor_buffers(num_copy_ops)

        src = batch_src[:num_copy_ops]
        dst = batch_dst[:num_copy_ops]
        sizes = batch_sizes[:num_copy_ops]
        all_src = src.numpy()
        all_dst = dst.numpy()
        all_sizes = sizes.numpy()

        src_offset = 0
        dst_offset = 0
        op_idx = 0
        # count total number of bytes copied
        num_transfer_bytes = 0
        for g_idx, (group_size, block_idx) in enumerate(
            zip(group_sizes, block_indices)
        ):
            if group_size == 0:
                continue

            src_logical_blocks_to_skip = block_idx % self.src_blocks_per_chunk
            dst_logical_blocks_to_skip = block_idx % self.dst_blocks_per_chunk
            src_logical_blocks_count = group_size + src_logical_blocks_to_skip
            dst_logical_blocks_count = group_size + dst_logical_blocks_to_skip

            dst_blocks_count = cdiv(dst_logical_blocks_count, self.dst_blocks_per_chunk)
            dst_end_offset = dst_offset + dst_blocks_count
            assert dst_end_offset <= num_dst_blocks

            src_blocks_count = cdiv(src_logical_blocks_count, self.src_blocks_per_chunk)
            src_end_offset = src_offset + src_blocks_count
            assert src_end_offset <= num_src_blocks

            op_idx, group_bytes = self._fill_group_ops(
                g_idx,
                group_src=src_blocks[src_offset:src_end_offset],
                group_dst=dst_blocks[dst_offset:dst_end_offset],
                group_size=group_size,
                src_skip_count=src_logical_blocks_to_skip,
                dst_skip_count=dst_logical_blocks_to_skip,
                all_src=all_src,
                all_dst=all_dst,
                all_sizes=all_sizes,
                op_idx=op_idx,
            )
            num_transfer_bytes += group_bytes

            src_offset = src_end_offset
            dst_offset = dst_end_offset

        assert src_offset == num_src_blocks
        assert dst_offset == num_dst_blocks
        # Writer rotation may skip non-writer blocks, leaving op_idx below
        # the sized upper bound
        assert op_idx <= num_copy_ops
        src = src[:op_idx]
        dst = dst[:op_idx]
        sizes = sizes[:op_idx]

        stream = (
            self._stream_pool.pop() if self._stream_pool else current_platform.Stream()
        )
        start_event = (
            self._event_pool.pop()
            if self._event_pool
            else torch.Event(enable_timing=True)
        )
        end_event = (
            self._event_pool.pop()
            if self._event_pool
            else torch.Event(enable_timing=True)
        )

        # Stores must wait for the model to finish writing the KV they read.
        # Loads must wait for pending writes (including zeroing) to their
        # destination blocks: the scheduler leaves a block unzeroed only in
        # the step that allocates it for the load and cannot retract zeroing
        # shipped in an earlier step for a since-reallocated block; with async
        # scheduling nothing else orders that zeroing against this copy.
        settled = self.gpu_to_cpu and getattr(gpu_spec, "settled", False)
        # With the copy thread, the waits for compute and for the previous
        # transfer are made on the host, never enqueued: the copy engine runs
        # the commands of every stream in one order, so a copy queued behind a
        # device-side wait for compute holds back every other stream's copies
        # until that compute ends (measured: an independent D2H finished with
        # the matmul it did not depend on).
        compute_event = None
        if not settled:
            if COPY_THREAD:
                compute_event = torch.Event()
                compute_event.record(current_platform.current_stream())
            else:
                stream.wait_stream(current_platform.current_stream())
        last_event = self._lane_last[settled]
        self._lane_last[settled] = end_event
        # CPU->GPU reads from host memory, which is never written
        # by a concurrent GPU stream, so CU_MEMCPY_SRC_ACCESS_ORDER_ANY is
        # safe and lets the driver pipeline source reads. GPU->CPU reads
        # from the live GPU KV cache, which the compute stream keeps
        # writing; we must keep STREAM ordering so source reads are gated
        # by the transfer stream's wait_stream(compute) barrier.
        is_src_access_order_any = not self.gpu_to_cpu

        def copy() -> None:
            if compute_event is not None:
                compute_event.synchronize()
            if last_event is not None:
                # assure job will start only after the previous one completes
                if COPY_THREAD:
                    last_event.synchronize()
                else:
                    stream.wait_event(last_event)
            with current_platform.stream(stream):
                start_event.record(stream)
                if op_idx > 0:
                    self._swap_blocks_batch(
                        src,
                        dst,
                        sizes,
                        is_src_access_order_any=is_src_access_order_any,
                    )
                end_event.record(stream)

        submitted = None
        if COPY_THREAD:
            submitted = self._lane_thread(settled, stream).submit(copy)
        else:
            copy()

        self._transfer_events[job_id] = end_event
        self._transfers.append(
            Transfer(
                job_id=job_id,
                stream=stream,
                start_event=start_event,
                end_event=end_event,
                num_bytes=num_transfer_bytes,
                batch_src=batch_src,
                batch_dst=batch_dst,
                batch_sizes=batch_sizes,
                submitted=submitted,
                settled=settled,
            )
        )

        # success
        return True

    # --- staged canonical copies ---

    def _staged_items(
        self,
        src_blocks: np.ndarray,
        dst_blocks: np.ndarray,
        group_sizes: Sequence[int],
    ) -> list[tuple[_StagedRef, np.ndarray, np.ndarray]]:
        """Segments (plan, gpu blocks, cpu chunks): the pages this rank copies,
        one segment per data ref in group and ref order; a store keeps only the
        pages this rank is elected to write."""
        assert self._staged_refs is not None
        gpu_ids_all, cpu_ids_all = (
            (src_blocks, dst_blocks) if self.gpu_to_cpu else (dst_blocks, src_blocks)
        )
        segments: list[tuple[_StagedRef, np.ndarray, np.ndarray]] = []
        offset = 0
        for g_idx, group_size in enumerate(group_sizes):
            if group_size == 0:
                continue
            gpu_ids = np.asarray(gpu_ids_all[offset : offset + group_size], dtype=np.int64)
            cpu_ids = np.asarray(cpu_ids_all[offset : offset + group_size], dtype=np.int64)
            offset += group_size
            for plan in self._staged_refs[g_idx]:
                if plan is None:
                    continue
                g_sel, c_sel = gpu_ids, cpu_ids
                mapping = plan.mapping
                if self.gpu_to_cpu and mapping.num_writers > 1:
                    mask = c_sel % mapping.num_writers == mapping.writer_index
                    g_sel, c_sel = g_sel[mask], c_sel[mask]
                if len(g_sel):
                    segments.append((plan, g_sel, c_sel))
        return segments

    def _lane_thread(self, settled: bool, stream) -> "_CopyThread":
        """The copy thread of a lane, started on first use."""
        name = "_band_thread" if settled else "_copy_thread"
        thread = getattr(self, name)
        if thread is None:
            kind = "d2h" if self.gpu_to_cpu else "h2d"
            thread = _CopyThread(
                f"kv-copy-{kind}{'-band' if settled else ''}",
                getattr(stream, "device", None),
            )
            setattr(self, name, thread)
        return thread

    def _slot(self, idx: int, nbytes: int, settled: bool = False) -> torch.Tensor:
        slots = self._lane_slots[settled]
        events = self._lane_slot_events[settled]
        while len(slots) <= idx:
            slots.append(torch.empty(0, dtype=torch.int8))
        if slots[idx].numel() < nbytes:
            event = events[idx]
            if event is not None:
                event.synchronize()
                events[idx] = None
            nbytes = -(-nbytes // _STAGED_ALIGN) * _STAGED_ALIGN
            # the view cache keeps the old slot alive; rebuild it
            self.__dict__[f"_np_rows_cache{int(settled)}"] = {}
            slots[idx] = torch.empty(nbytes, dtype=torch.int8, pin_memory=True)
        return slots[idx]

    def _np_rows(self, tensor: torch.Tensor, size: int, settled: bool = False) -> np.ndarray:
        """The tensor's whole storage as a numpy array of ``size``-byte rows
        (8-byte elements when the size allows), cached per storage and lane."""
        storage = tensor.untyped_storage()
        key = (storage.data_ptr(), storage.nbytes(), size)
        cache = self.__dict__.setdefault(f"_np_rows_cache{int(settled)}", {})
        view = cache.get(key)
        if view is None:
            elt = 8 if size % 8 == 0 else 1
            dtype = torch.int64 if elt == 8 else torch.int8
            view = cache[key] = _rows_view(storage, dtype, size // elt).numpy()
        return view

    def _staged_batches(
        self, segments: list[tuple[_StagedRef, np.ndarray, np.ndarray]]
    ) -> list[list[tuple[_StagedRef, np.ndarray, np.ndarray, int]]]:
        """Pack segments into slot-sized batches of (plan, gpu blocks, cpu
        chunks, first slot offset); page i of a piece sits at first offset +
        i * the plan's aligned span. A page larger than a slot is a batch alone."""
        batches: list[list[tuple[_StagedRef, np.ndarray, np.ndarray, int]]] = []
        batch: list[tuple[_StagedRef, np.ndarray, np.ndarray, int]] = []
        used = 0
        for plan, g_ids, c_ids in segments:
            step = -(-plan.span // _STAGED_ALIGN) * _STAGED_ALIGN
            start = 0
            while start < len(g_ids):
                fit = (STAGED_SLOT_BYTES - used) // step
                if fit <= 0:
                    if batch:
                        batches.append(batch)
                        batch, used = [], 0
                        continue
                    fit = 1
                end = min(len(g_ids), start + fit)
                batch.append((plan, g_ids[start:end], c_ids[start:end], used))
                used += (end - start) * step
                start = end
        if batch:
            batches.append(batch)
        return batches

    def _staged_dma(
        self, batch: list[tuple[_StagedRef, np.ndarray, np.ndarray, int]], slot: torch.Tensor
    ) -> None:
        """One command per page between the device and the slot."""
        # device addresses can exceed int64: uint64 arithmetic throughout
        slot_base = np.uint64(slot.data_ptr())
        gpu_tensors = self.src_tensors if self.gpu_to_cpu else self.dst_tensors
        gpu_parts, host_parts, size_parts = [], [], []
        for plan, g_ids, _c, off in batch:
            t = gpu_tensors[plan.tensor_idx]
            step = np.uint64(-(-plan.span // _STAGED_ALIGN) * _STAGED_ALIGN)
            n = len(g_ids)
            gpu_parts.append(
                np.uint64(t.data_ptr() + plan.span_lo)
                + g_ids.astype(np.uint64) * np.uint64(t.stride(0))
            )
            host_parts.append(
                slot_base + np.uint64(off) + np.arange(n, dtype=np.uint64) * step
            )
            size_parts.append(np.full(n, plan.span, dtype=np.uint64))
        gpu_side = np.concatenate(gpu_parts)
        host_side = np.concatenate(host_parts)
        sizes = np.concatenate(size_parts)
        src, dst = (gpu_side, host_side) if self.gpu_to_cpu else (host_side, gpu_side)
        self._swap_blocks_batch(
            torch.from_numpy(src),
            torch.from_numpy(dst),
            torch.from_numpy(sizes),
            is_src_access_order_any=not self.gpu_to_cpu,
        )

    def _staged_cpu(
        self,
        batch: list[tuple[_StagedRef, np.ndarray, np.ndarray, int]],
        slot: torch.Tensor,
        settled: bool = False,
    ) -> None:
        """Scatter (store) or gather (load) the batch's fragments between the
        slot, laid out like the local pages, and the canonical arena rows: one
        numpy fancy copy per (fragment size, arena storage), the fragments
        being rows of a 2D view of each side. A fancy copy releases the GIL
        once; every torch call releases and retakes it, and on the copy thread
        each retake queues behind the forward thread. A class whose offsets are
        not all multiples of its size is copied run by run instead."""
        cpu_tensors = self.dst_tensors if self.gpu_to_cpu else self.src_tensors
        classes: dict[tuple[int, int], tuple[torch.Tensor, list, list]] = {}
        slow = []
        for plan, _g, c_ids, off in batch:
            cpu = cpu_tensors[plan.tensor_idx]
            step = -(-plan.span // _STAGED_ALIGN) * _STAGED_ALIGN
            offs = off + np.arange(len(c_ids), dtype=np.int64) * step
            rows = cpu.storage_offset() + c_ids * cpu.stride(0)
            for size, local, canon in plan.frags:
                local_abs = (offs[:, None] + local[None, :]).ravel()
                canon_abs = (rows[:, None] + canon[None, :]).ravel()
                if (local_abs % size).any() or (canon_abs % size).any():
                    slow.append((plan, cpu, rows, offs))
                    continue
                key = (size, cpu.untyped_storage().data_ptr())
                entry = classes.setdefault(key, (cpu, [], []))
                entry[1].append(local_abs // size)
                entry[2].append(canon_abs // size)
        for (size, _), (cpu, local_rows, canon_rows) in classes.items():
            stage2d = self._np_rows(slot, size, settled)
            arena2d = self._np_rows(cpu, size, settled)
            local_idx = np.concatenate(local_rows)
            canon_idx = np.concatenate(canon_rows)
            n = len(local_idx)
            first = int(local_idx[0])
            if int(local_idx[-1]) - first + 1 == n and (np.diff(local_idx) == 1).all():
                stage = stage2d[first : first + n]  # the slot holds them in order
                if self.gpu_to_cpu:
                    arena2d[canon_idx] = stage
                else:
                    np.take(arena2d, canon_idx, axis=0, out=stage)
            elif self.gpu_to_cpu:
                arena2d[canon_idx] = stage2d[local_idx]
            else:
                stage2d[local_idx] = arena2d[canon_idx]
        done = set()
        for plan, cpu, rows, offs in slow:
            if id(plan) in done:
                continue
            done.add(id(plan))
            for row, o in zip(rows.tolist(), offs.tolist()):
                local = slot.storage_offset() + o - plan.span_lo
                for run in plan.runs:
                    shape = (run.num_fragments, run.fragment_size)
                    canon = cpu.as_strided(
                        shape, (run.canonical_stride, 1), row + run.canonical_offset
                    )
                    stage = slot.as_strided(
                        shape, (run.local_stride, 1), local + run.local_offset
                    )
                    if self.gpu_to_cpu:
                        canon.copy_(stage)
                    else:
                        stage.copy_(canon)

    def _staged_run(
        self,
        segments: list[tuple[_StagedRef, np.ndarray, np.ndarray]],
        stream,
        settled: bool = False,
    ) -> None:
        """Runs on the copy thread inside the transfer stream. Store: DMA batch
        k+1 while scattering batch k. Load: gather batch k+1 while batch k's
        DMA runs (a slot is refilled only once its previous DMA is done)."""
        st = self.__dict__.setdefault(
            f"_staged_stats{int(settled)}", [0, 0, 0.0, 0.0, 0.0, 0.0, 0]
        )
        t_start = time.perf_counter()
        wait = cpu = 0.0
        batches = self._staged_batches(segments)
        if self.gpu_to_cpu:
            pending: deque = deque()
            for k, batch in enumerate(batches):
                slot = self._slot(k % 2, _batch_bytes(batch), settled)
                self._staged_dma(batch, slot)
                event = torch.Event()
                event.record(stream)
                pending.append((batch, slot, event))
                while len(pending) == 2 or (pending and k == len(batches) - 1):
                    done_batch, done_slot, done_event = pending.popleft()
                    t0 = time.perf_counter()
                    done_event.synchronize()
                    t1 = time.perf_counter()
                    self._staged_cpu(done_batch, done_slot, settled)
                    wait += t1 - t0
                    cpu += time.perf_counter() - t1
        else:
            for k, batch in enumerate(batches):
                idx = k % 2
                slot = self._slot(idx, _batch_bytes(batch))
                event = self._lane_slot_events[False][idx]
                t0 = time.perf_counter()
                if event is not None:
                    event.synchronize()
                t1 = time.perf_counter()
                self._staged_cpu(batch, slot)
                wait += t1 - t0
                cpu += time.perf_counter() - t1
                self._staged_dma(batch, slot)
                event = torch.Event()
                event.record(stream)
                self._lane_slot_events[False][idx] = event
        now = time.perf_counter()
        st[0] += 1
        st[1] += len(batches)
        st[2] += now - t_start
        st[3] += wait
        st[4] += cpu
        if now - st[5] >= 30.0:
            st[5] = now
            logger.info(
                "KV offload staged %s%s: jobs=%d batches=%d wall=%.2fs "
                "slot_wait=%.2fs cpu=%.2fs (cumulative, copy thread)",
                "store" if self.gpu_to_cpu else "load",
                " (settled lane)" if settled else "",
                st[0], st[1], st[2], st[3], st[4],
            )

    def _transfer_staged(
        self,
        job_id: int,
        src_blocks: np.ndarray,
        dst_blocks: np.ndarray,
        group_sizes: Sequence[int],
        block_indices: Sequence[int],
        settled: bool = False,
    ) -> bool:
        del block_indices  # whole blocks: no partial first chunk at 1 block per chunk
        items = self._staged_items(src_blocks, dst_blocks, group_sizes)
        num_transfer_bytes = sum(plan.total_bytes * len(g) for plan, g, _ in items)
        stream = (
            self._stream_pool.pop() if self._stream_pool else current_platform.Stream()
        )
        start_event = (
            self._event_pool.pop() if self._event_pool else torch.Event(enable_timing=True)
        )
        end_event = (
            self._event_pool.pop() if self._event_pool else torch.Event(enable_timing=True)
        )
        settled = self.gpu_to_cpu and settled
        # Host-side waits only (see transfer_async). The lane's copy thread runs
        # jobs in order and waits for each staged batch it reuses, so no
        # chaining to the previous transfer is needed.
        compute_event = None
        if not settled:
            compute_event = torch.Event()
            compute_event.record(current_platform.current_stream())
        self._lane_last[settled] = end_event

        def copy() -> None:
            if compute_event is not None:
                compute_event.synchronize()
            with current_platform.stream(stream):
                start_event.record(stream)
                if items:
                    self._staged_run(items, stream, settled)
                end_event.record(stream)

        submitted = self._lane_thread(settled, stream).submit(copy)
        self._transfer_events[job_id] = end_event
        empty = torch.empty(0, dtype=torch.int64)
        self._transfers.append(
            Transfer(
                job_id=job_id,
                stream=stream,
                start_event=start_event,
                end_event=end_event,
                num_bytes=num_transfer_bytes,
                batch_src=empty,
                batch_dst=empty,
                batch_sizes=empty,
                submitted=submitted,
                settled=settled,
            )
        )
        return True

    def _submitted(self, transfer: Transfer, wait: bool = False) -> bool:
        if transfer.submitted is None:
            return True
        if wait:
            transfer.submitted.wait()
        elif not transfer.submitted.is_set():
            return False
        for thread in (self._copy_thread, self._band_thread):
            if thread is not None and thread.error is not None:
                raise RuntimeError("KV offload copy failed") from thread.error
        return True

    def get_finished(self) -> list[TransferResult]:
        results: list[TransferResult] = []
        # Each lane completes in order; the settled lane may finish before
        # earlier ordered stores, so every lane's head is checked.
        done: list[Transfer] = []
        for lane in (False, True):
            for transfer in self._transfers:
                if transfer.settled != lane:
                    continue
                if not (self._submitted(transfer) and transfer.end_event.query()):
                    break
                done.append(transfer)
        if done:
            finished = {id(t) for t in done}
            self._transfers = deque(t for t in self._transfers if id(t) not in finished)
        for transfer in done:
            if self._lane_last[transfer.settled] is transfer.end_event:
                self._lane_last[transfer.settled] = None  # the event is pooled again
            transfer_time = (
                transfer.start_event.elapsed_time(transfer.end_event) * 1e-3
            )  # elapsed_time is in milliseconds
            result = TransferResult(
                job_id=transfer.job_id,
                success=True,
                transfer_size=transfer.num_bytes,
                transfer_time=transfer_time,
            )

            results.append(result)
            self._stream_pool.append(transfer.stream)
            if not COPY_THREAD:
                # a queued copy closure may still wait on a finished transfer's
                # end event, so pooled events are only safe without the thread
                self._event_pool.append(transfer.end_event)
                self._event_pool.append(transfer.start_event)
            if transfer.batch_src.numel():
                self._buffer_pool.append(
                    (transfer.batch_src, transfer.batch_dst, transfer.batch_sizes)
                )
            del self._transfer_events[transfer.job_id]
        return results

    def wait(self, job_ids: set[int]):
        by_id = {t.job_id: t for t in self._transfers}
        for job_id in job_ids:
            event = self._transfer_events.get(job_id)
            if event is not None:
                transfer = by_id.get(job_id)
                if transfer is not None:
                    self._submitted(transfer, wait=True)
                event.synchronize()

    def shutdown(self) -> None:
        """Drain this direction and release its transfer-side resources."""
        sync_error: Exception | None = None
        while self._transfers:
            transfer = self._transfers[0]
            try:
                self._submitted(transfer, wait=True)
                transfer.end_event.synchronize()
            except Exception as e:
                logger.exception(
                    "Failed to synchronize transfer end event; "
                    "skipping %d remaining transfers",
                    len(self._transfers) - 1,
                )
                self._transfers.clear()
                sync_error = e
                break
            self._transfers.popleft()

        for name in ("_copy_thread", "_band_thread"):
            thread = getattr(self, name)
            if thread is not None:
                thread.stop()
                setattr(self, name, None)
        self._lane_last = [None, None]
        self._transfer_events.clear()
        self._stream_pool.clear()
        self._event_pool.clear()
        self._buffer_pool.clear()
        self.src_tensors.clear()
        self.dst_tensors.clear()
        if sync_error is not None:
            raise sync_error


class CPUOffloadingWorker(OffloadingWorker):
    """OffloadingWorker for CPU offloading.

    Composes two SingleDirectionOffloadingHandler instances (one for each
    direction) and exposes them through the explicit submit_store /
    submit_load API.
    """

    def __init__(
        self,
        kv_caches: CanonicalKVCaches,
        blocks_per_chunk: int,
        num_cpu_chunks: int,
        mmap_region: SharedOffloadRegion | None = None,
        canonical_layout: bool = False,
    ):
        assert not canonical_layout or mmap_region is not None
        # The caller owns mmap_region until this constructor returns. After a
        # successful construction, the worker is the sole owner and releases
        # it after both transfer directions have stopped.
        self._mmap_region = mmap_region
        pin_memory = PIN_MEMORY
        logger.info("Allocating %d CPU tensors...", len(kv_caches.tensors))
        if mmap_region is not None and pin_memory and REGISTER_REGION:
            pin_mmap_region(mmap_region)
        host_memory_is_pinned = pin_memory and (
            mmap_region is None or mmap_region.is_pinned
        )

        canonical_bytes_per_block = (
            _canonical_block_sizes(kv_caches.group_data_refs, len(kv_caches.tensors))
            if canonical_layout
            else None
        )

        gpu_tensors: list[torch.Tensor] = []
        cpu_tensors: list[torch.Tensor] = []
        for t_idx, kv_cache_tensor in enumerate(kv_caches.tensors):
            gpu_page_size_bytes = kv_cache_tensor.page_size_bytes
            gpu_tensor = kv_cache_tensor.tensor.view(torch.int8).view(
                (-1, gpu_page_size_bytes)
            )
            cpu_page_size_bytes = gpu_page_size_bytes * blocks_per_chunk

            if canonical_bytes_per_block is not None:
                assert mmap_region is not None
                cpu_tensor = mmap_region.create_next_canonical_view(
                    canonical_bytes_per_block[t_idx] * blocks_per_chunk
                )
            elif mmap_region is not None:
                cpu_tensor = mmap_region.create_next_worker_view(cpu_page_size_bytes)
            else:
                t0 = time.monotonic()
                cpu_tensor = torch.zeros(
                    (num_cpu_chunks, cpu_page_size_bytes),
                    dtype=torch.int8,
                    device="cpu",
                    pin_memory=pin_memory,
                )
                logger.debug(
                    "torch.zeros pinned tensor %d×%d (%.2f GB): %.3f s",
                    num_cpu_chunks,
                    cpu_page_size_bytes,
                    num_cpu_chunks * cpu_page_size_bytes / 1e9,
                    time.monotonic() - t0,
                )

            gpu_tensors.append(gpu_tensor)
            cpu_tensors.append(cpu_tensor)

        self._store_handler = SingleDirectionOffloadingHandler(
            gpu_tensors=gpu_tensors,
            cpu_tensors=cpu_tensors,
            blocks_per_chunk=blocks_per_chunk,
            layer_refs_per_group=kv_caches.group_data_refs,
            gpu_to_cpu=True,
            canonical_layout=canonical_layout,
            host_memory_is_pinned=host_memory_is_pinned,
        )

        self._load_handler = SingleDirectionOffloadingHandler(
            gpu_tensors=gpu_tensors,
            cpu_tensors=cpu_tensors,
            blocks_per_chunk=blocks_per_chunk,
            layer_refs_per_group=kv_caches.group_data_refs,
            gpu_to_cpu=False,
            canonical_layout=canonical_layout,
            host_memory_is_pinned=host_memory_is_pinned,
        )

    def submit_store(
        self, job_id: int, src_spec: GPULoadStoreSpec, dst_spec: LoadStoreSpec
    ) -> bool:
        """Async GPU -> CPU."""
        return self._store_handler.transfer_async(job_id, src_spec, dst_spec)

    def submit_load(
        self, job_id: int, src_spec: LoadStoreSpec, dst_spec: GPULoadStoreSpec
    ) -> bool:
        """Async CPU -> GPU."""
        return self._load_handler.transfer_async(job_id, src_spec, dst_spec)

    def get_finished(self) -> list[TransferResult]:
        return self._store_handler.get_finished() + self._load_handler.get_finished()

    def wait(self, job_ids: set[int]) -> None:
        self._store_handler.wait(job_ids)
        self._load_handler.wait(job_ids)

    def shutdown(self) -> None:
        handler_failed = False
        try:
            self._store_handler.shutdown()
        except Exception:
            logger.exception("Failed to shut down store offloading handler")
            handler_failed = True

        try:
            self._load_handler.shutdown()
        except Exception:
            logger.exception("Failed to shut down load offloading handler")
            handler_failed = True

        if self._mmap_region is not None:
            if handler_failed:
                try:
                    torch.accelerator.synchronize()
                except Exception:
                    logger.warning(
                        "Device sync before mmap cleanup failed; "
                        "proceeding with cleanup anyway",
                        exc_info=True,
                    )
            self._mmap_region.cleanup()
            self._mmap_region = None
