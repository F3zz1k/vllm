# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B70-PLE-RAM on vLLM 0.32: the INT8 row-scale PLE table native in pinned host RAM.

Port of Lumnus 0008 (``B70_PLE_INT8``, v0.30). PLE is either RAM native
or on SSD: ``B70_PLE_INT8=1`` is the RAM side, ``B70_PLE_INT8_NVME=1``
(b70_ple_nvme.py) the SSD side; both set is refused at init.

At boot each TP rank reads its owned row range [tp_start, tp_end) of
``B70_PLE_INT8_PATH`` (160 int8 + float32 scale = 164 B per row) into pinned
slabs (``ple_ram.PleRamTable``: O_DIRECT sequential chunks, power-of-two
slabs so the caching host allocator does not round 12.2 GiB up to 16), then
checks every owned row's scale, a random sample against O_DIRECT row reads of
the file, and (with ``PLE_TABLE_PATH``) 4096 rows against the BF16 table.

Per step the producer/consumer of the 0.32 host-prepared path (the NVMe class,
inherited unchanged) runs with one difference: ``_fill_stage`` gathers the
looked-up rows straight out of the slabs by row id (``row - tp_start``) into
the pinned stage. Then one H2D of the stage; the consumer (an eager break in
the FULL graph) all-reduces the packed bytes over ETP (one owner per row, the
int8 byte sum is exact) and dequantises per row, q * s in float32, one rounding
to BF16.

Gather choice: host gather into the pinned stage + one H2D, not 0008's
device-side UVA Triton gather. The 0.32 B70 step already takes its one host
sync for the n-gram ids (B70_PLE_HOST_IDS=1 hashes on the host), so the ids
are on the host for free; a host gather from DRAM costs ~0.1 us/row and the
H2D is one contiguous DMA, where a UVA gather issues one PCIe read per 164-B
row from the GPU. The stock 0.32 pinned-UVA class (CUDA stream API) has not
run on XPU, the host-prepared path has (FULL replay gated on the NVMe engine).

Env: B70_PLE_INT8=1, B70_PLE_INT8_PATH (the table),
B70_PLE_INT8_LOAD_THREADS (reads in flight per rank, default 4),
B70_PLE_INT8_LOAD_CHUNK_MIB (default 64), B70_PLE_INT8_BOOT_SAMPLE (rows
re-read with O_DIRECT and compared, default 4096), PLE_TABLE_PATH +
B70_PLE_INT8_CROSSCHECK + B70_PLE_INT8_MAX_REL_ERR (BF16 cross-check).
"""

import os
import time
from typing import ClassVar

import numpy as np
import torch

from vllm.logger import init_logger

from .b70_ple_nvme import (
    INT8_FORMAT,
    SCALE_BYTES,
    Qwen4ExpPLEInt8NVMeEmbedding,
    bf16_crosscheck,
    safetensors_metadata,
)

logger = init_logger(__name__)


def enabled() -> bool:
    """B70_PLE_INT8=1 (RAM native). Raises when B70_PLE_INT8_NVME=1 is also set."""
    on = os.environ.get("B70_PLE_INT8", "0") == "1"
    if on and os.environ.get("B70_PLE_INT8_NVME", "0") == "1":
        raise ValueError(
            "B70_PLE_INT8=1 and B70_PLE_INT8_NVME=1 are exclusive: the INT8 PLE table is "
            "either RAM native (B70_PLE_INT8) or on SSD (B70_PLE_INT8_NVME); unset one")
    if on and not os.environ.get("B70_PLE_INT8_PATH"):
        raise ValueError("B70_PLE_INT8=1 requires B70_PLE_INT8_PATH (the INT8 PLE .safetensors)")
    return on


class Qwen4ExpPLEInt8RAMEmbedding(Qwen4ExpPLEInt8NVMeEmbedding):
    """INT8 row-scale PLE rows resident in pinned host RAM (B70-PLE-RAM, port of 0008)."""

    _flag: ClassVar[str] = "B70_PLE_INT8"
    _label: ClassVar[str] = "B70 PLE INT8 RAM"

    def __init__(self, *args, **kwargs) -> None:
        enabled()  # the flag conflict, also when the class is built directly
        super().__init__(*args, **kwargs)
        self._table = None
        self._slab_tensors: list[torch.Tensor] = []
        self._devgather = None  # B70-V7-PLEG
        self._pleg_checked = False

    def _pinned_slab(self, rows: int, row_bytes: int) -> np.ndarray:
        t = torch.empty(rows, row_bytes, dtype=torch.uint8, device="cpu", pin_memory=True)
        if not t.is_pinned():
            raise RuntimeError(f"B70 PLE INT8 RAM: {rows * row_bytes / 2**30:.2f} GiB slab "
                               "failed to pin")
        self._slab_tensors.append(t)  # the numpy view does not own the storage
        return t.numpy()

    def weight_loader(self, *args, **kwargs) -> None:
        raise RuntimeError("B70 PLE INT8 RAM rejects checkpoint PLE rows (filtered index expected)")

    def host_loaded(self) -> bool:
        return self._table is not None

    def load_host_table(self, owner) -> None:
        """Read the rank's rows into pinned slabs and check them (once)."""
        from vllm.distributed import get_tp_group

        from .ple_nvme import PleNvmeRowStore, owned_heads, safetensors_tensor_location
        from .ple_ram import PleRamTable, pinned_bytes

        if self._table is not None:
            raise RuntimeError("B70 PLE INT8 RAM rejects reload")
        path = os.environ["B70_PLE_INT8_PATH"]
        fmt = safetensors_metadata(path).get("format")
        if fmt != INT8_FORMAT:
            raise ValueError(f"INT8 PLE file {path} has format {fmt!r}, expected {INT8_FORMAT!r}")
        dim = self.embedding_dim
        data_start, shape, dtype = safetensors_tensor_location(path, "table")
        if dtype != "U8" or shape != [self.org_vocab_size, self._storage]:
            raise ValueError(f"INT8 PLE {path}: table {dtype} {shape}, expected U8 "
                             f"[{self.org_vocab_size}, {self._storage}]")
        tp_start = self.shard_indices.org_vocab_start_index
        tp_end = self.shard_indices.org_vocab_end_index
        rank = get_tp_group().rank_in_group
        t0 = time.perf_counter()
        table = PleRamTable(tp_start, tp_end, self._storage, alloc=self._pinned_slab)
        alloc_s = time.perf_counter() - t0
        info = table.load(
            path, data_start,
            threads=int(os.environ.get("B70_PLE_INT8_LOAD_THREADS", "4")),
            chunk_bytes=int(float(os.environ.get("B70_PLE_INT8_LOAD_CHUNK_MIB", "64")) * (1 << 20)))
        loaded_s = time.perf_counter() - t0
        t1 = time.perf_counter()
        table.scan_scales(dim)
        sample = int(os.environ.get("B70_PLE_INT8_BOOT_SAMPLE", "4096"))
        if sample > 0 and tp_end > tp_start:
            g = torch.Generator().manual_seed(20260929 + tp_start)
            own = torch.unique(torch.randint(tp_start, tp_end, (sample,), generator=g)).numpy()
            store = PleNvmeRowStore(path, data_start, self._storage, shape[0], io_threads=8,
                                    reader="py")
            try:
                if not np.array_equal(table.rows(own), store.read_rows(own)):
                    raise ValueError(f"INT8 PLE {path}: loaded rows differ from row reads "
                                     f"(rank {rank}, {own.shape[0]} sampled rows)")
            finally:
                store.close()
        if tp_end > tp_start:
            bf16_crosscheck(lambda rows: _crosscheck_rows(table, rows, path), dim,
                            self.org_vocab_size, path, self._label)
        check_s = time.perf_counter() - t1
        sizes = owner.ngram_heads_vocab_sizes.detach().cpu().numpy()
        offsets = owner.ngram_heads_offsets.detach().cpu().numpy()
        self._own_heads = owned_heads(sizes, offsets, tp_start, tp_end)
        self._table = table
        pinned = pinned_bytes(table.slab_rows, self._storage)
        logger.info(
            "B70 PLE INT8 RAM (rank %d): %d rows, %.2f GiB pinned, loaded in %.1f s "
            "(alloc %.1f s, read %.2f GB/s%s, %d slabs; checks %.1f s), own heads %s, "
            "rows [%d, %d), %s", rank, table.num_rows, pinned / 2**30, loaded_s, alloc_s,
            info["gbps"], "" if info["direct"] else " buffered", info["slabs"], check_s,
            self._own_heads.tolist(), tp_start, tp_end, path)
        from . import b70_ple_devgather

        if b70_ple_devgather.enabled():
            self._devgather = b70_ple_devgather.DeviceGather(
                self._slab_tensors, table.bounds, tp_start, tp_end, self._storage)
            b70_ple_devgather.log_once(
                f"rank {rank}, {len(self._slab_tensors)} slabs, rows [{tp_start}, {tp_end}), "
                f"{self._storage} B/row; prepare_inputs no longer syncs for PLE")

    # -- B70-V7-PLEG: device producer ------------------------------------
    def device_fill(self, ids: torch.Tensor, query_start_loc: torch.Tensor,
                    out: torch.Tensor | None = None) -> None:
        """In-forward producer: rows of ``ids`` [T, heads] into the packed buffer."""
        if out is None:
            out = self._prefetch_buffer[: ids.shape[0]]
        self._devgather(ids, query_start_loc, out)

    def prepare_device(self, num_tokens_padded: int) -> None:
        """B70-V7-PLEG bookkeeping for the consumer: no host work, no copy."""
        self._check_capacity(num_tokens_padded)
        self._prepared_tokens = num_tokens_padded

    def check_device_gather(self, ids_dev: torch.Tensor, query_start_loc: torch.Tensor,
                            num_tokens_padded: int) -> None:
        """Once, on the first real step after ``prepare`` filled the stage on the host:
        the device ids + gather must reproduce the stage bytes exactly."""
        self._pleg_checked = True
        tmp = torch.empty_like(self._prefetch_buffer[:num_tokens_padded])
        self.device_fill(ids_dev, query_start_loc, tmp)
        got = tmp.cpu()
        want = self._stage[:num_tokens_padded]
        bad = int((got != want).any(dim=-1).sum())
        if bad:
            raise RuntimeError(
                f"B70-V7-PLEG device PLE gather differs from the host stage in {bad} of "
                f"{num_tokens_padded * self._heads} (token, head) slots")
        logger.info("B70-V7-PLEG check: device ids + gather equal the host stage "
                    "(%d tokens x %d heads)", num_tokens_padded, self._heads)

    def _fill_stage(self, ids: np.ndarray, num_tokens: int, num_padded: int) -> None:
        stage = self._stage_np
        stage[: max(self._dirty, num_padded)].fill(0)
        self._dirty = 0
        if num_tokens == 0 or self._table is None:
            return
        self._table.gather_into(stage, ids[:num_tokens][:, self._own_heads], self._own_heads)
        self._dirty = num_tokens


def _crosscheck_rows(table, rows: np.ndarray, path: str) -> np.ndarray:
    """Cross-check rows: the rank's own rows from RAM, the others read from the file."""
    rows = np.asarray(rows, dtype=np.int64)
    out = np.empty((rows.size, table.row_bytes), dtype=np.uint8)
    own = (rows >= table.tp_start) & (rows < table.tp_end)
    if own.any():
        out[own] = table.rows(rows[own])
    if (~own).any():
        from .ple_nvme import PleNvmeRowStore, safetensors_tensor_location

        start, shape, _ = safetensors_tensor_location(path, "table")
        store = PleNvmeRowStore(path, start, table.row_bytes, shape[0], io_threads=8, reader="py")
        try:
            out[~own] = store.read_rows(rows[~own])
        finally:
            store.close()
    return out


__all__ = ["Qwen4ExpPLEInt8RAMEmbedding", "enabled", "SCALE_BYTES"]
