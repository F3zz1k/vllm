# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B70 0013 on vLLM 0.32: the INT8 row-scale PLE table served from NVMe.

Port of Lumnus B70-0008/0013 (v0.30 line) onto the 0.32 PLE split and wu1ff's
1.0.5 host-prepared producer pattern (b70_disk.py, MIT): the model state calls
``prepare()`` with the step's device n-gram ids before every forward and FULL
replay; ``prepare()`` resolves this rank's rows through the 0013 row cache
(``ple_nvme.PleNvmeServer``, O_DIRECT reads of the misses), gathers the packed
INT8 rows (160 int8 + float32 scale = 164 B) into a pinned stage and starts one
H2D copy. The consumer (an eager break) reduces the packed bytes over ETP (one
owner per row, so the int8 byte sum is exact), dequantises once (q * s in
float32) and copies BF16 into graph-owned output.

Differences from the v0.30 0013: ids come from the device hash (one D2H per
step, as wu1ff's disk producer) instead of the host hash; no 0013b lookahead
(serving runs LOOKAHEAD=0); no in-RAM SYNC_ONLY mode. Any dispatch size up to
max_num_batched_tokens, any slot count (no 4-slot / DFlash-only limit).

Env (same names as v0.30): B70_PLE_INT8_NVME=1, B70_PLE_INT8_NVME_PATH (or
B70_PLE_INT8_PATH), B70_PLE_INT8_NVME_CACHE_GIB (total over ranks, default 8),
B70_PLE_INT8_NVME_IO_THREADS, B70_PLE_INT8_NVME_READER, B70_PLE_INT8_NVME_STATS,
B70_PLE_INT8_NVME_STATS_S, B70_PLE_INT8_NVME_BOOT_SAMPLE, PLE_TABLE_PATH (BF16
cross-check, optional), B70_PLE_INT8_CROSSCHECK, B70_PLE_INT8_MAX_REL_ERR.

B70-PLE-PRELOAD: B70_PLE_INT8_NVME_PRELOAD=1 (default 0) reads the rank's whole
owned row range into the pinned cache at boot when the cache holds it
(``ple_nvme.preload_cache``; B70_PLE_INT8_NVME_PRELOAD_THREADS,
B70_PLE_INT8_NVME_PRELOAD_CHUNK_MIB); the boot sample rows are then compared
byte for byte against the preloaded slab.
"""

import json
import os
import struct
import threading
from typing import ClassVar

import numpy as np
import torch

from vllm.compilation.breakable_cudagraph import eager_break_during_capture
from vllm.config import get_current_vllm_config
from vllm.logger import init_logger
from vllm.platforms import current_platform

from ..common.ngram_embedding import (
    Qwen4ExpPLEEmbedding,
    Qwen4ExpPLEEmbeddingMethod,
    Qwen4ExpPLEPinnedHostEmbedding,
    Qwen4ExpPLEUnquantizedEmbeddingMethod,
)

logger = init_logger(__name__)

SCALE_BYTES = 4
INT8_FORMAT = "lumnus-ple-int8-rowscale/v1"
CONFLICTING = ("B70_PLE_DISK_BF16", "B70_PLE_DIRECT_PINNED", "B70_PLE_FP8",
               "VLLM_PLE_CPU_OFFLOAD", "B70_PLE_INT8_NVME_LOOKAHEAD",
               "B70_PLE_INT8_NVME_SYNC_ONLY")


def enabled() -> bool:
    return os.environ.get("B70_PLE_INT8_NVME", "0") == "1"


def int8_dequantize(packed: torch.Tensor, logical_dim: int,
                    output_dtype: torch.dtype) -> torch.Tensor:
    """uint8 [..., logical_dim + 4] rows -> [..., logical_dim] (q * s, fp32, one rounding)."""
    values = packed[..., :logical_dim].view(torch.int8).to(torch.float32)
    scales = packed[..., logical_dim:].contiguous().view(torch.float32)
    return (values * scales).to(output_dtype)


def bf16_crosscheck(read_rows, dim: int, org_vocab_size: int, path: str,
                    label: str) -> float | None:
    """Rel-L2 of dequantised INT8 rows vs the BF16 table (PLE_TABLE_PATH, 4096 rows).

    ``read_rows(global_rows: np.ndarray) -> uint8 [n, dim + 4]``. Skipped (None)
    without PLE_TABLE_PATH or with B70_PLE_INT8_CROSSCHECK=0; raises above
    B70_PLE_INT8_MAX_REL_ERR (default 0.02; the 76427c71 table measures ~0.0066).
    """
    ref_path = os.environ.get("PLE_TABLE_PATH")
    if not ref_path or os.environ.get("B70_PLE_INT8_CROSSCHECK", "1") != "1":
        return None
    ref = torch.load(ref_path, mmap=True, weights_only=True)
    ref = ref["table"] if isinstance(ref, dict) else ref
    g = torch.Generator().manual_seed(20260928)
    rows = torch.randint(0, min(ref.shape[0], org_vocab_size), (4096,), generator=g)
    picked = torch.from_numpy(np.ascontiguousarray(read_rows(rows.numpy())))
    got = int8_dequantize(picked, dim, torch.float32)
    want = ref.index_select(0, rows).to(torch.float32)
    rel = float(torch.linalg.vector_norm(got - want)) / max(
        float(torch.linalg.vector_norm(want)), 1e-30)
    lim = float(os.environ.get("B70_PLE_INT8_MAX_REL_ERR", "0.02"))
    logger.info("%s cross-check vs %s: rel L2 %.4f (limit %.3f)", label, ref_path, rel, lim)
    if not rel <= lim:
        raise ValueError(f"INT8 PLE {path} disagrees with {ref_path}: {rel:.4f} > {lim}")
    return rel


def safetensors_metadata(path: str) -> dict:
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        return json.loads(f.read(n)).get("__metadata__", {}) or {}


class Qwen4ExpPLEInt8NVMeEmbedding(Qwen4ExpPLEPinnedHostEmbedding):
    """INT8 row-scale PLE rows from NVMe through a pinned row cache (B70 0013)."""

    host_prepared: ClassVar[bool] = True
    # The env flag and log label of this storage mode (B70-PLE-RAM subclasses).
    _flag: ClassVar[str] = "B70_PLE_INT8_NVME"
    _label: ClassVar[str] = "B70 PLE NVMe"

    def __init__(self, num_embeddings: int, embedding_dim: int, *,
                 params_dtype: torch.dtype, padding_size: int, prefix: str,
                 embedding_method: Qwen4ExpPLEEmbeddingMethod,
                 num_ngram_heads: int = 1, max_total_tokens: int = 0,
                 data_parallel_rank: int = 0, reduce_results: bool = True) -> None:
        bad = [n for n in CONFLICTING if os.environ.get(n, "0") == "1"]
        if bad:
            raise ValueError(f"{self._flag}=1 conflicts with {bad}")
        if not current_platform.is_xpu():
            raise ValueError(f"{self._flag}=1 is XPU-only")
        if not isinstance(embedding_method, Qwen4ExpPLEUnquantizedEmbeddingMethod):
            raise ValueError(f"{self._flag}=1 needs the unquantized (BF16) PLE method "
                             f"in the checkpoint, got {type(embedding_method).__name__}")
        config = get_current_vllm_config()
        if config.parallel_config.data_parallel_size != 1:
            raise ValueError(f"{self._flag}=1 requires DP=1")
        Qwen4ExpPLEEmbedding.__init__(
            self, num_embeddings, embedding_dim, params_dtype=params_dtype,
            padding_size=padding_size, prefix=prefix, embedding_method=embedding_method,
            num_ngram_heads=num_ngram_heads, max_total_tokens=max_total_tokens,
            data_parallel_rank=data_parallel_rank, reduce_results=reduce_results)
        if self.etp_data_parallel_size != 1:
            raise ValueError(f"{self._flag}=1 requires ETP = TP (no DP span)")
        self._stream_mod = torch.xpu
        self._out_dtype = params_dtype
        self._capacity = int(max_total_tokens)
        self._heads = int(num_ngram_heads)
        self._storage = int(embedding_dim) + SCALE_BYTES
        self._output_dim = self._heads * int(embedding_dim)
        self._prefetch_buffer = torch.empty(self._capacity, self._heads, self._storage,
                                            dtype=torch.uint8, device="xpu")
        self._stage = torch.zeros(self._capacity, self._heads, self._storage,
                                  dtype=torch.uint8, device="cpu", pin_memory=True)
        if not self._stage.is_pinned():
            raise RuntimeError(f"{self._label} stage failed to pin")
        self._stage_np = self._stage.numpy()
        self._server = None
        self._slab_np = None
        self._own_heads = None
        self._pending = False
        self._copy_event = torch.xpu.Event()
        self._owner_stream = None
        self._owner_thread = None
        self._lock = threading.Lock()
        self._failed = False
        self._prepared_tokens = 0
        self._dirty = 0  # stage rows that may hold non-zero bytes

    # -- loading -----------------------------------------------------------
    def allocate_embedding_weight(self, num_embeddings, embedding_dim, dtype):
        return torch.empty(0, embedding_dim, dtype=dtype, device="cpu")

    def weight_loader(self, *args, **kwargs) -> None:
        raise RuntimeError("B70 PLE NVMe rejects checkpoint PLE rows (filtered index expected)")

    def host_loaded(self) -> bool:
        return self._server is not None

    def load_host_table(self, owner) -> None:
        """Open the INT8 table, check it, build the cache + server (once)."""
        from vllm.distributed import get_tp_group

        from .ple_nvme import (PleNvmeRowStore, PleNvmeServer, PleNvmeStats,
                               PleRowCache, cache_rows_per_rank, owned_heads,
                               preload_cache, safetensors_tensor_location)

        if self._server is not None:
            raise RuntimeError("B70 PLE NVMe rejects reload")
        path = os.environ.get("B70_PLE_INT8_NVME_PATH") or os.environ["B70_PLE_INT8_PATH"]
        fmt = safetensors_metadata(path).get("format")
        if fmt != INT8_FORMAT:
            raise ValueError(f"INT8 PLE file {path} has format {fmt!r}, expected {INT8_FORMAT!r}")
        dim = self.embedding_dim
        data_start, shape, dtype = safetensors_tensor_location(path, "table")
        if dtype != "U8" or shape != [self.org_vocab_size, self._storage]:
            raise ValueError(f"INT8 PLE {path}: table {dtype} {shape}, expected U8 "
                             f"[{self.org_vocab_size}, {self._storage}]")
        store = PleNvmeRowStore(
            path, data_start, self._storage, shape[0],
            io_threads=int(os.environ.get("B70_PLE_INT8_NVME_IO_THREADS", "16")),
            reader=os.environ.get("B70_PLE_INT8_NVME_READER", "py"))
        tp_start = self.shard_indices.org_vocab_start_index
        tp_end = self.shard_indices.org_vocab_end_index
        bf16_crosscheck(store.read_rows, dim, self.org_vocab_size, path, self._label)
        sample = int(os.environ.get("B70_PLE_INT8_NVME_BOOT_SAMPLE", "65536"))
        own = own_bytes = None
        if sample > 0 and tp_end > tp_start:
            g = torch.Generator().manual_seed(20260929 + tp_start)
            own = torch.unique(torch.randint(tp_start, tp_end, (sample,), generator=g)).numpy()
            own_bytes = store.read_rows(own)
            sc = torch.from_numpy(own_bytes)[:, dim:].contiguous().view(torch.float32)
            if not bool(torch.isfinite(sc).all()) or bool((sc < 0).any()):
                raise ValueError(f"INT8 PLE {path}: non-finite or negative row scale")
        sizes = owner.ngram_heads_vocab_sizes.detach().cpu().numpy()
        offsets = owner.ngram_heads_offsets.detach().cpu().numpy()
        self._own_heads = owned_heads(sizes, offsets, tp_start, tp_end)
        total_gib = float(os.environ.get("B70_PLE_INT8_NVME_CACHE_GIB", "8"))
        capacity = cache_rows_per_rank(total_gib, self.tp_size, self._storage)
        needed = 2 * self._capacity * max(1, self._own_heads.shape[0])
        if capacity < needed:
            raise RuntimeError(f"B70_PLE_INT8_NVME_CACHE_GIB={total_gib}: {capacity} rows/rank "
                               f"< {needed} (2 steps x {self._capacity} tokens x own heads)")
        slab = torch.empty(capacity, self._storage, dtype=torch.uint8, device="cpu", pin_memory=True)
        self._slab = slab
        self._slab_np = slab.numpy()
        cache = PleRowCache(tp_end - tp_start, capacity, self._storage, slab=self._slab_np)
        rank = get_tp_group().rank_in_group
        preloaded = None
        if os.environ.get("B70_PLE_INT8_NVME_PRELOAD", "0") == "1":
            preloaded = preload_cache(
                cache, store, tp_start, rank=rank,
                threads=int(os.environ.get("B70_PLE_INT8_NVME_PRELOAD_THREADS", "4")),
                chunk_bytes=int(float(os.environ.get(
                    "B70_PLE_INT8_NVME_PRELOAD_CHUNK_MIB", "64")) * (1 << 20)))
            # The O_DIRECT row reads of the boot sample vs the bulk preload.
            if (preloaded is not None and own is not None
                    and not np.array_equal(self._slab_np[own - tp_start], own_bytes)):
                raise ValueError(f"INT8 PLE {path}: preloaded rows differ from row reads "
                                 f"(rank {rank}, {own.shape[0]} sampled rows)")
        stats = PleNvmeStats(rank, float(os.environ.get("B70_PLE_INT8_NVME_STATS_S", "60")),
                             os.environ.get("B70_PLE_INT8_NVME_STATS", "0") == "1")
        self._server = PleNvmeServer(
            tp_start=tp_start, tp_end=tp_end,
            multipliers=owner.layer_multipliers.detach().cpu().numpy(),
            sizes=sizes, offsets=offsets, eos_token_id=owner.eos_token_id,
            heads_per_ngram=owner.heads_per_ngram, cache=cache, store=store, stats=stats)
        logger.info("B70 PLE NVMe ready (rank %d, %s): cache %d rows (%.2f GiB pinned), "
                    "own heads %s, rows [%d, %d), stage %d tokens%s", rank, path, capacity,
                    capacity * self._storage / 2**30, self._own_heads.tolist(), tp_start,
                    tp_end, self._capacity, ", preloaded" if preloaded is not None else "")

    # -- producer (host, before forward / FULL replay) ---------------------
    def _check_capacity(self, num_tokens: int) -> None:
        if not 0 <= num_tokens <= self._capacity:
            raise ValueError(f"{self._label} stage overflow: {num_tokens} > {self._capacity}")
        if self._failed:
            raise RuntimeError(f"{self._label} stage is quarantined")
        if self._stream_mod.is_current_stream_capturing():
            raise RuntimeError(f"{self._label} host preparation cannot run inside capture")

    def _wait_for_stage(self) -> None:
        if self._pending:
            try:
                self._copy_event.synchronize()
            except BaseException:
                self._failed = True
                raise
            self._pending = False

    def _fill_stage(self, ids: np.ndarray, num_tokens: int, num_padded: int) -> None:
        stage = self._stage_np
        stage[: max(self._dirty, num_padded)].fill(0)
        self._dirty = 0
        if num_tokens == 0 or self._server is None:
            return
        own_ids = ids[:num_tokens][:, self._own_heads]
        slots, mask = self._server.resolve_ids(own_ids)
        tok, col = np.nonzero(mask)
        if tok.size:
            stage[tok, self._own_heads[col]] = self._slab_np[slots[tok, col]]
        self._dirty = num_tokens

    def prepare(self, ngram_ids: torch.Tensor | None, num_tokens: int,
                num_tokens_padded: int, host_ids: np.ndarray | None = None) -> None:
        self._check_capacity(num_tokens_padded)
        if not 0 <= num_tokens <= num_tokens_padded:
            raise ValueError(f"{self._label} invalid actual/padded token counts")
        if not self._lock.acquire(blocking=False):
            raise RuntimeError(f"Concurrent {self._label} stage preparation")
        try:
            stream = self._stream_mod.current_stream()
            thread = threading.get_ident()
            self._wait_for_stage()
            if self._owner_stream is not None and (
                    stream != self._owner_stream or thread != self._owner_thread):
                self._owner_stream.synchronize()
            self._owner_stream, self._owner_thread = stream, thread
            if host_ids is not None:
                if tuple(host_ids.shape) != (num_tokens, self._heads):
                    raise ValueError(f"{self._label}: host ids shape mismatch")
                self._fill_stage(host_ids, num_tokens, num_tokens_padded)
            elif ngram_ids is None:
                self._fill_stage(None, 0, num_tokens_padded)
            else:
                if tuple(ngram_ids.shape) != (num_tokens_padded, self._heads):
                    raise ValueError(f"{self._label}: dispatch ids shape mismatch")
                ids = ngram_ids[:num_tokens].to("cpu", non_blocking=False).numpy()
                self._fill_stage(ids, num_tokens, num_tokens_padded)
            if num_tokens_padded:
                self._pending = True
                try:
                    self._prefetch_buffer[:num_tokens_padded].copy_(
                        self._stage[:num_tokens_padded], non_blocking=True)
                    self._copy_event.record(stream)
                except BaseException:
                    self._failed = True
                    raise
            self._prepared_tokens = num_tokens_padded
        finally:
            self._lock.release()

    def start_prefetch(self, ngram_ids) -> None:
        return None

    # -- consumer (eager break inside the forward / replay) ----------------
    @eager_break_during_capture
    def _finalize_prefetch(self, prefetch_output, output) -> None:
        if self._failed or self._prepared_tokens != output.shape[0]:
            raise RuntimeError(f"{self._label} consumer has no matching prepared dispatch")
        packed = prefetch_output[: output.shape[0]]
        if self.tp_size > 1:
            packed = self.parallel_group.all_reduce(packed.view(torch.int8)).view(torch.uint8)
        output.copy_(int8_dequantize(packed, self.embedding_dim, output.dtype).flatten(-2))

    def finalize_prefetch(self, total_tokens: int) -> torch.Tensor:
        output = torch.empty((total_tokens, self._output_dim), dtype=self._out_dtype,
                             device=self._prefetch_buffer.device)
        self._finalize_prefetch(self._prefetch_buffer, output)
        return output
