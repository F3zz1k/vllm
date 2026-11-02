# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OffloadingSpec for the shared host-RAM KV pool.

Configure it through kv_connector_extra_config::

    {"spec_name": "SharedPoolOffloadingSpec",
     "shared_pool": {"socket": "/run/kvpool/kvpool.sock", "pool": "qwen3.8-27b"}}

``pool`` names the layout class in the server's config. The server owns one
arena file shared by every class; each engine maps it whole at its own row
stride and is only ever handed rows its class owns. The arena size is set on
the server, not here (cpu_bytes_to_use is ignored).

Engines share rows only when their layout key matches: the same model, KV
dtype, cache groups, block and chunk geometry and parallel layout, so a chunk
row written by any of them has the same meaning to all of them. The server
refuses an engine whose key differs from its class's.

With "canonical_layout": true in kv_connector_extra_config, and every layer
certified portable, rows hold canonical (TP=1) pages and the parallel layout
leaves the key: engines of different TP sizes share one class.

The pool is optional at run time: when the server is out of reach, or does
not answer ATTACH/INFO within VLLM_KVPOOL_ATTACH_TIMEOUT_SEC (the server
defers both until its arena is ready), the engine serves without offloading.
"""

import hashlib
import json
import os
from typing import Any

from typing_extensions import override

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.v1.kv_offload.base import CanonicalKVCaches, OffloadingManager
from vllm.v1.kv_offload.config import OffloadingConfig
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec, envs_b70_private_pinned
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.shared_pool import client as C
from vllm.v1.kv_offload.shared_pool import protocol as P
from vllm.v1.kv_offload.shared_pool.client import (
    PoolClient,
    PoolServerError,
    pool_info,
)
from vllm.v1.kv_offload.shared_pool.manager import SharedPoolOffloadingManager

logger = init_logger(__name__)

DEFAULT_SOCKET = "/run/kvpool/kvpool.sock"
_LAYOUT_VERSION = 2
# A reattach of a running engine must not stall its scheduler for long.
_REATTACH_TIMEOUT_SEC = 5.0

# What a worker reports to the scheduler side after mapping the arena:
# (st_dev, st_ino) of the file it mapped and its row count, or None when it
# runs without the pool.
ArenaIdentity = tuple[tuple[int, int], int]


def _build_identity(model_name: str) -> dict[str, Any]:
    """What else decides the meaning of a chunk's bytes beyond its geometry:
    the vLLM build (kernels and KV formats) and the checkpoint behind the model
    path. The KV cache dtype is already in the FileMapper fields (model.dtype).
    Every part must come out the same in the scheduler and in the workers, so
    nothing here may depend on per-process state such as the current config.
    Each part is best effort: a missing one is recorded as None."""
    out: dict[str, Any] = {}
    try:
        import vllm

        out["vllm_version"] = vllm.__version__
    except Exception:
        out["vllm_version"] = None
    try:
        with open(os.path.join(model_name, "config.json"), "rb") as f:
            out["model_config_sha"] = hashlib.sha256(f.read()).hexdigest()[:16]
    except OSError:
        out["model_config_sha"] = None
    return out


def _file_identity(path: str) -> tuple[int, int]:
    st = os.stat(path)
    return st.st_dev, st.st_ino


class SharedPoolOffloadingSpec(CPUOffloadingSpec):
    def __init__(self, config: OffloadingConfig):
        super().__init__(config)
        pool_cfg: dict[str, Any] = dict(self.extra_config.get("shared_pool") or {})
        if not pool_cfg.get("pool"):
            raise ValueError(
                "SharedPoolOffloadingSpec needs shared_pool.pool in "
                "kv_connector_extra_config"
            )
        self.pool_name: str = str(pool_cfg["pool"])
        self.socket_path: str = str(pool_cfg.get("socket", DEFAULT_SOCKET))
        # Per-request timeout once attached; the attach itself has its own.
        self.timeout_sec = float(pool_cfg.get("timeout_sec", 600.0))
        self.layout_key = self._layout_key()
        # Scheduler side: the arena this engine's workers mapped (from their
        # handshakes), the only one a reattach may accept.
        self._worker_arena: ArenaIdentity | None = None
        # Scheduler side: the arena its own attach found.
        self._scheduler_arena: ArenaIdentity | None = None
        # Worker side: what this worker mapped, for its handshake.
        self._mapped_arena: ArenaIdentity | None = None

    @override
    def _cpu_bytes_to_use(self) -> int:
        # Sized by the server: num_chunks comes with the attach or info reply.
        return 0

    @override
    def _uses_shared_region(self) -> bool:
        # The pool is a shared region by definition, so the private pinned
        # tier switch (B70_OFFLOAD_PRIVATE_PINNED) does not apply to it.
        if current_platform.is_xpu() and envs_b70_private_pinned():
            logger.warning_once(
                "B70_OFFLOAD_PRIVATE_PINNED is ignored by the shared KV pool"
            )
        return (
            current_platform.is_cuda_alike() and not current_platform.is_rocm()
        ) or current_platform.is_xpu()

    @property
    def _portable(self) -> bool:
        """Whether every row holds canonical pages certified identical under
        any TP size, so engines of different TP sizes share the pool."""
        return (
            self.config.canonical_layout
            and self.config.parallel.is_parallelism_agnostic
        )

    def _layout_key(self) -> str:
        fields = FileMapper.from_offloading_spec(
            "", self, parallel_agnostic=self._portable
        ).get_run_config()
        fields.update(
            layout_version=_LAYOUT_VERSION,
            kv_bytes_per_chunk=self.kv_bytes_per_chunk,
            blocks_per_chunk=self.blocks_per_chunk,
            replicated_layout=self.replicated_layout,
            kv_cache_layout=self.config.kv_cache_layout,
            **_build_identity(self.config.model.name),
        )
        if not self._portable:
            fields.update(
                cpu_page_size_per_worker=self.cpu_page_size_per_worker,
                world_size=self.config.parallel.world_size,
            )
        blob = json.dumps(fields, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]

    def _pool_args(self) -> dict[str, Any]:
        if self.kv_bytes_per_chunk <= 0:
            raise RuntimeError("Shared KV pool needs a known KV chunk size")
        return {
            "pool": self.pool_name,
            "layout_key": self.layout_key,
            "row_stride": self.kv_bytes_per_chunk,
        }

    def _attach_args(self) -> dict[str, Any]:
        return {
            **self._pool_args(),
            "version": C.PROTOCOL_VERSION,
            "name": self.config.engine_id,
        }

    # --- scheduler side ---

    def _attach(self, timeout: float) -> tuple[PoolClient, dict[str, Any]] | None:
        """ATTACH this engine; None when the pool is out of reach or refuses."""
        client = None
        try:
            client = PoolClient(self.socket_path, timeout=timeout)
            reply = client.call(P.OP_ATTACH, self._attach_args())
            if not isinstance(reply, dict) or "path" not in reply:
                raise ValueError(f"malformed attach reply {reply!r}")
            arena = _file_identity(str(reply["path"]))
        except (*C.UNREACHABLE, PoolServerError) as exc:
            if client is not None:
                client.close()
            logger.warning(
                "Shared KV pool %s at %s is unavailable (%s: %s)",
                self.pool_name,
                self.socket_path,
                type(exc).__name__,
                exc,
            )
            return None
        client.settimeout(self.timeout_sec)
        reply["_arena"] = (arena, int(reply["num_chunks"]))
        return client, reply

    def _reattach(self) -> PoolClient | None:
        """Attach again after losing the server, if it serves the arena the
        workers mapped. A different arena never comes back: stop trying."""
        expected = self._worker_arena
        if expected is None:
            return None
        attached = self._attach(_REATTACH_TIMEOUT_SEC)
        if attached is None:
            return None
        client, reply = attached
        if reply["_arena"] != expected:
            client.close()
            logger.error(
                "Shared KV pool %s now serves another arena than the one this "
                "engine's workers mapped; the engine runs without offloading "
                "until it is restarted",
                self.pool_name,
            )
            self._worker_arena = None
            return None
        return client

    @override
    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            attached = self._attach(C.ATTACH_TIMEOUT_SEC)
            client, reply = attached if attached is not None else (None, {})
            if client is not None:
                self.num_chunks = int(reply["num_chunks"])
                logger.info(
                    "Attached to shared KV pool class %s as client %s: arena %s "
                    "(generation %s), %d rows of %d bytes, %d KV groups, layout "
                    "%s (canonical %s, shared across TP sizes %s)",
                    self.pool_name,
                    reply.get("client_id"),
                    reply.get("path"),
                    reply.get("generation"),
                    self.num_chunks,
                    self.kv_bytes_per_chunk,
                    len(self.config.groups),
                    self.layout_key,
                    self.config.canonical_layout,
                    self._portable,
                )
            self._manager = SharedPoolOffloadingManager(
                client,
                pool=self.pool_name,
                engine_tag=self.config.engine_id,
                enable_events=self.kv_events_config.enable_kv_cache_events,
                free_fraction=reply.get("class_free_fraction"),
                reconnect=self._reattach,
            )
            self._scheduler_arena = reply.get("_arena")
        return self._manager

    @override
    def set_worker_handshakes(self, handshakes: dict[int, Any]) -> None:
        """Keep the pool only if every worker mapped the arena this scheduler
        attached to; a worker without it would fail any transfer."""
        manager = self.get_manager()
        assert isinstance(manager, SharedPoolOffloadingManager)
        arenas = {handshake.get("arena") for handshake in handshakes.values()}
        if (
            len(handshakes) == self.config.parallel.world_size
            and len(arenas) == 1
            and None not in arenas
        ):
            self._worker_arena = arenas.pop()
        if self._worker_arena is None:
            manager.reconnect = None
            manager.disable("the workers did not all map one arena")
        elif self._worker_arena != self._scheduler_arena:
            manager.disable("the workers mapped another arena than this attach")

    # --- worker side ---

    def _map_arena(self) -> SharedOffloadRegion | None:
        """Map the arena whole at this engine's row stride, or None when the
        pool is out of reach (the worker then runs without it)."""
        try:
            info = pool_info(self.socket_path, self._pool_args())
            self.num_chunks = int(info["num_chunks"])
            region = SharedOffloadRegion(
                engine_id=self.config.engine_id,
                num_chunks=self.num_chunks,
                rank=0 if self.replicated_layout else self.config.parallel.rank,
                kv_bytes_per_chunk=self.kv_bytes_per_chunk,
                cpu_page_size=self.cpu_page_size_per_worker,
                external_path=str(info["path"]),
            )
        except (*C.UNREACHABLE, PoolServerError, RuntimeError) as exc:
            logger.warning(
                "Shared KV pool %s at %s is unavailable (%s: %s); this worker "
                "runs without offloading",
                self.pool_name,
                self.socket_path,
                type(exc).__name__,
                exc,
            )
            self.num_chunks = 0
            return None
        assert region.fd is not None
        st = os.fstat(region.fd)
        self._mapped_arena = ((st.st_dev, st.st_ino), self.num_chunks)
        return region

    @override
    def create_worker(self, kv_caches: CanonicalKVCaches) -> CPUOffloadingWorker:
        if not self._uses_shared_region():
            raise RuntimeError("The shared KV pool needs a CUDA or XPU platform")
        if self.config.canonical_layout:
            self._check_canonical_refs(kv_caches)
        mmap_region = self._map_arena()
        if mmap_region is None:
            # No rows: the scheduler side disables the pool on our handshake,
            # so this worker is never handed a transfer.
            return CPUOffloadingWorker(
                kv_caches=kv_caches,
                blocks_per_chunk=self.blocks_per_chunk,
                num_cpu_chunks=0,
            )
        try:
            # Registers every row before any transfer can exist.
            worker = CPUOffloadingWorker(
                kv_caches=kv_caches,
                blocks_per_chunk=self.blocks_per_chunk,
                num_cpu_chunks=self.num_chunks,
                mmap_region=mmap_region,
                canonical_layout=self.config.canonical_layout,
            )
        except Exception:
            mmap_region.cleanup()
            raise
        if current_platform.is_xpu():
            from vllm._custom_ops import _xpu_swap_blocks_batch_has_snapshot_src

            # Loads read the registered rows directly (vllm-xpu-kernels #662,
            # snapshot_src=False); otherwise every load is staged through
            # torch's caching host allocator, whose blocks stay cached.
            direct = mmap_region.is_pinned and _xpu_swap_blocks_batch_has_snapshot_src()
            (logger.info if direct else logger.warning)(
                "Shared KV pool rank %d: %d rows registered: %s; loads %s",
                self.config.parallel.rank,
                self.num_chunks,
                mmap_region.is_pinned,
                "copy directly from the arena" if direct else "are staged",
            )
        return worker

    @override
    def get_worker_handshake(self) -> dict[str, Any]:
        # Sent even without an arena, so the scheduler side hears from every
        # worker.
        return {"arena": self._mapped_arena}

    def _check_canonical_refs(self, kv_caches: CanonicalKVCaches) -> None:
        """Refuse to write anything but what the layout key promised: canonical
        pages for every layer, portable ones when the key is shared across TP
        sizes (the static gate certified them before any worker existed)."""
        refs = [ref for group_refs in kv_caches.group_data_refs for ref in group_refs]
        if any(ref.mapping is None for ref in refs):
            raise RuntimeError(
                "canonical_layout: a KV cache layer has no canonical mapping "
                "(packed or cross-layer KV layout); remove canonical_layout"
            )
        if self._portable and not all(
            ref.mapping is not None and ref.mapping.parallelism_agnostic
            for ref in refs
        ):
            raise RuntimeError(
                "canonical_layout: a layer is not portable across TP sizes, "
                "but the pool layout key is shared across them"
            )
        logger.info(
            "Shared KV pool rows are canonical (shared across TP sizes: %s)",
            self._portable,
        )
