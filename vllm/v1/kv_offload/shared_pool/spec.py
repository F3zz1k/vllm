# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""OffloadingSpec for the shared host-RAM KV pool.

Configure it through kv_connector_extra_config::

    {"spec_name": "SharedPoolOffloadingSpec",
     "shared_pool": {"socket": "/run/kvpool/kvpool.sock", "pool": "qwen3.8-27b"}}

The pool size is set on the server, not here (cpu_bytes_to_use is ignored).
Engines share a pool only when their layout key matches: the same model, KV
dtype, cache groups, block and chunk geometry and parallel layout, so a chunk
row written by any of them has the same meaning to all of them. The server
refuses an engine whose key differs from the pool's.

With "canonical_layout": true in kv_connector_extra_config, and every layer
certified portable, rows hold canonical (TP=1) pages and the parallel layout
leaves the key: engines of different TP sizes share one pool.
"""

import hashlib
import json
import os
from typing import Any

from typing_extensions import override

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import CanonicalKVCaches, OffloadingManager
from vllm.v1.kv_offload.config import OffloadingConfig
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion
from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.shared_pool import protocol as P
from vllm.v1.kv_offload.shared_pool.client import PoolClient, pool_info
from vllm.v1.kv_offload.shared_pool.manager import SharedPoolOffloadingManager

logger = init_logger(__name__)

DEFAULT_SOCKET = "/run/kvpool/kvpool.sock"
_LAYOUT_VERSION = 2


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
        self.timeout_sec = float(pool_cfg.get("timeout_sec", 600.0))
        self.layout_key = self._layout_key()
        self._region_path: str | None = None

    @override
    def _cpu_bytes_to_use(self) -> int:
        # Sized by the server once the row size is known (_ensure_pool).
        return 0

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

    def _ensure_pool(self) -> None:
        if self._region_path is not None:
            return
        info = pool_info(self.socket_path, self._pool_args(), self.timeout_sec)
        self._apply_pool_info(info)

    def _apply_pool_info(self, info: dict[str, Any]) -> None:
        self.num_chunks = int(info["num_chunks"])
        self._region_path = str(info["path"])

    @override
    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            client = PoolClient(self.socket_path, timeout=self.timeout_sec)
            reply = client.call(
                P.OP_ATTACH,
                {
                    **self._pool_args(),
                    "version": P.PROTOCOL_VERSION,
                    "name": self.config.engine_id,
                },
            )
            assert isinstance(reply, dict)
            self._apply_pool_info(reply)
            logger.info(
                "Attached to shared KV pool %s (%d chunks of %d bytes, layout %s) "
                "at %s as client %s",
                self.pool_name,
                self.num_chunks,
                self.kv_bytes_per_chunk,
                self.layout_key,
                self.socket_path,
                reply["client_id"],
            )
            self._manager = SharedPoolOffloadingManager(
                client,
                num_chunks=self.num_chunks,
                enable_events=self.kv_events_config.enable_kv_cache_events,
            )
        return self._manager

    @override
    def create_worker(self, kv_caches: CanonicalKVCaches) -> CPUOffloadingWorker:
        if not self._uses_shared_region():
            raise RuntimeError("The shared KV pool needs a CUDA or XPU platform")
        if self.config.canonical_layout:
            self._check_canonical_refs(kv_caches)
        self._ensure_pool()
        assert self._region_path is not None
        worker_index = self.config.parallel.rank
        mmap_region = SharedOffloadRegion(
            engine_id=self.config.engine_id,
            num_chunks=self.num_chunks,
            rank=0 if self.replicated_layout else worker_index,
            kv_bytes_per_chunk=self.kv_bytes_per_chunk,
            cpu_page_size=self.cpu_page_size_per_worker,
            external_path=self._region_path,
        )
        try:
            return CPUOffloadingWorker(
                kv_caches=kv_caches,
                blocks_per_chunk=self.blocks_per_chunk,
                num_cpu_chunks=self.num_chunks,
                mmap_region=mmap_region,
                canonical_layout=self.config.canonical_layout,
            )
        except Exception:
            mmap_region.cleanup()
            raise

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
