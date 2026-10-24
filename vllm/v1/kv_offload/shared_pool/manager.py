# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side OffloadingManager backed by the shared KV pool server.

All index state lives in the server; this class translates the manager calls
into requests. A lookup HIT is held by the server for this client until the
manager lock region that made it closes, which is the window in which the
scheduler relies on the chunk staying present (see OffloadingManager.lock).

If the server goes away, the manager degrades instead of failing the engine:
every lookup misses and nothing is stored, so requests prefill from scratch.
"""

import time
from collections.abc import Collection, Iterable, Sequence
from types import TracebackType
from typing import Any

from typing_extensions import override

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    ConfigInfoMapping,
    LoadStoreSpec,
    LookupResult,
    Medium,
    OffloadingEvent,
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
)
from vllm.v1.kv_offload.cpu.common import (
    CPULoadStoreSpec,
    CPUOffloadingInfo,
    CPUOffloadingMetrics,
)
from vllm.v1.kv_offload.shared_pool import protocol as P
from vllm.v1.kv_offload.shared_pool.client import PoolClient, PoolServerError
from vllm.v1.metrics.cache_hit_source import CacheHitSource

logger = init_logger(__name__)

_LOOKUP_RESULTS = {
    P.MISS: LookupResult.MISS,
    P.HIT: LookupResult.HIT,
    P.HIT_PENDING: LookupResult.HIT_PENDING,
    P.RETRY: LookupResult.RETRY,
}
_STATS_INTERVAL_SEC = 1.0


class _Session:
    """The manager lock: closing it ends the server-side lookup holds."""

    def __init__(self, manager: "SharedPoolOffloadingManager") -> None:
        self._manager = manager

    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._manager._end_session()


class SharedPoolOffloadingManager(OffloadingManager):
    def __init__(
        self,
        client: PoolClient,
        num_chunks: int,
        enable_events: bool = False,
    ) -> None:
        self.medium: Medium = Medium.CPU
        self._client: PoolClient | None = client
        self._num_chunks = num_chunks
        self._config_info: ConfigInfoMapping = CPUOffloadingInfo(
            cpu_num_chunks=num_chunks
        ).as_config_info()
        self.events: list[OffloadingEvent] | None = [] if enable_events else None
        self._session = _Session(self)
        self._holds_open = False
        self._stats: dict[str, Any] | None = None
        self._stats_at = 0.0
        self._allocation_sizes: list[int] = []

    # --- transport ---

    def _call(self, op: str, args: object = None) -> object:
        if self._client is None:
            raise ConnectionError("shared KV pool server is unavailable")
        try:
            return self._client.call(op, args)
        except (OSError, EOFError, ValueError) as exc:
            logger.error(
                "Lost the shared KV pool server (%s: %s); offloading is disabled "
                "for this engine until it restarts",
                type(exc).__name__,
                exc,
            )
            self._client.close()
            self._client = None
            raise ConnectionError("shared KV pool server is unavailable") from exc

    @property
    def available(self) -> bool:
        return self._client is not None

    def _call_soft(self, op: str, args: object = None) -> object:
        """A call whose failure must not end the engine: a rejection by the
        server is logged and reported as ConnectionError to the caller, which
        then degrades the same way as for a lost server."""
        try:
            return self._call(op, args)
        except PoolServerError as exc:
            logger.error("Shared KV pool rejected %s: %s", op, exc)
            raise ConnectionError(str(exc)) from exc

    def _end_session(self) -> None:
        if not self._holds_open:
            return
        self._holds_open = False
        try:
            self._call_soft(P.OP_END_SESSION)
        except ConnectionError:
            pass

    # --- OffloadingManager interface ---

    @property
    @override
    def lock(self) -> _Session:
        return self._session

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        try:
            (result,) = self._call_soft(P.OP_LOOKUP, [bytes(key)])  # type: ignore[misc]
        except ConnectionError:
            return LookupResult.MISS
        if result == P.HIT:
            self._holds_open = True
        return _LOOKUP_RESULTS[result]

    def lookup_batch(
        self,
        keys: Sequence[OffloadKey],
        req_context: ReqContext,
        stop_at_miss: bool = False,
    ) -> list[LookupResult]:
        """Look up many keys in one request (see the scheduler's
        _lookup_results). With ``stop_at_miss`` the reply ends at the first
        MISS, so it can be shorter than ``keys``."""
        if not keys:
            return []
        op = P.OP_LOOKUP_PREFIX if stop_at_miss else P.OP_LOOKUP
        try:
            results = self._call_soft(op, [bytes(k) for k in keys])
        except ConnectionError:
            return [LookupResult.MISS] * (1 if stop_at_miss else len(keys))
        if P.HIT in results:  # type: ignore[operator]
            self._holds_open = True
        return [_LOOKUP_RESULTS[r] for r in results]  # type: ignore[union-attr]

    @override
    def get_load_source(
        self, key: OffloadKey, req_context: ReqContext
    ) -> CacheHitSource:
        return CacheHitSource.HOST

    @override
    def prepare_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> LoadStoreSpec:
        # A lost server here cannot be degraded: the scheduler was told these
        # chunks are present. The error ends the engine, as a bug would.
        chunk_ids = self._call(P.OP_PREPARE_LOAD, [bytes(k) for k in keys])
        return CPULoadStoreSpec(list(chunk_ids))  # type: ignore[arg-type]

    @override
    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        if not keys:
            return
        try:
            self._call_soft(P.OP_TOUCH, [bytes(k) for k in keys])
        except ConnectionError:
            pass

    @override
    def complete_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        try:
            self._call_soft(P.OP_COMPLETE_LOAD, [bytes(k) for k in keys])
        except ConnectionError:
            pass

    def prepare_store_run(
        self, keys: Collection[OffloadKey], tag: OffloadKey
    ) -> PrepareStoreOutput | None:
        """Store one conversation ("run") as a unit (exclusive pool mode)."""
        return self.prepare_store(keys, None, tag=tag)

    @override
    def prepare_store(
        self, keys: Collection[OffloadKey], req_context: ReqContext, tag: OffloadKey | None = None
    ) -> PrepareStoreOutput | None:
        try:
            if tag is None:
                result = self._call_soft(P.OP_PREPARE_STORE, [bytes(k) for k in keys])
            else:
                result = self._call_soft(
                    P.OP_PREPARE_STORE_TAGGED, [[bytes(k) for k in keys], bytes(tag)]
                )
        except ConnectionError:
            return None
        if result is None:
            return None
        new_keys, chunk_ids, evicted = result  # type: ignore[misc]
        if new_keys:
            self._allocation_sizes.append(len(new_keys))
        to_store = [OffloadKey(k) for k in new_keys]
        evicted_keys = [OffloadKey(k) for k in evicted]
        if evicted_keys and self.events is not None:
            self.events.append(
                OffloadingEvent(keys=evicted_keys, medium=self.medium, removed=True)
            )
        return PrepareStoreOutput(
            keys_to_store=to_store,
            store_spec=CPULoadStoreSpec(list(chunk_ids)),
            evicted_keys=evicted_keys,
        )

    @override
    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        keys = list(keys)
        try:
            self._call_soft(P.OP_COMPLETE_STORE, ([bytes(k) for k in keys], success))
        except ConnectionError:
            return
        if success and keys and self.events is not None:
            self.events.append(
                OffloadingEvent(keys=keys, medium=self.medium, removed=False)
            )

    @override
    def on_request_finished(self, req_context: ReqContext) -> None:
        return None

    @override
    def reset_cache(self) -> None:
        # The pool is shared, so a reset drops only what this engine holds:
        # its lookup holds, read references and unfinished stores. Chunks
        # other engines rely on stay.
        self._holds_open = False
        try:
            self._call_soft(P.OP_RELEASE_ALL)
        except ConnectionError:
            pass

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()

    @override
    def config_info(self) -> Sequence[ConfigInfoMapping]:
        return [self._config_info]

    def pool_stats(self, max_age: float = _STATS_INTERVAL_SEC) -> dict[str, Any] | None:
        now = time.monotonic()
        if self._stats is None or now - self._stats_at >= max_age:
            try:
                self._stats = self._call_soft(P.OP_STATS)  # type: ignore[assignment]
            except ConnectionError:
                self._stats = None
            self._stats_at = now
        return self._stats

    @override
    def get_stats(self) -> OffloadingConnectorStats | None:
        stats = OffloadingConnectorStats()
        for size in self._allocation_sizes:
            stats.observe_histogram(CPUOffloadingMetrics.CPU_ALLOCATION_SIZE, size)
        self._allocation_sizes.clear()
        pool = self.pool_stats()
        if pool is not None and pool["num_chunks"] > 0:
            n = pool["num_chunks"]
            usage = (pool["held"] + pool["write_pending"]) / n
            write_usage = pool["write_pending"] / n
            stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC, usage)
            stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_WRITE_USAGE_PERC, write_usage)
            stats.set_gauge(
                CPUOffloadingMetrics.CPU_CACHE_READ_USAGE_PERC,
                max(usage - write_usage, 0.0),
            )
        return stats

    @override
    def shutdown(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None
