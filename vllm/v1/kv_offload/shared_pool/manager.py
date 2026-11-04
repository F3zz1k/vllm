# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Scheduler-side OffloadingManager backed by the shared KV pool server.

All index state lives in the server; this class translates the manager calls
into requests. A lookup HIT is held by the server for this client until the
end of the scheduler step that made it (on_schedule_end), which covers the
window in which the scheduler relies on the chunk staying present: the
prepare_load of a hit runs in the same step.

Every store names a run: a compute-time store the request that makes it, a
write-back store the conversation it belongs to. When a request finishes, the
scheduler moves the request's run into its conversation (merge_tag), so the
server evicts whole conversations.

If the server goes away, the manager degrades instead of failing the engine:
every lookup misses and nothing is stored, so requests prefill from scratch.
While degraded it retries through ``reconnect`` (set by the spec), which only
succeeds when the server again serves the arena the workers mapped.
"""

import time
from collections.abc import Callable, Collection, Iterable, Sequence
from typing import Any

from typing_extensions import override

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    LookupResult,
    Medium,
    OffloadingEvent,
    OffloadingManager,
    OffloadKey,
    OffloadPolicy,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    ScheduleEndContext,
)
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec, CPUOffloadingMetrics
from vllm.v1.kv_offload.shared_pool import client as C
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
_STATE_INTERVAL_SEC = 1.0
_RECONNECT_INTERVAL_SEC = 30.0


class SharedPoolOffloadingManager(OffloadingManager):
    def __init__(
        self,
        client: PoolClient | None,
        pool: str,
        engine_tag: str,
        enable_events: bool = False,
        free_fraction: float | None = None,
        reconnect: Callable[[], PoolClient | None] | None = None,
    ) -> None:
        self.medium: Medium = Medium.CPU
        self._client = client
        self._pool = pool
        self._engine_tag = engine_tag
        self.events: list[OffloadingEvent] | None = [] if enable_events else None
        self._holds_open = False
        self._free_fraction = free_fraction
        self._state_at = time.monotonic()
        self._allocation_sizes: list[int] = []
        # Called while degraded; returns an attached client or None.
        self.reconnect = reconnect
        self._reconnect_at = time.monotonic()

    # --- transport ---

    def _call(self, op: str, args: object = None) -> object:
        if self._client is None:
            raise ConnectionError("shared KV pool server is unavailable")
        try:
            return self._client.call(op, args)
        except C.UNREACHABLE as exc:
            self.disable(f"lost the server ({type(exc).__name__}: {exc})")
            raise ConnectionError("shared KV pool server is unavailable") from exc

    @property
    def available(self) -> bool:
        return self._client is not None

    def disable(self, reason: str) -> None:
        """Run without the pool from now on (until a reconnect succeeds)."""
        if self._client is None:
            return
        logger.error(
            "Shared KV pool %s: %s; offloading is disabled for this engine",
            self._pool,
            reason,
        )
        self._client.close()
        self._client = None
        self._holds_open = False
        self._free_fraction = None
        self._reconnect_at = time.monotonic()

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

    def _maybe_reconnect(self) -> None:
        now = time.monotonic()
        if (
            self._client is not None
            or self.reconnect is None
            or now - self._reconnect_at < _RECONNECT_INTERVAL_SEC
        ):
            return
        self._reconnect_at = now
        client = self.reconnect()
        if client is not None:
            self._client = client
            self._state_at = 0.0
            logger.info("Reattached to shared KV pool %s", self._pool)

    def _request_tag(self, req_context: ReqContext) -> str:
        # Request ids are unique per engine only; conversation tags (bytes
        # keys) are shared by every engine of a layout on purpose.
        return f"{self._engine_tag}/{req_context.req_id}"

    # --- OffloadingManager interface ---

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        # Offer the chunks loaded from the pool too: the server already holds
        # them, so nothing is copied, and offering them makes them members of
        # this request's run, so a conversation's run always starts at its root.
        return RequestOffloadingContext(policy=OffloadPolicy.REQUEST_LEVEL)

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

    @override
    def prepare_store(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> PrepareStoreOutput | None:
        """A compute-time store, in the run of the request that makes it."""
        return self._prepare_store(keys, self._request_tag(req_context))

    def prepare_store_run(
        self, keys: Collection[OffloadKey], tag: OffloadKey, whole: bool = False
    ) -> PrepareStoreOutput | None:
        """Store keys in the run of ``tag``, all or nothing. ``whole`` marks a
        whole conversation, which the server refuses when the run could not
        restore anything (no recurrent state at any of its boundaries)."""
        return self._prepare_store(keys, bytes(tag), whole)

    def _prepare_store(
        self, keys: Collection[OffloadKey], tag: str | bytes, whole: bool = False
    ) -> PrepareStoreOutput | None:
        args: list[Any] = [[bytes(k) for k in keys], tag]
        if whole:
            args.append(True)
        try:
            result = self._call_soft(P.OP_PREPARE_STORE_TAGGED, args)
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

    def merge_tag(self, req_context: ReqContext, tag: OffloadKey) -> None:
        """Move everything the finished request stored into the run of
        conversation ``tag``."""
        try:
            self._call_soft(
                P.OP_MERGE_TAG, (self._request_tag(req_context), bytes(tag))
            )
        except ConnectionError:
            pass

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
    def on_schedule_end(self, context: ScheduleEndContext) -> None:
        self._end_session()
        self._maybe_reconnect()

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

    def pool_free_fraction(
        self, max_age: float = _STATE_INTERVAL_SEC
    ) -> float | None:
        """This class's share of rows a store can take without evicting a live
        run of its own (STATE admit_fraction, else free_fraction), at most one
        call per ``max_age``; None when unknown."""
        if self._client is None:
            return None
        now = time.monotonic()
        if now - self._state_at >= max_age:
            self._state_at = now
            try:
                state: Any = self._call_soft(P.OP_STATE)
                cls_state = state["classes"][self._pool]
                # admit_fraction also counts dead runs and other classes'
                # idle runs (a joint arena full of another model's cold runs
                # is not full for this one); older servers send only
                # free_fraction.
                self._free_fraction = float(
                    cls_state.get("admit_fraction", cls_state["free_fraction"])
                )
            except ConnectionError:
                self._free_fraction = None
            except (KeyError, TypeError, ValueError):
                logger.warning_once(
                    "Shared KV pool STATE names no free fraction for class %s",
                    self._pool,
                )
                self._free_fraction = None
        return self._free_fraction

    def pool_admit_rows(self) -> int | None:
        """Rows (chunks) of this class a store can take now without evicting
        a live run of its own, from a fresh STATE; None when unknown."""
        if self._client is None:
            return None
        try:
            state: Any = self._call_soft(P.OP_STATE)
            cls_state = state["classes"][self._pool]
            fraction = float(
                cls_state.get("admit_fraction", cls_state["free_fraction"])
            )
            return int(fraction * int(cls_state["rows_total"]))
        except ConnectionError:
            return None
        except (KeyError, TypeError, ValueError):
            return None

    @override
    def get_stats(self) -> OffloadingConnectorStats | None:
        stats = OffloadingConnectorStats()
        for size in self._allocation_sizes:
            stats.observe_histogram(CPUOffloadingMetrics.CPU_ALLOCATION_SIZE, size)
        self._allocation_sizes.clear()
        return stats

    @override
    def shutdown(self) -> None:
        self.reconnect = None
        if self._client is not None:
            self._client.close()
            self._client = None
