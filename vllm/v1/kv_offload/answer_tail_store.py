# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The RAM side of the answer tail: the seam a finished turn's tail blocks are
offered through, the hit a lookup returns, and the null default that keeps the
feature inert until a real store exists.

The store is a notifier. Arena rows are allocated by the connector's offloading
manager and bytes move only in a worker's transfer job, so a store called from
the engine core can neither reserve a row nor take lane time: it records the
offer and the connector turns it into a store.

A tail key is lookup-only. Nothing here publishes a KV event, offloads a block
under a tail key or pins a GPU block."""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from vllm.utils.math_utils import cdiv
from vllm.v1.core.kv_cache_utils import BlockHash, BlockHashWithGroupId
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    CircularBufferSpec,
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
    SlidingWindowSpec,
)


def tail_rows_for_group(
    kv_cache_spec: KVCacheSpec, tokens_per_block: int
) -> int | None:
    """How many arena rows one group's tail state costs.

    A group whose state at a position is one block costs one row. A windowed
    group's state at a position is the window, so it costs the blocks the
    window spans: ``window / block_size`` rounded up. None when the spec does
    not say, which is a geometry the store cannot size and must refuse.
    """
    if isinstance(kv_cache_spec, SlidingWindowSpec):
        window = kv_cache_spec.sliding_window
    elif isinstance(kv_cache_spec, ChunkedLocalAttentionSpec):
        window = kv_cache_spec.attention_chunk_size
    elif isinstance(kv_cache_spec, (FullAttentionSpec, MambaSpec, CircularBufferSpec)):
        return 1
    else:
        return None
    if window <= 0 or tokens_per_block <= 0:
        return None
    return cdiv(window, tokens_per_block)


@dataclass(frozen=True)
class AnswerTailHandoff:
    """One finished request's tail, as the engine core names it.

    The hash is the tail's own, chained on the last hash-boundary parent and
    salted so that it never equals a full-unit key. Only the core can compute
    it: it needs the finished request's tokens, its extra keys and its parent
    chain, none of which survive the request's release.
    """

    request_id: str
    """The request the tail was registered for, so the connector can find its
    offload state, its run and the blocks to fence."""

    tail_hash: BlockHash
    """The tail's hash, the same one every group's key is made from."""

    parent: BlockHash | None
    """The hash the tail is chained on, None for a tail that starts the
    conversation."""

    length: int
    """Tokens the tail covers, the last token included."""

    resume_token: int
    """Where a resume restarts: one token before the finished request's last
    token, which is the position the tail's blocks hold the state at."""


@dataclass(frozen=True)
class AnswerTailHit:
    """A tail found in the RAM tier: GPU-bound block ids to restore per group."""

    tail_key: BlockHash
    """The tail hash without its marker, as the hit probe computes it."""

    resume_token: int
    """Where the resumed request starts: one token before the finished
    request's last token."""

    group_block_ids: dict[int, list[int]]
    """The RAM tier's block ids per KV cache group."""

    num_tokens: int
    """The finished request's token count, the tail's own length included."""

    generation: int = 0
    """Which prefix cache the tier was in when it answered. A record of a tail
    names keys from the cache that was here when it was made, and that cache can
    be wiped without the tier being asked, so a record from an older cache names
    keys nothing can reach again. The tier stamps its own number because only it
    knows, and the connector that reads the record is the one that fences it."""


class AnswerTailStore(ABC):
    """The tier that keeps finished requests' tail blocks after the GPU drops
    them. One key per (tail, group): a tail is offered once per group."""

    @abstractmethod
    def store_tail(
        self,
        handoff: AnswerTailHandoff,
        group_id: int,
        block_ids: Sequence[int],
    ) -> bool:
        """Offer one group's tail rows to the tier, oldest block first. False
        if it refused.

        The list is the group's state at the tail: one block for a group whose
        state is one block, the window's worth of blocks for a windowed group
        (``tail_rows_for_group``). A refusal costs the registration nothing:
        the GPU-side tail key stands on its own, and a tier that cannot take
        the tail today loses only the resume the RAM tier would have served.
        """

    @abstractmethod
    def lookup_tail(
        self, tail_key: BlockHash, group_ids: Sequence[int]
    ) -> AnswerTailHit | None:
        """The tail held by every one of ``group_ids``, or None. Lookup-only:
        no KV event, no pin."""

    @abstractmethod
    def start_load(
        self, hit: AnswerTailHit, dest_block_ids: Mapping[int, Sequence[int]]
    ) -> bool:
        """Copy a hit into the GPU blocks the resumed request holds. False if
        the tier cannot serve it."""

    @abstractmethod
    def wait(self, hit: AnswerTailHit) -> bool:
        """Whether a started load has landed. False while it is in flight."""

    @abstractmethod
    def forget_tail(self, tail_key: BlockHashWithGroupId) -> None:
        """Drop the RAM copy when the GPU registry forgets the key."""

    @abstractmethod
    def pin(self, tail_key: BlockHashWithGroupId) -> None:
        """Hold a tail against eviction while a store or load is in flight."""

    @abstractmethod
    def release(self, tail_key: BlockHashWithGroupId) -> None:
        """Undo a pin."""

    @abstractmethod
    def reset(self) -> None:
        """Wipe the tier together with the prefix cache."""

    @abstractmethod
    def shutdown(self) -> None:
        """Give everything back and stop."""


class NullAnswerTailStore(AnswerTailStore):
    """The default: no RAM tier, so every offer is refused, every lookup misses
    and every other call does nothing."""

    def store_tail(
        self,
        handoff: AnswerTailHandoff,
        group_id: int,
        block_ids: Sequence[int],
    ) -> bool:
        return False

    def lookup_tail(
        self, tail_key: BlockHash, group_ids: Sequence[int]
    ) -> AnswerTailHit | None:
        return None

    def start_load(
        self, hit: AnswerTailHit, dest_block_ids: Mapping[int, Sequence[int]]
    ) -> bool:
        return False

    def wait(self, hit: AnswerTailHit) -> bool:
        return False

    def forget_tail(self, tail_key: BlockHashWithGroupId) -> None:
        return None

    def pin(self, tail_key: BlockHashWithGroupId) -> None:
        return None

    def release(self, tail_key: BlockHashWithGroupId) -> None:
        return None

    def reset(self) -> None:
        return None

    def shutdown(self) -> None:
        return None
