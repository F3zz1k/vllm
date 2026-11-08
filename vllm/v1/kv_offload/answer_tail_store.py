# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The RAM side of the answer tail: the store the finished turns' tail blocks
are offered to, the hit a lookup returns, and the null default that keeps the
feature inert until a real store exists.

A tail key is lookup-only. Nothing here publishes a KV event, offloads a block
under a tail key or pins a GPU block."""

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from vllm.v1.core.kv_cache_utils import BlockHash, BlockHashWithGroupId


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


class AnswerTailStore(ABC):
    """The tier that keeps finished requests' tail blocks after the GPU drops
    them. One key per (tail, group): a tail is registered once per group."""

    @abstractmethod
    def store_tail(
        self,
        tail_key: BlockHashWithGroupId,
        parent: BlockHash | None,
        length: int,
        block_id: int,
    ) -> bool:
        """Offer one finished-tail GPU block to the tier. False if it refused."""

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
        tail_key: BlockHashWithGroupId,
        parent: BlockHash | None,
        length: int,
        block_id: int,
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
