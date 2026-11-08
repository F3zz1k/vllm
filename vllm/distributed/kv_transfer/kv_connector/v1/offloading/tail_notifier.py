# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The connector's answer-tail store: a notifier, and nothing more.

Arena rows are allocated by the connector's offloading manager and bytes move
only in a worker's transfer job, so the engine core cannot store a tail: it
offers one here when it registers the finished turn, and the connector scheduler
turns the offer into a store on the step that follows. The queue is all the
state this object holds.

Restore is refused until a byte-level check says a tail comes back whole. A
tail that returns missing a group, or carrying a state that is not the one the
resume position needs, is read silently: the resume is wrong and nothing
reports it. So the load methods refuse by name, count it once per attempt, and
hand back nothing, which leaves no half tail to reach a request.
"""

from collections import Counter, OrderedDict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from vllm.v1.core.kv_cache_utils import BlockHash, BlockHashWithGroupId
from vllm.v1.kv_offload.answer_tail_store import (
    AnswerTailHandoff,
    AnswerTailHit,
    AnswerTailStore,
)

# Tails offered and not yet stored. One per finished turn that registered a
# tail, so at the queue's bound the oldest offers are dropped rather than
# crowding the scheduler's memory: a dropped offer costs one resume, not a
# conversation.
_MAX_OFFERS = 1 << 12


@dataclass
class TailOffer:
    """One tail waiting to become a store: the hand-off the core named, the
    GPU block per group the tail's state sits in, and how many steps the pool
    has refused it for."""

    handoff: AnswerTailHandoff
    block_ids: dict[int, int] = field(default_factory=dict)
    tries: int = 0


class AnswerTailNotifier(AnswerTailStore):
    """The queue the engine core hands a finished turn's tail to."""

    def __init__(self, counts: Counter, max_offers: int = _MAX_OFFERS) -> None:
        """``counts`` is the connector's partial-tail counter: the refusals this
        object makes on the load side have to reach /metrics through the same
        place every other tail event is reported from."""
        self.counts = counts
        self.pending: OrderedDict[tuple[str, bytes], TailOffer] = OrderedDict()
        self._max_offers = max_offers

    # --- the store side: record the offer ---

    def store_tail(
        self,
        handoff: AnswerTailHandoff,
        group_id: int,
        block_id: int,
    ) -> bool:
        """Queue one group's tail block. The tail is complete when every group
        the core registered for it has been offered, which is true by the time
        the connector drains the queue on its next step."""
        key = (handoff.request_id, bytes(handoff.tail_hash))
        offer = self.pending.get(key)
        if offer is None:
            if len(self.pending) >= self._max_offers:
                return False
            offer = self.pending[key] = TailOffer(handoff)
        else:
            self.pending.move_to_end(key)
        offer.block_ids[group_id] = block_id
        return True

    def take_offers(self) -> list[TailOffer]:
        """Every offer queued since the last step, oldest first."""
        offers = list(self.pending.values())
        self.pending.clear()
        return offers

    def requeue(self, offer: TailOffer) -> bool:
        """Put back an offer the pool refused, behind anything offered after
        it, for the store's bounded retries. False when the queue is full, so
        the caller gives up on the tail and counts it rather than losing it
        here."""
        if len(self.pending) >= self._max_offers:
            return False
        self.pending[(offer.handoff.request_id, bytes(offer.handoff.tail_hash))] = (
            offer
        )
        return True

    def forget_tail(self, tail_key: BlockHashWithGroupId) -> None:
        """Drop a still-queued offer whose tail the GPU has forgotten.

        The queue key names the request as well as the hash, and the core's
        forget hook has only the key, so this walks the queue: it holds at most
        ``_MAX_OFFERS`` tails and the hook is rare, a block losing its hashes.
        Every group of one tail shares the tail's hash, so one block losing it
        drops the whole offer: a tail missing a group is never stored.
        """
        wanted = bytes(tail_key)[:-4]
        for key in [k for k in self.pending if k[1] == wanted]:
            del self.pending[key]

    def has_offer_for(self, request_id: str) -> bool:
        """Whether a tail of this request is still waiting to be stored.

        The queue is the only index of what is waiting, so this walks it: at
        most ``_MAX_OFFERS`` keys, and it drains every step.
        """
        return any(request_id == key[0] for key in self.pending)

    # --- the load side: refused, by name ---

    def lookup_tail(
        self, tail_key: BlockHash, group_ids: Sequence[int]
    ) -> AnswerTailHit | None:
        """No hit while restore is closed. Miss here rather than hand back a
        hit that ``start_load`` would then refuse: a caller that believes the
        tail is coming will go on to reserve the tokens it covers."""
        return None

    def start_load(
        self, hit: AnswerTailHit, dest_block_ids: Mapping[int, Sequence[int]]
    ) -> bool:
        """Refuse the restore and count it.

        The count sits here rather than on the lookup because a lookup runs on
        every resume probe and would read as a gauge; one call here is one
        restore that was asked for and turned down. Nothing is copied, so
        nothing can be half copied: the resume falls back to computing the
        tokens the tail would have covered.
        """
        self.counts["tail_restore_refused"] += 1
        return False

    def wait(self, hit: AnswerTailHit) -> bool:
        """Nothing is ever in flight: no load starts while restore is closed."""
        return False

    # --- pins, reset, shutdown ---

    def pin(self, tail_key: BlockHashWithGroupId) -> None:
        """Pinning a tail's rows is the store route's, through the pool's own
        pin; the engine core holds no tail against eviction."""
        return None

    def release(self, tail_key: BlockHashWithGroupId) -> None:
        """The partner of ``pin``, refused for the same reason."""
        return None

    def reset(self) -> None:
        """The prefix cache is being wiped: nothing queued survives it."""
        self.pending.clear()

    def shutdown(self) -> None:
        self.pending.clear()
