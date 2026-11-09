# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""The connector's answer-tail store: a notifier, and nothing more.

Arena rows are allocated by the connector's offloading manager and bytes move
only in a worker's transfer job, so the engine core cannot store a tail: it
offers one here when it registers the finished turn, and the connector scheduler
turns the offer into a store on the step that follows. The queue is all the
state this object holds.

The load side is a fence the connector holds, not one held here. A tail that
returns missing a group, or carrying a state that is not the one the resume
position needs, is read silently: the resume is wrong and nothing reports it. So
the tier answers only whether the rows are here, the connector decides over every
row of a tail or none of them, and it decides before the scheduler allocates the
destination blocks. The rows then travel as the resumed request's own load, so
there is one wait surface, the connector's job machinery, and no second gate that
could report a tail landed that has not.
"""

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
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

# Restores already committed, remembered so a lookup retried over blocks that
# are already being filled joins the load in flight instead of counting and
# issuing the same rows again. Bounded the same way the offer queue is: what
# losing one costs is a duplicate count, never a wrong byte.
_MAX_RESTORES = 1 << 12


@dataclass
class TailOffer:
    """One tail waiting to become a store: the hand-off the core named, the GPU
    blocks the tail's state sits in, oldest row first and one list per group,
    and how many steps the pool has refused it for."""

    handoff: AnswerTailHandoff
    block_ids: dict[int, list[int]] = field(default_factory=dict)
    tries: int = 0


class AnswerTailNotifier(AnswerTailStore):
    """The queue the engine core hands a finished turn's tail to."""

    def __init__(
        self,
        report: Callable[[str, int], bool],
        on_reset: Callable[[], None],
        holds: Callable[[BlockHash, Sequence[int]], bool],
        generation: Callable[[], int],
        tails: Callable[
            [Sequence[BlockHash | None]], Mapping[BlockHash | None, tuple[int, ...]]
        ],
        max_offers: int = _MAX_OFFERS,
    ) -> None:
        """``report`` is the connector's tail reporter: the refusals this object
        makes on the load side have to reach /metrics through the same predicate
        every other tail event is counted by, including the rule that a flag-off
        deployment reports nothing at all.

        ``on_reset`` is the connector's own reset step. This queue is not the
        only thing keyed on the prefix cache: the connector remembers which keys
        are tails, which run owns them and which rows it believes the arena
        holds, and a reset of that cache takes those facts with it whether or
        not the queue happened to hold anything.

        ``holds`` answers whether the arena holds every row of a tail, which is
        what a lookup is asking: the rows are the connector's to know, and this
        object only holds the offers that have not become stores yet.

        ``generation`` is the connector's prefix-cache counter. A hit is a
        promise about rows in a particular cache, and a reset can happen without
        this object being asked, so the number travels on the answer and the
        connector that reads it is the one that turns down a stale record.

        ``tails`` answers which lengths the tier holds under each parent the
        probe names. A tail key cannot be enumerated by the side that did not
        finish the turn, so this is the only way a brand-new engine finds a tail
        its own registry never saw.
        """
        self.report = report
        self.on_reset = on_reset
        self.holds = holds
        self.generation = generation
        self.tails = tails
        self.pending: OrderedDict[tuple[str, bytes], TailOffer] = OrderedDict()
        self._restored: OrderedDict[tuple[bytes, tuple[tuple[int, ...], ...]], bool] = (
            OrderedDict()
        )
        self._max_offers = max_offers

    # --- the store side: record the offer ---

    def store_tail(
        self,
        handoff: AnswerTailHandoff,
        group_id: int,
        block_ids: Sequence[int],
    ) -> bool:
        """Queue one group's tail rows, oldest block first. The tail is
        complete when every group the core registered for it has been offered,
        which is true by the time the connector drains the queue on its next
        step."""
        key = (handoff.request_id, bytes(handoff.tail_hash))
        offer = self.pending.get(key)
        if offer is None:
            if len(self.pending) >= self._max_offers:
                return False
            offer = self.pending[key] = TailOffer(handoff)
        else:
            self.pending.move_to_end(key)
        offer.block_ids[group_id] = list(block_ids)
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
        Every group of one tail shares the tail's hash, and every row of one
        group shares that group's tail key, so one block losing it drops the
        whole offer: a tail missing a group, or a group missing a row, is never
        stored. The rows a windowed group takes before its tail block are keyed
        by their own chunk hashes, which are not tail keys and so never reach
        here; a store that has to wait for room re-reads those blocks instead
        (see the connector's ``_tail_rows_on_gpu``).
        """
        wanted = bytes(tail_key)[:-4]
        for key in [k for k in self.pending if k[1] == wanted]:
            del self.pending[key]

    def has_offer_for(self, request_id: str) -> bool:
        """Whether a tail of this request is still waiting to be stored.

        The connector holds a finished request's offload state while one of its
        offers is queued, because the retry builds its store into the run that
        state names. The queue is the only index of what is waiting, so this
        walks it: at most ``_MAX_OFFERS`` keys, and it drains every step.
        """
        return any(request_id == key[0] for key in self.pending)

    # --- the load side: the tier answers, the connector fences ---

    def answers_lookups(self) -> bool:
        """The connector can put a probe to the arena, so the core walks the
        candidates a turn finished elsewhere would have left behind."""
        return True

    def tail_lengths(
        self, parents: Sequence[BlockHash | None]
    ) -> Mapping[BlockHash | None, tuple[int, ...]]:
        """Which of these parents the tier holds a tail under, and how long.

        The record is the pool's, and this object only forwards: a tail is
        stored under a key the finished turn's tokens made, so the lengths have
        to come from the side that kept the rows.
        """
        return self.tails(parents)

    def lookup_tail(
        self, tail_key: BlockHash, group_ids: Sequence[int]
    ) -> AnswerTailHit | None:
        """The tail the arena holds rows for, whole or not at all.

        The hit carries no resume position: the tail was stored under a key the
        finished turn's tokens made, and where a resume restarts from it is the
        core's to know, because the core is the one that repeated the turn. The
        caller reads the position off the tail it probed and leaves this hit as
        the answer to "do you have it", which is all the connector can say: the
        rows are named by the key, and their count is the layout's.

        Nothing is counted here. A lookup runs on every probe of every request,
        so it reads as a gauge; what a miss costs is a tail that recomputes.
        """
        if not self.holds(tail_key, group_ids):
            return None
        return AnswerTailHit(
            tail_key=tail_key,
            resume_token=0,
            group_block_ids={},
            num_tokens=0,
            generation=self.generation(),
        )

    def start_load(
        self, hit: AnswerTailHit, dest_block_ids: Mapping[int, Sequence[int]]
    ) -> bool:
        """Commit a restore the connector has already fenced and built.

        The copy is the resumed request's own load job, so this records that the
        rows are on their way to the blocks they land on and counts the restore
        once. A second call for the same rows into the same blocks is the same
        load asked twice, which happens when a lookup is retried, so it joins
        what is in flight and counts nothing: two counts for one copy would
        report a tail restored that was not.

        False is still the answer for a restore this object will not stand
        behind, and nothing here copies: a tail that cannot come back whole is
        refused by the connector before it allocates, never half loaded.
        """
        destination = tuple(
            (group_id, tuple(block_ids))
            for group_id, block_ids in sorted(dest_block_ids.items())
        )
        if not destination:
            # No block to land on is no restore. The caller fenced a resume it
            # could not fill, and counting one here would report a tail restored
            # that no request was given.
            return False
        key = (bytes(hit.tail_key), destination)
        if key in self._restored:
            return True
        # The reporter is the connector's one flag-off predicate, so a tier that
        # is not supposed to be serving says so here as well as in its counters.
        if not self.report("tail_restored", 1):
            return False
        if len(self._restored) >= _MAX_RESTORES:
            self._restored.popitem(last=False)
        self._restored[key] = True
        return True

    def wait(self, hit: AnswerTailHit) -> bool:
        """Nothing is ever in flight here: the copy is the request's own load
        job, and the connector's job machinery is the one wait surface."""
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
        """The prefix cache is being wiped: nothing queued survives it, and the
        connector's bookkeeping over that cache goes in the same step, from the
        same call, whichever way the reset was asked for."""
        self.pending.clear()
        self._restored.clear()
        self.on_reset()

    def discard_offers(self) -> None:
        """Drop the queue and nothing else. The connector's own view of the
        cache is not in question here, so a caller that cannot serve the offers
        it holds still believes what it believed about the pool."""
        self.pending.clear()

    def shutdown(self) -> None:
        self.pending.clear()
