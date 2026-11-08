# The answer-tail key and bookkeeping contract

Phase 2 step A, branch `answer-tail`: one section per item of `spec-phase2-A.md`, at this
branch's line numbers. No store hand-off here: step 3, blocked on the Flash-Next window.

## 1. Tail key naming, and who computes it on lookup

**Bytes and hash.** Unchanged bytes: `make_offload_key(tail_hash, group_idx)`, the hash
then the group index as four big-endian bytes. `BlockHashWithGroupId` and `OffloadKey` are
the same layout, so a tail key is already legal on the pool map and the arena reads it with
`_group(key)` and `key[:-4]` unchanged. The hash is `AnswerTail._finish_tail_hash`
(`vllm/v1/core/answer_tail.py:319`): the tail's token range, chained on the last
hash-boundary parent (`_finish_tail_parent`) and salted with `FINISH_TAIL_EXTRA_KEY`
(`answer_tail.py:49`), a position that is not a hash boundary, which is the whole
difficulty.

**Owner on lookup: the engine core.** `AnswerTail._find_finish_tail_hit`
(`answer_tail.py:329`) probes the tails `cache_finished` registered and returns the resume
length, one token short of the finished request's last token. The connector cannot own it,
for two reasons that do not overlap. It names keys only from `request.block_hashes`, which
the engine fills one `tokens_per_hash` step at a time and the connector walks one chunk at
a time (`_lookup_complete_chunks_impl`, `offloading/scheduler.py:2249`), so a position
between two steps has no entry to name. And it could not recompute one: the hash needs
`request.all_token_ids[start:end]`, the extra keys `generate_block_hash_extra_keys`
derives, and the parent chain, all of which belong to a finished request the connector has
already released.

**Why the external path is undisturbed.** The offloading connector leaves
`supports_divergent_local_hybrid_hits` False (`kv_connector/v1/base.py:188`), so the
scheduler uses `get_computed_blocks` and the core finds a local tail hit before the
connector is asked for anything. Today the connector only writes tail keys and never names
one on lookup, and that is what makes a pseudo-boundary hash safe on a map shared with
chunk keys. Step 3 must not lose this: naming a tail key on lookup would bring the
finish-tail hash and that request lifetime onto the connector.

decided

## 2. Band membership

**The test.** `_storable_free_block` (`offloading/scheduler.py:1583`) is the band's
membership test: hashed, chunk-aligned, free. A tail key is refused there. The GPU block a
tail key names is the tail block, and the pool map records it under a chunk-aligned token
count, so the alignment test passes on a tail key and the band would copy that block a
second time under the tail's own key, from no whole-chunk source, into rows no restore can
read. The refusal is by key identity, held in `_tail_run`, which `_tail_store_job`
(`scheduler.py:3200`) fills through `_note_tail_keys` (`scheduler.py:1389`).

**`_pool_has` and `_state_sets`.** A tail key belongs in `_pool_has` once its store
completes, by the ordinary job-completion path (`_note_in_pool`): the pool does hold those
rows. It never enters a boundary's state set. `_note_states` (`scheduler.py:1377`) sends a
tail key to `_pool_has` and no further, because `_state_sets` is what `_restore_chain` reads
to decide that a boundary can restore and a tail hash is not a boundary; recording one there
builds a restore chain on a position nothing can restore from.

**`_offer_keys`.** Never includes a tail key (`scheduler.py:1893`), for two independent
reasons. The band has no whole-chunk source for a tail, so it could offer only rows it
cannot write. And a whole-run store asks the pool to admit a run on the strength of the
keys offered with it (`prepare_store_run(whole=True)` and `Run.restorable_with`), where a
tail's pseudo-boundary completes a boundary the conversation's own keys do not, admitting a
run of nothing but tail rows. So a tail joins its run through its own store, alongside that
run's other keys in the same call, never as part of a whole-run offer. That is what "a tail
must not be offerable as if it were a whole run's state" means for this band, and a test
pins it by seeding a tail key into a run's chain.

**The eviction feed.** `_forget_in_pool` (`scheduler.py:1625`) drops the tail key from
`_tail_run` and from its run's list, and counts `tail_evicted`. A tail has no GPU hash to be
re-offered from, so the feed is the only way the band learns the arena dropped it; without
the counter a dropped tail is invisible. A feed overflow resets `_pool_has` and `_state_sets`
but not `_tail_run`: which keys are tails is a fact about the keys, not about the pool.

decided

## 3. Run tagging and lifetime

**A tail belongs to a run.** A run's tag is the request's last full-attention chunk key
(`_run_tag`, `scheduler.py:1153`); a session spans one run per finished turn.
`_note_tail_keys` records every stored tail key against the tag its request names at store
time, and `_run_tails` is the reverse. `_record_run` (`scheduler.py:1219`) moves that record
with the rows: a tail stored mid-decode was named after the attention key of its own step,
and the finish moves the request's rows into the run named after its last one (`merge_tag`,
`shared_pool/manager.py:285`). Without the re-tag the band forgets a tail the pool still
holds and reads it as a chunk key again. Only the band keeps run records, so only the band
re-tags: with the band off the tag is read by nothing.

**A tail leaves when its run leaves (D17).** `_drop_run` (`scheduler.py:1259`) forgets a
run's tail keys with its chain and its older restore points. No later run takes a tail
over: the band cannot write one, and a tail left behind by a dropped run is unreachable.

**A tail with no run.** A request that has keyed no full-attention chunk names no run, so
its tail is recorded without an owner and counted as `tail_run_missing`. Storing it under a
tag of its own would stand up a run of nothing but tail rows, the shape section 4 is about.
The counter is live now; refusing such a store is step 3's call, since it changes what the
phase-1 finish hand-off accepts.

**Arena accounting.** `Arena.refresh` (`shared_pool/server.py:1061`), reached from
`complete_store`, moves the whole run to the newest end of `Arena.order` and drops it from
`Arena.held`. A tail store into a session's run is therefore an accounting event for
everything already saved for that session, not only its own rows. That ordering and that
marking are what `_candidates` (`server.py:1148`) reads for the D4 fix, so one tail store
can move a state-only stub out of the eviction path for a whole conversation. Tested
against `Arena` directly: the D4 tests own the eviction side.

open, because the arena's run for a tail store is still the request's own (`prepare_store`
tags by `_request_tag`, `manager.py:228` and `manager.py:150`), so a tail reaches the
conversation run only through `merge_tag` at finish. Making the arena agree with the
band's tag at store time needs the widened hand-off, which is step 3.

## 4. The D9 leak guard

**The shape and the predicate.** A run whose keys are only a tail's rows holds recurrent
state and restores nothing, is the pool's only copy of that state, so the ordering that
protects a `state_only` run protects it and the spill path skips it, and nothing can
complete it from inside: both unrestorable and immortal. `Run.holds_state_and_restores_nothing`
(`shared_pool/server.py:267`) is the predicate: the class has state groups, the run holds
state rows, and no boundary it covers is complete. It is deliberately wider than
`state_only` (`server.py:261`), which ends the moment the band's first attention row lands
while the state still sits at a deeper boundary than that row reaches.
`Run.state_stub_protected(now)` adds `STATE_STUB_MAX_AGE_SEC` (`server.py:104`), from the
run's last store, so a session that keeps finishing turns keeps its state and a stalled one
gives its rows back.

**What is wired, and what is not.** The predicate and the bound are implemented and tested,
and P1b wired them: `Arena._deferred` (`server.py:1131`) is `Run.state_stub_protected`, and
`_candidates` (`server.py:1173`) and both walks of `admit_rows` (`server.py:1140`) ask it, so
a protected run leaves the arena last and its rows are not space a store may have. What step
3 still owes is the counter `state_stub_expired`, incremented where `_candidates` passes an
expired stub back into the reclaimable list. The band's live counters for this shape are
`tail_run_missing` (a tail with no run to belong to) and `tail_evicted` (a tail the arena
dropped).

**What the predicate cannot see.** A tail carrying an attention row as well as the state
rows completes a boundary at the tail's own hash, so the run reads as restorable and the
predicate is False. Tested, because it is why the guard cannot be this predicate alone: the
guard is that a tail never stands up a run of its own, which is section 2's offer rule and
section 3's `tail_run_missing`.

## 5. The invariant, in every new test

`sum(group_sizes) == len(block_ids) == len(keys)`: the per-group split of a store's keys,
the GPU blocks it reads, and the keys themselves are one size. This is the phase-1 P0 class,
where a store's split did not add up to its keys and arena rows were mis-paired with GPU
blocks. Every new test asserts it through a helper that reads a real spec, named
`_assert_one_row_per_key` in `test_write_back.py:970` for band and tail store jobs and
`assert_one_row_per_key` in `test_server.py:154` for the arena's own rows. A test about
bookkeeping rather than a transfer first stores a real finish tail
(`_finish_tail_store`, `test_write_back.py:980`) and asserts on that job, so the invariant
is checked on a store that happened, not on a table the test wrote.

decided
