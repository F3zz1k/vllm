# The answer-tail key and bookkeeping contract

Phase 2 step A, branch `answer-tail`: one section per item of `spec-phase2-A.md`, at this
branch's line numbers. Sections 1 to 5 predate the store hand-off, section 6 is the rows a
tail costs, and section 7 is the block a state row may be copied from.

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
run of nothing but tail rows. So a tail joins its run through its own store, never as part
of a whole-run offer, and its own store offers only the tail's keys (section 3). That is what
"a tail must not be offerable as if it were a whole run's state" means for this band, and a
test pins it by seeding a tail key into a run's chain.

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
Step 3 decided the refusal: the store route turns such an offer down before it asks the pool
for anything, so the counter now counts a tail that was not stored as well as one that was.
The phase-1 finish hand-off keeps its own record, because it stores the tail under a
hash-boundary key rather than the tail's own.

**Arena accounting.** `Arena.refresh` (`shared_pool/server.py:1061`), reached from
`complete_store`, moves the whole run to the newest end of `Arena.order` and drops it from
`Arena.held`. A tail store into a session's run is therefore an accounting event for
everything already saved for that session, not only its own rows. That ordering and that
marking are what `_candidates` (`server.py:1148`) reads for the D4 fix, so one tail store
can move a state-only stub out of the eviction path for a whole conversation. Tested
against `Arena` directly: the D4 tests own the eviction side.

**The store route names the run at store time.** The widened hand-off (step 3) stores a
tail through `prepare_store_run(keys, tag)` with the same tag `_note_tail_keys` records, so
the arena's run and the band's record agree the moment the rows are taken rather than
waiting for `merge_tag` at the request's release. The tail's keys are the only ones the call
offers: a store takes a row for every offered key the pool lacks, and a tail job writes
only its tail's rows, so the run's other members join through their own stores. The tail
still reaches the conversation's run because it is offered under the run's tag.

**The pin is short, and the pool does not release it for us.** The arena's only pin a
client can reach is `OP_PIN`/`OP_UNPIN` naming the class (`protocol.py:63`,
`server.py:2756`); `held=True` on a store is the opposite, it tells the arena the GPU has a
copy so the run may go first. A pin is class state, not client state: `release_client`
(`server.py:1553`), reached from `_drop` on a detach and from `OP_RELEASE_ALL`, drops a
client's loads, writes and holds and never touches `cls.pins`, so the "released N
references" in the detach log counts none of them. A pin therefore outlives the client that
took it, which is why the tail store pins for its copy only and unpins on every path that
ends the job (completion, cache reset, shutdown), counted as `tail_pinned` and
`tail_unpinned` with `tail_pin_leaked` for a release with nothing held. A pin held for a
tail's whole life would be safe only if the pool released pins on detach, which it does not.

decided

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

## 6. The rows a tail costs

**The count, and where it comes from.** A tail is stored as the finished turn's state at
the tail, and one group's state there is not always one block: a resume reads a windowed
group's whole window, so that group costs `ceil(window / block_size)` arena rows and every
other group one. The number is `GroupOffloadConfig.tail_rows`
(`offloading/scheduler.py:206`), read once from the real `kv_cache_config` spec by
`tail_rows_for_group` (`kv_offload/answer_tail_store.py:31`), which takes the window from
the spec object that carries it and never from a model's name. A spec that does not say
yields None, and the store refuses the tail with `window_rows_unresolved` rather than
storing it at a shape guessed from another layout.

**Only the last row of a group carries the tail key.** The rows before it are whole blocks
ending at hash-aligned positions, so each is stored under the key its own block owns at
that position: the key the core registers the same block under for the GPU registry
(`answer_tail.py:318-330`) and the key `_build_boundary_attention_jobs` writes for the same
block at an aligned boundary. A new key form was rejected (D10), and sharing one namespace
is what makes the pool's own dedup the tail's dedup: a window row the band already wrote is
not taken twice. Only the tail's own keys reach `_note_tail_keys`, because a window row's
key is a chunk key and the band must stay free to copy that block later.

**Production behaviour for a sliding-window layout is unchanged.** A layout with a
`SlidingWindowSpec` group still fails the sliding-window clause of `supports_partial_tail`
(`offloading/scheduler.py:452-456`), and `_build_answer_tail_store_jobs` refuses every offer
with `layout_unsupported` before it names a key (`offloading/scheduler.py:3610`). Lifting
that clause is its own commit (D12); what this section fixes is the arithmetic that commit
will find in place.

**What the gpu-gate finding was, and what it was not.** The finding at `answer_tail.py:291`
is about the GPU-side registration: the core registers one row at the tail hash per group,
which is right for the GPU, because `_finish_tail_blocks` reads the window's earlier blocks
from the prefix cache under their own keys. The store route never stored a windowed tail at
the wrong row count, because with the clause closed it stores no windowed tail at all. The
offer now carries the ordered list of blocks per group (`TailOffer.block_ids`), and the
store allocates, pairs and pins that many rows, so the clause's commit inherits a store that
counts correctly instead of one that has to be fixed alongside it.

decided

## 7. The block a state row may be copied from

**The rule.** A recurrent state row is stored only from the block the finish checkpoint
wrote. The worker's FINALIZE pass (`postprocess_mamba_fused_kernel`, FINALIZE=True) copies
the state after the exact last computed token into that token's block in the step that
detected the finish, before any later forward, and that is the only place a sub-block
position ever becomes bytes. The step route therefore stores no state row at all: every
tail it is handed is refused whole with the counter `live_state_source`
(`offloading/scheduler.py`, `_build_partial_tail_store_jobs`).

**Why the step route has no correct source.** Inside a block a recurrent group has one
state slot, and it holds the position the last forward reached -- never a position between
two forwards. The boundary a partial tail names is a *hashing* position: the manager
re-keys at `get_mamba_prefill_checkpoint_position(prompt, hash_block_size, drop_eagle)`,
which sits below the end of the chunk that computed it. So the slot the manager registers
as the producer's source holds the state at the chunk end, and its copy-on-write block --
which is what the hand-off actually carries, the request's table being append-only -- holds
the same bytes however early the copy is taken. Being durable is not being the boundary.

**What the two GPU proofs decided.** Functional-test-2 arm A poisoned rows 4, 5 and 6 under
key `80facefd82a3de48` with groups 0, 1 and 2, at boundary 23808 of a 23909-token prompt.
The first fix refused a hand-off that named the request's own table row at
`boundary // block_size`, and that premise is now retracted too: functional-test-3 ran the
fix and `live_state_source` fired zero times while the same three keys were written again
and the same degenerate reply came back (`ids_sha 52fc9fb9`, 2 distinct ids in 640 tokens).
The hand-off had named the manager's copy-on-write block, which is in no position of the
request's table, so the identity check passed it. The pool log puts that store 2.4 s after
the aligned store for the same request at 20800 and 13.7 s before the request finished
generating, so the copy was taken during the producer's own decode, from a slot 101 tokens
past the boundary it was keyed at.

**What stays storable.** The finish route: `finalize_partial_tail_offload`
(`single_type_kv_cache_manager.py:1931`) returns a source only when nothing is in flight
and the request computed exactly up to the boundary, which is exactly when that block is
the FINALIZE target. The aligned route at exact block boundaries is unaffected, because a
block boundary is a position a forward ends at by construction, and full-attention rows
need no capture at all: their KV for those tokens already sits in the block. The answer
tail keeps its own route.

**What the refusal costs.** Nothing selectable. The probe in `_lookup` names the boundary
key for every tail group and misses without the state row, so no partial restore is
offered and no load asks for a row the store refused. With the flag off a refusal is
reported nowhere, and now stores nothing either: a flag-off engine writes no tail rows at
all, which is the point, since the rows it used to write there were the poison.

**The tests.** `test_partial_tail_step_path_refuses_the_state_row_the_request_holds` and
`test_partial_tail_step_path_refuses_a_copy_of_that_row_made_mid_run` are the two hand-off
shapes the rig produced, and both are refused.
`test_partial_tail_a_step_tail_leaves_the_restore_no_state_row` and
`test_partial_tail_a_finished_tail_is_stored_and_the_restore_loads_it` run store and
restore against one pool, one conversation asking twice, so the refused key is absent from
a later load and the finish route's key is present.

decided
## 8. The marker positive control, and the arena it needed

Phase 2 commit 5 was to be the per-group marker positive control that `gpu-gate-1.md`
section 7.2 asks for: seed each group's tail block with a distinguishable payload, run the
store, read the rows back, and require group g's row to carry g's payload byte for byte. It
exists now, at `b9229c8512`, as the last section of
`tests/v1/kv_connector/unit/offloading_connector/test_write_back.py`: one mmap-backed arena,
the positive control and three injected faults, with the review's P2 batch at `4c780f241d`
adding the fourth fault and pinning the boundaries the harness had assumed. Why it could
not be written then is worth keeping, because it decides the shape it has.

**What the control has to prove.** Not that a row round-trips, which the pool already
proves, but that the connector paired the right source block with the right destination row
for each group. A permutation of the groups that is applied to both the store and the load
is invisible to every assertion the connector tests make today: the key set matches, the
row count matches, `group_sizes` sums to the number of keys, and the tail restores a state
that is one group's state standing in for another's. Nothing downstream can see it, because
a recurrent state carries no group label. That is the residual the note says byte equality
never closed, and it is why the control must be payload-level rather than key-level.

**What it proves.** Each group's payload is seeded into the GPU block the core's offer
names, stored through the job `_build_answer_tail_store_jobs` builds, and read back through
the job `update_state_after_alloc` builds, into a destination buffer the harness holds. Both
legs are the product's own jobs, the arena answers for every row, and rows are addressed
through the server's own `LayoutClass.offset`, so what is proved is which block id is paired
with which arena row for every group, in bytes. That covers the residual above, a
permutation applied to both legs alike, because the expectation is anchored to the arena's
key-to-row index and not to a comparison between the legs: a permutation is wrong on the
store side before it can be consistent anywhere.

**What it does not reach.** The GPU tensors and the fragment walk under the two specs
(`gpu_worker.py:700-770`) are stood in for by `_StandInBlocks` and
`_copy_as_the_worker_would`, a CPU transcription that pairs by position, so the control says
nothing about the CUDA copy engine, the per-layer page layout, or the bytes inside a row.
Three limits follow from that and are disclosed in the test's own section text rather than
argued away. The drafter exemption leaves this fixture two tail groups, so the permutation
the control can express is a swap of two, and the faults injected differ by path (a store
swap, a restore swap, a row left unwritten, a tail row copied twice into a chunk block);
a layout with three tail groups would widen that space and has no control here. A windowed
group costs the window's worth of rows and the fence refuses a resume whose tail holds more
than one row for a group, so no multi-row restore control is reachable: one row per group is
all a resume has a destination for. And every group's row is the same harness-chosen
`ROW_BYTES`, while a `group_sizes` that preserves the flat order pairs the same rows in this
transcription and in the worker because both partition on that one field, so a wrong
per-group row size and that whole `group_sizes` class are blind here by construction.

**Why the connector's suite cannot host it.** The byte copy lives in a worker between a GPU
block and an arena row, and `tests/v1/kv_connector/unit/offloading_connector/` runs against
`MockOffloadingSpec`. Its manager returns rows from `_cpu_store_output`, which hands back
`CPULoadStoreSpec(range(len(keys)))`: row numbers, not a buffer. There is no memory on
either side of that job to seed and read, so a test there can assert the pairing of a key to
a row number, which is what `_assert_one_row_per_key` already does, and cannot assert where
a byte went. Writing a payload into a mock is the tautology the note warns about.

**What it needed.** The shared arena has the bytes. `tests/v1/kv_offload/shared_pool/`
builds an `Arena` over a real `mmap` and its tests already write and read payload through
`arena.mm[row * ROW:(row + 1) * ROW]`, so the destination side is genuinely there. The
missing half is the source: the arena is given rows to allocate, and the bytes that reach
them come from a GPU block the CPU tests have no equivalent of. The control therefore needed
a second buffer standing in for GPU memory, a copy that walks the *job's own* source and
destination specs rather than the test's own knowledge of the layout, and the tail keys of a
store the connector built, so that the pairing under test is the connector's and not one the
test wrote down. Those three pieces did not exist together anywhere then, and the section's
last two paragraphs are the shape they took.

**The shape that was built.** One arena over a real `mmap` with a class whose row
size matches the group stride, a byte array standing in for the GPU blocks, a tail store
built by `OffloadingConnectorScheduler._answer_tail_store_job` rather than by hand, a copy
loop that reads `job.src_spec.block_ids` and `job.dst_spec.block_ids` in the order the
transfer layer will read them, and one assertion per group: the row the arena now holds under
`make_offload_key(tail_hash, g)` carries the payload that was written into group g's block,
and no payload that belongs to any other group. The failure it must catch is the one the
existing suite cannot express: every count right, every key present, and group 1's state
sitting in group 2's row.

**What is not a substitute.** Byte equality between two runs, which `gpu-gate-1.md` section
7.1 records as unproven for a finish checkpoint and section 7.2 replaces; key-level pairing
assertions, which a common permutation passes; and a per-group row count, which is section 6
and says nothing about which row is whose.
