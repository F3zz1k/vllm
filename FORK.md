# This fork

vLLM for a rig of six Intel Arc Pro B70 (XPU) cards. Base: the Lumnus B70 fork, tag `v0.32.0-b70.5` (vLLM v0.32.0 plus
B70/XPU fixes). Branch `answer-tail-2` adds 66 commits on top (about 8.4k lines of code, 4.5k of tests). Production
image: `vllm-xpu:b70-skv-<commit>`, tagged `vllm-<commit>` here. Models served: Qwen3.8-27B (FP8, hybrid Gated
DeltaNet + attention, TP=2 and TP=4) and Qwen3.8-Flash-Next (hybrid with QSA sparse attention, TP=4 + EP).

## What it adds

**Host-RAM KV cache shared by every engine** (`vllm/v1/kv_offload/shared_pool/`, connector `OffloadingConnector` with
`spec_name: SharedPoolOffloadingSpec`)
- A pool server (`server.py`, run as its own service) owns one arena in `/dev/shm` for every model ("classes").
  Engines attach over a Unix socket, map the arena and copy rows in and out themselves.
- Rows use a canonical layout, independent of TP size, including Gated DeltaNet recurrent state. So any instance
  restores what any other stored, TP=2 or TP=4.
- Eviction works on whole conversations ("runs"), oldest first, across models. Rows a run stops offering age out on
  their own recency, so a branch nobody resumes ages out while the live branch keeps its rows.
- Pins for long shared prefixes go in a pin zone with an NVMe cold tier. On RELEASE the newest runs spill to NVMe, are
  read back when the arena is re-created, and the RAM is freed progressively.

**RAM as an extension of VRAM: the write-back band** (`kv_connector/v1/offloading/scheduler.py`)
- `VLLM_KV_WRITE_BACK=1` copies a conversation to RAM when its blocks are about to leave the GPU, on its own copy
  lane, never during compute. RAM extends VRAM instead of mirroring it.
- Hybrid models resume only where a recurrent-state snapshot exists, so each run is stored whole with its restore
  points. Up to `VLLM_KV_WRITE_BACK_KEEP_POINTS` older points are kept, and a broken run is re-trimmed to a
  restorable boundary or dropped.
- A finished turn's run is pinned and stored a couple of steps after it ends. So a follow-up on another instance
  restores from RAM.
- Flush on stop (`VLLM_KV_FLUSH_ON_STOP_SEC`): a graceful stop writes the GPU's cached conversations to the pool,
  newest first.

**Answer caching on hybrid models**
- `--enable-mamba-finish-checkpoint` keeps the recurrent state at the end of each generated answer, as well as at the
  end of the prompt. This includes models with sliding-window and circular-buffer groups.
- `--enable-answer-tail-offload` stores that answer-end tail in RAM as the run's deepest restore point. A follow-up
  that repeats the answer (located through its stop tokens) resumes there on any engine. Logic in
  `vllm/v1/core/answer_tail.py`.

**KV copies on XPU** (`vllm/v1/kv_offload/cpu/`)
- `VLLM_KV_OFFLOAD_COPY_LIST=1`: offload copies run on a Level Zero copy list of our own in the engine's context
  (`l0copy.c`, built into `/opt/b70/libl0copy.so`). They therefore don't queue behind the forward pass on the single
  user copy engine.
- Staged canonical copies use one DMA per page with a CPU scatter/gather on the copy thread. Host registration is done
  row by row (`VLLM_KV_OFFLOAD_REGISTER_ROWS`).

**Smaller changes**
- `--tool-call-lossless-args`: tool-call arguments are passed through byte for byte.
- An XPU path from FP8 block weights to per-channel W8A16.
- The Qwen3.5/3.8 FP8 head swap happens at load, before the memory profile.
- Workers start in parallel (`B70_SERIAL_WORKER_SPAWN` restores serial start).
- A backport of upstream `derive_mamba_conv_split`.

**Diagnostics**
- `B70_OFFLOAD_TRACE=1` logs every lookup (hit depth per group, why it stopped), every load, band store, pool eviction
  and dropped run.
- `vllm:kv_offload_write_back_events_total{event=...}` counts band, pin, tail and flush events.

## Knobs

- **Arena and connector:** `kv_connector_extra_config.shared_pool` = `{socket, pool}` and `canonical_layout`.
- **Write-back band:** `VLLM_KV_WRITE_BACK`, `_KEEP_POINTS`, `_MAX_HELD`, `_SCAN`, `_SCAN_MAX`, `_PIN_HOLD_S`,
  `_HANDOFF`, `_LEAD_FACTOR`, `_FREE_SHARE`, `_PROMPT_BOUNDARY`, `_RESCUE`, `_MIRROR`.
- **Copies:** `VLLM_KV_OFFLOAD_COPY_LIST` (and `_RING`), `_STAGED_COPY`, `_STAGED_RING_MB`, `_STAGED_SLOT_MB`,
  `_COPY_THREAD`, `_REGISTER`, `_REGISTER_ROWS`, `_EAGLE_POP`, `_CHECK_BOUNDS`.
- **Other:** `VLLM_KV_WRITE_THROUGH_MIN_FREE` and `VLLM_KV_FLUSH_ON_STOP_SEC`.

Defaults and reasons are next to each definition. Bump `_LAYOUT_VERSION` (`shared_pool/spec.py`) whenever the
meaning of a stored row changes.

## Where it is used and measured

The serving stack (router, registry, pool-server service, deploy scripts, measurements) lives in a separate repo.
Measured there: 27B TP=2 + TP=4 serving 32 agent sessions of 60k context reach 42-50k total tokens/s at 0.97-0.98
cache. Every follow-up turn re-prefills only its new message, on whichever instance serves it.
