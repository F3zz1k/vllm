# SPDX-License-Identifier: Apache-2.0
"""Vocab-parallel greedy verification with the ordinary proposal/state protocol."""

import os
import numpy as np
import torch
from vllm.triton_utils import triton, tl
from vllm.v1.worker.gpu.sample.output import SamplerOutput


@triton.jit
def _partials(
    logits,
    pairs,
    M: tl.constexpr,
    N: tl.constexpr,
    VALID: tl.constexpr,
    START: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    cols = tile * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(logits + row * N + cols, cols < N, other=-float("inf")).to(
        tl.float32
    )
    values = tl.where(cols < VALID, values, -float("inf"))
    maximum = tl.max(values, 0)
    index = tl.min(
        tl.where((values == maximum) & (cols < VALID), cols + START, 16777215), 0
    )
    offset = (row * tl.cdiv(N, BLOCK) + tile) * 2
    tl.store(pairs + offset, maximum)
    tl.store(pairs + offset + 1, index.to(tl.float32))


@triton.jit
def _select_and_verify(
    pairs,
    cu,
    logits_indices,
    input_ids,
    positions,
    local_pos,
    idx_mapping,
    seq_lens,
    prefill_lens,
    out,
    sampled,
    rejected,
    M: tl.constexpr,
    TILES: tl.constexpr,
    TP: tl.constexpr,
    PAIRS: tl.constexpr,
):
    req = tl.program_id(0)
    start = tl.load(cu + req)
    end = tl.load(cu + req + 1)
    n = end - start
    rows = tl.arange(0, 8)
    cols = tl.arange(0, PAIRS)
    rank = cols // TILES
    tile = cols % TILES
    offset = ((rank[None, :] * M + start + rows[:, None]) * TILES + tile[None, :]) * 2
    mask = (rows[:, None] < n) & (cols[None, :] < TP * TILES)
    values = tl.load(pairs + offset, mask, other=-float("inf"))
    ids = tl.load(pairs + offset + 1, mask, other=16777215.0).to(tl.int32)
    maximum = tl.max(values, 1)
    top = tl.min(tl.where(values == maximum[:, None], ids, 16777215), 1)
    state_idx = tl.load(idx_mapping + req)
    prefill = tl.load(prefill_lens + state_idx)
    ix = tl.load(logits_indices + start + rows + 1, rows < n - 1, other=0)
    draft = tl.load(input_ids + ix, rows < n - 1, other=-1)
    pos = tl.load(positions + ix, rows < n - 1, other=0)
    local = tl.load(local_pos + start + rows + 1, rows < n - 1, other=0)
    placeholder = (local > 0) & (pos - local < prefill)
    mismatch = (rows < n - 1) & ((draft != top) | (draft < 0) | placeholder)
    first_rejected = tl.min(tl.where(mismatch, rows, n - 1), 0)
    count = first_rejected + 1
    prefilling = tl.load(seq_lens + req) < prefill
    tl.store(out + req * 8 + rows, tl.where(rows < count, top, -1))
    tl.store(sampled + req, tl.where(prefilling, 0, count))
    tl.store(rejected + req, tl.where(prefilling, 0, n - count))


def eligible(runner, batch, grammar):
    if os.getenv("VLLM_XPU_ASTRA_GREEDY", "1") != "1":
        return False
    sampler = runner.sampler
    reject = runner.rejection_sampler
    if (
        grammar is not None
        or runner.batch_sharder is not None
        or sampler is None
        or reject is None
        or batch.num_reqs < 1
        or batch.num_draft_tokens < 1
    ):
        return False
    if (
        reject.num_speculative_steps != 7
        or reject.enable_adaptive_verification
        or reject.synthetic_conditional_rates is not None
        or reject.use_block_verification
        or reject.watermark_key is not None
        or sampler.compute_nans
        or sampler.return_sampling_mask
        or sampler.trace_replay_state is not None
    ):
        return False
    idx = batch.idx_mapping_np
    if np.any(idx < 0):
        return False
    if not np.all(sampler.sampling_states.temperature.np[idx] == 0):
        return False
    if np.any(sampler.needs_logits_processing[idx]):
        return False
    # B70-CJK: the greedy head bypasses the logits; masked rows take the
    # ordinary path.
    cjk_mask = getattr(sampler, "cjk_mask", None)
    if cjk_mask is not None and cjk_mask.any_flagged(idx):
        return False
    if sampler.get_logprobs_dims(idx) is not None:
        return False
    lengths = np.diff(batch.cu_num_logits_np)
    if np.any(lengths < 1) or np.any(lengths > 8):
        return False
    lm = getattr(runner.model, "language_model", None)
    if lm is None or not hasattr(lm, "lm_head") or not hasattr(lm, "logits_processor"):
        return False
    head = lm.lm_head
    if getattr(head, "num_added_embeddings", 0) != 0:
        return False
    # The ordinary verifier truncates to the common target/draft vocabulary.
    # Keep that path for models whose vocabularies differ or are remapped.
    draft_logits = getattr(runner.speculator, "draft_logits", None)
    if (
        draft_logits is None
        or draft_logits.shape[-1] != runner.vocab_size
        or head.org_vocab_size != runner.vocab_size
        or lm.logits_processor.org_vocab_size != runner.vocab_size
        or lm.logits_processor.logits_as_input
    ):
        return False
    return (
        lm.logits_processor.scale == 1.0
        and lm.logits_processor.soft_cap is None
        and runner.vocab_size < 2**24
    )


def sample(runner, hidden, batch):
    from vllm.distributed import get_tp_group
    import torch.distributed as dist

    model = runner.model.language_model
    selected = hidden[batch.logits_indices]
    head = model.lm_head
    logits = model.logits_processor._apply_head(head, selected, None)
    rows, columns = logits.shape
    tiles = triton.cdiv(columns, 4096)
    pairs = torch.empty((rows, tiles, 2), dtype=torch.float32, device=hidden.device)
    _partials[(rows, tiles)](
        logits,
        pairs,
        rows,
        columns,
        head.shard_indices.org_vocab_end_index
        - head.shard_indices.org_vocab_start_index,
        head.shard_indices.org_vocab_start_index,
        4096,
        num_warps=4,
    )
    # Keep rank-major layout to avoid an all-gather transpose/copy.
    tp = head.tp_size
    gathered = torch.empty(
        (tp * rows, tiles, 2), dtype=pairs.dtype, device=hidden.device
    )
    dist.all_gather_into_tensor(gathered, pairs, group=get_tp_group().device_group)
    out = torch.empty((batch.num_reqs, 8), dtype=torch.int64, device=hidden.device)
    count = torch.empty((batch.num_reqs,), dtype=torch.int32, device=hidden.device)
    rejected = torch.empty_like(count)
    _select_and_verify[(batch.num_reqs,)](
        gathered,
        batch.cu_num_logits,
        batch.logits_indices,
        batch.input_ids,
        batch.positions,
        batch.expanded_local_pos,
        batch.idx_mapping,
        batch.seq_lens,
        runner.sampler.req_states.prefill_len.gpu,
        out,
        count,
        rejected,
        rows,
        tiles,
        tp,
        triton.next_power_of_2(tp * tiles),
        num_warps=4,
    )
    result = SamplerOutput(out, None, None, count, rejected)
    return result, count, rejected
