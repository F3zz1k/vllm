# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B70-CJK: serving-side Han mask, worker side (env ``B70_CJK_MASK=1``).

One device bool mask [vocab] built at boot from the tokenizer, one per-request
flag. The model runner applies it in place on the target logits right after
``compute_logits``, next to the grammar bitmask, so it covers the plain
sampler, every speculative verification row (drafted Han then has target
probability 0 and is rejected, and the residual resample cannot pick it), and
both the stock and the ``B70_SAMPLER_FAST`` sampling paths. The astra greedy
shortcut builds its own head and is declined for batches with a flagged row.

Cost: nothing when no row of the batch is flagged (host check on the numpy
mirror); otherwise one launch, grid (rows, V / 4096), that reads the mask and
stores -inf on masked ids of flagged rows only.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton
from vllm.v1.sample import b70_cjk
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.sampling_params import SamplingParams

logger = init_logger(__name__)

BLOCK = 4096


@triton.jit
def _cjk_mask_kernel(
    logits_ptr,
    logits_stride,
    vocab_size,
    expanded_idx_mapping_ptr,
    flags_ptr,
    mask_ptr,
    BLOCK_SIZE: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    blk = tl.program_id(1)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + row)
    flag = tl.load(flags_ptr + req_state_idx)
    if flag != 0:
        offset = blk * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        in_vocab = offset < vocab_size
        m = tl.load(mask_ptr + offset, mask=in_vocab, other=0)
        tl.store(
            logits_ptr + row * logits_stride + offset,
            -float("inf"),
            mask=in_vocab & (m != 0),
        )


def apply_cjk_mask(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    flags: torch.Tensor,
    mask_u8: torch.Tensor,
) -> None:
    """In place: logits[r, v] = -inf where flags[map[r]] and mask[v]."""
    num_rows = logits.shape[0]
    vocab_size = min(logits.shape[1], mask_u8.shape[0])
    if num_rows == 0 or vocab_size == 0:
        return
    assert logits.stride(-1) == 1
    grid = (num_rows, triton.cdiv(vocab_size, BLOCK))
    _cjk_mask_kernel[grid](
        logits,
        logits.stride(0),
        vocab_size,
        expanded_idx_mapping,
        flags,
        mask_u8,
        BLOCK_SIZE=BLOCK,
    )


def apply_cjk_mask_reference(
    logits: torch.Tensor,
    expanded_idx_mapping: torch.Tensor,
    flags: torch.Tensor,
    mask: torch.Tensor,
) -> None:
    """Torch reference of the kernel (tests, CPU)."""
    v = min(logits.shape[1], mask.shape[0])
    rows = flags[expanded_idx_mapping.long()] != 0
    if not bool(rows.any()):
        return
    region = logits[:, :v]
    region.masked_fill_(rows[:, None] & mask[None, :v].bool(), float("-inf"))


class CjkMaskState:
    def __init__(
        self,
        vllm_config: VllmConfig | None,
        max_num_reqs: int,
        vocab_size: int,
        device: torch.device,
        tokenizer=None,
    ):
        if tokenizer is None:
            assert vllm_config is not None
            from vllm.tokenizers import cached_tokenizer_from_config

            tokenizer = cached_tokenizer_from_config(vllm_config.model_config)
            if tokenizer is None:
                raise RuntimeError(
                    f"{b70_cjk.ENV}=1 needs the tokenizer (skip_tokenizer_init is set)"
                )
        mask_np, counts = b70_cjk.build_mask(tokenizer, vocab_size)
        self.counts = counts
        self.mask = torch.from_numpy(mask_np).to(device)
        self.mask_u8 = self.mask.view(torch.uint8)
        self.flags = UvaBackedTensor(max_num_reqs, dtype=torch.int32)
        self.use_torch_reference = device.type == "cpu"
        logger.info(
            "B70-CJK: Han mask active: %d of %d vocabulary ids masked "
            "(%d Han, %d partial UTF-8; kana and special tokens kept); "
            "per request: vllm_xargs.%s, auto-off when the prompt has Han",
            counts["masked"],
            vocab_size,
            counts["han"],
            counts["partial"],
            b70_cjk.XARG,
        )

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> bool:
        extra = sampling_params.extra_args or {}
        # The API server resolves the flag; a missing key (an engine-direct
        # request) defaults to masked.
        flag = b70_cjk.parse_flag(extra.get(b70_cjk.XARG))
        flag = True if flag is None else flag
        self.flags.np[req_idx] = int(flag)
        return flag

    def apply_staged_writes(self) -> None:
        self.flags.copy_to_uva()

    def any_flagged(self, idx_mapping_np: np.ndarray) -> bool:
        return bool(np.any(self.flags.np[idx_mapping_np]))

    def apply(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
    ) -> None:
        if logits.shape[0] == 0 or not self.any_flagged(idx_mapping_np):
            return
        if self.use_torch_reference:
            apply_cjk_mask_reference(
                logits, expanded_idx_mapping, self.flags.gpu, self.mask
            )
        else:
            apply_cjk_mask(logits, expanded_idx_mapping, self.flags.gpu, self.mask_u8)

    def warmup(self, logits: torch.Tensor) -> None:
        """JIT the kernel for this logits dtype (writes into the dummy logits)."""
        n = logits.shape[0]
        if n == 0 or self.use_torch_reference:
            return
        rows = torch.zeros(n, dtype=torch.int32, device=logits.device)
        ones = torch.ones(1, dtype=torch.int32, device=logits.device)
        apply_cjk_mask(logits, rows, ones, self.mask_u8)
