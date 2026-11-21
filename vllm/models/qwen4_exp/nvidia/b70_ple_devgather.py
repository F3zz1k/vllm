# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B70-V7-PLEG (``B70_PLE_DEVICE_GATHER=1``): the RAM-native INT8 PLE producer
inside the forward, on the device.

The host-prepared producer (b70_ple_nvme ``prepare``) runs in ``prepare_inputs``
before every replay: a blocking D2H of the step's tokens (it waits for the
drafter, the last work of the previous step), the n-gram hash and the row
gather on the host, then one H2D. Measured with B70_PHASE_TIMING on the deployed engine (RAM
mode, 4 cards): 0.8 ms per step at c=1 and 1.0-1.4 ms at c=8-12 with the GPU
idle, because the verify rows' tokens (last sampled + 7 drafts) only exist
once the drafter has finished, and the consumer is inlined into the FULL graph
(``eager_break_during_capture`` does not break FULL captures), so its bytes
must be resident before the replay starts.

Here the producer moves into the graph, next to the consumer:

1. ``ple_ngram_ids`` (the fused Triton hash, one launch) computes the ids from
   the same ``input_ids`` / ``query_start_loc`` / ``ngram_context`` buffers
   the host path reads;
2. ``device_gather`` (one Triton launch) reads each owned row straight from
   the pinned slabs through their UVA device views
   (``get_accelerator_view_from_cpu_tensor``) into the packed prefetch buffer:
   owned (token, head) slots get the 164-B row, every other slot and every
   row past the actual token count gets zeros, exactly what ``_fill_stage``
   leaves in the stage;
3. the consumer is unchanged (int8 all-reduce over ETP, dequant).

No host sync, no host hash, no H2D per step. The host path stays the
reference: the first real step runs both and compares the bytes
(one sync, once), logging the result.

Gates: RAM mode only (B70_PLE_INT8=1; the NVMe row cache is not addressable
by row id), at most ``_MAX_SLABS`` slabs, UVA available.
"""

import os

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

_MAX_SLABS = 8


def enabled() -> bool:
    return os.environ.get("B70_PLE_DEVICE_GATHER", "0") == "1"


@triton.jit
def _pick(vals, slab, lo, hi, local, b, bmask, owned, ROW: tl.constexpr):
    hit = owned & (local >= lo) & (local < hi)
    v = tl.load(slab + (local - lo) * ROW + b, mask=bmask & hit, other=0)
    return tl.where(hit, v, vals)


@triton.jit
def _ple_device_gather_kernel(
    ids_ptr,  # int64 [T, H] global row ids
    qsl_ptr,  # int32 query_start_loc, actual tokens at [num_reqs]
    out_ptr,  # uint8 [T, H, ROW]
    s0,
    s1,
    s2,
    s3,
    s4,
    s5,
    s6,
    s7,  # uint8 slab device views (unused repeat s0)
    b1,
    b2,
    b3,
    b4,
    b5,
    b6,
    b7,
    b8,  # local row starts of slabs 1..7, then the end
    tp_start,
    tp_end,
    num_reqs,
    H: tl.constexpr,
    ROW: tl.constexpr,
    ROW_P2: tl.constexpr,
    NUM_SLABS: tl.constexpr,
):
    t = tl.program_id(0)
    h = tl.program_id(1)
    n_actual = tl.load(qsl_ptr + num_reqs)
    rid = tl.load(ids_ptr + t * H + h)
    owned = (t < n_actual) & (rid >= tp_start) & (rid < tp_end)
    local = tl.where(owned, rid - tp_start, 0).to(tl.int64)
    b = tl.arange(0, ROW_P2)
    bmask = b < ROW
    vals = tl.zeros([ROW_P2], dtype=tl.uint8)
    vals = _pick(vals, s0, 0, b1, local, b, bmask, owned, ROW)
    if NUM_SLABS > 1:
        vals = _pick(vals, s1, b1, b2, local, b, bmask, owned, ROW)
    if NUM_SLABS > 2:
        vals = _pick(vals, s2, b2, b3, local, b, bmask, owned, ROW)
    if NUM_SLABS > 3:
        vals = _pick(vals, s3, b3, b4, local, b, bmask, owned, ROW)
    if NUM_SLABS > 4:
        vals = _pick(vals, s4, b4, b5, local, b, bmask, owned, ROW)
    if NUM_SLABS > 5:
        vals = _pick(vals, s5, b5, b6, local, b, bmask, owned, ROW)
    if NUM_SLABS > 6:
        vals = _pick(vals, s6, b6, b7, local, b, bmask, owned, ROW)
    if NUM_SLABS > 7:
        vals = _pick(vals, s7, b7, b8, local, b, bmask, owned, ROW)
    tl.store(out_ptr + (t * H + h).to(tl.int64) * ROW + b, vals, mask=bmask)


class DeviceGather:
    """UVA views of one rank's pinned slabs plus the launch."""

    def __init__(
        self,
        slab_tensors: list[torch.Tensor],
        bounds,
        tp_start: int,
        tp_end: int,
        row_bytes: int,
        view_fn=None,
    ) -> None:
        n = len(slab_tensors)
        if not 1 <= n <= _MAX_SLABS:
            raise ValueError(f"B70-V7-PLEG: {n} slabs (1..{_MAX_SLABS})")
        if view_fn is None:
            from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

            view_fn = get_accelerator_view_from_cpu_tensor
        self.views = [view_fn(t) for t in slab_tensors]
        bnd = [int(x) for x in list(bounds)]
        if len(bnd) != n + 1 or bnd[0] != 0:
            raise ValueError(f"B70-V7-PLEG: bounds {bnd[:4]}... do not match {n} slabs")
        # starts of slabs 1..n-1, then the end; unused slots never match (lo = hi).
        self.bounds = bnd[1:] + [bnd[-1]] * (_MAX_SLABS - n)
        self.tp_start = int(tp_start)
        self.tp_end = int(tp_end)
        self.row = int(row_bytes)

    def __call__(
        self, ids: torch.Tensor, query_start_loc: torch.Tensor, out: torch.Tensor
    ) -> None:
        T, H = ids.shape
        if T == 0:
            return
        v = self.views + [self.views[0]] * (_MAX_SLABS - len(self.views))
        _ple_device_gather_kernel[(T, H)](
            ids,
            query_start_loc,
            out,
            *v,
            *self.bounds,
            self.tp_start,
            self.tp_end,
            query_start_loc.numel() - 1,
            H=H,
            ROW=self.row,
            ROW_P2=triton.next_power_of_2(self.row),
            NUM_SLABS=len(self.views),
        )


_logged = False


def log_once(rank_info: str) -> None:
    global _logged
    if not _logged:
        _logged = True
        logger.info(
            "B70-V7-PLEG PLE producer on the device (in-graph hash + UVA "
            "row gather, no per-step host sync): %s",
            rank_info,
        )
