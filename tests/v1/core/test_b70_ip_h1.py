# SPDX-License-Identifier: Apache-2.0
"""B70-IP-H1: INT8 lm_head copy (CPU): per-row int8 codes + scales, BF16 shard released, accuracy vs bf16."""

import torch

from vllm.models.qwen4_exp.nvidia.b70_fp8_head import B70Int8Head, b70_int8_head_of


class _Head(torch.nn.Module):
    def __init__(self, rows, k):
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.randn(rows, k, dtype=torch.bfloat16) * 0.02, requires_grad=False
        )


def test_int8_head_codes_scales_release():
    torch.manual_seed(0)
    h = _Head(20000, 256)
    ref = h.weight.data.float().clone()
    head = b70_int8_head_of(h)
    assert isinstance(head, B70Int8Head)
    assert h.weight.numel() == 0  # BF16 shard released
    assert head.wq.dtype == torch.int8 and head.wq.shape == (20000, 256)
    assert head.scale.shape == (20000, 1)
    assert head.wq.abs().max() <= 127
    deq = head.wq.float() * head.scale
    rel = (deq - ref).norm() / ref.norm()
    assert rel < 0.01, rel
    assert b70_int8_head_of(h) is head  # cached
