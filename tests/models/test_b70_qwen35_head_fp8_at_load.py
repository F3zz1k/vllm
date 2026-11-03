# SPDX-License-Identifier: Apache-2.0
"""B70-Q27-0007: CPU proofs that the B70_HEAD_FP8 swap runs at load.

1. With B70_HEAD_FP8=1 (XPU faked) b70_qwen35_head_fp8_at_load swaps the Qwen3_5
   lm_head for the FP8 copy, releases the BF16 shard and returns the freed segment
   (empty_cache), through a ConditionalGeneration wrapper (get_language_model).
2. A drafter that shares the head module (MTP shared_head.head) sees the same FP8
   quant_method; the FP8 rows dequantise to the BF16 rows within E4M3 rounding.
3. The forward-time hook afterwards is a flag check: no second conversion.
4. Env unset, or a model that is not Qwen3_5: nothing changes.
5. The model runner calls it after speculator.load_model (source order), so a
   drafter reading the BF16 head at load still sees BF16.
No device is opened: everything runs on CPU tensors.
"""

import inspect
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.models import qwen3_5 as q35
from vllm.models.qwen4_exp.nvidia import b70_fp8_head as fh

ROWS, K = 96, 64


class _Head(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = nn.Parameter(weight, requires_grad=False)
        self.quant_method = "bf16"


def _lm(weight):
    lm = q35.Qwen3_5ForCausalLM.__new__(q35.Qwen3_5ForCausalLM)
    nn.Module.__init__(lm)
    lm.lm_head = _Head(weight)
    lm.config = SimpleNamespace(tie_word_embeddings=False)
    lm.model = nn.Module()
    lm.logits_processor = SimpleNamespace(head_dtype=None)
    return lm


class _Wrapper(nn.Module):
    def __init__(self, lm):
        super().__init__()
        self.language_model = lm

    def get_language_model(self):
        return self.language_model


@pytest.fixture
def fake_xpu(monkeypatch):
    from vllm.platforms import current_platform

    calls = {"empty_cache": 0}
    monkeypatch.setattr(current_platform, "is_xpu", lambda: True, raising=False)
    monkeypatch.setattr(
        torch.xpu, "is_current_stream_capturing", lambda: False, raising=False
    )
    monkeypatch.setattr(torch.accelerator, "memory_allocated", lambda *a: 0)

    def _empty():
        calls["empty_cache"] += 1

    monkeypatch.setattr(torch.accelerator, "empty_cache", _empty)
    return calls


def _weight():
    torch.manual_seed(0)
    return (torch.randn(ROWS, K) * 0.05).to(torch.bfloat16)


def test_swap_at_load_releases_bf16_and_shares(monkeypatch, fake_xpu):
    monkeypatch.setenv("B70_HEAD_FP8", "1")
    w = _weight()
    lm = _lm(w.clone())
    drafter_shared_head = lm.lm_head  # what MTP shared_head.head holds
    assert q35.b70_qwen35_head_fp8_at_load(_Wrapper(lm)) is True
    head = lm.lm_head
    assert isinstance(head.quant_method, fh.B70Fp8Head)
    assert head.weight.numel() == 0 and head.weight.shape[1] == K
    assert drafter_shared_head.quant_method is head.quant_method
    assert fake_xpu["empty_cache"] == 1
    h8 = head.quant_method
    assert h8.w8.shape == (K, ROWS) and h8.w8.dtype == torch.float8_e4m3fn
    deq = (h8.w8.float() * h8.scale).t()
    ref = w.float()
    # E4M3 keeps 3 mantissa bits: relative error per element <= 2^-4 of the row max
    assert torch.all((deq - ref).abs() <= ref.abs().amax(1, keepdim=True) / 16 + 1e-6)

    # 3. the forward / compute_logits hook is now a flag check
    q8 = head.quant_method
    q35._b70_qwen35_head_fp8(lm)
    assert head.quant_method is q8 and fake_xpu["empty_cache"] == 1


def test_env_unset_or_other_model_untouched(monkeypatch, fake_xpu):
    monkeypatch.delenv("B70_HEAD_FP8", raising=False)
    lm = _lm(_weight())
    assert q35.b70_qwen35_head_fp8_at_load(_Wrapper(lm)) is False
    assert lm.lm_head.quant_method == "bf16" and lm.lm_head.weight.numel() == ROWS * K

    monkeypatch.setenv("B70_HEAD_FP8", "1")
    other = nn.Module()
    other.lm_head = _Head(_weight())
    assert q35.b70_qwen35_head_fp8_at_load(other) is False
    assert other.lm_head.quant_method == "bf16"
    assert fake_xpu["empty_cache"] == 0


def test_runner_calls_it_after_the_drafter_loads():
    from vllm.v1.worker.gpu import model_runner as mr

    src = inspect.getsource(mr.GPUModelRunner.load_model)
    i_spec = src.index("self.speculator.load_model(self.model)")
    i_head = src.index("b70_qwen35_head_fp8_at_load(self.model)")
    i_mem = src.index("self.model_memory_usage = m.consumed_memory")
    assert i_spec < i_head < i_mem
