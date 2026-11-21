# SPDX-License-Identifier: Apache-2.0
"""B70-TC: opaque custom ops so the XPU Qwen4Exp target can run under VLLM_COMPILE.

Off unless B70_TORCH_COMPILE=1 (read at import). With it unset, nothing here is
reached and the model is the upstream breakable-graph build.

Why ops and not plain tracing: vLLM compiles with guards dropped, so every
Python branch on a batch dimension is frozen at the trace size (the profile
run, 4096 rows). The B70 HC, GDN-gate and PLE paths dispatch on row count
(fixed-order DFlash2 HC down GEMM <= B70_HC_ORDERED_MAX_ROWS, astra HC up at
<= 8 rows, parallel HC GEMM at <= 32 rows, ordered in_proj_ba above 8 rows) and
PLE keeps host-side state. Behind an op the branch runs at call time with the
real width; Inductor fuses the glue around the ops.

PLE is a splitting op (listed in CompilationConfig._attention_ops): its short
conv reads per-request metadata, like attention, so it must stay outside the
PIECEWISE graphs. HC and the GDN gate are graph-safe kernels, not splitting ops.
"""

import os

import torch
from torch import nn

from vllm.utils.torch_utils import direct_register_custom_op

ENABLED = os.environ.get("B70_TORCH_COMPILE", "0") == "1"

_MODULES: dict[str, nn.Module] = {}


def register(name: str, module: nn.Module) -> None:
    if ENABLED:
        if name in _MODULES and _MODULES[name] is not module:
            raise RuntimeError(f"B70-TC: duplicate module name {name!r}")
        _MODULES[name] = module


def compile_enabled(vllm_config) -> bool:  # support_torch_compile(enable_if=...)
    return ENABLED


# -- HC mix / combine_and_mix ------------------------------------------------
def _hc_mix(hidden_states: torch.Tensor, layer_name: str) -> list[torch.Tensor]:
    m = _MODULES[layer_name]
    _, block_input, injection = m._mix_eager(hidden_states)
    if injection is None:
        injection = hidden_states.new_empty((hidden_states.shape[0], 0))
    return [block_input, injection]


def _hc_mix_fake(hidden_states: torch.Tensor, layer_name: str) -> list[torch.Tensor]:
    m = _MODULES[layer_name]
    t = hidden_states.shape[0]
    block_input = hidden_states.new_empty((t, m.hidden_size))
    if not m.use_combine:
        return [block_input, hidden_states.new_empty((t, 0))]
    # Same strides as the real split view of the merged down projection.
    down = hidden_states.new_empty((t, m.lora_rank + m.hc_count + m.pad_size))
    return [block_input, down[:, m.lora_rank : m.lora_rank + m.hc_count]]


def _hc_combine_and_mix(
    hidden_states: torch.Tensor,
    prev_block_output: torch.Tensor,
    prev_injection: torch.Tensor | None,
    layer_name: str,
) -> list[torch.Tensor]:
    m = _MODULES[layer_name]
    out, block_input, injection = m._combine_and_mix_eager(
        hidden_states, prev_block_output, prev_injection
    )
    if injection is None:
        injection = hidden_states.new_empty((hidden_states.shape[0], 0))
    return [out, block_input, injection]


def _hc_combine_and_mix_fake(
    hidden_states: torch.Tensor,
    prev_block_output: torch.Tensor,
    prev_injection: torch.Tensor | None,
    layer_name: str,
) -> list[torch.Tensor]:
    return [torch.empty_like(hidden_states, memory_format=torch.contiguous_format)] + (
        _hc_mix_fake(hidden_states, layer_name)
    )


direct_register_custom_op(
    op_name="b70_hc_mix", op_func=_hc_mix, mutates_args=[], fake_impl=_hc_mix_fake
)
direct_register_custom_op(
    op_name="b70_hc_combine_and_mix",
    op_func=_hc_combine_and_mix,
    mutates_args=[],
    fake_impl=_hc_combine_and_mix_fake,
)


# -- GDN in_proj_ba (size-dispatched: F.linear <= 8 rows, ordered Triton above) --
def _gdn_in_proj_ba(
    hidden_states: torch.Tensor, out_features: int, layer_name: str
) -> torch.Tensor:
    return _MODULES[layer_name]._b70_in_proj_ba_eager(hidden_states)


def _gdn_in_proj_ba_fake(
    hidden_states: torch.Tensor, out_features: int, layer_name: str
) -> torch.Tensor:
    return hidden_states.new_empty((hidden_states.shape[0], out_features))


direct_register_custom_op(
    op_name="b70_gdn_in_proj_ba",
    op_func=_gdn_in_proj_ba,
    mutates_args=[],
    fake_impl=_gdn_in_proj_ba_fake,
)


# -- PLE layer (splitting op; writes graph-owned storage) ---------------------
def _qwen4_exp_ple(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    output.copy_(
        _MODULES[layer_name](hidden_states, input_ids, query_start_loc, ngram_context)
    )


def _qwen4_exp_ple_fake(
    hidden_states: torch.Tensor,
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    ngram_context: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
) -> None:
    return None


direct_register_custom_op(
    op_name="b70_qwen4_exp_ple",
    op_func=_qwen4_exp_ple,
    mutates_args=["output"],
    fake_impl=_qwen4_exp_ple_fake,
)
