# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B70-V2-MOE: fewer small kernels around the XPU EP MoE (B70_MOE_KCUT=1, default off).

Per MoE layer the stock EP path launches, around the routing:

1. ``torch.where(topk_ids < 0, sentinel, topk_ids)`` (xpu_moe.py): compare, scalar
   fill, where = 3 kernels;
2. ``(topk_ids < 0).all(1)`` + out-of-place ``masked_fill`` (XpuFusedMoe._apply_kernel
   and both W4A8 applies): compare, reduce, D2D copy, masked fill = 4 kernels;
3. ``torch.zeros(rows_per_expert)``: 1 fill.

With B70_MOE_KCUT=1 one Triton kernel writes the sentinel-mapped ids AND zeroes a
fresh ``rows_per_expert``. Step 2 is dropped only where it is provably dead: after
the sentinel rewrite every id is >= 0 (the sentinel is 0 or E-1), so
``(ids < 0).all(1)`` is False on every row and the masked_fill is the identity. Rows
that a FULL-graph replay pads (any hidden content, NaN included) are routed exactly
as on the stock path: the ids are the same integers, the remap skips the sentinel
expert on this rank. 8 kernels -> 1 per layer, bitwise-identical output.

The stock bodies live in the vllm_xpu_kernels wheel. The replicas below are stock
minus steps 2 and 3. Each one is armed only if the AST fingerprint of the wheel
function it replaces equals the one it was copied from (``_FP``); a different wheel
(e.g. a new W4A8 apply) logs once and keeps the wheel's own function.
"""

import ast
import hashlib
import inspect
import os
import textwrap

import torch

from vllm.logger import init_logger
from vllm.triton_utils import tl, triton

logger = init_logger(__name__)

# AST fingerprints of the wheel functions the replicas were copied from
# (the v6 vllm_xpu_kernels wheel).
_FP = {
    "XpuFusedMoe._apply_kernel": {"32c13063d42ad5ae"},  # v6, v7 (unchanged)
    # v6; v7 (adds the B70_W4A8_FUSED_ACT branch)
    "b70_moe_w4a8.apply": {"8e77c64b832b5251", "2bdb73e8f7e4f4b8"},
}

_ENABLED = os.environ.get("B70_MOE_KCUT", "0") == "1"
_LOGGED: set[str] = set()


def enabled() -> bool:
    return _ENABLED


def _log_once(key: str, msg: str, *args) -> None:
    if key not in _LOGGED:
        _LOGGED.add(key)
        logger.info(msg, *args)


def fingerprint_node(node: ast.AST) -> str:
    return hashlib.sha256(ast.dump(node).encode()).hexdigest()[:16]


def fingerprint(func) -> str:
    """AST fingerprint of a function (line numbers and comments excluded)."""
    src = textwrap.dedent(inspect.getsource(func))
    tree = ast.parse(src)
    return fingerprint_node(tree.body[0])


_FP_OK: dict[str, bool] = {}


def _armed(key: str, func) -> bool:
    ok = _FP_OK.get(key)
    if ok is None:
        try:
            got = fingerprint(func)
        except (OSError, TypeError, SyntaxError) as e:
            got = f"unreadable ({e})"
        ok = got in _FP[key]
        _FP_OK[key] = ok
        if not ok:
            logger.warning(
                "B70-V2-MOE: %s fingerprint %s not in %s (wheel changed); "
                "the wheel's own function runs",
                key,
                got,
                _FP[key],
            )
    return ok


# n_ids is not specialised: one compiled variant serves every batch size.
@triton.jit(do_not_specialize=["n_ids"])
def _sentinel_zero_kernel(
    ids_ptr, out_ptr, n_ids, sentinel, rpe_ptr, n_exp, BLOCK: tl.constexpr
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = offs < n_ids
    x = tl.load(ids_ptr + offs, mask=m, other=0)
    tl.store(out_ptr + offs, tl.where(x < 0, sentinel, x).to(x.dtype), mask=m)
    tl.store(rpe_ptr + offs, tl.zeros([BLOCK], dtype=tl.int32), mask=offs < n_exp)


def sentinel_and_zero(topk_ids: torch.Tensor, sentinel: int, num_local_experts: int):
    """where(ids < 0, sentinel, ids) and a zeroed int32 rows_per_expert, one launch."""
    ids = topk_ids if topk_ids.is_contiguous() else topk_ids.contiguous()
    out = torch.empty_like(ids)
    rpe = torch.empty((num_local_experts,), dtype=torch.int32, device=ids.device)
    n = ids.numel()
    block = 256
    grid = (max(triton.cdiv(n, block), triton.cdiv(num_local_experts, block), 1),)
    _sentinel_zero_kernel[grid](
        ids, out, n, sentinel, rpe, num_local_experts, BLOCK=block, num_warps=1
    )
    _log_once(
        "on",
        "B70-V2-MOE: EP sentinel + rows_per_expert zero in one kernel; "
        "dead padding mask "
        "skipped (B70_MOE_KCUT=1)",
    )
    return out.view(topk_ids.shape), rpe


def stock_kernel_ok(impl, a1q_scale) -> bool:
    """The replica of XpuFusedMoe._apply_kernel may run for this call."""
    if getattr(impl, "_use_ref", True):
        return False
    return _armed("XpuFusedMoe._apply_kernel", type(impl)._apply_kernel)


def apply_kernel(
    impl,
    output,
    hidden_states,
    topk_weights,
    topk_ids,
    rows_per_expert,
    expert_map=None,
    a1q_scale=None,
):
    """XpuFusedMoe._apply_kernel without the padding mask (dead after the sentinel
    rewrite), with ``rows_per_expert`` zeroed by the caller. Otherwise the wheel's
    body verbatim."""
    from vllm_xpu_kernels import fused_moe_interface as fmi

    self = impl
    num_rows, hidden_size = hidden_states.shape
    num_moe_inputs = self.n_experts_per_token * num_rows
    act_quant = a1q_scale is not None

    if expert_map is None and self.ep_size > 1:
        expert_map = self.expert_map

    if act_quant:
        remapped_scales = torch.empty(
            (num_rows * self.n_experts_per_token, a1q_scale.shape[1]),
            dtype=a1q_scale.dtype,
            device=a1q_scale.device,
        )
    else:
        remapped_scales = None
    remapped_hidden_states = torch.empty(
        (num_rows * self.n_experts_per_token, hidden_size),
        dtype=hidden_states.dtype,
        device=hidden_states.device,
    )

    # [stock: invalid_padding_rows mask + masked_fill; dead, see module doc]
    # [stock: rows_per_expert = torch.zeros(...); zeroed by sentinel_and_zero]
    unpermuted_row_to_permuted_row = torch.empty(
        (num_rows, self.n_experts_per_token),
        dtype=torch.int32,
        device=hidden_states.device,
    )

    torch.ops._moe_C.remap_hidden_states(
        hidden_states=hidden_states,
        hidden_states_scales=a1q_scale,
        remapped_hidden_states=remapped_hidden_states,
        remapped_hidden_states_scales=remapped_scales,
        expert_map=expert_map,
        rows_per_expert=rows_per_expert,
        unpermuted_row_to_permuted_row=unpermuted_row_to_permuted_row,
        topk_ids=topk_ids,
        total_experts_num=self.total_experts_num,
        local_experts_num=self.local_experts_num,
    )

    if remapped_scales is not None and self.is_mxfp8:
        remapped_hidden_states = fmi.dequant_mxfp8(
            remapped_hidden_states, remapped_scales
        ).to(output.dtype)
        remapped_scales = None
    elif remapped_scales is not None and self.is_block_fp8:
        remapped_hidden_states = fmi.dequant_fp8_block_act(
            remapped_hidden_states, remapped_scales
        ).to(output.dtype)
        remapped_scales = None

    gemm1_output = torch.empty(
        (num_moe_inputs, 2 * self.inter_size), dtype=output.dtype, device=output.device
    )
    self._grouped_gemm(
        remapped_hidden_states,
        remapped_scales,
        self.w13,
        self.gemm1_wei_scales,
        self.w13_bias,
        gemm1_output,
        rows_per_expert,
        2 * self.inter_size,
        hidden_size,
    )

    if self.gemm1_clamp_limit is not None and self.gemm1_clamp_limit > 0:
        gate = gemm1_output[:, : self.inter_size]
        up = gemm1_output[:, self.inter_size :]
        gate.clamp_(max=self.gemm1_clamp_limit)
        up.clamp_(min=-self.gemm1_clamp_limit, max=self.gemm1_clamp_limit)

    act_output = torch.empty(
        (num_moe_inputs, self.inter_size * self.inter_size_scale),
        dtype=gemm1_output.dtype,
        device=gemm1_output.device,
    )
    if self.activation == "situ":
        self.act_func(
            act_output,
            gemm1_output,
            self.activation_situ_beta,
            -1.0
            if self.activation_situ_linear_beta is None
            else self.activation_situ_linear_beta,
        )
    else:
        self.act_func(act_output, gemm1_output)

    gemm2_output = torch.empty(
        (num_moe_inputs, hidden_size), dtype=output.dtype, device=output.device
    )

    if act_quant:
        act_output, gemm2_act_scale = fmi.quant_act_xpu(act_output, self.recipe)
        if self.is_mxfp8 and gemm2_act_scale is not None:
            act_output = fmi.dequant_mxfp8(act_output, gemm2_act_scale).to(output.dtype)
            gemm2_act_scale = None
        elif self.is_block_fp8 and gemm2_act_scale is not None:
            act_output = fmi.dequant_fp8_block_act(act_output, gemm2_act_scale).to(
                output.dtype
            )
            gemm2_act_scale = None
    else:
        gemm2_act_scale = None
    self._grouped_gemm(
        act_output,
        gemm2_act_scale,
        self.w2,
        self.gemm2_wei_scales,
        self.w2_bias,
        gemm2_output,
        rows_per_expert,
        hidden_size,
        self.inter_size * self.inter_size_scale,
    )

    torch.ops._moe_C.moe_gather(
        output,
        gemm2_output,
        topk_weights,
        unpermuted_row_to_permuted_row,
        self.num_experts,
    )


def w4a8_cutlass_ok(mod) -> bool:
    return _armed("b70_moe_w4a8.apply", mod.apply)


def apply_w4a8_cutlass(
    mod,
    impl,
    output,
    hidden_states,
    topk_weights,
    topk_ids,
    rows_per_expert,
    expert_map=None,
) -> None:
    """vllm_xpu_kernels.b70_moe_w4a8.apply without the dead padding mask and with
    ``rows_per_expert`` zeroed by the caller. The GEMM, quant and gather calls are
    the wheel's."""
    op = mod.load_op()
    num_rows, hidden = hidden_states.shape
    topk = impl.n_experts_per_token
    n_moe = num_rows * topk
    inter = impl.inter_size
    dev = hidden_states.device
    if expert_map is None and impl.ep_size > 1:
        expert_map = impl.expert_map
    # [stock: invalid mask + masked_fill; dead, see module doc]
    hq, hsc = mod.quant_rows(hidden_states)
    remapped = torch.empty((n_moe, hidden), dtype=torch.float8_e4m3fn, device=dev)
    remapped_sc = torch.empty((n_moe, 1), dtype=torch.float32, device=dev)
    # [stock: rows_per_expert = torch.zeros(...); zeroed by sentinel_and_zero]
    unp = torch.empty((num_rows, topk), dtype=torch.int32, device=dev)
    torch.ops._moe_C.remap_hidden_states(
        hidden_states=hq.view(torch.float8_e4m3fn),
        hidden_states_scales=hsc,
        remapped_hidden_states=remapped,
        remapped_hidden_states_scales=remapped_sc,
        expert_map=expert_map,
        rows_per_expert=rows_per_expert,
        unpermuted_row_to_permuted_row=unp,
        topk_ids=topk_ids,
        total_experts_num=impl.total_experts_num,
        local_experts_num=impl.local_experts_num,
    )

    # v7: optional fused gemm1 epilogue (silu*up -> int8); v6 has no load_glu_op.
    glu = None
    if hasattr(mod, "load_glu_op") and mod.fused_act_enabled() and inter % 128 == 0:
        glu = mod.load_glu_op()
    if glu is not None:
        if not getattr(mod, "_LOGGED_FUSED", True):
            mod._LOGGED_FUSED = True
            import logging

            logging.getLogger("vllm_xpu_kernels").warning(
                "B70-K1: W4A8 MoE fused gemm1 epilogue on (silu*up -> int8, "
                "per-row x 128-col scales; inter=%d)",
                inter,
            )
        aq = torch.empty((n_moe, inter), dtype=torch.int8, device=dev)
        asc = torch.empty((n_moe, inter // 128), dtype=torch.float32, device=dev)
        glu(
            remapped.view(torch.int8),
            remapped_sc,
            impl.w13,
            impl.gemm1_wei_scales,
            impl.w13_bias,
            aq,
            asc,
            rows_per_expert,
            inter,
            hidden,
            impl.num_experts,
        )
    else:
        g1 = torch.empty((n_moe, 2 * inter), dtype=output.dtype, device=dev)
        op(
            remapped.view(torch.int8),
            remapped_sc,
            impl.w13,
            impl.gemm1_wei_scales,
            impl.w13_bias,
            g1,
            rows_per_expert,
            2 * inter,
            hidden,
            impl.num_experts,
        )
        act = torch.empty((n_moe, inter), dtype=output.dtype, device=dev)
        impl.act_func(act, g1)
        aq, asc = mod.quant_rows(act)

    g2 = torch.empty((n_moe, hidden), dtype=output.dtype, device=dev)
    op(
        aq,
        asc,
        impl.w2,
        impl.gemm2_wei_scales,
        impl.w2_bias,
        g2,
        rows_per_expert,
        hidden,
        inter,
        impl.num_experts,
    )
    torch.ops._moe_C.moe_gather(output, g2, topk_weights, unp, impl.num_experts)
