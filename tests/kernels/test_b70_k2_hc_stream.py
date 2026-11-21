# SPDX-License-Identifier: Apache-2.0
"""B70-K2 HC down int8 with per-(token, stream) scales: CPU tests (TRITON_INTERPRET=1).

1. The quantizing norm producers equal hc_combine_norm / grouped_gemma_rmsnorm bitwise,
   and their int8 rows / [M, 4] scales equal the XPU scaled_int8_quant reference per
   (token, stream).
2. The split-K GEMM equals an exact torch model of it (int64 dot per split, the same
   fp32 scaling and fixed-order fp32 sum), bitwise; and is batch-invariant.
3. GatedResidual with B70_HC_DOWN_INT8=1 + B70_HC_DOWN_INT8_STREAM_SCALE=1 runs it,
   is batch-invariant, and stays close to the BF16 path.
"""

import importlib.util
import os

os.environ.setdefault("TRITON_INTERPRET", "1")
os.environ["B70_I8_PRODUCER_QUANT"] = "1"
os.environ["B70_HC_SILU_FUSE"] = "1"
os.environ["B70_HC_ORDERED_MAX_ROWS"] = "32"
os.environ["B70_HC_DOWN_INT8"] = "1"
os.environ["B70_HC_DOWN_INT8_STREAM_SCALE"] = "1"

import pytest  # noqa: E402
import torch  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "k2_pq_base",
    os.path.join(os.path.dirname(__file__), "test_b70_k2_producer_quant.py"),
)
base = importlib.util.module_from_spec(_spec)  # the bf16 interpreter patch + ref quant
_spec.loader.exec_module(base)

from vllm.model_executor.layers import b70_i8_producer as pq  # noqa: E402
from vllm.models.qwen4_exp.nvidia.ops import hc, hc_i8  # noqa: E402

HC, HS = 4, 2560


def _gen(s):
    return torch.Generator().manual_seed(s)


def ref_group_quant(y, groups):
    m, k = y.shape
    q, s = base.ref_quant(y.reshape(m * groups, k // groups))
    return q.reshape(m, k), s.reshape(m, groups)


def stream_gains(g):
    """Per-branch (1 + w) with stream gains ~0.4x .. 2.5x, like the deployed checkpoint."""
    gains = torch.tensor([0.45, 0.6, 2.5, 0.5]).repeat_interleave(HS)
    return ((gains - 1) + 0.05 * torch.randn(HC * HS, generator=g)).to(torch.bfloat16)


@pytest.mark.parametrize("rows", [1, 8, 33])
@pytest.mark.parametrize("with_inj", [True, False])
def test_hc_combine_norm_quant(rows, with_inj):
    g = _gen(rows + 10 * with_inj)
    res = torch.randn(rows, HC * HS, generator=g).to(torch.bfloat16)
    blk = torch.randn(rows, HS, generator=g).to(torch.bfloat16)
    inj = torch.randn(rows, HC, generator=g).to(torch.bfloat16) if with_inj else None
    w = stream_gains(g)
    out, y, xq, xs = hc_i8.hc_combine_norm_quant(res, blk, inj, w, 1e-6, HC)
    r_out, r_y = hc._hc_combine_norm(res, blk, inj, w, 1e-6, HC)
    assert torch.equal(base.bits(out), base.bits(r_out))
    assert torch.equal(base.bits(y), base.bits(r_y))
    rq, rs = ref_group_quant(y, HC)
    assert torch.equal(xs, rs) and torch.equal(xq, rq)


@pytest.mark.parametrize("rows", [1, 8, 33])
def test_grouped_gemma_rmsnorm_quant(rows):
    g = _gen(50 + rows)
    x = torch.randn(rows, HC * HS, generator=g).to(torch.bfloat16)
    w = stream_gains(g)
    y, xq, xs = hc_i8.grouped_gemma_rmsnorm_quant(x, w, 1e-6, HC)
    assert torch.equal(
        base.bits(y), base.bits(hc._grouped_gemma_rmsnorm(x, w, 1e-6, HC))
    )
    rq, rs = ref_group_quant(y, HC)
    assert torch.equal(xs, rs) and torch.equal(xq, rq)
    gq, gs = hc_i8.group_quant(y, HC)
    assert torch.equal(gq, rq) and torch.equal(gs, rs)


def exact_model(xq, xs, wq, ws, sps):
    """The kernel's arithmetic in torch: int64 dot per split -> fp32 * xs * ws, fp32
    sum over splits in order, bf16 store."""
    m, k = xq.shape
    g = xs.shape[1]
    ks = k // (g * sps)
    acc = torch.zeros(m, wq.shape[0], dtype=torch.float32)
    for s in range(g * sps):
        sl = slice(s * ks, (s + 1) * ks)
        dot = (xq[:, sl].long() @ wq[:, sl].long().t()).to(torch.float32)
        acc = acc + dot * xs[:, s // sps, None] * ws[None, :, s // sps]
    return acc.to(torch.bfloat16)


@pytest.mark.parametrize("rows", [1, 8, 32, 64])
@pytest.mark.parametrize("sps", [1, 4])
def test_hc_down_i8s_exact_and_batch_invariant(rows, sps):
    g = _gen(100 + rows + sps)
    x = (torch.randn(rows, HC * HS, generator=g) * (1 + stream_gains(g).float())).to(
        torch.bfloat16
    )
    w = (torch.randn(336, HC * HS, generator=g) * 0.02).to(torch.bfloat16)
    w[324:] = 0  # merged-weight padding rows
    xq, xs = hc_i8.group_quant(x, HC)
    wq, ws = hc_i8.quantize_weight_int8_grouped(w, HC)
    out = hc_i8.hc_down_i8s(xq, xs, wq, ws, sps)
    assert torch.equal(base.bits(out), base.bits(exact_model(xq, xs, wq, ws, sps)))
    one = hc_i8.hc_down_i8s(xq[-1:].contiguous(), xs[-1:].contiguous(), wq, ws, sps)
    assert torch.equal(base.bits(one), base.bits(out[-1:]))
    d, lora, lq, ls = hc_i8.hc_down_i8s_silu(xq, xs, wq, ws, 320, HC, True, sps)
    assert torch.equal(base.bits(d), base.bits(out))
    r_lora = hc._hc_silu(out[:, :320].contiguous(), HC)
    assert torch.equal(base.bits(lora), base.bits(r_lora))
    rq, rs = base.ref_quant(lora)
    assert torch.equal(lq, rq) and torch.equal(ls, rs)
    # precision: per-stream int8 vs the fp32 product of the bf16 operands
    ref = x.float() @ w.float().t()
    rel = ((out.float() - ref).norm() / ref.norm()).item()
    assert rel < 0.08, rel


@pytest.fixture(scope="module")
def module():
    import types

    from vllm import _custom_ops as ops
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import parallel_state as ps
    from vllm.model_executor.layers import b70_dense_int8 as D
    from vllm.models.qwen4_exp.common.hyperconnection import HyperConnectionConfig
    from vllm.models.qwen4_exp.nvidia import b70_hyperconnection as bhc

    ops.scaled_int8_quant = lambda x, s=None, a=None, sym=True: (
        *base.ref_quant(x),
        None,
    )
    bhc.get_current_vllm_config = lambda: types.SimpleNamespace(speculative_config=None)
    cfg = HyperConnectionConfig(
        hc_count=HC,
        hidden_size=HS,
        params_dtype=torch.bfloat16,
        hc_lowrank=320,
        rms_norm_eps=1e-6,
        hc_per_branch_norm=True,
    )
    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        if not ps.model_parallel_is_initialized():
            ps.init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method="tcp://127.0.0.1:29518",
                local_rank=0,
                backend="gloo",
            )
            ps.initialize_model_parallel(1, 1, backend="gloo")
        gr = bhc.GatedResidual(cfg, prefix="model.layers.0.attn_hyper_connection")
    g = _gen(9)
    with torch.no_grad():
        gr.hc_norm.weight.copy_(stream_gains(g))
        down = gr.input_mix_weight_down_block_inject.weight
        down.copy_((torch.randn(down.shape, generator=g) * 0.02).to(down.dtype))
        down[324:] = 0
        up = gr.input_mix_weight_up
        wup = (torch.randn(up.weight.shape, generator=g) * 0.05).to(torch.bfloat16)
    q, s = D.quantize_weight_int8(wup)
    up.weight = torch.nn.Parameter(q, requires_grad=False)
    up.b70_int8_scale = s
    up.quant_method = D.B70Int8DenseMethod(64, False)
    gr._ordered_hc = True
    gr._astra_hc_up_prepared = True
    gr._b70_i8_block = True
    gr._prepare_kdown_ksplit()  # load-time hook: makes the per-(row, stream) int8 copy
    assert gr._kdown_q is not None and tuple(gr._kdown_qs.shape) == (336, HC)
    return gr


def test_gated_residual_stream_path(module):
    rows = 8
    g = _gen(77)
    h = torch.randn(rows, HC * HS, generator=g).to(torch.bfloat16)
    prev = torch.randn(rows, HS, generator=g).to(torch.bfloat16)
    inj = torch.randn(rows, HC, generator=g).to(torch.bfloat16)
    pq._logged_sites.clear()
    out = module._combine_and_mix_eager(h, prev, inj)
    assert pq._logged_sites == {"hc_up", "hc_block"}
    # batch invariance: row 3 alone gives the same bits
    one = module._combine_and_mix_eager(h[3:4], prev[3:4], inj[3:4])
    for a, b in zip(out, one):
        assert torch.equal(base.bits(a[3:4]), base.bits(b))
    # against the fp32 product of the bf16 operands (torch). Not against the BF16
    # Triton HC down: the interpreter has no bf16 tl.dot (it multiplies the uint16
    # storage), so that path is meaningless on CPU.
    xn = hc._hc_combine_norm(h, prev, inj, module.hc_norm.weight, 1e-6, HC)[1]
    lora, injection = module._down_and_silu(xn)
    w = module.input_mix_weight_down_block_inject.weight.float()
    d = xn.float() @ w.t()
    v = d[:, :320] / HC
    rels = {
        "lora_silu": (
            (lora.float() - v * torch.sigmoid(v)).norm() / (v * torch.sigmoid(v)).norm()
        ).item(),
        "injection": (
            (injection.float() - d[:, 320:324]).norm() / d[:, 320:324].norm()
        ).item(),
    }
    print("K2_STREAM_REL", rels)
    assert all(r < 0.06 for r in rels.values()), rels
