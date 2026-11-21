# SPDX-License-Identifier: Apache-2.0
"""B70-K2 wiring through GatedResidual (CPU, TRITON_INTERPRET=1).

One GatedResidual with the deployed settings (ordered DFlash2 HC, B70_HC_SILU_FUSE=1,
B70_HC_ORDERED_MAX_ROWS=32, HC up converted to W1 int8) runs combine_and_mix
with B70-K2 on and off. With K2 on, the HC up GEMM and the block input's consumer
get producer-quantized rows; every output must be bitwise equal to K2 off
(W1's own quant + the same GEMM), at rows on the parallel route (1, 8, 32) and
on the stock-down route (64).
"""

import os

os.environ.setdefault("TRITON_INTERPRET", "1")
os.environ["B70_I8_PRODUCER_QUANT"] = "1"
os.environ["B70_HC_SILU_FUSE"] = "1"
os.environ["B70_HC_ORDERED_MAX_ROWS"] = "32"

import importlib.util  # noqa: E402

import pytest  # noqa: E402
import torch  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "k2_pq_base",
    os.path.join(os.path.dirname(__file__), "test_b70_k2_producer_quant.py"),
)
base = importlib.util.module_from_spec(_spec)  # the bf16 interpreter patch + ref quant
_spec.loader.exec_module(base)
from vllm.model_executor.layers import b70_i8_producer as pq  # noqa: E402


@pytest.fixture(scope="module")
def module():
    import types

    from vllm import _custom_ops as ops
    from vllm.model_executor.layers import b70_dense_int8 as D
    from vllm.models.qwen4_exp.common.hyperconnection import HyperConnectionConfig
    from vllm.models.qwen4_exp.nvidia import b70_hyperconnection as bhc
    from vllm.models.qwen4_exp.nvidia.b70_hyperconnection import GatedResidual

    # W1's quant on CPU: the XPU reference (same bits as the native op).
    ops.scaled_int8_quant = lambda x, s=None, a=None, sym=True: (
        *base.ref_quant(x),
        None,
    )
    cfg = HyperConnectionConfig(
        hc_count=4,
        hidden_size=2560,
        params_dtype=torch.bfloat16,
        hc_lowrank=320,
        rms_norm_eps=1e-6,
        hc_per_branch_norm=True,
    )
    # No device for a VllmConfig on this host: GatedResidual only reads the
    # speculative config (ordered HC is set by hand below, as the DFlash2 target sets it).
    bhc.get_current_vllm_config = lambda: types.SimpleNamespace(speculative_config=None)
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.distributed import parallel_state as ps

    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        if not ps.model_parallel_is_initialized():  # the Linear modules ask for TP
            ps.init_distributed_environment(
                world_size=1,
                rank=0,
                distributed_init_method="tcp://127.0.0.1:29517",
                local_rank=0,
                backend="gloo",
            )
            ps.initialize_model_parallel(1, 1, backend="gloo")
        gr = GatedResidual(cfg, prefix="model.layers.0.attn_hyper_connection")
    g = torch.Generator().manual_seed(3)
    with torch.no_grad():
        gr.hc_norm.weight.copy_(0.1 * torch.randn(10240, generator=g))
        down = gr.input_mix_weight_down_block_inject.weight
        down.copy_((torch.randn(down.shape, generator=g) * 0.02).to(down.dtype))
        up = gr.input_mix_weight_up
        wup = (torch.randn(up.weight.shape, generator=g) * 0.05).to(torch.bfloat16)
    q, s = D.quantize_weight_int8(wup)
    up.weight = torch.nn.Parameter(q, requires_grad=False)
    up.b70_int8_scale = s
    up.quant_method = D.B70Int8DenseMethod(64, False)
    gr._ordered_hc = True  # DFlash2 target
    gr._astra_hc_up_prepared = True  # int8 up: astra's BF16 fused up never applies
    gr._b70_i8_block = True  # set by the decoder when the block consumer is W1
    return gr


@pytest.mark.parametrize("rows", [1, 8, 32, 64])
def test_combine_and_mix_k2_bitwise(module, rows):
    g = torch.Generator().manual_seed(rows)
    h = torch.randn(rows, 10240, generator=g).to(torch.bfloat16)
    prev = torch.randn(rows, 2560, generator=g).to(torch.bfloat16)
    inj = torch.randn(rows, 4, generator=g).to(torch.bfloat16)

    pq.ENABLED = False
    ref = module._combine_and_mix_eager(h, prev, inj)
    assert pq.take(ref[1]) is None
    pq.ENABLED = True
    pq._logged_sites.clear()
    out = module._combine_and_mix_eager(h, prev, inj)
    assert pq._logged_sites == {"hc_up", "hc_block"}
    got = pq.take(out[1])
    assert got is not None
    rq, rs = base.ref_quant(out[1])
    assert torch.equal(got[0], rq) and torch.equal(got[1], rs)
    for a, b in zip(out, ref):
        assert torch.equal(base.bits(a), base.bits(b))
