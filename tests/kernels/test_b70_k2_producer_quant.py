# SPDX-License-Identifier: Apache-2.0
"""B70-K2 producer quant: CPU equivalence tests (TRITON_INTERPRET=1).

For every producer kernel of ``b70_i8_producer`` / ``ops/hc_i8``:
1. the bf16 output equals the unfused producer it replaces, bitwise;
2. the int8 rows and fp32 scales equal the XPU ``scaled_int8_quant`` reference
   (``xpu_ops.dynamic_per_token_int8_quant_ref``, executed from the tree's own
   ``vllm/_xpu_ops.py`` source) applied to that bf16 output, bitwise;
3. producer -> W1 GEMM on the int8 rows equals unfused producer -> quant -> W1 GEMM.
Plus the quant rule's edge cases (zero rows, ties, the -128 corner) and the
carrier (attach / take) guards.
"""

import ast
import os
import textwrap

os.environ.setdefault("TRITON_INTERPRET", "1")
os.environ["B70_I8_PRODUCER_QUANT"] = "1"

import pytest  # noqa: E402
import torch  # noqa: E402

from vllm.model_executor.layers import b70_i8_producer as pq  # noqa: E402
from vllm.models.qwen4_exp.nvidia import b70_hc_gemm  # noqa: E402
from vllm.models.qwen4_exp.nvidia.ops import hc, hc_i8  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _faithful_interpreter_bf16():
    """The Triton 3.8 interpreter casts fp32 -> bf16 by truncation (``cast_impl``
    passes no rounding mode; its "RTNE" branch rounds half up without carry).
    The device rounds to nearest even, like torch. Patch the interpreter's
    converter for that one direction so CPU results mean what device results mean.
    Every Triton kernel in these tests (old and new) runs with the same patch."""
    import numpy as np
    import triton.language as tl
    from triton.runtime import interpreter as it

    orig = it._convert_float

    def conv(input, input_dtype, output_dtype, rounding_mode):
        if input_dtype == tl.float32 and output_dtype == tl.bfloat16:
            f = torch.from_numpy(np.ascontiguousarray(input).view(np.float32).copy())
            return f.to(torch.bfloat16).view(torch.int16).numpy().view(np.uint16)
        return orig(input, input_dtype, output_dtype, rounding_mode)

    it._convert_float = conv


if os.environ.get("TRITON_INTERPRET") == "1":
    _faithful_interpreter_bf16()


def _load_ref():
    """xpu_ops.dynamic_per_token_int8_quant_ref, from the tree's source (no wheel)."""
    with open(os.path.join(ROOT, "vllm/_xpu_ops.py")) as f:
        src = f.read()
    for node in ast.walk(ast.parse(src)):
        if (
            isinstance(node, ast.FunctionDef)
            and node.name == "dynamic_per_token_int8_quant_ref"
        ):
            body = textwrap.dedent(ast.get_source_segment(src, node))
            ns = {"torch": torch}
            exec(body, ns)
            return ns["dynamic_per_token_int8_quant_ref"]
    raise AssertionError("reference not found")


_REF = _load_ref()


def ref_quant(y: torch.Tensor):
    """W1's scaled_int8_quant on XPU: (int8 rows, fp32 [M, 1] scale)."""
    q, s, _ = _REF(y.contiguous(), True, 8)
    return q, s.reshape(-1, 1).to(torch.float32)


def bits(t: torch.Tensor) -> torch.Tensor:
    return t.view(torch.int16) if t.dtype == torch.bfloat16 else t


def assert_quant_equal(y, xq, xs):
    rq, rs = ref_quant(y)
    assert xs.shape == rs.shape and xs.dtype == torch.float32
    assert torch.equal(xs, rs), (xs - rs).abs().max()
    assert torch.equal(xq, rq), int((xq != rq).sum())


def _gen(seed):
    return torch.Generator().manual_seed(seed)


# ------------------------------------------------------------- quant rule edges
def test_quant_edges_via_identity_gate():
    """attn_gate_quant with sigmoid(gate) == 1.0 exactly is y = a: drive chosen rows."""
    d = 1536
    a = torch.randn(6, d, generator=_gen(1)).to(torch.bfloat16)
    a[0] = 0  # amax 0 -> scale clamps to 1e-5 (bf16), q = 0
    # ties: amax 127 -> s = 1.0; k + 0.5 values round half to even
    a[1] = 0
    a[1, :8] = torch.tensor([127.0, 0.5, 1.5, 2.5, -0.5, -1.5, -2.5, 126.5])
    # bf16(amax / 127) rounds below amax / 127, so |amax / s| > 127; bf16's unit
    # roundoff 2**-8 keeps it under 127.5, so the [-128, 127] clamp never binds.
    a[2] = 0
    a[2, 0] = -1.0546875
    a[3] = 1e-30  # tiny: the 1e-5 floor
    a[4] *= 1000.0
    gate = torch.full_like(a, 30.0)
    y, xq, xs = pq.attn_gate_quant(a, gate)
    assert torch.equal(bits(y), bits(a))
    assert_quant_equal(y, xq, xs)
    assert xq[0].abs().sum() == 0
    assert xq[1, :8].tolist() == [127, 0, 2, 2, 0, -2, -2, 126]
    assert int(xq[2, 0]) == -127 and int(xq.min()) >= -127


# ------------------------------------------------------------ attention o_proj
@pytest.mark.parametrize("rows", [1, 8, 13, 64, 200])
def test_attn_gate_quant(rows):
    g = _gen(rows)
    a = (torch.randn(rows, 1536, generator=g) * 3).to(torch.bfloat16)
    gate = (torch.randn(rows, 1536, generator=g) * 4).to(torch.bfloat16)
    y, xq, xs = pq.attn_gate_quant(a, gate)
    ref = a * torch.sigmoid(gate)  # the eager pair it replaces (CPU torch)
    # Triton's sigmoid vs torch's: compare and record; the quant must be exact on y.
    diff = (bits(y) != bits(ref)).float().mean().item()
    print(f"ATTN_GATE rows={rows} bf16 mismatch vs torch sigmoid*mul: {diff:.2e}")
    assert diff < 1e-3, diff
    if diff:
        ulp = (y.float() - ref.float()).abs() / ref.float().abs().clamp(min=1e-30)
        assert ulp.max() <= 2**-7, ulp.max()
    assert_quant_equal(y, xq, xs)


# ------------------------------------------------------------- GDN out_proj
@pytest.mark.parametrize("rows", [1, 8, 13, 64])
@pytest.mark.parametrize("act", ["silu", "sigmoid"])
def test_gated_rmsnorm_quant(rows, act):
    from vllm.third_party.flash_linear_attention.ops import layernorm_guard as lg

    # The FLA launcher sizes ROWS_PER_BLOCK from the device's compute units, which a
    # CPU run lacks; give it the B70's 32 Xe cores (1, 2 or 4 rows per block here).
    lg.num_compute_units = lambda index=None: 32
    rmsnorm_fn = lg.rmsnorm_fn

    g = _gen(100 + rows)
    x = torch.randn(rows, 12, 128, generator=g).to(torch.bfloat16)
    z = torch.randn(rows, 12, 128, generator=g).to(torch.bfloat16)
    w = (1 + 0.1 * torch.randn(128, generator=g)).to(torch.bfloat16)
    eps = 1e-6
    y, xq, xs = pq.gated_rmsnorm_quant(x, z, w, eps, act)
    try:
        ref = rmsnorm_fn(
            x,
            w,
            None,
            z=z,
            eps=eps,
            group_size=None,
            norm_before_gate=True,
            activation=act,
        )
    except Exception as e:  # pragma: no cover - FLA launcher needs a device
        pytest.skip(f"FLA rmsnorm_fn not runnable on CPU: {e!r}")
    assert torch.equal(bits(y), bits(ref.flatten(-2)))
    assert_quant_equal(y, xq, xs)


# ------------------------------------------------------------- HC block input
@pytest.mark.parametrize("rows", [1, 8, 13, 64, 300])
def test_hc_gate_mix_quant(rows):
    g = _gen(200 + rows)
    x = torch.randn(rows, 10240, generator=g).to(torch.bfloat16)
    gate = (torch.randn(rows, 10240, generator=g) * 2).to(torch.bfloat16)
    y, xq, xs = hc_i8.hc_gate_mix_quant(x, gate, 4)
    ref = hc._hc_gate_mix(x, gate, 4)
    assert y.shape == ref.shape == (rows, 2560)
    assert torch.equal(bits(y), bits(ref))
    assert_quant_equal(y, xq, xs)


# ------------------------------------------------------------- HC up input
@pytest.mark.parametrize("rows", [1, 8, 33, 64])
def test_hc_silu_quant_strided(rows):
    """Lora is the [:, :320] split view of the 336-wide down output."""
    g = _gen(300 + rows)
    down = (torch.randn(rows, 336, generator=g) * 6).to(torch.bfloat16)
    lora = down.split([320, 4, 12], dim=-1)[0]
    assert lora.stride(0) == 336
    y, xq, xs = hc_i8.hc_silu_quant(lora, 4)
    ref = hc._hc_silu(lora, 4)
    assert torch.equal(bits(y), bits(ref))
    assert_quant_equal(y, xq, xs)


@pytest.mark.parametrize("rows", [1, 2, 8, 9, 16, 32])
@pytest.mark.parametrize("shape", [(336, 324, 320), (320, 320, 320)])
def test_parallel_hc_gemm_silu_quant(rows, shape):
    columns, valid, lora_rank = shape
    g = _gen(400 + rows + columns)
    x = torch.randn(rows, 10240, generator=g).to(torch.bfloat16)
    w = (torch.randn(columns, 10240, generator=g) * 0.02).to(torch.bfloat16)
    out, lora, xq, xs = hc_i8.parallel_hc_gemm_silu_quant(x, w, valid, lora_rank, 4)
    r_out, r_lora = b70_hc_gemm.parallel_hc_gemm_silu(x, w, valid, lora_rank, 4)
    assert torch.equal(bits(out), bits(r_out))
    assert torch.equal(bits(lora), bits(r_lora))
    assert_quant_equal(lora, xq, xs)


# ------------------------------------------------------- carrier and consumer
def test_attach_take_guards():
    y = torch.randn(4, 64).to(torch.bfloat16)
    assert pq.take(y) is None
    q, s = ref_quant(y)
    pq.attach(y, q, s, "test")
    got = pq.take(y)
    assert got is not None and got[0] is q and got[1] is s
    assert pq.take(y.clone()) is None  # a new tensor never inherits it
    assert pq.take(y.view(2, 2, 64)) is None  # a view is another object
    y2 = torch.randn(4, 64).to(torch.bfloat16)
    pq.attach(y2, q[:3], s[:3], "test")
    assert pq.take(y2) is None  # shape mismatch is refused


@pytest.mark.parametrize("rows", [1, 8, 64])
def test_consumer_w1_gemm_bitwise(rows):
    """Producer -> b70_int8_dense_mm_q(use_k1=False) equals
    producer -> quant -> W1 Triton GEMM, bitwise."""
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
        triton_scaled_mm,
    )

    g = _gen(500 + rows)
    a = torch.randn(rows, 1536, generator=g).to(torch.bfloat16)
    gate = torch.randn(rows, 1536, generator=g).to(torch.bfloat16)
    wf = torch.randn(256, 1536, generator=g) * 0.02
    ws = (wf.abs().amax(1, keepdim=True) / 127).clamp(min=1e-8)
    wq = torch.round(wf / ws).clamp(-127, 127).to(torch.int8)
    y, xq, xs = pq.attn_gate_quant(a, gate)
    got = pq._b70_int8_dense_mm_q(xq, xs, wq, ws, None, 64, False)
    rq, rs = ref_quant(y)
    ref = triton_scaled_mm(rq, wq.t(), rs, ws, torch.bfloat16, None)
    assert torch.equal(bits(got), bits(ref))


def test_site_switches():
    assert pq.ENABLED
    for s in pq.ALL_SITES:
        assert pq.site_on(s)
    assert not pq.site_on("nope")
    lin = torch.nn.Linear(4, 4)
    assert not pq.consumer_is_w1(lin) and not pq.consumer_is_w1(None)
    assert pq.emits("attn_o", 8) and not pq.emits("attn_o", 0)


def test_w1_apply_takes_prequant():
    """B70Int8DenseMethod.apply on a tensor the producer marked runs the _q op
    (no quant), with the K1 op absent (CPU) falling back to W1's Triton GEMM."""
    from vllm.model_executor.layers.b70_dense_int8 import B70Int8DenseMethod
    from vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm import (  # noqa: E501
        triton_scaled_mm,
    )

    g = _gen(7)
    layer = torch.nn.Module()
    wf = torch.randn(256, 1536, generator=g) * 0.02
    ws = (wf.abs().amax(1, keepdim=True) / 127).clamp(min=1e-8)
    layer.weight = torch.nn.Parameter(
        torch.round(wf / ws).clamp(-127, 127).to(torch.int8), requires_grad=False
    )
    layer.b70_int8_scale = ws
    layer.prefix = "model.layers.3.self_attn.o_proj"
    a = torch.randn(8, 1536, generator=g).to(torch.bfloat16)
    gate = torch.randn(8, 1536, generator=g).to(torch.bfloat16)
    y, xq, xs = pq.attn_gate_quant(a, gate)
    pq.attach(y, xq, xs, "attn_o")
    out = B70Int8DenseMethod(64, False).apply(layer, y)
    assert layer.__dict__["_b70_pq_k1"] is True  # K1 chosen; op absent -> W1 GEMM
    rq, rs = ref_quant(y)
    ref = triton_scaled_mm(rq, layer.weight.t(), rs, ws, torch.bfloat16, None)
    assert torch.equal(bits(out), bits(ref))
