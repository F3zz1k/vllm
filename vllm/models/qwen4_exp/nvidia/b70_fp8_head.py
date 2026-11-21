# SPDX-License-Identifier: Apache-2.0
"""B70-0048/0053 (V32 port): the target's lm_head shard as FP8 W8A16 (E4M3, one scale per vocab row).

B70_HEAD_FP8=1 (XPU): built once at the first forward (the eager memory-profile run, before any activation) and the
BF16 shard released, so vLLM's profile sees the shard gone and the saving lands in KV. Every caller of the head
(compute_logits, astra greedy, the DFlash2 drafter sharing the module) goes through head.quant_method.apply."""
import os

import torch


class B70Fp8Head:
    """What LogitsProcessor needs of an lm_head (quant_method.apply), backed by an FP8 W8A16 copy of the shard."""

    def __init__(self, w8: torch.Tensor, scale: torch.Tensor) -> None:
        self.w8, self.scale = w8, scale

    def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        shape = x.shape
        out = torch.ops._xpu_C.fp8_gemm_w8a16(x.reshape(-1, shape[-1]), self.w8, self.scale, None)
        if bias is not None:
            out = out + bias
        return out.reshape(*shape[:-1], -1)


class B70Int8Head:
    """B70-IP-H1: the lm_head shard as INT8 W8A8 (one scale per vocab row, activations int8 per token), native int8
    DPAS through B70-IP-W1's opaque op (Triton int8 at decode M, oneDNN int8_scaled_mm above). Per-row int8 is ~4x
    more accurate than e4m3 on embedding-like rows (PLE study: 0.66 % vs 2.66 % rel-L2) and the math stays int8."""

    def __init__(self, wq: torch.Tensor, scale: torch.Tensor, triton_max_m: int) -> None:
        self.wq, self.scale, self.triton_max_m = wq, scale, triton_max_m

    def apply(self, layer, x: torch.Tensor, bias: torch.Tensor | None = None) -> torch.Tensor:
        import vllm.model_executor.layers.b70_dense_int8  # noqa: F401  (registers the op)

        shape = x.shape
        out = torch.ops.vllm.b70_int8_dense_mm(
            x.reshape(-1, shape[-1]), self.wq, self.scale, bias, self.triton_max_m
        )
        return out.reshape(*shape[:-1], -1)


def b70_int8_head_of(lm_head) -> "B70Int8Head | None":
    """INT8 copy of an lm_head shard (built chunked, BF16 moved to host first so the two never coexist in VRAM)."""
    head8 = getattr(lm_head, "_b70_int8_head", None)
    weight = getattr(lm_head, "weight", None)
    if head8 is not None:
        return head8
    if weight is None or weight.dim() != 2 or weight.dtype != torch.bfloat16 or weight.numel() == 0:
        return None
    rows, k = weight.shape
    device = weight.device
    src = weight.data.to("cpu")
    weight.data = torch.empty((0, k), dtype=weight.dtype, device=device)
    wq = torch.empty((rows, k), dtype=torch.int8, device=device)
    scale = torch.empty((rows, 1), dtype=torch.float32, device=device)
    step = 8192
    for i in range(0, rows, step):
        chunk = src[i : i + step].to(device).float()
        sc = (chunk.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-8)
        wq[i : i + step] = torch.round(chunk / sc).clamp(-127, 127).to(torch.int8)
        scale[i : i + step] = sc
        del chunk
    del src
    head8 = B70Int8Head(wq, scale, int(os.environ.get("B70_DENSE_INT8_TRITON_MAX_M", "64")))
    lm_head._b70_int8_head = head8
    return head8


def b70_fp8_head_of(lm_head, release_bf16: bool = False) -> "B70Fp8Head | None":
    """The FP8 copy of an lm_head shard, cached on the module. release_bf16: the BF16 weight goes to host memory and
    is released on the device before the FP8 copy is built (the two never coexist in VRAM)."""
    head8 = getattr(lm_head, "_b70_fp8_head", None)
    weight = getattr(lm_head, "weight", None)
    if head8 is None:
        if weight is None or weight.dim() != 2 or weight.dtype != torch.bfloat16 or weight.numel() == 0:
            return None
        rows, k = weight.shape
        device = weight.device
        src = weight.data
        if release_bf16:
            src = weight.data.to("cpu")
            weight.data = torch.empty((0, k), dtype=weight.dtype, device=device)
        w8 = torch.empty((k, rows), dtype=torch.float8_e4m3fn, device=device)
        scale = torch.empty((1, rows), dtype=torch.float32, device=device)
        step = 8192
        for i in range(0, rows, step):
            chunk = src[i : i + step].to(device).float()
            s = chunk.abs().amax(dim=1).clamp(min=1e-12) / 448.0
            w8[:, i : i + step] = (chunk / s[:, None]).t().to(torch.float8_e4m3fn)
            scale[0, i : i + step] = s
            del chunk
        del src
        head8 = B70Fp8Head(w8, scale)
        lm_head._b70_fp8_head = head8
    if release_bf16 and weight is not None and weight.numel() != 0:
        weight.data = torch.empty((0, weight.shape[1]), dtype=weight.dtype, device=weight.device)
    return head8


def maybe_prepare_head_fp8(lm) -> None:
    """Once per model: B70_HEAD_FP8=1 on XPU swaps lm.lm_head.quant_method for the FP8 copy and releases BF16."""
    if getattr(lm, "_b70_head_fp8_checked", False):
        return
    lm._b70_head_fp8_checked = True
    from vllm.platforms import current_platform

    want_int8 = os.environ.get("B70_HEAD_INT8", "0") == "1"
    if (os.environ.get("B70_HEAD_FP8", "0") != "1" and not want_int8) or not current_platform.is_xpu():
        return
    if torch.xpu.is_current_stream_capturing():
        lm._b70_head_fp8_checked = False
        return
    from vllm.logger import init_logger

    logger = init_logger(__name__)
    head = lm.lm_head
    weight = head.weight
    released = weight.numel() * weight.element_size()
    alloc0 = torch.accelerator.memory_allocated(weight.device)
    if want_int8:
        # B70-IP-H1: INT8 W8A8 head wins over B70_HEAD_FP8 when both are set.
        head_i8 = b70_int8_head_of(head)
        if head_i8 is None:
            logger.warning("B70 head INT8: no BF16 lm_head weight; off")
            return
        head.quant_method = head_i8
        alloc1 = torch.accelerator.memory_allocated(weight.device)
        logger.info("B70-IP-H1 head INT8: target lm_head %d x %d as INT8 W8A8 (%.0f MB), BF16 shard released "
                    "(%.0f MB); torch allocated %.2f -> %.2f GiB", head_i8.wq.shape[0], head_i8.wq.shape[1],
                    head_i8.wq.numel() / 1e6, released / 1e6, alloc0 / 2**30, alloc1 / 2**30)
        return
    head8 = b70_fp8_head_of(head, release_bf16=True)
    if head8 is None:
        logger.warning("B70 head FP8: no BF16 lm_head weight; off")
        return
    head.quant_method = head8
    alloc1 = torch.accelerator.memory_allocated(weight.device)
    logger.info("B70 head FP8: target lm_head %d x %d as FP8 W8A16 (%.0f MB), BF16 shard released (%.0f MB); "
                "torch allocated %.2f -> %.2f GiB", head8.w8.shape[1], head8.w8.shape[0], head8.w8.numel() / 1e6,
                released / 1e6, alloc0 / 2**30, alloc1 / 2**30)
