"""B70 Fase S runtime helper: draft MTP LM head INT4 g128 sym.

Escribido por patch_draft_lmhead_int4.py dentro del contenedor. Provee la
cuantizacion one-time del lm_head fp16 compartido a GPTQ INT4 g128 sym y el
ruteo de las 4 pasadas del draft por ``int4_gemm_w4a16``. El target queda
fp16 (lossless).
"""

from __future__ import annotations

import os

import torch


def quantize_lmhead_to_int4(weight: torch.Tensor, group_size: int = 128):
    """Quantiza un lm_head fp16 [N, K] a GPTQ INT4 g128 sym.

    Returns (qweight, scales, qzeros, group_size):
      qweight: int32 [K//8, N] en layout NT (strides[-2] == 1), nibbles
               secuenciales LSB-first, valor almacenado = q + 8 (q in [-8, 7])
      scales:  fp16 [K//group_size, N]
      qzeros:  int8 tensor([8])  -> rama simetrica de int4_gemm_w4a16
    """
    device = weight.device
    N, K = weight.shape
    num_groups = K // group_size
    chunk = 4096
    shifts = torch.tensor(
        [0, 4, 8, 12, 16, 20, 24, 28], dtype=torch.int32, device=device
    )
    parts = []
    scale_parts = []
    for i in range(0, N, chunk):
        wc = weight[i : i + chunk].float()  # [c, K] fp32 (chunked: no 5 GB temp)
        wg = wc.view(wc.shape[0], num_groups, group_size)
        maxabs = wg.abs().amax(dim=-1)  # [c, g]
        scale = maxabs / 7.0
        q = (wg / scale.unsqueeze(-1)).round().clamp(-8, 7).to(torch.int32)
        stored = q + 8  # 0..15
        qv = stored.view(wc.shape[0], num_groups, group_size // 8, 8)
        packed = (qv << shifts).sum(dim=-1).to(torch.int32).reshape(wc.shape[0], K // 8)
        parts.append(packed)
        scale_parts.append(scale.half())
    qweight_contig = torch.cat(parts, dim=0)  # [N, K//8] int32
    scales_contig = torch.cat(scale_parts, dim=0)  # [N, g] fp16
    # Layout NT requerido por la op (strides[-2] == 1) + scales contiguas
    qweight = qweight_contig.t()  # [K//8, N], strides (1, K//8)
    scales = scales_contig.t().contiguous()  # [g, N]
    qzeros = torch.tensor([8], dtype=torch.int8, device=device)
    return qweight, scales, qzeros, group_size


def int4_lmhead_logits(
    x: torch.Tensor,
    qweight: torch.Tensor,
    scales: torch.Tensor,
    qzeros: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Logits [.., vocab] via int4_gemm_w4a16 (mismo formato que el cuerpo)."""
    flat = x.reshape(-1, x.shape[-1])
    logits = torch.ops._xpu_C.int4_gemm_w4a16(
        flat, qweight, None, scales, qzeros, group_size, None
    )
    return logits.reshape(*x.shape[:-1], qweight.shape[1])


@torch.no_grad()
def build_draft_lmhead_int4(model) -> None:
    """Cuantiza el lm_head fp16 compartido del draft (one-time, no-op si no
    hay env gate o si ya se construyo). Almacena en model._b70_lmhead_int4."""
    if os.environ.get("B70_DRAFT_LMHEAD_INT4") != "1":
        return
    if getattr(model, "_b70_lmhead_int4_tp_blocked", False):
        return
    parallel_config = getattr(
        getattr(model, "vllm_config", None), "parallel_config", None
    )
    tp_size = getattr(parallel_config, "tensor_parallel_size", None)
    if tp_size is None:
        model._b70_lmhead_int4_tp_blocked = True
        print(
            "[B70] draft LM head INT4: TP unknown; skipping (fail-closed, issue #9)",
            flush=True,
        )
        return
    if tp_size > 1:
        model._b70_lmhead_int4_tp_blocked = True
        print(
            f"[B70] draft LM head INT4: TP>1 (tp={tp_size}) detected; skipping "
            "(C1/TP1 only, issue #9)",
            flush=True,
        )
        return
    if getattr(model, "_b70_lmhead_int4", None) is not None:
        return
    head = getattr(model, "lm_head", None)
    weight = getattr(head, "weight", None)
    if weight is None:
        print(
            "[B70] draft LM head INT4: lm_head.weight no disponible; "
            "draft sigue por fp16",
            flush=True,
        )
        return
    # B70-V6x guard: the shared lm_head may already be owned by another path
    # (B70_HEAD_FP8 / B70_HEAD_INT8 release the BF16 shard at the first forward).
    # Only a plain 2-D fp16/bf16 weight is quantized here; anything else keeps
    # the stock logits path.
    if (
        not isinstance(weight, torch.Tensor)
        or weight.dim() != 2
        or weight.numel() == 0
        or weight.dtype not in (torch.float16, torch.bfloat16)
    ):
        model._b70_lmhead_int4_tp_blocked = True
        print(
            "[B70] draft LM head INT4: lm_head is not a plain fp16/bf16 "
            "weight (owned by another head path?); skipping",
            flush=True,
        )
        return
    print(
        "[B70] draft LM head INT4: cuantizando lm_head fp16 "
        f"{tuple(weight.shape)} -> INT4 g128 sym (one-time)",
        flush=True,
    )
    # B70_DRAFT_VOCAB
    _vf = os.environ.get("B70_DRAFT_VOCAB")
    if _vf == "64k":  # B70-V6x: the bundled cookbook subset (Qwen3.5/3.8 vocab)
        _vf = os.path.join(os.path.dirname(__file__), "b70_draft_vocab_64k.json")
    _w = weight.detach()
    if _vf:
        import json as _json

        with open(_vf) as _f:
            _vocab = _json.load(_f)
        _ids = torch.tensor(sorted(_vocab), dtype=torch.long, device=_w.device)
        model._b70_draft_vocab_ids = _ids
        model._b70_draft_vocab_full = _w.shape[0]
        _w = _w.index_select(0, _ids).contiguous()
        print(f"[B70] draft vocab: {_ids.numel()} / {weight.shape[0]} rows", flush=True)
    qweight, scales, qzeros, group_size = quantize_lmhead_to_int4(_w)
    model._b70_lmhead_int4 = (qweight, scales, qzeros, group_size)
    fp16_bytes = weight.numel() * weight.element_size()
    int4_bytes = qweight.numel() * qweight.element_size() + (
        scales.numel() * scales.element_size()
    )
    print(
        f"[B70] draft LM head INT4: listo. {fp16_bytes / 1e9:.2f} GB fp16 -> "
        f"{int4_bytes / 1e9:.2f} GB INT4 (ahorro "
        f"{(fp16_bytes - int4_bytes) / 1e6:.1f} MB/lectura)",
        flush=True,
    )


def draft_lmhead_int4_logits(model, hidden_states: torch.Tensor) -> torch.Tensor:
    """Logits del draft via la copia INT4 (4 pasadas/paso -> 0.66 GB c/u)."""
    qweight, scales, qzeros, group_size = model._b70_lmhead_int4
    logits = int4_lmhead_logits(hidden_states, qweight, scales, qzeros, group_size)
    _ids = getattr(model, "_b70_draft_vocab_ids", None)
    if _ids is not None:  # B70_DRAFT_VOCAB
        full = logits.new_full(
            (*logits.shape[:-1], model._b70_draft_vocab_full), float("-inf")
        )
        full.index_copy_(full.dim() - 1, _ids, logits)
        logits = full
    org = getattr(getattr(model, "logits_processor", None), "org_vocab_size", None)
    if org is not None and logits.shape[-1] > org:
        logits = logits[..., :org]
    return logits
