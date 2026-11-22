# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 vision encoder on XPU: upstream's PE-Lang-style ViT with a 2x2 merge
adapter (`nvidia/vision.py`), with its Triton glue written as torch ops (the
encoder runs once per image, eager) and attention on the XPU varlen kernel.

Module names mirror the checkpoint (`model.visual.<name>`), except that each
layer's q/k/v projections load into one fused `qkv_proj`. Rounding to bf16
happens at the same points as upstream's kernels."""

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any, NamedTuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from vllm.model_executor.model_loader.weight_utils import default_weight_loader

_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
# the image processor's CLIP std (float32-rounded): pixels are CLIP-normalised with it
# and rounded to bf16, then mapped to [-1, 1]
_PROCESSOR_CLIP_STD = (0.26862954, 0.2613026, 0.2757771)


def _linear(i: int, o: int, bias: bool = True) -> nn.Linear:
    return nn.Linear(i, o, bias=bias, dtype=torch.bfloat16)


def _bf16(x: torch.Tensor) -> torch.Tensor:
    return x.to(torch.bfloat16).float()


def _layer_norm(x: torch.Tensor, norm: nn.LayerNorm) -> torch.Tensor:
    """fp32 LayerNorm of fp32 rows (weights upcast), like upstream's kernel."""
    return F.layer_norm(x, x.shape[-1:], norm.weight.float(), norm.bias.float(), norm.eps)


class VisionMetadata(NamedTuple):
    meta: torch.Tensor  # int32 [T, 4] (y, x, height, width) in native patches; CLS: y = x = -1
    patch_rows: torch.Tensor  # per token row, its row in the patch embeddings
    pixel_offsets: torch.Tensor  # per native patch, the element offset of its top-left pixel
    pool_rows: torch.Tensor  # token rows in 2x2-merge order, for the adapter
    cu_seqlens: torch.Tensor
    max_seqlen: int


def gather_patches(pixels: torch.Tensor, offsets: torch.Tensor, patch: int) -> torch.Tensor:
    """Native patches of raw `[N, side, side, 3]` pixels as bf16 rows of (py, px, c)
    values in [-1, 1]: `2 * (bf16((v / 255 - mean) / clip_std) * std + mean) - 1`."""
    side = pixels.shape[1]
    f = torch.arange(patch * patch * 3, device=pixels.device)
    base = (f // (patch * 3)) * (side * 3) + f % (patch * 3)
    v = pixels.reshape(-1)[offsets.long()[:, None] + base[None, :]].float()
    c = f % 3
    mean = torch.tensor(_CLIP_MEAN, device=v.device)[c]
    clip_std = torch.tensor(_PROCESSOR_CLIP_STD, device=v.device)[c]
    std = torch.tensor(_CLIP_STD, device=v.device)[c]
    v = _bf16((v / 255.0 - mean) / clip_std)
    return (2.0 * (v * std + mean) - 1.0).to(torch.bfloat16)


def _source_coords(index: torch.Tensor, target: torch.Tensor, source: int):
    coord = (source * (index + 0.5) / target - 0.5).clamp(0.0, source - 1.0)
    s0 = coord.floor().long()
    s1 = (s0 + 1).clamp(max=source - 1)
    w1 = coord - s0.float()
    return s0, s1, 1.0 - w1, w1


def embed_tokens(patch_embeds: torch.Tensor, md: VisionMetadata, grid: torch.Tensor,
                 cls: torch.Tensor, cls_pos: torch.Tensor, ln_pre: nn.LayerNorm,
                 ln_1: nn.LayerNorm) -> tuple[torch.Tensor, torch.Tensor]:
    """(hidden, ln_1(hidden)) with hidden = ln_pre(patch embedding + bilinearly
    resampled position embedding); CLS rows take the class embedding."""
    meta = md.meta.long()
    sh, sw, dim = grid.shape
    y, x = meta[:, 0], meta[:, 1]
    h, w = meta[:, 2].float(), meta[:, 3].float()
    is_cls = y < 0
    yc, xc = y.clamp(min=0).float(), x.clamp(min=0).float()
    y0, y1, wy0, wy1 = _source_coords(yc, h, sh)
    x0, x1, wx0, wx1 = _source_coords(xc, w, sw)
    g = grid.float()

    def at(a, b):
        return g[a, b]

    top = at(y0, x0) * wx0[:, None] + at(y0, x1) * wx1[:, None]
    bottom = at(y1, x0) * wx0[:, None] + at(y1, x1) * wx1[:, None]
    pos = _bf16(top * wy0[:, None] + bottom * wy1[:, None])
    patch = patch_embeds.float()[md.patch_rows.long()]
    hidden = patch + pos
    hidden[is_cls] = (cls.float() + cls_pos.float()).expand(int(is_cls.sum()), dim)
    hidden = _bf16(_layer_norm(_bf16(hidden), ln_pre))
    return hidden.to(torch.bfloat16), _layer_norm(hidden, ln_1).to(torch.bfloat16)


def apply_rotary_qk_(qkv: torch.Tensor, meta: torch.Tensor, num_heads: int) -> None:
    """In-place interleaved 2D RoPE on the q and k thirds of `qkv [T, 3W]`: the first
    quarter of the pairs rotates by x + 1, the second by y + 1 (CLS: identity)."""
    T, width = qkv.shape
    hd = width // (3 * num_heads)
    pairs = hd // 2
    inv = 10000.0 ** (-torch.arange(0, hd, 4, device=qkv.device, dtype=torch.float32) / hd)
    p = torch.arange(pairs, device=qkv.device)
    freq = inv[p % (hd // 4)]
    axis = (p < hd // 4).long()  # 1 -> x, 0 -> y
    pos = meta.long()[:, axis].float() + 1.0  # [T, pairs]
    ang = pos * freq
    cos, sin = ang.cos()[:, None], ang.sin()[:, None]
    qk = qkv[:, :2 * num_heads * hd].view(T, 2 * num_heads, pairs, 2)
    e, o = qk[..., 0].float(), qk[..., 1].float()
    qk[..., 0] = (e * cos - o * sin).to(qkv.dtype)
    qk[..., 1] = (o * cos + e * sin).to(qkv.dtype)


def scaled_residual_norm_(hidden: torch.Tensor, update: torch.Tensor, gamma: torch.Tensor,
                          norm: nn.LayerNorm | None) -> torch.Tensor | None:
    """`hidden += update * gamma` in place (bf16 at upstream's points); returns
    `norm(hidden)`."""
    hs = _bf16(hidden.float() + _bf16(update.float() * gamma.float()))
    hidden.copy_(hs)
    return None if norm is None else _layer_norm(hs, norm).to(torch.bfloat16)


def _varlen_attention(q, k, v, md: VisionMetadata, scale: float) -> torch.Tensor:
    if q.is_xpu:
        from vllm._xpu_ops import xpu_ops

        return xpu_ops.flash_attn_varlen_func(
            q, k, v, md.cu_seqlens, md.max_seqlen, md.max_seqlen, softmax_scale=scale,
            causal=False, cu_seqlens_k=md.cu_seqlens)
    out = torch.empty_like(q)
    cu = md.cu_seqlens.tolist()
    for a, b in zip(cu[:-1], cu[1:]):
        out[a:b] = F.scaled_dot_product_attention(
            q[a:b].transpose(0, 1), k[a:b].transpose(0, 1), v[a:b].transpose(0, 1),
            scale=scale).transpose(0, 1)
    return out


class Inkling2VisionAttention(nn.Module):
    def __init__(self, width: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = width // num_heads
        self.qkv_proj = _linear(width, 3 * width)
        self.out_proj = _linear(width, width)

    def forward(self, hidden: torch.Tensor, md: VisionMetadata) -> torch.Tensor:
        qkv = self.qkv_proj(hidden)
        apply_rotary_qk_(qkv, md.meta, self.num_heads)
        q, k, v = qkv.view(-1, 3, self.num_heads, self.head_dim).unbind(1)
        out = _varlen_attention(q.contiguous(), k.contiguous(), v.contiguous(), md,
                                self.head_dim ** -0.5)
        return self.out_proj(out.reshape(hidden.shape))


class Inkling2VisionMLP(nn.Module):
    def __init__(self, width: int, intermediate: int) -> None:
        super().__init__()
        self.c_fc = _linear(width, intermediate)
        self.c_proj = _linear(intermediate, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(F.gelu(self.c_fc(x)))


class Inkling2VisionLayerScale(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.empty(width, dtype=torch.bfloat16))


class Inkling2VisionLayer(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        width, eps = config["hidden_size"], config["layer_norm_eps"]
        self.ln_1 = nn.LayerNorm(width, eps=eps, dtype=torch.bfloat16)
        self.self_attn = Inkling2VisionAttention(width, config["num_attention_heads"])
        self.ln_2 = nn.LayerNorm(width, eps=eps, dtype=torch.bfloat16)
        self.mlp = Inkling2VisionMLP(width, int(width * config["mlp_ratio"]))
        self.ls_1 = Inkling2VisionLayerScale(width)
        self.ls_2 = Inkling2VisionLayerScale(width)

    def forward(self, hidden, normed, md, next_norm):
        """Updates the residual `hidden` in place (`normed = ln_1(hidden)`); returns
        the next layer's `ln_1(hidden)`, if any."""
        normed = scaled_residual_norm_(hidden, self.self_attn(normed, md), self.ls_1.gamma,
                                       self.ln_2)
        return scaled_residual_norm_(hidden, self.mlp(normed), self.ls_2.gamma, next_norm)


class Inkling2VisionEmbeddings(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        width, grid = config["hidden_size"], config["posemb_grid_size"]
        self.patch_embedding = _linear(3 * config["native_patch_size"] ** 2, width, bias=False)
        self.position_embedding = nn.Parameter(torch.empty(grid * grid, width,
                                                           dtype=torch.bfloat16))
        self.class_embedding = nn.Parameter(torch.empty(width, dtype=torch.bfloat16))
        self.cls_position_embedding = nn.Parameter(torch.empty(width, dtype=torch.bfloat16))


class Inkling2VisionBackbone(nn.Module):
    def __init__(self, config: Mapping[str, Any]) -> None:
        super().__init__()
        self.embeddings = Inkling2VisionEmbeddings(config)
        self.ln_pre = nn.LayerNorm(config["hidden_size"], eps=config["layer_norm_eps"],
                                   dtype=torch.bfloat16)
        self.layers = nn.ModuleList(Inkling2VisionLayer(config)
                                    for _ in range(config["num_hidden_layers"]))


class Inkling2VisionEncoder(nn.Module):
    """Raw `[N, 40, 40, 3]` bf16 pixel cells (-1 = padding) to decoder embeddings: a ViT
    with a CLS token per image, then a 2x2 merge projection to the text width.
    Replicated on every TP rank. Stills only (the temporal frames are equal)."""

    def __init__(self, config: Mapping[str, Any], text_hidden_size: int) -> None:
        super().__init__()
        self.patch_size = config["native_patch_size"]
        self.merge_size = config["adapter_merge_size"]
        self.posemb_grid_size = config["posemb_grid_size"]
        self.backbone = Inkling2VisionBackbone(config)
        self.proj_in = _linear(self.merge_size ** 2 * config["hidden_size"],
                               config["projection_hidden_dim"], bias=config["projector_bias"])
        self.proj_out = _linear(config["projection_hidden_dim"], text_hidden_size,
                                bias=config["projector_bias"])

    def metadata(self, grid: list[list[int]], device: torch.device) -> VisionMetadata:
        m, p = self.merge_size, self.patch_size
        side = m * p
        frame = side * side * 3
        meta, patch_rows, offsets, pool_rows, lengths = [], [], [], [], []
        num_tokens = num_patches = num_rows = 0
        for _, h, w in grid:
            height, width = h * m, w * m
            y, x = np.divmod(np.arange(height * width, dtype=np.int64), width)
            im = np.empty((height * width + 1, 4), np.int64)
            im[0, :2] = -1
            im[1:, 0], im[1:, 1] = y, x
            im[:, 2], im[:, 3] = height, width
            meta.append(im)
            rows = np.arange(height * width + 1, dtype=np.int64) + num_rows - 1
            rows[0] = 0
            patch_rows.append(rows)
            cell = num_patches + (y // m) * w + x // m
            offsets.append(cell * frame + (y % m) * (p * side * 3) + (x % m) * (p * 3))
            merged = (y // m) * (w * m * m) + (x // m) * (m * m) + (y % m) * m + x % m
            pool = np.empty(height * width, np.int64)
            pool[merged] = num_tokens + 1 + np.arange(height * width)
            pool_rows.append(pool)
            lengths.append(height * width + 1)
            num_tokens += height * width + 1
            num_patches += h * w
            num_rows += height * width
        cu = np.cumsum([0, *lengths])

        def t(a):
            return torch.from_numpy(np.ascontiguousarray(a).astype(np.int32)).to(device)

        return VisionMetadata(t(np.concatenate(meta)), t(np.concatenate(patch_rows)),
                              t(np.concatenate(offsets)), t(np.concatenate(pool_rows)), t(cu),
                              int(max(lengths)))

    def forward(self, pixels: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
        side = self.patch_size * self.merge_size
        assert pixels.dtype == torch.bfloat16 and pixels.shape[1:] == (side, side, 3)
        grid = grid_thw.tolist()
        assert all(t == 1 for t, _, _ in grid)
        assert sum(h * w for _, h, w in grid) == pixels.shape[0]
        return self.encode(pixels.contiguous(), self.metadata(grid, pixels.device))

    def encode(self, pixels: torch.Tensor, md: VisionMetadata) -> torch.Tensor:
        bb = self.backbone
        emb = bb.embeddings
        patches = gather_patches(pixels, md.pixel_offsets, self.patch_size)
        g = self.posemb_grid_size
        hidden, normed = embed_tokens(emb.patch_embedding(patches), md,
                                      emb.position_embedding.view(g, g, -1),
                                      emb.class_embedding, emb.cls_position_embedding,
                                      bb.ln_pre, bb.layers[0].ln_1)
        layers = bb.layers
        for i, layer in enumerate(layers):
            normed = layer(hidden, normed, md, layers[i + 1].ln_1 if i + 1 < len(layers)
                           else None)
        pooled = hidden.index_select(0, md.pool_rows.long())
        pooled = pooled.view(-1, self.merge_size ** 2 * hidden.shape[-1])
        return self.proj_out(F.gelu(self.proj_in(pooled)))

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        """`model.visual.*` checkpoint tensors with that prefix stripped."""
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        qkv: Counter[str] = Counter()
        for name, weight in weights:
            for shard, proj in enumerate(("q_proj", "k_proj", "v_proj")):
                if f".self_attn.{proj}." in name:
                    name = name.replace(proj, "qkv_proj")
                    param = params[name]
                    n = param.shape[0] // 3
                    default_weight_loader(param.data[shard * n:(shard + 1) * n], weight)
                    qkv[name] += 1
                    break
            else:
                if name not in params:
                    raise ValueError(f"unexpected vision checkpoint tensor: {name}")
                default_weight_loader(params[name], weight)
                loaded.add(name)
        return loaded | {n for n, c in qkv.items() if c == 3}
