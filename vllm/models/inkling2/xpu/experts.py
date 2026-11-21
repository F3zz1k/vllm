# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 routed-expert storage formats (XPU port).

The checkpoint stores layers 0-19 in NVFP4 (two E2M1 codes per byte, low
nibble first, an E4M3 scale per 16 values and a per-expert factor
`amax / outer_scale_denominator`) and layers 20-23 in MXFP8 (E4M3 values, a
power-of-two scale `2 ** (s - 127)` per 32 values). The gate and up rows of
`w13` are interleaved (row 2i = gate i, row 2i + 1 = up i).

A layer is converted at load time into one storage format, chosen with
`VLLM_INKLING2_EXPERTS` (no code change, for A/B runs):

  bf16   dequantized weights (reference precision)
  fp8    E4M3 with one fp32 scale per expert and matrix
  nvfp4  the checkpoint bytes kept as they are (torch reference only for now);
         the MXFP8 layers are stored in bf16 (exact)

bf16 / fp8 are stored in XpuFusedMoe's layout: `w13` `(E, H, 2I)` with the
gate columns first, `w2` `(E, I, H)` (K-major); nvfp4 keeps the checkpoint
row layout `w13` `(E, 2I, H/2)` (de-interleaved to [gate; up]).
"""

from __future__ import annotations

import os

import torch

E2M1_VALUES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
               -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0)
FP8_MAX = 448.0
FORMATS = ("bf16", "fp8", "nvfp4")


def expert_format() -> str:
    fmt = os.environ.get("VLLM_INKLING2_EXPERTS", "fp8").lower()
    if fmt not in FORMATS:
        raise ValueError(f"VLLM_INKLING2_EXPERTS must be one of {FORMATS}, got {fmt!r}")
    return fmt


def _byte_table(device) -> torch.Tensor:
    vals = torch.tensor(E2M1_VALUES, device=device)
    byte = torch.arange(256, device=device)
    return torch.stack((vals[byte & 0xF], vals[byte >> 4]), dim=-1)  # (256, 2)


def deinterleave_rows(t: torch.Tensor) -> torch.Tensor:
    """(E, 2I, ...) with gate/up rows interleaved -> [gate; up]."""
    e, rows = t.shape[:2]
    return t.view(e, rows // 2, 2, *t.shape[2:]).transpose(1, 2).reshape(t.shape).contiguous()


def dequant_nvfp4(w: torch.Tensor, s: torch.Tensor, factor: torch.Tensor,
                  out_dtype=torch.float32, chunk: int = 8) -> torch.Tensor:
    """w uint8 (E, R, C/2), s e4m3 (E, R, C/16), factor (E,) -> (E, R, C)."""
    e, r = w.shape[:2]
    table = _byte_table(w.device)
    out = torch.empty(e, r, w.shape[2] * 2, dtype=out_dtype, device=w.device)
    for a in range(0, e, chunk):
        b = min(e, a + chunk)
        v = table[w[a:b].long()].view(b - a, r, -1, 16) * s[a:b].float()[..., None]
        out[a:b] = (v * factor[a:b, None, None, None].float()).reshape(b - a, r, -1).to(out_dtype)
    return out


def dequant_mxfp8(w: torch.Tensor, s: torch.Tensor, out_dtype=torch.float32,
                  chunk: int = 8) -> torch.Tensor:
    """w e4m3 (E, R, C), s uint8 exponent (E, R, C/32) -> (E, R, C)."""
    e, r, c = w.shape
    out = torch.empty(e, r, c, dtype=out_dtype, device=w.device)
    for a in range(0, e, chunk):
        b = min(e, a + chunk)
        v = w[a:b].float().view(b - a, r, -1, 32) * torch.exp2(s[a:b].float() - 127)[..., None]
        out[a:b] = v.reshape(b - a, r, c).to(out_dtype)
    return out


def quant_fp8_per_expert(w: torch.Tensor, chunk: int = 8) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32/bf16 (E, R, C) -> (e4m3 (E, R, C), fp32 scale (E,))."""
    e = w.shape[0]
    q = torch.empty(w.shape, dtype=torch.float8_e4m3fn, device=w.device)
    scale = torch.empty(e, dtype=torch.float32, device=w.device)
    for a in range(0, e, chunk):
        b = min(e, a + chunk)
        x = w[a:b].float()
        sc = x.abs().amax(dim=(1, 2)).clamp(min=1e-12) / FP8_MAX
        q[a:b] = (x / sc[:, None, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
        scale[a:b] = sc
    return q, scale


class ExpertStore:
    """One layer's routed experts in the chosen storage format."""

    def __init__(self, fmt: str):
        self.fmt = fmt
        self.t: dict[str, torch.Tensor] = {}

    # ---------------------------------------------------------------- loading

    @staticmethod
    def decode(parts: dict[str, torch.Tensor], name: str, out_dtype=torch.float32) -> torch.Tensor:
        """Full-precision `name` ('w13' or 'w2') from its checkpoint parts,
        with w13 de-interleaved to [gate; up]."""
        w = parts[f"{name}_weight"]
        s = parts[f"{name}_weight_scale"]
        if f"{name}_weight_amax" in parts:
            factor = parts[f"{name}_weight_amax"].float() / parts[
                f"{name}_weight_outer_scale_denominator"].float()
            full = dequant_nvfp4(w, s.view(torch.float8_e4m3fn), factor, out_dtype)
        else:
            full = dequant_mxfp8(w.view(torch.float8_e4m3fn), s.view(torch.uint8), out_dtype)
        return deinterleave_rows(full) if name == "w13" else full

    def build(self, parts: dict[str, torch.Tensor], device) -> None:
        """Convert a layer's checkpoint expert tensors into this format."""
        parts = {k: v.to(device) for k, v in parts.items()}
        nvfp4_src = "w13_weight_amax" in parts
        if self.fmt == "nvfp4":
            if not nvfp4_src:
                # MXFP8 layers have no NVFP4 bytes: store them exactly (bf16).
                self.fmt = "bf16"
            else:
                for name in ("w13", "w2"):
                    w, s = parts[f"{name}_weight"], parts[f"{name}_weight_scale"]
                    if name == "w13":
                        w, s = deinterleave_rows(w), deinterleave_rows(s)
                    self.t[f"{name}_packed"] = w.view(torch.uint8).contiguous()
                    self.t[f"{name}_bscale"] = s.view(torch.float8_e4m3fn).contiguous()
                    self.t[f"{name}_factor"] = (
                        parts[f"{name}_weight_amax"].float()
                        / parts[f"{name}_weight_outer_scale_denominator"].float())
                return
        for name in ("w13", "w2"):
            w = parts[f"{name}_weight"]
            e, rows = w.shape[:2]
            cols = w.shape[2] * (2 if nvfp4_src else 1)
            dt = torch.bfloat16 if self.fmt == "bf16" else torch.float8_e4m3fn
            out = torch.empty((e, cols, rows), dtype=dt, device=device)  # (E, K, N)
            scale = torch.empty(e, dtype=torch.float32, device=device)
            for a in range(0, e, 16):
                b = min(e, a + 16)
                sub = {k: (v[a:b] if v.dim() > 1 or k.endswith("amax") else v)
                       for k, v in parts.items() if k.startswith(name + "_")}
                full = self.decode(sub, name)  # (n, rows, cols) fp32
                if self.fmt == "bf16":
                    out[a:b] = full.transpose(1, 2).to(torch.bfloat16)
                else:
                    q, sc = quant_fp8_per_expert(full)
                    out[a:b] = q.transpose(1, 2)
                    scale[a:b] = sc
                del full
            self.t[name] = out
            if self.fmt == "fp8":
                self.t[f"{name}_scale"] = scale

    # -------------------------------------------------------------- reference

    def expert_weights(self, e: int) -> tuple[torch.Tensor, torch.Tensor]:
        """bf16 (w13 (2I, H), w2 (H, I)) of expert `e` (torch reference)."""
        out = []
        for name in ("w13", "w2"):
            if self.fmt == "bf16":
                out.append(self.t[name][e].t())
            elif self.fmt == "fp8":
                out.append((self.t[name][e].t().float() * self.t[f"{name}_scale"][e]).to(
                    torch.bfloat16))
            else:
                w = dequant_nvfp4(self.t[f"{name}_packed"][e:e + 1], self.t[f"{name}_bscale"][e:e + 1],
                                  self.t[f"{name}_factor"][e:e + 1], torch.bfloat16)
                out.append(w[0])
        return out[0], out[1]
