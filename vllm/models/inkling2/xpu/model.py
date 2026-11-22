# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2 (looped) for Intel XPU.

Structure follows transformers' `Inkling2Loop` (the reference): 24 decoder
layers tied by block attention residuals (AttnRes); layers
`[core_start, core_end)` run `num_passes` times, each pass from the state
after the prelude, passes after the first blending gated low-rank messages
from the previous pass into every sublayer input. Every (layer, pass) visit
has its own KV cache and short-conv state (`cache.py`).

The ops are dispatched through `ops` (torch references in `ops/ref.py`;
Triton kernels where available).
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import regex as re
import torch
import torch.nn.functional as F
from torch import nn

from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.sequence import IntermediateTensors

from ..configs import Inkling2MMConfig, Inkling2ModelConfig
from . import ops
from .cache import Inkling2ConvState, Inkling2KVCache, layer_metadata
from .experts import ExpertStore, expert_format
from .ops.ref import ConvLayout

logger = init_logger(__name__)

LOOP_STREAMS = ("attn", "k", "v", "mlp")


def _param(*shape, dtype=torch.bfloat16) -> nn.Parameter:
    return nn.Parameter(torch.empty(*shape, dtype=dtype), requires_grad=False)


class Inkling2Attention(nn.Module):
    """Relative-bias GQA with short-conv K/V, L2-clipped q/k/r, gated output
    and per-token sinks. One set of weights; one KV cache and one conv state
    per loop pass."""

    def __init__(self, config: Inkling2ModelConfig, layer_id: int, num_passes: int,
                 prefix: str) -> None:
        super().__init__()
        self.is_local = layer_id in config.local_layer_ids
        c = config
        self.num_heads = c.swa_num_attention_heads if self.is_local else c.num_attention_heads
        self.num_kv_heads = c.swa_num_key_value_heads if self.is_local else c.num_key_value_heads
        self.head_dim = c.swa_head_dim if self.is_local else c.head_dim
        self.d_rel = c.d_rel
        self.window = c.sliding_window_size if self.is_local else None
        self.rel_extent = c.sliding_window_size if self.is_local else c.rel_extent
        self.kernel = c.sconv_kernel_size
        self.scale = self.head_dim ** -0.5
        self.qk_clip = c.qk_l2_clip_threshold
        self.rel_clip = c.rel_l2_clip_threshold
        self.eps = c.rms_norm_eps
        H, Hk, D, R = self.num_heads, self.num_kv_heads, self.head_dim, c.d_rel
        dyn = 0 if self.is_local else Hk * self.kernel
        # [q | r | gate | sink] from the query input, [k | k taps], [v | v taps].
        self.q_sizes = (H * D, H * R, H * D, H)
        self.k_size, self.v_size, self.dyn = Hk * D, Hk * D, dyn
        rows = sum(self.q_sizes) + 2 * (Hk * D + dyn)
        self.qkvr = _param(rows, c.hidden_size)
        self.wo = _param(c.hidden_size, H * D)
        self.q_scale = _param(H, D)
        self.r_scale = _param(H, R)
        self.rel_proj = _param(R, self.rel_extent)
        if self.is_local:
            self.k_conv = _param(Hk * D, self.kernel, dtype=torch.float32)
            self.v_conv = _param(Hk * D, self.kernel, dtype=torch.float32)
        else:
            self.k_static = _param(self.kernel, Hk * D)
            self.v_static = _param(self.kernel, Hk * D)
        self.layout = ConvLayout.make(Hk, D, c.hidden_size)
        self.kv_caches = nn.ModuleList(
            Inkling2KVCache(f"{prefix}.kv.p{p}", H, Hk, D, self.window) for p in range(num_passes))
        self.conv_states = nn.ModuleList(
            Inkling2ConvState(f"{prefix}.conv.p{p}", self.layout, self.kernel)
            for p in range(num_passes))

    # Row offsets of the fused projection's shards.
    def shard_offsets(self) -> dict[str, tuple[int, int]]:
        names = ["q", "r", "gate", "sink"]
        out, off = {}, 0
        for n, s in zip(names, self.q_sizes):
            out[n] = (off, s)
            off += s
        out["k"] = (off, self.k_size)
        off += self.k_size
        if self.dyn:
            out["kdyn"] = (off, self.dyn)
            off += self.dyn
        out["v"] = (off, self.v_size)
        off += self.v_size
        if self.dyn:
            out["vdyn"] = (off, self.dyn)
        return out

    def _project(self, xq, xk, xv):
        nq = sum(self.q_sizes)
        nk = self.k_size + self.dyn
        if xq is xk and xk is xv:
            y = F.linear(xq, self.qkvr)
            return y[:, :nq], y[:, nq:nq + nk], y[:, nq + nk:]
        return (F.linear(xq, self.qkvr[:nq]), F.linear(xk, self.qkvr[nq:nq + nk]),
                F.linear(xv, self.qkvr[nq + nk:]))

    def forward(self, positions, xq, xk, xv, tau, pass_idx: int) -> torch.Tensor:
        T = xq.shape[0]
        H, Hk, D, R = self.num_heads, self.num_kv_heads, self.head_dim, self.d_rel
        yq, yk, yv = self._project(xq, xk, xv)
        q_lin, r_lin, gate_lin, sink = yq.split(self.q_sizes, -1)
        k_lin, v_lin = yk[:, :self.k_size], yv[:, :self.v_size]
        kv_owner = self.kv_caches[pass_idx]
        conv_owner = self.conv_states[pass_idx]
        conv_md = layer_metadata(conv_owner.prefix)
        attn_md = layer_metadata(kv_owner.prefix)

        # q / r: learned scale, L2 clip (+ log scaling on global layers), and the
        # relative logits (one fused kernel when the SYCL attention serves the cache)
        fused = attn_md is not None and ops.sycl_ready(kv_owner)
        q, rel_logits = ops.qkr(self, yq, tau, fused)

        # k / v: short conv (static or dynamic) over the conv cache, then
        # L2 clip (k) or unweighted RMS norm (v), stored into the KV cache.
        kw = dict(kv_owner=kv_owner, kv_md=attn_md)
        if self.is_local:
            ops.conv(conv_owner, conv_md, "k", k_lin, positions, self.k_conv, norm=1,
                     clip=self.qk_clip, **kw)
            ops.conv(conv_owner, conv_md, "v", v_lin, positions, self.v_conv, norm=2,
                     eps=self.eps, **kw)
        else:
            ops.conv(conv_owner, conv_md, "k", k_lin, positions, self.k_static,
                     dyn=yk[:, self.k_size:], norm=1, clip=self.qk_clip, **kw)
            ops.conv(conv_owner, conv_md, "v", v_lin, positions, self.v_static,
                     dyn=yv[:, self.v_size:], norm=2, eps=self.eps, **kw)

        if attn_md is None:
            out = torch.zeros(T, H * D, dtype=torch.bfloat16, device=xq.device)
        else:  # sinks and the sigmoid output gate applied inside
            out = ops.attention(kv_owner, attn_md, q, rel_logits, sink, self.scale,
                                self.window, self.rel_extent, gate=gate_lin)
        return F.linear(out, self.wo)


class Inkling2MoE(nn.Module):
    def __init__(self, config: Inkling2ModelConfig, layer_id: int) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.num_experts = config.n_routed_experts
        self.top_k = config.num_experts_per_tok
        self.route_scale = config.route_scale
        self.gate_cap = config.mlp_gate_softcap
        self.up_cap = config.mlp_up_softcap
        self.gate_weight = _param(self.num_experts, config.hidden_size, dtype=torch.float32)
        self.gate_bias = _param(self.num_experts, dtype=torch.float32)
        self.store = ExpertStore(expert_format())
        self._parts: dict[str, torch.Tensor] = {}
        self._kernel = None

    def load_part(self, name: str, tensor: torch.Tensor) -> None:
        """Collect a layer's checkpoint expert tensors; convert once complete
        (NVFP4 layers have 8 tensors, MXFP8 layers 4)."""
        self._parts[name] = tensor
        w13 = self._parts.get("w13_weight")
        if w13 is None:
            return
        need = 8 if w13.dtype == torch.uint8 else 4
        if len(self._parts) < need:
            return
        self.store.build(self._parts, self.gate_weight.device)
        self._parts = {}
        for k, v in self.store.t.items():
            self.register_buffer(f"e_{k}", v, persistent=False)

    def expert_weights(self, e: int):
        return self.store.expert_weights(e)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w, ids = ops.route_fused(x, self.gate_weight, self.gate_bias, self.top_k,
                                 self.route_scale)
        return ops.experts(self, x, w, ids)


class Inkling2DecoderLayer(nn.Module):
    def __init__(self, config: Inkling2ModelConfig, layer_id: int, prefix: str) -> None:
        super().__init__()
        self.layer_id = layer_id
        self.is_local = layer_id in config.local_layer_ids
        self.is_core = layer_id in config.loop_core_layers
        self.boundary = layer_id % config.n_layers_per_attn_res_block == 0
        self.num_passes = config.num_loop_passes if self.is_core else 1
        self.eps = config.rms_norm_eps
        D = config.hidden_size
        self.kernel = config.sconv_kernel_size
        self.attn = Inkling2Attention(config, layer_id, self.num_passes, f"{prefix}.attn")
        self.attn_norm = _param(1 if self.is_local else 3, D)
        self.mlp_norm = _param(D)
        self.attn_conv = _param(D, self.kernel, dtype=torch.float32)
        self.mlp_conv = _param(D, self.kernel, dtype=torch.float32)
        self.mlp = Inkling2MoE(config, layer_id)
        if self.is_core:
            r = config.loop_config["gate_rank"]
            self.ctrl_down = _param(len(LOOP_STREAMS), r, 2 * D)
            self.ctrl_gain = _param(len(LOOP_STREAMS), D, r, dtype=torch.float32)
            self.ctrl_forget = _param(len(LOOP_STREAMS), D, r, dtype=torch.float32)

    def forward(self, positions, h, stack, n: int, attn_query, mlp_query, messages, tau,
                pass_idx: int):
        """One visit. `h` is the layer input (bf16); `stack[:n]` the completed
        AttnRes blocks (a persistent buffer). Returns (layer output bf16, n):
        a block-boundary layer appends `h` to the stack."""
        conv_owner = self.attn.conv_states[pass_idx]
        conv_md = layer_metadata(conv_owner.prefix)
        if attn_query is None:  # layer 0: no mix
            xq = xk = xv = ops.rms_norm(h, self.attn_norm[0], self.eps)
        elif messages is None:  # mixes and their RMSNorms in one kernel
            xs = ops.attn_res(stack, n, h, None, attn_query, self.attn_norm, self.eps)
            xq, xk, xv = (xs[0], xs[0], xs[0]) if self.is_local else xs.unbind(0)
        else:
            mixes = ops.attn_res(stack, n, h, None, attn_query)  # fp32 (1 or 3, T, D)
            dests = mixes.expand(3, -1, -1) if self.is_local else mixes
            w = self.attn_norm.expand(3, -1) if self.is_local else self.attn_norm
            xq, xk, xv = ops.blend(self, slice(0, 3), dests, messages[:3], w, self.eps).unbind(0)

        a = self.attn(positions, xq, xk, xv, None if self.is_local else tau, pass_idx)
        a = ops.conv(conv_owner, conv_md, "attn", a, positions, self.attn_conv)
        if self.boundary:
            stack[n].copy_(h)
            n += 1
            last, addend = a, None
        else:
            last, addend = h, a
        if messages is None:
            m = ops.attn_res(stack, n, last, addend, mlp_query, self.mlp_norm.view(1, -1),
                             self.eps)[0]
        else:
            m = ops.attn_res(stack, n, last, addend, mlp_query)[0]
            m = ops.blend(self, slice(3, 4), m[None], messages[3:], self.mlp_norm.view(1, -1),
                          self.eps)[0]
        m = self.mlp(m)
        # the MLP-output conv also adds the residual (last (+ addend)): the layer output
        return ops.conv(conv_owner, conv_md, "mlp", m, positions, self.mlp_conv, last=last,
                        addend=addend), n


class Inkling2LoopModel(nn.Module):
    def __init__(self, config: Inkling2ModelConfig, prefix: str) -> None:
        super().__init__()
        self.config = config
        D = config.hidden_size
        self.eps = config.rms_norm_eps
        self.embed_tokens = VocabParallelEmbedding(config.padded_vocab_size, D)
        self.embed_norm = _param(D)
        self.norm = _param(D)
        self.layers = nn.ModuleList(
            Inkling2DecoderLayer(config, i, f"{prefix}.layers.{i}")
            for i in range(config.num_hidden_layers))
        local = set(config.local_layer_ids)
        n = config.num_hidden_layers
        # attn_queries[i] belongs to layer i + 1 (layer 0 mixes nothing).
        self.attn_queries = nn.ParameterList(
            _param(1 if i in local else 3, D, dtype=torch.float32) for i in range(1, n))
        self.mlp_queries = nn.ParameterList(_param(1, D, dtype=torch.float32) for _ in range(n))
        self.output_query = _param(1, D, dtype=torch.float32)
        core = config.loop_core_layers
        self.core = core
        self.cross_queries = _param(2 * len(core), D, dtype=torch.float32)
        self.cross_kv_queries = _param(2 * len(core), D, dtype=torch.float32)
        self.register_buffer("message_queries",
                             torch.zeros(len(core), len(LOOP_STREAMS), D, dtype=torch.float32),
                             persistent=False)
        self.scale = D ** -0.5
        # completed AttnRes blocks held at once (each pass restarts from the prelude's)
        nb = mx = 0
        for i in [*range(core.start), *core, *range(core.stop, config.num_hidden_layers)]:
            nb += i % config.n_layers_per_attn_res_block == 0
            mx = max(mx, nb)
        self.max_blocks = mx

    def finish_loading(self) -> None:
        for layer in self.layers:
            if layer.is_core:  # [gain; forget] stacked for one batched GEMM per stream set
                layer.register_buffer("ctrl_gf", torch.cat([layer.ctrl_gain.data,
                                                            layer.ctrl_forget.data], 1),
                                      persistent=False)
        with torch.no_grad():
            for p in [*self.attn_queries, *self.mlp_queries, self.output_query]:
                p.mul_(self.scale)
            q = self.cross_queries.view(-1, 2, self.config.hidden_size)
            kv = self.cross_kv_queries.view(-1, 2, self.config.hidden_size)
            stacked = torch.stack((q[:, 0], kv[:, 0], kv[:, 1], q[:, 1]), 1)
            self.message_queries.copy_(stacked * self.scale)

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return ops.rms_norm(self.embed_tokens(input_ids), self.embed_norm, self.eps)

    def _visit(self, i, p, positions, h, stack, n, messages, tau):
        attn_query = None if i == 0 else self.attn_queries[i - 1]
        return self.layers[i](positions, h, stack, n, attn_query, self.mlp_queries[i], messages,
                              tau, p)

    def forward(self, positions: torch.Tensor, inputs_embeds: torch.Tensor) -> torch.Tensor:
        c = self.config
        tau = None
        if c.log_scaling_n_floor is not None:
            tau = ops.log_scaling_tau(positions, c.log_scaling_n_floor, c.log_scaling_alpha)
        core = self.core
        T, D = inputs_embeds.shape
        stack = inputs_embeds.new_empty((self.max_blocks, T, D))
        h, n = inputs_embeds, 0
        for i in range(core.start):
            h, n = self._visit(i, 0, positions, h, stack, n, None, tau)
        pre_h, n_pre = h, n
        for i in core:
            h, n = self._visit(i, 0, positions, h, stack, n, None, tau)
        for p in range(1, c.num_loop_passes):
            # the previous pass's blocks after the prelude (+1: skip the pre-loop
            # partial its first layer appended) and its output form the bank
            bank, nb, prev_h = stack[n_pre + 1:n].clone(), n - n_pre - 1, h
            h, n = pre_h, n_pre
            for j, i in enumerate(core):
                messages = ops.attn_res(bank, nb, prev_h, None, self.message_queries[j])
                h, n = self._visit(i, p, positions, h, stack, n, messages, tau)
        for i in range(core.stop, c.num_hidden_layers):
            h, n = self._visit(i, 0, positions, h, stack, n, None, tau)
        return ops.attn_res(stack, n, h, None, self.output_query, self.norm.view(1, -1),
                            self.eps)[0]


class Inkling2LoopForConditionalGeneration(nn.Module):
    """Text path of Inkling 2 Mini on XPU (vision: not yet)."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super().__init__()
        config: Inkling2MMConfig = vllm_config.model_config.hf_config
        assert get_tensor_model_parallel_world_size() == 1, "TP > 1: not yet"
        self.config = config
        self.text_config = tc = config.text_config
        self.model = Inkling2LoopModel(tc, f"{prefix}model" if prefix else "model")
        self.lm_head = ParallelLMHead(tc.padded_vocab_size, tc.hidden_size)
        self.logits_processor = LogitsProcessor(
            tc.padded_vocab_size, org_vocab_size=tc.vocab_size,
            soft_cap=tc.final_logit_softcapping)
        # The config's fp32 head would cast the 0.9 GB weight to fp32 on every step
        # (~9 ms/token on XPU); the reference runs the head in bf16, so do the same.
        self.logits_processor.head_dtype = None

    def get_language_model(self) -> nn.Module:
        return self

    def embed_input_ids(self, input_ids: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(self, input_ids: torch.Tensor | None, positions: torch.Tensor,
                intermediate_tensors: IntermediateTensors | None = None,
                inputs_embeds: torch.Tensor | None = None, **kwargs) -> torch.Tensor:
        if inputs_embeds is None:
            inputs_embeds = self.model.embed_input_ids(input_ids)
        return self.model(positions, inputs_embeds)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        return self.logits_processor(self.lm_head, hidden_states)

    # ---------------------------------------------------------------- weights

    _simple = (
        (r"^model\.llm\.embed\.weight$", "model.embed_tokens.weight"),
        (r"^model\.llm\.embed_norm\.weight$", "model.embed_norm"),
        (r"^model\.llm\.norm\.weight$", "model.norm"),
        (r"^model\.llm\.unembed\.weight$", "lm_head.weight"),
        (r"^model\.llm\.attn_res\.attn_queries\.(\d+)$", r"model.attn_queries.\1"),
        (r"^model\.llm\.attn_res\.mlp_queries\.(\d+)$", r"model.mlp_queries.\1"),
        (r"^model\.llm\.attn_res\.output_query$", "model.output_query"),
        (r"^model\.llm\.middle_cycle_cross_pass_queries$", "model.cross_queries"),
        (r"^model\.llm\.middle_cycle_cross_pass_kv_queries$", "model.cross_kv_queries"),
        (r"^model\.llm\.layers\.(\d+)\.attn\.wo_ud\.weight$", r"model.layers.\1.attn.wo"),
        (r"^model\.llm\.layers\.(\d+)\.attn\.q_l2_clip\.scale$", r"model.layers.\1.attn.q_scale"),
        (r"^model\.llm\.layers\.(\d+)\.attn\.rel_l2_clip\.scale$", r"model.layers.\1.attn.r_scale"),
        (r"^model\.llm\.layers\.(\d+)\.attn\.rel_logits_proj\.proj$", r"model.layers.\1.attn.rel_proj"),
        (r"^model\.llm\.layers\.(\d+)\.attn\.k_sconv\.weight$", r"model.layers.\1.attn.k_conv"),
        (r"^model\.llm\.layers\.(\d+)\.attn\.v_sconv\.weight$", r"model.layers.\1.attn.v_conv"),
        (r"^model\.llm\.layers\.(\d+)\.attn\.k_sconv\.static_w$", r"model.layers.\1.attn.k_static"),
        (r"^model\.llm\.layers\.(\d+)\.attn\.v_sconv\.static_w$", r"model.layers.\1.attn.v_static"),
        (r"^model\.llm\.layers\.(\d+)\.attn_sconv\.weight$", r"model.layers.\1.attn_conv"),
        (r"^model\.llm\.layers\.(\d+)\.mlp_sconv\.weight$", r"model.layers.\1.mlp_conv"),
        (r"^model\.llm\.layers\.(\d+)\.mlp_norm\.weight$", r"model.layers.\1.mlp_norm"),
        (r"^model\.llm\.layers\.(\d+)\.mlp\.gate\.weight$", r"model.layers.\1.mlp.gate_weight"),
        (r"^model\.llm\.layers\.(\d+)\.mlp\.gate\.bias$", r"model.layers.\1.mlp.gate_bias"),
    )
    _qkvr = {"wq_du": "q", "wr_du": "r", "gate_attn": "gate", "attn_sink_proj": "sink",
             "wk_dv": "k", "wv_dv": "v"}
    _norm_rows = {"attn_norm": 0, "q_input_norm": 0, "k_input_norm": 1, "v_input_norm": 2}
    _ctrl = {"down_weight": "ctrl_down", "gain_weight": "ctrl_gain",
             "forget_weight": "ctrl_forget"}

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()

        def put(name: str, tensor: torch.Tensor, rows: tuple[int, int] | None = None,
                index: int | None = None) -> None:
            p = params[name]
            dst = p.data if index is None else p.data[index]
            if rows is not None:
                dst = dst[rows[0]:rows[0] + rows[1]]
            if name.endswith(("k_conv", "v_conv", "attn_conv", "mlp_conv")):
                tensor = tensor.squeeze(1)  # (C, 1, k) -> (C, k)
            if name == "lm_head.weight" or name == "model.embed_tokens.weight":
                default_weight_loader(p, tensor)
            else:
                if "quer" in name and dst.numel() == tensor.numel():
                    tensor = tensor.reshape(dst.shape)  # (D,) pseudo-queries -> (1, D)
                assert dst.shape == tensor.shape, (name, dst.shape, tensor.shape)
                dst.copy_(tensor)
            loaded.add(name)

        for ckpt_name, tensor in weights:
            if not ckpt_name.startswith("model.llm."):
                continue  # vision tower: not yet
            for pat, repl in self._simple:
                if re.match(pat, ckpt_name):
                    put(re.sub(pat, repl, ckpt_name), tensor)
                    break
            else:
                m = re.match(r"^model\.llm\.layers\.(\d+)\.(.+)$", ckpt_name)
                if m is None:
                    raise KeyError(f"unexpected weight {ckpt_name}")
                lid, rest = int(m.group(1)), m.group(2)
                layer = self.model.layers[lid]
                base = f"model.layers.{lid}"
                if (mm := re.match(r"^attn\.(\w+)\.weight$", rest)) and mm.group(1) in self._qkvr:
                    off = layer.attn.shard_offsets()[self._qkvr[mm.group(1)]]
                    put(f"{base}.attn.qkvr", tensor, rows=off)
                elif mm := re.match(r"^attn\.(k|v)_sconv\.weight_proj$", rest):
                    off = layer.attn.shard_offsets()[f"{mm.group(1)}dyn"]
                    put(f"{base}.attn.qkvr", tensor, rows=off)
                elif mm := re.match(r"^(attn_norm|[qkv]_input_norm)\.weight$", rest):
                    put(f"{base}.attn_norm", tensor, index=self._norm_rows[mm.group(1)])
                elif mm := re.match(r"^controllers\.(attn|k|v|mlp)\.(\w+)$", rest):
                    put(f"{base}.{self._ctrl[mm.group(2)]}", tensor,
                        index=LOOP_STREAMS.index(mm.group(1)))
                elif mm := re.match(r"^mlp\.experts\.(\w+)$", rest):
                    layer.mlp.load_part(mm.group(1), tensor)
                    loaded.add(f"{base}.mlp.experts")
                else:
                    raise KeyError(f"unexpected weight {ckpt_name}")
        self.model.finish_loading()
        missing = [n for n in params if n not in loaded]
        if missing:
            raise ValueError(f"Inkling2: weights not loaded: {missing[:10]} ({len(missing)})")
        for layer in self.model.layers:
            if layer.mlp._parts:
                raise ValueError(f"layer {layer.layer_id}: incomplete expert tensors")
        return loaded | set(params)
