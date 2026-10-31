# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Derivation of canonical page mappings for KV offloading.

The only place in the offloading stack that reasons about parallelism
(TP/DCP/PCP); everything downstream consumes byte mappings. The canonical
page of a layer is the full offloaded block without parallelism: all KV
heads, all block_size * dcp * pcp tokens, in the worker's page encoding.
For a gated delta net layer it is the TP=1 state page. Uncertifiable layers
get an opaque fallback mapping (fail closed).
"""

from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.ssm_conv_transfer_utils import (
    derive_mamba_conv_split,
)
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import (
    AttentionSpec,
    KVCacheConfig,
    KVCacheLayout,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.kv_offload.base import CanonicalPageMapping, CopyRun

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# Version of the canonical byte format; bump on any layout change
CANONICAL_FORMAT_VERSION = 1


def canonical_format_id(kv_cache_layout: str) -> str:
    """Identity of the canonical byte format, for namespacing persisted KV.
    Canonical pages keep the worker's KV layout family, so the id couples the
    format version with that family; consumers must match it exactly. The family
    keeps its historical NHD/HND spelling so ids stay stable for KV persisted
    before the layout enum existed."""
    layout = KVCacheLayout[kv_cache_layout]
    legacy = {KVCacheLayout.LBNHC: "nhd", KVCacheLayout.LBHNC: "hnd"}
    family = legacy.get(layout, layout.name.lower())
    return f"v{CANONICAL_FORMAT_VERSION}-{family}"


@dataclass(frozen=True)
class _RankContext:
    """Sharding parameters of one worker rank within the offload group."""

    tp_size: int
    dcp_size: int
    pcp_size: int
    interleave: int
    total_kv_heads: int
    rank: int

    @property
    def cp_size(self) -> int:
        return self.dcp_size * self.pcp_size

    @property
    def tp_rank(self) -> int:
        return self.rank % self.tp_size

    @property
    def total_cp_rank(self) -> int:
        pcp_rank = self.rank // self.tp_size
        return pcp_rank * self.dcp_size + self.tp_rank % self.dcp_size


@dataclass(frozen=True)
class ByteRegion:
    """A byte region within a page that repeats once per token."""

    local_offset: int
    canonical_offset: int
    bytes_per_token: int
    canonical_token_stride: int


def _coalesce_runs(runs: list[CopyRun]) -> tuple[CopyRun, ...]:
    """Collapse contiguous fragments within and across runs to minimize the
    number of copy ops (e.g. a single-rank mapping becomes one whole-page run).
    """
    out: list[CopyRun] = []
    for run in runs:
        if (
            run.num_fragments > 1
            and run.local_stride == run.fragment_size
            and run.canonical_stride == run.fragment_size
        ):
            size = run.fragment_size * run.num_fragments
            run = CopyRun(run.local_offset, run.canonical_offset, size, 1, size, size)
        prev = out[-1] if out else None
        if (
            prev is not None
            and prev.num_fragments == 1
            and run.num_fragments == 1
            and prev.local_offset + prev.fragment_size == run.local_offset
            and prev.canonical_offset + prev.fragment_size == run.canonical_offset
        ):
            size = prev.fragment_size + run.fragment_size
            out[-1] = CopyRun(
                prev.local_offset, prev.canonical_offset, size, 1, size, size
            )
        else:
            out.append(run)
    return tuple(out)


def _local_to_canonical_token(local_idx: int, ctx: _RankContext) -> int:
    """Canonical position of one of this rank's local token indices."""
    chunk, pos_in_chunk = divmod(local_idx, ctx.interleave)
    return (chunk * ctx.cp_size + ctx.total_cp_rank) * ctx.interleave + pos_in_chunk


def _interleave_cp_tokens(
    regions: list[ByteRegion],
    num_tokens: int,
    ctx: _RankContext,
) -> tuple[CopyRun, ...]:
    """Place each region's num_tokens rows at their canonical token positions,
    one run per chunk of interleaved tokens."""
    runs: list[CopyRun] = []
    for region in regions:
        if ctx.cp_size == 1:
            runs.append(
                CopyRun(
                    region.local_offset,
                    region.canonical_offset,
                    region.bytes_per_token,
                    num_tokens,
                    region.bytes_per_token,
                    region.canonical_token_stride,
                )
            )
            continue
        for chunk_start in range(0, num_tokens, ctx.interleave):
            canonical_token = _local_to_canonical_token(chunk_start, ctx)
            runs.append(
                CopyRun(
                    region.local_offset + chunk_start * region.bytes_per_token,
                    region.canonical_offset
                    + canonical_token * region.canonical_token_stride,
                    region.bytes_per_token,
                    ctx.interleave,
                    region.bytes_per_token,
                    region.canonical_token_stride,
                )
            )
    return _coalesce_runs(runs)


def _packed_kv_regions(
    kv_cache: torch.Tensor,
    spec: AttentionSpec,
    head_shard: int,
    num_head_shards: int,
    cp_size: int,
) -> list[ByteRegion] | None:
    """K and V adjacent per (token, head), in NHD or HND stride order."""
    bs, heads = spec.block_size, spec.num_kv_heads
    elem = kv_cache.element_size()
    head_elems = 2 * spec.head_size
    if heads * bs * head_elems * elem != spec.real_page_size_bytes:
        return None
    _, head_stride, token_stride, inner_stride = kv_cache.stride()
    if inner_stride != 1:
        return None
    head_bytes = head_elems * elem
    token_row_bytes = heads * head_bytes

    if head_stride == head_elems and token_stride == heads * head_elems:  # NHD
        return [
            ByteRegion(
                local_offset=0,
                canonical_offset=head_shard * token_row_bytes,
                bytes_per_token=token_row_bytes,
                canonical_token_stride=num_head_shards * token_row_bytes,
            )
        ]

    if head_stride == bs * head_elems and token_stride == head_elems:  # HND
        canonical_span = bs * cp_size  # canonical tokens per offloaded block
        return [
            ByteRegion(
                local_offset=head * bs * head_bytes,
                canonical_offset=(head_shard * heads + head)
                * canonical_span
                * head_bytes,
                bytes_per_token=head_bytes,
                canonical_token_stride=head_bytes,
            )
            for head in range(heads)
        ]
    return None


def _split_kv_regions(
    kv_cache: torch.Tensor,
    spec: AttentionSpec,
    head_shard: int,
    num_head_shards: int,
    cp_size: int,
) -> list[ByteRegion] | None:
    """K and V in separate page halves, in NHD or HND stride order."""
    bs, heads, head_size = spec.block_size, spec.num_kv_heads, spec.head_size
    elem = kv_cache.element_size()
    if 2 * bs * heads * head_size * elem != spec.real_page_size_bytes:
        return None
    _, half_stride, token_stride, head_stride, inner_stride = kv_cache.stride()
    if inner_stride != 1 or half_stride != bs * heads * head_size:
        return None
    head_bytes = head_size * elem
    token_row_bytes = heads * head_bytes
    canonical_span = bs * cp_size  # canonical tokens per offloaded block

    if token_stride == heads * head_size and head_stride == head_size:  # NHD
        canonical_half_bytes = canonical_span * num_head_shards * token_row_bytes
        return [
            ByteRegion(
                local_offset=half * bs * token_row_bytes,
                canonical_offset=half * canonical_half_bytes
                + head_shard * token_row_bytes,
                bytes_per_token=token_row_bytes,
                canonical_token_stride=num_head_shards * token_row_bytes,
            )
            for half in range(2)  # K, then V
        ]

    if head_stride == bs * head_size and token_stride == head_size:  # HND
        local_half_bytes = bs * heads * head_bytes
        total_heads = num_head_shards * heads
        return [
            ByteRegion(
                local_offset=half * local_half_bytes + head * bs * head_bytes,
                canonical_offset=(half * total_heads + head_shard * heads + head)
                * canonical_span
                * head_bytes,
                bytes_per_token=head_bytes,
                canonical_token_stride=head_bytes,
            )
            for half in range(2)
            for head in range(heads)
        ]
    return None


def _attention_byte_regions(
    kv_cache: torch.Tensor,
    spec: AttentionSpec,
    num_blocks: int,
    head_shard: int,
    num_head_shards: int,
    cp_size: int,
) -> list[ByteRegion] | None:
    """Byte regions of an attention page, given this rank's head shard.
    None when the physical layout is not recognized (fail closed)."""
    bs, heads, head_size = spec.block_size, spec.num_kv_heads, spec.head_size
    if tuple(kv_cache.shape) == (num_blocks, heads, bs, 2 * head_size):
        return _packed_kv_regions(kv_cache, spec, head_shard, num_head_shards, cp_size)
    if tuple(kv_cache.shape) == (num_blocks, 2, bs, heads, head_size):
        return _split_kv_regions(kv_cache, spec, head_shard, num_head_shards, cp_size)
    return None


def mamba_state_runs(
    spec: MambaSpec, tp_size: int, tp_rank: int
) -> tuple[CopyRun, ...] | None:
    """Runs placing one rank's gated delta net state page into the TP=1 page,
    or None when the layer cannot be certified.

    The state page is the conv state, (state_len, conv_dim) rows of
    [Q | K | V], then the recurrent state, (num_v_heads, v_dim, k_dim). TP
    shards every sub-projection of each conv row and the recurrent state's
    value heads, so rank r holds slice r of each. Only GDN is certified: its
    sub-projections always shard evenly (Mamba2 may replicate B and C). Only
    the SD conv layout is: a DS page holds different bytes for the same state,
    so it would need its own canonical format id.
    """
    if (
        spec.mamba_type != MambaAttentionBackendEnum.GDN_ATTN
        or spec.tp_replicated
        or len(spec.shapes) != 2
        or is_conv_state_dim_first()
    ):
        return None
    conv = derive_mamba_conv_split(spec, tp_size, require_dim_first=False)
    conv_bytes, ssm_bytes = conv.ssm_sizes
    row_bytes = conv.local_conv_dim * conv.conv_dtype_size
    runs: list[CopyRun] = []
    for row in range(conv.conv_rows):
        local_offset = row * row_bytes
        canonical_offset = row * tp_size * row_bytes
        for dim in conv.local_proj_dims:
            size = dim * conv.conv_dtype_size
            runs.append(
                CopyRun(
                    local_offset, canonical_offset + tp_rank * size, size, 1, size, size
                )
            )
            local_offset += size
            canonical_offset += tp_size * size
    runs.append(
        CopyRun(
            conv_bytes,
            tp_size * conv_bytes + tp_rank * ssm_bytes,
            ssm_bytes,
            1,
            ssm_bytes,
            ssm_bytes,
        ),
    )
    return _coalesce_runs(runs)


def _is_packed_state_page(
    spec: MambaSpec, kv_cache: torch.Tensor | list[torch.Tensor] | None, num_blocks: int
) -> bool:
    """Whether the layer cache is the raw (num_blocks, 1, 1, bytes) page view
    that MambaBase.bind_kv_cache unpacks, with the states packed from byte 0."""
    return (
        isinstance(kv_cache, torch.Tensor)
        and kv_cache.ndim == 4
        and tuple(kv_cache.shape[:3]) == (num_blocks, 1, 1)
        and kv_cache.stride(3) == 1
        and kv_cache.shape[3] * kv_cache.element_size() == spec.state_content_size_bytes
    )


def _layer_mapping(
    spec: KVCacheSpec,
    kv_cache: torch.Tensor | list[torch.Tensor] | None,
    num_blocks: int,
    ctx: _RankContext,
) -> CanonicalPageMapping | None:
    """Certified mapping for one layer at one rank, or None (fail closed)."""
    if isinstance(spec, MambaSpec):
        # A state is per request, not per token: CP has nothing to shard.
        if ctx.cp_size > 1 or not _is_packed_state_page(spec, kv_cache, num_blocks):
            return None
        runs = mamba_state_runs(spec, ctx.tp_size, ctx.tp_rank)
        if runs is None:
            return None
        page = spec.state_content_size_bytes
        return CanonicalPageMapping(
            canonical_page_size_bytes=ctx.tp_size * page,
            local_page_size_bytes=page,
            runs=runs,
            num_writers=1,
            writer_index=0,
            parallelism_agnostic=True,
        )

    if not isinstance(spec, AttentionSpec):
        return None
    bs = spec.block_size
    page = spec.real_page_size_bytes
    if ctx.cp_size > 1 and (ctx.interleave > bs or bs % ctx.interleave):
        return None

    if isinstance(spec, MLAAttentionSpec):
        # TP-replicated latent; CP shards its tokens across the DCP groups
        if (
            spec.tokens_per_state != 1
            or page % bs
            or ctx.tp_size % ctx.dcp_size
            or spec.kv_quant_mode.is_per_token_head
        ):
            return None
        row = page // bs
        return CanonicalPageMapping(
            canonical_page_size_bytes=ctx.cp_size * page,
            local_page_size_bytes=page,
            runs=_interleave_cp_tokens([ByteRegion(0, 0, row, row)], bs, ctx),
            num_writers=ctx.tp_size // ctx.dcp_size,
            writer_index=ctx.tp_rank // ctx.dcp_size,
            parallelism_agnostic=ctx.cp_size == 1,
        )

    if spec.kv_quant_mode.is_per_token_head or not isinstance(kv_cache, torch.Tensor):
        return None
    total, tp = ctx.total_kv_heads, ctx.tp_size
    if spec.num_kv_heads != max(1, total // tp):
        return None
    if total >= tp:
        if total % tp:
            return None
        num_head_shards, replication = tp, 1
    else:
        if tp % total:
            return None
        num_head_shards, replication = total, tp // total
    # DCP shards tokens across ranks holding replicated KV
    if replication % ctx.dcp_size:
        return None

    head_shard = ctx.tp_rank // replication
    regions = _attention_byte_regions(
        kv_cache, spec, num_blocks, head_shard, num_head_shards, ctx.cp_size
    )
    if regions is None:
        return None
    return CanonicalPageMapping(
        canonical_page_size_bytes=ctx.cp_size * num_head_shards * page,
        local_page_size_bytes=page,
        runs=_interleave_cp_tokens(regions, bs, ctx),
        num_writers=replication // ctx.dcp_size,
        writer_index=(ctx.tp_rank % replication) // ctx.dcp_size,
        parallelism_agnostic=ctx.cp_size == 1,
    )


def _opaque_fallback_mapping(
    page_size_bytes: int, num_ranks: int, rank: int
) -> CanonicalPageMapping:
    """Fallback: place the worker's page whole at a worker-exclusive offset."""
    run = CopyRun(
        0, rank * page_size_bytes, page_size_bytes, 1, page_size_bytes, page_size_bytes
    )
    return CanonicalPageMapping(
        canonical_page_size_bytes=num_ranks * page_size_bytes,
        local_page_size_bytes=page_size_bytes,
        runs=(run,),
        num_writers=1,
        writer_index=0,
        parallelism_agnostic=False,
    )


def _run_intervals(runs: tuple[CopyRun, ...], canonical: bool) -> list[tuple[int, int]]:
    intervals = []
    for run in runs:
        offset = run.canonical_offset if canonical else run.local_offset
        stride = run.canonical_stride if canonical else run.local_stride
        for i in range(run.num_fragments):
            start = offset + i * stride
            intervals.append((start, start + run.fragment_size))
    return sorted(intervals)


def _is_exact_partition(intervals: list[tuple[int, int]], size: int) -> bool:
    return (
        bool(intervals)
        and intervals[0][0] == 0
        and intervals[-1][1] == size
        and all(a[1] == b[0] for a, b in zip(intervals, intervals[1:]))
    )


def _verify_tiling(layer_name: str, per_rank: list[CanonicalPageMapping]) -> None:
    """Whichever ranks a block elects as writers must tile the canonical page
    exactly once, and each rank's runs must cover exactly its local page."""
    size = per_rank[0].canonical_page_size_bytes
    num_writers = per_rank[0].num_writers
    for mapping in per_rank:
        assert mapping.canonical_page_size_bytes == size
        assert mapping.num_writers == num_writers
        local = _run_intervals(mapping.runs, canonical=False)
        assert _is_exact_partition(local, mapping.local_page_size_bytes), (
            f"runs do not cover the local page of layer {layer_name}"
        )
    for block_id in range(num_writers):
        stored: list[tuple[int, int]] = []
        for mapping in per_rank:
            if mapping.is_writer(block_id):
                stored += _run_intervals(mapping.runs, canonical=True)
        stored.sort()
        assert _is_exact_partition(stored, size), (
            f"writers of block {block_id} do not tile the canonical page "
            f"of layer {layer_name}"
        )


def _unpadded_page_size(spec: KVCacheSpec) -> int | None:
    if isinstance(spec, AttentionSpec):
        return spec.unpadded_page_size_bytes
    if isinstance(spec, MambaSpec):
        return replace(spec, page_size_padded=None).page_size_bytes
    return None


def derive_canonical_mappings(
    vllm_config: "VllmConfig",
    kv_cache_config: KVCacheConfig,
    kv_caches: dict[str, torch.Tensor | list[torch.Tensor]],
) -> dict[str, CanonicalPageMapping]:
    """Per-layer canonical page mappings for this worker.

    Empty when the worker group is not exactly the TP x PCP grid; layers
    absent from the result have no canonical representation.
    """
    parallel_config = vllm_config.parallel_config
    tp_size = parallel_config.tensor_parallel_size
    pcp_size = parallel_config.prefill_context_parallel_size
    group_size = tp_size * pcp_size
    if parallel_config.world_size != group_size:
        return {}

    def ctx(rank: int) -> _RankContext:
        return _RankContext(
            tp_size=tp_size,
            dcp_size=parallel_config.decode_context_parallel_size,
            pcp_size=pcp_size,
            interleave=parallel_config.cp_kv_cache_interleave_size,
            total_kv_heads=vllm_config.model_config.get_total_num_kv_heads(),
            rank=rank,
        )

    my_rank = parallel_config.rank
    num_blocks = kv_cache_config.num_blocks

    mappings: dict[str, CanonicalPageMapping] = {}
    for kv_cache_group in kv_cache_config.kv_cache_groups:
        group_kv_cache_spec = kv_cache_group.kv_cache_spec
        if isinstance(group_kv_cache_spec, UniformTypeKVCacheSpecs):
            per_layer_specs = group_kv_cache_spec.kv_cache_specs
        else:
            per_layer_specs = {}
        for layer_name in kv_cache_group.layer_names:
            spec = per_layer_specs.get(layer_name, group_kv_cache_spec)
            per_rank: list[CanonicalPageMapping] = []
            for rank in range(group_size):
                mapping = _layer_mapping(
                    spec, kv_caches.get(layer_name), num_blocks, ctx(rank)
                )
                if mapping is None:
                    break
                per_rank.append(mapping)
            if len(per_rank) != group_size:
                page = _unpadded_page_size(spec)
                if page is None:
                    continue
                per_rank = [
                    _opaque_fallback_mapping(page, group_size, rank)
                    for rank in range(group_size)
                ]
            _verify_tiling(layer_name, per_rank)
            mappings[layer_name] = per_rank[my_rank]
    return mappings
