# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B70 0040: routed experts split between VRAM (hot) and pinned host RAM (cold).

For INT4 W4A16 MoE on XPU (XPUExpertsWNA16). Each rank keeps the
``B70_MOE_HOT_EXPERTS`` most-routed of its local experts per layer as the
layer's ordinary parameters (in VRAM) and the rest in pinned host memory,
which the same grouped-GEMM kernel reads through a UVA view (zero copy over
PCIe). Both calls see the full ``topk_ids``; each maps only its own experts
(``expert_map`` -1 elsewhere, the mechanism expert parallelism already uses),
and the two outputs are summed. The split is exact: a row's result does not
depend on where its experts live.

Load path (no full-size host or device copy of a layer ever exists):
  * ``create_weights`` is called with the hot count, so the device params hold
    only hot experts;
  * the weight loader writes a cold expert into a pageable CPU twin of the
    same param (GPTQ layout);
  * after loading, each twin is repacked into the kernel layout (transpose,
    unsigned -> signed int4) on the device, 16 experts at a time, copied into
    pinned memory and dropped (the host only copies: an all-core CPU repack on
    both ranks at once was the heaviest burst of the whole boot).

Switches: B70_MOE_HOTCOLD=1, B70_MOE_HOT_EXPERTS=<per layer per rank> (or, 0.32 port,
B70_MOE_COLD_EXPERTS=<cold per layer per rank>, which wins),
B70_MOE_HOTCOLD_RANKINGS=<json: layer -> global expert ids, hottest first>.
Off (default): nothing here runs.
"""

import json
import os
import re
from functools import cache

import torch

from vllm.logger import init_logger

logger = init_logger(__name__)

_LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
# Per-expert params of the GPTQ MoE method that get a cold twin.
_EXPERT_PARAMS = ("w13_qweight", "w13_scales", "w13_qzeros", "w13_bias",
                  "w2_qweight", "w2_scales", "w2_qzeros", "w2_bias")


def hotcold_enabled() -> bool:
    return os.environ.get("B70_MOE_HOTCOLD", "0") == "1"


@cache
def _rankings() -> dict[int, list[int]]:
    path = os.environ.get("B70_MOE_HOTCOLD_RANKINGS", "")
    if not path:
        return {}
    with open(path) as handle:
        raw = json.load(handle)
    return {int(k): [int(e) for e in v] for k, v in raw.items()}


def _rank() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank
        return get_tensor_model_parallel_rank()
    except Exception:
        return 0


@cache
def _learned_state() -> dict | None:
    """B70 0047: the long-term scores the swapper saved (all ranks merged; each rank's file holds only its
    own experts), or None (off, missing, or fewer than B70_MOE_HOTCOLD_STATE_MIN_PERIODS active periods)."""
    import glob

    prefix = os.environ.get("B70_MOE_HOTCOLD_STATE", "")
    if not prefix:
        return None
    layers: dict[int, dict[int, float]] = {}
    periods = None
    for path in sorted(glob.glob(f"{glob.escape(prefix)}-rank*.json")):
        try:
            with open(path) as handle:
                raw = json.load(handle)
        except (OSError, ValueError) as err:
            logger.warning("B70 hot/cold: ignoring state %s: %s", path, err)
            continue
        periods = int(raw.get("periods", 0)) if periods is None else min(periods, int(raw.get("periods", 0)))
        for layer, scores in raw.get("layers", {}).items():
            layers.setdefault(int(layer), {}).update({int(g): float(s) for g, s in scores.items()})
    need = int(os.environ.get("B70_MOE_HOTCOLD_STATE_MIN_PERIODS", "12"))
    if not layers or periods is None or periods < need:
        if layers:
            logger.info("B70 hot/cold: state %s has %s active periods (< %d): not used", prefix, periods, need)
        return None
    logger.info("B70 hot/cold: hot sets from the learned state %s (%d layers, %d active periods)",
                prefix, len(layers), periods)
    return {"periods": periods, "layers": layers}


class _PinnedArena:
    """Pinned host memory carved from 1 GiB chunks (first fit, 4 KiB aligned).

    torch's caching host allocator rounds every pinned block up to a power of
    two (a 197 MB cold w13 takes 256 MB), so per-tensor pinning wastes ~30 %
    of the cold experts' RAM. Chunks of exactly 2^30 B are not rounded.
    """

    CHUNK = 1 << 30
    ALIGN = 4096

    def __init__(self) -> None:
        self.chunks: list[torch.Tensor] = []
        self.used: list[int] = []

    def empty(self, shape, dtype: torch.dtype) -> torch.Tensor:
        nbytes = torch.Size(shape).numel() * torch.empty((), dtype=dtype).element_size()
        if nbytes > self.CHUNK:
            return torch.empty(shape, dtype=dtype, pin_memory=True)
        for i, used in enumerate(self.used):
            if used + nbytes <= self.CHUNK:
                break
        else:
            self.chunks.append(torch.empty(self.CHUNK, dtype=torch.uint8, pin_memory=True))
            self.used.append(0)
            i, used = len(self.chunks) - 1, 0
        self.used[i] = used + -(-nbytes // self.ALIGN) * self.ALIGN
        return self.chunks[i][used : used + nbytes].view(dtype).view(shape)

    @property
    def gib(self) -> float:
        return len(self.chunks) * self.CHUNK / 2**30


_ARENA = _PinnedArena()
_FINALIZED: list[str] = []
_COUNTED: list["HotColdPlan"] = []  # plans with a route counter (B70_MOE_ROUTE_COUNT)
_DUMP = {"t": 0.0}
_LIVE = {"on": False}  # set by the runner's first real step (maybe_dump_routes)


_PLANS: dict[int, "HotColdPlan"] = {}  # finalized plans by layer index


def cold_prefetch_min_tokens() -> int:
    """0 = off; else batches with at least this many tokens run the cold experts from VRAM staging."""
    if os.environ.get("B70_MOE_COLD_PREFETCH", "0") != "1":
        return 0
    return max(64, int(os.environ.get("B70_MOE_COLD_PREFETCH_MIN_TOKENS", "512")))


def _capturing() -> bool:
    """True while the current XPU stream is recording a graph (CUDA/XPU graph capture)."""
    return bool(torch.xpu.is_current_stream_capturing())


def _refuse_under_capture(what: str) -> None:
    """B70-V32-HC3: the prefetcher orders an eager side copy stream against the current stream
    with events. Recorded into a graph, those events never complete on the eager timeline, so the
    copy stream and the next eager forward wait forever. Fail loudly instead."""
    if _capturing():
        raise RuntimeError(f"B70 hot/cold: cold prefetch {what} called during graph capture")


class _ColdPrefetcher:
    """Two VRAM slots, each holding one layer's cold experts (kernel layout).

    Large batches (prefill) run the cold experts from a slot instead of reading
    pinned host memory through UVA: layer L computes from one slot while a
    copy stream DMAs layer L+1's cold experts into the other. Events order the
    two streams: a slot is refilled only after the GEMM that read it, and a
    GEMM starts only after its slot's copy. The math is unchanged (same kernel,
    same bytes); only where the kernel reads the weights from differs."""

    def __init__(self) -> None:
        self.bufs: list[dict[str, torch.Tensor]] | None = None
        self.owner: list[int | None] = [None, None]
        self.ready: list = [None, None]
        self.free: list = [None, None]
        self.last = 1
        self.stream = None

    def _ensure(self, plan: "HotColdPlan", device) -> None:
        if self.bufs is not None:
            return
        self.bufs = [{k: torch.empty(t.shape, dtype=t.dtype, device=device)
                      for k, t in plan.cold.items()} for _ in range(2)]
        self.stream = torch.xpu.Stream(device)
        gib = 2 * sum(t.numel() * t.element_size() for t in plan.cold.values()) / 2**30
        logger.info("B70 hot/cold: cold prefetch on (batches >= %d tokens), 2 VRAM slots, %.2f GiB",
                    cold_prefetch_min_tokens(), gib)

    def _load(self, slot: int, plan: "HotColdPlan") -> None:
        _refuse_under_capture("_load")
        compute = torch.xpu.current_stream()
        with torch.xpu.stream(self.stream):
            if self.free[slot] is not None:
                self.stream.wait_event(self.free[slot])
            for key, src in plan.cold.items():
                self.bufs[slot][key].copy_(src, non_blocking=True)
            event = torch.xpu.Event()
            event.record(self.stream)
        self.ready[slot] = event
        self.owner[slot] = plan.layer_idx
        del compute

    def acquire(self, plan: "HotColdPlan", device) -> dict[str, torch.Tensor]:
        _refuse_under_capture("acquire")
        self._ensure(plan, device)
        if plan.layer_idx in self.owner:
            slot = self.owner.index(plan.layer_idx)
        else:
            slot = 1 - self.last
            self._load(slot, plan)
        torch.xpu.current_stream().wait_event(self.ready[slot])
        self.last = slot
        return self.bufs[slot]

    def release_and_prefetch(self, plan: "HotColdPlan") -> None:
        _refuse_under_capture("release_and_prefetch")
        slot = self.last
        event = torch.xpu.Event()
        event.record(torch.xpu.current_stream())
        self.free[slot] = event
        keys = sorted(_PLANS)
        nxt = _PLANS[keys[(keys.index(plan.layer_idx) + 1) % len(keys)]]
        if nxt.layer_idx not in self.owner:
            self._load(1 - slot, nxt)


_PREFETCH = _ColdPrefetcher()


def swap_enabled() -> bool:
    return os.environ.get("B70_MOE_HOTCOLD_SWAP", "0") == "1"


def async_swap_enabled() -> bool:
    """B70-Q27-0003: B70_MOE_HOTCOLD_ASYNC_SWAP=1 (with B70_MOE_HOTCOLD_SWAP=1) moves experts without a host
    synchronize on the forward path (_Swapper._maybe_swap_async). Unset: the synchronous swap, unchanged."""
    return swap_enabled() and os.environ.get("B70_MOE_HOTCOLD_ASYNC_SWAP", "0") == "1"


def spare_rows() -> int:
    """B70-Q27-0003: free cold rows per layer for the asynchronous swap (0 when it is off). A hot expert is first
    copied into a free cold row while it is still served hot, so every expert stays routable at every step.
    B70_MOE_HOTCOLD_ASYNC_SPARE, default the swap budget; it also caps the swaps per layer and cycle."""
    if not async_swap_enabled():
        return 0
    default = os.environ.get("B70_MOE_HOTCOLD_SWAP_BUDGET", "8")
    return max(1, int(os.environ.get("B70_MOE_HOTCOLD_ASYNC_SPARE", default)))


def route_count_enabled() -> bool:
    """B70_MOE_ROUTE_COUNT: 1 = count eager batches and decode graphs; prefill = eager batches only
    (nothing is added to the decode graphs). B70_MOE_HOTCOLD_SWAP=1 implies prefill counting."""
    return os.environ.get("B70_MOE_ROUTE_COUNT", "0") in ("1", "prefill") or swap_enabled()


_DECODE_OP = {"max_rows": None}


def _cpu_has_avx512_bf16() -> bool:
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("flags"):
                    return "avx512_bf16" in line.split()
    except OSError:
        pass
    return False


def _cpu_path_setup() -> None:
    """B70 0049: the decode op's CPU pool for cold experts. B70_MOE_CPU=1 turns it on, anything else (0, unset) off:
    then every cold expert stays on the GPU path. The op itself also refuses CPUs without AVX-512 BF16; this sets
    B70_MOE_CPU=0 for them, so the state is logged here once per rank instead of silently in the op.
    The pool pins B70_MOE_CPU_THREADS threads from B70_MOE_CPU_FIRST_CORE: unset = 8 x TP rank (one 8-core CCD
    per rank on a 16-core Ryzen); one number = the same first core for every rank; a comma list (e.g. "0,8") =
    one first core per TP rank."""
    try:
        from vllm.distributed import get_tensor_model_parallel_rank
        rank = get_tensor_model_parallel_rank()
    except Exception:  # noqa: BLE001
        rank = 0
    if os.environ.get("B70_MOE_CPU", "0") != "1":
        logger.info("B70 decode op: CPU path off (B70_MOE_CPU=%s), rank %d: cold experts on the GPU",
                    os.environ.get("B70_MOE_CPU", "unset"), rank)
        return
    if not _cpu_has_avx512_bf16():
        os.environ["B70_MOE_CPU"] = "0"
        logger.warning("B70 decode op: CPU path requested but this CPU has no AVX-512 BF16 (avx512_bf16), rank %d: "
                       "cold experts on the GPU", rank)
        return
    first = os.environ.get("B70_MOE_CPU_FIRST_CORE", "").strip()
    if not first:
        first = str(8 * rank)
    elif "," in first:
        cores = [c.strip() for c in first.split(",")]
        first = cores[rank] if rank < len(cores) and cores[rank] else str(8 * rank)
    os.environ["B70_MOE_CPU_FIRST_CORE"] = first
    logger.info("B70 decode op: CPU path on, rank %d: %s threads from core %s, %s %% of the cold experts", rank,
                os.environ.get("B70_MOE_CPU_THREADS", "8"), first, os.environ.get("B70_MOE_CPU_PCT", "50"))


def decode_op_max_rows() -> int:
    """B70 0048: B70_MOE_DECODE_OP=<path to libb70_moe_decode.so> (or 1 = the image's copy) runs batches of at most
    B70_MOE_DECODE_OP_MAX_ROWS (default 16) tokens through one fused hot+cold kernel set instead of two stock
    XpuFusedMoe calls; 0 = off (no library loaded)."""
    rows = _DECODE_OP["max_rows"]
    if rows is None:
        rows, lib = 0, os.environ.get("B70_MOE_DECODE_OP", "0")
        if lib not in ("", "0"):
            if lib == "1":
                lib = os.path.join(os.path.dirname(__file__), "libb70_moe_decode.so")
            try:
                torch.ops.load_library(lib)
                rows = min(16, int(os.environ.get("B70_MOE_DECODE_OP_MAX_ROWS", "16")))
                logger.info("B70 decode op: %s, batches of <= %d tokens", lib, rows)
            except (OSError, RuntimeError) as e:
                logger.warning("B70 decode op: cannot load %s (%s); stock path", lib, e)
        if rows:
            _cpu_path_setup()
        _DECODE_OP["max_rows"] = rows
    return rows


_ONES: dict = {}  # device -> static int64 ones for the route counter (allocated at load time)
_ONES_N = 1 << 16


def route_ones(n: int, device) -> torch.Tensor:
    """n int64 ones for index_add_: a slice of one static buffer (no fill kernel per layer)."""
    buf = _ONES.get(str(device))
    if buf is None or n > buf.numel():
        return torch.ones(n, dtype=torch.int64, device=device)
    return buf[:n]


_TRACE = {"f": None, "last": None, "steps": 0}


def maybe_trace_routes() -> None:
    """B70_MOE_ROUTE_TRACE=<path>: one JSON line per decode step with the experts each
    layer's decode graph routed to in that step (needs B70_MOE_ROUTE_COUNT=1). Measurement only."""
    path = os.environ.get("B70_MOE_ROUTE_TRACE", "")
    if not path or not _COUNTED or not _COUNTED[0].count_graphs:
        return
    import numpy as np

    counts = np.stack([p.route_counts[1, :-1].cpu().numpy() for p in _COUNTED])
    if _TRACE["f"] is None:
        try:
            from vllm.distributed import get_tensor_model_parallel_rank
            rank = get_tensor_model_parallel_rank()
        except Exception:
            rank = 0
        _TRACE["f"] = open(f"{path}-rank{rank}.jsonl", "w")
        cold = {str(p.layer_idx): np.nonzero(p.cold_map.cpu().numpy() >= 0)[0].tolist() for p in _COUNTED}
        _TRACE["f"].write(json.dumps({"cold": cold}) + "\n")
        _TRACE["last"] = counts
        return
    delta = counts - _TRACE["last"]
    _TRACE["last"] = counts
    if delta.sum() == 0:
        return
    step = {str(p.layer_idx): np.nonzero(delta[i])[0].tolist() for i, p in enumerate(_COUNTED)}
    _TRACE["f"].write(json.dumps(step) + "\n")
    _TRACE["steps"] += 1
    if _TRACE["steps"] % 200 == 0:
        _TRACE["f"].flush()


def maybe_dump_routes() -> None:
    """Log the cold share of routes and write the cumulative counts (every
    B70_MOE_ROUTE_COUNT_S seconds). Called by the runner on the host right
    after its per-step sync, so the .cpu() below does not stall the GPU."""
    import time

    if not _COUNTED:
        return
    _LIVE["on"] = True
    now = time.monotonic()
    period = float(os.environ.get("B70_MOE_ROUTE_COUNT_S", "60"))
    if _DUMP["t"] == 0.0:
        _DUMP["t"] = now
        return
    if now - _DUMP["t"] < period:
        return
    _DUMP["t"] = now
    try:
        from vllm.distributed import get_tensor_model_parallel_rank
        rank = get_tensor_model_parallel_rank()
    except Exception:
        rank = 0
    out = {}
    local_all, cold_all = [0, 0], [0, 0]
    for plan in _COUNTED:
        counts = plan.route_counts[:, :-1].cpu()
        # this period only, against the placement now in effect (it changes with B70_MOE_HOTCOLD_SWAP)
        delta = counts - getattr(plan, "dump_last", torch.zeros_like(counts))
        plan.dump_last = counts
        local = (plan.hot_map >= 0).cpu() | (plan.cold_map >= 0).cpu()
        cold = (plan.cold_map >= 0).cpu()
        for row in (0, 1):
            local_all[row] += int(delta[row][local].sum())
            cold_all[row] += int(delta[row][cold].sum())
        out[str(plan.layer_idx)] = counts.tolist()
    share = [c / n if n else 0.0 for c, n in zip(cold_all, local_all)]
    logger.info("B70 routes rank %d (last period): prefill %d local routes, cold share %.3f; decode %d, "
                "cold share %.3f",
                rank, local_all[0], share[0], local_all[1], share[1])
    path = os.environ.get("B70_MOE_ROUTE_COUNT_PATH", "")
    if path:
        with open(f"{path}-rank{rank}.json", "w") as handle:
            json.dump(out, handle)


def _pin(t: torch.Tensor, extra: int = 0) -> torch.Tensor:
    """Pinned copy of ``t``; ``extra`` zeroed rows appended (B70-Q27-0003 spare cold rows)."""
    if not extra:
        pinned = _ARENA.empty(t.shape, t.dtype)
        pinned.copy_(t)
        return pinned
    pinned = _ARENA.empty((t.shape[0] + extra, *t.shape[1:]), t.dtype)
    pinned[: t.shape[0]].copy_(t)
    pinned[t.shape[0]:].zero_()
    return pinned


def _repack_int4_pinned(qweight: torch.Tensor, device: torch.device, extra: int = 0) -> torch.Tensor:
    """GPTQ int32 [E, in/8, out] (CPU) -> signed-int4 bytes [E, out, in/2] in pinned memory.

    The transforms of int_wna16._process_weights_xpu (transpose, int32 -> uint8 view)
    and vllm_xpu_kernels.implement_zp, run on ``device`` a few experts at a time: the
    host only copies, and the device temporaries stay at a few hundred MB.
    """
    from vllm_xpu_kernels.fused_moe_interface import implement_zp

    e, k8, n = qweight.shape
    out = _ARENA.empty((e + extra, n, k8 * 4), torch.uint8)
    step = 16
    for i in range(0, e, step):
        chunk = qweight[i : i + step].to(device)
        packed = chunk.transpose(1, 2).contiguous().view(torch.uint8)
        out[i : i + step].copy_(implement_zp(packed))
    if extra:
        out[e:].zero_()
    return out


class _Swapper:
    """B70 0043: the hot set follows the traffic.

    Every B70_MOE_HOTCOLD_SWAP_S seconds each rank turns the prefill route
    counts of the last period into a decayed score per (layer, local expert)
    (plus B70_MOE_HOTCOLD_SWAP_PRIOR x the rankings-file order) and swaps up to
    B70_MOE_HOTCOLD_SWAP_BUDGET experts per layer whose cold score beats a hot
    one by 10 %: the hot slot's bytes go to the cold expert's pinned slot and
    vice versa, and the two expert maps are rewritten in place. The decode
    graphs read weights and maps through the same pointers, so nothing is
    recaptured. Runs on the host right after the runner's per-step sync and
    synchronizes after the copies, so no forward ever sees a half-swapped
    layer. Results do not depend on placement (hot + cold is exact)."""

    KEYS = ("w13", "w2", "w13_scales", "w2_scales")

    def __init__(self) -> None:
        self.t = 0.0
        self.ready = False
        self.disabled = False
        self.saved_t = 0.0
        self.periods = 0

    # B70 0047 (decode-aware ranking): B70_MOE_HOTCOLD_SWAP_SOURCE=decode scores by the routes of the decode
    # graphs (needs B70_MOE_ROUTE_COUNT=1) plus B70_MOE_HOTCOLD_SWAP_PREFILL_W x the prefill routes, each
    # normalised per layer and period. Large prefill batches read the whole cold set from the 0042 VRAM slots
    # anyway, so only decode (and small batches) pay for a cold expert. B70_MOE_HOTCOLD_STATE=<prefix> keeps a
    # long-term score per expert (decay B70_MOE_HOTCOLD_STATE_DECAY per active period), writes it to
    # <prefix>-rank<R>.json every B70_MOE_HOTCOLD_STATE_S seconds; the next boot places the hot set by it
    # (maybe_plan) and seeds the short-term score with it, so a restart starts from this host's traffic.

    def _combined(self, plan, np, counts=None):
        """This period's score mass per global expert. ``counts``: a host snapshot of the route counters
        (async swap); None reads them from the device (a host sync, the synchronous swap)."""
        if counts is None:
            counts = plan.route_counts[:, :-1].cpu().numpy()
        delta, plan.last_counts = counts - plan.last_counts, counts.copy()
        if self.source != "decode":
            return delta[0].astype(np.float64)
        pre, dec = delta[0].astype(np.float64), delta[1].astype(np.float64)
        out = np.zeros(plan.num_global)
        if dec.sum() > 0:
            out += dec / dec.sum()
        if pre.sum() > 0 and self.prefill_w > 0:
            out += self.prefill_w * pre / pre.sum()
        return out

    def _save_state(self, np) -> None:
        rank = _rank()
        layers = {}
        for plan in self.plans:
            layers[str(plan.layer_idx)] = {str(int(g)): round(float(plan.lt[g]), 6)
                                           for g in plan.local_globals if plan.lt[g] > 0}
        path = f"{self.state_path}-rank{rank}.json"
        tmp = path + ".tmp"
        periods = self.periods + self.prev_periods
        try:
            with open(tmp, "w") as handle:
                json.dump({"version": 1, "source": self.source, "periods": periods, "layers": layers}, handle)
            os.replace(tmp, path)
        except OSError as err:
            logger.warning("B70 hot/cold: cannot write %s: %s", path, err)
            return
        logger.info("B70 hot/cold state rank %d: saved %s (%d active periods)", rank, path, periods)

    def _setup(self) -> bool:
        import time

        import numpy as np

        plans = [_PLANS[k] for k in sorted(_PLANS)]
        if not plans or any(p._impls is None or p.route_counts is None for p in plans):
            return False  # impls are built on the first forward
        for plan in plans:
            hot, _ = plan._impls
            plan.hot_tensors = {"w13": hot.w13.view(torch.uint8), "w2": hot.w2.view(torch.uint8),
                                "w13_scales": hot.gemm1_wei_scales, "w2_scales": hot.gemm2_wei_scales}
            for key in self.KEYS:
                h, c = plan.hot_tensors[key], plan.cold[key]
                if h.shape[1:] != c.shape[1:] or h.element_size() != c.element_size():
                    logger.warning("B70 hot/cold swap off: %s %s hot %s/%s vs cold %s/%s", plan.layer_name,
                                   key, tuple(h.shape), h.dtype, tuple(c.shape), c.dtype)
                    self.disabled = True
                    return False
            hot_map, cold_map = plan.hot_map.cpu().numpy(), plan.cold_map.cpu().numpy()
            plan.hot_owner = np.full(plan.num_hot, -1, np.int64)
            plan.cold_owner = np.full(plan.cold_rows, -1, np.int64)  # spare rows (async swap) stay -1
            for g in np.nonzero(hot_map >= 0)[0]:
                plan.hot_owner[hot_map[g]] = g
            for g in np.nonzero(cold_map >= 0)[0]:
                plan.cold_owner[cold_map[g]] = g
            owners = np.concatenate([plan.hot_owner, plan.cold_owner])
            plan.local_globals = np.sort(owners[owners >= 0])
            plan.last_counts = plan.route_counts[:, :-1].cpu().numpy().copy()
            plan.ema = np.zeros(plan.num_global)
            plan.lt = np.zeros(plan.num_global)
            ranked = [g for g in _rankings().get(plan.layer_idx, []) if g in set(plan.local_globals.tolist())]
            plan.prior = np.zeros(plan.num_global)
            for i, g in enumerate(ranked):
                plan.prior[g] = 1 - i / max(len(ranked), 1)
            if not plan.spare:  # the in-place exchange's VRAM temporaries; the async swap needs none
                plan.tmp = {k: torch.empty_like(plan.hot_tensors[k][0]) for k in self.KEYS}
        self.plans = plans
        self.period = float(os.environ.get("B70_MOE_HOTCOLD_SWAP_S", "5"))
        self.decay = float(os.environ.get("B70_MOE_HOTCOLD_SWAP_DECAY", "0.95"))
        self.budget = int(os.environ.get("B70_MOE_HOTCOLD_SWAP_BUDGET", "8"))
        self.prior_weight = float(os.environ.get("B70_MOE_HOTCOLD_SWAP_PRIOR", "0.01"))
        self.source = os.environ.get("B70_MOE_HOTCOLD_SWAP_SOURCE", "prefill")
        if self.source == "decode" and not plans[0].count_graphs:
            logger.warning("B70 hot/cold swap: SOURCE=decode needs B70_MOE_ROUTE_COUNT=1; using prefill")
            self.source = "prefill"
        self.prefill_w = float(os.environ.get("B70_MOE_HOTCOLD_SWAP_PREFILL_W", "0.1"))
        self.state_path = os.environ.get("B70_MOE_HOTCOLD_STATE", "")
        self.state_decay = float(os.environ.get("B70_MOE_HOTCOLD_STATE_DECAY", "0.999"))
        self.state_period = float(os.environ.get("B70_MOE_HOTCOLD_STATE_S", "300"))
        self.saved_t = time.monotonic()  # the first save after one full state period of uptime
        self.prev_periods = 0
        seeded = 0
        state = _learned_state()
        if state:
            self.prev_periods = int(state["periods"])
            for plan in plans:
                learned = state["layers"].get(plan.layer_idx, {})
                for g in plan.local_globals:
                    plan.lt[g] = learned.get(int(g), 0.0)
                total = plan.lt.sum()
                if total > 0 and self.source == "decode":
                    # the steady state of the short-term score under the long-term distribution
                    plan.ema = plan.lt / total * (1 + self.prefill_w) / max(1 - self.decay, 1e-3)
                    seeded += 1
        self.total = 0
        logger.info("B70 hot/cold swap on: every %.0f s, decay %.2f, up to %d swaps per layer, prior %.3f, "
                    "source %s (prefill weight %.2f), state %s (%d layers seeded, %d earlier periods)",
                    self.period, self.decay, self.budget, self.prior_weight, self.source, self.prefill_w,
                    self.state_path or "off", seeded, self.prev_periods)
        self.async_mode = False
        if async_swap_enabled():
            if all(p.spare > 0 for p in plans):
                self._setup_async(np)
            else:
                logger.warning("B70 hot/cold: B70_MOE_HOTCOLD_ASYNC_SWAP=1 but the cold tensors have no spare rows "
                               "(set before load?); synchronous swap")
        return True

    def _swap(self, plan, a: int, b: int) -> None:
        """a: global id of a hot expert, b: global id of a cold expert."""
        i = int(plan.hot_map_np[a])
        j = int(plan.cold_map_np[b])
        for key in self.KEYS:
            h, c, tmp = plan.hot_tensors[key], plan.cold[key], plan.tmp[key]
            tmp.copy_(h[i])
            h[i].copy_(c[j].view(h.dtype), non_blocking=True)
            c[j].view(h.dtype).copy_(tmp, non_blocking=True)
        plan.hot_map[a] = -1
        plan.hot_map[b] = i
        plan.cold_map[b] = -1
        plan.cold_map[a] = j
        plan.hot_map_np[a], plan.hot_map_np[b] = -1, i
        plan.cold_map_np[b], plan.cold_map_np[a] = -1, j
        plan.hot_owner[i], plan.cold_owner[j] = b, a

    def maybe_swap(self) -> None:
        import time

        import numpy as np

        if self.disabled:
            return
        now = time.monotonic()
        if not self.ready:
            if not self._setup():
                return
            self.ready = True
            self.t = now
            return
        if self.async_mode:
            self._maybe_swap_async(now, np)
            return
        if now - self.t < self.period:
            return
        self.t = now
        t0 = time.perf_counter()
        moves = self._moves(np, now)
        if not moves:
            return
        if _PREFETCH.stream is not None:
            torch.xpu.current_stream().wait_stream(_PREFETCH.stream)
            _PREFETCH.owner = [None, None]  # staged copies may hold swapped-out experts
        n = 0
        for plan, pairs in moves:
            plan.hot_map_np = plan.hot_map.cpu().numpy()
            plan.cold_map_np = plan.cold_map.cpu().numpy()
            for a, b in pairs:
                self._swap(plan, a, b)
                n += 1
        torch.xpu.current_stream().synchronize()
        self.total += n
        logger.info("B70 hot/cold swap rank %d: %d experts in %d layers in %.1f ms (%d since start)",
                    _rank(), n, len(moves), (time.perf_counter() - t0) * 1e3, self.total)

    def _moves(self, np, now: float, snapshot=None) -> list:
        """Fold this period's routes into the scores; [(plan, [(hot global a, cold global b), ...]), ...]."""
        active = False
        for k, plan in enumerate(self.plans):
            delta = self._combined(plan, np, None if snapshot is None else snapshot[k, :, :-1])
            if delta.sum() <= 0 and self.source == "decode":
                continue  # no traffic on this layer: keep the scores (idle time does not decay them)
            active = active or delta.sum() > 0
            plan.ema = self.decay * plan.ema + delta
            if self.state_path:
                total = delta.sum()
                plan.lt = self.state_decay * plan.lt + delta / total
        if active:
            self.periods += 1
        if self.state_path and active and now - self.saved_t >= self.state_period:
            self.saved_t = now
            self._save_state(np)
        top = max(float(p.ema.max()) for p in self.plans) or 1.0
        moves = []
        for plan in self.plans:
            score = plan.ema / top + self.prior_weight * plan.prior
            hot_g, cold_g = plan.hot_owner, plan.cold_owner
            cold_g = cold_g[cold_g >= 0]  # without the async swap's spare rows
            cold_best = cold_g[np.argsort(-score[cold_g])][: self.budget]
            hot_worst = hot_g[np.argsort(score[hot_g])][: self.budget]
            pairs = [(int(a), int(b)) for a, b in zip(hot_worst, cold_best) if score[b] > 1.1 * score[a]]
            if pairs:
                moves.append((plan, pairs))
        return moves

    # -- B70-Q27-0003: the asynchronous swap (B70_MOE_HOTCOLD_ASYNC_SWAP=1) ------------------------------------
    #
    # The synchronous swap above exchanges hot slot i and cold row j in place on the forward stream and then
    # synchronizes the host: every forward behind it waits for ~3 PCIe/VRAM copies per expert plus the host round
    # trip (160-940 ms per tick after a domain shift). Here one swap (hot a in slot i, cold b in row j) is
    # a cycle of phases, one per runner step at most, none of which waits on the host:
    #   snap   forward stream: the route counters -> a pinned snapshot, event (no .cpu() sync)
    #   copy1  side stream, after the last cycle's flip2: a's bytes, hot slot i -> a spare cold row s (D2H)
    #   flip1  forward stream, once copy1's event has completed (polled with query()): maps a -> cold row s
    #          (hot slot i is now unreferenced; forwards before this flip may still read it)
    #   copy2  side stream, after flip1's event (so after every forward that could read slot i): b's bytes,
    #          cold row j -> hot slot i (H2D)
    #   flip2  forward stream, once copy2 has completed: maps b -> hot slot i; row j becomes spare after flip2's event
    # Invariant: a forward reads only rows its maps (as of its place on the forward stream) reference, and a row is
    # written only while no map references it and after every forward that referenced it, so every expert stays
    # routable with complete bytes at every step. The maps are written in place from pinned staging (same storage, so
    # the captured graphs read them), and the weights in place too; nothing is recaptured. Host cost per step: a few
    # event queries and ~2 small H2D map copies per moved layer.

    def _setup_async(self, np) -> None:
        self.stream = torch.xpu.Stream(self.plans[0].hot_map.device)
        self.phase = "idle"
        self.ev = None  # the event the next phase waits for (polled)
        self.ev_flip = None  # forward stream, after flip1
        self.ev_spare = None  # forward stream, after the last flip2: the rows it released are free after it
        self.cycle = None
        self.snap = torch.empty((len(self.plans), *self.plans[0].route_counts.shape), dtype=torch.int64,
                                pin_memory=True)
        spare_bytes = 0
        for plan in self.plans:
            plan.hot_map_np = plan.hot_map.cpu().numpy().copy()
            plan.cold_map_np = plan.cold_map.cpu().numpy().copy()
            plan.spare_free = [r for r in range(plan.cold_rows) if plan.cold_owner[r] < 0]
            plan.map_stage = torch.empty((2, plan.num_global), dtype=plan.hot_map.dtype, pin_memory=True)
            plan.pending = []
            spare_bytes += sum(t[0].numel() * t.element_size() for t in plan.cold.values()) * plan.spare
        self.async_mode = True
        logger.info("B70 hot/cold async swap on (B70-Q27-0003): side-stream copies, routing flips on the forward "
                    "stream after their events, no host synchronize; %d spare cold rows per layer (%.2f GiB pinned), "
                    "at most that many swaps per layer and cycle", self.plans[0].spare, spare_bytes / 2**30)

    def _push_maps(self, plan) -> None:
        """The host mirrors -> the device maps, in place, on the current (forward) stream."""
        plan.map_stage[0].copy_(torch.from_numpy(plan.hot_map_np))
        plan.map_stage[1].copy_(torch.from_numpy(plan.cold_map_np))
        plan.hot_map.copy_(plan.map_stage[0], non_blocking=True)
        plan.cold_map.copy_(plan.map_stage[1], non_blocking=True)

    def _copy1(self, moves) -> int:
        side, n = self.stream, 0
        if self.ev_spare is not None:
            side.wait_event(self.ev_spare)
        with torch.xpu.stream(side):
            for plan, pairs in moves:
                plan.pending = []
                for a, b in pairs[: len(plan.spare_free)]:
                    i, j, s = int(plan.hot_map_np[a]), int(plan.cold_map_np[b]), plan.spare_free.pop(0)
                    for key in self.KEYS:
                        h, c = plan.hot_tensors[key], plan.cold[key]
                        c[s].view(h.dtype).copy_(h[i], non_blocking=True)
                    plan.pending.append((a, b, i, j, s))
                    n += 1
            self.ev = torch.xpu.Event()
            self.ev.record(side)
        return n

    def _flip1_copy2(self, fwd) -> None:
        fwd.wait_event(self.ev)  # already complete (polled): ordering only
        for plan in self.plans:
            if plan.pending:
                for a, _b, _i, _j, s in plan.pending:
                    plan.hot_map_np[a] = -1
                    plan.cold_map_np[a] = s
                self._push_maps(plan)
        _PREFETCH.owner = [None, None]  # a staged layer may predate the new cold rows
        self.ev_flip = torch.xpu.Event()
        self.ev_flip.record(fwd)
        side = self.stream
        side.wait_event(self.ev_flip)
        with torch.xpu.stream(side):
            for plan in self.plans:
                for _a, _b, i, j, _s in plan.pending:
                    for key in self.KEYS:
                        h, c = plan.hot_tensors[key], plan.cold[key]
                        h[i].copy_(c[j].view(h.dtype), non_blocking=True)
            self.ev = torch.xpu.Event()
            self.ev.record(side)

    def _flip2(self, fwd) -> tuple[int, int]:
        fwd.wait_event(self.ev)
        n = layers = 0
        for plan in self.plans:
            if not plan.pending:
                continue
            layers += 1
            for a, b, i, j, s in plan.pending:
                plan.hot_map_np[b] = i
                plan.cold_map_np[b] = -1
                plan.hot_owner[i], plan.cold_owner[s], plan.cold_owner[j] = b, a, -1
                plan.spare_free.append(j)
                n += 1
            self._push_maps(plan)
            plan.pending = []
        _PREFETCH.owner = [None, None]
        self.ev_spare = torch.xpu.Event()
        self.ev_spare.record(fwd)
        return n, layers

    def _maybe_swap_async(self, now: float, np) -> None:
        import time

        if _capturing():
            return
        t0 = time.perf_counter()
        fwd = torch.xpu.current_stream()
        if self.phase == "idle":
            if now - self.t < self.period:
                return
            self.t = now
            for k, plan in enumerate(self.plans):
                self.snap[k].copy_(plan.route_counts, non_blocking=True)
            self.ev = torch.xpu.Event()
            self.ev.record(fwd)
            self.phase = "snap"
            self.cycle = {"t0": t0, "host": 0.0, "steps": 0, "n": 0}
        elif not self.ev.query():
            pass  # copies still running: the forward goes on with the current placement
        elif self.phase == "snap":
            moves = self._moves(np, now, self.snap.numpy())
            if moves and self._copy1(moves):
                self.phase = "copy1"
            else:
                self.phase = "idle"
        elif self.phase == "copy1":
            self._flip1_copy2(fwd)
            self.phase = "copy2"
        elif self.phase == "copy2":
            n, layers = self._flip2(fwd)
            self.total += n
            self.phase = "idle"
            c = self.cycle
            c["host"] += time.perf_counter() - t0
            logger.info("B70 hot/cold swap rank %d: %d experts in %d layers async: %.1f ms wall over %d steps, "
                        "host %.2f ms (%d since start)", _rank(), n, layers, (time.perf_counter() - c["t0"]) * 1e3,
                        c["steps"] + 1, c["host"] * 1e3, self.total)
            return
        if self.cycle is not None:
            self.cycle["host"] += time.perf_counter() - t0
            self.cycle["steps"] += 1


_SWAPPER = _Swapper()


def maybe_swap() -> None:
    if swap_enabled():
        _SWAPPER.maybe_swap()


class HotColdPlan:
    """Which local experts of one layer live where, plus the cold storage."""

    def __init__(self, layer_name: str, local_ids: list[int], global_ids: list[int],
                 num_global: int, hot_local: list[int]):
        self.layer_name = layer_name
        self.num_global = num_global
        hot_set = set(hot_local)
        self.hot_local = sorted(hot_set)
        self.cold_local = [i for i in local_ids if i not in hot_set]
        # local id -> (is_cold, slot)
        self.slot: dict[int, tuple[bool, int]] = {}
        for s, i in enumerate(self.hot_local):
            self.slot[i] = (False, s)
        for s, i in enumerate(self.cold_local):
            self.slot[i] = (True, s)
        self.local_to_global = dict(zip(local_ids, global_ids))
        self.twins: dict[str, torch.nn.Parameter] = {}
        self._param_names: dict[int, str] = {}
        self.cold: dict[str, torch.Tensor] = {}  # pinned, kernel layout
        self.cold_views: dict[str, torch.Tensor] = {}
        self.hot_map: torch.Tensor | None = None
        self.cold_map: torch.Tensor | None = None
        self._impls = None
        self._decode_args = None
        self.spare = 0  # B70-Q27-0003: free cold rows after the cold experts (async swap only)

    @property
    def num_hot(self) -> int:
        return len(self.hot_local)

    @property
    def num_cold(self) -> int:
        return len(self.cold_local)

    @property
    def cold_rows(self) -> int:
        """Rows of the cold tensors: the cold experts plus the async swap's spare rows."""
        return self.num_cold + self.spare

    # -- load ---------------------------------------------------------------

    def attach_twins(self, layer: torch.nn.Module) -> None:
        """A pageable CPU twin [num_cold, ...] for every per-expert param."""
        for name, param in layer.named_parameters(recurse=False):
            if name not in _EXPERT_PARAMS or param.ndim == 0 or param.shape[0] != self.num_hot:
                continue
            twin = torch.nn.Parameter(
                # explicit CPU: the model is built under a default-device context
                torch.zeros((self.num_cold, *param.shape[1:]), dtype=param.dtype,
                            device="cpu"),
                requires_grad=False,
            )
            for key, value in vars(param).items():
                if not key.startswith("_"):
                    setattr(twin, key, value)
            self.twins[name] = twin
            self._param_names[id(param)] = name

    def route(self, param: torch.nn.Parameter, local_id: int):
        """(param, slot) the loader should write expert ``local_id`` into."""
        is_cold, slot = self.slot[local_id]
        if not is_cold:
            return param, slot
        name = self._param_names.get(id(param))
        if name is None:
            raise RuntimeError(
                f"B70 hot/cold: {self.layer_name}: no cold twin for a param of "
                f"shape {tuple(param.shape)}"
            )
        return self.twins[name], slot

    # -- after loading ---------------------------------------------------------

    def finalize(self, layer: torch.nn.Module, moe_kernel) -> None:
        """Repack the cold twins into pinned kernel-layout tensors; wire the kernel."""
        from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor

        for bias in ("w13_bias", "w2_bias"):
            if getattr(layer, bias, None) is not None:
                raise NotImplementedError(f"B70 hot/cold: {bias} is not supported")
        twins, self.twins, self._param_names = self.twins, {}, {}
        self.spare = spare_rows()
        if self.spare and self.num_cold + self.spare == 128:
            self.spare += 1  # 128 local experts would route the cold impl into the astra direct path
        # Same transforms as int_wna16._process_weights_xpu, then implement_zp.
        for key in ("w13", "w2"):
            self.cold[key] = _repack_int4_pinned(
                twins.pop(f"{key}_qweight").data, layer.w13_qweight.device, self.spare
            )
            scales = twins.pop(f"{key}_scales").data.transpose(1, 2).contiguous()
            self.cold[f"{key}_scales"] = _pin(scales.to(getattr(layer, f"{key}_scales").dtype), self.spare)
            del scales
        del twins  # qzeros: symmetric int4 has none on XPU
        for key, tensor in self.cold.items():
            self.cold_views[key] = get_accelerator_view_from_cpu_tensor(tensor)

        device = layer.w13_qweight.device
        hot_map = torch.full((self.num_global,), -1, dtype=torch.int32)
        cold_map = torch.full((self.num_global,), -1, dtype=torch.int32)
        for local_id, (is_cold, slot) in self.slot.items():
            (cold_map if is_cold else hot_map)[self.local_to_global[local_id]] = slot
        self.hot_map = hot_map.to(device)
        self.cold_map = cold_map.to(device)
        match = _LAYER_RE.search(self.layer_name)
        self.layer_idx = int(match.group(1)) if match else len(_PLANS)
        _PLANS[self.layer_idx] = self
        self.route_counts = None
        if route_count_enabled():
            # one bucket per global expert + one for negative (padding) ids
            self.route_counts = torch.zeros(2, self.num_global + 1, dtype=torch.int64,
                                            device=device)
            self.count_graphs = os.environ.get("B70_MOE_ROUTE_COUNT") == "1"
            _COUNTED.append(self)
            if str(device) not in _ONES:
                _ONES[str(device)] = torch.ones(_ONES_N, dtype=torch.int64, device=device)

        experts = getattr(moe_kernel, "fused_experts", None)
        if experts is None or not hasattr(experts, "b70_apply_hotcold"):
            raise NotImplementedError(
                f"B70 hot/cold needs XPUExpertsWNA16, got {type(experts).__name__}"
            )
        experts.b70_hotcold = self
        gib = sum(t.numel() * t.element_size() for t in self.cold.values()) / 2**30
        logger.debug("B70 hot/cold: %s: %d hot in VRAM, %d cold pinned (%.2f GiB); "
                     "arena %.0f GiB", self.layer_name, self.num_hot, self.num_cold,
                     gib, _ARENA.gib)
        _FINALIZED.append(self.layer_name)
        if len(_FINALIZED) % 12 == 0 or os.environ.get("B70_MOE_HOTCOLD_MEMLOG") == "1":
            free, _ = torch.xpu.mem_get_info(device)
            logger.info("B70 hot/cold: %d layers split (%d hot in VRAM, %d cold); "
                        "cold pinned arena %.0f GiB; device: torch allocated %.2f GiB, "
                        "reserved %.2f GiB, free %.2f GiB", len(_FINALIZED), self.num_hot,
                        self.num_cold, _ARENA.gib,
                        torch.xpu.memory_allocated(device) / 2**30,
                        torch.xpu.memory_reserved(device) / 2**30, free / 2**30)

    # -- forward ---------------------------------------------------------------

    def apply_cold(self, cold, output, hidden_states, topk_weights, topk_ids) -> None:
        """The cold experts' share: UVA reads (decode, graphs) or a VRAM slot (large eager batches).

        Graph capture always takes the UVA path, whatever the batch size (B70-V32-HC3): the
        prefetcher's side stream and events cannot live inside a graph. Same kernel, same bytes,
        same math; a captured 512/1024-token graph reads its cold experts over PCIe."""
        min_tokens = cold_prefetch_min_tokens()
        if not min_tokens or hidden_states.shape[0] < min_tokens or _capturing():
            if min_tokens and hidden_states.shape[0] >= min_tokens:
                logger.info_once("B70 hot/cold: graph capture at %d tokens reads cold experts over "
                                 "UVA (prefetch is eager-only)", hidden_states.shape[0])
            from vllm.model_executor.layers.fused_moe import b70_moe_w4a8 as _w4a8

            # B70-IP-K: W4A8 reads the cold weights over UVA only on request (B70_MOE_W4A8_UVA=1)
            if not (_w4a8.uva_enabled()
                    and _w4a8.maybe_apply(cold, output, hidden_states, topk_weights, topk_ids)):
                cold.apply(output=output, hidden_states=hidden_states,
                           topk_weights=topk_weights, topk_ids=topk_ids)
            return
        import copy

        buf = _PREFETCH.acquire(self, hidden_states.device)
        staged = copy.copy(cold)
        staged.w13, staged.w2 = buf["w13"].view(torch.int8), buf["w2"].view(torch.int8)
        staged.gemm1_wei_scales, staged.gemm2_wei_scales = buf["w13_scales"], buf["w2_scales"]
        from vllm.model_executor.layers.fused_moe import b70_moe_w4a8 as _w4a8

        if not _w4a8.maybe_apply(staged, output, hidden_states, topk_weights, topk_ids):
            staged.apply(output=output, hidden_states=hidden_states,
                         topk_weights=topk_weights, topk_ids=topk_ids)
        _PREFETCH.release_and_prefetch(self)

    def apply_decode_op(self, output, hidden_states, topk_weights, topk_ids) -> bool:
        """B70 0048: small batches through the fused hot+cold decode op (b70moe.moe_decode); False = not taken."""
        if hidden_states.shape[0] > decode_op_max_rows() or hidden_states.dtype != torch.bfloat16:
            return False
        args = self._decode_args
        if args is False:
            return False
        if args is None:
            hot, cold = self._impls
            args = (self.hot_map, self.cold_map,
                    hot.w13, hot.gemm1_wei_scales, hot.w2, hot.gemm2_wei_scales,
                    cold.w13, cold.gemm1_wei_scales, cold.w2, cold.gemm2_wei_scales)
            if any(t is None or not t.is_contiguous() for t in args) or any(
                    t.dtype != torch.bfloat16 for t in args[3::2]):
                logger.warning("B70 decode op: %s: unexpected weight layout; stock path", self.layer_name)
                self._decode_args = False
                return False
            self._decode_args = args
        ids = topk_ids if topk_ids.dtype == torch.int32 else topk_ids.to(torch.int32)
        torch.ops.b70moe.moe_decode(output, hidden_states.contiguous(), topk_weights.contiguous(),
                                    ids.contiguous(), *args)
        return True

    def impls(self, w1, w1_scale, w2, w2_scale, topk, activation, extra):
        """(hot, cold) XpuFusedMoe, built on first use like the stock path."""
        if self._impls is None:
            from vllm_xpu_kernels.fused_moe_interface import XpuFusedMoe

            # ep_size > 1 only makes XpuFusedMoe use the expert_map passed here (never init_expert_map).
            # On 0.32 the hot/cold impls never take wu1ff group ownership or the astra direct path:
            # both require exactly 128 local experts (astra_ownership.prepare, astra_structural.direct_moe).
            common = dict(n_experts_per_token=topk, activation=activation,
                          ep_rank=0, ep_size=2, **extra)
            hot = XpuFusedMoe(w13=w1, w13_scales=w1_scale, w13_bias=None, w2=w2,
                              w2_scales=w2_scale, w2_bias=None,
                              num_experts=self.num_hot, expert_map=self.hot_map,
                              **common)
            cw13, cw2 = self.cold_views["w13"], self.cold_views["w2"]
            for t in (cw13, cw2):
                t.xpu_fused_moe = True  # already signed int4
            cold = XpuFusedMoe(w13=cw13, w13_scales=self.cold_views["w13_scales"],
                               w13_bias=None, w2=cw2,
                               w2_scales=self.cold_views["w2_scales"], w2_bias=None,
                               num_experts=self.cold_rows, expert_map=self.cold_map,
                               **common)
            # sizes come from the uint8 view at init; the kernel reads int4 as int8
            cold.w13, cold.w2 = cw13.view(torch.int8), cw2.view(torch.int8)
            self._impls = (hot, cold)
        return self._impls


def hot_count_for(num_local: int) -> int:
    """B70-V32-HC1: how many of a rank's ``num_local`` experts per layer stay in VRAM.

    B70_MOE_COLD_EXPERTS=N (cold per layer per rank) wins over B70_MOE_HOT_EXPERTS (hot per layer per rank),
    so one number means the same VRAM saving whatever the EP size: N = 16 is 16 of 128 at TP4 and 16 of
    256 at TP2. Neither set: everything hot (no plan). Clamped to [1, num_local]."""
    cold = os.environ.get("B70_MOE_COLD_EXPERTS", "").strip()
    if cold:
        hot = num_local - int(cold)
        if os.environ.get("B70_MOE_HOT_EXPERTS", "").strip():
            _warn_once("B70 hot/cold: B70_MOE_COLD_EXPERTS=%s wins over B70_MOE_HOT_EXPERTS=%s",
                       cold, os.environ["B70_MOE_HOT_EXPERTS"])
    else:
        hot = int(os.environ.get("B70_MOE_HOT_EXPERTS", str(num_local)))
    return max(1, min(hot, num_local))


_WARNED: set = set()


def _warn_once(msg: str, *args) -> None:
    if msg not in _WARNED:
        _WARNED.add(msg)
        logger.warning(msg, *args)


def remap_unrouted(topk_ids: torch.Tensor, num_global: int, ep_rank: int) -> torch.Tensor:
    """B70-V32-HC1: 0.32's stock XPU path sends topk_ids == -1 to an expert this rank does not own (the last
    expert on rank 0, expert 0 elsewhere: linear EP placement). That expert is -1 in both the hot and the
    cold map, so both kernel calls skip the row, as the stock call does."""
    sentinel = num_global - 1 if ep_rank == 0 else 0
    return torch.where(topk_ids < 0, topk_ids.new_full((), sentinel), topk_ids)


def maybe_plan(layer_name: str, moe_config, expert_map_manager,
               quant_config) -> HotColdPlan | None:
    """A plan for this RoutedExperts layer, or None (switch off / not applicable)."""
    if not hotcold_enabled() or quant_config is None:
        return None
    from vllm.platforms import current_platform

    if not current_platform.is_xpu():
        return None
    match = _LAYER_RE.search(layer_name)
    if match is None or "mtp" in layer_name:
        return None  # the MTP draft layer stays fully in VRAM
    layer_idx = int(match.group(1))
    num_global = moe_config.num_experts
    local_of = {g: expert_map_manager.map_global_to_local(g) for g in range(num_global)}
    pairs = sorted((loc, g) for g, loc in local_of.items() if loc >= 0)
    local_ids = [loc for loc, _ in pairs]
    global_ids = [g for _, g in pairs]
    hot_count = hot_count_for(len(local_ids))
    if hot_count >= len(local_ids):
        return None
    ranked = [g for g in _rankings().get(layer_idx, []) if local_of.get(g, -1) >= 0]
    seen = set(ranked)
    ranked += [g for g in global_ids if g not in seen]  # never observed: by id
    state = _learned_state()
    learned = state["layers"].get(layer_idx, {}) if state else {}
    if learned:
        # this host's long-term scores first, the checkpoint order as a small prior (and the tie-break)
        top = max(learned.values()) or 1.0
        w = float(os.environ.get("B70_MOE_HOTCOLD_STATE_PRIOR", "0.05"))
        prior = {g: 1 - i / len(ranked) for i, g in enumerate(ranked)}
        ranked = sorted(ranked, key=lambda g: -(learned.get(g, 0.0) / top + w * prior[g]))
    hot_local = [local_of[g] for g in ranked[:hot_count]]
    return HotColdPlan(layer_name, local_ids, global_ids, num_global, hot_local)
