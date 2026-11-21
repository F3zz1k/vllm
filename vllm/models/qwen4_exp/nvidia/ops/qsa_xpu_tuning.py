# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""B70-QSA: one place for Intel XPU launch overrides of the QSA Triton kernels.

Measurement and tuning only. Nothing changes unless an override is set.

``B70_QSA_TUNE`` holds a JSON object (or ``@/path/to/file.json``) keyed by
kernel name; each value is a dict of launch choices. Example::

    B70_QSA_TUNE='{"sparse_gqa_prefill": {"block_n": 64, "splits": 1,
                   "num_warps": 8, "grf_mode": "256"},
                   "decode_score": {"num_warps": 4, "grf_mode": "auto"}}'

Kernel keys and the fields each one reads:

``decode_score``   num_warps, stages, block_n, tiles_per_prog, grf_mode
``prefill_score``  tile_r, block_n, k_tiles, stages, num_warps, grf_mode
``sparse_gqa_decode``   block_n, splits, num_warps, num_stages, grf_mode
                        (every base_programs region except the prefill one)
``sparse_gqa_prefill``  same fields; base_programs > 2048 with
                        use_prefill_config (the 4096-row prefill chunk)
``sparse_attn2_decode`` / ``sparse_attn2_prefill`` (B70_QSA_ATTN_IMPL=attn2)
                   block_n, num_dchunk (head_dim chunks), tiles_per_prog,
                   row_chunk, num_warps, num_stages, grf_mode
``sparse_attn2_i8_decode`` / ``sparse_attn2_i8_prefill`` (B70-IP-K8: the
                   same fields for the attn2 int8-dot launches, B70_QSA_I8DOT=1;
                   ``sparse_attn2_i8pv_*`` for B70_QSA_I8DOT=2)
``merge``          num_warps, grf_mode (also the attn2 merge)
``expand``         column_block, num_warps, grf_mode

The launch site and the JIT warmup read the same values, so warmup can never
compile a variant other than the one that runs. ``set_overrides`` is the
in-process form used by benchmarks/kernels/bench_qwen4_qsa_xpu.py; it clears
the cache of the dispatch helpers that memoise.

Only XPU reads overrides: ``grf_mode`` is an Intel backend option and would
be rejected by the CUDA backend.
"""

from __future__ import annotations

import json
import os
from typing import Any

from vllm.logger import init_logger
from vllm.platforms import current_platform

logger = init_logger(__name__)

_KNOWN = {
    "decode_score": {"num_warps", "stages", "block_n", "tiles_per_prog", "grf_mode"},
    "prefill_score": {
        "tile_r",
        "block_n",
        "k_tiles",
        "stages",
        "num_warps",
        "grf_mode",
    },
    "sparse_gqa_decode": {"block_n", "splits", "num_warps", "num_stages", "grf_mode"},
    "sparse_gqa_prefill": {"block_n", "splits", "num_warps", "num_stages", "grf_mode"},
    "sparse_attn2_decode": {
        "block_n",
        "num_dchunk",
        "tiles_per_prog",
        "row_chunk",
        "num_warps",
        "num_stages",
        "grf_mode",
    },
    "sparse_attn2_prefill": {
        "block_n",
        "num_dchunk",
        "tiles_per_prog",
        "row_chunk",
        "num_warps",
        "num_stages",
        "grf_mode",
    },
    # B70-IP-K8: the attn2 int8-dot (B70_QSA_I8DOT) launches tune separately.
    "sparse_attn2_i8_decode": {
        "block_n",
        "num_dchunk",
        "tiles_per_prog",
        "row_chunk",
        "num_warps",
        "num_stages",
        "grf_mode",
    },
    "sparse_attn2_i8_prefill": {
        "block_n",
        "num_dchunk",
        "tiles_per_prog",
        "row_chunk",
        "num_warps",
        "num_stages",
        "grf_mode",
    },
    "sparse_attn2_i8pv_decode": {
        "block_n",
        "num_dchunk",
        "tiles_per_prog",
        "row_chunk",
        "num_warps",
        "num_stages",
        "grf_mode",
    },
    "sparse_attn2_i8pv_prefill": {
        "block_n",
        "num_dchunk",
        "tiles_per_prog",
        "row_chunk",
        "num_warps",
        "num_stages",
        "grf_mode",
    },
    "merge": {"num_warps", "grf_mode"},
    "expand": {"column_block", "num_warps", "grf_mode"},
}
_GRF_MODES = {"default", "128", "256", "512", "auto"}

_overrides: dict[str, dict[str, Any]] | None = None


def _validate(raw: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for kernel, fields in raw.items():
        if kernel not in _KNOWN:
            raise ValueError(
                f"B70_QSA_TUNE: unknown kernel {kernel!r}; known: {sorted(_KNOWN)}"
            )
        if not isinstance(fields, dict):
            raise ValueError(f"B70_QSA_TUNE[{kernel!r}] must be an object")
        bad = set(fields) - _KNOWN[kernel]
        if bad:
            raise ValueError(
                f"B70_QSA_TUNE[{kernel!r}]: unknown fields {sorted(bad)}; "
                f"known: {sorted(_KNOWN[kernel])}"
            )
        grf = fields.get("grf_mode")
        if grf is not None and str(grf) not in _GRF_MODES:
            raise ValueError(f"B70_QSA_TUNE: grf_mode must be one of {_GRF_MODES}")
        out[kernel] = {
            k: (str(v) if k == "grf_mode" else int(v)) for k, v in fields.items()
        }
    return out


def _load() -> dict[str, dict[str, Any]]:
    global _overrides
    if _overrides is None:
        spec = os.environ.get("B70_QSA_TUNE", "").strip()
        if spec.startswith("@"):
            with open(spec[1:]) as f:
                spec = f.read()
        parsed = _validate(json.loads(spec)) if spec else {}
        if parsed and not current_platform.is_xpu():
            logger.warning_once("B70-QSA: B70_QSA_TUNE ignored off XPU")
            parsed = {}
        if parsed:
            logger.info_once(
                "B70-QSA: XPU launch overrides active: %s",
                json.dumps(parsed, sort_keys=True),
            )
        _overrides = parsed
    return _overrides


def set_overrides(raw: dict[str, dict[str, Any]] | None) -> None:
    """Replace the overrides in-process (benchmarks). None = re-read the env."""
    global _overrides
    _overrides = None if raw is None else _validate(raw)


def get(kernel: str, field: str, default: Any) -> Any:
    """Override for one field, or ``default``."""
    return _load().get(kernel, {}).get(field, default)


def launch_extra(kernel: str) -> dict[str, Any]:
    """Extra Triton launch kwargs (today only ``grf_mode``) for ``kernel``."""
    grf = _load().get(kernel, {}).get("grf_mode")
    return {"grf_mode": grf} if grf is not None and grf != "default" else {}
