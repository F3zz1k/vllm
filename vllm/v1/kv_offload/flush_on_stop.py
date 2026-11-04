# SPDX-License-Identifier: Apache-2.0
"""Flush on stop: on a graceful stop (SIGTERM), the offloading scheduler
writes the conversations still cached on the GPU to the KV pool before the
engine exits, newest first, through the write-back band. With the RAM tier as
a pure extension of VRAM, blocks otherwise reach the pool only when the GPU
evicts them, so a stop threw away every conversation still on the GPU.

VLLM_KV_FLUSH_ON_STOP_SEC bounds the flush (default 20 s, 0 disables). It
needs the band (VLLM_KV_WRITE_BACK=1); without it the stop is unchanged.

Kept free of heavy imports: the engine process manager reads it to know how
long to wait for the engine core before force-killing it.
"""

import os

# Seconds the engine core keeps running after it leaves its request loop,
# beyond the flush budget itself (worker and distributed teardown).
TEARDOWN_GRACE_SEC = 8.0


def flush_on_stop_sec() -> float:
    if os.environ.get("VLLM_KV_WRITE_BACK", "0") != "1":
        return 0.0
    try:
        value = float(os.environ.get("VLLM_KV_FLUSH_ON_STOP_SEC", "20"))
    except ValueError:
        return 0.0
    return max(0.0, value)


def engine_stop_grace_sec(vllm_config) -> float:
    """How long the process manager must wait for an engine core that may
    flush on stop (0 when it will not flush)."""
    if getattr(vllm_config, "kv_transfer_config", None) is None:
        return 0.0
    budget = flush_on_stop_sec()
    return budget + TEARDOWN_GRACE_SEC if budget > 0 else 0.0
