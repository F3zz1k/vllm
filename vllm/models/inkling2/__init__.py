# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Inkling2. This tree serves the looped checkpoints (Inkling 2 Mini) on Intel
XPU (`xpu/`); configs, processing, quantization metadata and the DSpark
speculator config are shared with upstream. The NVIDIA implementation
(`nvidia/` upstream) is not carried."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .xpu.model import Inkling2LoopForConditionalGeneration

__all__ = ["Inkling2LoopForConditionalGeneration"]


def __getattr__(name: str):
    if name == "Inkling2LoopForConditionalGeneration":
        from .xpu import model

        return model.Inkling2LoopForConditionalGeneration
    if name in ("Inkling2ForConditionalGeneration", "Inkling2DSparkModel"):
        raise NotImplementedError(f"{name} is not ported to this tree yet")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
