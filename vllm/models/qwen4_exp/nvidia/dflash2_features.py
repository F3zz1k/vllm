# SPDX-License-Identifier: Apache-2.0
"""DFlash2 features from completed Qwen4Exp hyper-connection layers."""

import os

import torch

from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import EagleModelMixin

TARGET_LAYER_IDS = (5, 15, 25, 35, 45)
# vLLM converts DFlash's completed zero-based layer IDs to these boundaries.
AUX_LAYER_BOUNDARIES = tuple(i + 1 for i in TARGET_LAYER_IDS)

logger = init_logger(__name__)
# B70-V7-AUX (B70_DFLASH2_AUX_REUSE=1): take each feature from the combine the
# next layer's attention HC computes anyway (combine_and_mix returns the
# materialized multi-stream state) instead of a second hc_combine.
_AUX_REUSE = os.environ.get("B70_DFLASH2_AUX_REUSE", "0") == "1"
_aux_logged = False


def aux_feature(completed: torch.Tensor) -> torch.Tensor:
    # HC is the outer axis; TP4 owns replicated full-width token rows.
    # mean allocates independent storage, so later workspace reuse cannot
    # overwrite these snapshots. Preserve packed token order verbatim.
    return completed.unflatten(-1, (4, 2560)).mean(dim=-2)


class DFlash2FeatureCapture(EagleModelMixin):
    def _set_aux_hidden_state_layers(self, layers: tuple[int, ...]) -> None:
        if layers and tuple(layers) != AUX_LAYER_BOUNDARIES:
            raise ValueError(
                "Flash-Next DFlash2 requires completed layers 5/15/25/35/45 "
                "as vLLM boundaries 6/16/26/36/46"
            )
        if layers and (
            self.config.hc_count != 4
            or self.config.hidden_size != 2560
            or self.start_layer != 0
            or self.end_layer != 48
            or self.use_hc_sequence_parallel
        ):
            raise ValueError("DFlash2 capture requires unsharded HC4/H2560, PP1")
        super()._set_aux_hidden_state_layers(layers)

    def capture_completed_layer(
        self,
        aux_hidden_states: list[torch.Tensor],
        layer_idx: int,
        layer: torch.nn.Module,
        hidden_states: torch.Tensor,
        block_output: torch.Tensor | None,
        injection: torch.Tensor | None,
    ) -> None:
        if layer_idx + 1 not in self.aux_hidden_state_layers:
            return
        if (
            _AUX_REUSE
            and block_output is not None
            and layer_idx + 1 < self.end_layer
            and getattr(self.layers[layer_idx + 1], "ple", None) is None
        ):
            # Deferred: collect_deferred_feature appends it right after the
            # next layer, before any later boundary, so the order is kept.
            nxt = self.layers[layer_idx + 1]
            nxt._b70_stash_combined = True
            self._b70_aux_defer = layer_idx + 1
            global _aux_logged
            if not _aux_logged:
                _aux_logged = True
                logger.info(
                    "B70-V7-AUX DFlash2 aux features reuse the next layer's HC "
                    "combine (boundaries %s)",
                    sorted(self.aux_hidden_state_layers),
                )
            return
        # A decoder returns a delayed MLP injection. Materialize it for the
        # feature without consuming or modifying the tuple passed downstream.
        completed = hidden_states
        if block_output is not None:
            completed = layer.mlp_hyper_connection.combine(
                hidden_states, block_output, injection
            )
        aux_hidden_states.append(aux_feature(completed))

    def collect_deferred_feature(
        self,
        aux_hidden_states: list[torch.Tensor],
        layer_idx: int,
        layer: torch.nn.Module,
    ) -> None:
        """B70-V7-AUX: append the feature deferred from the previous boundary,
        from the combined state ``layer`` materialized in its attention HC."""
        if getattr(self, "_b70_aux_defer", None) != layer_idx:
            return
        self._b70_aux_defer = None
        combined = layer.__dict__.pop("_b70_combined", None)
        if combined is None:
            raise RuntimeError(
                f"B70-V7-AUX: layer {layer_idx} did not stash its HC combine"
            )
        aux_hidden_states.append(aux_feature(combined))
