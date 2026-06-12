"""
Per-channel latent weighting and colour-subspace projection loss component.

Provides **two strategies** for reducing colour influence in the training
signal, both derived from the diagnostic output of
:mod:`diagnostics.channel_sensitivity`:

1. **Simple per-channel weighting** (``mode=channel_weight``)
   Each of the 16 VAE latent channels gets a multiplier in ``(0, 1]``.
   Colour-sensitive channels receive low weight.

2. **PCA colour-subspace projection** (``mode=pca_projection``)
   The loss is computed **only in the structure subspace** — the
   complement of the principal components that capture colour variation.
   This is a more principled approach: instead of down-weighting channels,
   the loss is *blind* to colour by projecting onto directions in latent
   space that are nearly colour-invariant.

Usage in config
---------------
::

    train_loss_function = composite
    # Simple per-channel weights (backward compat):
    train_loss_components = ch_weight(path=./weights/ch16.pt;base=MSE)
    # PCA projection (more principled — loss blind to colour):
    train_loss_components = ch_weight(path=./weights/ch16.pt;mode=pca_projection;base=MSE)
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import torch

from .base import LossComponent


class ChannelWeightLoss(LossComponent):
    """Weight / project latent channels to reduce colour influence.

    Two modes (set via ``mode`` config param):

    * ``channel_weight`` (default) — simple per-channel multiplier.
    * ``pca_projection`` — loss is computed in the structure subspace
      (complement of the colour PCA components).

    Args:
        analysis_dict: The full dictionary produced by
            :func:`diagnostics.channel_sensitivity.analyse_channel_sensitivity`.
            If provided, *weights* / *eigvecs* etc. are extracted from it.
        weights: ``[C]`` tensor of per-channel multipliers (used when
            *analysis_dict* is not provided, or in ``channel_weight`` mode).
        mode: ``"channel_weight"`` or ``"pca_projection"``.
        base_loss: Base loss function name.
    """

    def __init__(
        self,
        analysis_dict: Optional[Dict[str, Any]] = None,
        weights: Optional[torch.Tensor] = None,
        mode: str = "channel_weight",
        base_loss: str = "MSE",
    ):
        super().__init__(base_loss)
        self._mode = mode

        if analysis_dict is not None:
            self._init_from_analysis(analysis_dict)
        elif weights is not None:
            self._weights = weights.clone().detach().float()
            self._projection_matrix = None  # not used in simple mode
        else:
            # Fallback: uniform weights (no-op)
            self._weights = torch.ones(16, dtype=torch.float32)
            self._projection_matrix = None
            self._mode = "channel_weight"

    def _init_from_analysis(self, d: Dict[str, Any]):
        """Extract weights / PCA data from the diagnostic output dict."""
        self._weights = d.get("channel_weights", torch.ones(16)).clone().detach().float()

        # PCA projection matrix: eigenvectors spanning the structure subspace
        eigvecs = d.get("colour_subspace_eigvecs")  # [C, C]
        n_colour = d.get("n_colour_dimensions", 0)
        if eigvecs is not None and n_colour is not None and n_colour < eigvecs.shape[1]:
            # Structure subspace = trailing eigenvectors
            self._projection_matrix = eigvecs[:, n_colour:].clone().detach().float()
            # [C, C - n_colour]  — maps latent → structure-subspace coordinates
        else:
            self._projection_matrix = None

    # ------------------------------------------------------------------
    # Config factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        *,
        path: str = "",
        mode: str = "channel_weight",
        base_loss: str = "MSE",
        latent_channels: int = 16,
        **_kwargs,
    ) -> "ChannelWeightLoss":
        """Factory: parse config kwargs into an instance.

        If *path* points to a ``.pt`` file produced by the diagnostic, it
        is loaded and all derived quantities (weights, PCA, frequency
        profiles) are extracted.
        """
        if path and os.path.isfile(path):
            data = torch.load(path, map_location="cpu", weights_only=False)
            if isinstance(data, dict) and "channel_weights" in data:
                # Full analysis dict
                return cls(analysis_dict=data, mode=mode, base_loss=base_loss)
            elif isinstance(data, torch.Tensor) and data.numel() == latent_channels:
                # Legacy: just a [16] weight tensor
                return cls(weights=data.float(), mode=mode, base_loss=base_loss)

        # Fallback
        print(
            f"WARNING: ChannelWeightLoss — no valid weight file at {path!r}. "
            "Using uniform weights.  Run diagnostics/channel_sensitivity.py."
        )
        return cls(
            weights=torch.ones(latent_channels).float(),
            mode=mode,
            base_loss=base_loss,
        )

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **context: Any,
    ) -> torch.Tensor:
        loss = self._compute_base(pred, target, self._base_fn)  # [B, C, H, W]

        if self._mode == "pca_projection" and self._projection_matrix is not None:
            return self._forward_projection(loss)
        else:
            return self._forward_channel_weight(loss)

    def _forward_channel_weight(self, loss: torch.Tensor) -> torch.Tensor:
        """Simple per-channel weighting."""
        w = self._weights.to(loss.device, dtype=loss.dtype)
        return loss * w.view(1, -1, 1, 1)

    def _forward_projection(self, loss: torch.Tensor) -> torch.Tensor:
        """PCA projection: compute loss only in the structure subspace.

        The idea:
        1. Reshape loss ``[B, C, H, W]`` → ``[B*H*W, C]``
        2. Project onto structure basis: ``proj = loss_flat @ P``  where
           ``P`` is ``[C, C - n_colour]``
        3. The loss magnitude in the structure subspace is the norm of
           the projected coordinates.

        This makes the loss *blind* to colour directions altogether,
        rather than simply down-weighting them.
        """
        B, C, H, W = loss.shape
        P = self._projection_matrix.to(loss.device, dtype=loss.dtype)  # [C, S]

        # Flatten spatial+batch
        loss_flat = loss.view(B * H * W, C)       # [N, C]
        proj = loss_flat @ P                       # [N, S]   S = C - n_colour

        # Norm in structure subspace → per-pixel loss
        struct_loss = proj.norm(dim=1, keepdim=True)  # [N, 1]

        # Reshape back to [B, C, H, W] (broadcast single channel)
        # We keep C channels by repeating the structural loss
        struct_loss = struct_loss.view(B, 1, H, W).expand(B, C, H, W)
        return struct_loss

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def to(self, device: torch.device, dtype: Optional[torch.dtype] = None):
        self._weights = self._weights.to(device, dtype=dtype)
        if self._projection_matrix is not None:
            self._projection_matrix = self._projection_matrix.to(device, dtype=dtype)
        return self

    @property
    def device(self) -> torch.device:
        return self._weights.device

    def extra_repr(self) -> str:
        mode_str = f"mode={self._mode}"
        n_col = "N/A"
        if self._projection_matrix is not None:
            n_col = self._projection_matrix.shape[0] - self._projection_matrix.shape[1]
        return f"{mode_str}, base={self._base_loss_name}, colour_dims={n_col}"
