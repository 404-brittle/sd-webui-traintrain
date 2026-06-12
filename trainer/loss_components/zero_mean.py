"""
Zero-mean normalised loss component.

Centers each channel independently by subtracting its spatial mean *before*
computing the base loss.  This removes the DC (bias/offset) component from
every channel, so only the **pattern of variation around the mean** — the
"relative strength" — contributes to the loss.

This is **not** the same as :class:`SpatialHighPassLoss`, which operates on
the loss tensor *after* the base loss has been computed.  ``ZeroMeanLoss``
modifies the inputs *before* the base loss, making it complement any existing
component.

Key properties
--------------
- **Purely structural** — absolute brightness/bias per channel is ignored.
- **Complements Sobel/Census/SSIM** — those are already shift-invariant by
  design; this adds the same property to the raw base loss.
- **No learned parameters** — deterministic, stable, zero overhead at inference.

Usage in config
---------------
::

    train_loss_function = composite
    train_loss_components = zero_mean(base=MSE) * 0.3
        + sobel(base=MSE) * 0.3
        + census(window=3) * 0.4

Config parameters
-----------------
- ``base`` — Base loss function: ``"MSE"`` (default), ``"L1"``, ``"Smooth-L1"``.
"""

from __future__ import annotations

from typing import Any

from .base import LossComponent


class ZeroMeanLoss(LossComponent):
    """Zero-mean normalised base loss — purely structural (DC-invariant).

    Each channel is centered independently over the spatial dimensions
    before the base loss is applied::

        pred_c  = pred  - pred.mean(dim=[2, 3], keepdim=True)
        targ_c  = target - target.mean(dim=[2, 3], keepdim=True)
        loss    = base_fn(pred_c, targ_c)

    This ensures absolute per-channel bias does not contribute to the loss —
    only the *pattern* of variation matters.

    Args:
        base_loss: Base loss function name (``"MSE"``, ``"L1"``, ``"Smooth-L1"``).
    """

    def __init__(self, base_loss: str = "MSE"):
        super().__init__(base_loss)

    # ------------------------------------------------------------------
    # Config factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        *,
        base_loss: str = "MSE",
        **_kwargs: Any,
    ) -> "ZeroMeanLoss":
        """Factory for config-string parsing."""
        return cls(base_loss=base_loss)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **context: Any,
    ) -> torch.Tensor:
        # Center each channel independently over spatial dims
        pred_c = pred - pred.mean(dim=[2, 3], keepdim=True)
        targ_c = target - target.mean(dim=[2, 3], keepdim=True)

        # Element-wise base loss on the centered representations
        return self._compute_base(pred_c, targ_c, self._base_fn)

    def extra_repr(self) -> str:
        return f"base={self._base_loss_name}"
