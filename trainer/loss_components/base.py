"""
Abstract base class for all loss components.

Each component computes an **element-wise** loss tensor of shape ``[B, C, H, W]``
so that components can be composed (summed, weighted) **before** the spatial
reduction step in ``process_loss``.  This preserves compatibility with the
existing mask-weighting and timestep-weighting machinery.
"""

from __future__ import annotations

import abc
from typing import Any, Dict

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Lookup table  —  maps short names to base-loss functions
# ---------------------------------------------------------------------------
_BASE_LOSS_FN: Dict[str, Any] = {
    "MSE": F.mse_loss,
    "L1": F.l1_loss,
    "Smooth-L1": F.smooth_l1_loss,
}


def get_base_loss_fn(name: str):
    """Return the base-loss callable for *name* (e.g. ``"MSE"``)."""
    fn = _BASE_LOSS_FN.get(name)
    if fn is None:
        raise ValueError(
            f"Unknown base loss {name!r}.  Choose from {list(_BASE_LOSS_FN)}"
        )
    return fn


# ---------------------------------------------------------------------------
# Abstract component
# ---------------------------------------------------------------------------


class LossComponent(abc.ABC):
    """Interface for a single loss term.

    Subclasses must override ``forward()`` which returns an element-wise
    loss tensor of shape ``[B, C, H, W]``.
    """

    def __init__(self, base_loss: str = "MSE"):
        self._base_loss_name = base_loss
        self._base_fn = get_base_loss_fn(base_loss)

    # --- public API -------------------------------------------------------

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **context: Any,
    ) -> torch.Tensor:
        """Delegates to :meth:`forward` — makes instances callable."""
        return self.forward(pred, target, **context)

    @abc.abstractmethod
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **context: Any,
    ) -> torch.Tensor:
        """Compute the element-wise loss.

        Args:
            pred: Predicted velocity  ``[B, C, H, W]``.
            target: Target velocity  ``[B, C, H, W]``.
            context: Arbitrary keyword arguments provided by the training
                loop (e.g. ``timesteps``, ``latents``, ``noise``, ``step``,
                the trainer object ``t``).

        Returns:
            Element-wise loss  ``[B, C, H, W]``.
        """
        ...

    def extra_repr(self) -> str:
        """Short string for debugging / logging."""
        return f"base={self._base_loss_name}"

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.extra_repr()})"

    # --- helpers ----------------------------------------------------------

    @staticmethod
    def _compute_base(
        pred: torch.Tensor,
        target: torch.Tensor,
        base_fn: Any,
    ) -> torch.Tensor:
        """Convenience: apply the base loss with ``reduction='none'``."""
        return base_fn(pred, target, reduction="none")
