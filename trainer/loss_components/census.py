"""
Census transform + Hamming distance loss component.

The Census transform replaces each pixel/latent value with a bitmask
comparing it to its neighbours:

::

    neighbour values:   [1.2, 0.8, 1.5]
                         [0.5,  *, 1.1]
                         [0.9, 1.3, 0.7]
    pixel > neighbour?   [1,   0,   1,  0,  1,  0,  1,  0]
                        (NW,  N,  NE,  W,  E,  SW,  S,  SE)

Two Census maps are compared via **Hamming distance** — the fraction of
neighbour-relations that differ.  This makes the loss:

- **Purely structural** — only the *ordering* of values matters, not the
  values themselves.  Completely colour/brightness invariant.
- **Captures local texture patterns** — brushstrokes, hatching, linework
  patterns produce characteristic binary signatures.
- **Captures relative spatial relationships** — if two pixels swap order
  (e.g. an edge passes between them), many bits flip.

Usage
-----
::

    train_loss_function = composite
    train_loss_components = census(window=3;base=MSE)

Combined:
::

    train_loss_components = sobel(base=MSE) * 0.3
        + ms_ssim(window=7;scales=3) * 0.4
        + census(window=5) * 0.3

Config parameters
-----------------
- ``window`` — neighbourhood radius (3 = 3×3, 5 = 5×5, etc.).  Odd only.
  Larger windows capture coarser spatial patterns.  Default 3.
- ``base`` — base loss applied to the Hamming distance (informational,
  since the distance is already a perceptual metric).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .base import LossComponent


def _census_hamming(
    x: torch.Tensor,
    y: torch.Tensor,
    radius: int,
) -> torch.Tensor:
    """Compute Hamming distance between Census transforms of ``x`` and ``y``.

    Args:
        x: ``[B, C, H, W]``
        y: ``[B, C, H, W]``
        radius: Neighbourhood radius (1 = 3×3, 2 = 5×5, etc.)

    Returns:
        Element-wise loss ``[B, C, H, W]`` where each value is the
        fraction of neighbour comparisons that disagree (in ``[0, 1]``).
    """
    B, C, H, W = x.shape
    device = x.device

    # Build offsets for all neighbours in the window
    offsets = []
    for dy in range(-radius, radius + 1):
        for dx in range(-radius, radius + 1):
            if dy == 0 and dx == 0:
                continue
            offsets.append((dy, dx))

    n_neighbours = len(offsets)
    # Accumulate Hamming distance across all neighbour comparisons
    hamming_sum = torch.zeros(B, C, H, W, device=device, dtype=torch.float32)

    for dy, dx in offsets:
        # Shift x and y
        x_shifted = torch.roll(x, shifts=(dy, dx), dims=(2, 3))
        y_shifted = torch.roll(y, shifts=(dy, dx), dims=(2, 3))

        # Census bits: 1 if centre > neighbour
        bit_x = (x > x_shifted).float()
        bit_y = (y > y_shifted).float()

        # Hamming contribution: bits differ
        hamming_sum += (bit_x != bit_y).float()

    # Average across neighbours: fraction of disagreeing comparisons
    return hamming_sum / n_neighbours  # [B, C, H, W]


class CensusLoss(LossComponent):
    """Census transform + Hamming distance loss.

    Purely structural: only the relative ordering of values matters,
    not the values themselves.  Completely colour/brightness invariant.

    Args:
        window_size: Neighbourhood window size (odd, default 3).
            3 = 3×3 (8 neighbours), 5 = 5×5 (24 neighbours), etc.
        base_loss: Base loss name (informational, since the Hamming
            distance is already a metric in ``[0, 1]``).
    """

    def __init__(
        self,
        window_size: int = 3,
        base_loss: str = "MSE",
    ):
        super().__init__(base_loss)
        assert window_size % 2 == 1, "window_size must be odd"
        self._radius = window_size // 2  # e.g. 3→1, 5→2

    @classmethod
    def from_config(
        cls,
        *,
        window: int = 3,
        base_loss: str = "MSE",
        **_kwargs,
    ) -> "CensusLoss":
        return cls(
            window_size=int(window),
            base_loss=base_loss,
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **context: Any,
    ) -> torch.Tensor:
        return _census_hamming(pred, target, self._radius)

    def extra_repr(self) -> str:
        win = 2 * self._radius + 1
        return f"window={win}x{win}, base={self._base_loss_name}"
