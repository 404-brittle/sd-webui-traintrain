"""
Sobel gradient edge-magnitude loss component.

Computes the spatial gradient (edge magnitude) of both ``pred`` and ``target``
via Sobel operators, then returns the element-wise absolute difference between
the edge maps.

This loss **penalises differences in where edges are located**, not differences
in the pixel/latent values themselves.  A shifted contour produces a
double-edge in the difference map; a colour change with preserved edges
produces near-zero gradient loss.

It naturally captures:
- **Edge/contour positions** — where are the boundaries between features?
- **Local structure patterns** — edge orientations and sharpness

Usage
-----
::

    train_loss_function = composite
    train_loss_components = sobel(base=MSE)

Combined with other components:
::

    train_loss_components = sobel(base=MSE) * 0.3
        + ch_weight(path=./weights/ch16.pt;mode=pca_projection;base=MSE) * 0.7

Config parameters
-----------------
- ``base`` — base loss to apply on edge maps: ``"MSE"``, ``"L1"``, ``"Smooth-L1"``
- ``normalise_edges`` — if ``True`` (default), edge magnitudes are normalised
  to ``[0, 1]`` per channel before comparison, making the loss purely
  structural and completely magnitude-invariant.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .base import LossComponent

# ---------------------------------------------------------------------------
# Sobel kernels (3×3)
# ---------------------------------------------------------------------------
_SOBEL_X = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
_SOBEL_Y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)


def _edge_magnitude(x: torch.Tensor) -> torch.Tensor:
    """Compute Sobel edge magnitude for each channel independently.

    Args:
        x: ``[B, C, H, W]``

    Returns:
        Edge magnitude ``[B, C, H, W]``.
    """
    B, C, H, W = x.shape
    device, dtype = x.device, x.dtype

    # Prepare kernels: [1, 1, 3, 3] → expand to groups=C depthwise conv
    kx = _SOBEL_X.to(device, dtype=dtype).view(1, 1, 3, 3)  # [1, 1, 3, 3]
    ky = _SOBEL_Y.to(device, dtype=dtype).view(1, 1, 3, 3)

    # Pad for 'same' convolution
    x_pad = F.pad(x, (1, 1, 1, 1), mode="reflect")

    # Depthwise convolution: each channel convolved independently
    # We need to expand kernel to [C, 1, 3, 3] for groups=C
    w_x = kx.expand(C, 1, 3, 3).contiguous()
    w_y = ky.expand(C, 1, 3, 3).contiguous()

    gx = F.conv2d(x_pad, w_x, groups=C)  # [B, C, H, W]
    gy = F.conv2d(x_pad, w_y, groups=C)  # [B, C, H, W]

    mag = (gx.pow(2) + gy.pow(2) + 1e-8).sqrt()
    return mag


class SobelGradientLoss(LossComponent):
    """Loss on Sobel edge magnitudes — penalises structural displacement.

    Args:
        normalise_edges: If ``True``, normalise each channel's edge map to
            ``[0, 1]`` before comparison.  This makes the loss purely about
            *where* edges are, ignoring edge strength.
        base_loss: Base loss function name.
    """

    def __init__(
        self,
        normalise_edges: bool = True,
        base_loss: str = "MSE",
    ):
        super().__init__(base_loss)
        self._normalise = normalise_edges

    @classmethod
    def from_config(
        cls,
        *,
        normalise_edges: bool = True,
        base_loss: str = "MSE",
        **_kwargs,
    ) -> "SobelGradientLoss":
        return cls(
            normalise_edges=bool(normalise_edges)
            if not isinstance(normalise_edges, bool)
            else normalise_edges,
            base_loss=base_loss,
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **context: Any,
    ) -> torch.Tensor:
        # Compute edge magnitudes
        edges_pred = _edge_magnitude(pred)    # [B, C, H, W]
        edges_targ = _edge_magnitude(target)  # [B, C, H, W]

        # Optional normalisation per channel
        if self._normalise:
            def _norm(emap):
                min_v = emap.amin(dim=[2, 3], keepdim=True)
                max_v = emap.amax(dim=[2, 3], keepdim=True)
                return (emap - min_v) / (max_v - min_v + 1e-8)
            edges_pred = _norm(edges_pred)
            edges_targ = _norm(edges_targ)

        # Element-wise loss on edge maps
        loss = self._compute_base(edges_pred, edges_targ, self._base_fn)
        return loss

    def extra_repr(self) -> str:
        return (
            f"base={self._base_loss_name}, "
            f"normalise_edges={self._normalise}"
        )
