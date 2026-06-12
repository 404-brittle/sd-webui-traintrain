"""
Spatial high-pass loss component.

Colour information lives predominantly in **low spatial frequencies** (broad
uniform regions), while edges, texture, and structure occupy mid-to-high
frequencies.  This component applies a Gaussian low-pass filter to the
element-wise loss tensor, then up-weights the high-frequency residual:

    final = lowpass * (1 / strength) + (loss - lowpass) * strength

So mismatches in high-frequency (structural) detail are penalised more heavily
than mismatches in low-frequency (colour) fields.

Usage in config
---------------
::

    train_loss_function = composite
    train_loss_components = sp_hpf(kernel=5;sigma=2.0;strength=2.0;base=MSE)

"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from .base import LossComponent


def _gaussian_kernel_1d(kernel_size: int, sigma: float) -> torch.Tensor:
    """1-D Gaussian kernel (unnormalised)."""
    half = (kernel_size - 1) // 2
    x = torch.arange(-half, half + 1, dtype=torch.float32)
    g = torch.exp(-0.5 * (x / sigma).pow(2))
    return g / g.sum()


def _gaussian_blur_2d(x: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
    """Fast separable 2-D Gaussian blur on ``[B, C, H, W]``.

    Blurs each channel independently using depthwise (group-wise) convolution.
    """
    C = x.shape[1]
    # Build 1-D kernels
    k_base = _gaussian_kernel_1d(kernel_size, sigma).to(x.device, dtype=x.dtype)

    # Depthwise: each channel gets its own copy of the 1-D kernel
    k_v = k_base.view(1, 1, -1, 1).expand(C, 1, -1, 1).contiguous()   # [C, 1, K, 1]
    k_h = k_base.view(1, 1, 1, -1).expand(C, 1, 1, -1).contiguous()   # [C, 1, 1, K]

    # Pad
    pad_h = kernel_size // 2
    pad_w = kernel_size // 2
    x_pad = F.pad(x, (pad_w, pad_w, pad_h, pad_h), mode="reflect")

    # Separable depthwise convolution
    x_blur = F.conv2d(x_pad, k_v, groups=C)  # vertical
    x_blur = F.conv2d(x_blur, k_h, groups=C)  # horizontal
    return x_blur


class SpatialHighPassLoss(LossComponent):
    """Down-weight low-frequency (colour-field) loss, up-weight high-frequency
    (edge/texture) loss.

    Args:
        strength: How much to amplify the high-frequency residual.
            ``strength=1.0`` is a no-op (flat).  ``strength > 1`` penalises
            structural errors more.
        kernel_size: Gaussian kernel size (odd).  Larger = broader low-pass.
        sigma: Gaussian sigma in pixels (latent-space pixels, i.e. 8× pixel px).
        base_loss: Base loss function name.
    """

    def __init__(
        self,
        strength: float = 2.0,
        kernel_size: int = 5,
        sigma: float = 2.0,
        base_loss: str = "MSE",
    ):
        super().__init__(base_loss)
        assert kernel_size % 2 == 1, "kernel_size must be odd"
        self._strength = strength
        self._kernel_size = kernel_size
        self._sigma = sigma

    @classmethod
    def from_config(
        cls,
        *,
        strength: float = 2.0,
        kernel_size: int = 5,
        sigma: float = 2.0,
        base_loss: str = "MSE",
        **_kwargs,
    ) -> "SpatialHighPassLoss":
        return cls(
            strength=float(strength),
            kernel_size=int(kernel_size),
            sigma=float(sigma),
            base_loss=base_loss,
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **context: Any,
    ) -> torch.Tensor:
        loss = self._compute_base(pred, target, self._base_fn)  # [B, C, H, W]

        # Low-pass filter the loss tensor
        lowpass = _gaussian_blur_2d(loss, self._kernel_size, self._sigma)
        highpass = loss - lowpass

        # Recombine with emphasis on high frequencies
        s = self._strength
        return lowpass * (1.0 / s) + highpass * s

    def extra_repr(self) -> str:
        return (
            f"base={self._base_loss_name}, "
            f"strength={self._strength}, "
            f"kernel={self._kernel_size}, "
            f"sigma={self._sigma}"
        )
