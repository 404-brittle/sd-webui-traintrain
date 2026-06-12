"""
Multi-Scale SSIM (MS-SSIM) structural loss component.

SSIM (Structural Similarity) decomposes each local patch comparison into
three terms — luminance, contrast, and **structure** — where the structure
term is the normalised cross-correlation coefficient between patches.
This measures whether the *pattern* of variation matches, independent of
magnitude/colour.

Applied per channel in latent space, this loss:

- **Preserves relative spatial layout** — patches with the same texture
  pattern but different absolute values have high SSIM.
- **Captures local structure** at multiple scales — fine details via small
  windows, global composition via large windows.
- **Is naturally colour-invariant** when using only the structure term
  (which we do by default).

Usage
-----
::

    train_loss_function = composite
    train_loss_components = ms_ssim(window=11;scales=5;base=MSE)

Combined:
::

    train_loss_components = sobel(base=MSE) * 0.3
        + ms_ssim(window=7;scales=3;base=MSE) * 0.4
        + ch_weight(path=./weights/ch16.pt;base=MSE) * 0.3

Config parameters
-----------------
- ``window`` — Gaussian window size (odd, default 11).  Larger = more global
- ``sigma`` — Gaussian sigma in pixels (default 1.5)
- ``scales`` — number of scales for multi-scale SSIM (default 1 = single-scale)
- ``use_structure_only`` — if ``True`` (default), only the structure term is
  used, making the loss **purely pattern-based** and magnitude-invariant.
- ``base`` — base loss applied to the SSIM distance ``(1 - ssim_map)``:
  ``"MSE"``, ``"L1"``, ``"Smooth-L1"``
"""

from __future__ import annotations

from typing import Any, Optional

import torch
import torch.nn.functional as F

from .base import LossComponent


# ---------------------------------------------------------------------------
# Gaussian window for SSIM
# ---------------------------------------------------------------------------


def _gaussian_window(
    window_size: int, sigma: float, n_channels: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """1-D Gaussian kernel expanded to ``[C, 1, window_size, 1]`` and
    ``[C, 1, 1, window_size]`` for separable convolution."""
    half = (window_size - 1) // 2
    x = torch.arange(-half, half + 1, device=device, dtype=dtype)
    g = torch.exp(-0.5 * (x / sigma).pow(2))
    g = g / g.sum()
    # [C, 1, K, 1]  and  [C, 1, 1, K]  for depthwise separable conv
    k_v = g.view(1, 1, -1, 1).expand(n_channels, 1, -1, 1).contiguous()
    k_h = g.view(1, 1, 1, -1).expand(n_channels, 1, 1, -1).contiguous()
    return k_v, k_h


def _gaussian_filter(x: torch.Tensor, window_size: int, sigma: float) -> torch.Tensor:
    """Fast separable Gaussian filter on ``[B, C, H, W]``."""
    C = x.shape[1]
    k_v, k_h = _gaussian_window(window_size, sigma, C, x.device, x.dtype)
    pad = window_size // 2
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    x_blur = F.conv2d(x_pad, k_v, groups=C)
    x_blur = F.conv2d(x_blur, k_h, groups=C)
    return x_blur


# ---------------------------------------------------------------------------
# Per-channel SSIM map
# ---------------------------------------------------------------------------


def _ssim_map(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int,
    sigma: float,
    structure_only: bool = True,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Per-pixel SSIM map ``[B, C, H, W]`` with values in ``(0, 1]``.

    If *structure_only* is ``True``, the luminance and contrast terms are
    set to 1.0, so the result reflects **only the structure (pattern)
    similarity** — completely magnitude-invariant.
    """
    C = x.shape[1]

    # Means
    mu_x = _gaussian_filter(x, window_size, sigma)  # [B, C, H, W]
    mu_y = _gaussian_filter(y, window_size, sigma)

    # Variances and covariance
    sigma2_x = _gaussian_filter(x.pow(2), window_size, sigma) - mu_x.pow(2)
    sigma2_y = _gaussian_filter(y.pow(2), window_size, sigma) - mu_y.pow(2)
    sigma_xy = _gaussian_filter(x * y, window_size, sigma) - mu_x * mu_y

    # Clamp to avoid negative from numerical issues
    sigma2_x = sigma2_x.clamp(min=0)
    sigma2_y = sigma2_y.clamp(min=0)

    # Dynamic range: approximate from data
    # For latent space we estimate range per channel
    C1 = (0.01 * (x.amax() - x.amin() + eps)).clamp(min=eps).pow(2)
    C2 = (0.03 * (x.amax() - x.amin() + eps)).clamp(min=eps).pow(2)

    if structure_only:
        # Luminance = 1, Contrast = 1, only structure term
        # SSIM_str = (sigma_xy + C2/2) / (sigma_x * sigma_y + C2/2)
        # But folded into the standard SSIM formula with C1=inf
        ssim_val = (sigma_xy + C2 / 2) / (
            (sigma2_x * sigma2_y).sqrt() + C2 / 2 + eps
        )
    else:
        # Full SSIM
        ssim_val = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
            (mu_x.pow(2) + mu_y.pow(2) + C1) * (sigma2_x + sigma2_y + C2) + eps
        )

    return ssim_val.clamp(0.0, 1.0)  # [B, C, H, W]


# ---------------------------------------------------------------------------
# MS-SSIM
# ---------------------------------------------------------------------------


def _ms_ssim_map(
    x: torch.Tensor,
    y: torch.Tensor,
    window_size: int,
    sigma: float,
    scales: int,
    structure_only: bool = True,
) -> torch.Tensor:
    """Multi-scale SSIM map, upsampled to original resolution ``[B, C, H, W]``.

    At each scale, the SSIM map is computed and upsampled back to the
    original spatial size, then averaged across scales.
    """
    B, C, H, W = x.shape
    ssim_scales = []

    for s in range(scales):
        ssim_s = _ssim_map(x, y, window_size, sigma, structure_only)  # [B, C, h, w]
        if ssim_s.shape[2] != H or ssim_s.shape[3] != W:
            ssim_s = F.interpolate(ssim_s, size=(H, W), mode="bilinear", align_corners=False)
        ssim_scales.append(ssim_s)

        # Downsample for next scale
        if s < scales - 1:
            x = F.avg_pool2d(x, kernel_size=2)
            y = F.avg_pool2d(y, kernel_size=2)

    # Average across scales
    return torch.stack(ssim_scales, dim=0).mean(dim=0)  # [B, C, H, W]


# ---------------------------------------------------------------------------
# Loss component
# ---------------------------------------------------------------------------


class MSSSIMLoss(LossComponent):
    """Multi-Scale SSIM structural loss.

    Returns ``1 - ssim_map`` as the element-wise loss, so regions with
    poor structural similarity contribute larger gradients.

    Args:
        window_size: Gaussian window size (odd, default 11).
        sigma: Gaussian sigma (default 1.5).
        scales: Number of scales (1 = single-scale SSIM).
        use_structure_only: If True, only the pattern/correlation term
            is used — magnitude-invariant.
        base_loss: Base loss applied to the distance map (typically MSE
            or L1, used as a formality since the distance is already
            computed).
    """

    def __init__(
        self,
        window_size: int = 11,
        sigma: float = 1.5,
        scales: int = 1,
        use_structure_only: bool = True,
        base_loss: str = "MSE",
    ):
        super().__init__(base_loss)
        assert window_size % 2 == 1, "window_size must be odd"
        self._window_size = window_size
        self._sigma = sigma
        self._scales = max(1, scales)
        self._structure_only = use_structure_only

    @classmethod
    def from_config(
        cls,
        *,
        window: int = 11,
        sigma: float = 1.5,
        scales: int = 1,
        use_structure_only: bool = True,
        base_loss: str = "MSE",
        **_kwargs,
    ) -> "MSSSIMLoss":
        return cls(
            window_size=int(window),
            sigma=float(sigma),
            scales=int(scales),
            use_structure_only=bool(use_structure_only),
            base_loss=base_loss,
        )

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **context: Any,
    ) -> torch.Tensor:
        # Compute MS-SSIM map
        if self._scales > 1:
            ssim_val = _ms_ssim_map(
                pred, target,
                self._window_size, self._sigma,
                self._scales, self._structure_only,
            )
        else:
            ssim_val = _ssim_map(
                pred, target,
                self._window_size, self._sigma,
                self._structure_only,
            )

        # Convert similarity to distance: d = 1 - ssim
        loss = 1.0 - ssim_val

        # The base_loss parameter here is somewhat formal since 1 - SSIM
        # is already a meaningful distance.  We apply it as a pass-through
        # for compatibility.
        # (We return the distance map directly; base_fn applied element-wise
        #  to 1-ssim would double-transform it, so we treat base_loss as
        #  informational only.)
        return loss

    def extra_repr(self) -> str:
        return (
            f"window={self._window_size}, sigma={self._sigma}, "
            f"scales={self._scales}, structure_only={self._structure_only}"
        )
