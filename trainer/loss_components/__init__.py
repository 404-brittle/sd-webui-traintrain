"""
Composable loss components for the TrainTrain training loop.

Provides a pluggable, swappable system where each loss component implements
a standard interface and can be composed via CompositeLoss.  The default
(legacy) path — MSE / L1 / Smooth-L1 — remains unchanged when
``train_loss_function`` is not ``"composite"``.

Quick start
-----------
>>> from trainer.loss_components import CompositeLoss, build_composite_loss
>>> composite = build_composite_loss(
...     spec="ch_weight(path=./weights/ch16.pt;base=MSE)",
...     device="cuda",
...     latent_channels=16,
... )
>>> loss = composite(pred, target, timesteps=ts, latents=latents)

"""

from .base import LossComponent
from .channel_weight import ChannelWeightLoss
from .spatial_highpass import SpatialHighPassLoss
from .sobel_gradient import SobelGradientLoss
from .ms_ssim import MSSSIMLoss
from .census import CensusLoss
from .zero_mean import ZeroMeanLoss
from .composite import CompositeLoss, build_composite_loss, parse_component_specs

__all__ = [
    "LossComponent",
    "ChannelWeightLoss",
    "SpatialHighPassLoss",
    "SobelGradientLoss",
    "MSSSIMLoss",
    "CensusLoss",
    "ZeroMeanLoss",
    "CompositeLoss",
    "build_composite_loss",
    "parse_component_specs",
]
