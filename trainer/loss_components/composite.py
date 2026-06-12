"""
Composite loss — weighted sum of multiple LossComponents.

Parses a config string like ::

    ch_weight(path=./weights/ch16.pt;base=MSE) + sp_hpf(kernel=5;strength=2.0)

into a list of ``(component, coefficient)`` pairs and sums their element-wise
outputs.  The result is still ``[B, C, H, W]`` and is passed to the existing
mask + spatial reduction logic in ``process_loss``.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import torch

from .base import LossComponent
from .channel_weight import ChannelWeightLoss
from .spatial_highpass import SpatialHighPassLoss
from .sobel_gradient import SobelGradientLoss
from .ms_ssim import MSSSIMLoss
from .census import CensusLoss
from .zero_mean import ZeroMeanLoss

# ---------------------------------------------------------------------------
# Component registry
# ---------------------------------------------------------------------------

_COMPONENT_REGISTRY: Dict[str, Any] = {
    "ch_weight": ChannelWeightLoss,
    "sp_hpf": SpatialHighPassLoss,
    "sobel": SobelGradientLoss,
    "ms_ssim": MSSSIMLoss,
    "census": CensusLoss,
    "zero_mean": ZeroMeanLoss,
}


def register_component(name: str, cls: type):
    """Register a custom component so it can be referenced in config strings."""
    _COMPONENT_REGISTRY[name] = cls


# ---------------------------------------------------------------------------
# Config-string parser
# ---------------------------------------------------------------------------

# Regex:  component_name(key=val;key=val) + coefficient
# e.g.  ch_weight(path=./w.pt;base=MSE) * 0.8 + sp_hpf(strength=2.0)
_COMPONENT_RE = re.compile(
    r"(?P<name>\w+)"                     # component name
    r"\("                                # open paren
    r"(?P<params>[^)]*)"                 # key=val;key=val...
    r"\)"                                # close paren
    r"(?:\s*\*\s*(?P<coeff>[\d.]+))?"   # optional * coefficient
)


def _parse_kv(kv_string: str) -> Dict[str, Any]:
    """Parse ``key1=val1;key2=val2`` into a dict, inferring types."""
    parts = kv_string.split(";")
    out: Dict[str, Any] = {}
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if "=" not in part:
            continue
        k, v = part.split("=", 1)
        k = k.strip()
        v = v.strip()
        # Try numeric types
        try:
            if "." in v:
                out[k] = float(v)
            else:
                out[k] = int(v)
        except (ValueError, TypeError):
            out[k] = v  # keep as string
    return out


def parse_component_specs(spec: str) -> List[Tuple[str, dict, float]]:
    """Parse a composite-loss config string.

    Returns a list of ``(component_name, kwargs_dict, coefficient)`` tuples.
    """
    results: List[Tuple[str, dict, float]] = []
    # Split on '+' (trimming whitespace)
    tokens = re.split(r"\s*\+\s*", spec.strip())
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        m = _COMPONENT_RE.match(token)
        if not m:
            print(f"WARNING: cannot parse loss component spec: {token!r}  — skipping")
            continue
        name = m.group("name")
        params_str = m.group("params")
        coeff_str = m.group("coeff")
        coeff = float(coeff_str) if coeff_str else 1.0

        kwargs = _parse_kv(params_str)
        results.append((name, kwargs, coeff))
    return results


# ---------------------------------------------------------------------------
# Composite loss
# ---------------------------------------------------------------------------


class CompositeLoss:
    """Weighted sum of multiple :class:`LossComponent` instances.

    Args:
        components: List of ``(LossComponent, coefficient)`` pairs.
    """

    def __init__(
        self,
        components: List[Tuple[LossComponent, float]],
    ):
        self._components = components

    @property
    def is_noop(self) -> bool:
        """True when there are no components — falls back to legacy path."""
        return len(self._components) == 0

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        **context: Any,
    ) -> torch.Tensor:
        """Compute element-wise composite loss ``[B, C, H, W]``."""
        total: Optional[torch.Tensor] = None

        for comp, coeff in self._components:
            elem_loss = comp(pred, target, **context)  # [B, C, H, W]
            if coeff != 1.0:
                elem_loss = elem_loss * coeff
            total = elem_loss if total is None else total + elem_loss

        if total is None:
            # Fallback: plain MSE (shouldn't happen in practice)
            return torch.nn.functional.mse_loss(
                pred.float(), target.float(), reduction="none"
            )
        return total

    def to(self, device: torch.device):
        """Move all component buffers to *device*."""
        for comp, _ in self._components:
            if hasattr(comp, "to"):
                comp.to(device)
        return self


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_composite_loss(
    spec: str,
    device: Optional[torch.device] = None,
    latent_channels: int = 16,
) -> CompositeLoss:
    """Build a :class:`CompositeLoss` from a config string.

    Args:
        spec: Config string, e.g. ``"ch_weight(path=./w.pt;base=MSE) * 0.8"``.
        device: Target device for component buffers/weights.
        latent_channels: Number of VAE latent channels (default 16 for Anima).

    Returns:
        A :class:`CompositeLoss` instance (which is a no-op if *spec* is empty).
    """
    if not spec or not spec.strip():
        return CompositeLoss([])

    parsed = parse_component_specs(spec)
    components: List[Tuple[LossComponent, float]] = []

    for name, kwargs, coeff in parsed:
        cls = _COMPONENT_REGISTRY.get(name)
        if cls is None:
            print(
                f"WARNING: unknown loss component {name!r}.  "
                f"Available: {list(_COMPONENT_REGISTRY)}  — skipping"
            )
            continue

        # Provide latent_channels to any factory that needs it
        if "latent_channels" not in kwargs:
            kwargs["latent_channels"] = latent_channels

        # Use from_config if available, else direct constructor
        if hasattr(cls, "from_config") and callable(getattr(cls, "from_config")):
            instance = cls.from_config(**kwargs)
        else:
            instance = cls(**kwargs)

        components.append((instance, coeff))

    composite = CompositeLoss(components)
    if device is not None:
        composite.to(device)
    return composite
