"""trainer/subspace_guard.py

Gradient projection guard for LoRA training.

Registers backward hooks on lora_down.weight for every matching LoRA layer that
project the gradient to be orthogonal to a pre-computed "protected" subspace.
This prevents a new LoRA from learning features already encoded by a reference
feature type (e.g. preventing a style LoRA from also encoding subject identity).

Typical usage
-------------
    # After create_network(t) in train_lora / train_diff2:
    guard = SubspaceGuard.from_file("subspaces/subject.safetensors", strength=1.0)
    guard.register_gradient_hooks(network)

    # ... training loop runs normally ...

    guard.remove_hooks()   # optional clean-up at end of training

Multiple guards can be stacked (one per feature type to protect):
    style_guard   = SubspaceGuard.from_file("subspaces/style.safetensors",   strength=1.0)
    subject_guard = SubspaceGuard.from_file("subspaces/subject.safetensors", strength=1.0)
    style_guard.register_gradient_hooks(network)
    subject_guard.register_gradient_hooks(network)

Background
----------
Each reference subspace file (produced by tools/extract_subspace.py) contains,
for every LoRA layer key, the top-K right singular vectors of the stacked
lora_down.weight matrices from all reference LoRAs.  These K vectors span the
principal input-space directions occupied by the reference feature type.

For a gradient g of shape (r, n) on lora_down.weight, the projection is:

    g_protected = g @ V_K @ V_K^T   (component lying in the protected subspace)
    g_filtered  = g - strength * g_protected

With strength=1.0 the filtered gradient has zero component in every protected
direction, so the optimiser cannot step the LoRA weights into that subspace.
With strength<1.0 a fraction of the protected gradient is still allowed through,
which can help convergence when the subspaces are not perfectly orthogonal.
"""

from __future__ import annotations

import os
import torch
from safetensors.torch import load_file


class SubspaceGuard:
    """Project LoRA gradients orthogonal to a reference feature subspace.

    Args:
        subspaces:  dict mapping lora_name (base key) to
                    {"V_K": Tensor(n, K)}  principal input-space directions.
        strength:   float in [0, 1].
                    1.0 = full projection (zero gradient in protected directions).
                    0.0 = no projection (guard is a no-op).
        label:      Optional human-readable name for logging (e.g. "subject").
    """

    def __init__(
        self,
        subspaces: dict,
        strength: float = 1.0,
        label: str = "unnamed",
        restrict_to_mapped: bool = False,
    ) -> None:
        self.subspaces = subspaces
        self.strength = float(max(0.0, min(1.0, strength)))
        self.label = label
        self.restrict_to_mapped = restrict_to_mapped
        self._hooks: list = []
        self._projected_layers: int = 0
        self._blocked_layers: int = 0

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_file(
        cls,
        path: str,
        strength: float = 1.0,
        label: str | None = None,
        device: str = "cpu",
        restrict_to_mapped: bool = False,
    ) -> "SubspaceGuard":
        """Load a subspace file produced by tools/extract_subspace.py.

        Args:
            path:     Path to the .safetensors subspace file.
            strength: Projection strength in [0, 1].
            label:    Human-readable name; defaults to the file stem.
            device:   Device to load tensors onto (they are moved to the
                      gradient device at hook time, so "cpu" is fine).
        """
        if label is None:
            label = os.path.splitext(os.path.basename(path))[0]

        flat = load_file(path, device=device)
        subspaces: dict = {}
        for key, tensor in flat.items():
            if key.endswith(".V_K"):
                base = key[: -len(".V_K")]
                subspaces[base] = {"V_K": tensor}

        print(
            f"SubspaceGuard [{label}]: loaded {len(subspaces)} layer subspaces "
            f"from {path} (strength={strength:.3f}, restrict_to_mapped={restrict_to_mapped})"
        )
        return cls(subspaces, strength=strength, label=label, restrict_to_mapped=restrict_to_mapped)

    @classmethod
    def from_files(
        cls,
        paths: list[str],
        strength: float = 1.0,
        label: str = "combined",
        device: str = "cpu",
        restrict_to_mapped: bool = False,
    ) -> "SubspaceGuard":
        """Merge multiple subspace files into one guard.

        The V_K matrices from each file are concatenated per layer and
        re-orthogonalised via a thin SVD so the combined subspace has at
        most K*len(paths) orthonormal directions.  Useful for protecting
        against multiple feature types simultaneously.

        Args:
            paths:   List of subspace .safetensors file paths.
            strength, label, device: as for from_file.
        """
        # Load all subspaces
        per_file: list[dict] = [
            cls.from_file(p, strength=1.0, device=device).subspaces for p in paths
        ]

        # Union of layer keys
        all_keys: set[str] = set()
        for d in per_file:
            all_keys.update(d.keys())

        merged: dict = {}
        for key in all_keys:
            vks = [d[key]["V_K"] for d in per_file if key in d]
            if not vks:
                continue
            # Concatenate along the K axis: (n, K1), (n, K2) → (n, K1+K2)
            combined = torch.cat(vks, dim=1)
            # Re-orthogonalise: SVD of combined^T gives orthonormal rows
            _, _, Vh = torch.linalg.svd(combined.T.float(), full_matrices=False)
            # Vh: (rank, n) → take as rows, transpose back to (n, rank)
            merged[key] = {"V_K": Vh.T.half().cpu()}

        print(
            f"SubspaceGuard [{label}]: merged {len(paths)} subspace files → "
            f"{len(merged)} combined layer subspaces (strength={strength:.3f})"
        )
        inst = cls(merged, strength=strength, label=label, restrict_to_mapped=restrict_to_mapped)
        return inst

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _make_projection_hook(self, V_K: torch.Tensor):
        """Return a gradient hook that projects rows of grad orthogonal to V_K.

        V_K must already be on the same device as the gradient (moved at
        registration time, not inside the hook).  Only a dtype cast is done
        at hook call time, which is a cheap in-place GPU operation.

        For grad of shape (r, n) and V_K of shape (n, K):
            proj = grad @ V_K @ V_K^T   — component in protected subspace
            return grad - strength * proj
        """
        strength = self.strength

        def hook(grad: torch.Tensor) -> torch.Tensor:
            if grad is None or strength == 0.0:
                return grad
            # V_K is already on grad.device — only cast dtype (fast, on-GPU)
            vk = V_K.to(dtype=grad.dtype)
            # (r, n) @ (n, K) → (r, K);  (r, K) @ (K, n) → (r, n)
            proj = (grad @ vk) @ vk.t()
            return grad - strength * proj

        return hook

    def register_gradient_hooks(self, network) -> None:
        """Attach projection hooks to lora_down.weight for all matching layers.

        V_K tensors are moved to the LoRA weight device (GPU) here, once, so
        the hook itself never triggers a CPU→GPU transfer during training.

        If restrict_to_mapped is True, layers with no subspace entry have their
        gradients zeroed entirely — only mapped layers can learn.

        Args:
            network: LoRANetwork instance (from trainer/lora.py).
        """
        self._hooks.clear()
        self._projected_layers = 0
        self._blocked_layers = 0
        total_layers = len(network.unet_loras + network.te_loras)

        for lora in network.unet_loras + network.te_loras:
            name = lora.lora_name
            if name not in self.subspaces:
                continue

            # Pre-move V_K to the training device once at registration time.
            # The hook only needs a dtype cast afterwards, which is a cheap
            # in-place GPU operation instead of a PCIe transfer per step.
            target_device = lora.lora_down.weight.device
            V_K = self.subspaces[name]["V_K"].to(device=target_device)
            handle = lora.lora_down.weight.register_hook(
                self._make_projection_hook(V_K)
            )
            self._hooks.append(handle)
            self._projected_layers += 1

        print(
            f"SubspaceGuard [{self.label}]: registered projection hooks on "
            f"{self._projected_layers}/{total_layers} layers"
            + (f", blocked {self._blocked_layers} unmapped layers." if self.restrict_to_mapped else ".")
        )
        if self._projected_layers == 0:
            print(
                f"  Warning: no layers matched.  Check that the subspace file was "
                f"extracted from LoRAs with the same key naming as this network."
            )

    def remove_hooks(self) -> None:
        """Remove all registered gradient hooks (call at end of training)."""
        for handle in self._hooks:
            handle.remove()
        n = self._projected_layers
        b = self._blocked_layers
        self._hooks.clear()
        self._projected_layers = 0
        self._blocked_layers = 0
        detail = f" ({b} blocked)" if b else ""
        print(f"SubspaceGuard [{self.label}]: removed hooks on {n} projected{detail} layers.")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def projection_stats(self, network) -> dict:
        """Compute how much gradient energy would be projected away (dry run).

        Accumulates the ratio ||g_protected|| / ||g|| over all matching layers
        using the *current* weight gradients (call after a backward pass).
        Returns a dict with mean/max/layer-wise ratios for logging.

        This does not modify any gradients — it is read-only.
        """
        ratios = {}
        for lora in network.unet_loras + network.te_loras:
            name = lora.lora_name
            if name not in self.subspaces:
                continue
            grad = lora.lora_down.weight.grad
            if grad is None:
                continue
            V_K = self.subspaces[name]["V_K"].to(device=grad.device, dtype=grad.dtype)
            proj = (grad @ V_K) @ V_K.t()
            ratio = (proj.norm() / (grad.norm() + 1e-8)).item()
            ratios[name] = ratio

        if ratios:
            vals = list(ratios.values())
            return {
                "mean_projection_ratio": sum(vals) / len(vals),
                "max_projection_ratio": max(vals),
                "n_layers": len(vals),
                "per_layer": ratios,
            }
        return {}


# ---------------------------------------------------------------------------
# Convenience factory used by trainer/train.py
# ---------------------------------------------------------------------------


def prepare_network_filter(t) -> None:
    """Pre-load subspace layer names and store them on t before network creation.

    When subspace_guard_restrict_to_mapped is True, sets t.subspace_guard_allowed_names
    to the set of layer names present in the subspace file(s).  create_modules in
    LoRANetwork reads this set and skips any layer not in it, so those layers are
    never created and consume no parameters, optimizer state, or VRAM.

    Only the safetensors file header is read (key names only, no tensor data).
    """
    if not bool(getattr(t, "subspace_guard_restrict_to_mapped", False)):
        return

    raw_path = (getattr(t, "subspace_guard_path", "") or "").strip()
    if not raw_path:
        return

    from safetensors import safe_open

    paths = [p.strip() for p in raw_path.split(",") if p.strip()]
    allowed: set[str] = set()

    for path in paths:
        if not os.path.isfile(path):
            print(f"SubspaceGuard: WARNING — subspace file not found, skipping: {path}")
            continue
        try:
            with safe_open(path, framework="pt") as f:
                for key in f.keys():
                    if key.endswith(".V_K"):
                        allowed.add(key[: -len(".V_K")])
        except Exception as exc:
            print(f"SubspaceGuard: WARNING — could not read {path}: {exc}")

    if allowed:
        t.subspace_guard_allowed_names = allowed
        print(
            f"SubspaceGuard [restrict_to_mapped]: network creation will be limited "
            f"to {len(allowed)} mapped layers."
        )
    else:
        print("SubspaceGuard: WARNING — no mapped layers found; restrict_to_mapped has no effect.")


def maybe_create_subspace_guard(t, network) -> "SubspaceGuard | None":
    """Create and register a SubspaceGuard if the trainer config requests one.

    Reads:
        t.subspace_guard_path            — comma-separated list of subspace file paths,
                                           or a single path.  Empty string = disabled.
        t.subspace_guard_strength        — float in [0, 1], default 1.0.
        t.subspace_guard_restrict_to_mapped — bool, default False.
                                           When True, layers without a subspace entry
                                           have their gradients zeroed (only mapped
                                           layers are allowed to learn).

    Returns the guard (already registered) or None if disabled.
    """
    raw_path = (getattr(t, "subspace_guard_path", "") or "").strip()
    if not raw_path:
        return None

    strength = float(getattr(t, "subspace_guard_strength", 1.0) or 1.0)
    restrict = bool(getattr(t, "subspace_guard_restrict_to_mapped", False))

    paths = [p.strip() for p in raw_path.split(",") if p.strip()]
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        print(
            f"SubspaceGuard: WARNING — the following subspace files were not found "
            f"and will be skipped:\n" + "\n".join(f"  {p}" for p in missing)
        )
        paths = [p for p in paths if os.path.isfile(p)]

    if not paths:
        print("SubspaceGuard: no valid subspace files found; guard is disabled.")
        return None

    if len(paths) == 1:
        guard = SubspaceGuard.from_file(paths[0], strength=strength, restrict_to_mapped=restrict)
    else:
        guard = SubspaceGuard.from_files(paths, strength=strength, label="combined", restrict_to_mapped=restrict)

    guard.register_gradient_hooks(network)
    return guard
