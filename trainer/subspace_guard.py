"""trainer/subspace_guard.py

Gradient projection guard for LoRA training.

Each entry in the guard config maps a subspace file to one of two modes:

  exclude  (default)
      Projects the gradient *orthogonal* to the specified subspace.
      The LoRA cannot learn directions already encoded by the reference feature.
      Example: exclude a style subspace so the new LoRA only captures
      subject identity, not style.

  include
      Projects the gradient *into* the specified subspace.
      The LoRA can only learn within this subspace.
      Example: include a concept subspace to constrain a fine-tune to a
      pre-defined direction.

When both modes are present for a layer, include is applied first (constraining
the gradient to the include space), then exclude removes specific sub-directions
from within that constrained space.

Multiple files of the same mode are merged per-layer: their V_K matrices are
concatenated along the K axis and re-orthogonalised via a thin SVD so the
combined subspace is a proper orthonormal basis.

Config format (subspace_guard_path field, one entry per line)
-------------------------------------------------------------
    exclude: subspaces/style.safetensors
    exclude: subspaces/expressions.safetensors:0.75
    include: subspaces/my_concept.safetensors

    # bare path → exclude, strength from global setting
    subspaces/style.safetensors

Comma-separated bare paths (old format) are still accepted for backward
compatibility:
    subspaces/style.safetensors,subspaces/subject.safetensors

Per-entry strength override (append :float after the path):
    exclude: subspaces/soft_style.safetensors:0.5

Background
----------
Each subspace file (produced by tools/extract_subspace.py or
tools/extract_subspace_from_model.py) stores, for every LoRA layer key, the
top-K right singular vectors of the stacked weight/gradient matrices.  These K
vectors span the principal input-space directions of the reference feature type.

For grad of shape (r, n) and V_K of shape (n, K):

  Include projection (restrict to subspace):
      g = (g @ V_K) @ V_K^T

  Exclude projection (remove subspace component):
      g = g - strength * (g @ V_K) @ V_K^T
"""

from __future__ import annotations

import os
import re
import torch
from safetensors.torch import load_file


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_vk_from_file(path: str, device: str = "cpu") -> dict[str, torch.Tensor]:
    """Load {lora_name: V_K Tensor(n, K)} from a subspace .safetensors file."""
    flat = load_file(path, device=device)
    result: dict = {}
    for key, tensor in flat.items():
        if key.endswith(".V_K"):
            result[key[: -len(".V_K")]] = tensor
    return result


def _merge_vk_list(vk_list: list[torch.Tensor]) -> torch.Tensor:
    """Concatenate a list of V_K tensors and re-orthogonalise.

    All tensors must have the same n (input dimension).
    Returns an orthonormal (n, rank) basis for the union of the subspaces.
    """
    if len(vk_list) == 1:
        return vk_list[0]
    combined = torch.cat(vk_list, dim=1).float()  # (n, K_total)
    _, _, Vh = torch.linalg.svd(combined.T, full_matrices=False)
    # Vh: (rank, n) → transpose to (n, rank)
    return Vh.T.half().cpu().contiguous()


def _parse_guard_entries(raw: str) -> list[tuple[str, str, float | None]]:
    """Parse the subspace_guard_path field into a list of (path, mode, strength|None).

    Supported formats (one entry per line, or comma-separated for old compat):

        exclude: subspaces/style.safetensors
        include: subspaces/concept.safetensors
        subspaces/style.safetensors          # bare path → exclude
        exclude: subspaces/soft.safetensors:0.5  # per-entry strength override

    Blank lines and # comments are ignored.
    """
    entries: list = []
    # Support both newline and comma separators
    for line in re.split(r"[,\n]", raw):
        line = line.strip()
        if not line or line.startswith("#"):
            continue

        # Detect mode prefix: "exclude: ..." or "include: ..."
        m = re.match(r"^(exclude|include)\s*:\s*(.+)$", line, re.IGNORECASE)
        if m:
            mode = m.group(1).lower()
            rest = m.group(2).strip()
        else:
            mode = "exclude"
            rest = line

        # Detect optional per-entry strength suffix: "path.safetensors:0.75"
        # But be careful not to split Windows drive letters (C:\...)
        strength: float | None = None
        strength_m = re.search(r":([01](?:\.\d+)?)$", rest)
        if strength_m:
            try:
                strength = float(strength_m.group(1))
                rest = rest[: strength_m.start()].strip()
            except ValueError:
                pass

        if rest:
            entries.append((rest, mode, strength))

    return entries


# ---------------------------------------------------------------------------
# Main guard class
# ---------------------------------------------------------------------------

class SubspaceGuard:
    """Gradient projection guard supporting include and exclude subspace modes.

    Args:
        subspaces:  dict mapping lora_name (base key) to
                    {"V_exc": Tensor(n, K)|None,
                     "V_inc": Tensor(n, K)|None,
                     "strength_exc": float (per-layer override, or None → use self.strength)}
        strength:   Global exclude strength in [0, 1].
                    1.0 = full projection.  0.0 = no projection.
        label:      Human-readable name for logging.
        restrict_to_mapped:
                    When True, layers absent from the subspace dict were excluded
                    at network creation time and will not appear in the network.
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

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_entries(
        cls,
        entries: list[tuple[str, str, float | None]],
        strength: float = 1.0,
        label: str = "combined",
        device: str = "cpu",
        restrict_to_mapped: bool = False,
    ) -> "SubspaceGuard":
        """Build a guard from a list of (path, mode, per_entry_strength|None) tuples.

        Files with mode "exclude" are merged into a per-layer exclude subspace.
        Files with mode "include" are merged into a per-layer include subspace.
        """
        exc_vks: dict[str, list[torch.Tensor]] = {}  # name → [V_K, ...]
        inc_vks: dict[str, list[torch.Tensor]] = {}
        exc_strength_override: dict[str, float] = {}  # name → per-entry strength

        n_exc = sum(1 for _, m, _ in entries if m == "exclude")
        n_inc = sum(1 for _, m, _ in entries if m == "include")

        for path, mode, entry_strength in entries:
            vks = _load_vk_from_file(path, device=device)
            stem = os.path.splitext(os.path.basename(path))[0]
            print(
                f"SubspaceGuard [{label}]: [{mode}] {stem} — "
                f"{len(vks)} layers"
                + (f", strength={entry_strength:.3f}" if entry_strength is not None else "")
            )
            if mode == "exclude":
                for name, vk in vks.items():
                    exc_vks.setdefault(name, []).append(vk)
                    # Per-entry strength override: use the last seen if multiple
                    # files contribute to the same layer
                    if entry_strength is not None:
                        exc_strength_override[name] = entry_strength
            else:
                for name, vk in vks.items():
                    inc_vks.setdefault(name, []).append(vk)

        all_names: set[str] = set(exc_vks) | set(inc_vks)
        subspaces: dict = {}
        for name in all_names:
            V_exc = _merge_vk_list(exc_vks[name]) if name in exc_vks else None
            V_inc = _merge_vk_list(inc_vks[name]) if name in inc_vks else None
            subspaces[name] = {
                "V_exc": V_exc,
                "V_inc": V_inc,
                "strength_exc": exc_strength_override.get(name),  # None → use global
            }

        mode_summary = []
        if n_exc:
            mode_summary.append(f"{n_exc} exclude")
        if n_inc:
            mode_summary.append(f"{n_inc} include")
        print(
            f"SubspaceGuard [{label}]: "
            f"{' + '.join(mode_summary)} files → {len(subspaces)} layer entries "
            f"(global strength={strength:.3f}, restrict_to_mapped={restrict_to_mapped})"
        )
        return cls(subspaces, strength=strength, label=label, restrict_to_mapped=restrict_to_mapped)

    @classmethod
    def from_file(
        cls,
        path: str,
        strength: float = 1.0,
        label: str | None = None,
        device: str = "cpu",
        restrict_to_mapped: bool = False,
    ) -> "SubspaceGuard":
        """Load a single subspace file as an exclude guard (backward-compatible)."""
        if label is None:
            label = os.path.splitext(os.path.basename(path))[0]
        return cls.from_entries(
            [(path, "exclude", None)],
            strength=strength, label=label, device=device,
            restrict_to_mapped=restrict_to_mapped,
        )

    @classmethod
    def from_files(
        cls,
        paths: list[str],
        strength: float = 1.0,
        label: str = "combined",
        device: str = "cpu",
        restrict_to_mapped: bool = False,
    ) -> "SubspaceGuard":
        """Merge multiple subspace files as exclude guards (backward-compatible)."""
        return cls.from_entries(
            [(p, "exclude", None) for p in paths],
            strength=strength, label=label, device=device,
            restrict_to_mapped=restrict_to_mapped,
        )

    # ------------------------------------------------------------------
    # Hook management
    # ------------------------------------------------------------------

    def _make_projection_hook(
        self,
        V_exc: torch.Tensor | None,
        V_inc: torch.Tensor | None,
        strength_exc: float | None,
    ):
        """Return a backward hook applying include then exclude projections.

        Both tensors are already on the gradient device (moved at registration
        time).  Only a dtype cast happens inside the hook — no device transfers.

        Projection order:
            1. Include (if any):  g = (g @ V_inc) @ V_inc^T
               Restricts gradient to the include subspace.
            2. Exclude (if any):  g = g - strength * (g @ V_exc) @ V_exc^T
               Removes the excluded directions from the (possibly restricted) gradient.
        """
        global_strength = self.strength
        eff_strength = global_strength if strength_exc is None else float(strength_exc)

        def hook(grad: torch.Tensor) -> torch.Tensor:
            if grad is None:
                return grad
            g = grad
            if V_inc is not None:
                vk_inc = V_inc.to(dtype=g.dtype)
                g = (g @ vk_inc) @ vk_inc.t()
            if V_exc is not None and eff_strength > 0.0:
                vk_exc = V_exc.to(dtype=g.dtype)
                proj = (g @ vk_exc) @ vk_exc.t()
                g = g - eff_strength * proj
            return g

        return hook

    def register_gradient_hooks(self, network) -> None:
        """Attach projection hooks to lora_down.weight for all matching layers.

        V_K tensors are moved to the LoRA weight device (GPU) once here.
        The hook itself only performs a cheap dtype cast per step.
        """
        self._hooks.clear()
        self._projected_layers = 0
        total_layers = len(network.unet_loras + network.te_loras)

        for lora in network.unet_loras + network.te_loras:
            name = lora.lora_name
            if name not in self.subspaces:
                continue

            entry = self.subspaces[name]
            target_device = lora.lora_down.weight.device

            V_exc = entry["V_exc"].to(device=target_device) if entry["V_exc"] is not None else None
            V_inc = entry["V_inc"].to(device=target_device) if entry["V_inc"] is not None else None
            strength_exc = entry.get("strength_exc")

            handle = lora.lora_down.weight.register_hook(
                self._make_projection_hook(V_exc, V_inc, strength_exc)
            )
            self._hooks.append(handle)
            self._projected_layers += 1

        print(
            f"SubspaceGuard [{self.label}]: registered hooks on "
            f"{self._projected_layers}/{total_layers} layers."
        )
        if self._projected_layers == 0:
            print(
                "  Warning: no layers matched. Check that the subspace file was "
                "extracted with the same key naming as this network."
            )

    def remove_hooks(self) -> None:
        """Remove all registered gradient hooks (call at end of training)."""
        for handle in self._hooks:
            handle.remove()
        n = self._projected_layers
        self._hooks.clear()
        self._projected_layers = 0
        print(f"SubspaceGuard [{self.label}]: removed hooks on {n} layers.")

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def projection_stats(self, network) -> dict:
        """Compute gradient energy fractions (dry run, read-only).

        Returns per-mode ratios for logging after a backward pass.
        """
        exc_ratios: dict = {}
        inc_ratios: dict = {}

        for lora in network.unet_loras + network.te_loras:
            name = lora.lora_name
            if name not in self.subspaces:
                continue
            grad = lora.lora_down.weight.grad
            if grad is None:
                continue
            entry = self.subspaces[name]
            gnorm = grad.norm() + 1e-8

            if entry["V_exc"] is not None:
                vk = entry["V_exc"].to(device=grad.device, dtype=grad.dtype)
                proj = (grad @ vk) @ vk.t()
                exc_ratios[name] = (proj.norm() / gnorm).item()

            if entry["V_inc"] is not None:
                vk = entry["V_inc"].to(device=grad.device, dtype=grad.dtype)
                proj = (grad @ vk) @ vk.t()
                inc_ratios[name] = (proj.norm() / gnorm).item()

        result: dict = {}
        if exc_ratios:
            vals = list(exc_ratios.values())
            result["exclude_mean_ratio"] = sum(vals) / len(vals)
            result["exclude_max_ratio"] = max(vals)
        if inc_ratios:
            vals = list(inc_ratios.values())
            result["include_mean_ratio"] = sum(vals) / len(vals)
            result["include_max_ratio"] = max(vals)
        result["n_layers"] = len(set(exc_ratios) | set(inc_ratios))
        return result


# ---------------------------------------------------------------------------
# Convenience factories used by trainer/train.py
# ---------------------------------------------------------------------------


def prepare_network_filter(t) -> None:
    """Pre-load subspace layer names and store them on t before network creation.

    When subspace_guard_restrict_to_mapped is True, sets
    t.subspace_guard_allowed_names to the union of all layer names across all
    subspace files listed in subspace_guard_path (regardless of mode).
    Layers absent from this set are never created — no parameters, no optimizer
    state, no VRAM.

    Only the safetensors file headers are read (key names only, no tensors).
    """
    if not bool(getattr(t, "subspace_guard_restrict_to_mapped", False)):
        return

    raw_path = (getattr(t, "subspace_guard_path", "") or "").strip()
    if not raw_path:
        return

    from safetensors import safe_open

    entries = _parse_guard_entries(raw_path)
    allowed: set[str] = set()

    for path, _mode, _strength in entries:
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
            f"SubspaceGuard [restrict_to_mapped]: network creation limited "
            f"to {len(allowed)} mapped layers."
        )
    else:
        print("SubspaceGuard: WARNING — no mapped layers found; restrict_to_mapped has no effect.")


def maybe_create_subspace_guard(t, network) -> "SubspaceGuard | None":
    """Create and register a SubspaceGuard if the trainer config requests one.

    Reads:
        t.subspace_guard_path            — multi-line config (see module docstring),
                                           or legacy comma-separated paths (all exclude).
        t.subspace_guard_strength        — global float in [0, 1], default 1.0.
        t.subspace_guard_restrict_to_mapped — bool, default False.

    Returns the guard (already registered) or None if disabled.
    """
    raw_path = (getattr(t, "subspace_guard_path", "") or "").strip()
    if not raw_path:
        return None

    strength = float(getattr(t, "subspace_guard_strength", 1.0) or 1.0)
    restrict = bool(getattr(t, "subspace_guard_restrict_to_mapped", False))

    entries = _parse_guard_entries(raw_path)
    missing = [(p, m, s) for p, m, s in entries if not os.path.isfile(p)]
    if missing:
        print(
            "SubspaceGuard: WARNING — the following files were not found and will be skipped:\n"
            + "\n".join(f"  [{m}] {p}" for p, m, _ in missing)
        )
        entries = [(p, m, s) for p, m, s in entries if os.path.isfile(p)]

    if not entries:
        print("SubspaceGuard: no valid subspace files found; guard is disabled.")
        return None

    paths_for_label = [os.path.splitext(os.path.basename(p))[0] for p, _, _ in entries]
    label = "+".join(paths_for_label) if len(paths_for_label) <= 3 else f"{len(paths_for_label)}_files"

    guard = SubspaceGuard.from_entries(
        entries, strength=strength, label=label, restrict_to_mapped=restrict
    )
    guard.register_gradient_hooks(network)
    return guard
