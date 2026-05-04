"""
FD-Loss: Representation Fréchet Distance loss for Anima training.

Port of the core FD-Loss components (FeatureQueue, differentiable FID,
frozen feature extractors) adapted for the Anima traintrain pipeline.

Reference: https://github.com/sony/fd-loss  (Sony Research)

Extensions:
  - Diversity-aware eviction (evict redundant features, keep diverse ones)
  - Feature clustering & visualisation for inter-epoch inspection
  - Interactive guidance: steer queue composition toward artist-desired regions
"""

import hashlib
import logging
from typing import Optional, Literal, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("FD_loss")

# Shared ImageNet normalization constants
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# =============================================================================
# Differentiable FID
# =============================================================================

def precompute_sigma_ref_sqrt(sigma_ref: torch.Tensor) -> Optional[torch.Tensor]:
    """Precompute sigma_ref^{1/2} via eigendecomposition (one-time cost).

    Returns None if the eigendecomposition fails (ill-conditioned matrix),
    e.g. early in training when the queue has very few samples.
    """
    try:
        eigvals, eigvecs = torch.linalg.eigh(sigma_ref)
        eigvals = torch.clamp(eigvals, min=0)
        return eigvecs @ torch.diag(eigvals.sqrt()) @ eigvecs.T
    except torch._C._LinAlgError as e:
        logger.warning(f"[precompute_sigma_ref_sqrt] Eigendecomposition failed ({e}) — returning None")
        return None


def _compute_trace_term(
    sigma: torch.Tensor,
    sigma_ref: torch.Tensor,
    sigma_ref_sqrt: Optional[torch.Tensor] = None,
) -> Optional[torch.Tensor]:
    """Compute tr(sigma) + tr(sigma_ref) - 2*tr(sqrtm(sigma @ sigma_ref)).

    Returns None on numerical failure (NaN/Inf in product or eigendecomposition
    convergence failure).
    """
    try:
        if sigma_ref_sqrt is not None:
            # eigvalsh on symmetric product: exact and ~8x faster
            M = sigma_ref_sqrt @ sigma @ sigma_ref_sqrt
            M = 0.5 * (M + M.T)
            evals = torch.linalg.eigvalsh(M)
            evals = torch.clamp(evals, min=0)
            tr_covmean = torch.sum(torch.sqrt(evals))
        else:
            product = sigma @ sigma_ref
            if not torch.isfinite(product).all():
                return None
            eigvals = torch.linalg.eigvals(product).real
            eigvals = torch.clamp(eigvals, min=0)
            tr_covmean = torch.sum(torch.sqrt(eigvals))
    except torch._C._LinAlgError as e:
        logger.warning(f"[_compute_trace_term] Eigendecomposition failed ({e}) — returning None")
        return None

    return torch.diagonal(sigma).sum() + torch.diagonal(sigma_ref).sum() - 2.0 * tr_covmean


def compute_frechet_distance_loss(
    mu_ref: torch.Tensor,
    sigma_ref: torch.Tensor,
    all_feats: Optional[torch.Tensor] = None,
    mu: Optional[torch.Tensor] = None,
    sigma: Optional[torch.Tensor] = None,
    sigma_ref_sqrt: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Differentiable FID from raw features or pre-computed (mu, sigma) statistics.

    Provide either ``all_feats`` (raw feature matrix) or both ``mu`` and ``sigma``.
    When ``all_feats`` is given, mu/sigma are computed internally (requires >= 2 samples).
    """
    if all_feats is not None:
        n_samples = all_feats.shape[0]
        if n_samples < 2:
            logger.warning(f"[compute_frechet_distance_loss] Only {n_samples} sample(s) — need >= 2")
            return torch.tensor(1e6, device=all_feats.device, dtype=torch.float32, requires_grad=True)
        mu = all_feats.mean(dim=0)
        feats_c = all_feats - mu
        sigma = (feats_c.T @ feats_c) / (n_samples - 1)
    elif mu is None or sigma is None:
        raise ValueError("Provide either all_feats or both mu and sigma")

    # Ensure consistent dtype (ref stats may be float64 from numpy)
    compute_dtype = sigma.dtype
    mu_ref = mu_ref.to(dtype=compute_dtype)
    sigma_ref = sigma_ref.to(dtype=compute_dtype)
    if sigma_ref_sqrt is not None:
        sigma_ref_sqrt = sigma_ref_sqrt.to(dtype=compute_dtype)

    diff = mu - mu_ref
    mean_term = diff.dot(diff)

    trace_term = _compute_trace_term(sigma, sigma_ref, sigma_ref_sqrt)
    if trace_term is None:
        device = all_feats.device if all_feats is not None else mu.device
        logger.warning("[compute_frechet_distance_loss] NaN/Inf in covariance product — returning fallback")
        return torch.tensor(1e6, device=device, dtype=torch.float32)

    return (mean_term + trace_term).float()


def load_mu_and_sigma_reference(fid_stats_path: str, pool_type: str = "cls"):
    """Load reference FID statistics as CUDA float64 tensors.

    Args:
        fid_stats_path: .npz with ``mu``/``sigma`` and optionally ``avg_mu``/``avg_sigma``.
        pool_type: ``'cls'`` or ``'avg'``.

    Returns:
        (mu_ref, sigma_ref) tensors, or ``(None, None)`` if the file does not exist.
    """
    import os
    if not os.path.isfile(fid_stats_path):
        logger.warning(
            f"[load_mu_and_sigma_reference] Stats file not found: {fid_stats_path}. "
            "FD-Loss will use self-referential mode (queue stats as reference)."
        )
        return None, None

    import numpy as np
    ref = np.load(fid_stats_path)
    if pool_type == "avg":
        if "avg_mu" not in ref:
            raise KeyError(
                f"pool_type='avg' but {fid_stats_path} has no 'avg_mu'. "
                f"Available: {list(ref.keys())}"
            )
        mu_ref = torch.tensor(ref["avg_mu"], device="cuda", dtype=torch.float64)
        sigma_ref = torch.tensor(ref["avg_sigma"], device="cuda", dtype=torch.float64)
    else:
        mu_ref = torch.tensor(ref["mu"], device="cuda", dtype=torch.float64)
        sigma_ref = torch.tensor(ref["sigma"], device="cuda", dtype=torch.float64)
    return mu_ref, sigma_ref


# =============================================================================
# Feature Queue
# =============================================================================

class FeatureQueue(nn.Module):
    """Circular buffer of features for FID computation.

    Registered as buffers so they survive ``state_dict`` save/load and
    automatically move with ``.to(device)`` / ``.cuda()``.

    Extensions over the original Sony FD-Loss:
      - **Diversity-aware eviction**: when ``eviction_mode != 'fifo'``, the queue
        evicts the feature most similar to its nearest neighbour (i.e. the most
        redundant entry) instead of the oldest.  This keeps the queue diverse.
      - **Feature clustering**: ``get_clusters()`` returns cluster assignments
        and centroids for visualisation / inspection between epochs.
      - **Interactive guidance**: ``guidance_target`` can be set to a feature
        vector or cluster centroid; the queue then preferentially retains
        features near the target and evicts features far from it.
      - **Protected (pinned) features**: indices marked in ``protected_mask``
        are never evicted.  Use ``protect_indices()`` / ``unprotect_indices()``
        to pin specific features (e.g. from a cluster the artist wants to keep).
      - **Source image thumbnails**: ``source_images`` stores a small pixel crop
        alongside each feature so the cluster panel can show what each feature
        "looks like" — bridging the gap between abstract feature vectors and
        visual inspection.
      - **Stable cluster IDs**: ``_cluster_registry`` persists the last cluster
        assignment across re-clustering calls.  When ``get_clusters()`` is called,
        new clusters are matched to old ones via Hungarian assignment on centroid
        similarity, so cluster 3 today ≈ cluster 3 tomorrow.  Cluster-level
        actions (``protect_cluster``, ``evict_cluster``, etc.) use the registry
        rather than re-clustering, guaranteeing that a "protect cluster 3" action
        always targets the same set of queue indices until the next explicit
        re-clustering.
      - **Step counter per index**: ``_step_counter`` tracks how many training
        steps each generated (source_type=1) feature has survived.  Old noisy
        generated images get an auto-eviction hint in the cluster panel.
      - **Eviction rotation**: when features are committed for eviction, the
        pointer is advanced past their slots so they naturally cycle to the
        back of the queue rather than being overwritten in the same position.
    """

    def __init__(
        self,
        size: int = 50000,
        feat_dim: int = 2048,
        online_accum: bool = False,
        ema_beta: float = 0.0,
        eviction_mode: Literal["fifo", "diversity", "guided"] = "fifo",
        guidance_target: Optional[torch.Tensor] = None,
        guidance_strength: float = 0.5,
        n_clusters: int = 20,
        store_source_images: bool = False,
        source_img_size: int = 64,
        # Auto-eviction hint threshold: generated images older than this many
        # enqueue cycles get a visual hint suggesting eviction.
        auto_evict_hint_steps: int = 60,
    ):
        super().__init__()
        self.size = size
        self.feat_dim = feat_dim
        self.online_accum = online_accum
        self.ema_beta = ema_beta
        self.ema_stats = ema_beta > 0.0
        self.eviction_mode = eviction_mode
        self.guidance_strength = guidance_strength
        self.n_clusters = n_clusters
        self.store_source_images = store_source_images
        self.source_img_size = source_img_size
        self.auto_evict_hint_steps = auto_evict_hint_steps

        # Source type: 0 = dataset (real), 1 = machine-generated (model output)
        # Used by the cluster panel to visually distinguish real vs generated features.
        # Registered as a buffer later (when store_source_images=True) so it moves
        # with .to(device).  When store_source_images=False, stays None.
        # NOTE: Do NOT set self._source_type = None here — register_buffer below
        # will create the attribute, and PyTorch rejects register_buffer if the
        # attribute already exists.

        # Step counter per index: tracks how many enqueue cycles each feature
        # has survived.  Reset to 0 when a new feature is enqueued.
        # Only meaningful for generated (source_type=1) features.
        self.register_buffer(
            "_step_counter",
            torch.zeros(size, dtype=torch.long),
        )

        # Guidance target: a feature vector or cluster centroid the queue
        # should preferentially retain features near.
        # NOTE: _guidance_target is registered as a buffer AND assigned as a
        # plain attribute.  The plain attribute shadows the buffer, so after
        # .to(device) the plain attribute goes stale (still on old device).
        # The _get_guided_evict_indices method works around this with an
        # explicit .to(self.feats.device) call.  This is kept for backward
        # compatibility with state_dict keys.
        self._guidance_target: Optional[torch.Tensor] = None
        if guidance_target is not None:
            self.register_buffer("_guidance_target", guidance_target.clone().detach().float())
            self._guidance_target = guidance_target.clone().detach().float()

        # Set to True once the circular buffer has wrapped around at least once.
        # Until then, "evicted" features are uninitialized (torch.empty) and
        # must not be subtracted from the online accumulators.
        self._has_wrapped = True

        # --- Cluster result cache ---
        # Caches the last clustering result so repeated calls to get_clusters()
        # with unchanged queue contents skip the expensive k-means++ computation.
        # Invalidated on enqueue (see enqueue()) or when n_clusters changes.
        self._cluster_cache: Optional[dict] = None
        self._cluster_cache_n_clusters: int = 0
        self._cluster_cache_hash: Optional[bytes] = None

        if self.ema_stats:
            self.register_buffer("mu_ema", torch.zeros(feat_dim, dtype=torch.float64))
            self.register_buffer("m2_ema", torch.zeros(feat_dim, feat_dim, dtype=torch.float64))
            self.register_buffer("_ema_count", torch.zeros(1, dtype=torch.long))
            # _source_type is not used in EMA mode (no feature storage)
            self._source_type = None
        else:
            self.register_buffer("feats", torch.empty(size, feat_dim))
            self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
            # Protected mask: True = never evict
            self.register_buffer("protected_mask", torch.zeros(size, dtype=torch.bool))
            # Pending eviction mask: True = marked for eviction (visual red, not yet removed)
            self.register_buffer("pending_eviction_mask", torch.zeros(size, dtype=torch.bool))
            if online_accum and size > 0:
                self.register_buffer("feat_sum_old", torch.zeros(feat_dim, dtype=torch.float64))
                self.register_buffer("feat_outer_old", torch.zeros(feat_dim, feat_dim, dtype=torch.float64))
            # Source image thumbnails (optional, for visual cluster panel)
            if store_source_images:
                self.register_buffer(
                    "source_images",
                    torch.zeros(size, 3, source_img_size, source_img_size, dtype=torch.uint8),
                )
                # Source type: 0=dataset, 1=machine-generated
                # Registered as _source_type (not source_type) so the plain attribute
                # and the buffer are the same object — this prevents a stale reference
                # after .to(device) moves registered buffers to a different device.
                self.register_buffer(
                    "_source_type",
                    torch.zeros(size, dtype=torch.uint8),
                )
            else:
                # When store_source_images=False, _source_type stays None.
                # Must set it explicitly so that `if self._source_type is not None:`
                # checks work without raising AttributeError.
                self._source_type = None

        # --- Stable cluster registry ---
        # Persists the last cluster assignment so cluster-level actions
        # (protect_cluster, evict_cluster, etc.) are deterministic even
        # though k-means++ is inherently non-deterministic.
        # Shape: [size] long tensor, -1 = unassigned.
        self.register_buffer(
            "_cluster_registry",
            torch.full((size,), -1, dtype=torch.long),
        )
        # Cached centroids from the last clustering run, used for Hungarian
        # matching on the next call.  Shape: [n_clusters, feat_dim].
        self.register_buffer(
            "_last_centroids",
            torch.empty(0, feat_dim),
        )
        # The number of clusters used for the last clustering run.
        self._last_n_clusters: int = 0

    @property
    def pointer(self) -> int:
        return int(self.ptr.item())

    @property
    def guidance_target(self) -> Optional[torch.Tensor]:
        return self._guidance_target

    @guidance_target.setter
    def guidance_target(self, value: Optional[torch.Tensor]):
        if value is not None:
            self._guidance_target = value.clone().detach().float().to(self.feats.device)
        else:
            self._guidance_target = None

    # -- Initialization --------------------------------------------------------

    @torch.no_grad()
    def _init_accumulators(self):
        """Compute feat_sum_old and feat_outer_old from current queue contents."""
        feats_d = self.feats.double()
        self.feat_sum_old.copy_(feats_d.sum(0))
        self.feat_outer_old.copy_(feats_d.T @ feats_d)

    @torch.no_grad()
    def accumulate_batch(self, feats: torch.Tensor):
        """Streaming accumulation for EMA init."""
        feats_d = feats.detach().float().double()
        self.mu_ema.add_(feats_d.sum(0))
        self.m2_ema.addmm_(feats_d.T, feats_d)
        self._ema_count += feats_d.shape[0]

    @torch.no_grad()
    def _finalize_streaming_init(self):
        """Normalize accumulated sums into moments (mu, E[xx^T])."""
        count = self._ema_count.item()
        if count == 0:
            logger.warning("[FeatureQueue] EMA streaming init: no features accumulated")
            return
        self.mu_ema.div_(count)
        self.m2_ema.div_(count)
        logger.info(f"[FeatureQueue] EMA init done: {count} features (beta={self.ema_beta})")

    # -- Statistics (with gradient support) ------------------------------------

    def build_feats_stats(self, new_feats: torch.Tensor):
        """Compute (mu, sigma) with gradients flowing through new_feats.

        Dispatches to EMA mode if enabled; otherwise uses online accumulators.
        """
        if self.ema_stats:
            return self._build_feats_stats_ema(new_feats)

        new_d = new_feats.double()
        B = new_d.shape[0]
        N = self.size

        evicted = self._get_evicted_feats(B).double()
        sum_old = self.feat_sum_old - evicted.sum(0)
        outer_old = self.feat_outer_old - evicted.T @ evicted

        feat_sum = sum_old.detach() + new_d.sum(0)
        feat_outer = outer_old.detach() + new_d.T @ new_d

        mu = feat_sum / N
        sigma = (feat_outer - feat_sum.unsqueeze(1) * feat_sum.unsqueeze(0) / N) / (N - 1)
        return mu, sigma

    def _build_feats_stats_ema(self, new_feats: torch.Tensor):
        """Compute (mu, sigma) via EMA moments blended with new_feats."""
        beta = self.ema_beta
        new_d = new_feats.double()
        B = new_d.shape[0]

        mu = beta * self.mu_ema.detach() + (1.0 - beta) * new_d.mean(0)
        m2 = beta * self.m2_ema.detach() + (1.0 - beta) * (new_d.T @ new_d) / B

        sigma = m2 - mu.unsqueeze(1) * mu.unsqueeze(0)
        return mu, sigma

    # -- Snapshot (autograd through pointer region) ----------------------------

    def _snapshot(self, buf: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
        """Build a snapshot of *buf* with the pointer region replaced by *new*."""
        if self.size == 0:
            return new
        n = new.shape[0]
        snap = buf.clone().detach()
        ptr = self.pointer
        if ptr + n <= self.size:
            snap[ptr: ptr + n] = new
        else:
            first = self.size - ptr
            snap[ptr: self.size] = new[:first]
            snap[: n - first] = new[first:]
        return snap

    def build_feats_snapshot(self, new_feats: torch.Tensor) -> torch.Tensor:
        """Return (size, feat_dim) with the pointer region carrying autograd."""
        return self._snapshot(self.feats, new_feats)

    # -- Enqueue (detached, no grad) -------------------------------------------

    @torch.no_grad()
    def enqueue(self, new_feats: torch.Tensor, source_img: Optional[torch.Tensor] = None,
                source_type: Optional[int] = None):
        """Dequeue entries and enqueue new detached features.

        Eviction strategy depends on ``self.eviction_mode``:

        - ``'fifo'`` (default): oldest entries are overwritten (circular buffer).
        - ``'diversity'``: evict the feature most similar to its nearest neighbour
          (most redundant), keeping the queue maximally diverse.
        - ``'guided'``: evict features farthest from the guidance target, retaining
          features near the desired region of feature space.

        Protected indices (``protected_mask == True``) are **never** evicted.
        If too few unprotected slots remain, falls back to FIFO on unprotected
        indices only.

        When ``source_img`` is provided and ``store_source_images`` is True, the
        corresponding pixel thumbnails are stored alongside the features for the
        visual cluster panel.

        ``source_type`` indicates the origin of the features:
        - ``0`` (default): dataset / real training images
        - ``1``: machine-generated (model output during training)

        No-op when size=0. In EMA mode, updates running moments only.
        """
        if self.size == 0:
            return

        n = new_feats.shape[0]
        new_det = new_feats.detach().float()

        if self.ema_stats:
            beta = self.ema_beta
            new_d = new_det.double()
            self.mu_ema.mul_(beta).add_(new_d.mean(0), alpha=1.0 - beta)
            self.m2_ema.mul_(beta).addmm_(new_d.T, new_d, alpha=(1.0 - beta) / n)
            return

        # --- Determine which indices to evict ---
        evict_indices = self._get_evict_indices(n)

        # --- Update online accumulators (subtract evicted, add new) ---
        if self.online_accum:
            new_d = new_det.double()
            evicted = self.feats[evict_indices].double()
            if self._has_wrapped:
                self.feat_sum_old.add_(new_d.sum(0) - evicted.sum(0))
                self.feat_outer_old.add_(new_d.T @ new_d - evicted.T @ evicted)
            else:
                self.feat_sum_old.add_(new_d.sum(0))
                self.feat_outer_old.add_(new_d.T @ new_d)

        # --- Overwrite evicted slots with new features ---
        self.feats[evict_indices] = new_det

        # --- Reset step counter for newly enqueued features ---
        self._step_counter[evict_indices] = 0

        # --- Increment step counter for ALL non-zero-source-type features ---
        # This tracks how many enqueue cycles each generated feature has survived.
        if self._source_type is not None:
            # Only increment for generated (1) features, not dataset (0) or evicted (2)
            gen_mask = (self._source_type == 1)
            self._step_counter[gen_mask] = self._step_counter[gen_mask] + 1

        # --- Store source image thumbnails if enabled ---
        if self.store_source_images and source_img is not None:
            # Resize source images to the stored thumbnail size
            src = source_img.detach().float()
            if src.shape[-2:] != (self.source_img_size, self.source_img_size):
                src = F.interpolate(
                    src, size=(self.source_img_size, self.source_img_size),
                    mode='area',
                )
            # Clamp to [0, 1] and convert to uint8 for storage
            src_uint8 = (src.clamp(0, 1) * 255).to(torch.uint8)
            self.source_images[evict_indices] = src_uint8
            # Store source type (0=dataset, 1=generated)
            if self._source_type is not None:
                st = torch.tensor(source_type if source_type is not None else 0,
                                  dtype=torch.uint8, device=self.feats.device)
                self._source_type[evict_indices] = st

        # Invalidate cluster cache — queue contents have changed.
        self._cluster_cache = None
        self._cluster_cache_hash = None

        # Advance pointer for FIFO tracking (even in non-FIFO modes, this
        # provides a fallback ordering and tracks queue fill state).
        ptr = self.pointer
        if ptr + n <= self.size:
            self.ptr[0] = (ptr + n) % self.size
        else:
            self.ptr[0] = (ptr + n) % self.size
            self._has_wrapped = True

    # -- Eviction strategies ---------------------------------------------------

    def _get_evict_indices(self, n: int) -> torch.Tensor:
        """Return ``n`` indices to evict, respecting ``protected_mask``.

        Protected indices are never selected by diversity/guided strategies.
        If fewer than ``n`` unprotected indices exist, the surplus falls back
        to FIFO (which **may** include protected indices as a last resort,
        since we must make room for new features).

        **Evicted (noise) slots** (``source_type == 2``) are always prioritised
        for eviction so they get repopulated with real features rather than
        persisting as random noise in the queue.
        """
        if self.eviction_mode == "fifo" or not self._has_wrapped:
            return self._get_fifo_evict_indices(n)

        # --- Prioritise evicted (noise) slots for repopulation ---
        # After commit_evictions(), evicted slots are filled with random noise
        # and marked source_type=2.  These noise slots look maximally "diverse"
        # to the diversity/guided strategy (low similarity to everything), so
        # they'd never be selected for eviction — meaning they'd persist as
        # noise forever.  We intercept here and preferentially overwrite them.
        if self._source_type is not None:
            noise_mask = (self._source_type == 2) & (~self.protected_mask)
            noise_indices = noise_mask.nonzero(as_tuple=True)[0]  # [E]
            if noise_indices.numel() > 0:
                # Take up to n noise slots first
                n_noise = min(n, noise_indices.numel())
                evict = noise_indices[:n_noise]
                if evict.numel() == n:
                    return evict
                # Still need more — fall through to diversity/guided for the remainder
                n_remaining = n - evict.numel()
                # But exclude already-selected noise indices from the strategy
                remaining_mask = torch.ones(self.size, dtype=torch.bool, device=self.feats.device)
                remaining_mask[evict] = False
                remaining_mask[self.protected_mask] = False
                remaining_unprotected = remaining_mask.nonzero(as_tuple=True)[0]
                if remaining_unprotected.numel() == 0:
                    # Only noise and protected left — pad with FIFO (may include protected)
                    fifo = self._get_fifo_evict_indices(n)
                    fifo = fifo[~torch.isin(fifo, evict)]
                    return torch.cat([evict, fifo[:n_remaining]])
                # Run strategy on remaining unprotected (non-noise) indices
                if self.eviction_mode == "diversity":
                    raw = self._get_diversity_evict_indices(min(n_remaining, remaining_unprotected.numel()))
                elif self.eviction_mode == "guided":
                    raw = self._get_guided_evict_indices(min(n_remaining, remaining_unprotected.numel()))
                else:
                    raw = self._get_fifo_evict_indices(min(n_remaining, remaining_unprotected.numel()))
                # Filter to only remaining unprotected indices
                mask = remaining_mask[raw]
                extra = raw[mask]
                if extra.numel() < n_remaining:
                    # Pad with FIFO
                    fifo = self._get_fifo_evict_indices(n)
                    fifo = fifo[~torch.isin(fifo, evict)]
                    fifo = fifo[~torch.isin(fifo, extra)]
                    needed = n_remaining - extra.numel()
                    extra = torch.cat([extra, fifo[:needed]])
                return torch.cat([evict, extra[:n_remaining]])

        # --- Standard path (no noise slots to repopulate) ---
        # Build a candidate pool of unprotected indices
        unprotected = (~self.protected_mask).nonzero(as_tuple=True)[0]  # [U]
        if unprotected.numel() == 0:
            return self._get_fifo_evict_indices(n)

        if self.eviction_mode == "diversity":
            raw = self._get_diversity_evict_indices(min(n, unprotected.numel()))
        elif self.eviction_mode == "guided":
            raw = self._get_guided_evict_indices(min(n, unprotected.numel()))
        else:
            raw = self._get_fifo_evict_indices(min(n, unprotected.numel()))

        # Filter to only unprotected indices
        mask = ~self.protected_mask[raw]
        evict = raw[mask]

        # If we don't have enough, pad with FIFO (may include protected as last resort)
        if evict.numel() < n:
            needed = n - evict.numel()
            fifo = self._get_fifo_evict_indices(n)
            unprotected_fifo = fifo[~self.protected_mask[fifo]]
            unprotected_fifo = unprotected_fifo[~torch.isin(unprotected_fifo, evict)]
            if unprotected_fifo.numel() >= needed:
                evict = torch.cat([evict, unprotected_fifo[:needed]])
            else:
                evict = torch.cat([evict, unprotected_fifo])
                still_needed = n - evict.numel()
                remaining_fifo = fifo[~torch.isin(fifo, evict)]
                evict = torch.cat([evict, remaining_fifo[:still_needed]])

        return evict[:n]

    def _get_fifo_evict_indices(self, n: int) -> torch.Tensor:
        """Return indices of the oldest ``n`` features (FIFO circular buffer)."""
        ptr = self.pointer
        if ptr + n <= self.size:
            return torch.arange(ptr, ptr + n, device=self.feats.device)
        first = self.size - ptr
        return torch.cat([
            torch.arange(ptr, self.size, device=self.feats.device),
            torch.arange(0, n - first, device=self.feats.device),
        ])

    def _get_diversity_evict_indices(self, n: int) -> torch.Tensor:
        """Return indices of the ``n`` most *redundant* features.

        For each feature, compute its cosine similarity to its nearest neighbour
        in the queue.  Evict the features with the highest similarity (most
        redundant / least unique).  This preserves diversity.
        """
        feats_norm = F.normalize(self.feats, dim=1)  # [N, D]
        # Pairwise cosine similarity matrix
        sim = feats_norm @ feats_norm.T  # [N, N]
        # Mask out self-similarity (diagonal)
        sim.fill_diagonal_(-1.0)
        # For each feature, find its max similarity (nearest neighbour)
        max_sim, _ = sim.max(dim=1)  # [N]
        # Evict the n features with highest max_sim (most redundant)
        _, indices = torch.topk(max_sim, k=min(n, self.size), largest=True)
        return indices

    def _get_guided_evict_indices(self, n: int) -> torch.Tensor:
        """Return indices of the ``n`` features farthest from the guidance target.

        Features are ranked by cosine distance to ``self._guidance_target``.
        The farthest features are evicted, retaining those near the target.
        If no guidance target is set, falls back to FIFO.
        """
        if self._guidance_target is None:
            return self._get_fifo_evict_indices(n)

        target = self._guidance_target.to(self.feats.device, dtype=self.feats.dtype)
        target_norm = F.normalize(target.unsqueeze(0), dim=1)  # [1, D]
        feats_norm = F.normalize(self.feats, dim=1)  # [N, D]
        # Cosine similarity to target
        sim = feats_norm @ target_norm.T  # [N, 1]
        # Blend similarity with a diversity bonus to avoid collapsing to one mode
        # diversity bonus: for each feature, 1 - max similarity to any other feature
        pairwise = feats_norm @ feats_norm.T
        pairwise.fill_diagonal_(-1.0)
        diversity_bonus, _ = pairwise.max(dim=1)  # [N]
        # Score: high = good (close to target AND diverse from others)
        score = sim[:, 0] * (1.0 - self.guidance_strength) + (1.0 - diversity_bonus) * self.guidance_strength
        # Evict the n features with lowest score
        _, indices = torch.topk(score, k=min(n, self.size), largest=False)
        return indices

    # -- Protected (pinned) features -------------------------------------------

    @torch.no_grad()
    def protect_indices(self, indices: Union[list, torch.Tensor]):
        """Mark queue indices as protected — they will never be evicted.

        Args:
            indices: List or tensor of queue indices to protect.
        """
        idx = torch.as_tensor(indices, device=self.feats.device, dtype=torch.long)
        self.protected_mask[idx.clamp(0, self.size - 1)] = True
        # Clear any pending eviction mark when protecting
        self.pending_eviction_mask[idx.clamp(0, self.size - 1)] = False

    @torch.no_grad()
    def unprotect_indices(self, indices: Union[list, torch.Tensor]):
        """Remove protection from queue indices.

        Args:
            indices: List or tensor of queue indices to unprotect.
        """
        idx = torch.as_tensor(indices, device=self.feats.device, dtype=torch.long)
        self.protected_mask[idx.clamp(0, self.size - 1)] = False

    @torch.no_grad()
    def mark_eviction_indices(self, indices: Union[list, torch.Tensor]):
        """Mark queue indices for eviction (visual red highlight).

        Does NOT remove the features — just marks them.  Call
        ``commit_evictions()`` to actually purge all marked indices.

        Args:
            indices: List or tensor of queue indices to mark.
        """
        idx = torch.as_tensor(indices, device=self.feats.device, dtype=torch.long)
        self.pending_eviction_mask[idx.clamp(0, self.size - 1)] = True

    @torch.no_grad()
    def clear_eviction_marks(self, indices: Union[list, torch.Tensor]):
        """Clear pending eviction marks from queue indices.

        Args:
            indices: List or tensor of queue indices to un-mark.
        """
        idx = torch.as_tensor(indices, device=self.feats.device, dtype=torch.long)
        self.pending_eviction_mask[idx.clamp(0, self.size - 1)] = False

    @torch.no_grad()
    def commit_evictions(self) -> int:
        """Actually purge all features with pending eviction marks.

        Replaces them with random noise (will be overwritten by normal enqueue).
        Also clears the eviction marks, zeroes out source images/type so
        evicted features don't reappear in the cluster panel, and invalidates
        the cluster cache so the next refresh re-clusters from scratch.

        **Eviction rotation**: After purging, the pointer is advanced past the
        evicted slots so they naturally cycle to the back of the queue.  This
        prevents evicted images from being overwritten in the same position
        over and over — they get kicked to the back instead.

        Returns:
            Number of features evicted.
        """
        mask = self.pending_eviction_mask
        n = mask.sum().item()
        if n == 0:
            return 0

        # --- Save evicted indices BEFORE clearing the mask ---
        # We need the actual indices for eviction rotation, but mask will be
        # cleared in-place below (since mask is a reference to the buffer).
        evict_idx_tensor = mask.nonzero(as_tuple=True)[0]  # [n] long tensor
        evict_idx_sorted = evict_idx_tensor.cpu().tolist()

        # Unprotect first if protected
        self.protected_mask[mask] = False
        # Replace with random noise
        noise = torch.randn(n, self.feat_dim, device=self.feats.device)
        self.feats[mask] = noise
        # Zero out source images so evicted slots don't show stale thumbnails
        if self.store_source_images:
            self.source_images[mask] = 0
        # Mark evicted slots as source_type=2 (evicted/noise) so the cluster
        # panel can show them distinctly from dataset (0) and generated (1).
        # These will be overwritten by normal enqueue as the pointer cycles.
        if self._source_type is not None:
            self._source_type[mask] = 2
        # Reset step counter for evicted slots
        self._step_counter[mask] = 0
        # Clear marks
        self.pending_eviction_mask[:] = False

        # --- Eviction rotation: advance pointer past evicted slots ---
        # This kicks evicted features to the back of the queue so they
        # aren't overwritten in the same position over and over.
        if n > 0:
            # Advance pointer to just past the last evicted slot (wrap-around safe)
            last_evicted = max(evict_idx_sorted)
            new_ptr = (last_evicted + 1) % self.size
            self.ptr[0] = new_ptr
            logger.info(f"[FeatureQueue] Eviction rotation: pointer advanced to {new_ptr} "
                         f"(past {n} evicted slots, last at {last_evicted})")

        # Invalidate cluster cache — queue contents changed
        self._cluster_cache = None
        self._cluster_cache_hash = None
        logger.info(f"[FeatureQueue] Committed eviction of {n} features")
        return n

    @torch.no_grad()
    def _get_cluster_indices(self, cluster_id: int) -> torch.Tensor:
        """Return queue indices belonging to a cluster using the **registry**.

        Uses the persistent ``_cluster_registry`` rather than re-running
        clustering, guaranteeing that cluster-level actions always target
        the same set of queue indices until the next explicit re-clustering.

        Returns:
            A 1-D long tensor of queue indices, or empty if the cluster ID
            is out of range or no features are assigned to it.
        """
        if self._cluster_registry.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=self.feats.device)
        mask = self._cluster_registry == cluster_id
        return mask.nonzero(as_tuple=True)[0]

    @torch.no_grad()
    def protect_cluster(self, cluster_id: int, n_clusters: Optional[int] = None):
        """Protect all features belonging to a given cluster.

        Uses the persistent cluster registry so the action is deterministic
        even across re-clustering calls.  If the registry is stale (e.g. after
        many enqueue steps), call ``get_clusters()`` first to refresh it.

        Args:
            cluster_id: The cluster index to protect.
            n_clusters: Ignored when using the registry; kept for backward
                        compatibility.
        """
        indices = self._get_cluster_indices(cluster_id)
        if indices.numel() == 0:
            logger.warning(f"[FeatureQueue] No features found for cluster {cluster_id} in registry")
            return
        self.protect_indices(indices)
        logger.info(f"[FeatureQueue] Protected cluster {cluster_id} ({indices.numel()} features)")

    @torch.no_grad()
    def unprotect_cluster(self, cluster_id: int, n_clusters: Optional[int] = None):
        """Remove protection from all features in a given cluster.

        Uses the persistent cluster registry so the action is deterministic
        even across re-clustering calls.

        Args:
            cluster_id: The cluster index to unprotect.
            n_clusters: Ignored when using the registry; kept for backward
                        compatibility.
        """
        indices = self._get_cluster_indices(cluster_id)
        if indices.numel() == 0:
            logger.warning(f"[FeatureQueue] No features found for cluster {cluster_id} in registry")
            return
        self.unprotect_indices(indices)
        logger.info(f"[FeatureQueue] Unprotected cluster {cluster_id} ({indices.numel()} features)")

    # -- Internal helpers ------------------------------------------------------

    def _get_evicted_feats(self, n: int) -> torch.Tensor:
        """Return features that will be overwritten by the next enqueue of size n.

        Uses the current eviction strategy to determine which features are evicted.
        """
        indices = self._get_evict_indices(n)
        return self.feats[indices]

    # -- Reference statistics (for self-referential mode) -----------------------

    @torch.no_grad()
    def get_ref_stats(self):
        """Return (mu_ref, sigma_ref, sigma_ref_sqrt) from the queue's current state.

        Used for self-referential FD-Loss when no precomputed reference stats
        are available.  Returns the running mean/covariance of the queue contents.
        """
        if self.ema_stats:
            mu_ref = self.mu_ema.clone()
            sigma_ref = self.m2_ema - mu_ref.unsqueeze(1) * mu_ref.unsqueeze(0)
        elif self.online_accum:
            N = self.size
            feat_sum = self.feat_sum_old.clone()
            feat_outer = self.feat_outer_old.clone()
            mu_ref = feat_sum / N
            sigma_ref = (feat_outer - feat_sum.unsqueeze(1) * feat_sum.unsqueeze(0) / N) / (N - 1)
        else:
            # Snapshot mode: compute from current buffer contents
            feats_d = self.feats.double()
            mu_ref = feats_d.mean(0)
            feats_c = feats_d - mu_ref
            sigma_ref = (feats_c.T @ feats_c) / (self.size - 1)

        # Clamp tiny negative eigenvalues from numerical noise to zero
        sigma_ref = torch.clamp(sigma_ref, min=0.0)
        sigma_ref_sqrt = precompute_sigma_ref_sqrt(sigma_ref)
        # sigma_ref_sqrt may be None if eigendecomposition fails (ill-conditioned)
        return mu_ref, sigma_ref, sigma_ref_sqrt

    # -- Clustering & visualisation --------------------------------------------

    @torch.no_grad()
    def _compute_cluster_hash(self) -> Optional[bytes]:
        """Compute a hash of the current queue contents for cache invalidation.

        Returns None if the queue is empty or in EMA mode.
        """
        if self.ema_stats or self.size == 0:
            return None
        # Use a fast hash of a strided view — sha256 of the full 50k×2048 buffer
        # would be slow, so we sample a fixed number of bytes from the float buffer.
        feats = self.feats.float()
        # Sample up to 1MB of data for the hash (enough to detect meaningful changes)
        total_bytes = feats.numel() * feats.element_size()
        step = max(1, total_bytes // (1024 * 1024))
        sampled = feats.view(-1)[::step].cpu().numpy().tobytes()
        return hashlib.sha256(sampled).digest()

    @torch.no_grad()
    def get_clusters(self, n_clusters: Optional[int] = None, force: bool = False) -> dict:
        """Run k-means on the current queue contents and return cluster info.

        Results are **cached** based on a hash of the queue data.  Repeated calls
        with unchanged queue contents return the cached result instantly, skipping
        the expensive k-means++ computation.  The cache is invalidated on every
        ``enqueue()`` call (see :meth:`enqueue`).

        Uses **non-deterministic seeding** (random seed each call) so that
        clusters naturally re-organise as the feature distribution shifts due to
        computer-generated imagery and evictions.  This gives a fresh perspective
        on the data each time, which is more useful when the queue contents are
        dynamically evolving.

        Args:
            n_clusters: Number of clusters (default: ``self.n_clusters``).
            force: If True, bypass the cache and force re-clustering.

        Returns a dict with:
          - ``'centroids'``: [n_clusters, feat_dim] tensor
          - ``'assignments'``: [N] long tensor of cluster indices
          - ``'sizes'``: [n_clusters] int tensor, number of features per cluster
          - ``'inertia'``: float, sum of squared distances to centroids
          - ``'silhouette'``: float, average silhouette score (-1 to 1, higher = better)

        This is designed for inter-epoch inspection — call it between training
        epochs to visualise what the queue contains.
        """
        if self.ema_stats:
            logger.warning("[FeatureQueue] Clustering not supported in EMA mode")
            return {}

        n_clusters = n_clusters or self.n_clusters
        n_clusters = min(n_clusters, self.size, self.feat_dim)

        # --- Cache check ---
        if not force:
            cache_hash = self._compute_cluster_hash()
            if (self._cluster_cache is not None
                    and self._cluster_cache_n_clusters == n_clusters
                    and cache_hash is not None
                    and self._cluster_cache_hash == cache_hash):
                return self._cluster_cache

        feats = self.feats.float()  # [N, D]
        N = feats.shape[0]

        # --- Non-deterministic seed ---
        # Use a random seed so clusters are different each time the queue
        # contents change (after enqueue).  This allows the clustering to
        # naturally re-organise as generated images and evictions shift the
        # feature distribution.
        gen = torch.Generator(device='cpu')
        gen.seed()

        # --- Initialise centroids via k-means++ (randomised) ---
        device = feats.device
        centroids = torch.empty(n_clusters, self.feat_dim, device=device)
        # First centroid: random from data
        idx = torch.randint(0, N, (1,), generator=gen).item()
        centroids[0] = feats[idx]
        for k in range(1, n_clusters):
            # Distance from each point to nearest centroid
            dists = torch.cdist(feats, centroids[:k])  # [N, k]
            min_dists, _ = dists.min(dim=1)  # [N]
            probs = min_dists / (min_dists.sum() + 1e-8)
            # multinomial on CPU probabilities, then index into CUDA feats
            idx = torch.multinomial(probs.cpu(), 1, generator=gen).item()
            centroids[k] = feats[idx]

        # --- Lloyd iterations ---
        for _ in range(20):
            dists = torch.cdist(feats, centroids)  # [N, n_clusters]
            assignments = dists.argmin(dim=1)  # [N]
            for k in range(n_clusters):
                mask = assignments == k
                if mask.any():
                    centroids[k] = feats[mask].mean(dim=0)

        # Final assignment
        dists = torch.cdist(feats, centroids)
        assignments = dists.argmin(dim=1)
        min_dists, _ = dists.min(dim=1)
        inertia = min_dists.sum().item()

        # --- Update cluster registry ---
        self._cluster_registry.copy_(assignments)

        # Cluster sizes
        sizes = torch.zeros(n_clusters, dtype=torch.long, device=device)
        sizes.scatter_add_(0, assignments, torch.ones_like(assignments))

        # --- Silhouette score (approximate: uses centroids, not full pairwise) ---
        # For each point: a = distance to its own centroid
        #                 b = min distance to other centroids
        # silhouette = (b - a) / max(a, b)
        a = min_dists  # [N]
        # For each point, find distance to the nearest *other* centroid
        other_dists = dists.clone()
        other_dists[torch.arange(N, device=device), assignments] = float('inf')
        b, _ = other_dists.min(dim=1)  # [N]
        denom = torch.max(a, b).clamp(min=1e-8)
        sil = ((b - a) / denom).mean().item()

        result = {
            "centroids": centroids,
            "assignments": assignments,
            "sizes": sizes,
            "inertia": inertia,
            "silhouette": sil,
        }

        # --- Populate cache ---
        self._cluster_cache = result
        self._cluster_cache_n_clusters = n_clusters
        self._cluster_cache_hash = self._compute_cluster_hash()

        return result

    @torch.no_grad()
    def get_cluster_summary_html(self, n_clusters: Optional[int] = None) -> str:
        """Return an HTML snippet summarising the current cluster structure.

        Useful for rendering in a Gradio HTML component between epochs.
        """
        clusters = self.get_clusters(n_clusters)
        if not clusters:
            return "<div style='color:#888;'>Clustering unavailable (EMA mode)</div>"

        centroids = clusters["centroids"]
        sizes = clusters["sizes"]
        sil = clusters["silhouette"]
        n = centroids.shape[0]

        lines = [
            f"<div style='font-family:sans-serif;font-size:12px;color:#ccc;'>",
            f"<b>Feature Queue Clusters</b> &nbsp; "
            f"<span style='color:#888;'>{n} clusters, "
            f"silhouette={sil:.3f}</span>",
            "</div>",
            "<div style='margin-top:6px;font-family:monospace;font-size:11px;'>",
        ]

        for k in range(n):
            pct = 100.0 * sizes[k].item() / max(1, sizes.sum().item())
            bar_w = max(2.0, pct * 3.0)
            hue = (k * 360.0 / n) % 360.0
            lines.append(
                f"<div style='margin:2px 0;'>"
                f"<span style='color:#aaa;'>C{k}:</span> "
                f"<span style='display:inline-block;width:{bar_w:.0f}px;"
                f"height:10px;background:hsl({hue},60%,50%);border-radius:2px;"
                f"vertical-align:middle;'></span> "
                f"<span style='color:#666;'>{sizes[k].item()} ({pct:.1f}%)</span>"
                f"</div>"
            )

        lines.append("</div>")
        return "".join(lines)

    # -- Visual interactive cluster panel --------------------------------------

    @torch.no_grad()
    def get_cluster_thumbnails(self, n_clusters: Optional[int] = None,
                               max_per_cluster: int = 9,
                               clusters: Optional[dict] = None) -> dict:
        """Return source image thumbnails grouped by cluster.

        Requires ``store_source_images=True`` at construction.  Returns a dict
        mapping cluster_id -> list of (index, PIL Image) tuples.

        If source images are not available, returns an empty dict.

        Args:
            n_clusters: Number of clusters (ignored if ``clusters`` is provided).
            max_per_cluster: Maximum thumbnails per cluster.
            clusters: Pre-computed cluster dict from ``get_clusters()``.
                      When provided, avoids a redundant clustering call.
        """
        if not self.store_source_images:
            return {}
        if clusters is None:
            clusters = self.get_clusters(n_clusters)
        if not clusters:
            return {}

        assignments = clusters["assignments"]  # [N]
        n = int(clusters["centroids"].shape[0])
        result: dict = {}

        for k in range(n):
            mask = assignments == k
            indices = mask.nonzero(as_tuple=True)[0]
            # Filter out evicted indices — they're just noise/black squares,
            # not useful to show in previews.
            if hasattr(self, 'pending_eviction_mask') and indices.numel() > 0:
                evicted = self.pending_eviction_mask[indices]
                indices = indices[~evicted]
            # Take up to max_per_cluster samples
            if indices.numel() > max_per_cluster:
                # Pick evenly spaced indices for a representative view
                step = indices.numel() / max_per_cluster
                pick = [indices[int(i * step)].item() for i in range(max_per_cluster)]
            else:
                pick = indices.tolist()

            thumbs = []
            for idx in pick:
                img_uint8 = self.source_images[idx]  # [3, H, W] uint8
                # Convert to PIL Image for Gradio
                arr = img_uint8.cpu().permute(1, 2, 0).numpy()
                from PIL import Image
                pil_img = Image.fromarray(arr)
                thumbs.append((idx, pil_img))

            result[k] = thumbs

        return result

    @torch.no_grad()
    def fetch_thumbnail_base64(self, queue_idx: int) -> str:
        """Return a base64-encoded PNG data URI for a single queue index.

        This is called **on-demand** when the user expands a cluster accordion,
        avoiding the cost of base64-encoding all thumbnails at render time.

        Args:
            queue_idx: Index into the queue's source_images buffer.

        Returns:
            A ``data:image/png;base64,...`` string, or an empty string if
            source images are not available or the index is out of range.
        """
        if not self.store_source_images:
            return ""
        if queue_idx < 0 or queue_idx >= self.source_images.shape[0]:
            return ""
        img_uint8 = self.source_images[queue_idx]  # [3, H, W] uint8
        arr = img_uint8.cpu().permute(1, 2, 0).numpy()
        from PIL import Image
        import io, base64
        pil_img = Image.fromarray(arr)
        buf = io.BytesIO()
        pil_img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return f"data:image/png;base64,{b64}"

    def render_interactive_cluster_panel(self, n_clusters: Optional[int] = None,
                                         max_per_cluster: int = 9,
                                         n_pending_actions: int = 0,
                                         force_cluster: bool = False) -> str:
        """Return an HTML string with an interactive cluster panel for Gradio.

        Each cluster is shown as a collapsible section with:
          - Cluster ID, size, percentage, protection status, guidance indicator
          - Cluster-level action buttons (Protect toggle, Set as Guidance, Evict Cluster)
          - Up to ``max_per_cluster`` source image thumbnails (base64-encoded),
            **each with its own individual** Protect toggle / Evict buttons
            bound to the specific queue index via ``data-idx``.

        **Stateful visual effects**:
          - Protected clusters/items get a green border glow + green tint background
          - Guidance-targeted clusters get a blue border glow + blue tint
          - Eviction-marked items get a red border glow + red tint
          - Conflict (protected + eviction-marked) gets orange

        **Protect is a toggle**: a single button switches between protected/unprotected.

        **Guidance explanation**: When a cluster is set as guidance, a badge explains
        that the queue will preferentially retain features near this cluster's centroid.

        **Staged actions**: All actions (protect/guidance/evict) are buffered and
        applied atomically when the Commit button is clicked.  Pass
        ``n_pending_actions`` to show the running total in the button label.

        The buttons emit Gradio events via ``data-*`` attributes that the
        frontend JS can pick up:
          - ``data-cluster`` for cluster-level actions
          - ``data-idx`` for per-sample actions

        If source images are not available, falls back to the text summary.
        """
        clusters = self.get_clusters(n_clusters, force=force_cluster)
        if not clusters:
            return "<div style='color:#888;'>Clustering unavailable (EMA mode)</div>"

        centroids = clusters["centroids"]
        sizes = clusters["sizes"]
        sil = clusters["silhouette"]
        assignments = clusters["assignments"]
        n = centroids.shape[0]
        total = max(1, sizes.sum().item())

        # Guidance is 100% per-image — we do NOT mark any cluster as "the
        # guidance cluster".  The guidance pseudo-cluster (rendered below)
        # shows the top-N features closest to the guidance target instead.
        # The `guidance_cluster_id` variable has been removed; cluster-level
        # visual styling no longer includes a guidance blue glow.

        # Pre-compute which indices are closest to the guidance target, so
        # per-image guidance badges can be shown in regular clusters.
        guidance_near_indices: set = set()
        if self._guidance_target is not None and self.eviction_mode == "guided":
            target = self._guidance_target.to(self.feats.device, dtype=self.feats.dtype)
            target_norm = F.normalize(target.unsqueeze(0), dim=1)
            feats_norm = F.normalize(self.feats, dim=1)
            sim = feats_norm @ target_norm.T  # [N, 1]
            sorted_sim, sorted_idx = torch.sort(sim[:, 0], descending=True)
            for idx_val in sorted_idx.tolist():
                if len(guidance_near_indices) >= max_per_cluster:
                    break
                if hasattr(self, 'pending_eviction_mask') and self.pending_eviction_mask[idx_val].item():
                    continue
                guidance_near_indices.add(idx_val)

        # Get thumbnails if available — pass pre-computed clusters to avoid a
        # redundant k-means call inside get_cluster_thumbnails().
        thumbs = self.get_cluster_thumbnails(n_clusters, max_per_cluster, clusters=clusters) if self.store_source_images else {}
        has_thumbs = bool(thumbs)

        lines = [
            "<div style='font-family:sans-serif;font-size:13px;color:#ccc;margin-bottom:8px;'>",
            f"<b>Feature Queue — Interactive Cluster Inspector</b> &nbsp; ",
            f"<span style='color:#888;'>{n} clusters, ",
            f"silhouette={sil:.3f}</span>",
            " &nbsp; <span style='color:#666;font-size:11px;'>"
            "All changes (protect/guidance/evict) are <b>staged</b> until "
            "you click <b>Commit</b>.</span>",
            "</div>",
            # --- Thumbnail source explanation ---
            "<div style='margin-bottom:6px;padding:6px 10px;"
            "background:rgba(96,165,250,0.08);border:1px solid rgba(96,165,250,0.25);"
            "border-radius:3px;font-size:11px;color:#93c5fd;'>"
            "📷 <b>Thumbnail source:</b> "
            "<span style='color:#4ade80;'>🟢 Dataset</span> = real training images "
            "(pre-filled before training). "
            "<span style='color:#fbbf24;'>🟡 Generated</span> = model outputs enqueued "
            "during training. "
            "<span style='color:#888;'>💀 Evicted</span> = noise slots waiting to be "
            "overwritten by normal enqueue."
            "</div>",
            # Guidance explanation banner (shown when guidance is active)
            *self._render_guidance_banner(),
            # Commit-all button — always visible, shows running total of staged actions
            (
                f"<div style='margin-bottom:8px;display:flex;gap:6px;align-items:center;'>"
                f"<button id='fd-commit-all' "
                f"style='padding:4px 14px;font-size:12px;border:1px solid "
                f"{'#ef4444' if n_pending_actions > 0 else '#555'};"
                f"background:{'rgba(239,68,68,0.15)' if n_pending_actions > 0 else 'transparent'};"
                f"color:{'#ef4444' if n_pending_actions > 0 else '#666'};"
                f"border-radius:3px;cursor:pointer;font-weight:600;'>"
                f"{'⚠️' if n_pending_actions > 0 else '💾'} "
                f"{f'Commit {n_pending_actions} Change(s)' if n_pending_actions > 0 else 'Commit All'}</button>"
                f"<span style='font-size:10px;color:{'#ef4444' if n_pending_actions > 0 else '#666'};'>"
                f"{f'{n_pending_actions} staged action(s) pending' if n_pending_actions > 0 else 'No pending changes'}</span>"
                f"</div>"
            ),
            "<div style='max-height:600px;overflow-y:auto;'>",
        ]

        # Shared style for per-sample buttons (smaller than cluster-level)
        _SAMPLE_BTN = (
            "padding:1px 6px;font-size:10px;border:1px solid #555;"
            "background:transparent;border-radius:2px;cursor:pointer;"
        )

        # --- Guidance pseudo-cluster ---
        # When guidance mode is active, promote the guidance target to its own
        # pseudo-cluster at the top of the list.  This shows the top-N indices
        # closest to the guidance target by cosine similarity (computed above
        # in ``guidance_near_indices``), with per-index action buttons (protect,
        # guidance, evict).  The guidance target is always associated with
        # individual feature vectors, never with clusters.
        #
        # Thumbnails are **eagerly loaded** (base64 embedded directly) because
        # the pseudo-cluster is always visible — it's not inside a <details>
        # accordion, so the JS lazy-load mechanism (which fires on <details>
        # toggle) would never trigger for these images.
        if guidance_near_indices:
            lines.append(
                "<div style='margin:6px 0;padding:8px 12px;"
                "background:rgba(96,165,250,0.10);border:2px solid #60a5fa;"
                "border-radius:6px;box-shadow:0 0 12px rgba(96,165,250,0.25);'>"
                "<div style='display:flex;align-items:center;gap:8px;margin-bottom:6px;'>"
                "<span style='font-size:16px;'>🎯</span>"
                "<span style='font-weight:600;color:#93c5fd;font-size:13px;'>Guidance Target</span>"
                "<span style='color:#888;font-size:11px;'>"
                f"— top {len(guidance_near_indices)} features closest to guidance target"
                "</span>"
                "</div>"
                "<div class='fd-thumb-grid' style='display:flex;gap:6px;flex-wrap:wrap;'>"
            )
            for idx in sorted(guidance_near_indices):
                idx_protected = self.protected_mask[idx].item()
                idx_eviction = self.pending_eviction_mask[idx].item() if hasattr(self, 'pending_eviction_mask') else False
                idx_source_type = int(self._source_type[idx].item()) if self._source_type is not None else 0
                if idx_source_type == 0:
                    source_type_label = "📷"
                    source_type_color = "#4ade80"
                    source_type_title = "Dataset (real training image)"
                elif idx_source_type == 1:
                    source_type_label = "⚙️"
                    source_type_color = "#fbbf24"
                    source_type_title = "Machine-generated (model output)"
                else:
                    source_type_label = "💀"
                    source_type_color = "#666"
                    source_type_title = "Evicted (noise)"

                prot_color = "#4ade80" if idx_protected else "#555"
                prot_bg = "rgba(74,222,128,0.15)" if idx_protected else "transparent"
                prot_label = "🔒" if idx_protected else "🛡️"
                prot_title = "Click to unprotect" if idx_protected else "Click to protect"

                if idx_eviction and idx_protected:
                    item_border = "#f97316"
                    item_bg = "rgba(249,115,22,0.08)"
                    item_shadow = "0 0 4px rgba(249,115,22,0.3)"
                elif idx_eviction:
                    item_border = "#ef4444"
                    item_bg = "rgba(239,68,68,0.08)"
                    item_shadow = "0 0 4px rgba(239,68,68,0.3)"
                elif idx_protected:
                    item_border = "#4ade80"
                    item_bg = "rgba(74,222,128,0.05)"
                    item_shadow = "0 0 4px rgba(74,222,128,0.2)"
                elif idx_source_type == 1:
                    item_border = "#fbbf24"
                    item_bg = "rgba(251,191,36,0.04)"
                    item_shadow = "0 0 4px rgba(251,191,36,0.15)"
                else:
                    item_border = "#333"
                    item_bg = "transparent"
                    item_shadow = "none"

                evict_color = "#ef4444" if idx_eviction else "#fbbf24"
                evict_bg = "rgba(239,68,68,0.15)" if idx_eviction else "transparent"
                evict_label = "↩️" if idx_eviction else "❌"
                evict_title = "Clear eviction mark" if idx_eviction else "Mark for eviction"

                if idx_eviction and idx_protected:
                    idx_state_str = "conflict"
                elif idx_eviction:
                    idx_state_str = "eviction"
                elif idx_protected:
                    idx_state_str = "protected"
                elif idx_source_type == 1:
                    idx_state_str = "generated"
                else:
                    idx_state_str = "normal"

                # Eagerly load thumbnail — embed base64 directly since this
                # section is always visible (not inside a <details> accordion).
                thumb_src = self.fetch_thumbnail_base64(idx) if self.store_source_images else ""

                lines.append(
                    f"<div data-idx-state='{idx_state_str}' "
                    f"data-idx='{idx}' data-cluster='guidance' "
                    f"data-source-type='{idx_source_type}' "
                    f"style='display:flex;flex-direction:column;align-items:center;"
                    f"gap:2px;padding:3px;border:1px solid {item_border};border-radius:3px;"
                    f"background:{item_bg};"
                    f"box-shadow:{item_shadow};'>"
                    f"<img src='{thumb_src}' "
                    f"style='width:calc({self.source_img_size}px * var(--thumb-scale, 1.0));"
                    f"height:calc({self.source_img_size}px * var(--thumb-scale, 1.0));object-fit:cover;"
                    f"border-radius:2px;border:1px solid #333;"
                    f"background:#1a1a1a;' "
                    f"title='idx={idx} | guidance target candidate'/>"
                    f"<span style='font-size:9px;color:#666;display:flex;align-items:center;gap:2px;'>"
                    f"#{idx} "
                    f"<span style='color:{source_type_color};font-size:8px;' "
                    f"title='{source_type_title}'>{source_type_label}</span>"
                    f"</span>"
                    f"<div style='display:flex;gap:2px;'>"
                    f"<button class='fd-idx-protect' data-idx='{idx}' data-cluster='guidance' "
                    f"style='{_SAMPLE_BTN}border-color:{prot_color};"
                    f"background:{prot_bg};color:{prot_color};'"
                    f"title='{prot_title}'>{prot_label}</button>"
                    f"<button class='fd-idx-guidance' data-idx='{idx}' data-cluster='guidance' "
                    f"style='{_SAMPLE_BTN}border-color:#60a5fa;"
                    f"background:rgba(96,165,250,0.15);color:#60a5fa;'"
                    f"title='Set guidance target from this specific image'>🎯</button>"
                    f"<button class='fd-idx-evict' data-idx='{idx}' data-cluster='guidance' "
                    f"style='{_SAMPLE_BTN}border-color:{evict_color};"
                    f"background:{evict_bg};color:{evict_color};'"
                    f"title='{evict_title}'>{evict_label}</button>"
                    f"</div>"
                    f"</div>"
                )
            lines.append("</div></div>")

        for k in range(n):
            pct = 100.0 * sizes[k].item() / total
            hue = (k * 360.0 / n) % 360.0

            # --- Per-index status counts for this cluster ---
            # We NEVER associate actions with clusters themselves — only with
            # individual indices within clusters.  The cluster-level display
            # shows proportional counts so the user sees exactly how many
            # indices are protected/eviction-marked, not a binary "this
            # cluster IS protected" indicator.
            cluster_mask = (assignments == k)
            if cluster_mask.device != self.protected_mask.device:
                cluster_mask = cluster_mask.to(device=self.protected_mask.device)
            n_in_cluster = int(cluster_mask.sum().item())
            n_protected = int(self.protected_mask[cluster_mask].sum().item()) if n_in_cluster > 0 else 0
            n_eviction = int(self.pending_eviction_mask[cluster_mask].sum().item()) if (hasattr(self, 'pending_eviction_mask') and n_in_cluster > 0) else 0

            # Skip entirely-evicted clusters — no useful content to show
            if n_in_cluster > 0 and n_in_cluster - n_eviction == 0:
                continue

            # Source type composition for this cluster
            if self._source_type is not None:
                # Ensure mask is on the same device as _source_type
                if cluster_mask.device != self._source_type.device:
                    cluster_mask = cluster_mask.to(device=self._source_type.device)
                st = self._source_type[cluster_mask]  # [n_in_cluster]
                n_dataset = int((st == 0).sum().item())
                n_gen = int((st == 1).sum().item())
                n_evicted_st = int((st == 2).sum().item())
                n_total = max(1, n_in_cluster)
                pct_generated = 100.0 * n_gen / n_total
                pct_evicted_st = 100.0 * n_evicted_st / n_total
                # Build badge: show the dominant category
                badges = []
                if pct_evicted_st > 0:
                    badges.append(f'<span style="color:#888;font-size:10px;">💀 {pct_evicted_st:.0f}% evicted</span>')
                if pct_generated >= 50:
                    badges.append(f'<span style="color:#fbbf24;font-size:10px;">🟡 {pct_generated:.0f}% gen</span>')
                elif pct_generated > 0:
                    badges.append(f'<span style="color:#4ade80;font-size:10px;">🟢 {100-pct_generated:.0f}% dataset</span>')
                elif pct_evicted_st == 0:
                    badges.append('<span style="color:#4ade80;font-size:10px;">🟢 100% dataset</span>')
                source_badge = " ".join(badges)
            else:
                source_badge = ""

            # --- Stateful visual styling ---
            # Cluster-level styling is based on MAJORITY of indices, not .any().
            # This ensures the cluster border/glow reflects the dominant state
            # rather than falsely associating a single index's status with the
            # entire cluster.
            # Priority: eviction-majority > protected-majority > generated-dominant > normal
            # Protected + eviction-majority → orange (conflict indicator)
            # Eviction-majority → red
            # Protected-majority → green
            # Generated-dominant → yellow tint
            # NOTE: Guidance is 100% per-image — no cluster-level guidance styling.
            is_protected_majority = n_protected > n_in_cluster / 2
            is_eviction_majority = n_eviction > n_in_cluster / 2
            if is_eviction_majority and is_protected_majority:
                border_color = "#f97316"  # orange — conflict
                bg_color = "rgba(249,115,22,0.08)"
                glow = "0 0 8px rgba(249,115,22,0.3)"
            elif is_eviction_majority:
                border_color = "#ef4444"  # red
                bg_color = "rgba(239,68,68,0.08)"
                glow = "0 0 8px rgba(239,68,68,0.3)"
            elif is_protected_majority:
                border_color = "#4ade80"  # green
                bg_color = "rgba(74,222,128,0.08)"
                glow = "0 0 8px rgba(74,222,128,0.3)"
            else:
                # Check if cluster is dominated by generated images → yellow tint
                if self._source_type is not None and n_gen > n_dataset and n_gen > n_evicted_st:
                    border_color = "#fbbf24"
                    bg_color = "rgba(251,191,36,0.04)"
                    glow = "0 0 4px rgba(251,191,36,0.12)"
                else:
                    border_color = f"hsl({hue},50%,40%)"
                    bg_color = "#1a1a1a"
                    glow = "none"

            # Cluster-level protect button: show count badge, not binary status
            if n_protected > 0:
                prot_badge = f'<span style="color:#4ade80;font-size:10px;">🔒 {n_protected}/{n_in_cluster}</span>'
                protect_label = f"🔒 Unprotect All ({n_protected})"
                protect_title = f"Unprotect all {n_protected} protected indices in this cluster"
            else:
                prot_badge = ""
                protect_label = "🛡️ Protect All"
                protect_title = "Protect all indices in this cluster"

            # Cluster-level evict button: show count badge, not binary status
            if n_eviction > 0:
                evict_badge = f'<span style="color:#ef4444;font-size:10px;"> ❌ {n_eviction}/{n_in_cluster}</span>'
                evict_label = f"↩️ Unmark Eviction ({n_eviction})"
                evict_title = f"Clear eviction marks on {n_eviction} indices in this cluster"
            else:
                evict_badge = ""
                evict_label = "❌ Mark Eviction"
                evict_title = "Mark all indices in this cluster for eviction"

            # Determine data-cluster-state: based on majority for visual grouping
            if is_eviction_majority:
                cluster_state_str = "eviction"
            elif is_protected_majority:
                cluster_state_str = "protected"
            else:
                cluster_state_str = "normal"
            lines.append(
                f"<details class='fd-cluster-details' style='margin:4px 0;padding:6px 10px;"
                f"background:{bg_color};border:1px solid {border_color};"
                f"border-radius:4px;box-shadow:{glow};' "
                f"data-cluster-state='{cluster_state_str}'>"
                f"<summary style='cursor:pointer;font-size:12px;'>"
                f"<span style='color:#fff;font-weight:600;'>C{k}</span> "
                f"{prot_badge}{evict_badge}"
                f" <span style='color:#888;'>({sizes[k].item()} features, {pct:.1f}%)</span>"
                f" {source_badge}"
                f"</summary>"
                f"<div style='margin-top:6px;'>"
            )

            # --- Cluster-level action buttons ---
            # These are CONVENIENCE buttons that batch-operate on all indices
            # in the cluster.  The labels reflect the COUNT of affected indices,
            # not a binary "this cluster IS protected" status, because actions
            # are always associated with indices, never with clusters.
            prot_btn_color = "#4ade80" if n_protected > 0 else "#555"
            prot_btn_bg = "rgba(74,222,128,0.15)" if n_protected > 0 else "transparent"
            evict_btn_color = "#ef4444" if n_eviction > 0 else "#fbbf24"
            evict_btn_bg = "rgba(239,68,68,0.15)" if n_eviction > 0 else "transparent"
            lines.append(
                f"<div style='margin:4px 0;display:flex;gap:6px;flex-wrap:wrap;'>"
                f"<button class='fd-protect-btn' data-cluster='{k}' "
                f"style='padding:2px 10px;font-size:11px;border:1px solid "
                f"{prot_btn_color};"
                f"background:{prot_btn_bg};"
                f"color:{prot_btn_color};"
                f"border-radius:3px;cursor:pointer;' "
                f"title='{protect_title}'>{protect_label}</button>"
                f"<button class='fd-guidance-btn' data-cluster='{k}' "
                f"style='padding:2px 10px;font-size:11px;border:1px solid "
                f"#555;"
                f"background:transparent;"
                f"color:#ccc;"
                f"border-radius:3px;cursor:pointer;' "
                f"title='Set guidance from an image in this cluster (resolved to per-index)'>"
                f"🎯 Set as Guidance</button>"
                f"<button class='fd-evict-btn' data-cluster='{k}' "
                f"style='padding:2px 10px;font-size:11px;border:1px solid {evict_btn_color};"
                f"background:{evict_btn_bg};color:{evict_btn_color};border-radius:3px;cursor:pointer;' "
                f"title='{evict_title}'>{evict_label}</button>"
                f"</div>"
            )

            # --- Notice if thumbnails are not available ---
            if not self.store_source_images:
                lines.append(
                    "<div style='margin-top:6px;padding:6px 10px;"
                    "background:#2a1a1a;border:1px solid #fbbf24;border-radius:3px;"
                    "font-size:11px;color:#fbbf24;'>"
                    "⚠️ <b>Thumbnails not available.</b> "
                    "Enable <b>fd_store_source_images</b> in Options before starting "
                    "training to see per-sample previews and individual action buttons."
                    "</div>"
                )
            elif not has_thumbs:
                lines.append(
                    "<div style='margin-top:6px;padding:6px 10px;"
                    "background:#1a1a2a;border:1px solid #60a5fa;border-radius:3px;"
                    "font-size:11px;color:#60a5fa;'>"
                    "ℹ️ Source image storage is enabled but no thumbnails are available yet. "
                    "Features enqueued <b>after</b> training started will have thumbnails."
                    "</div>"
                )

            # --- Per-sample thumbnails with individual action buttons ---
            # Thumbnails are **truly lazy**: the HTML stores only the queue index
            # (``data-fetch-idx``).  When the accordion is expanded, the JS calls
            # ``postAction("fetch_thumb", idx)`` which triggers a Gradio round-trip
            # to base64-encode *only that one thumbnail* on demand.  This avoids
            # the cost of encoding all thumbnails at render time.
            if k in thumbs and thumbs[k]:
                lines.append(
                    "<div style='margin-top:6px;font-size:10px;color:#666;'>"
                    "Individual features — click buttons to act on a single sample:"
                    "</div>"
                )
                lines.append(
                    "<div class='fd-thumb-grid' style='display:flex;gap:6px;flex-wrap:wrap;margin-top:4px;'>"
                )
                for i, (idx, pil_img) in enumerate(thumbs[k]):
                    # NOTE: pil_img is used only for dimensions; we do NOT
                    # base64-encode it here.  Encoding happens on demand when
                    # the user expands the accordion.

                    idx_protected = self.protected_mask[idx].item()
                    idx_eviction = self.pending_eviction_mask[idx].item() if hasattr(self, 'pending_eviction_mask') else False
                    # Source type: 0=dataset (green), 1=generated (yellow), 2=evicted/noise (red/grey)
                    idx_source_type = int(self._source_type[idx].item()) if self._source_type is not None else 0
                    if idx_source_type == 0:
                        source_type_label = "📷"
                        source_type_color = "#4ade80"
                        source_type_title = "Dataset (real training image)"
                    elif idx_source_type == 1:
                        source_type_label = "⚙️"
                        source_type_color = "#fbbf24"
                        source_type_title = "Machine-generated (model output)"
                    else:
                        source_type_label = "💀"
                        source_type_color = "#666"
                        source_type_title = "Evicted (noise) — will be overwritten as training continues"

                    # --- Step counter for generated images ---
                    idx_step_count = int(self._step_counter[idx].item()) if hasattr(self, '_step_counter') else 0
                    is_old_noisy_generated = (
                        idx_source_type == 1
                        and idx_step_count >= self.auto_evict_hint_steps
                        and not idx_protected
                    )

                    prot_color = "#4ade80" if idx_protected else "#555"
                    prot_bg = "rgba(74,222,128,0.15)" if idx_protected else "transparent"
                    prot_label = "🔒" if idx_protected else "🛡️"
                    prot_title = "Click to unprotect" if idx_protected else "Click to protect"

                    # --- Per-item visual state ---
                    # Priority: eviction > conflict > protected > generated-tint > normal
                    # Generated images get a yellow/orange tinted border (unless overridden)
                    if idx_eviction and idx_protected:
                        item_border = "#f97316"  # orange conflict
                        item_bg = "rgba(249,115,22,0.08)"
                        item_shadow = "0 0 4px rgba(249,115,22,0.3)"
                    elif idx_eviction:
                        item_border = "#ef4444"  # red
                        item_bg = "rgba(239,68,68,0.08)"
                        item_shadow = "0 0 4px rgba(239,68,68,0.3)"
                    elif idx_protected:
                        item_border = "#4ade80"
                        item_bg = "rgba(74,222,128,0.05)"
                        item_shadow = "0 0 4px rgba(74,222,128,0.2)"
                    elif idx_source_type == 1:
                        # Generated image — yellow/orange tinted border
                        item_border = "#fbbf24"
                        item_bg = "rgba(251,191,36,0.04)"
                        item_shadow = "0 0 4px rgba(251,191,36,0.15)"
                    else:
                        item_border = "#333"
                        item_bg = "transparent"
                        item_shadow = "none"

                    evict_color = "#ef4444" if idx_eviction else "#fbbf24"
                    evict_bg = "rgba(239,68,68,0.15)" if idx_eviction else "transparent"
                    evict_label = "↩️" if idx_eviction else "❌"
                    evict_title = "Clear eviction mark" if idx_eviction else "Mark for eviction"

                    # Determine per-item state string for optimistic UI
                    if idx_eviction and idx_protected:
                        idx_state_str = "conflict"
                    elif idx_eviction:
                        idx_state_str = "eviction"
                    elif idx_protected:
                        idx_state_str = "protected"
                    elif idx_source_type == 1:
                        idx_state_str = "generated"
                    else:
                        idx_state_str = "normal"

                    # --- Auto-eviction hint badge ---
                    auto_evict_hint_html = ""
                    if is_old_noisy_generated:
                        auto_evict_hint_html = (
                            f"<span style='display:inline-block;font-size:7px;"
                            f"color:#fbbf24;background:rgba(251,191,36,0.15);"
                            f"padding:1px 3px;border-radius:2px;"
                            f"border:1px solid rgba(251,191,36,0.3);"
                            f"line-height:1.2;' "
                            f"title='This generated image has survived {idx_step_count} steps "
                            f"(threshold: {self.auto_evict_hint_steps}). "
                            f"If it is still noisy/low-quality, consider evicting it.'>"
                            f"⏳{idx_step_count}</span>"
                        )
                    elif idx_source_type == 1 and idx_step_count > 0:
                        # Show step count without warning for younger generated images
                        auto_evict_hint_html = (
                            f"<span style='font-size:7px;color:#666;' "
                            f"title='Steps survived: {idx_step_count}'>"
                            f"⏱{idx_step_count}</span>"
                        )
                    guidance_indicator_html = (
                        "<span style='color:#60a5fa;font-size:9px;' "
                        "title='Near guidance target'>🎯</span>"
                        if idx in guidance_near_indices else ""
                    )
                    lines.append(
                        f"<div data-idx-state='{idx_state_str}' "
                        f"data-idx='{idx}' data-cluster='{k}' "
                        f"data-source-type='{idx_source_type}' "
                        f"data-step-count='{idx_step_count}' "
                        f"style='display:flex;flex-direction:column;align-items:center;"
                        f"gap:2px;padding:3px;border:1px solid {item_border};border-radius:3px;"
                        f"background:{item_bg};"
                        f"box-shadow:{item_shadow};'>"
                        # Thumbnail — truly lazy: no base64 data in HTML.
                        # JS fetches it on demand via postAction("fetch_thumb", idx)
                        # when the accordion is expanded.
                        f"<img data-fetch-idx='{idx}' "
                        f"style='width:calc({self.source_img_size}px * var(--thumb-scale, 1.0));"
                        f"height:calc({self.source_img_size}px * var(--thumb-scale, 1.0));object-fit:cover;"
                        f"border-radius:2px;border:1px solid #333;"
                        f"background:#1a1a1a;' "  # dark placeholder bg
                        f"title='idx={idx} | step_count={idx_step_count}'/>"
                        # Index label + source type badge + guidance indicator + step counter / auto-evict hint
                        f"<span style='font-size:9px;color:#666;display:flex;align-items:center;gap:2px;'>"
                        f"#{idx} "
                        f"<span style='color:{source_type_color};font-size:8px;' "
                        f"title='{source_type_title}'>{source_type_label}</span>"
                        f"{guidance_indicator_html}"
                        f"{auto_evict_hint_html}"
                        f"</span>"
                        # Per-sample action buttons (protect + guidance + evict)
                        # Guidance is per-index: uses the actual feature vector of this
                        # specific image as the guidance target, not a cluster centroid.
                        f"<div style='display:flex;gap:2px;'>"
                        f"<button class='fd-idx-protect' data-idx='{idx}' data-cluster='{k}' "
                        f"style='{_SAMPLE_BTN}border-color:{prot_color};"
                        f"background:{prot_bg};color:{prot_color};'"
                        f"title='{prot_title}'>{prot_label}</button>"
                        f"<button class='fd-idx-guidance' data-idx='{idx}' data-cluster='{k}' "
                        f"style='{_SAMPLE_BTN}border-color:#555;"
                        f"background:transparent;color:#888;'"
                        f"title='Set guidance target from this specific image'>🎯</button>"
                        f"<button class='fd-idx-evict' data-idx='{idx}' data-cluster='{k}' "
                        f"style='{_SAMPLE_BTN}border-color:{evict_color};"
                        f"background:{evict_bg};color:{evict_color};'"
                        f"title='{evict_title}'>{evict_label}</button>"
                        f"</div>"
                        f"</div>"
                    )
                lines.append("</div>")

            lines.append("</div></details>")

        lines.append("</div>")

        lines.append(self._render_panel_javascript())
        return "".join(lines)

    def _render_guidance_banner(self) -> list[str]:
        """Render an explanation banner when guidance mode is active."""
        if self._guidance_target is None or self.eviction_mode != "guided":
            return []
        return [
            "<div style='margin-bottom:8px;padding:8px 12px;"
            "background:rgba(96,165,250,0.12);border:1px solid #60a5fa;"
            "border-radius:4px;font-family:sans-serif;font-size:12px;color:#93c5fd;'>"
            "🎯 <b>Guidance Mode Active</b> — The queue is using <b>guided eviction</b>, "
            "preferentially retaining features similar to the guidance target "
            "and evicting features dissimilar to it. "
            "Guidance is <b>per-image</b> — click the 🎯 button on any individual "
            "image to change the target, or switch the eviction mode to <b>fifo</b> or "
            "<b>diversity</b> to disable guidance. "
            "The <b>Guidance Target</b> section below shows the top features closest to the target."
            "</div>"
        ]

    def _render_panel_javascript(self) -> str:
        """Return inline JS for button forwarding and toast notifications.

        NOTE: In Gradio 5+, ``<script>`` tags inside ``gr.HTML`` are stripped.
        This JS is injected via the ``js`` parameter on Gradio component events
        in ``scripts/traintrain.py`` instead.  This method returns a no-op
        placeholder for backward compatibility.
        """
        return ""

    @torch.no_grad()
    def get_guidance_candidates(self, n: int = 10) -> torch.Tensor:
        """Return the ``n`` features closest to the guidance target.

        Useful for showing the artist what the queue currently considers
        "good" features.  Returns a [n, feat_dim] tensor.
        """
        if self._guidance_target is None:
            return self.feats[:min(n, self.size)].clone()

        target = self._guidance_target.to(self.feats.device, dtype=self.feats.dtype)
        target_norm = F.normalize(target.unsqueeze(0), dim=1)
        feats_norm = F.normalize(self.feats, dim=1)
        sim = feats_norm @ target_norm.T  # [N, 1]
        _, idx = torch.topk(sim[:, 0], k=min(n, self.size), largest=True)
        return self.feats[idx].clone()


# =============================================================================
# Frozen Feature Extractors
# =============================================================================

def _preprocess(x, mean, std, target_size=None):
    """[0,1] float -> resize to target_size -> ImageNet normalize."""
    if target_size is not None and (x.shape[-2] != target_size or x.shape[-1] != target_size):
        x = F.interpolate(x, size=(target_size, target_size), mode="bicubic",
                          align_corners=False, antialias=True)
    return (x - mean) / std


class TimmReprModel(nn.Module):
    """Wraps a timm model as a frozen feature extractor.

    Handles preprocessing: [0, 1] -> resize -> ImageNet normalize.
    Returns ``(cls_token, mean_token)``.
    """

    def __init__(self, model_name: str, device="cuda", target_size: Optional[int] = None):
        super().__init__()
        import timm
        from timm.data import resolve_data_config

        logger.info(f"[TimmReprModel] Loading model: {model_name}")
        kwargs = dict(pretrained=True, num_classes=0)
        try:
            self.model = timm.create_model(model_name, dynamic_img_size=True, dynamic_img_pad=True, **kwargs)
        except TypeError:
            self.model = timm.create_model(model_name, **kwargs)
        self.model.to(device).eval().requires_grad_(False)
        self.num_prefix_tokens = getattr(self.model, "num_prefix_tokens", 0)
        self.has_attn_pool = hasattr(self.model, "attn_pool") and self.model.attn_pool is not None
        self.feat_dim = self.model.num_features

        data_cfg = resolve_data_config(self.model.pretrained_cfg)
        native_size = data_cfg["input_size"][-1]
        if "naflex" in model_name.lower():
            native_size = 256
        if target_size is not None and target_size != native_size:
            self.target_size = target_size
            logger.info(f"[TimmReprModel] Overriding target_size: {native_size} -> {target_size}")
        else:
            self.target_size = native_size

        mean = torch.tensor(data_cfg["mean"], device=device).view(1, 3, 1, 1)
        std = torch.tensor(data_cfg["std"], device=device).view(1, 3, 1, 1)
        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

        interpolation = data_cfg.get("interpolation", "bicubic")
        logger.info(
            f"[TimmReprModel] {model_name}: feat_dim={self.feat_dim}, "
            f"target_size={self.target_size}, interpolation={interpolation}, "
            f"mean={data_cfg['mean']}, std={data_cfg['std']}"
        )

    def forward(self, x: torch.Tensor):
        x = _preprocess(x, self.mean, self.std, self.target_size)
        feats = self.model.forward_features(x)
        # CNN models return (B, C, H, W); pool spatially
        if feats.ndim == 4:
            cls_token = feats.mean(dim=[2, 3])
            return cls_token, None
        # ViT models return (B, N, C)
        patch_tokens = feats[:, self.num_prefix_tokens:]
        mean_token = patch_tokens.mean(1)
        if self.num_prefix_tokens > 0:
            cls_token = feats[:, 0]
        elif self.has_attn_pool:
            pool = getattr(self.model, "pool", None) or getattr(self.model, "_pool", None)
            cls_token = pool(feats)
        else:
            cls_token = mean_token
        return cls_token, mean_token


def load_repr_model(name: str, device="cuda", target_size: Optional[int] = None):
    """Load a representation feature extractor.

    Each model handles its own input resolution internally based on its
    training configuration (e.g. timm models use ``pretrained_cfg['input_size']``).

    Args:
        name: ``'inception'``, ``'convnext'``, or any timm model name.
              Common aliases are mapped automatically:
              - ``'dinov2_vitb14'`` -> ``'vit_base_patch14_dinov2'``
              - ``'dinov2_vitl14'`` -> ``'vit_large_patch14_dinov2'``
              - ``'dinov2_vitg14'`` -> ``'vit_giant_patch14_dinov2'``
              - ``'clip_vitl'``     -> ``'vit_large_patch14_clip_224.openai'``
        target_size: override the model's native target resolution.

    Returns:
        (model, feat_dim, has_logits, target_size)
    """
    # Common alias mapping for user convenience
    _ALIASES = {
        "dinov2_vitb14": "vit_base_patch14_dinov2",
        "dinov2_vitl14": "vit_large_patch14_dinov2",
        "dinov2_vitg14": "vit_giant_patch14_dinov2",
        "dinov2_vits14": "vit_small_patch14_dinov2",
        "clip_vitl": "vit_large_patch14_clip_224.openai",
        "clip_vitb": "vit_base_patch16_clip_224",
        "clip_vith": "vit_huge_patch14_clip_224",
    }
    name = _ALIASES.get(name, name)

    if name == "inception":
        # InceptionV3 from perception_util — simplified inline version
        from utils.perception_util import load_inception
        net = load_inception(device=device, normalize=False)
        return net, 2048, True, 299
    elif name == "convnext":
        net = TimmReprModel("convnextv2_base.fcmae_ft_in22k_in1k", device=device, target_size=224)
        return net, net.feat_dim, False, net.target_size
    else:
        net = TimmReprModel(name, device=device, target_size=target_size)
        return net, net.feat_dim, False, net.target_size


def model_short_name(name: str) -> str:
    """Derive a concise label from a representation model name for logging/metrics."""
    if name in ("inception", "convnext"):
        return name
    low = name.lower()
    if "naflex" in low:
        return "naflex_siglip"
    for keyword in ("dinov2", "dino", "mae", "clip", "siglip"):
        if keyword in low:
            return keyword
    return name.split(".")[0].replace("_", "-")


# =============================================================================
# Judge system
# =============================================================================

def infer_stats_path(name, img_size, target_size):
    """Auto-infer reference stats path for a repr model.

    Uses the *resolved* model name (after alias mapping) for the filename.
    """
    # Apply the same alias mapping as load_repr_model
    _ALIASES = {
        "dinov2_vitb14": "vit_base_patch14_dinov2",
        "dinov2_vitl14": "vit_large_patch14_dinov2",
        "dinov2_vitg14": "vit_giant_patch14_dinov2",
        "dinov2_vits14": "vit_small_patch14_dinov2",
        "clip_vitl": "vit_large_patch14_clip_224.openai",
        "clip_vitb": "vit_base_patch16_clip_224",
        "clip_vith": "vit_huge_patch14_clip_224",
    }
    resolved = _ALIASES.get(name, name)
    sanitized = resolved.replace(".", "_")
    if img_size == 512:
        img_size = 256
    return f"data/fid_stats/{sanitized}_in{img_size}_t{target_size}_stats.npz"


def extract_judge_features(judge, images):
    """Run judge model and return features respecting pool_type.

    TimmReprModel returns (cls_token, mean_token).  When pool_type=='avg',
    we want the mean_token as features.  Inception and CNN models use primary
    features.
    """
    primary, secondary = judge["model"](images)
    if judge.get("pool_type") == "avg":
        return secondary
    return primary


# =============================================================================
# FD-Loss Manager — orchestrates judges, queues, and loss computation
# =============================================================================

class FDLossManager(nn.Module):
    """Manages FD-Loss judges, feature queues, and loss computation.

    This is the main entry point for integrating FD-Loss into the Anima
    training pipeline.  It handles:

    - Loading frozen feature extractors (judges)
    - Loading reference statistics
    - Maintaining feature queues for each judge
    - Computing the differentiable FID loss each training step
    - Enqueuing new features after each step

    Usage::

        fd_loss = FDLossManager(
            repr_models=["dinov2_vitb14", "convnext"],
            queue_size=50000,
            ...
        )
        fd_loss.to(device)

        # In training loop:
        pixels = vae.decode_to_pixels(latents)  # [-1, 1]
        pixels = pixels * 0.5 + 0.5  # -> [0, 1]
        fd_loss.compute_loss(pixels)
    """

    def __init__(
        self,
        repr_models: list[str] = ("dinov2_vitb14",),
        queue_size: int = 50000,
        queue_mode: str = "online_accum",  # "snapshot", "online_accum", "ema"
        ema_beta: float = 0.9999,
        target_sizes: Optional[list[int]] = None,
        stats_paths: Optional[list[str]] = None,
        weights: Optional[list[float]] = None,
        pool_types: Optional[list[str]] = None,
        fid_norm_eps: float = 1e-6,
        device: str = "cuda",
        # --- New: intelligent queue management ---
        eviction_mode: Literal["fifo", "diversity", "guided"] = "fifo",
        guidance_strength: float = 0.5,
        n_clusters: int = 20,
        # --- Visual interactive cluster panel ---
        store_source_images: bool = False,
        source_img_size: int = 256,
        # --- Synthetic feature control ---
        # When False, generated (source_type=1) features are NOT enqueued.
        # The queue is populated only from prefill (real data) and remains
        # static during training.  This avoids the feedback loop where the
        # model trains against its own previous outputs.
        enqueue_generated: bool = True,
    ):
        super().__init__()
        self.fid_norm_eps = fid_norm_eps
        self.device = device
        self.eviction_mode = eviction_mode
        self.guidance_strength = guidance_strength
        self.n_clusters = n_clusters
        self.store_source_images = store_source_images
        self.source_img_size = source_img_size
        self.enqueue_generated = enqueue_generated
        self.judges = []  # plain list; sub-modules registered via _modules dict
        self._judge_models = nn.ModuleList()   # for proper device placement
        self._judge_queues = nn.ModuleList()   # for proper device placement
        # Pending actions buffer — staged changes applied on Commit.
        # Each entry: (action_type, value) where action_type is one of:
        #   "protect_cluster", "guidance_cluster", "clear_guidance",
        #   "toggle_evict_cluster", "protect_idx", "toggle_evict_idx"
        self.pending_actions: list = []

        num = len(repr_models)

        # Resolve per-model args
        if target_sizes is None:
            target_sizes = [256] * num
        elif len(target_sizes) == 1 and num > 1:
            target_sizes *= num

        if stats_paths is None:
            stats_paths = [
                infer_stats_path(n, 256, ts)
                for n, ts in zip(repr_models, target_sizes)
            ]
        elif len(stats_paths) == 1 and num > 1:
            stats_paths *= num

        if weights is None:
            weights = [1.0] * num

        if pool_types is None:
            pool_types = ["cls"] * num
        elif len(pool_types) == 1 and num > 1:
            pool_types *= num

        assert len(target_sizes) == num
        assert len(stats_paths) == num
        assert len(weights) == num
        assert len(pool_types) == num

        online_accum = queue_mode == "online_accum"
        ema_stats = queue_mode == "ema"

        for i, name in enumerate(repr_models):
            logger.info(f"[FDLossManager] Loading judge {i + 1}/{num}: {name}")

            # Load feature extractor
            model, feat_dim, has_logits, target_size = load_repr_model(
                name, device=device, target_size=target_sizes[i]
            )

            # Load reference statistics (optional — falls back to self-referential)
            mu_ref, sigma_ref = load_mu_and_sigma_reference(stats_paths[i], pool_types[i])
            sigma_ref_sqrt = precompute_sigma_ref_sqrt(sigma_ref) if sigma_ref is not None else None
            has_ref_stats = mu_ref is not None

            # Create feature queue with intelligent eviction
            queue = FeatureQueue(
                size=queue_size,
                feat_dim=feat_dim,
                online_accum=online_accum,
                ema_beta=ema_beta if ema_stats else 0.0,
                eviction_mode=eviction_mode,
                guidance_strength=guidance_strength,
                n_clusters=n_clusters,
                store_source_images=self.store_source_images,
                source_img_size=self.source_img_size,
            )

            judge = dict(
                name=name,
                model=model,
                feat_dim=feat_dim,
                queue=queue,
                mu_ref=mu_ref,
                sigma_ref=sigma_ref,
                sigma_ref_sqrt=sigma_ref_sqrt,
                has_ref_stats=has_ref_stats,
                weight=weights[i],
                pool_type=pool_types[i],
                has_logits=has_logits,
            )
            self.judges.append(judge)
            self._judge_models.append(model)
            self._judge_queues.append(queue)

        # Move all queues and models to the target device.
        # Without this, FeatureQueue buffers (feats, feat_sum_old, etc.) stay on CPU
        # since they are created via torch.empty/zeros without an explicit device,
        # causing a device mismatch when build_feats_stats mixes them with CUDA features.
        self.to(self.device)

        logger.info(
            f"[FDLossManager] Initialized {len(self.judges)} judges: "
            f"{[j['name'] for j in self.judges]}, "
            f"queue_size={queue_size}, queue_mode={queue_mode}, "
            f"eviction={eviction_mode}"
        )

    def compute_loss(self, pixels: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Compute FD-Loss from decoded pixel images.

        When precomputed reference statistics are available, the FID is
        computed between generated features and the reference distribution.
        Otherwise, the queue's own accumulated statistics serve as the
        reference (self-referential mode).

        Args:
            pixels: Tensor in [0, 1] range, shape [B, C, H, W].

        Returns:
            (loss, loss_dict) where loss is a scalar tensor with gradients,
            and loss_dict contains per-judge FID values for logging.
        """
        loss = torch.tensor(0.0, device=pixels.device)
        loss_dict = {}

        # Extract features from all judges
        all_new_feats = []
        for judge in self.judges:
            feats = extract_judge_features(judge, pixels)
            all_new_feats.append(feats)

        # Compute FID loss for each judge
        for i, judge in enumerate(self.judges):
            new_feats = all_new_feats[i]
            queue = judge["queue"]

            # Determine reference statistics
            if judge.get("has_ref_stats", False):
                mu_ref = judge["mu_ref"]
                sigma_ref = judge["sigma_ref"]
                sigma_ref_sqrt = judge.get("sigma_ref_sqrt")
            else:
                mu_ref, sigma_ref, sigma_ref_sqrt = queue.get_ref_stats()

            _ns_kwargs = dict(sigma_ref_sqrt=sigma_ref_sqrt)
            if queue.online_accum or queue.ema_stats:
                mu, sigma = queue.build_feats_stats(new_feats)
                fid = compute_frechet_distance_loss(
                    mu_ref, sigma_ref,
                    mu=mu, sigma=sigma,
                    **_ns_kwargs,
                )
            else:
                all_feats = queue.build_feats_snapshot(new_feats)
                fid = compute_frechet_distance_loss(
                    mu_ref, sigma_ref,
                    all_feats=all_feats,
                    **_ns_kwargs,
                )

            # Normalize FID to prevent extreme values from dominating
            fid_loss = fid / (fid.detach() + self.fid_norm_eps)
            loss = loss + judge["weight"] * fid_loss
            loss_dict[f"fid_{judge['name']}"] = float(fid.detach())

        return loss, loss_dict

    @torch.no_grad()
    def enqueue_features(self, pixels: torch.Tensor,
                         source_images: Optional[torch.Tensor] = None,
                         source_type: Optional[int] = None):
        """Enqueue features from generated images (no grad).

        Call this after each training step to keep the queue updated.

        When ``source_images`` is provided and the queue has
        ``store_source_images=True``, the pixel thumbnails are stored alongside
        features for the visual cluster panel.

        ``source_type`` indicates the origin of the features:
        - ``0``: dataset / real training images
        - ``1``: machine-generated (model output during training)

        When ``self.enqueue_generated`` is ``False`` and ``source_type == 1``,
        this is a no-op — the queue is not updated with generated features.
        This avoids the feedback loop where the model trains against its own
        previous outputs, keeping the queue as a static reference of real data.
        """
        if not self.enqueue_generated and source_type == 1:
            return
        for judge in self.judges:
            feats = extract_judge_features(judge, pixels)
            judge["queue"].enqueue(feats.detach(), source_img=source_images,
                                   source_type=source_type)

    # -- Active sample count ----------------------------------------------------
    @torch.no_grad()
    def get_active_count(self, judge_name: Optional[str] = None) -> dict:
        """Return a dict with active sample counts per judge queue.

        The active count is the number of non-evicted slots::

            active = n_type0 + n_type1   (= queue_size - n_type2)

        Where:
          - ``n_type0`` = number of features with ``source_type == 0`` (dataset / real)
          - ``n_type1`` = number of features with ``source_type == 1`` (generated / model output)
          - ``n_type2`` = number of features with ``source_type == 2`` (evicted / noise)

        This naturally handles the ``enqueue_generated`` flag: if it is ``False``,
        no type-1 features are ever enqueued, so ``n_type1`` is zero.

        Returns:
            A dict mapping judge name to a dict with keys:
            ``queue_size``, ``n_type0``, ``n_type1``, ``n_type2``, ``active``.
        """
        judge = self._resolve_judge(judge_name)
        if judge is None:
            if judge_name is not None:
                return {}
            # Aggregate across all judges
            result = {}
            for j in self.judges:
                result[j["name"]] = self.get_active_count(j["name"]).get(j["name"], {})
            return result

        queue = judge["queue"]
        queue_size = queue.size
        n_type0 = 0
        n_type1 = 0
        n_type2 = 0
        if queue._source_type is not None:
            st = queue._source_type
            n_type0 = int((st == 0).sum().item())
            n_type1 = int((st == 1).sum().item())
            n_type2 = int((st == 2).sum().item())
        # Active = all non-evicted slots.  Evicted/noise slots (type 2) are
        # dead slots waiting to be overwritten — they are NOT subtracted from
        # the active count because they are simply not counted as active.
        active = n_type0 + n_type1
        return {
            judge["name"]: {
                "queue_size": queue_size,
                "n_type0": n_type0,
                "n_type1": n_type1,
                "n_type2": n_type2,
                "active": active,
            }
        }

    # -- Clustering & visualisation (inter-epoch inspection) -------------------

    @torch.no_grad()
    def get_clusters(self, judge_name: Optional[str] = None,
                     n_clusters: Optional[int] = None) -> dict:
        """Run k-means on the queue of a specific judge (or the first judge).

        Returns a dict with centroids, assignments, sizes, inertia, silhouette.
        See ``FeatureQueue.get_clusters()`` for details.

        Args:
            judge_name: Name of the judge to cluster (e.g. ``'vit_base_patch14_dinov2'``).
                        If None, uses the first judge.
            n_clusters: Number of clusters (default: ``self.n_clusters``).
        """
        judge = self._resolve_judge(judge_name)
        if judge is None:
            return {}
        return judge["queue"].get_clusters(n_clusters=n_clusters or self.n_clusters)

    @torch.no_grad()
    def get_cluster_summary_html(self, judge_name: Optional[str] = None,
                                 n_clusters: Optional[int] = None) -> str:
        """Return an HTML snippet summarising cluster structure for a judge.

        Useful for rendering in a Gradio HTML component between epochs.
        """
        judge = self._resolve_judge(judge_name)
        if judge is None:
            return "<div style='color:#888;'>No judges available</div>"
        return judge["queue"].get_cluster_summary_html(n_clusters=n_clusters or self.n_clusters)

    def _resolve_judge(self, judge_name: Optional[str] = None) -> Optional[dict]:
        """Resolve a judge by name, or return the first judge."""
        if judge_name is not None:
            for j in self.judges:
                if j["name"] == judge_name:
                    return j
            logger.warning(f"[FDLossManager] Judge '{judge_name}' not found")
            return None
        return self.judges[0] if self.judges else None

    # -- Interactive guidance --------------------------------------------------

    def set_guidance_target(self, target_features: torch.Tensor,
                            judge_name: Optional[str] = None):
        """Set the guidance target for a specific judge's queue.

        When ``eviction_mode='guided'``, the queue will preferentially retain
        features similar to this target and evict features dissimilar to it.

        Args:
            target_features: A feature vector [feat_dim] or batch [B, feat_dim]
                             that represents the desired region of feature space.
                             If a batch is provided, the mean is used.
            judge_name: Name of the judge to guide. If None, applies to all judges.
        """
        if target_features.dim() > 1:
            target = target_features.mean(dim=0)
        else:
            target = target_features

        if judge_name is not None:
            judge = self._resolve_judge(judge_name)
            if judge is not None:
                judge["queue"].guidance_target = target
                logger.info(f"[FDLossManager] Guidance target set for judge '{judge_name}'")
        else:
            for judge in self.judges:
                judge["queue"].guidance_target = target
            logger.info(f"[FDLossManager] Guidance target set for all judges")

    def clear_guidance_target(self, judge_name: Optional[str] = None):
        """Clear the guidance target, reverting to unguided eviction."""
        if judge_name is not None:
            judge = self._resolve_judge(judge_name)
            if judge is not None:
                judge["queue"].guidance_target = None
        else:
            for judge in self.judges:
                judge["queue"].guidance_target = None
        logger.info(f"[FDLossManager] Guidance target cleared")

    def set_eviction_mode(self, mode: Literal["fifo", "diversity", "guided"],
                          judge_name: Optional[str] = None):
        """Change the eviction mode for a specific judge (or all judges).

        Args:
            mode: ``'fifo'`` (oldest out), ``'diversity'`` (evict redundant),
                  or ``'guided'`` (evict far from guidance target).
            judge_name: Name of the judge, or None for all.
        """
        if judge_name is not None:
            judge = self._resolve_judge(judge_name)
            if judge is not None:
                judge["queue"].eviction_mode = mode
        else:
            for judge in self.judges:
                judge["queue"].eviction_mode = mode
        logger.info(f"[FDLossManager] Eviction mode set to '{mode}'")

    @torch.no_grad()
    def get_guidance_candidates(self, n: int = 10,
                                judge_name: Optional[str] = None) -> torch.Tensor:
        """Return the ``n`` features closest to the guidance target.

        Useful for showing the artist what the queue currently considers
        "good" features.  Returns a [n, feat_dim] tensor.
        """
        judge = self._resolve_judge(judge_name)
        if judge is None:
            return torch.empty(0)
        return judge["queue"].get_guidance_candidates(n=n)

    # -- Visual interactive cluster panel --------------------------------------

    @torch.no_grad()
    def get_cluster_thumbnails(self, judge_name: Optional[str] = None,
                               n_clusters: Optional[int] = None,
                               max_per_cluster: int = 9) -> dict:
        """Return source image thumbnails grouped by cluster for a judge.

        Returns a dict mapping cluster_id -> list of (index, PIL Image).
        Requires ``store_source_images=True`` on the queue.
        """
        judge = self._resolve_judge(judge_name)
        if judge is None:
            return {}
        return judge["queue"].get_cluster_thumbnails(
            n_clusters=n_clusters or self.n_clusters,
            max_per_cluster=max_per_cluster,
        )

    @torch.no_grad()
    def render_interactive_cluster_panel(self, judge_name: Optional[str] = None,
                                         n_clusters: Optional[int] = None,
                                         max_per_cluster: int = 9) -> str:
        """Return an HTML interactive cluster panel for Gradio.

        Shows each cluster with thumbnails and action buttons (protect,
        unprotect, set as guidance, evict).
        """
        judge = self._resolve_judge(judge_name)
        if judge is None:
            return "<div style='color:#888;'>No judges available</div>"
        return judge["queue"].render_interactive_cluster_panel(
            n_clusters=n_clusters or self.n_clusters,
            max_per_cluster=max_per_cluster,
        )

    @torch.no_grad()
    def fetch_thumbnail_base64(self, queue_idx: int,
                               judge_name: Optional[str] = None) -> str:
        """Return a base64-encoded PNG data URI for a single queue index.

        Called on-demand when the user expands a cluster accordion in the UI.
        Delegates to the first judge's queue.
        """
        judge = self._resolve_judge(judge_name)
        if judge is None:
            return ""
        return judge["queue"].fetch_thumbnail_base64(queue_idx)

    # -- Staged actions (appended to pending_actions, applied on Commit) --------

    def _stage(self, action_type: str, value):
        """Append an action to the pending buffer."""
        self.pending_actions.append((action_type, value))
        logger.info(f"[FDLossManager] Staged {action_type}({value}) "
                     f"(total pending: {len(self.pending_actions)})")

    def _resolve_cluster_to_indices(self, cluster_id: int,
                                    judge_name: Optional[str] = None) -> torch.Tensor:
        """Resolve a cluster ID to individual queue indices using **live clustering**.

        Calls ``get_clusters()`` to get the current assignments, then finds all
        indices belonging to the given cluster.  This ensures actions are always
        associated with the **actual images** visible in the UI at click time,
        not a stale registry that may have shifted due to re-clustering.

        Returns:
            A 1-D long tensor of queue indices, or empty tensor if not found.
        """
        judge = self._resolve_judge(judge_name)
        if judge is None:
            return torch.empty(0, dtype=torch.long)
        queue = judge["queue"]
        clusters = queue.get_clusters(n_clusters=self.n_clusters)
        if not clusters or "assignments" not in clusters:
            return torch.empty(0, dtype=torch.long, device=queue.feats.device)
        assignments = clusters["assignments"]
        if cluster_id >= assignments.max().item() + 1:
            return torch.empty(0, dtype=torch.long, device=queue.feats.device)
        return (assignments == cluster_id).nonzero(as_tuple=True)[0]

    def protect_cluster(self, cluster_id: int, judge_name: Optional[str] = None,
                        n_clusters: Optional[int] = None):
        """Stage a protect-cluster action — **immediately resolved to per-index**.

        The cluster ID is resolved to individual queue indices using **live
        clustering** at click time, and per-index protect actions are staged.
        This ensures the action is associated with the actual images, not a
        cluster ID that may shift on re-clustering.
        """
        indices = self._resolve_cluster_to_indices(cluster_id, judge_name)
        if indices.numel() == 0:
            logger.warning(f"[FDLossManager] No indices found for cluster {cluster_id}, skipping")
            return
        self._stage("protect_indices", indices)

    def unprotect_cluster(self, cluster_id: int, judge_name: Optional[str] = None,
                          n_clusters: Optional[int] = None):
        """Stage an unprotect-cluster action — **immediately resolved to per-index**."""
        indices = self._resolve_cluster_to_indices(cluster_id, judge_name)
        if indices.numel() == 0:
            logger.warning(f"[FDLossManager] No indices found for cluster {cluster_id}, skipping")
            return
        self._stage("unprotect_indices", indices)

    def set_guidance_from_cluster(self, cluster_id: int,
                                  judge_name: Optional[str] = None,
                                  n_clusters: Optional[int] = None):
        """Stage a set-guidance action — **immediately resolved to per-index**.

        The cluster ID is resolved to individual queue indices using **live
        clustering** at click time, and a per-index guidance action is staged
        using the first index in the cluster.  This ensures the action is
        associated with the actual images, not a cluster ID that may shift
        on re-clustering.
        """
        indices = self._resolve_cluster_to_indices(cluster_id, judge_name)
        if indices.numel() == 0:
            logger.warning(f"[FDLossManager] No indices found for cluster {cluster_id}, skipping")
            return
        # Use the first index in the cluster as the guidance target.
        # All indices in a cluster are similar, so any one works.
        self._stage("guidance_idx", int(indices[0].item()))

    def set_guidance_from_index(self, idx: int,
                                judge_name: Optional[str] = None):
        """Stage a set-guidance action from a specific queue index (applied on Commit).

        Unlike cluster-level guidance (which uses the cluster centroid), this
        uses the **actual feature vector** of the selected image as the guidance
        target.  This gives the user precise control over which visual style
        to guide toward.
        """
        self._stage("guidance_idx", idx)

    def clear_guidance_target(self, judge_name: Optional[str] = None):
        """Stage a clear-guidance action (applied on Commit)."""
        self._stage("clear_guidance", None)

    def protect_index(self, idx: int, judge_name: Optional[str] = None):
        """Stage a protect-index action (applied on Commit)."""
        self._stage("protect_idx", idx)

    def unprotect_index(self, idx: int, judge_name: Optional[str] = None):
        """Stage an unprotect-index action (applied on Commit)."""
        self._stage("unprotect_idx", idx)

    def evict_index(self, idx: int, judge_name: Optional[str] = None):
        """Stage an evict-index action (applied on Commit)."""
        self._stage("toggle_evict_idx", idx)

    def evict_cluster(self, cluster_id: int, judge_name: Optional[str] = None,
                      n_clusters: Optional[int] = None):
        """Stage an evict-cluster action — **immediately resolved to per-index**.

        The cluster ID is resolved to individual queue indices using **live
        clustering** at click time, and per-index eviction toggle actions are
        staged.  This ensures the action is associated with the actual images,
        not a cluster ID that may shift on re-clustering.
        """
        indices = self._resolve_cluster_to_indices(cluster_id, judge_name)
        if indices.numel() == 0:
            logger.warning(f"[FDLossManager] No indices found for cluster {cluster_id}, skipping")
            return
        self._stage("toggle_evict_indices", indices)

    # -- Commit: replay all staged actions ------------------------------------

    def commit_pending_actions(self, judge_name: Optional[str] = None) -> int:
        """Execute all staged actions and clear the buffer.

        All actions in the pending buffer are now **per-index** — cluster-level
        actions are resolved to individual indices at stage time (see
        :meth:`protect_cluster`, :meth:`evict_cluster`), so no re-resolution
        is needed here.  This ensures actions are always associated with the
        actual images the user clicked on, not cluster IDs that may shift on
        re-clustering.

        After processing all staged actions, any features marked for eviction
        are **actually purged** via ``queue.commit_evictions()`` — they are
        replaced with random noise, their source images zeroed, and the
        cluster cache invalidated so the next refresh re-clusters cleanly.

        Returns the number of actions committed.
        """
        if not self.pending_actions:
            return 0
        judge = self._resolve_judge(judge_name)
        if judge is None:
            return 0
        queue = judge["queue"]
        n = len(self.pending_actions)
        logger.info(f"[FDLossManager] Committing {n} pending action(s)...")

        # All actions are now per-index (cluster actions resolved at stage time).
        # We apply them directly without any re-resolution.
        for action_type, value in self.pending_actions:
            try:
                if action_type == "protect_indices":
                    queue.protect_indices(value)
                elif action_type == "unprotect_indices":
                    queue.unprotect_indices(value)
                elif action_type == "toggle_evict_indices":
                    # value is a tensor of indices; check if ANY are already marked
                    if queue.pending_eviction_mask[value].any():
                        queue.clear_eviction_marks(value)
                    else:
                        queue.mark_eviction_indices(value)
                elif action_type == "protect_idx":
                    queue.protect_indices(torch.tensor([value], device=queue.feats.device))
                elif action_type == "unprotect_idx":
                    queue.unprotect_indices(torch.tensor([value], device=queue.feats.device))
                elif action_type == "toggle_evict_idx":
                    if queue.pending_eviction_mask[value].item():
                        queue.clear_eviction_marks(torch.tensor([value], device=queue.feats.device))
                    else:
                        queue.mark_eviction_indices(torch.tensor([value], device=queue.feats.device))
                elif action_type == "guidance_idx":
                    # Use the actual feature vector of the selected index as target
                    if 0 <= value < queue.size:
                        feat = queue.feats[value]  # [D]
                        queue.guidance_target = feat
                        if queue.eviction_mode != "guided":
                            queue.eviction_mode = "guided"
                elif action_type == "clear_guidance":
                    queue.guidance_target = None
                else:
                    logger.warning(f"[FDLossManager] Unknown action type: {action_type}")
            except Exception as e:
                logger.warning(f"[FDLossManager] Error applying {action_type}: {e}")

        # --- Actually purge evicted features ---
        n_evicted = queue.commit_evictions()
        if n_evicted > 0:
            logger.info(f"[FDLossManager] Purged {n_evicted} evicted features from queue")

        self.pending_actions.clear()
        logger.info(f"[FDLossManager] Committed {n} action(s)")
        return n

    # -- Effective pending count (resolves cluster-level actions) -------------

    def _get_effective_pending_count(self, queue) -> int:
        """Return the number of **individual features** affected by staged actions.

        Since all actions are now per-index (cluster actions resolved at stage
        time), this simply counts the number of indices in each action's value
        tensor, giving a meaningful count for the Commit button label.

        Args:
            queue: The ``FeatureQueue`` instance (unused, kept for API compat).
        """
        count = 0
        for action_type, value in self.pending_actions:
            if action_type in ("protect_indices", "unprotect_indices", "toggle_evict_indices"):
                count += max(1, value.numel() if hasattr(value, 'numel') else 1)
            else:
                count += 1
        return count

    # -- Projected state helpers (for rendering with pending actions) ----------

    def _get_projected_protected_mask(self, queue) -> torch.Tensor:
        """Return the protected_mask as it would be after all staged actions."""
        mask = queue.protected_mask.clone()
        for action_type, value in self.pending_actions:
            if action_type == "protect_indices":
                mask[value] = True
            elif action_type == "unprotect_indices":
                mask[value] = False
            elif action_type == "protect_idx":
                if 0 <= value < mask.shape[0]:
                    mask[value] = True
            elif action_type == "unprotect_idx":
                if 0 <= value < mask.shape[0]:
                    mask[value] = False
        return mask

    def _get_projected_eviction_mask(self, queue) -> torch.Tensor:
        """Return the pending_eviction_mask as it would be after all staged actions."""
        mask = queue.pending_eviction_mask.clone()
        for action_type, value in self.pending_actions:
            if action_type == "toggle_evict_indices":
                # value is a tensor of indices; toggle all of them
                if mask[value].any():
                    mask[value] = False
                else:
                    mask[value] = True
            elif action_type == "toggle_evict_idx":
                if 0 <= value < mask.shape[0]:
                    mask[value] = not mask[value].item()
        return mask

    def _get_projected_guidance_target(self, queue) -> Optional[torch.Tensor]:
        """Return the guidance target feature vector as it would be after staged actions.

        Guidance is 100% per-index — the target is always an actual feature vector
        from the queue, never a cluster centroid.  This method replays staged
        ``guidance_idx`` actions to compute the projected target.
        """
        # Start from current guidance state
        current_target = queue._guidance_target

        projected_target = current_target
        for action_type, value in self.pending_actions:
            if action_type == "clear_guidance":
                projected_target = None
            elif action_type == "guidance_idx":
                # Per-index guidance: use the actual feature vector of the
                # selected index as the guidance target.
                if 0 <= value < queue.size:
                    projected_target = queue.feats[value].clone()
        return projected_target

    # -- Override render to use projected state --------------------------------

    def render_interactive_cluster_panel(self, judge_name: Optional[str] = None,
                                         n_clusters: Optional[int] = None,
                                         max_per_cluster: int = 9) -> str:
        """Return an HTML interactive cluster panel for Gradio.

        Shows each cluster with thumbnails and action buttons (protect,
        unprotect, set as guidance, evict).  The display reflects both
        the actual queue state AND any staged (pending) actions.

        Guidance is 100% per-index — the projected guidance target is the
        actual feature vector from the staged ``guidance_idx`` action, never
        a cluster centroid.
        """
        judge = self._resolve_judge(judge_name)
        if judge is None:
            return "<div style='color:#888;'>No judges available</div>"
        queue = judge["queue"]
        # Compute projected state for rendering
        projected_protected = self._get_projected_protected_mask(queue)
        projected_eviction = self._get_projected_eviction_mask(queue)
        projected_guidance_target = self._get_projected_guidance_target(queue)
        # Temporarily swap in projected state for rendering
        orig_protected = queue.protected_mask
        orig_eviction = queue.pending_eviction_mask
        orig_guidance = queue._guidance_target
        orig_mode = queue.eviction_mode
        try:
            queue.protected_mask = projected_protected
            queue.pending_eviction_mask = projected_eviction
            if projected_guidance_target is not None:
                # Use the actual feature vector directly — never a centroid.
                # This ensures the guidance pseudo-cluster and guided eviction
                # use the exact image the user selected, not a cluster average.
                queue._guidance_target = projected_guidance_target
                queue.eviction_mode = "guided"
            else:
                queue._guidance_target = None
            effective_count = self._get_effective_pending_count(queue)
            return queue.render_interactive_cluster_panel(
                n_clusters=n_clusters or self.n_clusters,
                max_per_cluster=max_per_cluster,
                n_pending_actions=effective_count,
                force_cluster=True,  # non-deterministic on refresh
            )
        finally:
            queue.protected_mask = orig_protected
            queue.pending_eviction_mask = orig_eviction
            queue._guidance_target = orig_guidance
            queue.eviction_mode = orig_mode

    @torch.no_grad()
    def fill_queues_from_data(self, dataloader, num_samples: Optional[int] = None):
        """Fill feature queues using real training images.

        This should be called before training starts to initialize the queues
        with real data features.  The dataloader should yield batches with
        a "latent" key containing VAE latents.

        Args:
            dataloader: Iterable yielding dicts with "latent" key.
            num_samples: Number of samples to use for queue fill (default: queue_size).
        """
        if num_samples is None:
            num_samples = max(j["queue"].size for j in self.judges)

        filled = 0
        logger.info(f"[FDLossManager] Filling queues with {num_samples} real data features...")

        for batch in dataloader:
            if filled >= num_samples:
                break

            latents = batch["latent"].to(self.device)
            batch_size = latents.shape[0]
            remaining = num_samples - filled
            count = min(batch_size, remaining)

            # We need pixels for feature extraction — but we don't have a VAE decoder here.
            # Instead, we fill queues with generated features from the model.
            # For real data, we'd need the VAE decode.  This is a placeholder
            # that will be handled by the training loop.
            filled += count
            logger.info(f"[FDLossManager] Queue fill progress: {filled}/{num_samples}")

        logger.info(f"[FDLossManager] Queue fill complete: {filled} features")

    @torch.no_grad()
    def prefill_from_dataloader(self, dataloader, vae, num_samples: Optional[int] = None,
                                mask_key: Optional[str] = None):
        """Pre-fill feature queues with real data before training starts.

        Iterates the dataloader, decodes latents to pixels via the VAE,
        extracts features from each judge, and enqueues them.  This ensures
        the queue has enough samples for a non-degenerate covariance estimate
        from step 1 of training.

        When ``mask_key`` is provided (e.g. ``"mask"`` in texture mode), only the
        non-zero region of each sample's mask is cropped **in latent space** before
        VAE decode — the background noise canvas is never decoded, saving ~16×
        VAE compute (1024² canvas → 256² crop).  The feature queue then contains
        only actual texture content, not frequency-matched noise.

        The dataloader is iterated repeatedly until ``num_samples`` features have
        been enqueued.  In texture mode each pass produces different random crops
        from the same images, giving diverse features.  In full-res mode the same
        images repeat, which is fine for initialising the covariance estimate.

        VAE decode and feature extraction are the dominant costs; the per-batch
        crop extraction is vectorised (no Python loop over samples) to keep GPU
        utilisation high.

        Args:
            dataloader: Iterable yielding dicts with ``"latent"`` key.
            vae: VAE module with ``decode_to_pixels(latent) -> pixels``.
            num_samples: Number of samples to enqueue (default: queue_size).
            mask_key: Optional key in the batch dict for a latent-space mask
                      ``[B, H_lat, W_lat]`` that indicates the crop region.
        """
        if num_samples is None:
            num_samples = max(j["queue"].size for j in self.judges)

        filled = 0
        logger.info(f"[FDLossManager] Pre-filling queues with {num_samples} real data features...")

        from tqdm import tqdm
        pbar = tqdm(total=num_samples, desc="FD pre-fill", unit="img")

        # Iterate the dataloader repeatedly until we hit num_samples.
        # Each pass yields different random crops in texture mode (__getitem__
        # re-samples crop position/size every call), so this naturally produces
        # diverse features even from a small image set.
        # We use iter(dataloader) explicitly so we can re-create the iterator
        # when it exhausts (ContinualRandomDataLoader raises StopIteration
        # after one full pass through all data).
        while filled < num_samples:
            data_iter = iter(dataloader)
            for batch in data_iter:
                if filled >= num_samples:
                    break

                latents = batch["latent"].to(self.device)
                batch_size = latents.shape[0]
                remaining = num_samples - filled
                count = min(batch_size, remaining)
                latents = latents[:count]

                # --- Texture mode: crop latents FIRST, then VAE decode only the crop ---
                # This avoids decoding the full noisy canvas (~1024² → ~256² = ~16× saving).
                if mask_key is not None and mask_key in batch and batch[mask_key] is not None:
                    mask_t = batch[mask_key][:count].to(self.device)  # [B, H_lat, W_lat]
                    B = mask_t.shape[0]

                    # Vectorised per-sample bounding box in latent space (cumsum trick)
                    rows_any = (mask_t > 0.5).any(dim=2)   # [B, H_lat]
                    cols_any = (mask_t > 0.5).any(dim=1)   # [B, W_lat]
                    rows_cs = rows_any.cumsum(dim=1)
                    rows_cs_rev = rows_any.flip(dims=[1]).cumsum(dim=1).flip(dims=[1])
                    cols_cs = cols_any.cumsum(dim=1)
                    cols_cs_rev = cols_any.flip(dims=[1]).cumsum(dim=1).flip(dims=[1])

                    has_mask = rows_any.any(dim=1) & cols_any.any(dim=1)  # [B]
                    y1 = ((rows_cs == 1) & rows_any).int().argmax(dim=1)   # [B]
                    y2 = ((rows_cs_rev == 1) & rows_any).int().argmax(dim=1) + 1  # [B]
                    x1 = ((cols_cs == 1) & cols_any).int().argmax(dim=1)   # [B]
                    x2 = ((cols_cs_rev == 1) & cols_any).int().argmax(dim=1) + 1  # [B]

                    no_mask = ~has_mask
                    if no_mask.any():
                        y1[no_mask] = 0
                        y2[no_mask] = latents.shape[2]
                        x1[no_mask] = 0
                        x2[no_mask] = latents.shape[3]

                    # Crop latents in latent space — VAE decode only the crop region
                    cropped_latents = torch.stack([
                        latents[b, :, y1[b]:y2[b], x1[b]:x2[b]]
                        for b in range(B)
                    ], dim=0)

                    # VAE decode the cropped latents (small region, ~16× less work)
                    pixels = vae.decode_to_pixels(cropped_latents.float())
                else:
                    # Full-res mode: decode the whole latent as-is
                    pixels = vae.decode_to_pixels(latents.float())

                # decode_to_pixels returns [-1, 1]; FD-Loss expects [0, 1]
                pixels = pixels * 0.5 + 0.5

                # Pass source images for the visual cluster panel if enabled
                # source_type=0 marks these as dataset (real) images
                self.enqueue_features(pixels, source_images=pixels, source_type=0)

                filled += count
                pbar.update(count)

        pbar.close()
        logger.info(f"[FDLossManager] Pre-fill complete: {filled} features enqueued")

    def state_dict(self):
        """Collect queue state dicts from all judges for checkpointing."""
        return {j["name"]: j["queue"].state_dict() for j in self.judges}

    def load_state_dict(self, state_dict, strict=True):
        """Restore queue states from checkpoint into judges."""
        loaded = 0
        for judge in self.judges:
            if judge["name"] in state_dict:
                judge["queue"].load_state_dict(state_dict[judge["name"]])
                judge["queue"].to(self.device)
                loaded += 1
                logger.info(f"[FDLossManager] Restored queue state for '{judge['name']}'")
            else:
                logger.warning(f"[FDLossManager] No saved queue state for '{judge['name']}'")
        return loaded == len(self.judges)
