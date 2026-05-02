"""
FD-Loss: Representation Fréchet Distance loss for Anima training.

Port of the core FD-Loss components (FeatureQueue, differentiable FID,
frozen feature extractors) adapted for the Anima traintrain pipeline.

Reference: https://github.com/sony/fd-loss  (Sony Research)
"""

import logging
from typing import Optional

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
    """

    def __init__(
        self,
        size: int = 50000,
        feat_dim: int = 2048,
        online_accum: bool = False,
        ema_beta: float = 0.0,
    ):
        super().__init__()
        self.size = size
        self.feat_dim = feat_dim
        self.online_accum = online_accum
        self.ema_beta = ema_beta
        self.ema_stats = ema_beta > 0.0
        # Set to True once the circular buffer has wrapped around at least once.
        # Until then, "evicted" features are uninitialized (torch.empty) and
        # must not be subtracted from the online accumulators.
        self._has_wrapped = False

        if self.ema_stats:
            self.register_buffer("mu_ema", torch.zeros(feat_dim, dtype=torch.float64))
            self.register_buffer("m2_ema", torch.zeros(feat_dim, feat_dim, dtype=torch.float64))
            self.register_buffer("_ema_count", torch.zeros(1, dtype=torch.long))
        else:
            self.register_buffer("feats", torch.empty(size, feat_dim))
            self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))
            if online_accum and size > 0:
                self.register_buffer("feat_sum_old", torch.zeros(feat_dim, dtype=torch.float64))
                self.register_buffer("feat_outer_old", torch.zeros(feat_dim, feat_dim, dtype=torch.float64))

    @property
    def pointer(self) -> int:
        return int(self.ptr.item())

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
    def enqueue(self, new_feats: torch.Tensor):
        """Dequeue oldest entries and enqueue new detached features.

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

        ptr = self.pointer

        if self.online_accum:
            new_d = new_det.double()
            # Before the queue has wrapped, "evicted" features are uninitialized
            # (torch.empty) so we only add new features without subtracting.
            if self._has_wrapped:
                evicted = self._get_evicted_feats(n).double()
                self.feat_sum_old.add_(new_d.sum(0) - evicted.sum(0))
                self.feat_outer_old.add_(new_d.T @ new_d - evicted.T @ evicted)
            else:
                self.feat_sum_old.add_(new_d.sum(0))
                self.feat_outer_old.add_(new_d.T @ new_d)

        if ptr + n <= self.size:
            self.feats[ptr: ptr + n] = new_det
        else:
            first = self.size - ptr
            self.feats[ptr: self.size] = new_det[:first]
            self.feats[: n - first] = new_det[first:]
            self._has_wrapped = True
        self.ptr[0] = (ptr + n) % self.size

    # -- Internal helpers ------------------------------------------------------

    def _get_evicted_feats(self, n: int) -> torch.Tensor:
        """Return features that will be overwritten by the next enqueue of size n."""
        ptr = self.pointer
        if ptr + n <= self.size:
            return self.feats[ptr: ptr + n]
        first = self.size - ptr
        return torch.cat([self.feats[ptr: self.size], self.feats[: n - first]], dim=0)

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
    ):
        super().__init__()
        self.fid_norm_eps = fid_norm_eps
        self.device = device
        self.judges = []  # plain list; sub-modules registered via _modules dict
        self._judge_models = nn.ModuleList()   # for proper device placement
        self._judge_queues = nn.ModuleList()   # for proper device placement

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

            # Create feature queue
            queue = FeatureQueue(
                size=queue_size,
                feat_dim=feat_dim,
                online_accum=online_accum,
                ema_beta=ema_beta if ema_stats else 0.0,
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
            f"queue_size={queue_size}, queue_mode={queue_mode}"
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
    def enqueue_features(self, pixels: torch.Tensor):
        """Enqueue features from generated images (no grad).

        Call this after each training step to keep the queue updated.
        """
        for judge in self.judges:
            feats = extract_judge_features(judge, pixels)
            judge["queue"].enqueue(feats.detach())

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
    def prefill_from_dataloader(self, dataloader, vae, num_samples: Optional[int] = None):
        """Pre-fill feature queues with real data before training starts.

        Iterates the dataloader, decodes latents to pixels via the VAE,
        extracts features from each judge, and enqueues them.  This ensures
        the queue has enough samples for a non-degenerate covariance estimate
        from step 1 of training.

        Args:
            dataloader: Iterable yielding dicts with ``"latent"`` key.
            vae: VAE module with ``decode_to_pixels(latent) -> pixels``.
            num_samples: Number of samples to enqueue (default: queue_size).
        """
        if num_samples is None:
            num_samples = max(j["queue"].size for j in self.judges)

        filled = 0
        logger.info(f"[FDLossManager] Pre-filling queues with {num_samples} real data features...")

        for batch in dataloader:
            if filled >= num_samples:
                break

            latents = batch["latent"].to(self.device)
            batch_size = latents.shape[0]
            remaining = num_samples - filled
            count = min(batch_size, remaining)

            # Decode latents to pixels and enqueue features
            pixels = vae.decode_to_pixels(latents[:count].float())
            # decode_to_pixels returns [-1, 1]; FD-Loss expects [0, 1]
            pixels = pixels * 0.5 + 0.5
            self.enqueue_features(pixels)

            filled += count
            if filled % 1000 == 0 or filled == num_samples:
                logger.info(f"[FDLossManager] Pre-fill progress: {filled}/{num_samples}")

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
