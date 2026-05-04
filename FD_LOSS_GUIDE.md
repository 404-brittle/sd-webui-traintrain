# FD-Loss: Intelligent Feature Queue & Interactive Guidance

This document describes the extended FD-Loss (Fréchet Distance) system implemented for
[`TrainTrain`](scripts/traintrain.py).  The core idea is to replace the original blind-FIFO
feature queue with **three eviction strategies** and add **inter-epoch cluster inspection**
so you can steer training toward desirable outcomes.

---

## Quick Start

1. Enable FD-Loss in the **Options** tab: check `fd_loss_enable`.
2. Choose an **eviction mode** via the `fd_eviction_mode` dropdown:
   - `fifo` — original circular-buffer behaviour (safe default).
   - `diversity` — evict the most redundant features automatically.
   - `guided` — evict features *farthest* from a guidance target (see below).
3. Set `fd_cluster_log_interval` to a positive step count (e.g. `500`) to see
   feature-cluster summaries printed during training.
4. (Optional) Set `fd_n_clusters` to control the granularity of the cluster view
   (default `20`).

---

## Eviction Modes

### `fifo` (default, backward-compatible)

The original circular buffer: oldest features are overwritten first.  No intelligence,
but guaranteed stable and well-tested.

### `diversity`

Evicts the features that are **most similar to their nearest neighbour** in the queue.
This keeps the buffer filled with a diverse set of representations, preventing mode
collapse where the queue becomes dominated by a single visual style.

**When to use:** You have a varied dataset and want the FID reference distribution to
cover the full breadth of your training data.

### `guided`

Evicts features that are **farthest from a guidance target** (a feature vector you
provide), blended with a diversity bonus to prevent collapse.

The eviction score for each feature `f` is:

```
score(f) = (1 - α) · sim(f, target)  +  α · (1 - max_sim(f, queue))
```

Where:
- `sim(f, target)` = cosine similarity between `f` and the guidance target
- `max_sim(f, queue)` = cosine similarity to `f`'s nearest neighbour in the queue
- `α` = `guidance_strength` (0.0 = pure target-distance, 1.0 = pure diversity)

Features with the **lowest** score are evicted first — i.e. the queue preferentially
retains features that are *close* to the target and *diverse*.

**When to use:** You have a specific aesthetic or style you want to reinforce.  For
example, if you want the model to produce "sharp, high-contrast" outputs, you can
provide a guidance target derived from images that exhibit those qualities.

---

## Interactive Guidance

### Setting a guidance target at runtime

The [`FDLossManager`](trainer/fd_loss.py:1057) exposes:

```python
fd_manager.set_guidance_target(
    target_features,       # torch.Tensor of shape [N, D] — feature vectors
    judge_name=None,       # which judge's queue to target (None = all)
)
```

You can call this from a custom callback or a Gradio button during training.
The target is typically the **mean feature vector** of a batch of images you
consider "good".

### Clearing the target

```python
fd_manager.clear_guidance_target(judge_name=None)
```

Reverts the queue to `fifo` eviction (or the mode set via `set_eviction_mode`).

### Previewing candidates

```python
candidates = fd_manager.get_guidance_candidates(n=10, judge_name=None)
```

Returns the `n` features in the queue that are **closest** to the current guidance
target — useful for sanity-checking what the queue considers "desirable".

---

## Cluster Inspection

Between epochs (or at a regular step interval), the system runs **k-means++**
clustering on the feature queue and prints a summary to the console.

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fd_n_clusters` | `20` | Number of clusters for k-means++ |
| `fd_cluster_log_interval` | `0` | Log cluster summary every N steps (`0` = disabled) |

### Output format

When clustering fires, you'll see something like:

```
[Step 1500] FD queue clusters:
━━━ Cluster 0 ━━━ size: 1240 ━━━ inertia: 0.342
━━━ Cluster 1 ━━━ size: 987  ━━━ inertia: 0.287
━━━ Cluster 2 ━━━ size: 512  ━━━ inertia: 0.401
...
Silhouette: 0.31  |  Total inertia: 184.2
```

- **Size**: how many features belong to this cluster (larger = more common style).
- **Inertia**: within-cluster sum of squared distances (lower = tighter cluster).
- **Silhouette**: cluster separation quality, range [-1, 1] (higher = better
  separation).  A low silhouette suggests the queue is homogeneous; a high value
  suggests diverse modes.

### Interpreting clusters

- If one cluster dominates (>50% of features), the queue has **mode collapse** —
  consider switching to `diversity` eviction.
- If silhouette is very low (< 0.1), features are nearly uniformly distributed —
  the queue may not have enough samples to form meaningful clusters yet.
- If you see a cluster with very high inertia, it may contain outliers — consider
  whether those outliers are desirable or should be excluded from the reference.

---

## Visual Interactive Cluster Panel

The **visual interactive cluster panel** lets you inspect actual image thumbnails
grouped by feature cluster, and take direct action — protect clusters from eviction,
set them as guidance targets, or evict them entirely.

### Enabling source image storage

To use the visual panel, enable source image storage **before training starts**:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `fd_store_source_images` | `False` | Store pixel thumbnails alongside features in the queue |

When enabled, each feature enqueued also stores a 64×64 pixel crop of the source
image that produced it.  This consumes additional GPU memory:
`queue_size × 3 × 64 × 64 × 1 byte ≈ 600 MB` for a 50k queue.

> **Note:** Source images are only stored for features enqueued **after** this
> flag is enabled.  Pre-filled features from the dataloader will also have
> thumbnails if the flag was set at construction time.

### Per-Sample vs Cluster-Level Actions

The panel provides **two levels of granularity**:

| Level | Buttons | Scope |
|-------|---------|-------|
| **Cluster** | 🛡️ Protect All / 🔓 Unprotect All / 🎯 Set as Guidance / ❌ Evict Cluster | All features in the cluster |
| **Per-Sample** | 🛡️ / 🔓 / ❌ (below each thumbnail) | A single queue index |

This solves the "18 good, 2 bad" problem — you can protect the good samples
individually and evict only the bad ones, without throwing away the whole cluster.

### Programmatic API

All methods are available on [`FDLossManager`](trainer/fd_loss.py:1064):

```python
# --- Cluster-level actions ---

# Get thumbnails grouped by cluster
thumbnails = fd_manager.get_cluster_thumbnails(
    judge_name=None,       # which judge (None = first)
    n_clusters=None,       # override cluster count
    max_per_cluster=20,    # limit thumbnails per cluster
)
# Returns: {cluster_id: [(index, PIL.Image), ...], ...}

# Protect a cluster from eviction
fd_manager.protect_cluster(cluster_id=3, judge_name=None, n_clusters=None)

# Unprotect a cluster
fd_manager.unprotect_cluster(cluster_id=3, judge_name=None, n_clusters=None)

# Set a cluster's centroid as the guidance target
fd_manager.set_guidance_from_cluster(cluster_id=5, judge_name=None, n_clusters=None)
# This also auto-switches the queue to "guided" eviction mode.

# Evict an entire cluster (immediate removal)
fd_manager.evict_cluster(cluster_id=2, judge_name=None, n_clusters=None)
# Replaces cluster features with random noise — they'll be overwritten
# by normal enqueue on the next step.

# --- Per-sample actions (individual queue indices) ---

# Protect a single feature from eviction
fd_manager.protect_index(idx=42, judge_name=None)

# Remove protection from a single feature
fd_manager.unprotect_index(idx=42, judge_name=None)

# Evict a single feature immediately
fd_manager.evict_index(idx=42, judge_name=None)
# Replaces that one feature with noise; the rest of the cluster is untouched.
```

### Using the Gradio UI Tab

The interactive cluster panel is now available as a **"FD Cluster Inspector"** tab in the
Gradio UI (next to the "Train" tab).

**To use it:**

1. Enable **`fd_loss_enable`** and **`fd_store_source_images`** in the Options tab.
2. Start training.
3. Switch to the **"FD Cluster Inspector"** tab.
4. Click **🔄 Refresh Cluster Panel** to see the current cluster structure.

The panel shows:
- Each cluster as a collapsible section with a color-coded border
- Cluster size, percentage, and protection status
- **Cluster-level buttons**: Protect All, Unprotect All, Set as Guidance, Evict Cluster
- **Per-sample buttons** (below each thumbnail): Protect, Unprotect, Evict

> **Note:** The action buttons in the HTML panel emit `data-cluster` and `data-idx`
> attributes.  In the current Gradio integration, use the **Refresh** button to
> re-render the panel after making changes.  A future update may add live
> Gradio event wiring for the embedded buttons.

You can also adjust the **Clusters** and **Thumbnails per cluster** spinner values
and the panel will refresh automatically.

### Programmatic API

```python
html = fd_manager.render_interactive_cluster_panel(
    judge_name=None,
    n_clusters=None,
    max_per_cluster=20,
)
```

Returns a self-contained HTML string with:
- **Collapsible `<details>` sections** — one per cluster, sorted by size
- **Cluster-level action buttons** with `data-cluster` attributes:
  - 🛡️ **Protect All** — pin all features in this cluster from eviction
  - 🔓 **Unprotect All** — release protection for all features in this cluster
  - 🎯 **Set as Guidance** — use this cluster's centroid as guidance target
  - ❌ **Evict Cluster** — immediately remove this cluster's features
- **Per-sample thumbnails** (48×48 px) — each with its own individual buttons
  using `data-idx` attributes:
  - 🛡️ **Protect** — pin this single feature from eviction
  - 🔓 **Unprotect** — release protection for this single feature
  - ❌ **Evict** — immediately remove this single feature
- Each thumbnail shows its queue index (`#42`) and a green highlight if protected

### Protection mechanism

The [`protected_mask`](trainer/fd_loss.py:190) is a `torch.bool` buffer on
[`FeatureQueue`](trainer/fd_loss.py:161).  When an index is protected:

1. All eviction strategies (`fifo`, `diversity`, `guided`) skip protected indices.
2. The unified dispatcher [`_get_evict_indices(n)`](trainer/fd_loss.py:420)
   filters out protected indices before running the eviction strategy.
3. If not enough unprotected candidates exist, it falls back to FIFO on the
   unprotected pool.

Protected features remain in the queue indefinitely (until explicitly unprotected
or the queue is reset).

### Workflow example

1. **Enable** `fd_store_source_images` in the Options tab before training.
2. **Train** for a few hundred steps to populate the queue.
3. **Call** `fd_manager.render_interactive_cluster_panel()` from a Gradio button
   or callback to see the visual panel.
4. **Inspect** each cluster's thumbnails — do they look like a coherent style?
5. **Per-sample**: if a cluster has 18 good thumbnails and 2 bad ones, protect
   the 18 good ones individually and evict the 2 bad ones. The cluster-level
   "Protect All" would have protected the bad ones too.
6. **Cluster-level**: if an entire cluster is undesirable (e.g. all blurry),
   use "Evict Cluster" to purge it in one click.
7. **Set as Guidance** to steer the queue toward a specific cluster's centroid.
8. The queue will now preferentially retain features similar to the protected
   indices and evict features dissimilar to the guidance target.

---

## Architecture

### [`FeatureQueue`](trainer/fd_loss.py:161)

The core data structure.  Key additions:

| Method | Description |
|--------|-------------|
| [`_get_fifo_evict_indices(n)`](trainer/fd_loss.py:459) | Returns the `n` oldest indices |
| [`_get_diversity_evict_indices(n)`](trainer/fd_loss.py:470) | Returns indices of the `n` most redundant features |
| [`_get_guided_evict_indices(n)`](trainer/fd_loss.py:488) | Returns indices farthest from guidance target |
| [`_get_evict_indices(n)`](trainer/fd_loss.py:420) | **Unified dispatcher** — respects `protected_mask` |
| [`protect_indices(indices)`](trainer/fd_loss.py:516) | Mark indices as protected (never evicted) |
| [`unprotect_indices(indices)`](trainer/fd_loss.py:526) | Remove protection from indices |
| [`protect_cluster(cluster_id)`](trainer/fd_loss.py:536) | Protect all features in a cluster |
| [`unprotect_cluster(cluster_id)`](trainer/fd_loss.py:551) | Unprotect all features in a cluster |
| [`get_clusters(n_clusters)`](trainer/fd_loss.py:610) | k-means++ clustering with silhouette score |
| [`get_cluster_summary_html(n_clusters)`](trainer/fd_loss.py:685) | HTML bar-chart summary for Gradio |
| [`get_cluster_thumbnails(n_clusters)`](trainer/fd_loss.py:728) | Returns source image thumbnails grouped by cluster |
| [`render_interactive_cluster_panel(n_clusters)`](trainer/fd_loss.py:769) | Full HTML panel with thumbnails + action buttons |
| [`get_guidance_candidates(n)`](trainer/fd_loss.py:871) | Top-n features closest to guidance target |

### [`FDLossManager`](trainer/fd_loss.py:1064)

Orchestrates multiple judges (feature extractors).  New methods:

| Method | Description |
|--------|-------------|
| [`get_clusters(judge_name, n_clusters)`](trainer/fd_loss.py:1289) | Cluster features from one or all judges |
| [`get_cluster_summary_html(...)`](trainer/fd_loss.py:1307) | HTML summary for Gradio |
| [`get_cluster_thumbnails(...)`](trainer/fd_loss.py:1449) | Source image thumbnails grouped by cluster |
| [`render_interactive_cluster_panel(...)`](trainer/fd_loss.py:1466) | Full HTML panel with per-sample thumbnails + action buttons |
| [`protect_cluster(cluster_id, ...)`](trainer/fd_loss.py:1482) | Protect a cluster from eviction |
| [`unprotect_cluster(cluster_id, ...)`](trainer/fd_loss.py:1491) | Unprotect a cluster |
| [`set_guidance_from_cluster(cluster_id, ...)`](trainer/fd_loss.py:1500) | Set cluster centroid as guidance target |
| [`evict_cluster(cluster_id, ...)`](trainer/fd_loss.py:1540) | Immediately remove a cluster's features |
| [`protect_index(idx, judge_name)`](trainer/fd_loss.py:1523) | Protect a **single queue index** from eviction |
| [`unprotect_index(idx, judge_name)`](trainer/fd_loss.py:1531) | Remove protection from a single queue index |
| [`evict_index(idx, judge_name)`](trainer/fd_loss.py:1539) | Evict a single queue index immediately |
| [`set_guidance_target(target, judge_name)`](trainer/fd_loss.py:1330) | Set guidance target for guided eviction |
| [`clear_guidance_target(judge_name)`](trainer/fd_loss.py:1358) | Clear guidance target |
| [`set_eviction_mode(mode, judge_name)`](trainer/fd_loss.py:1369) | Change eviction mode at runtime |
| [`get_guidance_candidates(n, judge_name)`](trainer/fd_loss.py:1388) | Get features closest to target |

### Training loop integration

Both [`train_lora()`](trainer/train.py:246) and [`train_diff2()`](trainer/train.py:520)
read the new config parameters and pass them to `FDLossManager`.  When
`fd_store_source_images` is enabled, the decoded pixel crops are passed alongside
features during `enqueue_features()` so thumbnails are stored for the visual panel.

### UI controls

All new parameters appear in the **Options** tab of the Gradio UI
([`scripts/traintrain.py`](scripts/traintrain.py:165)):

| Control | Type | Description |
|---------|------|-------------|
| `fd_eviction_mode` | Dropdown | `fifo` / `diversity` / `guided` |
| `fd_guidance_strength` | Textbox | Diversity bonus weight (0.0–1.0) |
| `fd_n_clusters` | Textbox | Number of k-means clusters |
| `fd_cluster_log_interval` | Textbox | Steps between cluster logs (0 = off) |
| `fd_store_source_images` | Checkbox | Store pixel thumbnails for visual panel |

---

## Tips & Best Practices

1. **Start with `fifo`** to establish a baseline.  Then switch to `diversity` and
   compare the FID scores.
2. **Use `guided` mode sparingly** — it's a powerful tool but can bias the queue
   too aggressively if `guidance_strength` is too low (pure target-distance can
   collapse the queue to a single mode).
3. **Set `fd_cluster_log_interval` to ~500** for a 10k-step run — you'll get ~20
   snapshots to review without overwhelming the console.
4. **Watch the silhouette score**: if it drops over time, the queue is becoming
   more homogeneous.  Consider switching to `diversity` mode.
5. **Guidance targets** can be extracted from any batch of images you consider
   "good" — just run them through the same feature extractor and take the mean.
6. **Enable `fd_store_source_images`** only when you plan to use the visual
   cluster panel — it adds ~600 MB GPU memory overhead for a 50k queue.
7. **Protect clusters early** — once a desirable feature pattern is identified,
   protect it so it stays in the queue as a permanent reference point.
8. **Evict problem clusters** — if you see a cluster of blurry or distorted
   features, evict it immediately to prevent it from polluting the FID reference.
