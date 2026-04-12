import os
import gradio as gr
try:
    from modules import script_callbacks
    _HAS_WEBUI = True
except ImportError:
    _HAS_WEBUI = False
from trainer import train, trainer
from packaging import version

jsonspath = trainer.jsonspath
logspath = trainer.logspath
presetspath = trainer.presetspath

# Anima-only modes
MODES = ["LoRA", "ADDifT", "Multi-ADDifT"]

# Anima DiT block IDs: BASE (text encoder / global) + B00-B27 (28 transformer blocks)
from trainer.lora import generate_anima_preview_keys, _matches_module_filter

PRECISION_TYPES = ["fp32", "bf16", "fp16", "float32", "bfloat16", "float16"]
NETWORK_TYPES = ["lierla", "c3lier", "loha"]
NETWORK_DIMS = [str(2**x) for x in range(11)]
NETWORK_ALPHAS = [str(2**(x-5)) for x in range(16)]
NETWORK_ELEMENTS = ["Full", "CrossAttention", "SelfAttention"]
IMAGESTEPS = [str(x*64) for x in range(10)]
SEP = "--------------------------"
OPTIMIZERS = ["AdamW", "AdamW8bit", "AdaFactor", "Lion", "Prodigy", SEP,
              "DadaptAdam", "DadaptLion", "DAdaptAdaGrad", "DAdaptAdan", "DAdaptSGD", SEP,
              "Adam8bit", "SGDNesterov8bit", "Lion8bit", "PagedAdamW8bit", "PagedLion8bit", SEP,
              "RAdamScheduleFree", "AdamWScheduleFree", "SGDScheduleFree", SEP,
              "CAME", "Tiger", "AdamMini",
              "PagedAdamW", "PagedAdamW32bit", "SGDNesterov", "Adam"]
LOSS_FUNCTIONS = ["MSE", "L1", "Smooth-L1"]
SCHEDULERS = ["linear", "cosine_annealing", "cosine_annealing_with_restarts", "linear", "cosine",
              "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup",
              "piecewise_constant", "exponential", "step", "multi_step",
              "reduce_on_plateau", "cyclic", "one_cycle"]
ATTN_MODES = ["torch", "flash", "xformers"]

# --- Visibility flags: 3 entries — [LoRA, ADDifT, Multi-ADDifT] ---
ALL       = [True,  True,  True ]
LORA      = [True,  False, False]
ADIFT     = [False, True,  False]
MDIFF     = [False, False, True ]
LORA_MDIFF = [True, False, True ]
DIFF      = [False, True,  True ]   # ADDifT + Multi-ADDifT
NDIFF2    = [True,  True,  True ]   # same as ALL, kept for clarity
ALLN      = [False, False, False]

# Required parameters
lora_data_directory    = ["lora_data_directory",   "TX", None,              "",      str,   LORA_MDIFF]
lora_trigger_word      = ["lora_trigger_word",      "TX", None,              "",      str,   LORA_MDIFF]
diff_target_name       = ["diff_target_name",       "TX", None,              "",      str,   MDIFF]
network_rank           = ["network_rank",            "DD", NETWORK_DIMS[2:],  "16",    int,   ALL]
network_alpha          = ["network_alpha",           "DD", NETWORK_ALPHAS,    "8",     float, ALL]
image_size             = ["image_size(height, width)","TX", None,             512,     str,   ALL]
train_iterations       = ["train_iterations",        "TX", None,              1000,    int,   ALL]
train_batch_size       = ["train_batch_size",        "TX", None,              2,       int,   ALL]
train_learning_rate    = ["train_learning_rate",     "TX", None,              "1e-4",  float, ALL]
train_optimizer        = ["train_optimizer",         "DD", OPTIMIZERS,        "adamw", str,   ALL]
train_optimizer_settings = ["train_optimizer_settings","TX", None,            "",      str,   ALL]
train_lr_scheduler     = ["train_lr_scheduler",      "DD", SCHEDULERS,        "cosine",str,   ALL]
train_lr_scheduler_settings = ["train_lr_scheduler_settings","TX", None,      "",      str,   ALL]
save_lora_name         = ["save_lora_name",          "TX", None,              "",      str,   ALL]
use_gradient_checkpointing = ["use_gradient_checkpointing","CH", None,        False,   bool,  ALL]

# Anima model paths (rendered in dedicated UI row, prepended to all_configs)
qwen3_path         = ["qwen3_path",         "TX", None, "", str, ALL]
t5_tokenizer_path  = ["t5_tokenizer_path",  "TX", None, "", str, ALL]

# Option parameters
train_loss_function = ["train_loss_function","DD", LOSS_FUNCTIONS, "MSE", str, ALL]
train_seed          = ["train_seed",         "TX", None, -1,    int,   ALL]
train_model_precision = ["train_model_precision","DD", PRECISION_TYPES[:3], "bf16", str, ALL]
train_lora_precision  = ["train_lora_precision", "DD", PRECISION_TYPES[:3], "fp32", str, ALL]
image_buckets_step  = ["image_buckets_step", "DD", IMAGESTEPS, "256", int,  LORA_MDIFF]
image_mirroring     = ["image_mirroring",    "CH", None, False, bool, LORA_MDIFF]
image_use_filename_as_tag = ["image_use_filename_as_tag","CH", None, False, bool, LORA_MDIFF]
image_disable_upscale = ["image_disable_upscale","CH", None, False, bool, LORA_MDIFF]
texture_mode        = ["texture_mode",        "CH", None, False, bool, LORA_MDIFF]
texture_feather_latent_px = ["texture_feather_latent_px", "TX", None, 2, int, LORA_MDIFF]
texture_mask_directory = ["texture_mask_directory", "TX", None, "", str, LORA_MDIFF]
# tile_scale: multiply image dimensions before sampling (0.5=sharper, 2.0=chunkier)
texture_tile_scale  = ["texture_tile_scale",  "TX", None, 1.0, float, LORA_MDIFF]
# tile_resolution: pixel side of the square tile sampled from the scaled image (0=random range)
texture_tile_resolution = ["texture_tile_resolution", "TX", None, 0, int, LORA_MDIFF]
# energy_threshold: skip flat/empty tiles; reroll until detail > this (0.01 is a good start, 0 = disabled)
texture_energy_threshold = ["texture_energy_threshold", "TX", None, 0, float, LORA_MDIFF]
save_per_steps      = ["save_per_steps",     "TX", None, 0,    int,   ALL]
save_precision      = ["save_precision",     "DD", PRECISION_TYPES[:3], "fp16", str, ALL]
diff_revert_original_target = ["diff_revert_original_target","CH", None, False, bool, DIFF]
diff_use_diff_mask  = ["diff_use_diff_mask", "CH", None, False, bool, DIFF]
train_fixed_timsteps_in_batch = ["train_fixed_timsteps_in_batch","CH", None, False, bool, ALL]
train_repeat        = ["train_repeat",       "TX", None, 1,    int,   ALL]
gradient_accumulation_steps = ["gradient_accumulation_steps","TX", None, "1", str, ALL]
train_min_timesteps = ["train_min_timesteps","TX", None, 0,    int,   ALL]
train_max_timesteps = ["train_max_timesteps","TX", None, 1000, int,   ALL]
train_flow_shift    = ["train_flow_shift",   "TX", None, 3.0, float, ALL]
# Distribution used to sample training timesteps.
# "uniform"      — flat randint within [min, max]
# "flow_shift"   — bias toward high-noise end via shift factor (train_flow_shift)
# "logit_normal" — sigmoid of Normal(mean, std); params: mean=0.0 std=1.0
# "cosmap"       — cosine-based bijection; bias toward mid-noise
TIMESTEP_DISTRIBUTIONS = ["uniform", "flow_shift", "logit_normal", "cosmap", "beta"]
train_timestep_distribution = ["train_timestep_distribution", "DD", TIMESTEP_DISTRIBUTIONS, "flow_shift", str, ALL]
train_ts_dist_params = ["train_ts_dist_params(e.g. mean=0.0 std=1.0)", "TX", None, "", str, ALL]
# Inline timestep curriculum schedule.  When non-empty, overrides train_min/max_timesteps.
# Format: one entry per line — step_pct  t_min  t_max  [weight_fn]  [mode]
# weight_fn: flat | gaussian:center:sigma | linear:lo_w:hi_w
# mode: texture | fullres (only used when train_hybrid_mode is enabled)
train_ts_schedule   = ["train_ts_schedule",   "ML", None, "",   str,   ALL]
# Hybrid mode: use the same dataset as both texture patches (detail steps) and
# full-res images (layout steps), switching automatically based on the schedule mode column.
train_hybrid_mode   = ["train_hybrid_mode",   "CH", None, False, bool, ALL]
# Regex-based sub-layer filter.  Comma/newline separated; prefix with ! to exclude.
# Examples:  "!adaln_modulation"   →  skip adaln (recommended for Anima)
#            "attn, mlp"           →  attention + MLP only
#            "attn"                →  attention projections only
network_module_filter = ["network_module_filter(regex, !prefix=exclude)", "TX", None, "", str, ALL]
# Layer-wise LR decay: last block trains at base_lr, each earlier block is scaled by decay^depth.
# 1.0 = disabled (flat LR). 0.9 is a good starting point; lower values are more aggressive.
LLRD_DECAYS = ["1.0", "0.98", "0.95", "0.9", "0.85", "0.8"]
network_llrd_decay = ["network_llrd_decay", "DD", LLRD_DECAYS, "1.0", float, ALL]

r_column1 = [network_rank, network_alpha, lora_data_directory, diff_target_name, lora_trigger_word]
r_column2 = [image_size, train_iterations, train_batch_size, train_learning_rate]
r_column3 = [train_optimizer, train_optimizer_settings, train_lr_scheduler, train_lr_scheduler_settings, save_lora_name, use_gradient_checkpointing]

o_column1 = [image_buckets_step, image_mirroring, image_use_filename_as_tag, image_disable_upscale,
             train_fixed_timsteps_in_batch, texture_mode, texture_feather_latent_px, texture_mask_directory,
             texture_tile_scale, texture_tile_resolution, texture_energy_threshold]
o_column2 = [train_seed, train_loss_function, save_per_steps,
             diff_revert_original_target, diff_use_diff_mask]
o_column3 = [train_model_precision, train_lora_precision, save_precision,
             train_repeat, gradient_accumulation_steps]
o_ts_column    = [train_min_timesteps, train_max_timesteps, train_timestep_distribution, train_ts_dist_params, train_ts_schedule, train_hybrid_mode]
o_layer_column = [network_module_filter, network_llrd_decay]

model_column = [qwen3_path, t5_tokenizer_path]

trainer.all_configs = model_column + r_column1 + r_column2 + r_column3 + o_column1 + o_column2 + o_column3 + o_ts_column + o_layer_column

def makeui(sets, pas = 0):
    output = []
    add_id = "2_" if pas > 0 else "1_"
    for name, uitype, choices, value, _, visible in sets:
        visible = visible[pas]
        with gr.Row():
            if uitype == "DD":
                output.append(gr.Dropdown(label=name.replace("_"," "), choices=choices, value=value if value else choices[0] , elem_id="tt_" + name, visible = visible))
            if uitype == "TX":
                output.append(gr.Textbox(label=name.replace("_"," "),value = value, elem_id="tt_" +add_id + name, visible = visible))
            if uitype == "ML":
                output.append(gr.Textbox(label=name.replace("_"," "),value = value, elem_id="tt_" +add_id + name, visible = visible, lines=6))
            if uitype == "CH":
                output.append(gr.Checkbox(label=name.replace("_"," "),value = value, elem_id="tt_" + name, visible = visible))
            if uitype == "CB":
                output.append(gr.CheckboxGroup(label=name.replace("_"," "),choices=choices, value = value, elem_id="tt_" + name, type="value", visible = visible))
            if uitype == "RD":
                output.append(gr.Radio(label=name.replace("_"," "),choices=[x + " " for x in choices] if pas > 0 else choices, value = value, elem_id="tt_" + name,visible = visible))
    return output

_PREVIEW_KEYS = generate_anima_preview_keys()   # generated once at import time

# Accent colour per distribution type for the visualisation
_DIST_COLORS = {
    "uniform":      "#60a5fa",  # blue
    "flow_shift":   "#4ade80",  # green
    "logit_normal": "#f472b6",  # pink
    "cosmap":       "#fb923c",  # orange
    "beta":         "#a78bfa",  # violet
}


def _compute_ts_density(dist_type: str, ts_lo: int, ts_hi: int,
                         dist_params: str, n_bins: int = 50) -> list:
    """Return a list of n_bins normalised density values covering [0, 1000].

    All maths is pure Python so no extra imports are needed at module load time.
    """
    import math

    ts_lo = max(0, int(ts_lo))
    ts_hi = max(ts_lo + 1, min(1000, int(ts_hi)))
    span = ts_hi - ts_lo
    bin_size = 1000.0 / n_bins
    density = [0.0] * n_bins

    # Parse "key=value ..." pairs from dist_params
    params: dict = {}
    for item in (dist_params or "").replace(",", " ").split():
        if "=" in item:
            k, v = item.split("=", 1)
            try:
                params[k.strip()] = float(v.strip())
            except ValueError:
                pass

    if dist_type == "uniform":
        for b in range(n_bins):
            lo = b * bin_size
            hi = lo + bin_size
            overlap = max(0.0, min(float(ts_hi), hi) - max(float(ts_lo), lo))
            density[b] = overlap / bin_size

    elif dist_type == "flow_shift":
        shift = max(1e-3, params.get("shift", 3.0))
        # CDF in sigma space: F(s) = s*shift / (1 + (shift-1)*s)
        def _fs_cdf(s):
            return s * shift / (1.0 + (shift - 1.0) * s)
        for b in range(n_bins):
            lo = b * bin_size
            hi = lo + bin_size
            if hi <= ts_lo or lo >= ts_hi:
                continue
            s_lo = max(0.0, (max(lo, float(ts_lo)) - ts_lo) / span)
            s_hi = min(1.0, (min(hi, float(ts_hi)) - ts_lo) / span)
            if s_hi > s_lo:
                density[b] = (_fs_cdf(s_hi) - _fs_cdf(s_lo)) / (bin_size / 1000.0)

    elif dist_type == "logit_normal":
        mean = params.get("mean", 0.0)
        std = max(0.01, params.get("std", 1.0))
        INV_SQRT2PI = 1.0 / math.sqrt(2.0 * math.pi)

        def _normal_pdf(x):
            return INV_SQRT2PI * math.exp(-0.5 * x * x)

        for b in range(n_bins):
            lo = b * bin_size
            hi = lo + bin_size
            if hi <= ts_lo or lo >= ts_hi:
                continue
            # Numerical integration (8-point midpoint rule in sigma space)
            lo_t = max(float(ts_lo) + 0.001 * span, lo)
            hi_t = min(float(ts_hi) - 0.001 * span, hi)
            if hi_t <= lo_t:
                continue
            n_pts, val = 8, 0.0
            for i in range(n_pts):
                t = lo_t + (hi_t - lo_t) * (i + 0.5) / n_pts
                sigma = max(1e-6, min(1.0 - 1e-6, (t - ts_lo) / span))
                logit = math.log(sigma / (1.0 - sigma))
                pdf_sigma = _normal_pdf((logit - mean) / std) / (std * sigma * (1.0 - sigma))
                val += pdf_sigma / span * (hi_t - lo_t) / n_pts
            density[b] = val

    elif dist_type == "cosmap":
        # sigma = 1 - 1/(tan(pi/2*u)+1)  →  PDF in sigma = 2/pi / (s^2 + (1-s)^2)
        TWO_OVER_PI = 2.0 / math.pi
        for b in range(n_bins):
            lo = b * bin_size
            hi = lo + bin_size
            if hi <= ts_lo or lo >= ts_hi:
                continue
            lo_t = max(float(ts_lo), lo)
            hi_t = min(float(ts_hi), hi)
            if hi_t <= lo_t:
                continue
            n_pts, val = 8, 0.0
            for i in range(n_pts):
                t = lo_t + (hi_t - lo_t) * (i + 0.5) / n_pts
                sigma = max(1e-6, min(1.0 - 1e-6, (t - ts_lo) / span))
                pdf_sigma = TWO_OVER_PI / (sigma ** 2 + (1.0 - sigma) ** 2)
                val += pdf_sigma / span * (hi_t - lo_t) / n_pts
            density[b] = val

    elif dist_type == "beta":
        # Beta(alpha, beta) PDF: Γ(a+b)/(Γ(a)Γ(b)) * s^(a-1) * (1-s)^(b-1)
        # alpha=beta=0.5 → inverse bell (arcsine / U-shape)
        # alpha=beta=1   → uniform
        # alpha=beta>1   → bell; alpha≠beta → skewed
        a = max(0.01, params.get("alpha", 0.5))
        b = max(0.01, params.get("beta",  0.5))
        try:
            log_norm = math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        except ValueError:
            log_norm = 0.0
        for bn in range(n_bins):
            lo = bn * bin_size
            hi = lo + bin_size
            if hi <= ts_lo or lo >= ts_hi:
                continue
            lo_t = max(float(ts_lo), lo)
            hi_t = min(float(ts_hi), hi)
            if hi_t <= lo_t:
                continue
            n_pts, val = 8, 0.0
            for i in range(n_pts):
                t = lo_t + (hi_t - lo_t) * (i + 0.5) / n_pts
                sigma = max(1e-6, min(1.0 - 1e-6, (t - ts_lo) / span))
                log_pdf = log_norm + (a - 1.0) * math.log(sigma) + (b - 1.0) * math.log(1.0 - sigma)
                pdf_sigma = math.exp(log_pdf)
                val += pdf_sigma / span * (hi_t - lo_t) / n_pts
            density[bn] = val

    max_d = max(density) if any(d > 0 for d in density) else 1.0
    return [d / max_d for d in density]


def render_timestep_distribution(dist_type, ts_min, ts_max, dist_params) -> str:
    """Return an HTML snippet containing an SVG bar chart of the timestep distribution."""
    import math

    try:
        ts_lo = max(0, int(float(ts_min or 0)))
        ts_hi = max(ts_lo + 1, min(1000, int(float(ts_max or 1000))))
    except (ValueError, TypeError):
        ts_lo, ts_hi = 0, 1000

    dist_type = dist_type or "flow_shift"
    n_bins = 50
    density = _compute_ts_density(dist_type, ts_lo, ts_hi, dist_params or "", n_bins)

    # SVG layout
    W, H = 500, 74
    bar_area_h = 54
    bar_w = W / n_bins
    bin_size = 1000.0 / n_bins
    color = _DIST_COLORS.get(dist_type, "#60a5fa")

    bars = []
    for b, d in enumerate(density):
        bin_lo = b * bin_size
        bin_hi = bin_lo + bin_size
        in_range = bin_lo < ts_hi and bin_hi > ts_lo
        x = b * bar_w
        h = max(0.0, d * bar_area_h)
        y = bar_area_h - h
        fill = color if in_range else "#252525"
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w - 0.8:.1f}" height="{h:.1f}" fill="{fill}" rx="0.5"/>'
        )

    # Axis ticks and labels
    ticks = []
    for t_val, t_lbl in [(0, "0"), (250, "250"), (500, "500"), (750, "750"), (1000, "1000")]:
        x = t_val / 1000.0 * W
        ticks.append(
            f'<line x1="{x:.1f}" y1="{bar_area_h}" x2="{x:.1f}" y2="{bar_area_h + 3}" stroke="#555" stroke-width="0.5"/>'
            f'<text x="{x:.1f}" y="{bar_area_h + 12}" text-anchor="middle" fill="#555" font-size="8">{t_lbl}</text>'
        )

    # Active range boundary markers
    lo_x = ts_lo / 1000.0 * W
    hi_x = ts_hi / 1000.0 * W
    markers = (
        f'<line x1="{lo_x:.1f}" y1="0" x2="{lo_x:.1f}" y2="{bar_area_h}" stroke="#ffffff18" stroke-width="0.8" stroke-dasharray="2,2"/>'
        f'<line x1="{hi_x:.1f}" y1="0" x2="{hi_x:.1f}" y2="{bar_area_h}" stroke="#ffffff18" stroke-width="0.8" stroke-dasharray="2,2"/>'
    )

    svg = (
        f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;display:block;">'
        + markers + "".join(bars) + "".join(ticks)
        + f'<line x1="0" y1="{bar_area_h}" x2="{W}" y2="{bar_area_h}" stroke="#333" stroke-width="0.5"/>'
        + "</svg>"
    )

    # Human-readable parameter summary
    param_desc = ""
    if dist_type == "flow_shift":
        _fs_p: dict = {}
        for item in (dist_params or "").replace(",", " ").split():
            if "=" in item:
                k, v = item.split("=", 1)
                try:
                    _fs_p[k.strip()] = float(v.strip())
                except ValueError:
                    pass
        param_desc = f" &nbsp;<span style='color:#666;'>shift={_fs_p.get('shift', 3.0):.3g}</span>"
    elif dist_type == "logit_normal":
        params: dict = {}
        for item in (dist_params or "").replace(",", " ").split():
            if "=" in item:
                k, v = item.split("=", 1)
                try:
                    params[k.strip()] = float(v.strip())
                except ValueError:
                    pass
        m, s = params.get("mean", 0.0), params.get("std", 1.0)
        param_desc = f" &nbsp;<span style='color:#666;'>mean={m:.2g}&thinsp; std={s:.2g}</span>"
    elif dist_type == "beta":
        params2: dict = {}
        for item in (dist_params or "").replace(",", " ").split():
            if "=" in item:
                k, v = item.split("=", 1)
                try:
                    params2[k.strip()] = float(v.strip())
                except ValueError:
                    pass
        a2, b2 = params2.get("alpha", 0.5), params2.get("beta", 0.5)
        param_desc = f" &nbsp;<span style='color:#666;'>α={a2:.2g}&thinsp; β={b2:.2g}</span>"

    header = (
        f'<div style="font-size:11px;margin-bottom:4px;font-family:sans-serif;">'
        f'<span style="color:{color};font-weight:600;">{dist_type}</span>'
        f'{param_desc}'
        f'&nbsp;&nbsp;<span style="color:#555;font-size:10px;">range&thinsp;{ts_lo}–{ts_hi}</span>'
        f'</div>'
    )

    return (
        f'<div style="background:#111;border:1px solid #2a2a2a;border-radius:4px;padding:8px 12px;">'
        + header + svg + "</div>"
    )


def render_layer_preview(filter_str: str) -> str:
    """Return an HTML snippet showing every canonical Anima LoRA key,
    highlighted (green) when the current filter selects it or dimmed when it
    is filtered out.  Blocks are grouped with a small header label.
    """
    import re as _re

    # Validate regex — show error banner instead of crashing
    raw_patterns = [p.strip() for p in _re.split(r'[,\n]+', filter_str or '') if p.strip()]
    bad_patterns = []
    for p in raw_patterns:
        pat = p.lstrip('!')
        try:
            _re.compile(pat)
        except _re.error as e:
            bad_patterns.append(f"{pat!r}: {e}")

    error_html = ""
    if bad_patterns:
        msgs = "<br>".join(bad_patterns)
        error_html = (
            f'<div style="color:#f87171;font-family:monospace;font-size:11px;'
            f'margin-bottom:6px;padding:4px 8px;background:rgba(248,113,113,0.08);'
            f'border-radius:3px;">Invalid regex — {msgs}</div>'
        )

    active_count = 0
    rows = []
    prev_block = None

    for key in _PREVIEW_KEYS:
        m = _re.search(r'_blocks_(\d+)_', key)
        block = f"B{int(m.group(1)):02d}" if m else "BASE"

        if block != prev_block:
            rows.append(
                f'<div style="color:#666;font-size:10px;margin-top:6px;margin-bottom:1px;'
                f'font-family:monospace;letter-spacing:0.08em;">── {block} ──</div>'
            )
            prev_block = block

        active = _matches_module_filter(key, filter_str or '')
        if active:
            active_count += 1
            style = (
                'color:#4ade80;background:rgba(74,222,128,0.09);'
                'padding:1px 6px;border-radius:2px;display:block;'
                'margin:1px 0;font-family:monospace;font-size:11px;'
            )
        else:
            style = (
                'color:#3a3a3a;padding:1px 6px;display:block;'
                'margin:1px 0;font-family:monospace;font-size:11px;'
            )

        rows.append(f'<span style="{style}">{key}</span>')

    total = len(_PREVIEW_KEYS)
    frac_color = '#4ade80' if active_count == total else ('#facc15' if active_count > 0 else '#f87171')
    header = (
        f'<div style="font-size:12px;color:#aaa;margin-bottom:6px;font-family:sans-serif;">'
        f'<span style="color:{frac_color};font-weight:600;">{active_count}</span>'
        f'<span style="color:#666;">/{total}</span> layers active'
        f'&nbsp;&nbsp;<span style="color:#555;font-size:10px;">canonical architecture preview</span>'
        f'</div>'
    )
    container = (
        f'<div style="height:420px;overflow-y:auto;background:#111;border:1px solid #2a2a2a;'
        f'border-radius:4px;padding:8px 12px;box-sizing:border-box;">'
        + ''.join(rows)
        + '</div>'
    )
    return error_html + header + container


def ToolButton(value="", elem_classes=None, **kwargs):
    elem_classes = elem_classes or []
    return gr.Button(value=value, elem_classes=["tool", *elem_classes], **kwargs)

def on_ui_tabs():
    global result

    def load_preset(name):
        json_files = [f.replace(".json","") for f in os.listdir(presetspath) if f.endswith('.json')]
        if name is None:
            return json_files
        else:
            return trainer.import_json(name, preset = True)

    folder_symbol = '\U0001f4c2'
    load_symbol = '\u2199\ufe0f'   # ↙
    save_symbol = '\U0001f4be'     # 💾
    refresh_symbol = '\U0001f504'  # 🔄

    with gr.Blocks() as ui:
        with gr.Tab("Train"):
            result = gr.Textbox(label="Message")
            with gr.Row():
                start = gr.Button(value="Start Training", elem_classes=["compact_button"], variant='primary')
                stop = gr.Button(value="Stop", elem_classes=["compact_button"], variant='primary')
                stop_save = gr.Button(value="Stop and Save", elem_classes=["compact_button"], variant='primary')
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        presets = gr.Dropdown(choices=load_preset(None), show_label=False, elem_id="tt_preset")
                        loadpreset = ToolButton(value=load_symbol)
                        savepreset = ToolButton(value=save_symbol)
                        refleshpreset = ToolButton(value=refresh_symbol)
                with gr.Column():
                    with gr.Row():
                        sets_file = gr.Textbox(show_label=False)
                        openfolder = ToolButton(value=folder_symbol)
                        loadjson = ToolButton(value=load_symbol)
            with gr.Row(equal_height=True):
                with gr.Column():
                    mode = gr.Radio(label="Mode", choices=MODES, value="LoRA")
                with gr.Column():
                    model = gr.Textbox(label="Anima DiT Path (.safetensors)", elem_id="tt_anima_dit_path")
                with gr.Column():
                    vae = gr.Textbox(label="VAE Path (.safetensors)", elem_id="tt_anima_vae_path")

            with gr.Row(equal_height=True):
                with gr.Column():
                    qwen3_gr = gr.Textbox(label="Qwen3 Path", value="", elem_id="tt_qwen3_path")
                with gr.Column():
                    t5_gr = gr.Textbox(label="T5 Tokenizer Path (optional)", value="", elem_id="tt_t5_tokenizer_path")

            dummy = gr.Checkbox(visible=False, value=False)

            gr.HTML(value="Required Parameters")
            with gr.Row():
                with gr.Column(variant="compact"):
                    col1_r1 = makeui(r_column1)
                with gr.Column(variant="compact"):
                    col2_r1 = makeui(r_column2)
                with gr.Column(variant="compact"):
                    col3_r1 = makeui(r_column3)
            gr.HTML(value="Option Parameters")
            with gr.Row():
                with gr.Column(variant="compact"):
                    col1_o1 = makeui(o_column1)
                with gr.Column(variant="compact"):
                    col2_o1 = makeui(o_column2)
                with gr.Column(variant="compact"):
                    col3_o1 = makeui(o_column3)

            # --- Timestep Distribution ---
            gr.HTML(value="Timestep Distribution")
            with gr.Row():
                with gr.Column(variant="compact", scale=1):
                    col_ts = makeui(o_ts_column)
                with gr.Column(scale=2):
                    with gr.Group(visible=True) as g_fs_sliders:
                        fs_slider = gr.Slider(label="flow shift", minimum=0.05, maximum=5.0, step=0.05, value=3.0)
                    with gr.Group(visible=False) as g_logit_sliders:
                        mean_slider = gr.Slider(label="mean", minimum=-4.0, maximum=4.0,  step=0.1,  value=0.0)
                        std_slider  = gr.Slider(label="std",  minimum=0.1,  maximum=4.0,  step=0.05, value=1.0)
                    with gr.Group(visible=False) as g_beta_sliders:
                        alpha_slider    = gr.Slider(label="alpha", minimum=0.1, maximum=20.0, step=0.1, value=0.5)
                        beta_prm_slider = gr.Slider(label="beta",  minimum=0.1, maximum=20.0, step=0.1, value=0.5)
                    ts_dist_preview = gr.HTML(
                        value=render_timestep_distribution("flow_shift", 0, 1000, "shift=3.0"),
                    )

            # --- Layers ---
            gr.HTML(value="Layers")
            with gr.Row():
                with gr.Column(variant="compact", scale=1):
                    col_layer = makeui(o_layer_column)
                with gr.Column(scale=2):
                    layer_preview = gr.HTML(value=render_layer_preview(""))

            model_col_grs = [qwen3_gr, t5_gr]
            all_gr = model_col_grs + col1_r1 + col2_r1 + col3_r1 + col1_o1 + col2_o1 + col3_o1 + col_ts + col_layer
            train_settings_1 = all_gr + [dummy]

            # Look up config-tracked widgets by identity (immune to list reordering)
            dist_gr      = all_gr[trainer.all_configs.index(train_timestep_distribution)]
            ts_min_gr    = all_gr[trainer.all_configs.index(train_min_timesteps)]
            ts_max_gr    = all_gr[trainer.all_configs.index(train_max_timesteps)]
            ts_params_gr = all_gr[trainer.all_configs.index(train_ts_dist_params)]
            _filter_gr   = all_gr[trainer.all_configs.index(network_module_filter)]

            # Chart update: fired by any of the four config inputs
            _ts_dist_inputs = [dist_gr, ts_min_gr, ts_max_gr, ts_params_gr]
            for _inp in _ts_dist_inputs:
                _inp.change(render_timestep_distribution, inputs=_ts_dist_inputs, outputs=[ts_dist_preview])

            # When distribution type changes: reset params string + show correct slider group
            _DIST_PARAM_DEFAULTS = {
                "uniform":      "",
                "flow_shift":   "shift=3.0",
                "logit_normal": "mean=0.0 std=1.0",
                "cosmap":       "",
                "beta":         "alpha=0.5 beta=0.5",
            }
            def _on_dist_change(dist):
                show_fs    = dist == "flow_shift"
                show_logit = dist == "logit_normal"
                show_beta  = dist == "beta"
                return (
                    _DIST_PARAM_DEFAULTS.get(dist, ""),  # reset dist_params text
                    gr.update(visible=show_fs),
                    gr.update(visible=show_logit),
                    gr.update(visible=show_beta),
                    gr.update(value=3.0),   # fs_slider
                    gr.update(value=0.0),   # mean_slider
                    gr.update(value=1.0),   # std_slider
                    gr.update(value=0.5),   # alpha_slider
                    gr.update(value=0.5),   # beta_prm_slider
                )
            dist_gr.change(
                _on_dist_change, inputs=[dist_gr],
                outputs=[ts_params_gr, g_fs_sliders, g_logit_sliders, g_beta_sliders,
                         fs_slider, mean_slider, std_slider, alpha_slider, beta_prm_slider],
            )

            # flow_shift slider → dist_params (same pattern as logit_normal / beta)
            fs_slider.change(lambda v: f"shift={v:.3g}", inputs=[fs_slider], outputs=[ts_params_gr])

            # logit_normal sliders → dist_params
            def _logit_to_params(mean, std):
                return f"mean={mean:.2g} std={std:.2g}"
            mean_slider.change(_logit_to_params, inputs=[mean_slider, std_slider], outputs=[ts_params_gr])
            std_slider.change(_logit_to_params,  inputs=[mean_slider, std_slider], outputs=[ts_params_gr])

            # beta sliders → dist_params
            def _beta_to_params(alpha, beta_v):
                return f"alpha={alpha:.3g} beta={beta_v:.3g}"
            alpha_slider.change(_beta_to_params,    inputs=[alpha_slider, beta_prm_slider], outputs=[ts_params_gr])
            beta_prm_slider.change(_beta_to_params, inputs=[alpha_slider, beta_prm_slider], outputs=[ts_params_gr])

            # Layer preview
            _filter_gr.change(render_layer_preview, inputs=[_filter_gr], outputs=[layer_preview])

            with gr.Group(visible=False) as g_diff:
                gr.HTML(value="Image Pairs (Original → Target)")
                with gr.Row():
                    with gr.Column():
                        orig_image = gr.Image(label="Original Image", interactive=True)
                    with gr.Column():
                        targ_image = gr.Image(label="Target Image", interactive=True)

        dtrue = gr.Checkbox(value=True, visible=False)
        dfalse = gr.Checkbox(value=False, visible=False)

        in_images = [orig_image, targ_image]

        def savepreset_f(*args):
            train.train(*args)
            return gr.update(choices=load_preset(None))

        start.click(train.train, [dfalse, mode, model, vae, *train_settings_1, *in_images], [result])
        savepreset.click(savepreset_f, [dtrue, mode, model, vae, *train_settings_1, *in_images], [presets])
        refleshpreset.click(lambda: gr.update(choices=load_preset(None)), outputs=[presets])

        stop.click(train.stop_time, [dfalse], [result])
        stop_save.click(train.stop_time, [dtrue], [result])

        def change_the_mode(mode):
            mode_idx = MODES.index(mode)
            out = [gr.update(visible=x[5][mode_idx]) for x in trainer.all_configs]
            out.append(gr.update())  # dummy checkbox
            # g_diff (image pair picker) only for single-pair ADDifT
            out.append(gr.update(visible=(mode_idx == 1)))
            return out

        def openfolder_f():
            os.startfile(jsonspath)

        loadjson.click(trainer.import_json, [sets_file], [mode, model, vae] + train_settings_1)
        loadpreset.click(load_preset, [presets], [mode, model, vae] + train_settings_1)
        mode.change(change_the_mode, [mode], [*train_settings_1, g_diff])
        openfolder.click(openfolder_f)

    return (ui, "TrainTrain", "TrainTrain"),

if _HAS_WEBUI and __package__ == "traintrain":
    script_callbacks.on_ui_tabs(on_ui_tabs)
elif __name__ == "__main__":
    ui = on_ui_tabs()[0][0]
    ui.launch(css=".gradio-textbox {margin-bottom: 0 !important;}")
