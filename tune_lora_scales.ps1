# ============================================================
# tune_lora_scales.ps1
#
# Automated per-layer LoRA scale optimisation using the
# original training image-caption pairs.
#
# Learns a scalar α_i for each LoRA layer by minimising the
# denoising loss on a held-out validation split.  Layers that
# hurt quality converge toward 0 (cut); useful layers keep
# their scale.  The result is baked into a new .safetensors.
#
# See the report for the full algorithm description (Strategy 2A).
#
# Usage — run with defaults below:
#   .\tune_lora_scales.ps1
#
# Usage — specify LoRA and data directory directly:
#   .\tune_lora_scales.ps1 `
#       --lora_file  X:\loras\my_run.safetensors `
#       --data_dir   X:\datasets\my_training_images
#
# Usage — pass any args directly to the Python script:
#   .\tune_lora_scales.ps1 --steps 100 --lr 0.02 --reg_strength 0
# ============================================================

# ---------------------------------------------------------------
# MODEL PATHS — set these for your installation
# ---------------------------------------------------------------

$VENV_DIR        = ".venv"
$SD_SCRIPTS_PATH = "X:\SD\sd-scripts"
$DIT_PATH        = "X:\SD\sd-webui-forge-neo\models\Stable-diffusion\anima-preview2.safetensors"
$VAE_PATH        = "X:\SD\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\models\vae\Wan2_1_VAE_fp32.safetensors"
$QWEN3_PATH      = "X:\SD\AnimaLoraToolkit\models\text_encoders"
$T5_PATH         = "X:\SD\AnimaLoraToolkit\models\t5_tokenizer"

# ---------------------------------------------------------------
# INPUT — LoRA to tune and training data directory
# ---------------------------------------------------------------

$LORA_FILE = "X:\SD\ComfyUI_windows_portable_nvidia\ComfyUI_windows_portable\ComfyUI\models\loras\anima\_style\xxuanon_ap2 v2.safetensors"
$DATA_DIR  = "X:\d\xxu_anon\tmp\1_xxuanon\New folder"

$PREVIEW_PROMPT           = "xxuanon, newest, masterpiece, best quality, 1girl, standing, kasane teto, utau, red hair, red eyes, twin drills, white t-shirt, looking at viewer, upper body, starry night aurora background"

# Output path.  Leave blank to auto-name as <input_stem>_tuned.safetensors.
$OUTPUT = ""

# ---------------------------------------------------------------
# OPTIMISATION SETTINGS
#
# Three preset profiles — uncomment one block, or mix and match:
#
#   CONSERVATIVE  — gentle rescaling, rarely cuts layers
#     $STEPS=100  $LR=0.01  $BATCH_SIZE=4  $REG_STRENGTH=0.1
#     $ABLATION_SAMPLES=8
#
#   AGGRESSIVE    — cuts damaging layers, allows scale changes
#     $STEPS=200  $LR=0.05  $BATCH_SIZE=4  $REG_STRENGTH=0.01
#     $ABLATION_SAMPLES=16
#
#   MAX AGGRESSION — cuts freely, no regularisation pull toward 1
#     $STEPS=300  $LR=0.1   $BATCH_SIZE=8  $REG_STRENGTH=0.0
#     $ABLATION_SAMPLES=32
# ---------------------------------------------------------------

# Number of Adam steps.
# Conservative: 100 | Aggressive: 200 | Max: 300
$STEPS = 60

# Learning rate for the per-layer scale parameters.
# Conservative: 0.01 | Aggressive: 0.05 | Max: 0.1
$LR = 0.5

# LR warmup: ramps from 0 → $LR linearly, then holds constant.
# 0    — disabled (no warmup).
# 0.1  — 10% of $STEPS (e.g. 6 steps when STEPS=60).  Recommended for
#         perceptual metrics (gram_fft etc.) whose gradient signal is noisy
#         at the start of training.
# 10   — absolute step count (integer ≥ 1).
$WARMUP_STEPS = 0.1

# Denoising samples drawn per step.  Higher = lower variance gradient
# estimate but slower steps.
# Conservative: 4 | Aggressive: 4 | Max: 8
$BATCH_SIZE = 4

# L2 regularisation strength toward scale=1.0.
# Conservative: 0.1 — mostly rescales, rarely cuts.
# Aggressive:   0.01 — allows significant cuts.
# Max:          0.0 — unconstrained, most aggressive cutting.
$REG_STRENGTH = 0.0

# Early-stop patience: stop if val loss hasn't improved for this
# many steps.  Set to 0 to disable.
$PATIENCE = 0

# Fraction of the dataset held out as a validation set.
# 0    — no split: all images used for both train and val (default).
# 0.15 — 15% held out as validation, rest for training.
# Must be 0 or in the range (0, 1) exclusive.
$VAL_FRACTION = 0

# Timestep range for the denoising loss.
# [0, 1000] = full range (safe default).
# [50, 450] = low timesteps (texture / style focus).
# [400, 950] = high timesteps (structure / composition focus).
$MIN_TIMESTEP = 0
$MAX_TIMESTEP = 1000

# Image size: resize training images to this square size before
# VAE encoding.  Should match the resolution used during training.
$IMAGE_SIZE = 1024

# Layer filter.  Leave blank to tune all layers.
# "!adaln_modulation" — skip adaln layers (recommended if the
#   LoRA was trained without adaln).
# "self_attn, cross_attn" — attention layers only.
$MODULE_FILTER = ""

# Ablation cut pass: number of denoising samples used to test each
# layer for cutting.  A layer is cut only if zeroing it improves
# (or ties) val loss — no fixed threshold.
# Conservative: 8 | Aggressive: 16 | Max: 32
# 0 — skip ablation entirely (no layers will be cut).
$ABLATION_SAMPLES = 0

# Skip the Adam optimisation step and go straight to the ablation cut pass.
# $true  — cut-only mode: tests each layer at scale=1.0, cuts anything that
#           doesn't help.  Produces real cuts even when optimisation never cuts.
# $false — full mode: optimise scales first, then ablate (default).
$SKIP_OPTIMIZATION = $false

# Granularity for both optimisation and ablation.
# $false — layer-level: one scale per LoRA layer (~300+ parameters, default).
# $true  — block-level: one scale shared across all layers in each DiT block
#           (28 parameters).  Coarser but far faster and more robust.
$BLOCK_LEVEL = $false

# Magnitude threshold cut: zero any layer/block whose learned |scale| falls
# below this value after optimisation (before the ablation pass).
# No forward passes required — essentially free.
# 0.0  — disabled (default, no threshold cutting).
# 0.05 — cut layers/blocks at less than 5% of original strength (recommended).
# 0.1  — more aggressive; cuts anything at less than 10%.
$SCALE_THRESHOLD = 0.05

# Metric used for both optimisation and ablation.
# "mse"      — fast single-step denoising MSE; no extra dependencies.
# "lpips"    — VGG perceptual distance.  Requires: pip install lpips
#              Spatially biased; misses fine texture and adverse features.
# "dists"    — Structure + texture similarity.  Requires: pip install piq
# "gram"     — Multi-scale VGG Gram matrix loss.  No extra install needed.
#              Fully positionally invariant.  Best for stylistic LoRAs:
#              blur, grain, glitch, DoF, lens artefacts, tonal style.
# "fft"      — 2-D FFT log-power-spectrum.  No network, near-zero cost.
#              Perfectly shift-invariant.  Best for blur/grain/noise/glitch.
# "gram_fft" — Gram + FFT combined.  Most comprehensive style metric.
#              Recommended for adverse-image / heavily stylised LoRAs.
$METRIC = "gram_fft"

# Preview image generation.
# Save a fully denoised image at each new optimisation best and each accepted cut.
# Files land in the output directory as <stem>_<unix_time>_<stage_label>.png.
#
# $PREVIEW_SAMPLER:
#   "euler"            — deterministic ODE step (fast, slightly flat).
#   "euler_ancestral"  — stochastic / ER_SDE: reconstructs x0, re-noises each step.
#                        More natural textures; recommended for Anima.
#
# $PREVIEW_SCHEDULE:
#   "uniform" — linearly-spaced timesteps.
#   "beta"    — Beta(0.6,0.6) schedule (denser near t≈0.5); matches ComfyUI beta.
#               Requires: pip install scipy  (falls back to uniform if missing).
#
# $PREVIEW_PROMPT          — positive prompt for preview generation.
#                            Leave blank to use the first training caption.
# $PREVIEW_NEGATIVE_PROMPT — negative prompt (only used when CFG scale > 1).
# $PREVIEW_CFG_SCALE       — classifier-free guidance scale.
#                            1.0  = disabled (single forward pass per step).
#                            5.0  = typical Anima value (two passes per step).
#                            Range 3.5–7.0 recommended.
$SAVE_PREVIEWS            = $true
$PREVIEW_STEPS            = 30
$PREVIEW_SAMPLER          = "er_sde"
$PREVIEW_SCHEDULE         = "beta"
$PREVIEW_CFG_SCALE        = 4.0
$PREVIEW_NEGATIVE_PROMPT  = "bad hands, missing finger, bad anatomy, sketch, wip, unfinished, chromatic aberration, signature, watermark, artist name, censored, censor bar, artist name, text, sound effects, speech bubble, retro graphics, lowpoly, blocky graphics, pixel art"

$PRECISION = "bfloat16"

# ---------------------------------------------------------------
# Resolve Python
# ---------------------------------------------------------------
$pythonExe = if (Test-Path "$PSScriptRoot\$VENV_DIR\Scripts\python.exe") {
    "$PSScriptRoot\$VENV_DIR\Scripts\python.exe"
} else {
    "python"
}

# ---------------------------------------------------------------
# Pass-through: if any args were supplied, relay them directly
# ---------------------------------------------------------------
if ($args.Count -gt 0) {
    $env:PYTHONPATH      = $PSScriptRoot
    $env:SD_SCRIPTS_PATH = $SD_SCRIPTS_PATH
    & $pythonExe "$PSScriptRoot\tools\tune_lora_scales.py" `
        --dit_path   $DIT_PATH   `
        --vae_path   $VAE_PATH   `
        --qwen3_path $QWEN3_PATH `
        $(if ($T5_PATH) { "--t5_tokenizer_path", $T5_PATH }) `
        @args
    exit $LASTEXITCODE
}

# ---------------------------------------------------------------
# Build argument list
# ---------------------------------------------------------------
$pyArgs = @(
    "$PSScriptRoot\tools\tune_lora_scales.py",
    "--lora_file",     $LORA_FILE,
    "--data_dir",      $DATA_DIR,
    "--dit_path",      $DIT_PATH,
    "--vae_path",      $VAE_PATH,
    "--qwen3_path",    $QWEN3_PATH,
    "--steps",         $STEPS,
    "--lr",            $LR,
    "--warmup_steps",  $WARMUP_STEPS,
    "--batch_size",    $BATCH_SIZE,
    "--reg_strength",     $REG_STRENGTH,
    "--patience",         $PATIENCE,
    "--val_fraction",     $VAL_FRACTION,
    "--ablation_samples", $ABLATION_SAMPLES,
    "--min_timestep",     $MIN_TIMESTEP,
    "--max_timestep",  $MAX_TIMESTEP,
    "--image_size",    $IMAGE_SIZE,
    "--precision",     $PRECISION
)

if ($T5_PATH) {
    $pyArgs += "--t5_tokenizer_path", $T5_PATH
}
if ($OUTPUT) {
    $pyArgs += "--output", $OUTPUT
}
if ($MODULE_FILTER) {
    $pyArgs += "--module_filter", $MODULE_FILTER
}
if ($SKIP_OPTIMIZATION) {
    $pyArgs += "--skip_optimization"
}
if ($BLOCK_LEVEL) {
    $pyArgs += "--block_level"
}
if ($SCALE_THRESHOLD -gt 0) {
    $pyArgs += "--scale_threshold", $SCALE_THRESHOLD
}
if ($METRIC -ne "mse") {
    $pyArgs += "--metric", $METRIC
}
if ($SAVE_PREVIEWS) {
    $pyArgs += "--save_previews"
    $pyArgs += "--preview_steps",    $PREVIEW_STEPS
    $pyArgs += "--preview_sampler",  $PREVIEW_SAMPLER
    $pyArgs += "--preview_schedule", $PREVIEW_SCHEDULE
    $pyArgs += "--preview_cfg_scale", $PREVIEW_CFG_SCALE
    if ($PREVIEW_PROMPT) {
        $pyArgs += "--preview_prompt", $PREVIEW_PROMPT
    }
    if ($PREVIEW_NEGATIVE_PROMPT) {
        $pyArgs += "--preview_negative_prompt", $PREVIEW_NEGATIVE_PROMPT
    }
}

# ---------------------------------------------------------------
# Print summary
# ---------------------------------------------------------------
Write-Host ""
Write-Host "=== LoRA Scale Optimisation ==="
Write-Host "  LoRA file       : $LORA_FILE"
Write-Host "  Training data   : $DATA_DIR"
Write-Host "  DiT             : $DIT_PATH"
Write-Host "  VAE             : $VAE_PATH"
Write-Host "  Qwen3           : $QWEN3_PATH"
Write-Host "  Output          : $(if ($OUTPUT) { $OUTPUT } else { '<auto: _tuned.safetensors>' })"
Write-Host "  Mode            : $(if ($SKIP_OPTIMIZATION) { 'cut-only (skip optimisation)' } else { 'optimise + ablate' })"
Write-Host "  Steps           : $(if ($SKIP_OPTIMIZATION) { 'N/A' } else { $STEPS })"
Write-Host "  LR              : $(if ($SKIP_OPTIMIZATION) { 'N/A' } else { $LR })"
Write-Host "  Warmup          : $(if ($SKIP_OPTIMIZATION) { 'N/A' } elseif ($WARMUP_STEPS -gt 0) { $WARMUP_STEPS } else { 'disabled' })"
Write-Host "  Batch size      : $BATCH_SIZE"
Write-Host "  Reg strength    : $(if ($SKIP_OPTIMIZATION) { 'N/A' } else { $REG_STRENGTH })"
Write-Host "  Patience        : $(if ($PATIENCE -gt 0) { $PATIENCE } else { 'disabled' })"
Write-Host "  Val fraction    : $VAL_FRACTION"
Write-Host "  Ablation samples: $(if ($ABLATION_SAMPLES -gt 0) { $ABLATION_SAMPLES } else { 'disabled' })"
Write-Host "  Ablation metric : $METRIC"
Write-Host "  Granularity     : $(if ($BLOCK_LEVEL) { 'block (0-27)' } else { 'layer' })"
Write-Host "  Scale threshold : $(if ($SCALE_THRESHOLD -gt 0) { $SCALE_THRESHOLD } else { 'disabled' })"
Write-Host "  Previews        : $(if ($SAVE_PREVIEWS) { "$PREVIEW_STEPS steps / $PREVIEW_SAMPLER / $PREVIEW_SCHEDULE / cfg=$PREVIEW_CFG_SCALE" } else { 'disabled' })"
Write-Host "  Preview prompt  : $(if ($PREVIEW_PROMPT) { $PREVIEW_PROMPT } else { '<first training caption>' })"
Write-Host "  Negative prompt : $(if ($PREVIEW_NEGATIVE_PROMPT) { $PREVIEW_NEGATIVE_PROMPT } else { '(none)' })"
Write-Host "  Timestep range  : [$MIN_TIMESTEP, $MAX_TIMESTEP]"
Write-Host "  Image size      : $IMAGE_SIZE"
Write-Host "  Module filter   : $(if ($MODULE_FILTER) { $MODULE_FILTER } else { '(all layers)' })"
Write-Host "  Precision       : $PRECISION"
Write-Host ""

$env:PYTHONPATH      = $PSScriptRoot
$env:SD_SCRIPTS_PATH = $SD_SCRIPTS_PATH
& $pythonExe @pyArgs

Write-Host ""
if ($LASTEXITCODE -ne 0) {
    Write-Host "FAILED with error code $LASTEXITCODE."
} else {
    $outPath = if ($OUTPUT) { $OUTPUT } else {
        $stem = [System.IO.Path]::GetFileNameWithoutExtension($LORA_FILE)
        $dir  = [System.IO.Path]::GetDirectoryName($LORA_FILE)
        "$dir\${stem}_tuned.safetensors"
    }
    Write-Host "Done.  Tuned LoRA saved to:"
    Write-Host "  $outPath"
    Write-Host ""
    Write-Host "Tip: a _scales.csv file was saved alongside the output."
    Write-Host "     Open it to inspect per-layer scales and identify cut layers."
}
