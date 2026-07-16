"""
TrainTrain — Configuration constants and module-level setup.

This module defines all the config constants, visibility flags, and
column groupings used by the frontend and trainer modules.

Previously this file contained the Gradio UI.  The UI has been moved
to a vanilla HTML/JS/CSS frontend served by frontend/server.py.
"""

import os
from trainer import trainer

# Anima-only modes
MODES = ["LoRA", "ADDifT", "Multi-ADDifT"]

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

# Anima model paths
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
texture_mode        = ["texture_mode",        "CH", None, True, bool, LORA_MDIFF]
texture_min_tile   = ["texture_min_tile(px)", "TX", None, 256, int, LORA_MDIFF]
texture_max_tile   = ["texture_max_tile(px)", "TX", None, 1024, int, LORA_MDIFF]
texture_tile_snap  = ["texture_tile_snap(px)", "TX", None, 128, int, LORA_MDIFF]
texture_min_shift  = ["texture_min_shift",    "TX", None, 0.5, float, LORA_MDIFF]
texture_max_shift  = ["texture_max_shift",    "TX", None, 3.0, float, LORA_MDIFF]
texture_tile_step_epochs  = ["texture_tile_step_epochs",  "TX", None, 5, int, LORA_MDIFF]
texture_shift_step_epochs = ["texture_shift_step_epochs", "TX", None, 5, int, LORA_MDIFF]
texture_feather_latent_px = ["texture_feather_latent_px", "TX", None, 2, int, LORA_MDIFF]
texture_mask_directory = ["texture_mask_directory", "TX", None, "", str, LORA_MDIFF]
texture_energy_threshold = ["texture_energy_threshold", "TX", None, 0, float, LORA_MDIFF]
texture_avoid_masked = ["texture_avoid_masked", "CH", None, True, bool, LORA_MDIFF]
texture_crop_aspect = ["texture_crop_aspect(e.g. 2:1,3:1)", "TX", None, "", str, LORA_MDIFF]
save_per_steps      = ["save_per_steps",     "TX", None, 0,    int,   ALL]
save_precision      = ["save_precision",     "DD", PRECISION_TYPES[:3], "fp16", str, ALL]
diff_revert_original_target = ["diff_revert_original_target","CH", None, False, bool, DIFF]
diff_use_diff_mask  = ["diff_use_diff_mask", "CH", None, False, bool, DIFF]
train_fixed_timsteps_in_batch = ["train_fixed_timsteps_in_batch","CH", None, False, bool, ALL]
train_repeat        = ["train_repeat",       "TX", None, 1,    int,   ALL]
gradient_accumulation_steps = ["gradient_accumulation_steps","TX", None, "1", str, ALL]
train_min_timesteps = ["train_min_timesteps","TX", None, 0,    int,   ALL]
train_max_timesteps = ["train_max_timesteps","TX", None, 1000, int,   ALL]
TIMESTEP_DISTRIBUTIONS = ["uniform", "flow_shift", "logit_normal", "cosmap", "beta"]
train_timestep_distribution = ["train_timestep_distribution", "DD", TIMESTEP_DISTRIBUTIONS, "flow_shift", str, ALL]
train_ts_dist_params = ["train_ts_dist_params(e.g. mean=0.0 std=1.0)", "TX", None, "", str, ALL]
train_ts_schedule   = ["train_ts_schedule",   "ML", None, "",   str,   ALL]
train_hybrid_mode   = ["train_hybrid_mode(legacy, unused)",   "CH", None, False, bool, ALLN]
network_module_filter = ["network_module_filter(regex, !prefix=exclude)", "TX", None, "", str, ALL]

# ── Score smoothing (Section 3.1: score smoothing via NN regularization) ──
score_smoothing_penalty      = ["score_smoothing_penalty",      "TX", None, 0.0,    float, ALL]
score_smoothing_kappa        = ["score_smoothing_kappa",        "TX", None, 1.44,   float, ALL]
score_smoothing_mc           = ["score_smoothing_mc",           "CH", None, False,  bool,  ALL]
score_smoothing_mc_samples   = ["score_smoothing_mc_samples",   "TX", None, 4,      int,   ALL]

# Column groupings
r_column1 = [network_rank, network_alpha, lora_data_directory, diff_target_name, lora_trigger_word]
r_column2 = [image_size, train_iterations, train_batch_size, train_learning_rate]
r_column3 = [train_optimizer, train_optimizer_settings, train_lr_scheduler, train_lr_scheduler_settings, save_lora_name]

# Bucket / Texture parameters
b_column = [image_buckets_step, image_mirroring, image_use_filename_as_tag, image_disable_upscale,
            train_fixed_timsteps_in_batch, texture_mode, texture_min_tile, texture_max_tile, texture_tile_snap,
            texture_min_shift, texture_max_shift, texture_tile_step_epochs, texture_shift_step_epochs,
            texture_feather_latent_px, texture_mask_directory, texture_energy_threshold,
            texture_avoid_masked, texture_crop_aspect]

# Other Options (everything else)
o_column1 = [train_seed, train_loss_function, save_per_steps,
             diff_revert_original_target, diff_use_diff_mask]
o_column2 = [train_model_precision, train_lora_precision, save_precision,
             train_repeat, gradient_accumulation_steps, use_gradient_checkpointing]

o_ts_column    = [train_min_timesteps, train_max_timesteps, train_timestep_distribution, train_ts_dist_params, train_ts_schedule, train_hybrid_mode]
o_layer_column = [network_module_filter]

model_column = [qwen3_path, t5_tokenizer_path]

# Score smoothing column (paper-inspired regularization)
ss_column = [score_smoothing_penalty, score_smoothing_kappa, score_smoothing_mc, score_smoothing_mc_samples]

trainer.all_configs = model_column + r_column1 + r_column2 + r_column3 + b_column + o_column1 + o_column2 + o_ts_column + o_layer_column + ss_column
