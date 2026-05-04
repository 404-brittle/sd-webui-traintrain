/* ================================================================
   TrainTrain — Frontend Application
   ================================================================ */

// ── Configuration (mirrors scripts/traintrain.py) ──────────────

const MODES = ["LoRA", "ADDifT", "Multi-ADDifT"];

const PRECISION_TYPES = ["fp32", "bf16", "fp16", "float32", "bfloat16", "float16"];
const NETWORK_DIMS = Array.from({length: 11}, (_, i) => String(2 ** i));
const NETWORK_ALPHAS = Array.from({length: 16}, (_, i) => String(2 ** (i - 5)));
const IMAGESTEPS = Array.from({length: 10}, (_, i) => String(i * 64));
const SEP = "--------------------------";
const OPTIMIZERS = [
  "AdamW", "AdamW8bit", "AdaFactor", "Lion", "Prodigy", SEP,
  "DadaptAdam", "DadaptLion", "DAdaptAdaGrad", "DAdaptAdan", "DAdaptSGD", SEP,
  "Adam8bit", "SGDNesterov8bit", "Lion8bit", "PagedAdamW8bit", "PagedLion8bit", SEP,
  "RAdamScheduleFree", "AdamWScheduleFree", "SGDScheduleFree", SEP,
  "CAME", "Tiger", "AdamMini",
  "PagedAdamW", "PagedAdamW32bit", "SGDNesterov", "Adam"
];
const LOSS_FUNCTIONS = ["MSE", "L1", "Smooth-L1"];
const SCHEDULERS = [
  "linear", "cosine_annealing", "cosine_annealing_with_restarts", "linear", "cosine",
  "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup",
  "piecewise_constant", "exponential", "step", "multi_step",
  "reduce_on_plateau", "cyclic", "one_cycle"
];
const ATTN_MODES = ["torch", "flash", "xformers"];
const TIMESTEP_DISTRIBUTIONS = ["uniform", "flow_shift", "logit_normal", "cosmap", "beta"];
const FD_QUEUE_MODES = ["snapshot", "online_accum", "ema"];
const FD_EVICTION_MODES = ["fifo", "diversity", "guided"];

// Visibility flags: [LoRA, ADDifT, Multi-ADDifT]
const ALL       = [true,  true,  true ];
const LORA      = [true,  false, false];
const ADIFT     = [false, true,  false];
const MDIFF     = [false, false, true ];
const LORA_MDIFF = [true, false, true ];
const DIFF      = [false, true,  true ];
const ALLN      = [false, false, false];

// ── Config definitions ─────────────────────────────────────────
// Each entry: [name, uitype, choices, default, dtype, visibility]

const ALL_CONFIGS = [
  ["lora_data_directory",   "TX", null, "",              "str",   LORA_MDIFF],
  ["lora_trigger_word",     "TX", null, "",              "str",   LORA_MDIFF],
  ["diff_target_name",      "TX", null, "",              "str",   MDIFF],
  ["network_rank",           "DD", NETWORK_DIMS.slice(2), "16",    "int",   ALL],
  ["network_alpha",          "DD", NETWORK_ALPHAS,        "8",     "float", ALL],
  ["image_size(height, width)","TX", null, "512",         "str",   ALL],
  ["train_iterations",       "TX", null, 1000,            "int",   ALL],
  ["train_batch_size",       "TX", null, 2,               "int",   ALL],
  ["train_learning_rate",    "TX", null, "1e-4",          "float", ALL],
  ["train_optimizer",        "DD", OPTIMIZERS,            "adamw", "str",   ALL],
  ["train_optimizer_settings","TX", null, "",             "str",   ALL],
  ["train_lr_scheduler",     "DD", SCHEDULERS,            "cosine","str",   ALL],
  ["train_lr_scheduler_settings","TX", null, "",          "str",   ALL],
  ["save_lora_name",         "TX", null, "",              "str",   ALL],
  ["use_gradient_checkpointing","CH", null, false,        "bool",  ALL],
  ["qwen3_path",         "TX", null, "", "str", ALL],
  ["t5_tokenizer_path",  "TX", null, "", "str", ALL],
  ["train_loss_function", "DD", LOSS_FUNCTIONS, "MSE", "str", ALL],
  ["train_seed",          "TX", null, -1,    "int",   ALL],
  ["train_model_precision","DD", PRECISION_TYPES.slice(0,3), "bf16", "str", ALL],
  ["train_lora_precision", "DD", PRECISION_TYPES.slice(0,3), "fp32", "str", ALL],
  ["image_buckets_step",  "DD", IMAGESTEPS, "256", "int",  LORA_MDIFF],
  ["image_mirroring",     "CH", null, false, "bool", LORA_MDIFF],
  ["image_use_filename_as_tag","CH", null, false, "bool", LORA_MDIFF],
  ["image_disable_upscale","CH", null, false, "bool", LORA_MDIFF],
  ["texture_mode",        "CH", null, false, "bool", LORA_MDIFF],
  ["texture_feather_latent_px", "TX", null, 2, "int", LORA_MDIFF],
  ["texture_mask_directory", "TX", null, "", "str", LORA_MDIFF],
  ["texture_tile_scale",  "TX", null, 1.0, "float", LORA_MDIFF],
  ["texture_tile_resolution", "TX", null, 0, "int", LORA_MDIFF],
  ["texture_energy_threshold", "TX", null, 0, "float", LORA_MDIFF],
  ["save_per_steps",      "TX", null, 0,    "int",   ALL],
  ["save_precision",      "DD", PRECISION_TYPES.slice(0,3), "fp16", "str", ALL],
  ["diff_revert_original_target","CH", null, false, "bool", DIFF],
  ["diff_use_diff_mask",  "CH", null, false, "bool", DIFF],
  ["train_fixed_timsteps_in_batch","CH", null, false, "bool", ALL],
  ["train_repeat",        "TX", null, 1,    "int",   ALL],
  ["gradient_accumulation_steps","TX", null, "1", "str", ALL],
  ["train_min_timesteps", "TX", null, 0,    "int",   ALL],
  ["train_max_timesteps", "TX", null, 1000, "int",   ALL],
  ["train_timestep_distribution", "DD", TIMESTEP_DISTRIBUTIONS, "flow_shift", "str", ALL],
  ["train_ts_dist_params(e.g. mean=0.0 std=1.0)", "TX", null, "", "str", ALL],
  ["train_ts_schedule",   "ML", null, "",   "str",   ALL],
  ["train_hybrid_mode",   "CH", null, false, "bool", ALL],
  ["network_module_filter(regex, !prefix=exclude)", "TX", null, "", "str", ALL],
  ["fd_loss_enable",      "CH", null, false, "bool", ALL],
  ["fd_repr_models(repr models, comma-sep)", "TX", null, "vit_base_patch14_dinov2", "str", ALL],
  ["fd_queue_size",       "TX", null, 50000, "int", ALL],
  ["fd_queue_mode",       "DD", FD_QUEUE_MODES, "online_accum", "str", ALL],
  ["fd_ema_beta",         "TX", null, 0.9999, "float", ALL],
  ["fd_loss_weight",      "TX", null, 0.1, "float", ALL],
  ["fd_fid_norm_eps",     "TX", null, 1e-6, "float", ALL],
  ["fd_eviction_mode",    "DD", FD_EVICTION_MODES, "fifo", "str", ALL],
  ["fd_guidance_strength","TX", null, 0.5, "float", ALL],
  ["fd_n_clusters",       "TX", null, 20, "int", ALL],
  ["fd_cluster_log_interval","TX", null, 0, "int", ALL],
  ["fd_store_source_images","CH", null, false, "bool", ALL],
  ["fd_enqueue_generated",  "CH", null, true,  "bool", ALL],
  ["fd_pause_interval",   "TX", null, 0, "int", ALL],
  ["fd_warmup_steps",     "TX", null, 0, "int", ALL],
];

// ── Column groupings ───────────────────────────────────────────

const R_COLUMN1 = ["network_rank", "network_alpha", "lora_data_directory", "diff_target_name", "lora_trigger_word"];
const R_COLUMN2 = ["image_size(height, width)", "train_iterations", "train_batch_size", "train_learning_rate"];
const R_COLUMN3 = ["train_optimizer", "train_optimizer_settings", "train_lr_scheduler", "train_lr_scheduler_settings", "save_lora_name"];

// Bucket / Texture parameters
const B_COLUMN = ["image_buckets_step", "image_mirroring", "image_use_filename_as_tag", "image_disable_upscale",
                  "train_fixed_timsteps_in_batch", "texture_mode", "texture_feather_latent_px", "texture_mask_directory",
                  "texture_tile_scale", "texture_tile_resolution", "texture_energy_threshold"];

// FD-Loss parameters
const F_COLUMN = ["fd_loss_enable", "fd_repr_models(repr models, comma-sep)", "fd_queue_size", "fd_queue_mode",
                  "fd_ema_beta", "fd_loss_weight", "fd_fid_norm_eps",
                  "fd_eviction_mode", "fd_guidance_strength", "fd_n_clusters",
                  "fd_cluster_log_interval", "fd_store_source_images",
                  "fd_enqueue_generated", "fd_pause_interval", "fd_warmup_steps"];

// Other Options (everything else)
const O_COLUMN1 = ["train_seed", "train_loss_function", "save_per_steps",
                   "diff_revert_original_target", "diff_use_diff_mask"];
const O_COLUMN2 = ["train_model_precision", "train_lora_precision", "save_precision",
                   "train_repeat", "gradient_accumulation_steps", "use_gradient_checkpointing"];

const O_TS_COLUMN = ["train_min_timesteps", "train_max_timesteps", "train_timestep_distribution", "train_ts_dist_params(e.g. mean=0.0 std=1.0)", "train_ts_schedule", "train_hybrid_mode"];
const O_LAYER_COLUMN = ["network_module_filter(regex, !prefix=exclude)"];

// ── Dist colors ────────────────────────────────────────────────

const DIST_COLORS = {
  "uniform":      "#60a5fa",
  "flow_shift":   "#4ade80",
  "logit_normal": "#f472b6",
  "cosmap":       "#fb923c",
  "beta":         "#a78bfa",
};

// ── State ──────────────────────────────────────────────────────

let currentMode = "LoRA";
let origImageData = null;
let targImageData = null;

// FD Cluster pending actions count (uncommitted changes tracker)
let fdPendingCount = 0;

// ── Helpers ────────────────────────────────────────────────────

function getConfig(name) {
  return ALL_CONFIGS.find(c => c[0] === name);
}

function labelFromName(name) {
  return name.replace(/_/g, ' ');
}

// ── Build Form Controls ────────────────────────────────────────

function buildField(config) {
  const [name, uitype, choices, defaultVal, dtype, visibility] = config;
  const modeIdx = MODES.indexOf(currentMode);
  const visible = visibility[modeIdx];
  const label = labelFromName(name);

  const wrapper = document.createElement('div');
  wrapper.className = 'field-group';
  wrapper.dataset.configName = name;
  wrapper.style.display = visible ? '' : 'none';

  const lbl = document.createElement('label');
  lbl.textContent = label;
  wrapper.appendChild(lbl);

  let input;

  if (uitype === 'DD') {
    input = document.createElement('select');
    input.className = 'form-select';
    input.autocomplete = 'off';
    (choices || []).forEach(opt => {
      const o = document.createElement('option');
      o.value = opt;
      o.textContent = opt;
      if (String(opt).toLowerCase() === String(defaultVal).toLowerCase()) {
        o.selected = true;
      }
      input.appendChild(o);
    });
  } else if (uitype === 'CH') {
    wrapper.className = 'form-checkbox';
    wrapper.dataset.configName = name;
    input = document.createElement('input');
    input.type = 'checkbox';
    input.checked = !!defaultVal;
    const span = document.createElement('span');
    span.textContent = label;
    wrapper.appendChild(input);
    wrapper.appendChild(span);
    wrapper.removeChild(lbl);
    return wrapper;
  } else if (uitype === 'ML') {
    input = document.createElement('textarea');
    input.className = 'form-input';
    input.autocomplete = 'off';
    input.rows = 6;
    input.value = defaultVal !== null && defaultVal !== undefined ? String(defaultVal) : '';
  } else {
    input = document.createElement('input');
    input.type = 'text';
    input.className = 'form-input';
    input.autocomplete = 'off';
    input.value = defaultVal !== null && defaultVal !== undefined ? String(defaultVal) : '';
  }

  input.dataset.configName = name;
  wrapper.appendChild(input);
  return wrapper;
}

/**
 * Reset all form fields to their default values (from ALL_CONFIGS).
 * Called on page load to prevent browser autofill from restoring stale values.
 */
function resetFormToDefaults() {
  ALL_CONFIGS.forEach(config => {
    const [name, uitype, , defaultVal] = config;
    // Check dedicated HTML inputs first (qwen3_path, t5_tokenizer_path)
    const dedicatedId = name === 'qwen3_path' ? 'qwen3-path' : (name === 't5_tokenizer_path' ? 't5-path' : null);
    if (dedicatedId) {
      const dedicatedEl = document.getElementById(dedicatedId);
      if (dedicatedEl) {
        dedicatedEl.value = defaultVal !== null && defaultVal !== undefined ? String(defaultVal) : '';
        return;
      }
    }
    const el = document.querySelector(`[data-config-name="${name}"]`);
    if (!el) return;
    if (uitype === 'CH') {
      const cb = el.querySelector('input[type="checkbox"]') || el;
      cb.checked = !!defaultVal;
    } else if (uitype === 'DD') {
      const sel = el.querySelector('select') || el;
      sel.value = String(defaultVal);
    } else {
      const inp = el.querySelector('input, textarea') || el;
      inp.value = defaultVal !== null && defaultVal !== undefined ? String(defaultVal) : '';
    }
  });
  // Also reset the static inputs
  const modelPath = document.getElementById('model-path');
  if (modelPath) modelPath.value = '';
  const vaePath = document.getElementById('vae-path');
  if (vaePath) vaePath.value = '';
  const modeSelect = document.getElementById('mode-select');
  if (modeSelect) modeSelect.value = 'LoRA';
}

function buildSection(containerId, configNames, cols) {
  const container = document.getElementById(containerId);
  container.innerHTML = '';
  container.className = `section-grid cols-${cols}`;
  configNames.forEach(name => {
    const config = getConfig(name);
    if (!config) return;
    const field = buildField(config);
    container.appendChild(field);
  });
}

// ── Get all values as flat array ───────────────────────────────

function collectValues() {
  const values = [];
  ALL_CONFIGS.forEach(config => {
    const [name, uitype] = config;
    // Check dedicated HTML inputs first (qwen3_path, t5_tokenizer_path)
    const dedicatedId = name === 'qwen3_path' ? 'qwen3-path' : (name === 't5_tokenizer_path' ? 't5-path' : null);
    if (dedicatedId) {
      const dedicatedEl = document.getElementById(dedicatedId);
      if (dedicatedEl) {
        let v = dedicatedEl.value;
        if (config[4] === 'int') { v = parseInt(v, 10); if (isNaN(v)) v = config[3]; }
        else if (config[4] === 'float') { v = parseFloat(v); if (isNaN(v)) v = config[3]; }
        values.push(v);
        return;
      }
    }
    const el = document.querySelector(`[data-config-name="${name}"]`);
    if (!el) { values.push(config[3]); return; }
    if (uitype === 'CH') {
      const cb = el.querySelector('input[type="checkbox"]') || el;
      values.push(cb.checked);
    } else if (uitype === 'DD') {
      const sel = el.querySelector('select') || el;
      values.push(sel.value);
    } else {
      const inp = el.querySelector('input, textarea') || el;
      let v = inp.value;
      if (config[4] === 'int') { v = parseInt(v, 10); if (isNaN(v)) v = config[3]; }
      else if (config[4] === 'float') { v = parseFloat(v); if (isNaN(v)) v = config[3]; }
      values.push(v);
    }
  });
  return values;
}

// ── Set values from array ──────────────────────────────────────

function setValues(values) {
  ALL_CONFIGS.forEach((config, i) => {
    const [name, uitype] = config;
    // Check dedicated HTML inputs first (qwen3_path, t5_tokenizer_path)
    const dedicatedId = name === 'qwen3_path' ? 'qwen3-path' : (name === 't5_tokenizer_path' ? 't5-path' : null);
    if (dedicatedId) {
      const dedicatedEl = document.getElementById(dedicatedId);
      if (dedicatedEl) {
        const v = values[i] !== undefined ? values[i] : config[3];
        dedicatedEl.value = String(v);
        return;
      }
    }
    const el = document.querySelector(`[data-config-name="${name}"]`);
    if (!el) return;
    const v = values[i] !== undefined ? values[i] : config[3];
    if (uitype === 'CH') {
      const cb = el.querySelector('input[type="checkbox"]') || el;
      cb.checked = !!v;
    } else if (uitype === 'DD') {
      const sel = el.querySelector('select') || el;
      sel.value = String(v);
    } else {
      const inp = el.querySelector('input, textarea') || el;
      inp.value = String(v);
    }
  });
}

// ── Build JSON payload ─────────────────────────────────────────

function buildPayload() {
  const payload = {};
  ALL_CONFIGS.forEach(config => {
    const [name, uitype] = config;
    const dtype = config[4];
    // Check dedicated HTML inputs first (qwen3_path, t5_tokenizer_path)
    const dedicatedId = name === 'qwen3_path' ? 'qwen3-path' : (name === 't5_tokenizer_path' ? 't5-path' : null);
    if (dedicatedId) {
      const dedicatedEl = document.getElementById(dedicatedId);
      if (dedicatedEl) {
        let v = dedicatedEl.value;
        if (dtype === 'int') { const n = parseInt(v, 10); v = isNaN(n) ? config[3] : n; }
        else if (dtype === 'float') { const n = parseFloat(v); v = isNaN(n) ? config[3] : n; }
        // str: keep as-is (do NOT run isNaN on string paths!)
        payload[name] = v;
        return;
      }
    }
    const el = document.querySelector(`[data-config-name="${name}"]`);
    if (!el) { payload[name] = config[3]; return; }
    if (uitype === 'CH') {
      const cb = el.querySelector('input[type="checkbox"]') || el;
      payload[name] = cb.checked;
    } else if (uitype === 'DD') {
      const sel = el.querySelector('select') || el;
      payload[name] = sel.value;
    } else {
      const inp = el.querySelector('input, textarea') || el;
      let v = inp.value;
      if (dtype === 'int') { const n = parseInt(v, 10); v = isNaN(n) ? config[3] : n; }
      else if (dtype === 'float') { const n = parseFloat(v); v = isNaN(n) ? config[3] : n; }
      // str: keep as-is
      payload[name] = v;
    }
  });
  payload['mode'] = document.getElementById('mode-select').value;
  payload['model'] = document.getElementById('model-path').value;
  payload['vae'] = document.getElementById('vae-path').value;
  return payload;
}

// ── Apply payload to form ──────────────────────────────────────

function applyPayload(data) {
  if (data.mode !== undefined) document.getElementById('mode-select').value = data.mode;
  if (data.model !== undefined) document.getElementById('model-path').value = data.model;
  if (data.vae !== undefined) document.getElementById('vae-path').value = data.vae;
  // Handle dedicated Qwen3/T5 inputs (not built by buildSection)
  if (data.qwen3_path !== undefined) document.getElementById('qwen3-path').value = data.qwen3_path;
  if (data.t5_tokenizer_path !== undefined) document.getElementById('t5-path').value = data.t5_tokenizer_path;
  ALL_CONFIGS.forEach(config => {
    const [name] = config;
    if (data[name] === undefined) return;
    const el = document.querySelector(`[data-config-name="${name}"]`);
    if (!el) return;
    const [, uitype] = config;
    if (uitype === 'CH') {
      const cb = el.querySelector('input[type="checkbox"]') || el;
      cb.checked = !!data[name];
    } else if (uitype === 'DD') {
      const sel = el.querySelector('select') || el;
      sel.value = String(data[name]);
    } else {
      const inp = el.querySelector('input, textarea') || el;
      inp.value = String(data[name]);
    }
  });
  updateModeVisibility();
  updateTsPreview();
  updateLayerPreview();
}

// ── Mode visibility ────────────────────────────────────────────

function updateModeVisibility() {
  const modeIdx = MODES.indexOf(currentMode);
  ALL_CONFIGS.forEach(config => {
    const [name] = config;
    const visible = config[5][modeIdx];
    const el = document.querySelector(`[data-config-name="${name}"]`);
    if (el) el.style.display = visible ? '' : 'none';
  });
  document.getElementById('diff-section').style.display = (modeIdx === 1) ? '' : 'none';
}

// ── Timestep Distribution Preview ──────────────────────────────

function computeTsDensity(distType, tsLo, tsHi, distParams, nBins) {
  nBins = nBins || 50;
  tsLo = Math.max(0, parseInt(tsLo) || 0);
  tsHi = Math.max(tsLo + 1, Math.min(1000, parseInt(tsHi) || 1000));
  const span = tsHi - tsLo;
  const binSize = 1000.0 / nBins;
  const density = new Array(nBins).fill(0);
  const params = {};
  (distParams || '').replace(/,/g, ' ').split(/\s+/).forEach(item => {
    const eq = item.indexOf('=');
    if (eq > 0) params[item.slice(0, eq).trim()] = parseFloat(item.slice(eq + 1).trim());
  });

  if (distType === 'uniform') {
    for (let b = 0; b < nBins; b++) {
      const lo = b * binSize, hi = lo + binSize;
      density[b] = Math.max(0, Math.min(tsHi, hi) - Math.max(tsLo, lo)) / binSize;
    }
  } else if (distType === 'flow_shift') {
    const shift = Math.max(1e-3, params['shift'] || 3.0);
    const fsCdf = (s) => s * shift / (1.0 + (shift - 1.0) * s);
    for (let b = 0; b < nBins; b++) {
      const lo = b * binSize, hi = lo + binSize;
      if (hi <= tsLo || lo >= tsHi) continue;
      const sLo = Math.max(0, (Math.max(lo, tsLo) - tsLo) / span);
      const sHi = Math.min(1, (Math.min(hi, tsHi) - tsLo) / span);
      if (sHi > sLo) density[b] = (fsCdf(sHi) - fsCdf(sLo)) / (binSize / 1000.0);
    }
  } else if (distType === 'logit_normal') {
    const mean = params['mean'] || 0.0, std = Math.max(0.01, params['std'] || 1.0);
    const INV_SQRT2PI = 1.0 / Math.sqrt(2.0 * Math.PI);
    const normalPdf = (x) => INV_SQRT2PI * Math.exp(-0.5 * x * x);
    for (let b = 0; b < nBins; b++) {
      const lo = b * binSize, hi = lo + binSize;
      if (hi <= tsLo || lo >= tsHi) continue;
      const loT = Math.max(tsLo + 0.001 * span, lo), hiT = Math.min(tsHi - 0.001 * span, hi);
      if (hiT <= loT) continue;
      let val = 0;
      for (let i = 0; i < 8; i++) {
        const t = loT + (hiT - loT) * (i + 0.5) / 8;
        const sigma = Math.max(1e-6, Math.min(1 - 1e-6, (t - tsLo) / span));
        const logit = Math.log(sigma / (1 - sigma));
        val += normalPdf((logit - mean) / std) / (std * sigma * (1 - sigma)) / span * (hiT - loT) / 8;
      }
      density[b] = val;
    }
  } else if (distType === 'cosmap') {
    const TWO_OVER_PI = 2.0 / Math.PI;
    for (let b = 0; b < nBins; b++) {
      const lo = b * binSize, hi = lo + binSize;
      if (hi <= tsLo || lo >= tsHi) continue;
      const loT = Math.max(tsLo, lo), hiT = Math.min(tsHi, hi);
      if (hiT <= loT) continue;
      let val = 0;
      for (let i = 0; i < 8; i++) {
        const t = loT + (hiT - loT) * (i + 0.5) / 8;
        const sigma = Math.max(1e-6, Math.min(1 - 1e-6, (t - tsLo) / span));
        val += (TWO_OVER_PI / (sigma ** 2 + (1 - sigma) ** 2)) / span * (hiT - loT) / 8;
      }
      density[b] = val;
    }
  } else if (distType === 'beta') {
    const a = Math.max(0.01, params['alpha'] || 0.5), b = Math.max(0.01, params['beta'] || 0.5);
    const logGamma = (x) => x <= 0 ? 0 : (x - 0.5) * Math.log(x) - x + 0.5 * Math.log(2 * Math.PI);
    const logNorm = logGamma(a + b) - logGamma(a) - logGamma(b);
    for (let bn = 0; bn < nBins; bn++) {
      const lo = bn * binSize, hi = lo + binSize;
      if (hi <= tsLo || lo >= tsHi) continue;
      const loT = Math.max(tsLo, lo), hiT = Math.min(tsHi, hi);
      if (hiT <= loT) continue;
      let val = 0;
      for (let i = 0; i < 8; i++) {
        const t = loT + (hiT - loT) * (i + 0.5) / 8;
        const sigma = Math.max(1e-6, Math.min(1 - 1e-6, (t - tsLo) / span));
        val += Math.exp(logNorm + (a - 1) * Math.log(sigma) + (b - 1) * Math.log(1 - sigma)) / span * (hiT - loT) / 8;
      }
      density[bn] = val;
    }
  }

  const maxD = Math.max(...density, 1);
  return density.map(d => d / maxD);
}

function renderTsPreview() {
  const distEl = document.querySelector('[data-config-name="train_timestep_distribution"] select');
  const minEl = document.querySelector('[data-config-name="train_min_timesteps"] input');
  const maxEl = document.querySelector('[data-config-name="train_max_timesteps"] input');
  const paramsEl = document.querySelector('[data-config-name="train_ts_dist_params(e.g. mean=0.0 std=1.0)"] input');
  const distType = distEl ? distEl.value : 'flow_shift';
  const tsMin = minEl ? minEl.value : 0;
  const tsMax = maxEl ? maxEl.value : 1000;
  const distParams = paramsEl ? paramsEl.value : '';
  const tsLo = Math.max(0, parseInt(tsMin) || 0);
  const tsHi = Math.max(tsLo + 1, Math.min(1000, parseInt(tsMax) || 1000));
  const nBins = 50;
  const density = computeTsDensity(distType, tsLo, tsHi, distParams, nBins);
  const W = 500, H = 74, barAreaH = 54, barW = W / nBins, binSize = 1000.0 / nBins;
  const color = DIST_COLORS[distType] || '#60a5fa';
  let bars = '';
  for (let b = 0; b < nBins; b++) {
    const binLo = b * binSize, binHi = binLo + binSize;
    const inRange = binLo < tsHi && binHi > tsLo;
    const x = b * barW, h = Math.max(0, density[b] * barAreaH), y = barAreaH - h;
    bars += `<rect x="${x.toFixed(1)}" y="${y.toFixed(1)}" width="${(barW - 0.8).toFixed(1)}" height="${h.toFixed(1)}" fill="${inRange ? color : '#252525'}" rx="0.5"/>`;
  }
  let ticks = '';
  [[0, '0'], [250, '250'], [500, '500'], [750, '750'], [1000, '1000']].forEach(([tVal, tLbl]) => {
    const x = tVal / 1000.0 * W;
    ticks += `<line x1="${x.toFixed(1)}" y1="${barAreaH}" x2="${x.toFixed(1)}" y2="${barAreaH + 3}" stroke="#555" stroke-width="0.5"/>`;
    ticks += `<text x="${x.toFixed(1)}" y="${barAreaH + 12}" text-anchor="middle" fill="#555" font-size="8">${tLbl}</text>`;
  });
  const loX = tsLo / 1000.0 * W, hiX = tsHi / 1000.0 * W;
  const markers = `<line x1="${loX.toFixed(1)}" y1="0" x2="${loX.toFixed(1)}" y2="${barAreaH}" stroke="#ffffff18" stroke-width="0.8" stroke-dasharray="2,2"/><line x1="${hiX.toFixed(1)}" y1="0" x2="${hiX.toFixed(1)}" y2="${barAreaH}" stroke="#ffffff18" stroke-width="0.8" stroke-dasharray="2,2"/>`;
  const svg = `<svg viewBox="0 0 ${W} ${H}" xmlns="http://www.w3.org/2000/svg" style="width:100%;display:block;">${markers}${bars}${ticks}<line x1="0" y1="${barAreaH}" x2="${W}" y2="${barAreaH}" stroke="#333" stroke-width="0.5"/></svg>`;
  let paramDesc = '';
  const parsedParams = {};
  (distParams || '').replace(/,/g, ' ').split(/\s+/).forEach(item => {
    const eq = item.indexOf('=');
    if (eq > 0) parsedParams[item.slice(0, eq).trim()] = parseFloat(item.slice(eq + 1).trim());
  });
  if (distType === 'flow_shift') paramDesc = ` &nbsp;<span style="color:#666;">shift=${(parsedParams['shift'] || 3.0).toFixed(3)}</span>`;
  else if (distType === 'logit_normal') paramDesc = ` &nbsp;<span style="color:#666;">mean=${(parsedParams['mean'] || 0.0).toFixed(2)}&thinsp; std=${(parsedParams['std'] || 1.0).toFixed(2)}</span>`;
  else if (distType === 'beta') paramDesc = ` &nbsp;<span style="color:#666;">α=${(parsedParams['alpha'] || 0.5).toFixed(2)}&thinsp; β=${(parsedParams['beta'] || 0.5).toFixed(2)}</span>`;
  const header = `<div style="font-size:11px;margin-bottom:4px;font-family:sans-serif;"><span style="color:${color};font-weight:600;">${distType}</span>${paramDesc}&nbsp;&nbsp;<span style="color:#555;font-size:10px;">range&thinsp;${tsLo}–${tsHi}</span></div>`;
  document.getElementById('ts-preview').innerHTML = `<div class="ts-svg-container">${header}${svg}</div>`;
}

function updateTsPreview() { renderTsPreview(); }

// ── Layer Preview ──────────────────────────────────────────────

// Anima LoRA sublayer names (16 per block) — matches trainer/lora.py _ANIMA_BLOCK_SUBLAYERS
const ANIMA_SUBLAYERS = [
  "adaln_modulation_cross_attn_1", "adaln_modulation_cross_attn_2",
  "adaln_modulation_mlp_1", "adaln_modulation_mlp_2",
  "adaln_modulation_self_attn_1", "adaln_modulation_self_attn_2",
  "cross_attn_k_proj", "cross_attn_output_proj",
  "cross_attn_q_proj", "cross_attn_v_proj",
  "mlp_layer1", "mlp_layer2",
  "self_attn_k_proj", "self_attn_output_proj",
  "self_attn_q_proj", "self_attn_v_proj",
];

// Generate all 448 Anima LoRA preview keys: lora_unet_blocks_{b}_{sublayer}
const PREVIEW_KEYS = (() => {
  const keys = [];
  for (let b = 0; b < 28; b++) {
    for (const sl of ANIMA_SUBLAYERS) {
      keys.push(`lora_unet_blocks_${b}_${sl}`);
    }
  }
  return keys;
})();

function matchesModuleFilter(key, filterStr) {
  if (!filterStr || !filterStr.trim()) return true;
  const patterns = filterStr.split(/[,\n]+/).map(p => p.trim()).filter(p => p);
  // If no positive patterns (all are exclusions starting with !), default to include everything
  const hasPositive = patterns.some(p => !p.startsWith('!'));
  let included = !hasPositive; // default: include if only exclusions
  for (const p of patterns) {
    const exclude = p.startsWith('!');
    const pat = exclude ? p.slice(1) : p;
    try {
      const re = new RegExp(pat);
      if (re.test(key)) {
        if (exclude) return false; // exclusion match → hide
        included = true;           // positive match → show
      }
    } catch (e) {}
  }
  return included;
}

function renderLayerPreview() {
  const filterEl = document.querySelector('[data-config-name="network_module_filter(regex, !prefix=exclude)"] input');
  const filterStr = filterEl ? filterEl.value : '';
  const rawPatterns = filterStr.split(/[,\n]+/).map(p => p.trim()).filter(p => p);
  const badPatterns = [];
  for (const p of rawPatterns) {
    const pat = p.startsWith('!') ? p.slice(1) : p;
    try { new RegExp(pat); } catch (e) { badPatterns.push(`${pat}: ${e.message}`); }
  }
  let errorHtml = '';
  if (badPatterns.length > 0) errorHtml = `<div class="layer-error">Invalid regex — ${badPatterns.join('<br>')}</div>`;
  let activeCount = 0, rows = '', prevBlock = null;
  for (const key of PREVIEW_KEYS) {
    const m = key.match(/_blocks_(\d+)_/);
    const block = m ? `B${String(parseInt(m[1])).padStart(2, '0')}` : 'BASE';
    if (block !== prevBlock) {
      rows += `<div class="layer-block-label">── ${block} ──</div>`;
      prevBlock = block;
    }
    const active = matchesModuleFilter(key, filterStr);
    if (active) activeCount++;
    const cls = active ? 'layer-key-active' : 'layer-key-inactive';
    rows += `<span class="${cls}">${key}</span>`;
  }
  const total = PREVIEW_KEYS.length;
  const fracColor = activeCount === total ? '#4ade80' : (activeCount > 0 ? '#facc15' : '#f87171');
  const header = `<div class="layer-preview-header"><span style="color:${fracColor};font-weight:600;">${activeCount}</span><span style="color:#666;">/${total}</span> layers active&nbsp;&nbsp;<span style="color:#555;font-size:10px;">canonical architecture preview</span></div>`;
  document.getElementById('layer-preview').innerHTML = errorHtml + header + `<div class="layer-preview-container">${rows}</div>`;
}

function updateLayerPreview() { renderLayerPreview(); }

// ── API calls ──────────────────────────────────────────────────

// API base URL: use injected value from server, or fall back to same-origin
const API_BASE = (typeof window.TRAINTRAIN_API_BASE !== 'undefined' && window.TRAINTRAIN_API_BASE) || '';

async function apiCall(endpoint, method, body) {
  const opts = { method, headers: {} };
  if (body) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  }
  const resp = await fetch(`${API_BASE}${endpoint}`, opts);
  if (!resp.ok) {
    const text = await resp.text();
    throw new Error(`API error ${resp.status}: ${text}`);
  }
  const ct = resp.headers.get('content-type') || '';
  if (ct.includes('application/json')) return resp.json();
  return resp.text();
}

function showMessage(msg, type) {
  const bar = document.getElementById('message-bar');
  bar.textContent = msg;
  bar.className = 'message-bar ' + (type || 'info');
  bar.style.display = 'block';
}

// ── Start Training ─────────────────────────────────────────────

async function startTraining() {
  const config = buildPayload();
  const mode = document.getElementById('mode-select').value;
  const model = document.getElementById('model-path').value;
  const vae = document.getElementById('vae-path').value;

  // Send the typed named payload directly — no fragile positional args.
  const body = {
    mode: mode,
    model: model,
    vae: vae,
    config: config,
    orig_image: origImageData,
    targ_image: targImageData,
  };

  try {
    showMessage('Starting training...', 'info');
    const result = await apiCall('/api/train', 'POST', body);
    showMessage(result, 'success');
  } catch (e) {
    showMessage('Error: ' + e.message, 'error');
  }
}

async function stopTraining(save) {
  try {
    const result = await apiCall('/api/stop', 'POST', { save });
    showMessage(result, 'info');
  } catch (e) {
    showMessage('Error: ' + e.message, 'error');
  }
}

// ── Presets ────────────────────────────────────────────────────

async function loadPresets() {
  try {
    const presets = await apiCall('/api/presets', 'GET');
    const sel = document.getElementById('preset-select');
    sel.innerHTML = '';
    presets.forEach(p => {
      const o = document.createElement('option');
      o.value = p;
      o.textContent = p;
      sel.appendChild(o);
    });
  } catch (e) {
    console.error('Failed to load presets:', e);
  }
}

async function loadPreset() {
  const name = document.getElementById('preset-select').value;
  if (!name) return;
  try {
    const data = await apiCall('/api/preset/' + encodeURIComponent(name), 'GET');
    applyPayload(data);
    showMessage('Preset loaded: ' + name, 'success');
  } catch (e) {
    showMessage('Error loading preset: ' + e.message, 'error');
  }
}

async function savePreset() {
  const config = buildPayload();
  const mode = document.getElementById('mode-select').value;
  const model = document.getElementById('model-path').value;
  const vae = document.getElementById('vae-path').value;
  const body = { mode, model, vae, config };
  try {
    const result = await apiCall('/api/preset', 'POST', body);
    showMessage(result, 'success');
    loadPresets();
  } catch (e) {
    showMessage('Error saving preset: ' + e.message, 'error');
  }
}

// ── JSON load ──────────────────────────────────────────────────

async function loadJson() {
  const name = document.getElementById('json-file').value;
  if (!name) return;
  try {
    const data = await apiCall('/api/json/' + encodeURIComponent(name), 'GET');
    applyPayload(data);
    showMessage('JSON loaded: ' + name, 'success');
  } catch (e) {
    showMessage('Error loading JSON: ' + e.message, 'error');
  }
}

function openFolder() {
  apiCall('/api/open-folder', 'POST').catch(e => console.error(e));
}

// ── FD Cluster Inspector ───────────────────────────────────────

/**
 * Check whether the fd_loss_enable checkbox in the Train tab is checked.
 * Returns true if FD-Loss is enabled in the form.
 */
function isFdLossEnabled() {
  const fdCheckbox = document.querySelector('[data-config-name="fd_loss_enable"] input[type="checkbox"]');
  return fdCheckbox ? fdCheckbox.checked : false;
}

/**
 * Update the FD Cluster tab status indicator based on the fd_loss_enable checkbox state.
 */
function updateFdClusterStatus() {
  const statusEl = document.getElementById('fd-pause-status');
  const container = document.getElementById('fd-cluster-html');
  const enabled = isFdLossEnabled();

  if (!enabled) {
    // Show a clear warning that FD-Loss is not enabled
    statusEl.innerHTML = '<div style="color:#facc15;font-family:sans-serif;font-size:13px;padding:8px 0;">⚠️ <strong>fd_loss_enable</strong> is not checked in the Train tab. Enable it and start training to use the Cluster Inspector.</div>';
    // Only replace placeholder if it's the default one
    const placeholder = container.querySelector('.fd-placeholder');
    if (!placeholder) {
      container.innerHTML = '<div class="fd-placeholder">FD-Loss not active. Enable <strong>fd_loss_enable</strong> and start training.</div>';
    }
    // Hide active count when FD is disabled
    const activeEl = document.getElementById('fd-active-count');
    if (activeEl) activeEl.style.display = 'none';
  } else {
    statusEl.innerHTML = '<div style="color:#4ade80;font-family:sans-serif;font-size:13px;padding:8px 0;">✅ <strong>fd_loss_enable</strong> is checked. Start training then click <strong>Refresh</strong> to inspect clusters.</div>';
    // Try to fetch active count — will show once training has started
    updateFdActiveCount();
  }
}

async function fdRefresh() {
  // First check if FD-Loss is enabled
  if (!isFdLossEnabled()) {
    showMessage('FD-Loss is not enabled. Check fd_loss_enable in the Train tab first.', 'error');
    return;
  }

  const nClusters = document.getElementById('fd-n-clusters').value;
  const maxThumbs = document.getElementById('fd-max-thumbs').value;
  try {
    const data = await apiCall('/api/fd/refresh', 'POST', { n_clusters: parseInt(nClusters), max_thumbs: parseInt(maxThumbs) });
    if (data.status === 'error') {
      showMessage('Error: ' + (data.error || 'Unknown error'), 'error');
      return;
    }
    document.getElementById('fd-cluster-html').innerHTML = data.html || data;
    // Re-attach event listeners for the newly rendered cluster panel
    initFdClusterPanel();
    // Sync the pending count from the server-rendered HTML
    syncFdPendingCountFromHtml();
    // Fetch and display the active sample count
    updateFdActiveCount();
    showMessage('Cluster panel refreshed', 'info');
  } catch (e) {
    showMessage('Error: ' + e.message, 'error');
  }
}

async function fdResume() {
  try {
    const data = await apiCall('/api/fd/resume', 'POST');
    const html = data.html || data;
    document.getElementById('fd-pause-status').innerHTML = html;
    showMessage('Training resumed', 'info');
  } catch (e) {
    showMessage('Error: ' + e.message, 'error');
  }
}

// ── FD Cluster Panel: Lazy image loading & button actions ──────

/**
 * Fetch a single thumbnail from the server and set it as the img src.
 */
async function fdLoadThumbnail(imgEl) {
  const idx = imgEl.getAttribute('data-fetch-idx');
  if (!idx || imgEl.src) return; // already loaded or no idx
  try {
    const data = await apiCall('/api/fd/thumbnail/' + idx, 'GET');
    if (data && data.data_uri) {
      imgEl.src = data.data_uri;
    }
  } catch (e) {
    console.error('Failed to load thumbnail idx=' + idx, e);
  }
}

/**
 * Fetch and display the active sample count from the server.
 * Active = all non-evicted slots: n_type0 + n_type1 (= queue_size - n_type2).
 * Called after fdRefresh() and after commit actions.
 */
async function updateFdActiveCount() {
  const el = document.getElementById('fd-active-count');
  if (!el) return;
  try {
    const data = await apiCall('/api/fd/active_count', 'GET');
    if (data && data.status === 'ok' && data.counts) {
      const entries = Object.entries(data.counts);
      if (entries.length === 0) {
        el.style.display = 'none';
        return;
      }
      // Build a summary line for each judge
      const parts = entries.map(([name, c]) => {
        const active = c.active;
        const total = c.queue_size;
        const n0 = c.n_type0 || 0;
        const n1 = c.n_type1 || 0;
        const n2 = c.n_type2 || 0;
        let label = `${name}: <b>${active}</b> / ${total}`;
        // Show breakdown with colored badges matching the cluster panel legend
        if (n0 > 0) {
          label += ` <span style='color:#4ade80;font-size:11px;'>🟢${n0}</span>`;
        }
        if (n1 > 0) {
          label += ` <span style='color:#fbbf24;font-size:11px;'>🟡${n1}</span>`;
        }
        if (n2 > 0) {
          label += ` <span style='color:#888;font-size:11px;'>💀${n2}</span>`;
        }
        return label;
      });
      el.innerHTML = '📊 Active samples: ' + parts.join(' &nbsp;|&nbsp; ');
      el.style.display = 'block';
    } else {
      el.style.display = 'none';
    }
  } catch (e) {
    console.error('Failed to fetch active count', e);
    el.style.display = 'none';
  }
}

/**
 * Update the uncommitted changes tracker UI in the FD cluster panel.
 * Finds the commit button and pending status span and updates their
 * labels/styles to reflect the current pending count.
 */
function updateFdPendingCountUI(count) {
  fdPendingCount = count;
  const container = document.getElementById('fd-cluster-html');
  if (!container) return;

  // Find the commit-all button
  const commitBtn = container.querySelector('#fd-commit-all');
  if (commitBtn) {
    const hasPending = count > 0;
    commitBtn.style.borderColor = hasPending ? '#ef4444' : '#555';
    commitBtn.style.background = hasPending ? 'rgba(239,68,68,0.15)' : 'transparent';
    commitBtn.style.color = hasPending ? '#ef4444' : '#666';
    commitBtn.textContent = hasPending ? `⚠️ Commit ${count} Change(s)` : '💾 Commit All';
  }

  // Find the pending status span (sibling of the commit button container)
  const statusSpan = container.querySelector('#fd-commit-all + span, .fd-commit-status');
  // If no dedicated span, look for the span that shows pending status
  // It's in the same div as the commit button
  const commitDiv = commitBtn ? commitBtn.closest('div') : null;
  if (commitDiv) {
    let statusEl = commitDiv.querySelector('span:last-child');
    if (statusEl) {
      const hasPending = count > 0;
      statusEl.style.color = hasPending ? '#ef4444' : '#666';
      statusEl.textContent = hasPending ? `${count} staged action(s) pending` : 'No pending changes';
    }
  }
}

/**
 * Sync the pending count from the server-rendered HTML.
 * Called after fdRefresh() to initialize the frontend state
 * from the n_pending_actions value embedded in the commit button text.
 */
function syncFdPendingCountFromHtml() {
  const container = document.getElementById('fd-cluster-html');
  if (!container) return;
  const commitBtn = container.querySelector('#fd-commit-all');
  if (!commitBtn) {
    fdPendingCount = 0;
    return;
  }
  // Extract count from button text like "⚠️ Commit 3 Change(s)" or "💾 Commit All"
  const text = commitBtn.textContent || '';
  const match = text.match(/Commit\s+(\d+)\s+Change/);
  if (match) {
    fdPendingCount = parseInt(match[1], 10);
  } else {
    fdPendingCount = 0;
  }
}

/**
 * Send an action to the server (protect/evict/guidance/commit).
 * Updates the uncommitted changes tracker from the server response.
 */
async function fdSendAction(action, extra) {
  const body = { action: action, ...(extra || {}) };
  try {
    const data = await apiCall('/api/fd/action', 'POST', body);
    if (data.status === 'ok') {
      showMessage(data.result || 'Action sent', 'info');
      // Update the pending count from the server response
      if (data.pending_count !== undefined) {
        updateFdPendingCountUI(data.pending_count);
      }
      // After a commit, evictions are applied so refresh the active count
      if (action === 'commit') {
        updateFdActiveCount();
      }
    } else {
      showMessage('Error: ' + (data.error || 'Unknown'), 'error');
    }
  } catch (e) {
    showMessage('Error: ' + e.message, 'error');
  }
}

/**
 * Initialize event listeners for the FD cluster panel.
 * Called after fdRefresh() replaces the innerHTML.
 */
function initFdClusterPanel() {
  const container = document.getElementById('fd-cluster-html');
  if (!container) return;

  // --- Lazy thumbnail loading: when a <details> accordion is opened ---
  container.querySelectorAll('details.fd-cluster-details').forEach(details => {
    details.addEventListener('toggle', function() {
      if (this.open) {
        // Load all thumbnails inside this cluster
        this.querySelectorAll('img[data-fetch-idx]').forEach(img => {
          fdLoadThumbnail(img);
        });
      }
    });
  });

  // --- Cluster-level protect button ---
  // NOTE: Actions are always associated with INDICES, not clusters.
  // The cluster-level button is a CONVENIENCE that batch-operates on all
  // indices in the cluster.  The label shows a count of affected indices.
  container.querySelectorAll('.fd-protect-btn').forEach(btn => {
    btn.addEventListener('click', function(e) {
      e.stopPropagation();
      const clusterId = this.getAttribute('data-cluster');
      const isUnprotect = this.textContent.includes('Unprotect All');
      // Gather all per-sample indices in this cluster from the DOM
      const details = this.closest('details.fd-cluster-details');
      const indices = [];
      if (details) {
        details.querySelectorAll('.fd-idx-protect').forEach(idxBtn => {
          const idx = parseInt(idxBtn.getAttribute('data-idx'));
          if (!isNaN(idx)) indices.push(idx);
        });
      }
      // Optimistic UI: toggle button state immediately
      if (isUnprotect) {
        this.textContent = '🛡️ Protect All';
        this.style.borderColor = '#555';
        this.style.background = 'transparent';
        this.style.color = '#555';
        this.title = 'Protect all indices in this cluster';
        // Send per-index actions directly — no cluster-level actions
        fdSendAction('unprotect_indices', { indices: indices });
      } else {
        this.textContent = '🔒 Unprotect All (' + indices.length + ')';
        this.style.borderColor = '#4ade80';
        this.style.background = 'rgba(74,222,128,0.15)';
        this.style.color = '#4ade80';
        this.title = 'Unprotect all indices in this cluster';
        fdSendAction('protect_indices', { indices: indices });
      }
      // Also toggle all per-sample protect buttons inside this cluster's accordion
      if (details) {
        details.querySelectorAll('.fd-idx-protect').forEach(idxBtn => {
          const idxIsProtected = idxBtn.textContent.includes('🔒');
          if (isUnprotect && idxIsProtected) {
            // Unprotect this individual sample
            idxBtn.textContent = '🛡️';
            idxBtn.style.borderColor = '#555';
            idxBtn.style.background = 'transparent';
            idxBtn.style.color = '#555';
            idxBtn.title = 'Click to protect';
            // Update parent container state
            const parentDiv = idxBtn.closest('div[data-idx-state]');
            if (parentDiv) {
              const sourceType = parseInt(parentDiv.getAttribute('data-source-type'));
              const evictBtn = parentDiv.querySelector('.fd-idx-evict');
              const isEviction = evictBtn && evictBtn.textContent.includes('↩️');
              if (isEviction) {
                parentDiv.setAttribute('data-idx-state', 'eviction');
              } else if (sourceType === 1) {
                parentDiv.setAttribute('data-idx-state', 'generated');
              } else {
                parentDiv.setAttribute('data-idx-state', 'normal');
              }
            }
          } else if (!isUnprotect && !idxIsProtected) {
            // Protect this individual sample
            idxBtn.textContent = '🔒';
            idxBtn.style.borderColor = '#4ade80';
            idxBtn.style.background = 'rgba(74,222,128,0.15)';
            idxBtn.style.color = '#4ade80';
            idxBtn.title = 'Click to unprotect';
            // Update parent container state
            const parentDiv = idxBtn.closest('div[data-idx-state]');
            if (parentDiv) {
              const evictBtn = parentDiv.querySelector('.fd-idx-evict');
              const isEviction = evictBtn && evictBtn.textContent.includes('↩️');
              if (isEviction) {
                parentDiv.setAttribute('data-idx-state', 'conflict');
              } else {
                parentDiv.setAttribute('data-idx-state', 'protected');
              }
            }
          }
        });
      }
    });
  });

  // --- Cluster-level guidance button ---
  container.querySelectorAll('.fd-guidance-btn').forEach(btn => {
    btn.addEventListener('click', function(e) {
      e.stopPropagation();
      const clusterId = this.getAttribute('data-cluster');
      const isGuidance = this.textContent.includes('Guidance Active');
      if (isGuidance) {
        // Optimistic UI: clear guidance immediately
        this.textContent = '🎯 Set as Guidance';
        this.style.borderColor = '#555';
        this.style.background = 'transparent';
        this.style.color = '#ccc';
        this.title = 'Set this cluster as guidance target';
        fdSendAction('clear_guidance', {});
      } else {
        // Optimistic UI: set guidance immediately
        this.textContent = '✅ Guidance Active';
        this.style.borderColor = '#60a5fa';
        this.style.background = 'rgba(96,165,250,0.15)';
        this.style.color = '#60a5fa';
        this.title = 'Guidance is active on this cluster';
        fdSendAction('guidance_cluster', { cluster_id: parseInt(clusterId) });
      }
    });
  });

  // --- Cluster-level evict button ---
  // NOTE: Actions are always associated with INDICES, not clusters.
  // The cluster-level button is a CONVENIENCE that batch-operates on all
  // indices in the cluster.  The label shows a count of affected indices.
  container.querySelectorAll('.fd-evict-btn').forEach(btn => {
    btn.addEventListener('click', function(e) {
      e.stopPropagation();
      const clusterId = this.getAttribute('data-cluster');
      const isEviction = this.textContent.includes('Unmark Eviction');
      // Gather all per-sample indices in this cluster from the DOM
      const details = this.closest('details.fd-cluster-details');
      const indices = [];
      if (details) {
        details.querySelectorAll('.fd-idx-evict').forEach(idxBtn => {
          const idx = parseInt(idxBtn.getAttribute('data-idx'));
          if (!isNaN(idx)) indices.push(idx);
        });
      }
      // Optimistic UI: toggle eviction state immediately
      if (isEviction) {
        this.textContent = '❌ Mark Eviction';
        this.style.borderColor = '#fbbf24';
        this.style.background = 'transparent';
        this.style.color = '#fbbf24';
        this.title = 'Mark all indices in this cluster for eviction';
      } else {
        this.textContent = '↩️ Unmark Eviction (' + indices.length + ')';
        this.style.borderColor = '#ef4444';
        this.style.background = 'rgba(239,68,68,0.15)';
        this.style.color = '#ef4444';
        this.title = 'Clear eviction marks on all indices in this cluster';
      }
      // Send per-index actions directly — no cluster-level actions
      fdSendAction('toggle_evict_indices', { indices: indices });
      // Also toggle all per-sample evict buttons inside this cluster's accordion
      if (details) {
        details.querySelectorAll('.fd-idx-evict').forEach(idxBtn => {
          const idxIsEviction = idxBtn.textContent.includes('↩️');
          const parentDiv = idxBtn.closest('div[data-idx-state]');
          if (isEviction && idxIsEviction) {
            // Unmark eviction for this individual sample
            idxBtn.textContent = '❌';
            idxBtn.style.borderColor = '#fbbf24';
            idxBtn.style.background = 'transparent';
            idxBtn.style.color = '#fbbf24';
            idxBtn.title = 'Mark for eviction';
            if (parentDiv) {
              const sourceType = parseInt(parentDiv.getAttribute('data-source-type'));
              if (sourceType === 1) {
                parentDiv.setAttribute('data-idx-state', 'generated');
              } else {
                parentDiv.setAttribute('data-idx-state', 'normal');
              }
            }
          } else if (!isEviction && !idxIsEviction) {
            // Mark eviction for this individual sample
            idxBtn.textContent = '↩️';
            idxBtn.style.borderColor = '#ef4444';
            idxBtn.style.background = 'rgba(239,68,68,0.15)';
            idxBtn.style.color = '#ef4444';
            idxBtn.title = 'Clear eviction mark';
            if (parentDiv) {
              const protBtn = parentDiv.querySelector('.fd-idx-protect');
              const isProtected = protBtn && protBtn.textContent.includes('🔒');
              if (isProtected) {
                parentDiv.setAttribute('data-idx-state', 'conflict');
              } else {
                parentDiv.setAttribute('data-idx-state', 'eviction');
              }
            }
          }
        });
      }
    });
  });

  // --- Per-sample protect button ---
  container.querySelectorAll('.fd-idx-protect').forEach(btn => {
    btn.addEventListener('click', function(e) {
      e.stopPropagation();
      const idx = parseInt(this.getAttribute('data-idx'));
      const isProtected = this.textContent.includes('🔒');
      const parentDiv = this.closest('div[data-idx-state]');
      // Optimistic UI: toggle protect state immediately
      if (isProtected) {
        this.textContent = '🛡️';
        this.style.borderColor = '#555';
        this.style.background = 'transparent';
        this.style.color = '#555';
        this.title = 'Click to protect';
        // Update parent container state
        if (parentDiv) {
          const sourceType = parseInt(parentDiv.getAttribute('data-source-type'));
          if (parentDiv.getAttribute('data-idx-state') === 'eviction' || parentDiv.getAttribute('data-idx-state') === 'conflict') {
            // If eviction is still marked, stay in eviction state
            parentDiv.setAttribute('data-idx-state', 'eviction');
          } else if (sourceType === 1) {
            parentDiv.setAttribute('data-idx-state', 'generated');
          } else {
            parentDiv.setAttribute('data-idx-state', 'normal');
          }
        }
        fdSendAction('unprotect_idx', { idx: idx });
      } else {
        this.textContent = '🔒';
        this.style.borderColor = '#4ade80';
        this.style.background = 'rgba(74,222,128,0.15)';
        this.style.color = '#4ade80';
        this.title = 'Click to unprotect';
        // Update parent container state
        if (parentDiv) {
          const evictBtn = parentDiv.querySelector('.fd-idx-evict');
          const isEviction = evictBtn && evictBtn.textContent.includes('↩️');
          if (isEviction) {
            parentDiv.setAttribute('data-idx-state', 'conflict');
          } else {
            parentDiv.setAttribute('data-idx-state', 'protected');
          }
        }
        fdSendAction('protect_idx', { idx: idx });
      }
    });
  });

  // --- Per-sample evict button ---
  container.querySelectorAll('.fd-idx-evict').forEach(btn => {
    btn.addEventListener('click', function(e) {
      e.stopPropagation();
      const idx = parseInt(this.getAttribute('data-idx'));
      const isEviction = this.textContent.includes('↩️');
      const parentDiv = this.closest('div[data-idx-state]');
      // Optimistic UI: toggle eviction state immediately
      if (isEviction) {
        this.textContent = '❌';
        this.style.borderColor = '#fbbf24';
        this.style.background = 'transparent';
        this.style.color = '#fbbf24';
        this.title = 'Mark for eviction';
        // Update parent container state
        if (parentDiv) {
          const sourceType = parseInt(parentDiv.getAttribute('data-source-type'));
          if (sourceType === 1) {
            parentDiv.setAttribute('data-idx-state', 'generated');
          } else {
            parentDiv.setAttribute('data-idx-state', 'normal');
          }
        }
      } else {
        this.textContent = '↩️';
        this.style.borderColor = '#ef4444';
        this.style.background = 'rgba(239,68,68,0.15)';
        this.style.color = '#ef4444';
        this.title = 'Clear eviction mark';
        // Update parent container state
        if (parentDiv) {
          const protBtn = parentDiv.querySelector('.fd-idx-protect');
          const isProtected = protBtn && protBtn.textContent.includes('🔒');
          if (isProtected) {
            parentDiv.setAttribute('data-idx-state', 'conflict');
          } else {
            parentDiv.setAttribute('data-idx-state', 'eviction');
          }
        }
      }
      fdSendAction('toggle_evict_idx', { idx: idx });
    });
  });

  // --- Per-sample guidance button ---
  // Sets the guidance target from this specific image's feature vector,
  // NOT from a cluster centroid.  This gives precise control over which
  // visual style to guide toward.
  container.querySelectorAll('.fd-idx-guidance').forEach(btn => {
    btn.addEventListener('click', function(e) {
      e.stopPropagation();
      const idx = parseInt(this.getAttribute('data-idx'));
      // Optimistic UI: show visual feedback immediately
      this.textContent = '✅';
      this.style.borderColor = '#60a5fa';
      this.style.background = 'rgba(96,165,250,0.15)';
      this.style.color = '#60a5fa';
      this.title = 'Guidance target set from this image';
      fdSendAction('guidance_idx', { idx: idx });
    });
  });

  // --- Commit all button ---
  const commitBtn = container.querySelector('#fd-commit-all');
  if (commitBtn) {
    commitBtn.addEventListener('click', function(e) {
      e.stopPropagation();
      fdSendAction('commit', {});
    });
  }
}

// ── Image upload ───────────────────────────────────────────────

function handleImageUpload(inputId, previewId, stateSetter) {
  const input = document.getElementById(inputId);
  input.addEventListener('change', function(e) {
    const file = e.target.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = function(ev) {
      const preview = document.getElementById(previewId);
      preview.src = ev.target.result;
      preview.style.display = 'block';
      // Store as base64 data URI
      stateSetter(ev.target.result);
    };
    reader.readAsDataURL(file);
  });
}

// ── Slider helpers ─────────────────────────────────────────────

function setupSlider(sliderId, valId, callback) {
  const slider = document.getElementById(sliderId);
  const val = document.getElementById(valId);
  slider.addEventListener('input', function() {
    val.textContent = parseFloat(this.value).toFixed(2);
    if (callback) callback();
  });
}

// ── Tab switching ──────────────────────────────────────────────

function initTabs() {
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', function() {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-content').forEach(t => t.classList.remove('active'));
      this.classList.add('active');
      document.getElementById('tab-' + this.dataset.tab).classList.add('active');

      // When switching to FD Cluster tab, update the status indicator
      // and initialize cluster panel event listeners
      if (this.dataset.tab === 'fd-cluster') {
        updateFdClusterStatus();
        initFdClusterPanel();
      }
    });
  });
}

// ── Initialize ─────────────────────────────────────────────────

function init() {
  // Build form sections
  buildSection('required-params', R_COLUMN1.concat(R_COLUMN2).concat(R_COLUMN3), 3);
  buildSection('bucket-params', B_COLUMN, 3);
  buildSection('fd-params', F_COLUMN, 3);
  buildSection('option-params', O_COLUMN1.concat(O_COLUMN2), 3);
  buildSection('ts-controls', O_TS_COLUMN, 1);
  buildSection('layer-controls', O_LAYER_COLUMN, 1);

  // Initial previews
  renderTsPreview();
  renderLayerPreview();

  // Tab switching
  initTabs();

  // Mode change
  document.getElementById('mode-select').addEventListener('change', function() {
    currentMode = this.value;
    updateModeVisibility();
  });

  // TS distribution change → update preview + slider visibility
  const distSelect = document.querySelector('[data-config-name="train_timestep_distribution"] select');
  if (distSelect) {
    distSelect.addEventListener('change', function() {
      updateTsPreview();
      const dist = this.value;
      document.getElementById('slider-group-flow-shift').style.display = dist === 'flow_shift' ? '' : 'none';
      document.getElementById('slider-group-logit').style.display = dist === 'logit_normal' ? '' : 'none';
      document.getElementById('slider-group-beta').style.display = dist === 'beta' ? '' : 'none';
      // Reset params
      const defaults = {
        'uniform': '', 'flow_shift': 'shift=3.0', 'logit_normal': 'mean=0.0 std=1.0',
        'cosmap': '', 'beta': 'alpha=0.5 beta=0.5'
      };
      const paramsInput = document.querySelector('[data-config-name="train_ts_dist_params(e.g. mean=0.0 std=1.0)"] input');
      if (paramsInput) paramsInput.value = defaults[dist] || '';
      updateTsPreview();
    });
  }

  // TS min/max/params change → update preview
  document.querySelectorAll('[data-config-name="train_min_timesteps"] input, [data-config-name="train_max_timesteps"] input, [data-config-name="train_ts_dist_params(e.g. mean=0.0 std=1.0)"] input').forEach(el => {
    el.addEventListener('input', updateTsPreview);
  });

  // TS schedule textarea change → update preview
  const tsSchedule = document.querySelector('[data-config-name="train_ts_schedule"] textarea');
  if (tsSchedule) tsSchedule.addEventListener('input', updateTsPreview);

  // Layer filter change → update preview
  const filterInput = document.querySelector('[data-config-name="network_module_filter(regex, !prefix=exclude)"] input');
  if (filterInput) filterInput.addEventListener('input', updateLayerPreview);

  // Sliders
  setupSlider('slider-fs', 'slider-fs-val', () => {
    const paramsInput = document.querySelector('[data-config-name="train_ts_dist_params(e.g. mean=0.0 std=1.0)"] input');
    if (paramsInput) paramsInput.value = `shift=${parseFloat(document.getElementById('slider-fs').value).toFixed(3)}`;
    updateTsPreview();
  });
  setupSlider('slider-mean', 'slider-mean-val', updateLogitParams);
  setupSlider('slider-std', 'slider-std-val', updateLogitParams);
  setupSlider('slider-alpha', 'slider-alpha-val', updateBetaParams);
  setupSlider('slider-beta', 'slider-beta-val', updateBetaParams);

  // FD thumb scale slider
  setupSlider('fd-thumb-scale', 'fd-thumb-scale-val');

  // Image uploads
  handleImageUpload('orig-image', 'orig-image-preview', (v) => { origImageData = v; });
  handleImageUpload('targ-image', 'targ-image-preview', (v) => { targImageData = v; });

  // Buttons
  document.getElementById('btn-start').addEventListener('click', startTraining);
  document.getElementById('btn-stop').addEventListener('click', () => stopTraining(false));
  document.getElementById('btn-stop-save').addEventListener('click', () => stopTraining(true));
  document.getElementById('btn-load-preset').addEventListener('click', loadPreset);
  document.getElementById('btn-save-preset').addEventListener('click', savePreset);
  document.getElementById('btn-refresh-presets').addEventListener('click', loadPresets);
  document.getElementById('btn-load-json').addEventListener('click', loadJson);
  document.getElementById('btn-open-folder').addEventListener('click', openFolder);
  document.getElementById('fd-refresh-btn').addEventListener('click', fdRefresh);
  document.getElementById('fd-resume-btn').addEventListener('click', fdResume);

  // Listen for changes to the fd_loss_enable checkbox so status updates live
  const fdCheckbox = document.querySelector('[data-config-name="fd_loss_enable"] input[type="checkbox"]');
  if (fdCheckbox) {
    fdCheckbox.addEventListener('change', updateFdClusterStatus);
  }

  // Reset all form fields to defaults (prevents browser autofill from restoring stale values)
  resetFormToDefaults();

  // Initialize FD cluster panel event listeners (if any content already rendered)
  initFdClusterPanel();

  // Load presets on startup
  loadPresets();
}

function updateLogitParams() {
  const mean = document.getElementById('slider-mean').value;
  const std = document.getElementById('slider-std').value;
  const paramsInput = document.querySelector('[data-config-name="train_ts_dist_params(e.g. mean=0.0 std=1.0)"] input');
  if (paramsInput) paramsInput.value = `mean=${parseFloat(mean).toFixed(2)} std=${parseFloat(std).toFixed(2)}`;
  updateTsPreview();
}

function updateBetaParams() {
  const alpha = document.getElementById('slider-alpha').value;
  const beta = document.getElementById('slider-beta').value;
  const paramsInput = document.querySelector('[data-config-name="train_ts_dist_params(e.g. mean=0.0 std=1.0)"] input');
  if (paramsInput) paramsInput.value = `alpha=${parseFloat(alpha).toFixed(3)} beta=${parseFloat(beta).toFixed(3)}`;
  updateTsPreview();
}

// ── Start ──────────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', init);
