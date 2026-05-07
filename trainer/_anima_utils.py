"""Anima model loading utilities (in-housed from sd-scripts).

Provides only what traintrain needs:
- load_anima_model: load Anima DiT from safetensors checkpoint
- load_qwen3_text_encoder: load Qwen3 text encoder
- load_qwen3_tokenizer: load Qwen3 tokenizer
- load_t5_tokenizer: load T5 tokenizer

Stripped of fp8 optimization, LoRA merging during model load, and other
features not used by traintrain.
"""

import logging
import os
from typing import Dict, List, Optional, Union

import torch
from accelerate import init_empty_weights
from safetensors.torch import load_file

from trainer import anima_models
from trainer._safetensors_utils import WeightTransformHooks

logger = logging.getLogger(__name__)


def load_anima_model(
    device: Union[str, torch.device],
    dit_path: str,
    attn_mode: str,
    split_attn: bool,
    loading_device: Union[str, torch.device],
    dit_weight_dtype: Optional[torch.dtype],
    fp8_scaled: bool = False,
    lora_weights_list: Optional[List[Dict[str, torch.Tensor]]] = None,
    lora_multipliers: Optional[List[float]] = None,
) -> anima_models.Anima:
    """Load Anima model from the specified checkpoint.

    Simplified version: no fp8, no LoRA merging during model load.
    """
    assert not fp8_scaled, "fp8_scaled is not supported in this in-house version"
    assert lora_weights_list is None, "LoRA merging during model load is not supported in this in-house version"

    device = torch.device(device)
    loading_device = torch.device(loading_device)

    # Fixed DiT config for Anima models
    dit_config = {
        "max_img_h": 512,
        "max_img_w": 512,
        "max_frames": 128,
        "in_channels": 16,
        "out_channels": 16,
        "patch_spatial": 2,
        "patch_temporal": 1,
        "model_channels": 2048,
        "concat_padding_mask": True,
        "crossattn_emb_channels": 1024,
        "pos_emb_cls": "rope3d",
        "pos_emb_learnable": True,
        "pos_emb_interpolation": "crop",
        "min_fps": 1,
        "max_fps": 30,
        "use_adaln_lora": True,
        "adaln_lora_dim": 256,
        "num_blocks": 28,
        "num_heads": 16,
        "extra_per_block_abs_pos_emb": False,
        "rope_h_extrapolation_ratio": 4.0,
        "rope_w_extrapolation_ratio": 4.0,
        "rope_t_extrapolation_ratio": 1.0,
        "extra_h_extrapolation_ratio": 1.0,
        "extra_w_extrapolation_ratio": 1.0,
        "extra_t_extrapolation_ratio": 1.0,
        "rope_enable_fps_modulation": False,
        "use_llm_adapter": True,
        "attn_mode": attn_mode,
        "split_attn": split_attn,
    }

    with init_empty_weights():
        model = anima_models.Anima(**dit_config)
        if dit_weight_dtype is not None:
            model.to(dit_weight_dtype)

    # Load model weights
    logger.info(f"Loading DiT model from {dit_path}, device={loading_device}")
    rename_hooks = WeightTransformHooks(
        rename_hook=lambda k: k[len("net."):] if k.startswith("net.") else k
    )

    # Load state dict directly
    sd = load_file(dit_path, device="cpu")

    # Apply rename hook
    if rename_hooks.rename_hook is not None:
        sd = {rename_hooks.rename_hook(k): v for k, v in sd.items()}

    # Cast to target dtype if specified
    if dit_weight_dtype is not None:
        for key in sd.keys():
            sd[key] = sd[key].to(dtype=dit_weight_dtype)

    # Move to loading device
    if loading_device.type != "cpu":
        for key in sd.keys():
            sd[key] = sd[key].to(loading_device)

    missing, unexpected = model.load_state_dict(sd, strict=False, assign=True)
    if missing:
        unexpected_missing = [
            k
            for k in missing
            if not any(buf_name in k for buf_name in ("seq", "dim_spatial_range", "dim_temporal_range", "inv_freq"))
        ]
        if unexpected_missing:
            raise RuntimeError(
                f"Missing keys in checkpoint: {unexpected_missing[:10]}"
                + ("..." if len(unexpected_missing) > 10 else "")
            )
    if unexpected:
        raise RuntimeError(
            f"Unexpected keys in checkpoint: {unexpected[:5]}"
            + ("..." if len(unexpected) > 5 else "")
        )
    logger.info(
        f"Loaded DiT model from {dit_path}, "
        f"unexpected missing keys: {len(missing)}, unexpected keys: {len(unexpected)}"
    )

    return model


def load_qwen3_tokenizer(qwen3_path: str):
    """Load Qwen3 tokenizer only (without the text encoder model)."""
    from transformers import AutoTokenizer

    if os.path.isdir(qwen3_path):
        tokenizer = AutoTokenizer.from_pretrained(qwen3_path, local_files_only=True)
    else:
        config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "qwen3_06b")
        if not os.path.exists(config_dir):
            raise FileNotFoundError(
                f"Qwen3 config directory not found at {config_dir}. "
                "Expected configs/qwen3_06b/ with config.json, tokenizer.json, etc. "
                "You can download these from the Qwen3-0.6B HuggingFace repository."
            )
        tokenizer = AutoTokenizer.from_pretrained(config_dir, local_files_only=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer


def load_qwen3_text_encoder(
    qwen3_path: str,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cpu",
    lora_weights: Optional[List[Dict[str, torch.Tensor]]] = None,
    lora_multipliers: Optional[List[float]] = None,
):
    """Load Qwen3-0.6B text encoder.

    Simplified version: no LoRA merging during load.
    """
    import transformers
    from transformers import AutoTokenizer

    logger.info(f"Loading Qwen3 text encoder from {qwen3_path}")

    if os.path.isdir(qwen3_path):
        tokenizer = AutoTokenizer.from_pretrained(qwen3_path, local_files_only=True)
        model = transformers.AutoModelForCausalLM.from_pretrained(
            qwen3_path, torch_dtype=dtype, local_files_only=True
        ).model
    else:
        config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "qwen3_06b")
        if not os.path.exists(config_dir):
            raise FileNotFoundError(
                f"Qwen3 config directory not found at {config_dir}. "
                "Expected configs/qwen3_06b/ with config.json, tokenizer.json, etc. "
                "You can download these from the Qwen3-0.6B HuggingFace repository."
            )

        tokenizer = AutoTokenizer.from_pretrained(config_dir, local_files_only=True)
        qwen3_config = transformers.Qwen3Config.from_pretrained(config_dir, local_files_only=True)
        model = transformers.Qwen3ForCausalLM(qwen3_config).model

        if qwen3_path.endswith(".safetensors"):
            state_dict = load_file(qwen3_path, device="cpu")
        else:
            state_dict = torch.load(qwen3_path, map_location="cpu", weights_only=True)

        new_sd = {}
        for k, v in state_dict.items():
            if k.startswith("model."):
                new_sd[k[len("model."):]] = v
            else:
                new_sd[k] = v

        info = model.load_state_dict(new_sd, strict=False)
        logger.info(f"Loaded Qwen3 state dict: {info}")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.use_cache = False
    model = model.requires_grad_(False).to(device, dtype=dtype)

    logger.info(f"Loaded Qwen3 text encoder. Parameters: {sum(p.numel() for p in model.parameters()):,}")
    return model, tokenizer


def load_t5_tokenizer(t5_tokenizer_path: Optional[str] = None):
    """Load T5 tokenizer for LLM Adapter target tokens."""
    from transformers import T5TokenizerFast

    if t5_tokenizer_path is not None:
        return T5TokenizerFast.from_pretrained(t5_tokenizer_path, local_files_only=True)

    config_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "configs", "t5_old")
    if os.path.exists(config_dir):
        return T5TokenizerFast(
            vocab_file=os.path.join(config_dir, "spiece.model"),
            tokenizer_file=os.path.join(config_dir, "tokenizer.json"),
        )

    raise FileNotFoundError(
        f"T5 tokenizer config directory not found at {config_dir}. "
        "Expected configs/t5_old/ with spiece.model and tokenizer.json."
    )
