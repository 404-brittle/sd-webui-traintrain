"""Anima-specific support for traintrain: flow matching, forward pass, text encoding."""

import os
import sys
import torch
import torch.nn as nn

# Add sd-scripts root to path so we can import library.*
_TRAINTRAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SD_SCRIPTS_ROOT = os.environ.get("SD_SCRIPTS_PATH") or os.path.dirname(_TRAINTRAIN_DIR)
if _SD_SCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, _SD_SCRIPTS_ROOT)


# ---------------------------------------------------------------------------
# Flow matching noise scheduler
# ---------------------------------------------------------------------------

class AnimaFlowScheduler:
    """Replaces DDPMScheduler for Anima's rectified flow matching.

    Timesteps are integers in [0, 999] in the training loop (for UI consistency)
    and converted to [0, 1] inside add_noise / anima_forward.
    """

    def add_noise(self, clean_latents: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between clean latents and noise (flow matching).

        noisy = (1 - t) * clean + t * noise   where t = timestep / 1000
        """
        t = timesteps.float() / 1000.0
        t = t.view(-1, *([1] * (clean_latents.dim() - 1)))
        return (1.0 - t) * clean_latents + t * noise


# ---------------------------------------------------------------------------
# Conditioning helpers
# ---------------------------------------------------------------------------

def expand_cond(cond, batch_size: int):
    """Expand a conditioning object (tuple/list or tensor) to the required batch size."""
    if isinstance(cond, (tuple, list)):
        out = []
        for c in cond:
            if c is None:
                out.append(None)
            elif c.dim() == 0 or c.shape[0] == batch_size:
                out.append(c)
            elif c.shape[0] == 1:
                out.append(c.expand(batch_size, *c.shape[1:]))
            else:
                out.append(c)
        return tuple(out)
    return torch.cat([cond] * batch_size)


def move_cond_to_device(cond, device, dtype=None):
    """Move a conditioning tuple/list to the specified device/dtype.
    Raw strings (BASE/text-encoder mode) are passed through unchanged.
    """
    if isinstance(cond, str):
        return cond
    if isinstance(cond, (tuple, list)):
        result = []
        for c in cond:
            if c is None or isinstance(c, str):
                result.append(c)
            elif c.dtype.is_floating_point:
                result.append(c.to(device, dtype=dtype) if dtype else c.to(device))
            else:
                result.append(c.to(device))
        return tuple(result)
    if cond.dtype.is_floating_point:
        return cond.to(device, dtype=dtype) if dtype else cond.to(device)
    return cond.to(device)


# ---------------------------------------------------------------------------
# Anima DiT forward pass
# ---------------------------------------------------------------------------

def anima_forward(t, noisy_latents: torch.Tensor, timesteps: torch.Tensor, conds) -> torch.Tensor:
    """Run Anima DiT forward pass.

    Args:
        t: Trainer object (must have t.unet = raw Anima DiT model).
        noisy_latents: [B, C, H, W] — 4D noisy latents.
        timesteps: [B] integer timesteps in [0, 999].
        conds: tuple (prompt_embeds, qwen3_attn_mask, t5_input_ids or None, t5_attn_mask or None)

    Returns:
        [B, C, H, W] model prediction (velocity for flow matching).
    """
    import torch.nn.functional as F

    # BASE mode: conditioning is a raw string or a DataLoader-collated list of
    # strings (default_collate returns list[str] for string batch items).
    if isinstance(conds, str):
        conds, _ = t.text_model.encode_text([conds])
        conds = move_cond_to_device(conds, noisy_latents.device, noisy_latents.dtype)
    elif isinstance(conds, (tuple, list)) and conds and isinstance(conds[0], str):
        conds, _ = t.text_model.encode_text(list(conds))
        conds = move_cond_to_device(conds, noisy_latents.device, noisy_latents.dtype)

    prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = conds

    # Fuse Qwen3 embeddings + T5 token IDs via the LLM adapter.
    # Only pass T5 IDs when they are available — adapter embedding layer
    # cannot receive None (mirrors preprocess_text_embeds guard logic).
    if hasattr(t.unet, "preprocess_text_embeds"):
        cross = t.unet.preprocess_text_embeds(prompt_embeds, t5_input_ids)
    elif hasattr(t.unet, "llm_adapter") and t5_input_ids is not None:
        cross = t.unet.llm_adapter(
            prompt_embeds, t5_input_ids,
            target_attention_mask=t5_attn_mask,
            source_attention_mask=attn_mask,
        )
    else:
        cross = prompt_embeds

    if cross.shape[1] < 512:
        cross = F.pad(cross, (0, 0, 0, 512 - cross.shape[1]))

    x_5d = noisy_latents.unsqueeze(2)             # [B, C, 1, H, W]
    timesteps_01 = timesteps.float() / 1000.0     # [B] in [0, 1]

    B, _, H, W = noisy_latents.shape
    padding_mask = torch.zeros(B, 1, H, W, dtype=noisy_latents.dtype, device=noisy_latents.device)

    pred = t.unet(x_5d, timesteps_01, cross, padding_mask=padding_mask)
    return pred.squeeze(2)  # [B, C, 1, H, W] -> [B, C, H, W]


def anima_forward_refcn(
    t,
    noisy_latents: torch.Tensor,
    ref_latents: torch.Tensor,
    timesteps: torch.Tensor,
    conds,
) -> torch.Tensor:
    """Anima DiT forward pass for Reference ControlNet training.

    Concatenates the clean reference latent to the noisy target latent along
    the temporal axis (making a 2-frame sequence), runs the model, then trims
    the output back to the target frame only.  The sd-scripts Anima model does
    not accept ref_latents natively — we replicate the Forge forward() logic.

    Args:
        t:             Trainer object.
        noisy_latents: [B, C, H, W] — noisy target latent.
        ref_latents:   [B, C, H, W] — clean reference/control latent (no noise).
        timesteps:     [B] integer timesteps in [0, 999].
        conds:         (prompt_embeds, qwen3_attn_mask, t5_ids or None, t5_mask or None)

    Returns:
        [B, C, H, W] velocity prediction for the target frames only.
    """
    import torch.nn.functional as F

    if isinstance(conds, str):
        conds, _ = t.text_model.encode_text([conds])
        conds = move_cond_to_device(conds, noisy_latents.device, noisy_latents.dtype)
    elif isinstance(conds, (tuple, list)) and conds and isinstance(conds[0], str):
        conds, _ = t.text_model.encode_text(list(conds))
        conds = move_cond_to_device(conds, noisy_latents.device, noisy_latents.dtype)

    prompt_embeds, attn_mask, t5_input_ids, t5_attn_mask = conds

    if hasattr(t.unet, "preprocess_text_embeds"):
        cross = t.unet.preprocess_text_embeds(prompt_embeds, t5_input_ids)
    elif hasattr(t.unet, "llm_adapter") and t5_input_ids is not None:
        cross = t.unet.llm_adapter(
            prompt_embeds, t5_input_ids,
            target_attention_mask=t5_attn_mask,
            source_attention_mask=attn_mask,
        )
    else:
        cross = prompt_embeds

    if cross.shape[1] < 512:
        cross = F.pad(cross, (0, 0, 0, 512 - cross.shape[1]))

    # [B, C, 1, H, W] — the sd-scripts Anima model's forward_mini_train_dit
    # does not accept ref_latents.  We replicate the Forge forward() logic:
    # concat reference along the temporal axis, run the model, then trim.
    ref = ref_latents.to(dtype=noisy_latents.dtype, device=noisy_latents.device)
    if getattr(t, "refcn_zero_mean_ref", False):
        ref = ref - ref.mean(dim=(-2, -1), keepdim=True)
    ref_5d   = ref.unsqueeze(2)                 # [B, C, 1, H, W]
    tgt_5d   = noisy_latents.unsqueeze(2)       # [B, C, 1, H, W]
    x_5d     = torch.cat([tgt_5d, ref_5d], dim=2)  # [B, C, 2, H, W]

    timesteps_01 = timesteps.float() / 1000.0  # [B] in [0, 1]

    B, _, H, W = noisy_latents.shape
    # padding_mask is [B, 1, H, W] — prepare_embedded_sequence repeats it
    # along the temporal axis internally, so we do NOT put T in here.
    padding_mask = torch.zeros(B, 1, H, W, dtype=noisy_latents.dtype, device=noisy_latents.device)

    # sd-scripts forward() passes **kwargs → forward_mini_train_dit; no ref_latents kwarg needed.
    out = t.unet(x_5d, timesteps_01, cross, padding_mask=padding_mask)
    # Trim output back to the target frame only: [B, C, 2, H, W] → [B, C, 1, H, W]
    return out[:, :, :1, :, :].squeeze(2)  # [B, C, H, W]


# ---------------------------------------------------------------------------
# Text model wrapper
# ---------------------------------------------------------------------------

def _parse_weighted_tag(tag: str):
    """Parse (tag:weight), (tag), [tag] syntax. Returns (clean_text, weight)."""
    import re
    s = tag.strip()
    if not s:
        return "", 1.0
    m = re.fullmatch(r"\(\s*(.+?)\s*:\s*([+-]?\d+(?:\.\d+)?)\s*\)", s)
    if m:
        return m.group(1).strip(), float(m.group(2))
    w = 1.0
    while True:
        s2 = s.strip()
        if len(s2) >= 2 and s2[0] == "(" and s2[-1] == ")":
            s = s2[1:-1].strip(); w *= 1.1; continue
        if len(s2) >= 2 and s2[0] == "[" and s2[-1] == "]":
            s = s2[1:-1].strip(); w /= 1.1; continue
        break
    return s.strip(), float(w)


def _tokenize_t5_weighted(tokenizer, texts, max_length=512):
    """Per-tag T5 tokenization with weight support, matching AnimaLoraToolkit."""
    if isinstance(texts, str):
        texts = [texts]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 1
    all_ids, all_w = [], []
    for text in texts:
        tags = [t.strip() for t in str(text).split(",") if t.strip()]
        ids, ws = [], []
        for tag in tags:
            clean_tag, weight = _parse_weighted_tag(tag)
            if not clean_tag:
                continue
            tok = tokenizer(clean_tag, add_special_tokens=False)
            for tid in tok["input_ids"]:
                ids.append(int(tid)); ws.append(float(weight))
        ids.append(int(eos_id)); ws.append(1.0)
        if max_length and len(ids) > max_length:
            ids = ids[:max_length - 1] + [int(eos_id)]
            ws  = ws[:max_length - 1] + [1.0]
        all_ids.append(torch.tensor(ids, dtype=torch.long))
        all_w.append(torch.tensor(ws, dtype=torch.float32))
    max_len = max(x.numel() for x in all_ids) if all_ids else 1
    input_ids     = torch.full((len(all_ids), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(all_ids), max_len), dtype=torch.long)
    for i, (ids, ws) in enumerate(zip(all_ids, all_w)):
        L = ids.numel()
        input_ids[i, :L] = ids
        attention_mask[i, :L] = 1
    return input_ids, attention_mask


class AnimaTextModel:
    """Wrapper around Qwen3 (+ optional T5) encoders providing the encode_text() interface.

    encode_text() returns (cond_tuple, None) where cond_tuple is:
        (prompt_embeds [B, L, D], qwen3_attn_mask [B, L], t5_input_ids [B, T] or None, t5_attn_mask [B, T] or None)
    """

    def __init__(self, qwen3_encoder, qwen3_path, t5_tokenizer_path, device, dtype):
        from transformers import AutoTokenizer, T5Tokenizer
        self.encoder = qwen3_encoder
        self.qwen3_tokenizer = AutoTokenizer.from_pretrained(qwen3_path, trust_remote_code=True)
        self.t5_tokenizer = T5Tokenizer.from_pretrained(t5_tokenizer_path) if t5_tokenizer_path else None
        self.device = device
        self.dtype = dtype
        # Compatibility with LoRANetwork which reads text_encoders[0]
        self.text_encoders = [qwen3_encoder]

    def encode_text(self, prompts):
        """Encode text prompts into Anima conditioning tensors."""
        if isinstance(prompts, str):
            prompts = [prompts]
        prompts = [p if p.strip() else " " for p in prompts]

        # Qwen3 — clean text (strip weights), no special tokens
        qwen_texts = []
        for p in prompts:
            parts = [_parse_weighted_tag(t)[0] for t in p.split(",") if t.strip()]
            qwen_texts.append(", ".join(x for x in parts if x) or " ")

        tokens = self.qwen3_tokenizer(
            qwen_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=512, add_special_tokens=False,
        )
        qwen3_ids  = tokens["input_ids"].to(self.device)
        qwen3_mask = tokens["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.encoder(
                input_ids=qwen3_ids, attention_mask=qwen3_mask,
                output_hidden_states=True, return_dict=True, use_cache=False,
            )
            prompt_embeds = outputs.hidden_states[-1]
            prompt_embeds = prompt_embeds * qwen3_mask.unsqueeze(-1)

        prompt_embeds = prompt_embeds.to(self.dtype)

        if self.t5_tokenizer is not None:
            t5_ids, t5_mask = _tokenize_t5_weighted(self.t5_tokenizer, prompts)
            t5_ids  = t5_ids.to(self.device)
            t5_mask = t5_mask.to(self.device)
        else:
            t5_ids = t5_mask = None

        cond = (prompt_embeds, qwen3_mask, t5_ids, t5_mask)
        return cond, None

    def to(self, device=None, dtype=None):
        if device is not None:
            self.device = device
        if dtype is not None:
            self.dtype = dtype
        return self

    def requires_grad_(self, val: bool):
        self.encoder.requires_grad_(val)
        return self

    def eval(self):
        self.encoder.eval()
        return self

    def train(self):
        self.encoder.train()
        return self

    def gradient_checkpointing_enable(self):
        if hasattr(self.encoder, "gradient_checkpointing_enable"):
            self.encoder.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        if hasattr(self.encoder, "gradient_checkpointing_disable"):
            self.encoder.gradient_checkpointing_disable()
