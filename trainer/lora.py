import os
import re
import math
from typing import Literal
import torch
import torch.nn as nn
from safetensors.torch import save_file

BLOCKID26=["BASE","IN00","IN01","IN02","IN03","IN04","IN05","IN06","IN07","IN08","IN09","IN10","IN11","M00","OUT00","OUT01","OUT02","OUT03","OUT04","OUT05","OUT06","OUT07","OUT08","OUT09","OUT10","OUT11"]

# Anima: BASE (text encoder) + 28 DiT blocks (B00-B27)
BLOCKID_ANIMA = ["BASE"] + ["B{:02d}".format(x) for x in range(28)]

ANIMA_TARGET_REPLACE_MODULE = ["Block"]

# ---------------------------------------------------------------------------
# Anima layer-type reference  (verified against anima_keys.txt dump)
# ---------------------------------------------------------------------------
# Each of the 28 DiT blocks (B00–B27) contains exactly these 16 LoRA keys:
#
#   self_attn_q_proj              )
#   self_attn_k_proj              ) self-attention projections  ← GOOD to train
#   self_attn_v_proj              )
#   self_attn_output_proj         )
#
#   cross_attn_q_proj             )
#   cross_attn_k_proj             ) cross-attention projections ← GOOD to train
#   cross_attn_v_proj             )
#   cross_attn_output_proj        )
#
#   mlp_layer1                    ) feed-forward MLP            ← GOOD to train
#   mlp_layer2                    )
#
#   adaln_modulation_self_attn_1  )
#   adaln_modulation_self_attn_2  )
#   adaln_modulation_cross_attn_1 ) per-component AdaLN         ← AVOID (see note)
#   adaln_modulation_cross_attn_2 )
#   adaln_modulation_mlp_1        )
#   adaln_modulation_mlp_2        )
#
# adaln_modulation note:
#   Each of the 6 adaln_modulation layers converts the shared timestep
#   embedding into per-component scale+shift parameters for one sub-layer's
#   layer norm.  Together they represent 168 of the 448 total LoRA keys
#   (37.5 %).  Training them with LoRA offsets the denoising schedule in a
#   way that accumulates across all 28 blocks and all timesteps — including
#   those not seen in training — which tends to produce artefacts.  The
#   standard recommendation is to exclude them.
#
# Common filter strings (set network_module_filter in the UI):
#   ""                                  → train all 448 layers (default)
#   "!adaln_modulation"                 → skip adaln (recommended) — 280 layers
#   "self_attn, cross_attn"             → attention only — 224 layers
#   "self_attn, cross_attn, mlp"        → attn + MLP — 280 layers (same count,
#                                          different set from !adaln_modulation)
#   "self_attn"                         → self-attention only — 112 layers
# ---------------------------------------------------------------------------


def _matches_module_filter(lora_name: str, filter_str: str) -> bool:
    """Return True if *lora_name* passes the user-supplied regex filter.

    *filter_str* is a comma- or newline-separated list of regex patterns.
    Patterns prefixed with ``!`` are *exclusions*; all others are *inclusions*.

    Matching rules
    --------------
    * Empty filter            → include everything (default behaviour).
    * Inclusion patterns only → include only modules that match at least one.
    * Exclusion patterns only → include everything *except* excluded modules.
    * Mixed                   → a module must match an inclusion pattern AND
                                 must not match any exclusion pattern.

    Matching is case-insensitive.
    """
    patterns = [p.strip() for p in re.split(r'[,\n]+', filter_str) if p.strip()]
    if not patterns:
        return True

    inclusions = [p for p in patterns if not p.startswith('!')]
    exclusions = [p[1:] for p in patterns if p.startswith('!')]

    def _safe_search(pattern, text):
        try:
            return bool(re.search(pattern, text, re.IGNORECASE))
        except re.error:
            return False

    # Inclusion gate
    if inclusions:
        if not any(_safe_search(inc, lora_name) for inc in inclusions):
            return False

    # Exclusion gate
    for exc in exclusions:
        if exc and _safe_search(exc, lora_name):
            return False

    return True


# ---------------------------------------------------------------------------
# Literal LoRA key list for all 28 Anima DiT blocks
# Sourced directly from anima_keys.txt (448 keys, 16 per block).
# ---------------------------------------------------------------------------

# Exact sublayer names in the order they appear per block.
_ANIMA_BLOCK_SUBLAYERS = (
    "adaln_modulation_cross_attn_1",
    "adaln_modulation_cross_attn_2",
    "adaln_modulation_mlp_1",
    "adaln_modulation_mlp_2",
    "adaln_modulation_self_attn_1",
    "adaln_modulation_self_attn_2",
    "cross_attn_k_proj",
    "cross_attn_output_proj",
    "cross_attn_q_proj",
    "cross_attn_v_proj",
    "mlp_layer1",
    "mlp_layer2",
    "self_attn_k_proj",
    "self_attn_output_proj",
    "self_attn_q_proj",
    "self_attn_v_proj",
)


def generate_anima_preview_keys():
    """Return the full list of 448 Anima LoRA key names (16 per block × 28 blocks).

    Keys are identical to those produced during training; sourced from
    anima_keys.txt.
    """
    return [
        f"lora_unet_blocks_{b}_{sl}"
        for b in range(28)
        for sl in _ANIMA_BLOCK_SUBLAYERS
    ]

BLOCKS=["encoder",
"diffusion_model_input_blocks_0_",
"diffusion_model_input_blocks_1_",
"diffusion_model_input_blocks_2_",
"diffusion_model_input_blocks_3_",
"diffusion_model_input_blocks_4_",
"diffusion_model_input_blocks_5_",
"diffusion_model_input_blocks_6_",
"diffusion_model_input_blocks_7_",
"diffusion_model_input_blocks_8_",
"diffusion_model_input_blocks_9_",
"diffusion_model_input_blocks_10_",
"diffusion_model_input_blocks_11_",
"diffusion_model_middle_block_",
"diffusion_model_output_blocks_0_",
"diffusion_model_output_blocks_1_",
"diffusion_model_output_blocks_2_",
"diffusion_model_output_blocks_3_",
"diffusion_model_output_blocks_4_",
"diffusion_model_output_blocks_5_",
"diffusion_model_output_blocks_6_",
"diffusion_model_output_blocks_7_",
"diffusion_model_output_blocks_8_",
"diffusion_model_output_blocks_9_",
"diffusion_model_output_blocks_10_",
"diffusion_model_output_blocks_11_",
"embedders"]

def to26(ratios):
    ids = BLOCKIDS[BLOCKNUMS.index(len(ratios))]
    output = [0]*26
    for i, id in enumerate(ids):
        output[BLOCKID26.index(id)] = ratios[i]
    output = [bool(x) for x in output]
    return output

UNET_TARGET_REPLACE_MODULE = ["Transformer2DModel"]
UNET_TARGET_REPLACE_MODULE_CONV2D_3X3 = ["ResnetBlock2D", "Downsample2D", "Upsample2D"]
UNET_TARGET_REPLACE_MODULE_C3 = UNET_TARGET_REPLACE_MODULE + UNET_TARGET_REPLACE_MODULE_CONV2D_3X3
TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
LORA_PREFIX_UNET = "lora_unet"
LORA_PREFIX_TEXT_ENCODER = "lora_te"
# SDXL: must starts with LORA_PREFIX_TEXT_ENCODER
LORA_PREFIX_TEXT_ENCODER1 = "lora_te1"
LORA_PREFIX_TEXT_ENCODER2 = "lora_te2"

LORA_LINEAR = ["Linear", "LoRACompatibleLinear"]
LORA_CONV = ["Conv2d", "LoRACompatibleConv"]

MMDIT_TARGET_REPLACE_MODULE = ["JointTransformerBlock", "FluxTransformerBlock", "FluxSingleTransformerBlock", "AuraFlowJointTransformerBlock", "AuraFlowSingleTransformerBlock"]
LORA_PREFIX_MMDIT = 'lora_transformer'
# Anima LoRAs use lora_unet_ prefix (index 0) for compatibility with sd-scripts

PREFIXLIST = [
    LORA_PREFIX_UNET,
    LORA_PREFIX_TEXT_ENCODER,
    LORA_PREFIX_TEXT_ENCODER1,
    LORA_PREFIX_TEXT_ENCODER2,
    LORA_PREFIX_MMDIT
]

TRAINING_METHODS = Literal[
    "noxattn",  # train all layers except x-attns and time_embed layers
    "innoxattn",  # train all layers except self attention layers
    "selfattn",  # ESD-u, train only self attention layers
    "xattn",  # ESD-x, train only x attention layers
    "full",  #  train all layers
    # "notime",
    # "xlayer",
    # "outxattn",
    # "outsattn",
    # "inxattn",
    # "inmidsattn",
    # "selflayer",
]

class LoRAModule(nn.Module):
    """
    replaces forward method of the original Linear, instead of replacing the original Linear module.
    """

    def __init__(
        self,
        lora_name,
        org_module: nn.Module,
        multiplier=1.0,
        lora_dim=4,
        alpha=1,
        up_weight = None,
        down_weight = None
    ):
        """if alpha == 0 or None, alpha is rank (no scaling)."""
        super().__init__()
        self.lora_name = lora_name
        self.lora_dim = lora_dim

        if org_module.__class__.__name__ in LORA_LINEAR:
            in_dim = org_module.in_features
            out_dim = org_module.out_features
            self.lora_down = nn.Linear(in_dim, lora_dim, bias=False)
            self.lora_up = nn.Linear(lora_dim, out_dim, bias=False)

        elif org_module.__class__.__name__ in LORA_CONV:
            in_dim = org_module.in_channels
            out_dim = org_module.out_channels

            kernel_size = org_module.kernel_size
            stride = org_module.stride
            padding = org_module.padding
            self.lora_down = nn.Conv2d(in_dim, self.lora_dim, kernel_size, stride, padding, bias=False)
            self.lora_up = nn.Conv2d(self.lora_dim, out_dim, (1, 1), (1, 1), bias=False)

        if type(alpha) == torch.Tensor:
            alpha = alpha.detach().numpy()
        alpha = lora_dim if alpha is None or alpha == 0 else alpha
        self.scale = alpha / self.lora_dim
        self.register_buffer("alpha", torch.tensor(alpha))  # 定数として扱える

        # same as microsoft's
        if down_weight is None:
            nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        else:
            self.lora_down.weight = torch.nn.Parameter(down_weight)
        if up_weight is None:
            nn.init.zeros_(self.lora_up.weight)
        else:
            self.lora_up.weight = torch.nn.Parameter(up_weight)

        self.multiplier = multiplier
        self.org_module = org_module  # remove in applying

    def apply_to(self):
        self.org_forward = self.org_module.forward
        self.org_module.forward = self.forward
        del self.org_module

    def forward(self, x, scale = None):
        return (
            self.org_forward(x)
            + self.lora_up(self.lora_down(x)) * self.multiplier * self.scale
        )

class LoRANetwork(nn.Module):
    def __init__(self, t):
        super().__init__()

        print("Creating LoRA Network")

        self.multiplier = 1
        self.lora_dim = t.network_rank
        self.alpha = t.network_alpha
        self.conv_lora_dim = t.network_rank   # Anima DiT has no conv layers; mirrors rank
        self.conv_alpha = t.network_alpha
        self.unet_lr = t.train_learning_rate
        self.te_lr = getattr(t, 'train_textencoder_learning_rate', None) or t.train_learning_rate

        self.module = LoRAModule
        self.is_anima = getattr(t, 'is_anima', False)
        self.llrd_decay = float(getattr(t, 'network_llrd_decay', 1.0))

        if self.is_anima:
            t.lora_unet_target = ANIMA_TARGET_REPLACE_MODULE
        elif t.is_dit:
            t.lora_unet_target = MMDIT_TARGET_REPLACE_MODULE
        else:
            t.lora_unet_target = UNET_TARGET_REPLACE_MODULE_C3 if t.network_type == "c3lier" else UNET_TARGET_REPLACE_MODULE

        t.lora_te_target = TEXT_ENCODER_TARGET_REPLACE_MODULE

        # Anima uses ut=0 (lora_unet prefix); other DiT models use ut=4 (lora_transformer prefix)
        ut_unet = 0 if getattr(t, 'is_anima', False) else (4 if t.is_dit else 0)
        self.unet_loras = self.create_modules(t, ut_unet)

        if "BASE" in t.network_blocks:
            self.te_loras = self.create_modules(t, 2 if t.is_te2 else 1)
            if t.is_te2:
                self.te_loras += self.create_modules(t, 3)
        else:
            self.te_loras = []
        
        print(f" Create LoRA for U-Net or DiT: {len(self.unet_loras)} modules.")
        print(f" Create LoRA for Textencoder : {len(self.te_loras)} modules.")

        lora_names = set()
        for lora in self.unet_loras + self.te_loras:
            assert (
                lora.lora_name not in lora_names
            ), f"duplicated lora name: {lora.lora_name}. {lora_names}"
            lora_names.add(lora.lora_name)

        for lora in self.unet_loras + self.te_loras:
            lora.apply_to()
            self.add_module(lora.lora_name,lora)

        torch.cuda.empty_cache()

    def create_modules(self, t, ut):

        loras = []
        elements = "Full"

        prefix = PREFIXLIST[ut]

        target = t.lora_te_target if ut > 0 else t.lora_unet_target

        if ut == 0 or ut == 4:
            root_modules = t.unet.named_modules()
        elif ut == 1 or ut == 2:
            root_modules = t.text_model.text_encoders[0].named_modules()
        elif ut == 3:
            root_modules = t.text_model.text_encoders[1].named_modules()

        for name, module in root_modules:
            if module.__class__.__name__ in target:
                for child_name, child_module in module.named_modules():
                    is_linear = child_module.__class__.__name__ in LORA_LINEAR
                    is_conv2d = child_module.__class__.__name__ in LORA_CONV
                    is_conv2d_1x1 = is_conv2d and child_module.kernel_size == (1, 1)

                    if elements == "SelfAttention":
                        if "self_attn" not in child_name and "attn1" not in child_name:
                            continue
                    elif elements == "CrossAttention":
                        if "cross_attn" not in child_name and "attn2" not in child_name:
                            continue
                    elif "Full" in elements:
                        pass

                    if is_linear or is_conv2d:
                        lora_name = prefix + "." + name + "." + child_name
                        lora_name = lora_name.replace(".", "_")

                        if getattr(t, 'is_anima', False):
                            # Anima: detect block by _blocks_N_ pattern
                            m = re.search(r'_blocks_(\d+)_', lora_name)
                            if m:
                                currentblock = "B{:02d}".format(int(m.group(1)))
                            else:
                                currentblock = "BASE"
                        else:
                            key = convert_diffusers_name_to_compvis(lora_name, t.is_sd2)
                            currentblock = "BASE"
                            for i, block in enumerate(BLOCKS):
                                if block in key:
                                    if i == 26:
                                        i = 0
                                    currentblock = BLOCKID26[i]

                        if currentblock not in t.network_blocks:
                            continue

                        # Per-layer regex filter (Anima: skip adaln_modulation etc.)
                        module_filter = getattr(t, 'network_module_filter', '') or ''
                        if module_filter.strip() and not _matches_module_filter(lora_name, module_filter):
                            continue

                        # Subspace guard: restrict to mapped layers only
                        allowed = getattr(t, 'subspace_guard_allowed_names', None)
                        if allowed is not None and lora_name not in allowed:
                            continue

                        if is_linear or is_conv2d_1x1:
                            dim = self.lora_dim
                            alpha = self.alpha
                        elif self.conv_lora_dim is not None:
                            dim = self.conv_lora_dim
                            alpha = self.conv_alpha

                        t.db(key if not getattr(t, 'is_anima', False) else lora_name)
                        lora = self.module(
                            lora_name, child_module, 1, dim, alpha)
                        loras.append(lora)

        return loras

    def load_fromfile(self, t, ut):
        loras = []

        file = t.network_resume

        prefix = PREFIXLIST[ut]

        target = t.lora_te_target if ut > 0 else t.lora_unet_target

        if ut == 0 or ut == 4:
            root_modules = t.unet.named_modules()
        elif ut == 1 or ut == 2:
            root_modules = t.text_model.text_encoders[0].named_modules()
        elif ut == 3:
            root_modules = t.text_model.text_encoders[1].named_modules()

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file, safe_open
            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

        for name, module in root_modules:
            if module.__class__.__name__ in target:
                for child_name, child_module in module.named_modules():
                    lora_name = prefix + "." + name + "." + child_name
                    lora_name = lora_name.replace(".", "_")

                    if lora_name + ".alpha" in weights_sd:
                        alpha = weights_sd[lora_name + ".alpha"]
                        up = weights_sd[lora_name + ".lora_up.weight"]
                        down = weights_sd[lora_name + ".lora_down.weight"]
                        dim = down.shape[0]
                        lora = self.module(
                            lora_name, child_module, 1, dim, alpha,  up_weight = up, down_weight = down)
                        loras.append(lora)
        return loras

    def prepare_optimizer_params(self):
        all_params = []

        if self.te_loras:
            params = []
            [params.extend(lora.parameters()) for lora in self.te_loras]
            param_data = {'params': params}
            if self.te_lr is not None:
                param_data['lr'] = self.te_lr
            all_params.append(param_data)

        if self.unet_loras:
            if self.is_anima and self.llrd_decay != 1.0:
                # Layer-wise Learning Rate Decay (LLRD):
                # Group LoRA modules by block index, then apply a geometric LR
                # multiplier so the last block trains at base_lr and each earlier
                # block is scaled down by decay^(max_block - block_idx).
                # This compensates for uneven gradient magnitudes across depth.
                block_loras = {}  # block_idx (int) -> [loras]
                unmatched = []
                for lora in self.unet_loras:
                    m = re.search(r'_blocks_(\d+)_', lora.lora_name)
                    if m:
                        block_loras.setdefault(int(m.group(1)), []).append(lora)
                    else:
                        unmatched.append(lora)

                if block_loras:
                    max_block = max(block_loras.keys())
                    lr_min = self.unet_lr * (self.llrd_decay ** max_block)
                    lr_max = self.unet_lr
                    print(f"LLRD: {len(block_loras)} block groups, decay={self.llrd_decay}, "
                          f"LR range [{lr_min:.2e} – {lr_max:.2e}]")
                    for block_idx in sorted(block_loras.keys()):
                        depth = max_block - block_idx
                        block_lr = self.unet_lr * (self.llrd_decay ** depth)
                        print(f"block_idx: {block_idx}, depth={depth}, block_lr={block_lr}")
                        params = []
                        [params.extend(lora.parameters()) for lora in block_loras[block_idx]]
                        all_params.append({'params': params, 'lr': block_lr})

                if unmatched:
                    params = []
                    [params.extend(lora.parameters()) for lora in unmatched]
                    all_params.append({'params': params, 'lr': self.unet_lr})
            else:
                params = []
                [params.extend(lora.parameters()) for lora in self.unet_loras]
                param_data = {'params': params}
                if self.unet_lr is not None:
                    param_data['lr'] = self.unet_lr
                all_params.append(param_data)

        return all_params

    def save_weights(self, file, t, metaname):
        state_dict = self.state_dict()
        dtype = t.save_precision
        metadata = t.metadata

        for key in metadata:
            metadata[key] = str(metadata[key])
        metadata["ss_output_name"] = metaname

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        for key in list(state_dict.keys()):
            if not key.startswith("lora"):
                del state_dict[key]

        # network_strength hardcoded to 1 — no weight scaling applied

        base, ext = os.path.splitext(file)
        attempt = 0
        while True:
            try:
                if ext == ".safetensors":
                    save_file(state_dict, file, metadata)  # safetensors用の保存処理
                else:
                    torch.save(state_dict, file)  # 通常のtorch.saveで保存
                print(f"Successfully saved to {file}")
                break  # 保存が成功したらループを抜ける
            except Exception as e:
                print(f"Error saving to {file}: {e}")
                attempt += 1
                file = f"{base}_{attempt}{ext}"  # ファイル名を変更
                print(f"Retrying with filename {file}...")
        
        return file

    def __enter__(self):
        for lora in self.unet_loras + self.te_loras:
            lora.multiplier = 1.0

    def __exit__(self, exc_type, exc_value, tb):
        for lora in self.unet_loras + self.te_loras:
            lora.multiplier = 0

    def check_weight(self, te = False):
        sums = []
        for lora in self.te_loras if te else self.unet_loras:
            sums.append(torch.sum(lora.lora_up.weight).item())
        return sums

    def set_multiplier(self, num):
        for lora in self.unet_loras + self.te_loras:
            lora.multiplier = num

from lycoris.modules.loha import LohaModule
from lycoris.modules.norms import NormModule

network_module_dict = {
    "loha": LohaModule,
}

HADAMEN = ["alpha","hada_t1","hada_w1_a","hada_w1_b","hada_t2","hada_w2_a","hada_w2_b"]

class LycorisNetwork(torch.nn.Module):
    ENABLE_CONV = True
    UNET_TARGET_REPLACE_MODULE = [
            "Transformer2DModel",
            "ResnetBlock2D",
            "Downsample2D",
            "Upsample2D",
        ]
    UNET_TARGET_REPLACE_NAME =[]
    TEXT_ENCODER_TARGET_REPLACE_MODULE = ["CLIPAttention", "CLIPMLP"]
    TEXT_ENCODER_TARGET_REPLACE_NAME = []
    LORA_PREFIX = "lora"
    MODULE_ALGO_MAP = {}
    NAME_ALGO_MAP = {}

    def __init__(
        self,
        t,
        use_tucker=True,
        dropout=0,
        rank_dropout=0,
        module_dropout=0,
        norm_modules=NormModule,
        train_norm=False,
        **kwargs,
    ) -> None:
        super().__init__()
        network_module = t.network_type
        root_kwargs = kwargs
        self.multiplier = 1
        self.lora_dim = t.network_rank
        conv_lora_dim = t.network_rank   # Anima DiT has no conv layers

        network_module_dict[network_module].forward = forwards[network_module]

        t.lora_unet_target = UNET_TARGET_REPLACE_MODULE_C3
        t.lora_te_target = TEXT_ENCODER_TARGET_REPLACE_MODULE

        self.unet_lr = t.train_learning_rate
        self.te_lr = getattr(t, 'train_textencoder_learning_rate', None) or t.train_learning_rate

        if not self.ENABLE_CONV:
            conv_lora_dim = 0

        self.conv_lora_dim = int(conv_lora_dim)
        if self.conv_lora_dim and self.conv_lora_dim != self.lora_dim:
            print("Apply different lora dim for conv layer")
            print(f"Conv Dim: {conv_lora_dim}, Linear Dim: {t.network_rank}")
        elif self.conv_lora_dim == 0:
            print("Disable conv layer")

        self.alpha = t.network_alpha
        self.conv_alpha = float(t.network_alpha)
        if self.conv_lora_dim and self.alpha != self.conv_alpha:
            print("Apply different alpha value for conv layer")
            print(f"Conv alpha: {self.conv_alpha}, Linear alpha: {t.network_alpha}")

        if 1 >= dropout >= 0:
            print(f"Use Dropout value: {dropout}")
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout

        self.use_tucker = use_tucker

        print("Create LyCORIS Module")

        def create_single_module(
            lora_name: str,
            module: torch.nn.Module,
            algo_name,
            dim=None,
            alpha=None,
            use_tucker=self.use_tucker,
            **kwargs,
        ):
            for k, v in root_kwargs.items():
                if k in kwargs:
                    continue
                kwargs[k] = v

            if train_norm and "Norm" in module.__class__.__name__:
                return norm_modules(
                    lora_name,
                    module,
                    self.multiplier,
                    self.rank_dropout,
                    self.module_dropout,
                    **kwargs,
                )
            lora = None
            if module.__class__.__name__ in LORA_LINEAR and t.network_rank > 0:
                dim = dim or t.network_rank
                alpha = alpha or self.alpha
                if module.__class__.__name__ == LORA_LINEAR[1]:
                    module.__class__.__name__ = LORA_LINEAR[0]
            elif module.__class__.__name__ in LORA_CONV:
                k_size, *_ = module.kernel_size
                if module.__class__.__name__ == LORA_CONV[1]:
                    module.__class__.__name__ = LORA_CONV[0]
                if k_size == 1 and t.network_rank > 0:
                    dim = dim or t.network_rank
                    alpha = alpha or self.alpha
                elif conv_lora_dim > 0 or dim:
                    dim = dim or conv_lora_dim
                    alpha = alpha or self.conv_alpha
                else:
                    return None
            else:
                return None
            lora = network_module_dict[algo_name](
                lora_name,
                module,
                self.multiplier,
                dim,
                alpha,
                self.dropout,
                self.rank_dropout,
                self.module_dropout,
                use_tucker,
                **kwargs,
            )
            return lora

        def create_modules_(
            prefix: str,
            root_module: torch.nn.Module,
            algo,
            configs={},
        ):
            loras = {}
            lora_names = []
            for name, module in root_module.named_modules():
                module_name = module.__class__.__name__
                if module_name in self.MODULE_ALGO_MAP:
                    next_config = self.MODULE_ALGO_MAP[module_name]
                    next_algo = next_config.get("algo", algo)
                    new_loras, new_lora_names = create_modules_(
                        f"{prefix}_{name}", module, next_algo, next_config
                    )
                    for lora_name, lora in zip(new_lora_names, new_loras):
                        if lora_name not in loras:
                            loras[lora_name] = lora
                            lora_names.append(lora_name)
                    continue
                lora_name = prefix + "." + name
                lora_name = lora_name.replace(".", "_")
                if lora_name in loras:
                    continue
                lora_name = lora_name.replace("text_encoder_","")
                lora = create_single_module(lora_name, module, algo, **configs)
                if lora is not None:
                    loras[lora_name] = lora
                    lora_names.append(lora_name)
            return [loras[lora_name] for lora_name in lora_names], lora_names

        # create module instances
        def create_modules(self, t, ut, target_replace_names = []):

            loras = []
            elements = "Full"

            prefix = PREFIXLIST[ut]

            target = t.lora_te_target if ut > 0 else t.lora_unet_target

            if ut == 0:
                root_modules = t.unet.named_modules()
            elif ut ==1 or ut == 2:
                root_modules = t.text_model.text_encoders[0].named_modules()
            elif ut ==3:
                root_modules = t.text_model.text_encoders[1].named_modules()

            loras = []
            next_config = {}
            for name, module in root_modules:
                module_name = module.__class__.__name__
                if module_name in target:
                    if module_name in self.MODULE_ALGO_MAP:
                        next_config = self.MODULE_ALGO_MAP[module_name]
                        algo = next_config.get("algo", network_module)
                    else:
                        algo = network_module
                    loras.extend(
                        create_modules_(f"{prefix}_{name}", module, algo, next_config)[
                            0
                        ]
                    )
                    next_config = {}
                elif name in target_replace_names or any(
                    re.match(t, name) for t in target_replace_names
                ):
                    if name in self.NAME_ALGO_MAP:
                        next_config = self.NAME_ALGO_MAP[name]
                        algo = next_config.get("algo", network_module)
                    elif module_name in self.MODULE_ALGO_MAP:
                        next_config = self.MODULE_ALGO_MAP[module_name]
                        algo = next_config.get("algo", network_module)
                    else:
                        algo = network_module
                    lora_name = prefix + "." + name
                    lora_name = lora_name.replace(".", "_")
                    lora = create_single_module(lora_name, module, algo, **next_config)
                    next_config = {}
                    if lora is not None:
                        loras.append(lora)
            return loras

        self.unet_loras = create_modules(self, t, 0)
        self.apply_block_weight(t)
        
        if "BASE" in t.network_blocks and t.mode == "LoRA":
            self.te_loras = create_modules(self, t, 2 if t.is_te2 else 1)
            if t.is_te2:
                self.te_loras += create_modules(self, t, 3)
        else:
            self.te_loras = []


        print(f"Create LyCORIS U-Net or DiT: {len(self.unet_loras)} modules.")
        print(f"Create LyCORIS TextEncoder: {len(self.te_loras)} modules.")

        algo_table = {}
        for lora in self.unet_loras:
            algo_table[lora.__class__.__name__] = (
                algo_table.get(lora.__class__.__name__, 0) + 1
            )
        print(f"module type table: {algo_table}")

        self.weights_sd = None

        # assertion
        names = set()
        for lora in self.unet_loras:
            assert (
                lora.lora_name not in names
            ), f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)

        for lora in self.unet_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

        for lora in self.te_loras:
            lora.apply_to()
            self.add_module(lora.lora_name, lora)

    def apply_block_weight(self, t):
        new_unet = []
        for lora in self.unet_loras:
            if getattr(t, 'is_anima', False):
                m = re.search(r'_blocks_(\d+)_', lora.lora_name)
                currentblock = "B{:02d}".format(int(m.group(1))) if m else "BASE"
            else:
                key = convert_diffusers_name_to_compvis(lora.lora_name, t.is_sd2)
                currentblock = "BASE"
                for i, block in enumerate(BLOCKS):
                    if block in key:
                        if i == 26:
                            i = 0
                        currentblock = BLOCKID26[i]
            if currentblock in t.network_blocks:
                new_unet.append(lora)

        self.unet_loras = new_unet

    def load_fromfile(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file, safe_open
            state_dict = load_file(file)
        else:
            state_dict = torch.load(file, map_location="cpu")

        new_unet = []
        new_te = []
        for lora in self.unet_loras:
            if lora.lora_name + ".alpha" in state_dict:
                for men in HADAMEN:
                    if lora.lora_name + "." + men in state_dict:
                        setattr(lora, men, torch.nn.Parameter(state_dict[lora.lora_name + "." + men]))
                new_unet.append(lora)

        for lora in self.te_loras:
            if lora.lora_name + ".alpha" in state_dict:
                for men in HADAMEN:
                    if lora.lora_name + "." + men in state_dict:
                        setattr(lora, men, state_dict[lora.lora_name + "." + men])
                new_te.append(lora)
        
        self.unet_loras = new_unet
        self.te_loras = new_te

    def load_weights(self, file):
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file, safe_open

            self.weights_sd = load_file(file)
        else:
            self.weights_sd = torch.load(file, map_location="cpu")
        missing, unexpected = self.load_state_dict(self.weights_sd, strict=False)
        state = {}
        if missing:
            state["missing keys"] = missing
        if unexpected:
            state["unexpected keys"] = unexpected
        return state

    def enable_gradient_checkpointing(self):
        # not supported
        def make_ckpt(module):
            if isinstance(module, torch.nn.Module):
                module.grad_ckpt = True

        self.apply(make_ckpt)
        pass

    def prepare_optimizer_params(self):
        def enumerate_params(loras):
            params = []
            for lora in loras:
                params.extend(lora.parameters())
            return params

        self.requires_grad_(True)
        all_params = []

        if self.unet_loras:
            param_data = {"params": enumerate_params(self.unet_loras)}
            param_data["lr"] = self.unet_lr
            all_params.append(param_data)

        if self.te_loras:
            param_data = {"params": enumerate_params(self.te_loras)}
            param_data["lr"] = self.te_lr
            all_params.append(param_data)

        return all_params

    def __enter__(self):
        for lora in self.unet_loras + self.te_loras:
            lora.multiplier = 1

    def __exit__(self, exc_type, exc_value, tb):
        for lora in self.unet_loras + self.te_loras:
            lora.multiplier = 0

    def set_multiplier(self, num):
        for lora in self.unet_loras + self.te_loras:
            lora.multiplier = num

    def prepare_grad_etc(self):
        self.requires_grad_(True)

    def on_epoch_start(self):
        self.train()

    def get_trainable_params(self):
        return self.parameters()

    def save_weights(self, file, t, metaname):
        state_dict = self.state_dict()
        dtype = t.save_precision
        metadata = t.metadata

        for key in metadata:
            metadata[key] = str(metadata[key])
        metadata["ss_output_name"] = metaname

        if dtype is not None:
            for key in list(state_dict.keys()):
                v = state_dict[key]
                v = v.detach().clone().to("cpu").to(dtype)
                state_dict[key] = v

        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import save_file

            save_file(state_dict, file, metadata)
        else:
            torch.save(state_dict, file)

###forwards####################################
def loha_forward(self, x: torch.Tensor, *args, **kwargs):
    if self.module_dropout and self.training:
        if torch.rand(1) < self.module_dropout:
            return self.op(
                x,
                self.org_module[0].weight.data,
                (
                    None
                    if self.org_module[0].bias is None
                    else self.org_module[0].bias.data
                ),
            )
    if self.bypass_mode:
        return self.bypass_forward(x, scale=self.multiplier)
    else:
        diff_weight = self.get_weight(self.shape).to(self.dtype) * self.scalar
        weight = self.org_module[0].weight.data.to(self.dtype)
        if self.wd:
            weight = self.apply_weight_decompose(
                weight + diff_weight, self.multiplier
            )
        else:
            weight = weight + diff_weight * self.multiplier
        bias = (
            None
            if self.org_module[0].bias is None
            else self.org_module[0].bias.data
        )
        return self.op(x, weight, bias, **self.kw_dict)

forwards = {"loha": loha_forward}


re_digits = re.compile(r"\d+")
re_x_proj = re.compile(r"(.*)_([qkv]_proj)$")
re_compiled = {}

suffix_conversion = {
    "attentions": {},
    "resnets": {
        "conv1": "in_layers_2",
        "conv2": "out_layers_3",
        "norm1": "in_layers_0",
        "norm2": "out_layers_0",
        "time_emb_proj": "emb_layers_1",
        "conv_shortcut": "skip_connection",
    }
}

def convert_diffusers_name_to_compvis(key, is_sd2):
    def match(match_list, regex_text):
        regex = re_compiled.get(regex_text)
        if regex is None:
            regex = re.compile(regex_text)
            re_compiled[regex_text] = regex

        r = re.match(regex, key)
        if not r:
            return False

        match_list.clear()
        match_list.extend([int(x) if re.match(re_digits, x) else x for x in r.groups()])
        return True

    m = []

    if match(m, r"lora_unet_conv_in(.*)"):
        return f'diffusion_model_input_blocks_0_0{m[0]}'

    if match(m, r"lora_unet_conv_out(.*)"):
        return f'diffusion_model_out_2{m[0]}'

    if match(m, r"lora_unet_time_embedding_linear_(\d+)(.*)"):
        return f"diffusion_model_time_embed_{m[0] * 2 - 2}{m[1]}"

    if match(m, r"lora_unet_down_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"diffusion_model_input_blocks_{1 + m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"

    if match(m, r"lora_unet_mid_block_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[0], {}).get(m[2], m[2])
        return f"diffusion_model_middle_block_{1 if m[0] == 'attentions' else m[1] * 2}_{suffix}"

    if match(m, r"lora_unet_up_blocks_(\d+)_(attentions|resnets)_(\d+)_(.+)"):
        suffix = suffix_conversion.get(m[1], {}).get(m[3], m[3])
        return f"diffusion_model_output_blocks_{m[0] * 3 + m[2]}_{1 if m[1] == 'attentions' else 0}_{suffix}"

    if match(m, r"lora_unet_down_blocks_(\d+)_downsamplers_0_conv"):
        return f"diffusion_model_input_blocks_{3 + m[0] * 3}_0_op"

    if match(m, r"lora_unet_up_blocks_(\d+)_upsamplers_0_conv"):
        return f"diffusion_model_output_blocks_{2 + m[0] * 3}_{2 if m[0]>0 else 1}_conv"

    if match(m, r"lora_te_text_model_encoder_layers_(\d+)_(.+)"):
        if is_sd2:
            if 'mlp_fc1' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc1', 'mlp_c_fc')}"
            elif 'mlp_fc2' in m[1]:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc2', 'mlp_c_proj')}"
            else:
                return f"model_transformer_resblocks_{m[0]}_{m[1].replace('self_attn', 'attn')}"

        return f"transformer_text_model_encoder_layers_{m[0]}_{m[1]}"

    if match(m, r"lora_te2_text_model_encoder_layers_(\d+)_(.+)"):
        if 'mlp_fc1' in m[1]:
            return f"1_model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc1', 'mlp_c_fc')}"
        elif 'mlp_fc2' in m[1]:
            return f"1_model_transformer_resblocks_{m[0]}_{m[1].replace('mlp_fc2', 'mlp_c_proj')}"
        else:
            return f"1_model_transformer_resblocks_{m[0]}_{m[1].replace('self_attn', 'attn')}"

    return key
