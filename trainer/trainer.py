import json
import os
import ast
import warnings
import torch
import subprocess
import sys
import torch.nn as nn
import gradio as gr
from datetime import datetime
from typing import Literal
from diffusers.optimization import get_scheduler
from transformers.optimization import AdafactorSchedule
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, CosineAnnealingWarmRestarts, StepLR, MultiStepLR, ReduceLROnPlateau, CyclicLR, OneCycleLR
from pprint import pprint
from accelerate import Accelerator
from trainer.lora import BLOCKID_ANIMA
warnings.filterwarnings("ignore", category=FutureWarning)

all_configs = []

PASS2 = "2nd pass"
POs = ["came", "tiger", "adammini"]

path_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
jsonspath = os.path.join(path_root,"jsons")
os.makedirs(jsonspath, exist_ok=True)
logspath = os.path.join(path_root,"logs")
presetspath = os.path.join(path_root,"presets")
os.makedirs(presetspath, exist_ok=True)

class Trainer():
    def __init__(self, jsononly, model, vae, mode, values):
        self.values = values
        self.mode = mode
        self.use_8bit = False
        self.count_dict = {}
        self.metadata = {}

        self.save_dir = os.environ.get("LORA_DIR", os.path.join(path_root, "output"))
        os.makedirs(self.save_dir, exist_ok=True)
        self.setpass(0)

        self.image_size = [int(x) for x in self.image_size.split(",")]
        if len(self.image_size) == 1:
            self.image_size = self.image_size * 2
        self.image_size.sort()

        # Hardcoded: removed from UI
        self.save_overwrite         = True
        self.save_as_json           = True
        self.logging_save_csv       = False
        self.logging_verbose        = False
        self.train_lr_scheduler_power = 1.0
        self.sub_image_num          = 0
        self.image_max_ratio        = 2.0
        self.image_min_length       = int(self.image_size[0] / self.image_max_ratio)
        self.network_type           = "lierla"
        self.diff_alt_ratio         = 1.0

        self.gradient_accumulation_steps = 1
        self.train_repeat = 1
        self.total_images = 0
        
        self.checkfile()

        # values = [all_configs..., dummy, orig_image, targ_image]
        clen = len(all_configs) + 1  # +1 for dummy checkbox

        self.images = values[clen:]  # [orig_image, targ_image]

        self.add_dcit = {"mode": mode, "model": model, "vae": vae}

        self.export_json(jsononly)

    def setpass(self, pas, set = True):
        values_0 = self.values[:len(all_configs)]
        values_1 = self.values[len(all_configs):len(all_configs) * 2]
        if pas == 1:
            if values_1[-1]:
                if set: print("Use 2nd pass settings")
            else:
                return
        jdict = {}
        for i, (sets, value) in enumerate(zip(all_configs, values_1 if pas > 0 else values_0)):
            jdict[sets[0]] = value
    
            if pas > 0:
                if not sets[5][3]:
                    value = values_0[i]

            if not isinstance(value, sets[4]):
                try:
                    value = sets[4](value)
                except:
                    if not sets[0] == "train_textencoder_learning_rate":
                        print(f"ERROR, input value for {sets[0]} : {sets[4]} is invalid, use default value {sets[3]}")
                    value = sets[3]
            if "precision" in sets[0]:
                if sets[0] == "train_model_precision" and value == "fp8":
                    self.use_8bit == True
                    print("Use 8bit Model Precision")
                value = parse_precision(value)

            if "train_optimizer" == sets[0]:
                value = value.lower()
            
            if "train_optimizer_settings" == sets[0] or "train_lr_scheduler_settings" == sets[0]:
                dvalue = {}
                if value is not None and len(value.strip()) > 0:
                    # 改行や空白を取り除きつつ処理
                    value = value.replace(" ", "").replace(";","\n")
                    args = value.split("\n")
                    for arg in args:
                        if "=" in arg:  # "=" が存在しない場合は無視
                            key, val = arg.split("=", 1)
                            val = ast.literal_eval(val)  # リテラル評価で型を適切に変換
                            dvalue[key] = val
                value = dvalue
            if set:
                setattr(self, sets[0].split("(")[0], value)

        if set:
            self.network_blocks = list(BLOCKID_ANIMA)
        self.mode_fixer()
        return jdict

    savedata = ["model", "vae", ]
    
    def export_json(self, jsononly):
        current_time = datetime.now()
        outdict = self.setpass(0, set=False)
        outdict.update(self.add_dcit)
        today = current_time.strftime("%Y%m%d")
        time = current_time.strftime("%Y%m%d_%H%M%S")
        add = "" if jsononly else f"-{time}"
        jsonpath = os.path.join(presetspath, self.save_lora_name + add + ".json")  if jsononly else os.path.join(jsonspath, today, self.save_lora_name + add + ".json")
        self.csvpath = os.path.join(logspath ,today, self.save_lora_name + add + ".csv")
        
        if self.save_as_json:
            directory = os.path.dirname(jsonpath)
            if not os.path.exists(directory):
                os.makedirs(directory)
            with open(jsonpath, 'w') as file:
                json.dump(outdict, file, indent=4)
        
        if jsononly:
            with open(jsonpath, 'w') as file:
                json.dump(outdict, file, indent=4)  

    def db(self, *obj, pp = False):
        if self.logging_verbose:
            if pp:
                pprint(*obj)
            else:
                print(*obj)

    def checkfile(self):
        if self.save_lora_name == "":
            self.save_lora_name = "untitled"

        filename = os.path.join(self.save_dir, f"{self.save_lora_name}.safetensors")

        self.isfile = os.path.isfile(filename) and not self.save_overwrite
    
    def tagcount(self, prompt):
        tags = [p.strip() for p in prompt.split(",")]

        for tag in tags:
            if tag in self.count_dict:
                self.count_dict[tag] += 1
            else:
                self.count_dict[tag] = 1
    
    # Anima modes: ["LoRA", "ADDifT", "Multi-ADDifT"]
    def mode_fixer(self):
        if self.mode == "ADDifT":
            if self.lora_trigger_word == "" and "BASE" in self.network_blocks:
                self.network_blocks.remove("BASE")

    def sd_typer(self):
        """Set model type properties for Anima."""
        self.is_anima = True
        self.is_dit = True   # Anima is a DiT model
        self.is_sdxl = False
        self.is_sd3 = False
        self.is_sd2 = False
        self.is_sd1 = False
        self.is_flux = False
        self.is_te2 = False
        self.model_version = "anima"
        # Anima VAE uses its own normalization internally; no scale/shift needed here
        self.vae_scale_factor = 1.0
        self.vae_shift_factor = 0.0
        self.sdver = 5  # 5 = Anima
        print(f"Model type: {self.model_version}")

def import_json(name, preset = False):
    def find_files(file_name):
        for root, dirs, files in os.walk(jsonspath):
            if file_name in files:
                return os.path.join(root, file_name)
        return None
    if preset:
        filepath = os.path.join(presetspath, name + ".json")
    else:
        filepath = find_files(name if ".json" in name else name + ".json")

    output = []

    # Return [mode, model, vae] + all_configs + [dummy] on failure
    null_len = 3 + len(all_configs) + 1
    if filepath is None:
        return [gr.update()] * null_len
    with open(filepath, 'r', encoding='utf-8') as file:
        data = json.load(file)

    def setconfigs(data, output):
        for key, gtype, _, default, dtype, _ in all_configs:
            from scripts.traintrain import OPTIMIZERS
            if key in data:
                if key == "train_optimizer":
                    for optim in OPTIMIZERS:
                        if data[key].lower() == optim.lower():
                            data[key] = optim
                if gtype == "DD" or "learning rate" in key:
                    dtype = str
                try:
                    output.append(dtype(data[key]))
                except Exception:
                    output.append(default)
            else:
                output.append(default)

    setconfigs(data, output)

    head = [
        data.get("mode", "LoRA"),
        data.get("model", ""),
        data.get("vae", ""),
    ]

    # dummy checkbox value at end of train_settings_1
    return head + output + [False]

def get_optimizer(name: str, trainable_params, lr, optimizer_kwargs, network):
    name = name.lower()
    if name.startswith("dadapt"):
        import dadaptation
        if name == "dadaptadam":
            optim = dadaptation.DAdaptAdam
        elif name == "dadaptlion":
            optim = dadaptation.DAdaptLion               
        elif name == "DAdaptAdaGrad".lower():
            optim = dadaptation.DAdaptAdaGrad
        elif name == "DAdaptAdan".lower():
            optim = dadaptation.DAdaptAdan
        elif name == "DAdaptSGD".lower():
            optim = dadaptation.DAdaptSGD

    elif name.endswith("8bit"): 
        import bitsandbytes as bnb
        try:
            if name == "adam8bit":
                optim = bnb.optim.Adam8bit
            elif name == "adamw8bit":  
                optim = bnb.optim.AdamW8bit
            elif name == "SGDNesterov8bit".lower():
                optim = bnb.optim.SGD8bit
                if "momentum" not in optimizer_kwargs:
                    optimizer_kwargs["momentum"] = 0.9
                optimizer_kwargs["nesterov"] = True
            elif name == "Lion8bit".lower():
                optim = bnb.optim.Lion8bit
            elif name == "PagedAdamW8bit".lower():
                optim = bnb.optim.PagedAdamW8bit
            elif name == "PagedLion8bit".lower():
                optim  = bnb.optim.PagedLion8bit

        except AttributeError:
            raise AttributeError(
                f"No {name}. The version of bitsandbytes installed seems to be old. Please install newest. / {name}が見つかりません。インストールされているbitsandbytesのバージョンが古いようです。最新版をインストールしてください。"
            )

    elif name.lower() == "adafactor":
        import transformers
        optim = transformers.optimization.Adafactor

    elif name == "PagedAdamW".lower():
        import bitsandbytes as bnb
        optim = bnb.optim.PagedAdamW
    elif name == "PagedAdamW32bit".lower():
        import bitsandbytes as bnb
        optim = bnb.optim.PagedAdamW32bit

    elif name == "SGDNesterov".lower():
        if "momentum" not in optimizer_kwargs:
            optimizer_kwargs["momentum"] = 0.9
        optimizer_kwargs["nesterov"] = True
        optim = torch.optim.SGD

    elif name.endswith("schedulefree".lower()):
        import schedulefree as sf
        if name == "RAdamScheduleFree".lower():
            optim = sf.RAdamScheduleFree
        elif name == "AdamWScheduleFree".lower():
            optim = sf.AdamWScheduleFree
        elif name == "SGDScheduleFree".lower():
            optim = sf.SGDScheduleFree

    elif name in POs:
        import pytorch_optimizer as po    
        if name == "CAME".lower():
            optim = po.CAME        
        elif name == "Tiger".lower():
            optim = po.Tiger        
        elif name == "AdamMini".lower():
            optim = po.AdamMini   
        
    else:
        if name == "adam":
            optim = torch.optim.Adam
        elif name == "adamw":
            optim = torch.optim.AdamW  
        elif name == "lion":
            from lion_pytorch import Lion
            optim = Lion
        elif name == "prodigy":
            import prodigyopt
            optim = prodigyopt.Prodigy

    
    if name.startswith("DAdapt".lower()) or name == "Prodigy".lower():
    # check lr and lr_count, and logger.info warning
        actual_lr = lr
        lr_count = 1
        if type(trainable_params) == list and type(trainable_params[0]) == dict:
            lrs = set()
            actual_lr = trainable_params[0].get("lr", actual_lr)
            for group in trainable_params:
                lrs.add(group.get("lr", actual_lr))
            lr_count = len(lrs)

        if actual_lr <= 0.1:
            print(
                f"learning rate is too low. If using D-Adaptation or Prodigy, set learning rate around 1.0 / 学習率が低すぎるようです。D-AdaptationまたはProdigyの使用時は1.0前後の値を指定してください: lr={actual_lr}"
            )
            print("recommend option: lr=1.0 / 推奨は1.0です")
        if lr_count > 1:
            print(
                f"when multiple learning rates are specified with dadaptation (e.g. for Text Encoder and U-Net), only the first one will take effect / D-AdaptationまたはProdigyで複数の学習率を指定した場合（Text EncoderとU-Netなど）、最初の学習率のみが有効になります: lr={actual_lr}"
            )

    elif name == "Adafactor".lower():
        # 引数を確認して適宜補正する
        if "relative_step" not in optimizer_kwargs:
            optimizer_kwargs["relative_step"] = True  # default
        if not optimizer_kwargs["relative_step"] and optimizer_kwargs.get("warmup_init", False):
            print(
                f"set relative_step to True because warmup_init is True / warmup_initがTrueのためrelative_stepをTrueにします"
            )
            optimizer_kwargs["relative_step"] = True
 
        if optimizer_kwargs["relative_step"]:
            print(f"relative_step is true / relative_stepがtrueです")
            if lr != 0.0:
                print(f"learning rate is used as initial_lr / 指定したlearning rateはinitial_lrとして使用されます")
  

            # trainable_paramsがgroupだった時の処理：lrを削除する
            if type(trainable_params) == list and type(trainable_params[0]) == dict:
                has_group_lr = False
                for group in trainable_params:
                    p = group.pop("lr", None)
                    has_group_lr = has_group_lr or (p is not None)

            lr = None
        #TODO

        # else:
        #     if args.max_grad_norm != 0.0:
        #         logger.warning(
        #             f"because max_grad_norm is set, clip_grad_norm is enabled. consider set to 0 / max_grad_normが設定されているためclip_grad_normが有効になります。0に設定して無効にしたほうがいいかもしれません"
        #         )
        #     if args.lr_scheduler != "constant_with_warmup":
        #         logger.warning(f"constant_with_warmup will be good / スケジューラはconstant_with_warmupが良いかもしれません")
        #     if optimizer_kwargs.get("clip_threshold", 1.0) != 1.0:
        #         logger.warning(f"clip_threshold=1.0 will be good / clip_thresholdは1.0が良いかもしれません")


    return optim(network, lr = lr, **optimizer_kwargs) if name == "AdamMini".lower() else  optim(trainable_params, lr = lr, **optimizer_kwargs) 


def get_random_resolution_in_bucket(bucket_resolution: int = 512) -> tuple[int, int]:
    max_resolution = bucket_resolution
    min_resolution = bucket_resolution // 2

    step = 64

    min_step = min_resolution // step
    max_step = max_resolution // step

    height = torch.randint(min_step, max_step, (1,)).item() * step
    width = torch.randint(min_step, max_step, (1,)).item() * step

    return height, width


def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

def make_accelerator(t):
    from accelerate.state import AcceleratorState
    AcceleratorState._reset_state(reset_partial_state=True)

    accelerator = Accelerator(
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        mixed_precision=parse_precision(t.train_model_precision, mode = False)
    )

    return accelerator

def parse_precision(precision, mode = True):
    if mode:
        if precision == "fp32" or precision == "float32":
            return torch.float32
        elif precision == "fp16" or precision == "float16" or precision == "fp8":
            return torch.float16
        elif precision == "bf16" or precision == "bfloat16":
            return torch.bfloat16
    else:
        if precision == torch.float32 or precision in ("fp32", "float32"):
            return 'no'
        elif precision == torch.float16 or precision in ("fp16", "float16", "fp8"):
            return 'fp16'
        elif precision == torch.bfloat16 or precision in ("bf16", "bfloat16"):
            return 'bf16'

    raise ValueError(f"Invalid precision type: {precision}")
    
def load_lr_scheduler(t, optimizer):
    if t.train_optimizer == "adafactor":
        return AdafactorSchedule(optimizer)
    
    args = t.train_lr_scheduler_settings
    print(f"LR Scheduler args: {args}")
    
    # アニーリング系のスケジューラを追加
    if t.train_lr_scheduler == "cosine_annealing":
        return CosineAnnealingLR(
            optimizer,
            T_max=t.train_iterations,
            **args
        )
    elif t.train_lr_scheduler == "cosine_annealing_with_restarts":
        return CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.pop("T_0") if "T_0" in args else 10,
            **args
        )
    elif t.train_lr_scheduler == "exponential":
        return ExponentialLR(
            optimizer,
            gamma=args.pop("gamma") if "gamma" in args else 0.9,
            **args
        )
    elif t.train_lr_scheduler == "step":
        return StepLR(
            optimizer,
            step_size=args.pop("step_size") if "step_size" in args else 10,
            **args
        )
    elif t.train_lr_scheduler == "multi_step":
        return MultiStepLR(
            optimizer,
            milestones=args.pop("milestones") if "milestones" in args else [30, 60, 90],
            **args
        )
    elif t.train_lr_scheduler == "reduce_on_plateau":
        return ReduceLROnPlateau(
            optimizer,
            mode='min',
            **args
        )
    elif t.train_lr_scheduler == "cyclic":
        return CyclicLR(
            optimizer,
            base_lr=args.pop("base_lr") if "base_lr" in args else  1e-5,
            max_lr=args.pop("max_lr") if "max_lr" in args else  1e-3,
            mode='triangular',
            **args
        )
    elif t.train_lr_scheduler == "one_cycle":
        return OneCycleLR(
            optimizer,
            max_lr=args.pop("max_lr") if "max_lr" in args else  1e-3,
            total_steps=t.train_iterations,
            **args
        )
    
    return get_scheduler(
        name=t.train_lr_scheduler,
        optimizer=optimizer,
        step_rules="",
        num_warmup_steps=0,
        num_training_steps=t.train_iterations,
        num_cycles=1,
        power=t.train_lr_scheduler_power if t.train_lr_scheduler_power > 0 else 1.0,
        **t.train_lr_scheduler_settings
    )