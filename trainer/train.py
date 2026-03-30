import os
import csv
import random
import sys
import time
import numpy
import gc
import json
from PIL import Image
import traceback
import torch
import torch.nn.functional as F
from torch.nn import ModuleList
from tqdm import tqdm
from trainer.lora import LoRANetwork, LycorisNetwork
from trainer import trainer, dataset
from trainer.anima_support import (
    AnimaFlowScheduler,
    AnimaTextModel,
    anima_forward,
    expand_cond,
    move_cond_to_device,
)
from pprint import pprint
from accelerate.utils import set_seed

# Add sd-scripts root to path for library imports
_TRAINTRAIN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SD_SCRIPTS_ROOT = os.environ.get("SD_SCRIPTS_PATH") or os.path.dirname(_TRAINTRAIN_DIR)
if _SD_SCRIPTS_ROOT not in sys.path:
    sys.path.insert(0, _SD_SCRIPTS_ROOT)

try:
    from modules import shared
    _HAS_WEBUI = True
except ImportError:
    _HAS_WEBUI = False

MAX_DENOISING_STEPS = 1000
ML = "LoRA"

jsonspath = trainer.jsonspath
logspath = trainer.logspath
presetspath = trainer.presetspath

stoptimer = 0

CUDA = torch.device("cuda:0")

queue_list = []
current_name = None


def get_name_index(wanted):
    for i, name in enumerate(trainer.all_configs):
        if name[0] == wanted:
            return i


def queue(*args):
    global queue_list
    name_index = get_name_index("save_lora_name") + 4
    dup = args[name_index] == current_name
    for queue in queue_list:
        if queue[name_index] == args[name_index]:
            dup = True
    if dup:
        return "Duplicated LoRA name! Could not add to queue."
    queue_list.append(args)
    return "Added to Queue"


def get_del_queue_list(del_name=None):
    global queue_list
    name_index = get_name_index("save_lora_name")
    out = []
    del_index = None
    for i, q in enumerate(queue_list):
        data = [*q[1:-2]]
        name = data[name_index + 3]
        data = [name] + data
        if del_name and name == del_name:
            del_index = i
        else:
            out.append(data)
    if del_index:
        del queue_list[del_index]
    return out


def setcurrentname(args):
    name_index = get_name_index("save_lora_name") + 4
    global current_name
    current_name = args[name_index]


def train(*args):
    if not args[0]:
        setcurrentname(args)
    result = train_main(*args)
    while len(queue_list) > 0:
        settings = queue_list.pop(0)
        result += "\n" + train_main(*settings)
    return result


def train_main(jsononly, mode, modelname, vaename, *args):
    t = trainer.Trainer(jsononly, modelname, vaename, mode, args)

    if jsononly:
        return "Preset saved"

    if t.isfile:
        return "File exist!"

    if modelname == "":
        return "No Model Selected."

    print(" Start Training!")

    # ------------------------------------------------------------------ #
    # Load Anima-specific libraries                                        #
    # ------------------------------------------------------------------ #
    from library import anima_utils, qwen_image_autoencoder_kl

    t.sd_typer()

    # ------------------------------------------------------------------ #
    # Load VAE                                                             #
    # ------------------------------------------------------------------ #
    print(f"Loading VAE from {vaename}")
    vae = qwen_image_autoencoder_kl.load_vae(vaename, device="cpu", disable_mmap=True)
    vae = vae.to(CUDA, dtype=t.train_model_precision)
    vae.requires_grad_(False)
    vae.eval()
    t.vae = vae

    # ------------------------------------------------------------------ #
    # Load Qwen3 text encoder + tokenizers                                 #
    # ------------------------------------------------------------------ #
    qwen3_path = getattr(t, "qwen3_path", "")
    print(f"Loading Qwen3 text encoder from {qwen3_path}")
    qwen3_encoder, _ = anima_utils.load_qwen3_text_encoder(
        qwen3_path, dtype=t.train_model_precision, device="cpu"
    )
    qwen3_encoder = qwen3_encoder.to(CUDA)
    qwen3_encoder.requires_grad_(False)
    qwen3_encoder.eval()

    t5_tokenizer_path = getattr(t, "t5_tokenizer_path", None) or None

    t.text_model = AnimaTextModel(
        qwen3_encoder,
        qwen3_path,
        t5_tokenizer_path,
        device=CUDA,
        dtype=t.train_model_precision,
    )

    # ------------------------------------------------------------------ #
    # Load Anima DiT                                                       #
    # ------------------------------------------------------------------ #
    attn_mode = "torch"
    print(f"Loading Anima DiT from {modelname}")
    dit = anima_utils.load_anima_model(
        device=CUDA,
        dit_path=modelname,
        attn_mode=attn_mode,
        split_attn=False,
        loading_device=CUDA,
        dit_weight_dtype=t.train_model_precision,
    )
    dit.requires_grad_(False)
    dit.eval()
    t.unet = dit  # raw DiT — LoRA modules patch this directly

    if t.use_gradient_checkpointing:
        dit.train()
        # Anima DiT gradient checkpointing (if supported)
        if hasattr(dit, "enable_gradient_checkpointing"):
            dit.enable_gradient_checkpointing()
        t.text_model.train()
        t.text_model.gradient_checkpointing_enable()

    # ------------------------------------------------------------------ #
    # Encode prompts                                                       #
    # ------------------------------------------------------------------ #
    trigger = getattr(t, 'lora_trigger_word', '') or ''
    if t.mode in ("ADDifT", "Multi-ADDifT"):
        # orig = source/before condition, targ = destination/after condition.
        # For ADDifT single-pair, diff_target_name is the "before" text prompt.
        # For Multi-ADDifT, per-pair conditioning comes from batch captions;
        # these serve as fallbacks.
        before_text = getattr(t, 'diff_target_name', '') or ''
        t.orig_cond, _ = text2cond(t, before_text if before_text else trigger)
        t.targ_cond, _ = text2cond(t, trigger)
    else:
        t.orig_cond, _ = text2cond(t, trigger)
        t.targ_cond = t.orig_cond
    t.un_cond, _ = text2cond(t, '')

    # ------------------------------------------------------------------ #
    # Noise scheduler (flow matching)                                      #
    # ------------------------------------------------------------------ #
    t.noise_scheduler = AnimaFlowScheduler()

    # ------------------------------------------------------------------ #
    # Accelerator                                                          #
    # ------------------------------------------------------------------ #
    t.a = trainer.make_accelerator(t)
    t.unet = t.a.prepare(t.unet)

    if 0 > t.train_seed:
        t.train_seed = random.randint(0, 2**32)
    set_seed(t.train_seed)
    makesavelist(t)

    # Store helpers on t for dataset
    t.text2cond = text2cond
    t.image2latent = image2latent

    try:
        if t.mode == ML:
            result = train_lora(t)
        elif t.mode == "ADDifT" or t.mode == "Multi-ADDifT":
            result = train_diff2(t)
        else:
            result = "Test mode"

        print("Done.")
    except Exception as e:
        print(traceback.format_exc())
        result = f"Error: {e}"

    del t
    flush()

    return result


# --------------------------------------------------------------------------- #
# Train modes                                                                  #
# --------------------------------------------------------------------------- #

def train_lora(t):
    global stoptimer
    stoptimer = 0

    t.a.print("Preparing image latents and text-conditional...")
    dataloaders = dataset.make_dataloaders(t)
    t.dataloader = dataset.ContinualRandomDataLoader(dataloaders)
    t.dataloader = t.a.prepare(t.dataloader)

    t.a.print("Train Anima LoRA Start")

    network, optimizer, lr_scheduler = create_network(t)

    if not t.dataloader.data:
        return "No data!"

    loss_ema = None
    loss_velocity = None

    del t.vae
    if "BASE" not in t.network_blocks:
        del t.text_model
    flush()

    pbar = tqdm(range(t.train_iterations))
    while t.train_iterations >= pbar.n:
        for batch in t.dataloader:
            for i in range(t.train_repeat):
                latents = batch["latent"].to(CUDA, dtype=t.train_lora_precision)
                conds1 = batch["cond1"] if "cond1" in batch else None

                noise = torch.randn_like(latents)
                batch_size = latents.shape[0]

                ts_lo = max(0, t.train_min_timesteps)
                ts_hi = max(ts_lo + 1, min(1000, t.train_max_timesteps))
                flow_shift = getattr(t, 'train_flow_shift', 3.0)
                n_ts = 1 if t.train_fixed_timsteps_in_batch else batch_size
                timesteps = _sample_timesteps(ts_lo, ts_hi, n_ts, flow_shift, CUDA)
                timesteps = torch.cat([timesteps.long()] * (batch_size if t.train_fixed_timsteps_in_batch else 1))

                noisy_latents = t.noise_scheduler.add_noise(latents, noise, timesteps)

                # Resolve conditioning
                if conds1 is None:
                    conds1 = expand_cond(t.orig_cond, batch_size)
                elif isinstance(conds1, str) or (isinstance(conds1, list) and isinstance(conds1[0], str)):
                    conds1, _ = t.text_model.encode_text(conds1 if isinstance(conds1, list) else [conds1])
                elif isinstance(conds1, (tuple, list)):
                    conds1 = move_cond_to_device(conds1, CUDA, t.train_lora_precision)

                with network, t.a.autocast():
                    model_pred = anima_forward(t, noisy_latents, timesteps, conds1)

                # Flow matching loss target: velocity = noise - latents
                velocity_target = (noise - latents).to(torch.float32)
                loss, loss_ema, loss_velocity = process_loss(
                    t, model_pred, velocity_target, timesteps, loss_ema, loss_velocity
                )

                c_lrs = [f"{x:.2e}" for x in lr_scheduler.get_last_lr()]
                pbar.set_description(
                    f"Loss EMA * 1000: {loss_ema * 1000:.4f}, Current LR: " + ", ".join(c_lrs) +
                    f", Epoch: {t.dataloader.epoch}"
                )
                pbar.update(1)

                if t.logging_save_csv:
                    savecsv(t, pbar.n, loss_ema,
                            [x.cpu().item() if isinstance(x, torch.Tensor) else x for x in lr_scheduler.get_last_lr()],
                            t.csvpath)

                t.a.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                del model_pred
                flush()

                result = finisher(network, t, pbar.n)
                if result is not None:
                    return result

            if pbar.n >= t.train_iterations:
                break

    return savecount(network, t, 0)


def train_diff2(t):
    global stoptimer
    stoptimer = 0

    if t.mode == "ADDifT":
        t.orig_latent = image2latent(t, t.images[0]).to(t.train_model_precision)
        t.targ_latent = image2latent(t, t.images[1]).to(t.train_model_precision)
        data = dataset.LatentsConds(t, [([t.orig_latent, None, t.orig_cond, None],
                                         [t.targ_latent, None, t.targ_cond, None])])
        dataloaders = [dataset.DataLoader(data, batch_size=t.train_batch_size, shuffle=True)]
    else:
        t.a.print("Preparing image latents and text-conditional...")
        dataloaders = dataset.make_dataloaders(t)

    t.dataloader = dataset.ContinualRandomDataLoader(dataloaders)
    t.dataloader = t.a.prepare(t.dataloader)

    t.a.print("Train Anima Multi-ADDifT Start")

    if not t.dataloader.data:
        return "No data!"

    del t.vae
    if "BASE" not in t.network_blocks:
        del t.text_model
    flush()

    network, optimizer, lr_scheduler = create_network(t)

    loss_ema = None
    noise = None
    loss_velocity = None

    ts_range_lo = max(0, t.train_min_timesteps)
    ts_range_hi = max(ts_range_lo + 1, min(1000, t.train_max_timesteps))
    ts_span = ts_range_hi - ts_range_lo
    num_bands = max(1, ts_span // 100)

    time_min = ts_range_lo
    time_max = ts_range_hi

    pbar = tqdm(range(t.train_iterations))
    epoch = 0
    while t.train_iterations >= pbar.n + 1:
        for batch in t.dataloader:
            orig_latent = batch["orig_latent"]
            targ_latent = batch["targ_latent"]

            batch_size = orig_latent.shape[0]

            # For Multi-ADDifT the dataset provides per-pair conditioning from
            # each image's caption; use it when present so the model sees the
            # actual semantic difference, not identical trigger-word embeddings.
            if "orig_cond1" in batch and batch["orig_cond1"] is not None:
                orig_conds1 = batch["orig_cond1"]
                targ_conds1 = batch.get("targ_cond1", orig_conds1)
            else:
                orig_conds1 = expand_cond(t.orig_cond, batch_size)
                targ_conds1 = expand_cond(t.targ_cond, batch_size)

            orig_conds1 = move_cond_to_device(orig_conds1, CUDA, t.train_model_precision)
            targ_conds1 = move_cond_to_device(targ_conds1, CUDA, t.train_model_precision)

            optimizer.zero_grad()
            noise = torch.randn_like(orig_latent)

            turn = pbar.n % 2 == 0

            if turn:
                band_span = ts_span // num_bands
                index = (pbar.n // 2) % num_bands
                time_min = ts_range_lo + band_span * index
                time_max = time_min + band_span
                time_max = max(time_min + 1, min(time_max, ts_range_hi))

            flow_shift = getattr(t, 'train_flow_shift', 3.0)
            n_ts = 1 if t.train_fixed_timsteps_in_batch else batch_size
            band_lo = int(min(time_min, ts_range_hi - 1))
            band_hi = int(max(time_max, ts_range_lo + 1))
            timesteps = _sample_timesteps(band_lo, band_hi, n_ts, flow_shift, CUDA)
            timesteps = torch.cat([timesteps.long()] * (batch_size if t.train_fixed_timsteps_in_batch else 1))

            orig_noisy_latents = t.noise_scheduler.add_noise(
                orig_latent if turn else targ_latent, noise, timesteps
            )
            targ_noisy_latents = t.noise_scheduler.add_noise(
                targ_latent if turn else orig_latent, noise, timesteps
            )

            orig_noisy_latents = orig_noisy_latents.to(CUDA, dtype=t.train_model_precision)
            targ_noisy_latents = targ_noisy_latents.to(CUDA, dtype=t.train_model_precision)

            # Baseline: base model prediction (no LoRA) on source latent.
            # multiplier must be 0 here — LoRANetwork starts at 1, so set explicitly.
            network.set_multiplier(0)
            with torch.no_grad(), t.a.autocast():
                orig_noise_pred = anima_forward(t, orig_noisy_latents, timesteps, orig_conds1)

            # LoRA-modified prediction on target latent, alternating direction each step.
            network.set_multiplier(0.25 if turn else -0.25)
            with t.a.autocast():
                targ_noise_pred = anima_forward(t, targ_noisy_latents, timesteps, targ_conds1)

            network.set_multiplier(0)

            if t.diff_use_diff_mask and "mask" in batch:
                mask = F.interpolate(batch["mask"].to(CUDA).unsqueeze(1).float(), size=targ_noise_pred.shape[2:], mode='nearest')
                targ_noise_pred = targ_noise_pred * mask
                orig_noise_pred = orig_noise_pred * mask

            loss, loss_ema, loss_velocity = process_loss(
                t, targ_noise_pred, orig_noise_pred, timesteps, loss_ema, loss_velocity
            )

            c_lrs = [f"{x:.2e}" for x in lr_scheduler.get_last_lr()]
            pbar.set_description(
                f"Loss EMA * 1000: {loss_ema * 1000:.4f}, Loss Velocity: {loss_velocity * 1000:.4f}, "
                f"Current LR: " + ", ".join(c_lrs) + f", Epoch: {epoch}"
            )
            pbar.update(1)

            if t.logging_save_csv:
                savecsv(t, pbar.n, loss_ema,
                        [x.cpu().item() if isinstance(x, torch.Tensor) else x for x in lr_scheduler.get_last_lr()],
                        t.csvpath)

            t.a.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            flush()

            result = finisher(network, t, pbar.n)
            if result is not None:
                del optimizer, lr_scheduler
                return result

        epoch += 1

    return savecount(network, t, 0)


# --------------------------------------------------------------------------- #
# Network / optimizer / scheduler helpers                                      #
# --------------------------------------------------------------------------- #

def flush():
    torch.cuda.empty_cache()
    gc.collect()


def _sample_timesteps(ts_lo: int, ts_hi: int, n: int, flow_shift: float, device) -> torch.Tensor:
    """Sample n integer timesteps in [ts_lo, ts_hi) with optional discrete flow shift.

    flow_shift == 1.0  →  uniform (equivalent to torch.randint)
    flow_shift  > 1.0  →  biased toward high-noise (high-t) end of the range,
                          matching the sd-scripts discrete_flow_shift convention:
                          sigma_shifted = (u * shift) / (1 + (shift - 1) * u)
    """
    if flow_shift == 1.0:
        return torch.randint(ts_lo, ts_hi, (n,), device=device)
    u = torch.rand(n, device=device)
    sigma = (u * flow_shift) / (1.0 + (flow_shift - 1.0) * u)
    ts = (sigma * (ts_hi - ts_lo) + ts_lo).long().clamp(ts_lo, ts_hi - 1)
    return ts


def create_network(t):
    network = load_network(t)
    optimizer = trainer.get_optimizer(
        t.train_optimizer,
        network.prepare_optimizer_params(),
        t.train_learning_rate,
        t.train_optimizer_settings,
        network,
    )

    t.is_schedulefree = t.train_optimizer.endswith("schedulefree".lower())

    if t.is_schedulefree:
        optimizer.train()
    else:
        lr_scheduler = trainer.load_lr_scheduler(t, optimizer)

    print(f"Optimizer : {type(optimizer).__name__}")
    print(f"Optimizer Settings : {t.train_optimizer_settings}")

    network, optimizer, lr_scheduler = t.a.prepare(
        network, optimizer, None if t.is_schedulefree else lr_scheduler
    )

    return network, optimizer, DummyScheduler(optimizer) if t.is_schedulefree else lr_scheduler


class DummyScheduler:
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def get_last_lr(self):
        return [p["scheduled_lr"] for p in self.optimizer.param_groups]

    def step(self):
        pass


def load_network(t):
    # Anima DiT uses standard linear LoRA (lierla) — no convolutions, no loha
    return LoRANetwork(t).to(CUDA, dtype=t.train_lora_precision)


def stop_time(save):
    global stoptimer
    stoptimer = 2 if save else 1


def finisher(network, t, i, copy=False):
    if t.save_list and i >= t.save_list[0]:
        savecount(network, t, t.save_list.pop(0), copy)

    if stoptimer > 0:
        if stoptimer > 1:
            result = ". " + savecount(network, t, i, copy)
        else:
            result = ""
        return "Stopped" + result


def savecount(network, t, i, copy=False):
    if t.metadata == {}:
        metadator(t)
    if copy and False:  # diff_save_1st_pass removed (DDPM 2-pass concept)
        return "Not save copy"
    add = "_copy" if copy else ""
    add = f"{add}_{i}steps" if i > 0 else add
    filename = os.path.join(t.save_dir, f"{t.save_lora_name}{add}.safetensors")
    print(f" Saving to {filename}")
    metaname = f"{t.save_lora_name}{add}"
    filename = network.save_weights(filename, t, metaname)
    return f"Successfully created to {filename}"


def makesavelist(t):
    if t.save_per_steps > 0:
        t.save_list = [x * t.save_per_steps for x in range(1, t.train_iterations // t.save_per_steps + 1)]
        if t.train_iterations in t.save_list:
            t.save_list.remove(t.train_iterations)
    else:
        t.save_list = []


def process_loss(t, original, target, timesteps, loss_ema, loss_velocity, copy=False):
    if t.train_loss_function == "MSE":
        loss = torch.nn.functional.mse_loss(original.float(), target.float(), reduction="none")
    elif t.train_loss_function == "L1":
        loss = torch.nn.functional.l1_loss(original.float(), target.float(), reduction="none")
    elif t.train_loss_function == "Smooth-L1":
        loss = torch.nn.functional.smooth_l1_loss(original.float(), target.float(), reduction="none")
    else:
        loss = torch.nn.functional.mse_loss(original.float(), target.float(), reduction="none")

    loss = loss.mean([1, 2, 3])
    loss = loss.mean()

    if loss_ema is None:
        loss_ema = loss.item()
        loss_velocity = 0
    else:
        loss_velocity = loss_velocity * 0.9 + (loss_ema - (loss_ema * 0.9 + loss.item() * 0.1)) * 0.1
        loss_ema = loss_ema * 0.9 + loss.item() * 0.1

    return loss, loss_ema, loss_velocity


# --------------------------------------------------------------------------- #
# Encode helpers                                                               #
# --------------------------------------------------------------------------- #

def image2latent(t, image):
    """Encode an image (PIL or path) to Anima latents using the Anima VAE."""
    if isinstance(image, str):
        with Image.open(image) as img:
            image = img.convert("RGB")
    elif hasattr(image, "convert"):
        image = image.convert("RGB")

    image_np = numpy.array(image).astype(numpy.float32) / 255.0
    image_tensor = torch.from_numpy(image_np).permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
    image_tensor = image_tensor * 2.0 - 1.0  # [0, 1] -> [-1, 1] (Anima VAE expects [-1, 1])
    image_tensor = image_tensor.to(CUDA, dtype=t.train_model_precision)

    with torch.no_grad():
        latent = t.vae.encode_pixels_to_latents(image_tensor)  # [1, C, H, W]

    return latent


def text2cond(t, prompt):
    """Encode a text prompt into Anima conditioning tensors."""
    cond, _ = t.text_model.encode_text(prompt if isinstance(prompt, list) else [prompt])
    return cond, None


# --------------------------------------------------------------------------- #
# Debug / logging                                                              #
# --------------------------------------------------------------------------- #

CSVHEADS = ["network_rank", "network_alpha", "train_learning_rate", "train_iterations",
            "train_lr_scheduler", "model_version", "train_optimizer", "save_lora_name"]


def savecsv(t, step, loss, lr, csvpath, copy=False):
    header = []
    for key in CSVHEADS:
        header.append([key, getattr(t, key, "")])
    header.append(["Step", "Loss"] + ["Learning Rate " + str(i + 1) for i in range(len(lr))])

    if copy:
        csvpath = csvpath.replace(".csv", "_copy.csv")

    directory = os.path.dirname(csvpath)
    if not os.path.exists(directory):
        os.makedirs(directory)

    file_exists = os.path.isfile(csvpath)
    with open(csvpath, "a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if not file_exists:
            for head in header:
                writer.writerow(head)
        writer.writerow([step, loss] + lr)


def metadator(t):
    t.metadata = {
        "ss_session_id": random.randint(0, 2**32),
        "ss_training_started_at": time.time(),
        "ss_output_name": t.save_lora_name,
        "ss_learning_rate": t.train_learning_rate,
        "ss_max_train_steps": t.train_iterations,
        "ss_lr_warmup_steps": 0,
        "ss_lr_scheduler": t.train_lr_scheduler,
        "ss_network_module": "network.lora",
        "ss_network_dim": t.network_rank,
        "ss_network_alpha": t.network_alpha,
        "ss_mixed_precision": t.train_lora_precision,
        "ss_lr_step_rules": "",
        "ss_lr_scheduler_num_cycles": 1,
        "ss_lr_scheduler_power": t.train_lr_scheduler_power,
        "ss_v2": False,
        "ss_base_model_version": t.model_version,
        "ss_seed": t.train_seed,
        "ss_optimizer": t.train_optimizer,
        "ss_min_snr_gamma": 0,
        "ss_tag_frequency": json.dumps({1: t.count_dict}),
    }
