import os
import pandas as pd
import matplotlib.pyplot as plt
import gradio as gr
from PIL import Image, ImageChops
import random
import numpy as np
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
from trainer.lora import BLOCKID_ANIMA, generate_anima_preview_keys, _matches_module_filter

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
SCHEDULERS = ["cosine_annealing", "cosine_annealing_with_restarts", "linear", "cosine",
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
save_per_steps      = ["save_per_steps",     "TX", None, 0,    int,   ALL]
save_precision      = ["save_precision",     "DD", PRECISION_TYPES[:3], "fp16", str, ALL]
diff_revert_original_target = ["diff_revert_original_target","CH", None, False, bool, DIFF]
diff_use_diff_mask  = ["diff_use_diff_mask", "CH", None, False, bool, DIFF]
train_fixed_timsteps_in_batch = ["train_fixed_timsteps_in_batch","CH", None, False, bool, ALL]
train_repeat        = ["train_repeat",       "TX", None, 1,    int,   ALL]
gradient_accumulation_steps = ["gradient_accumulation_steps","TX", None, "1", str, ALL]
train_min_timesteps = ["train_min_timesteps","TX", None, 0,    int,   ALL]
train_max_timesteps = ["train_max_timesteps","TX", None, 1000, int,   ALL]
# Regex-based sub-layer filter.  Comma/newline separated; prefix with ! to exclude.
# Examples:  "!adaln_modulation"   →  skip adaln (recommended for Anima)
#            "attn, mlp"           →  attention + MLP only
#            "attn"                →  attention projections only
network_module_filter = ["network_module_filter(regex, !prefix=exclude)", "TX", None, "", str, ALL]

r_column1 = [network_rank, network_alpha, lora_data_directory, diff_target_name, lora_trigger_word]
r_column2 = [image_size, train_iterations, train_batch_size, train_learning_rate]
r_column3 = [train_optimizer, train_optimizer_settings, train_lr_scheduler, train_lr_scheduler_settings, save_lora_name, use_gradient_checkpointing]
row1      = [["network_blocks(BASE = TextEncoder)", "CB", BLOCKID_ANIMA, BLOCKID_ANIMA, list, ALL]]

o_column1 = [image_buckets_step, image_mirroring, image_use_filename_as_tag, image_disable_upscale,
             train_fixed_timsteps_in_batch]
o_column2 = [train_seed, train_loss_function, save_per_steps,
             diff_revert_original_target, diff_use_diff_mask]
o_column3 = [train_model_precision, train_lora_precision, save_precision,
             train_repeat, gradient_accumulation_steps]
o_column4 = [train_min_timesteps, train_max_timesteps, network_module_filter]

model_column = [qwen3_path, t5_tokenizer_path]

trainer.all_configs = model_column + r_column1 + r_column2 + r_column3 + row1 + o_column1 + o_column2 + o_column3 + o_column4

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
            if uitype == "CH":
                output.append(gr.Checkbox(label=name.replace("_"," "),value = value, elem_id="tt_" + name, visible = visible))
            if uitype == "CB":
                output.append(gr.CheckboxGroup(label=name.replace("_"," "),choices=choices, value = value, elem_id="tt_" + name, type="value", visible = visible))
            if uitype == "RD":
                output.append(gr.Radio(label=name.replace("_"," "),choices=[x + " " for x in choices] if pas > 0 else choices, value = value, elem_id="tt_" + name,visible = visible))
    return output

_PREVIEW_KEYS = generate_anima_preview_keys()   # generated once at import time


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
                    queue = gr.Button(value="Add to Queue", elem_classes=["compact_button"], variant='primary')
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
            with gr.Row():
                blocks_grs_1 = makeui(row1)

            gr.HTML(value="Option Parameters")
            with gr.Row():
                with gr.Column(variant="compact"):
                    col1_o1 = makeui(o_column1)
                with gr.Column(variant="compact"):
                    col2_o1 = makeui(o_column2)
                with gr.Column(variant="compact"):
                    col3_o1 = makeui(o_column3)
                with gr.Column(variant="compact"):
                    col4_o1 = makeui(o_column4)

            model_col_grs = [qwen3_gr, t5_gr]
            train_settings_1 = model_col_grs + col1_r1 + col2_r1 + col3_r1 + blocks_grs_1 + col1_o1 + col2_o1 + col3_o1 + col4_o1 + [dummy]

            # --- Layer browser: live regex preview --------------------------------
            # col4_o1 indices: 0=train_min_timesteps, 1=train_max_timesteps,
            #                  2=network_module_filter
            _filter_gr = col4_o1[2]
            with gr.Row():
                layer_preview = gr.HTML(
                    value=render_layer_preview(""),
                    label="Layer browser",
                )
            _filter_gr.change(render_layer_preview, inputs=[_filter_gr], outputs=[layer_preview])
            # ----------------------------------------------------------------------

            with gr.Group(visible=False) as g_diff:
                gr.HTML(value="Image Pairs (Original → Target)")
                with gr.Row():
                    with gr.Column():
                        orig_image = gr.Image(label="Original Image", interactive=True)
                    with gr.Column():
                        targ_image = gr.Image(label="Target Image", interactive=True)

        with gr.Tab("Queue"):
            with gr.Row():
                reload_queue = gr.Button(value="Reload Queue", elem_classes=["compact_button"], variant='primary')
                delete_queue = gr.Button(value="Delete Queue", elem_classes=["compact_button"], variant='primary')
                delete_name = gr.Textbox(label="Name of Queue to delete")
            with gr.Row():
                queue_list = gr.DataFrame(headers=["Name", "Mode", "Model", "VAE"] + [x[0] for x in trainer.all_configs])

        with gr.Tab("Plot"):
            with gr.Row():
                reload_plot = gr.Button(value="Reload Plot", elem_classes=["compact_button"], variant='primary')
                plot_file = gr.Textbox(label="Name of logfile, blank for last or current training")
            with gr.Row():
                plot = gr.Plot()

        with gr.Tab("Image"):
            gr.HTML(value="Rotate random angle and scaling")
            image_result = gr.Textbox(label="Message")
            with gr.Row():
                with gr.Column(variant="compact"):
                    angle_bg = gr.Button(value="From Directory", elem_classes=["compact_button"], variant='primary')
                with gr.Column(variant="compact"):
                    angle_bg_i = gr.Button(value="From File", elem_classes=["compact_button"], variant='primary')
                with gr.Column(variant="compact"):
                    fix_side = gr.Radio(label="fix side", value="none", choices=["none", "right", "left", "top", "bottom"])
            with gr.Row():
                with gr.Column(variant="compact"):
                    image_dir = gr.Textbox(label="Image directory")
                with gr.Column(variant="compact"):
                    output_name = gr.Textbox(label="Output name")
                with gr.Column(variant="compact"):
                    save_dir = gr.Textbox(label="Output directory")
            with gr.Row():
                num_of_images = gr.Slider(label="number of images", maximum=1000, minimum=0, step=1, value=5)
                max_tilting_angle = gr.Slider(label="max_tilting_angle", maximum=180, minimum=0, step=1, value=180)
                min_scale = gr.Slider(label="minimun downscale ratio", maximum=1, minimum=0, step=0.01, value=0.4)
            with gr.Row():
                change_angle = gr.Checkbox(label="change angle", value=False)
                change_scale = gr.Checkbox(label="change scale", value=False)

            input_image = gr.Image(label="Input Image", interactive=True, type="pil", image_mode="RGBA")

        dtrue = gr.Checkbox(value=True, visible=False)
        dfalse = gr.Checkbox(value=False, visible=False)

        in_images = [orig_image, targ_image]

        def savepreset_f(*args):
            train.train(*args)
            return gr.update(choices=load_preset(None))

        angle_bg.click(change_angle_bg, [dtrue, image_dir, save_dir, input_image, output_name, num_of_images, change_angle, max_tilting_angle, change_scale, min_scale, fix_side], [image_result])
        angle_bg_i.click(change_angle_bg, [dfalse, image_dir, save_dir, input_image, output_name, num_of_images, change_angle, max_tilting_angle, change_scale, min_scale, fix_side], [image_result])

        start.click(train.train, [dfalse, mode, model, vae, *train_settings_1, *in_images], [result])
        queue.click(train.queue, [dfalse, mode, model, vae, *train_settings_1, *in_images], [result])
        savepreset.click(savepreset_f, [dtrue, mode, model, vae, *train_settings_1, *in_images], [presets])
        refleshpreset.click(lambda: gr.update(choices=load_preset(None)), outputs=[presets])

        reload_queue.click(train.get_del_queue_list, outputs=queue_list)
        delete_queue.click(train.get_del_queue_list, inputs=[delete_name], outputs=queue_list)

        stop.click(train.stop_time, [dfalse], [result])
        stop_save.click(train.stop_time, [dtrue], [result])

        def change_the_mode(mode):
            mode_idx = MODES.index(mode)
            out = [gr.update(visible=x[5][mode_idx]) for x in trainer.all_configs]
            out.append(gr.update())  # dummy checkbox
            # g_diff (image pair picker) only for single-pair ADDifT
            out.append(gr.update(visible=(mode_idx == 1)))
            return out

        def change_the_block(type, select):
            blocks = BLOCKID_ANIMA
            return gr.update(choices=blocks, value=[x for x in select if x in blocks])

        def openfolder_f():
            os.startfile(jsonspath)

        loadjson.click(trainer.import_json, [sets_file], [mode, model, vae] + train_settings_1)
        loadpreset.click(load_preset, [presets], [mode, model, vae] + train_settings_1)
        mode.change(change_the_mode, [mode], [*train_settings_1, g_diff])
        openfolder.click(openfolder_f)

        reload_plot.click(plot_csv, [plot_file], [plot])

    return (ui, "TrainTrain", "TrainTrain"),

def plot_csv(csv_path, logspath="."):
    def get_csv(csv_path):
        csv_path = csv_path if ".csv" in csv_path else csv_path + ".csv"
        if csv_path:
            for root, dirs, files in os.walk(logspath):
                if csv_path in files:
                    return os.path.join(root, csv_path)

        # 指定されたファイルが見つからない場合、最新の CSV ファイルを探す
        latest_csv = None
        latest_time = 0

        for root, dirs, files in os.walk(logspath):
            for file in files:
                if file.endswith(".csv"):
                    file_path = os.path.join(root, file)
                    file_time = os.path.getmtime(file_path)

                    if file_time > latest_time:
                        latest_csv = file_path
                        latest_time = file_time

        return latest_csv

    csv_file = get_csv(csv_path)

    # **"Step" から始まる行を見つける**
    header_row = 0  # デフォルトは0行目
    with open(csv_file, "r") as f:
        for i, line in enumerate(f):
            if line.startswith("Step"):
                header_row = i
                break  # "Step" が見つかったら終了

    # **CSVを適切なヘッダー行から読み込む**
    df = pd.read_csv(csv_file, skiprows=header_row)

    # x 軸にするカラム名（通常 "Step"）
    x = df.columns[0]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # 主要な y 軸 (2 列目)
    color = 'tab:red'
    ax1.set_xlabel(x)
    ax1.set_ylabel(df.columns[1], color=color)
    ax1.plot(df[x], df[df.columns[1]], color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # 追加の y 軸 (3 列目以降)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Learning Rates', color=color)
    for column in df.columns[2:]:
        ax2.plot(df[x], df[column], label=column)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title("Training Result")
    fig.tight_layout()
    plt.legend()
    plt.grid(True)

    return plt.gcf()


# ここに必要な追加の関数を定義します。
def downscale_image(image, min_scale, fix_side=None):
    import random
    from PIL import Image

    scale = random.uniform(min_scale, 1)
    original_size = image.size
    new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
    downscaled_image = image.resize(new_size, Image.ANTIALIAS)
    new_image = Image.new("RGBA", original_size, (0, 0, 0, 0))

    # 配置位置を決定する
    if fix_side == "right":
        x_position = original_size[0] - new_size[0]
        y_position = random.randint(0, original_size[1] - new_size[1])
    elif fix_side == "top":
        x_position = random.randint(0, original_size[0] - new_size[0])
        y_position = 0
    elif fix_side == "left":
        x_position = 0
        y_position = random.randint(0, original_size[1] - new_size[1])
    elif fix_side == "bottom":
        x_position = random.randint(0, original_size[0] - new_size[0])
        y_position = original_size[1] - new_size[1]
    else:
        # fix_sideがNoneまたは無効な値の場合、ランダムな位置に配置
        x_position = random.randint(0, original_size[0] - new_size[0])
        y_position = random.randint(0, original_size[1] - new_size[1])

    new_image.paste(downscaled_image, (x_position, y_position))
    return new_image


MARGIN = 5

def marginer(bbox, image):
    return (
        max(bbox[0] - MARGIN, 0),  # 左
        max(bbox[1] - MARGIN, 0),  # 上
        min(bbox[2] + MARGIN, image.width),  # 右
        min(bbox[3] + MARGIN, image.height)  # 下
    )


def change_angle_bg(from_dir, image_dir, save_dir, input_image, output_name, num_of_images ,
                                change_angle, max_tilting_angle, change_scale, min_scale, fix_side):

    if from_dir:
        image_files = [file for file in os.listdir(image_dir) if file.endswith((".png", ".jpg", ".jpeg"))]
    else:
        image_files = [input_image]

    for file in image_files:
        if isinstance(file, str):
            modified_folder_path = os.path.join(image_dir, "modified")
            os.makedirs(modified_folder_path, exist_ok=True)

            path = os.path.join(image_dir, file)
            name, extention = os.path.splitext(file)
            with Image.open(path) as img:
                img = img.convert("RGBA")
        else:
            modified_folder_path = save_dir
            os.makedirs(modified_folder_path, exist_ok=True)

            img = file
            name = output_name
            extention = "png"

        for i in range(num_of_images):
            modified_img = img

            #画像を回転
            if change_angle:
                angle = random.uniform(-max_tilting_angle, max_tilting_angle)
                modified_img = modified_img.rotate(angle, expand=True)

            if change_scale:
                modified_img = downscale_image(modified_img, min_scale, fix_side)

            # 変更した画像を保存
            save_path = os.path.join(modified_folder_path, f"{name}_id_{i}.{extention}")
            modified_img.save(save_path)

    return f"Images saved in {modified_folder_path}"


def getjsonlist():
    if not os.path.isdir(jsonspath):
        return []
    json_files = [f for f in os.listdir(jsonspath) if f.endswith('.json')]
    json_files = [f.replace(".json", "") for f in json_files]
    return json_files

if _HAS_WEBUI and __package__ == "traintrain":
    script_callbacks.on_ui_tabs(on_ui_tabs)
elif __name__ == "__main__":
    ui = on_ui_tabs()[0][0]
    ui.launch(css=".gradio-textbox {margin-bottom: 0 !important;}")
