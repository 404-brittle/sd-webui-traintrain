@echo off
setlocal

:: ============================================================
:: extract_subspace_model.bat
::
:: Extracts a principal subspace directly from the Anima DiT
:: model's internal response to concept prompts — no pre-trained
:: LoRAs required. Eliminates the bootstrapping problem.
::
:: Tag-swap delta mode (recommended): supply a NEUTRAL prompt.
:: The script runs both the concept and neutral prompt through
:: the model with identical noise, then subtracts the neutral
:: gradient to isolate concept-specific weight directions.
::
:: Usage:
::   extract_subspace_model.bat
::       (uses defaults defined below)
::
::   Or pass args directly:
::   extract_subspace_model.bat --prompts tools/prompts/style_artists.txt --output subspaces/style.safetensors
::
:: Output:
::   A .safetensors subspace file, same format as extract_subspace.bat.
::   Set subspace_guard_path to this file in the training UI.
::
:: Template prompt files are in tools/prompts/:
::   style_artists.txt    — painting styles and artists
::   style_rendering.txt  — rendering media (oil, 3D, photo, etc.)
::   subject_person.txt   — person/character identity features
::   pose_action.txt      — poses, gestures, actions
::   neutral_style.txt    — neutral examples for style delta
::   neutral_subject.txt  — neutral examples for subject/pose delta
:: ============================================================

set VENV_DIR=.venv
set SD_SCRIPTS_PATH=X:\SD\sd-scripts

:: ---------------------------------------------------------------
:: MODEL PATHS — required, set these for your installation
:: ---------------------------------------------------------------

set DIT_PATH=X:\SD\sd-webui-forge-neo\models\Stable-diffusion\anima-preview2.safetensors
set QWEN3_PATH=X:\SD\AnimaLoraToolkit\models\text_encoders
set T5_PATH=X:\SD\AnimaLoraToolkit\models\t5_tokenizer

:: ---------------------------------------------------------------
:: CONCEPT PROMPTS
:: Choose a prompt file from tools/prompts/ or point to your own.
:: ---------------------------------------------------------------

:: style_artists.txt    — painting styles, art movements
:: style_rendering.txt  — rendering media (oil, 3D, photo)
:: subject_person.txt   — person identity and appearance
:: pose_action.txt      — poses, gestures, actions
set PROMPTS=tools\prompts\style_artists.txt

:: ---------------------------------------------------------------
:: NEUTRAL PROMPT (tag-swap delta, recommended)
:: Leave blank to use raw gradients without subtraction.
:: See tools/prompts/neutral_style.txt for suggestions.
:: ---------------------------------------------------------------
set NEUTRAL=1girl, portrait

:: ---------------------------------------------------------------
:: OUTPUT
:: ---------------------------------------------------------------
set OUTPUT=subspaces\style_model.safetensors

:: ---------------------------------------------------------------
:: EXTRACTION SETTINGS
:: ---------------------------------------------------------------

:: Principal components per layer. Use --plot_decay to inspect
:: the singular value decay curve and pick where it flattens.
set N_COMPONENTS=16

:: Forward passes per prompt. More = richer estimate, slower.
:: 4 is a good default. Raise to 8 for small prompt lists.
set SAMPLES_PER_PROMPT=8

:: Timestep range. Match to the training preset this guard will
:: be used with. Low range (50-450) emphasises texture/style.
:: High range (400-950) emphasises structure/pose.
set MIN_TIMESTEP=0
set MAX_TIMESTEP=600

:: Latent shape — must match the model's expected latent dimensions.
:: Default: 16 channels, 32x32 spatial (= 512px with 16x VAE).
set LATENT_CHANNELS=16
set LATENT_HEIGHT=32
set LATENT_WIDTH=32

:: Layer filter — same syntax as network_module_filter in the UI.
:: Default excludes adaln_modulation (recommended).
:: For pose subspace: "self_attn, !adaln_modulation"
:: For style subspace: "mlp, cross_attn, !adaln_modulation"
set MODULE_FILTER=mlp, cross_attn, !adaln_modulation

:: Model precision.
set PRECISION=bfloat16

:: Set to 1 to save a singular value decay plot alongside the output.
set PLOT_DECAY=1


:: ---------------------------------------------------------------
:: Pass any command-line arguments through unchanged
:: ---------------------------------------------------------------
if not "%~1"=="" goto :run_with_args

:: ---------------------------------------------------------------
:: Build optional flags
:: ---------------------------------------------------------------
set PLOT_FLAG=
if "%PLOT_DECAY%"=="1" set PLOT_FLAG=--plot_decay

set T5_FLAG=
if not "%T5_PATH%"=="" set T5_FLAG=--t5_tokenizer_path "%T5_PATH%"

set NEUTRAL_FLAG=
if not "%NEUTRAL%"=="" set NEUTRAL_FLAG=--neutral "%NEUTRAL%"

:: ---------------------------------------------------------------
:: Resolve Python
:: ---------------------------------------------------------------
if exist "%VENV_DIR%\Scripts\python.exe" (
    set PYTHON=%VENV_DIR%\Scripts\python.exe
) else (
    set PYTHON=python
)

:: ---------------------------------------------------------------
:: Print summary
:: ---------------------------------------------------------------
echo.
echo === Subspace Extraction from Model ===
echo   DiT             : %DIT_PATH%
echo   Qwen3           : %QWEN3_PATH%
echo   Prompts         : %PROMPTS%
echo   Neutral         : %NEUTRAL%
echo   Output          : %OUTPUT%
echo   Components      : %N_COMPONENTS%
echo   Samples/prompt  : %SAMPLES_PER_PROMPT%
echo   Timestep range  : [%MIN_TIMESTEP%, %MAX_TIMESTEP%]
echo   Latent shape    : %LATENT_CHANNELS%x%LATENT_HEIGHT%x%LATENT_WIDTH%
echo   Module filter   : %MODULE_FILTER%
echo   Precision       : %PRECISION%
echo   Plot decay      : %PLOT_DECAY%
echo.

set PYTHONPATH=%CD%
"%PYTHON%" tools\extract_subspace_from_model.py ^
    --dit_path          "%DIT_PATH%"            ^
    --qwen3_path        "%QWEN3_PATH%"           ^
    --prompts           "%PROMPTS%"              ^
    --output            "%OUTPUT%"               ^
    --n_components      %N_COMPONENTS%           ^
    --samples_per_prompt %SAMPLES_PER_PROMPT%    ^
    --min_timestep      %MIN_TIMESTEP%           ^
    --max_timestep      %MAX_TIMESTEP%           ^
    --latent_channels   %LATENT_CHANNELS%        ^
    --latent_height     %LATENT_HEIGHT%          ^
    --latent_width      %LATENT_WIDTH%           ^
    --module_filter     "%MODULE_FILTER%"        ^
    --precision         %PRECISION%              ^
    %T5_FLAG% %NEUTRAL_FLAG% %PLOT_FLAG%

goto :done

:: ---------------------------------------------------------------
:: Pass-through: relay explicit args directly to the script
:: ---------------------------------------------------------------
:run_with_args
if exist "%VENV_DIR%\Scripts\python.exe" (
    set PYTHON=%VENV_DIR%\Scripts\python.exe
) else (
    set PYTHON=python
)
set PYTHONPATH=%CD%
"%PYTHON%" tools\extract_subspace_from_model.py %*

:done
echo.
if %ERRORLEVEL% neq 0 (
    echo FAILED with error code %ERRORLEVEL%.
) else (
    echo Done. Set subspace_guard_path in the training UI to:
    echo   %OUTPUT%
)
echo.
pause
