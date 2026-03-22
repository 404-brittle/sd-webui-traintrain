@echo off
setlocal

:: ============================================================
:: extract_subspace.bat
::
:: Extracts a principal subspace from a folder of reference
:: LoRA .safetensors files, saving a subspace file that the
:: SubspaceGuard uses during training to prevent feature bleed.
::
:: Usage:
::   extract_subspace.bat
::       (uses defaults defined below, prompts if blank)
::
::   Or override on the command line:
::   extract_subspace.bat --lora_dir D:\loras\style --output subspaces\style.safetensors --n_components 16
::
:: Output:
::   A .safetensors subspace file.
::   Set subspace_guard_path to this file in the training UI.
:: ============================================================

set VENV_DIR=.venv
set SD_SCRIPTS_PATH=X:\SD\sd-scripts

:: ---------------------------------------------------------------
:: DEFAULT SETTINGS — edit these to match your workflow
:: ---------------------------------------------------------------

:: Folder containing your reference LoRAs (all .safetensors in
:: this folder will be used). Leave blank to be prompted.
set LORA_DIR=X:\SD\sd-webui-traintrain\output\_style

:: Output subspace file path.
set OUTPUT=subspaces\subspace.safetensors

:: Number of principal components to extract per layer.
:: 8–16 is a good starting point. Use --plot_decay to inspect
:: the singular value decay curve and tune this value.
set N_COMPONENTS=16

:: Minimum number of reference LoRAs a layer must appear in to
:: be included. Raise if you have many LoRAs and want cleaner stats.
set MIN_REFERENCES=2

:: Set to 1 to save a singular value decay plot alongside the output.
:: Requires matplotlib. Useful for choosing N_COMPONENTS.
set PLOT_DECAY=1

:: Device for SVD computation. Use "cuda" for large models / many LoRAs.
set DEVICE=cuda

:: ---------------------------------------------------------------
:: If LORA_DIR is empty, prompt the user
:: ---------------------------------------------------------------
if "%LORA_DIR%"=="" (
    echo.
    echo No LORA_DIR set. Enter the path to a folder of reference LoRA files.
    echo Example: D:\loras\style_reference
    echo.
    set /p LORA_DIR="LoRA folder: "
)

if "%LORA_DIR%"=="" (
    echo ERROR: No LoRA folder specified. Exiting.
    pause
    exit /b 1
)

if not exist "%LORA_DIR%" (
    echo ERROR: Folder not found: %LORA_DIR%
    pause
    exit /b 1
)

:: ---------------------------------------------------------------
:: Build plot_decay flag
:: ---------------------------------------------------------------
set PLOT_FLAG=
if "%PLOT_DECAY%"=="1" set PLOT_FLAG=--plot_decay

:: ---------------------------------------------------------------
:: Pass any extra command-line arguments through unchanged,
:: so the bat can also be called with explicit --lora_dir etc.
:: ---------------------------------------------------------------
if not "%~1"=="" goto :run_with_args

:: ---------------------------------------------------------------
:: Run with defaults
:: ---------------------------------------------------------------
echo.
echo === Subspace Extraction ===
echo   LoRA folder   : %LORA_DIR%
echo   Output        : %OUTPUT%
echo   Components    : %N_COMPONENTS%
echo   Min references: %MIN_REFERENCES%
echo   Device        : %DEVICE%
echo   Plot decay    : %PLOT_DECAY%
echo.

:run_defaults
if exist "%VENV_DIR%\Scripts\python.exe" (
    set PYTHON=%VENV_DIR%\Scripts\python.exe
) else (
    set PYTHON=python
)

set PYTHONPATH=%CD%
"%PYTHON%" tools\extract_subspace.py ^
    --lora_dir      "%LORA_DIR%"     ^
    --output        "%OUTPUT%"       ^
    --n_components  %N_COMPONENTS%   ^
    --min_references %MIN_REFERENCES% ^
    --device        %DEVICE%         ^
    %PLOT_FLAG%

goto :done

:: ---------------------------------------------------------------
:: Pass-through: called with explicit args, just relay them
:: ---------------------------------------------------------------
:run_with_args
if exist "%VENV_DIR%\Scripts\python.exe" (
    set PYTHON=%VENV_DIR%\Scripts\python.exe
) else (
    set PYTHON=python
)

set PYTHONPATH=%CD%
"%PYTHON%" tools\extract_subspace.py %*

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
