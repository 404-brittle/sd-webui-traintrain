@echo off
setlocal

set VENV_DIR=.venv
set SD_SCRIPTS_PATH=X:\SD\sd-scripts

:: Path to the Anima sd-scripts repo containing the 'library' folder.
:: Override by setting SD_SCRIPTS_PATH before running, e.g.:
::   set SD_SCRIPTS_PATH=X:\SD\sd-scripts && launch.bat
if "%SD_SCRIPTS_PATH%"=="" (
    echo WARNING: SD_SCRIPTS_PATH is not set. Set it to the sd-scripts repo path, e.g.:
    echo   set SD_SCRIPTS_PATH=X:\SD\sd-scripts
    echo.
)

:: Optional: set port (default 7861)
if "%TRAINTRAIN_PORT%"=="" set TRAINTRAIN_PORT=7861

where uv >nul 2>&1
if %ERRORLEVEL% == 0 (
    echo Using uv...
    if not exist "%VENV_DIR%" uv venv "%VENV_DIR%"
    uv pip install --python "%VENV_DIR%\Scripts\python.exe" -r requirements.txt
    set PYTHONPATH=%CD%
    echo.
    echo Starting TrainTrain frontend server on http://localhost:%TRAINTRAIN_PORT%
    echo Open your browser to that address.
    echo.
    "%VENV_DIR%\Scripts\python.exe" frontend\server.py %*
) else (
    echo Using pip...
    if not exist "%VENV_DIR%" python -m venv "%VENV_DIR%"
    "%VENV_DIR%\Scripts\pip.exe" install -r requirements.txt
    set PYTHONPATH=%CD%
    echo.
    echo Starting TrainTrain frontend server on http://localhost:%TRAINTRAIN_PORT%
    echo Open your browser to that address.
    echo.
    "%VENV_DIR%\Scripts\python.exe" frontend\server.py %*
)
