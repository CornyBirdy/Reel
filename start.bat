@echo off
rem Reel - one-click launcher.
rem First run: creates .venv, installs the app dependencies, then starts Reel.
rem Every run after that goes straight to starting Reel.
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title Reel

set "PYTHONUTF8=1"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"
set "VENV_DIR=%~dp0.venv"
set "VENV_PY=%VENV_DIR%\Scripts\python.exe"
set "STAMP=%VENV_DIR%\requirements.installed"

if exist "%VENV_PY%" goto :have_venv

call :find_python
if not defined BASE_PY goto :no_python

echo [Reel] First run - creating the app environment with %BASE_PY% ...
%BASE_PY% -m venv "%VENV_DIR%"
if errorlevel 1 goto :fail

:have_venv
fc /b "%~dp0requirements.txt" "%STAMP%" >nul 2>&1
if not errorlevel 1 goto :launch

echo [Reel] Installing app dependencies ...
"%VENV_PY%" -m pip install -r "%~dp0requirements.txt"
if errorlevel 1 goto :fail
copy /y "%~dp0requirements.txt" "%STAMP%" >nul

:launch
"%VENV_PY%" -m reel
if errorlevel 1 goto :fail
exit /b 0

:no_python
echo.
echo [Reel] Python 3.10 - 3.13 was not found.
echo         Reel needs Python to run. Install Python 3.12 and run start.bat again:
echo           winget install -e --id Python.Python.3.12
echo         or download it from https://www.python.org/downloads/
echo         ^(tick "Add python.exe to PATH" in the installer^).
echo.
choice /c YN /n /m "Run the winget install now? [Y/N] "
if errorlevel 2 goto :fail
winget install -e --id Python.Python.3.12
echo.
echo [Reel] Close this window and run start.bat again so the new Python is picked up.
pause
exit /b 1

:fail
echo.
echo [Reel] Something went wrong - see the messages above.
pause
exit /b 1

:find_python
set "BASE_PY="
for %%V in (3.12 3.11 3.13 3.10) do (
    if not defined BASE_PY (
        py -%%V -c "import sys" >nul 2>&1 && set "BASE_PY=py -%%V"
    )
)
if not defined BASE_PY (
    python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)" >nul 2>&1 && set "BASE_PY=python"
)
exit /b 0
