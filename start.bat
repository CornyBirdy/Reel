@echo off
rem Reel - one-click launcher.
rem First run: creates .venv, installs the app dependencies, then starts Reel.
rem Every run after that goes straight to starting Reel.
setlocal EnableExtensions EnableDelayedExpansion
cd /d "%~dp0"
title Reel

rem Started again by the in-app updater: do not open a second window, the old one reconnects by itself.
if /i "%~1"=="restarted" set "REEL_SKIP_OPEN=1"

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
set "REEL_LAUNCHER=bat"
rem Everything that happens after Reel exits is inside ONE parenthesised block. cmd reads a block into
rem memory before running it, so Reel can safely replace this very file while it is running (updates).
(
    "%VENV_PY%" -m reel
    set "REEL_RC=!ERRORLEVEL!"
    if "!REEL_RC!"=="75" (
        echo [Reel] Restarting to finish the update ...
        start "" "%~f0" restarted
        exit
    )
    if not "!REEL_RC!"=="0" (
        echo.
        echo [Reel] Reel stopped unexpectedly ^(code !REEL_RC!^) - see the messages above.
        pause
        exit /b 1
    )
    exit /b 0
)

:no_python
echo.
echo [Reel] Python 3.10 - 3.13 was not found.
echo         Reel needs Python to run. Install Python 3.12 and run start.bat again:
echo           winget install -e --id Python.Python.3.12
echo         or download it from https://www.python.org/downloads/
echo         ^(tick "Add python.exe to PATH" in the installer^).
echo.
choice /c YN /n /m "Run the winget install now? [Y/N] "
if errorlevel 2 goto :declined
winget install -e --id Python.Python.3.12
echo.
echo [Reel] Close this window and run start.bat again so the new Python is picked up.
pause
exit /b 1

:declined
echo.
echo [Reel] Install Python, then run start.bat again.
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
