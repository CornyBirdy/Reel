@echo off
rem Creates a "Reel" shortcut on your Desktop that starts Reel with its icon.
setlocal
set "ROOT=%~dp0"
if "%ROOT:~-1%"=="\" set "ROOT=%ROOT:~0,-1%"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$desktop = [Environment]::GetFolderPath('Desktop');" ^
  "$lnk = (New-Object -ComObject WScript.Shell).CreateShortcut((Join-Path $desktop 'Reel.lnk'));" ^
  "$lnk.TargetPath = '%ROOT%\start.bat';" ^
  "$lnk.WorkingDirectory = '%ROOT%';" ^
  "$lnk.IconLocation = '%ROOT%\reel\static\icons\reel.ico,0';" ^
  "$lnk.Description = 'Reel - local AI video generation';" ^
  "$lnk.Save();" ^
  "Write-Host ('Created ' + (Join-Path $desktop 'Reel.lnk'))"

if errorlevel 1 (
    echo Could not create the shortcut.
    pause
    exit /b 1
)
timeout /t 3 >nul
