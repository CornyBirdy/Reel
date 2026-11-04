@echo off
rem Starts Reel in this console window, so its messages are visible. Closing the window stops Reel.
set "REEL_CONSOLE=1"
call "%~dp0start.bat" %*
