@echo off
rem Starts Reel in a browser window instead of the native window.
rem Use this if the native window ever misbehaves (blank window, crashes).
set "REEL_WINDOW=browser"
call "%~dp0start.bat" %*
