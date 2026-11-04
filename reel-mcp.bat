@echo off
rem Reel as an MCP server, so an AI assistant (Claude Desktop, Claude Code, ...) can make images and videos with Reel's models.
rem An assistant starts this itself; it is not meant to be double-clicked. Run "reel-mcp.bat --config" to get the text to paste
rem into the assistant's settings. If Reel is not running, the server starts it in the background (no window).
rem Nothing but the protocol may be printed to stdout, so every message here goes to stderr.
setlocal
cd /d "%~dp0"
set "PYTHONUTF8=1"
if not exist "%~dp0.venv\Scripts\python.exe" (
    echo [reel-mcp] Run start.bat once first: it sets Reel up. 1>&2
    exit /b 1
)
"%~dp0.venv\Scripts\python.exe" -m reel.mcp_server %*
exit /b %ERRORLEVEL%
