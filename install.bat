@echo off
rem Double-click to install the CALPHAD agent on Windows (runs install.ps1).
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install.ps1" %*
pause
