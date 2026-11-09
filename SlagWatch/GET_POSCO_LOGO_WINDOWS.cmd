@echo off
cd /d "%~dp0"
if not exist "assets" mkdir "assets"
powershell.exe -NoProfile -Command "$ErrorActionPreference='Stop'; Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/tmdgns104/posco_hackerton/dohyeon/SlagWatch/assets/posco-logo.png' -OutFile 'assets/posco-logo.png'; Add-Type -AssemblyName System.Drawing; $img=[System.Drawing.Image]::FromFile((Resolve-Path 'assets/posco-logo.png')); $img.Dispose(); Write-Host 'POSCO logo saved (offline ready).'"
if errorlevel 1 (echo Failed to download POSCO logo. Check your internet connection. & pause & exit /b 1)
pause
