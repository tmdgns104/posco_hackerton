@echo off
setlocal EnableExtensions
cd /d "%~dp0" || goto :fail
py -3.12 -c "import sys,struct; assert sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8"
if errorlevel 1 goto :fail
py -3.12 -m venv .venv_g7
if errorlevel 1 goto :fail
".venv_g7\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :fail
echo Installing CPU-compatible PyTorch. CPU inference works even on RTX hardware.
".venv_g7\Scripts\python.exe" -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 goto :fail
".venv_g7\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :fail
".venv_g7\Scripts\python.exe" -m pip check
if errorlevel 1 goto :fail
echo Setup complete. Launch START_WINDOWS.cmd
pause
exit /b 0
:fail
echo Setup failed. Check your network, 64-bit Python 3.12, and pip output.
pause
exit /b 1
