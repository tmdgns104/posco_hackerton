@echo off
setlocal EnableExtensions
cd /d "%~dp0" || goto :fail
if not exist ".venv_g7\Scripts\python.exe" (
  echo Python 3.12 environment missing. Run SETUP_G7_PY312_WINDOWS.cmd first.
  goto :fail
)
".venv_g7\Scripts\python.exe" -c "import sys; assert sys.version_info[:2]==(3,12); import streamlit,torch,sklearn,plotly,pandas"
if errorlevel 1 goto :fail
echo Launching SteelTrust G7...
".venv_g7\Scripts\python.exe" -m streamlit run app_demo.py --server.address 127.0.0.1
if errorlevel 1 goto :fail
exit /b 0
:fail
echo SteelTrust G7 failed to start. Please read README_G7_KO.md.
pause
exit /b 1
