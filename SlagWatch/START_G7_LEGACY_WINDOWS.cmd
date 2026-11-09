@echo off
setlocal EnableExtensions
cd /d "%~dp0" || goto :fail
if not exist ".venv_g7\Scripts\python.exe" (
 echo Legacy G7 environment missing. Run SETUP_G7_PY312_WINDOWS.cmd.
 goto :fail
)
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"
".venv_g7\Scripts\python.exe" -m streamlit run app_demo.py --server.address 127.0.0.1 --server.port 8502 --browser.gatherUsageStats=false
if errorlevel 1 goto :fail
exit /b 0
:fail
pause
exit /b 1
