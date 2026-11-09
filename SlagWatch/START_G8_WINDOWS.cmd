@echo off
setlocal EnableExtensions
cd /d "%~dp0" || goto :fail
set "STREAMLIT_BROWSER_GATHER_USAGE_STATS=false"
if not exist ".venv_g8\Scripts\python.exe" (
 echo [SteelTrust] Python 3.12 setup is missing.
 echo Run SETUP_G8_PY312_WINDOWS.cmd first.
 goto :fail
)
".venv_g8\Scripts\python.exe" -c "import sys; assert sys.version_info[:2]==(3,12); import streamlit,torch,sklearn,pandas,scipy,cv2"
if errorlevel 1 goto :fail
set "STREAMLIT_SERVER_ADDRESS=127.0.0.1"
set "STREAMLIT_SERVER_PORT=8501"
echo.
echo  SteelTrust G8 - Simple Model Reliability Console
 echo  Open http://127.0.0.1:8501 in your browser.
echo.
".venv_g8\Scripts\python.exe" -m streamlit run app_easy.py --server.address 127.0.0.1 --server.port 8501 --browser.gatherUsageStats=false
if errorlevel 1 goto :fail
exit /b 0
:fail
echo.
echo [ERROR] SteelTrust did not start. Review the error above.
pause
exit /b 1
