@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv_g7\Scripts\python.exe" (echo Run SETUP_G7_PY312_WINDOWS.cmd first & pause & exit /b 1)
".venv_g7\Scripts\python.exe" -m src.slag_compare_cli report
if errorlevel 1 (echo G7 verification failed & pause & exit /b 1)
echo Verified; results saved under reports
pause
