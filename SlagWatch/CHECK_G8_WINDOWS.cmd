@echo off
setlocal EnableExtensions
cd /d "%~dp0" || goto :fail
if not exist ".venv_g8\Scripts\python.exe" (
 echo Run SETUP_G8_PY312_WINDOWS.cmd first.
 goto :fail
)
".venv_g8\Scripts\python.exe" -m unittest tests.test_g8_easy_core tests.test_g8_catalog tests.test_g8_model_matrix_ui tests.test_slag_compare tests.test_adaptive_redteam -v
if errorlevel 1 goto :fail
".venv_g8\Scripts\python.exe" -m compileall -q app_easy.py src\g8_core.py src\g8_easy.py src\g8_catalog.py src\g8_model_matrix_ui.py
if errorlevel 1 goto :fail
echo SteelTrust G8.2 model matrix, evidence and syntax checks passed.
pause
exit /b 0
:fail
echo G8 verification failed.
pause
exit /b 1
