@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0" || goto :path_error
if not exist "logs" mkdir "logs"
if not exist "logs" goto :path_error
if exist ".venv\Scripts\python.exe" goto :run
echo Checking Python 3.11 64-bit...
py -3.11 -c "import sys,struct; sys.exit(0 if sys.version_info[:2]==(3,11) and struct.calcsize('P')==8 else 1)" >>"logs\bootstrap.log" 2>&1
if errorlevel 1 goto :python_error
py -3.11 -m venv ".venv" >>"logs\bootstrap.log" 2>&1
if errorlevel 1 goto :venv_error
:run
".venv\Scripts\python.exe" -c "import sys,struct; sys.exit(0 if sys.version_info[:2]==(3,11) and struct.calcsize('P')==8 else 1)" >>"logs\bootstrap.log" 2>&1
if errorlevel 1 goto :runtime_error
".venv\Scripts\python.exe" "start_demo.py" --bootstrap
if errorlevel 1 goto :launch_error
exit /b 0
:python_error
echo Python 3.11 64-bit and the Windows py launcher are required. See logs\bootstrap.log
goto :failed
:venv_error
echo Could not create .venv. See logs\bootstrap.log
goto :failed
:runtime_error
echo Existing .venv could not start Python 3.11 64-bit. See logs\bootstrap.log
goto :failed
:launch_error
echo SteelTrust startup failed. See logs\startup.log in this folder.
goto :failed
:path_error
echo Cannot open the project folder or create its logs folder.
:failed
pause
exit /b 1
