@echo off
setlocal
cd /d "%~dp0"
title Market Regime Dashboard Launcher

echo ================================================
echo      MARKET REGIME DASHBOARD - LAUNCHER
echo ================================================
echo.
echo Working folder: %CD%
echo.

set "PY_CMD="
where py >nul 2>&1
if %errorlevel%==0 set "PY_CMD=py"

if not defined PY_CMD (
    where python >nul 2>&1
    if %errorlevel%==0 set "PY_CMD=python"
)

if not defined PY_CMD (
    echo ERROR: Python was not found on this computer.
    echo Install Python 3.11 or newer from python.org and tick "Add Python to PATH".
    echo.
    pause
    exit /b 1
)

echo Python command found: %PY_CMD%
%PY_CMD% --version
echo.

echo [1/2] Installing/checking required packages...
%PY_CMD% -m pip install --upgrade pip
if errorlevel 1 goto :pip_error

%PY_CMD% -m pip install -r requirements.txt
if errorlevel 1 goto :pip_error

echo.
echo [2/2] Starting dashboard...
echo Open this in browser if it does not open automatically:
echo http://localhost:8501
echo.
%PY_CMD% -m streamlit run app.py

if errorlevel 1 goto :app_error

goto :end

:pip_error
echo.
echo ERROR: Package installation failed.
echo Please take a screenshot of the error above.
echo.
pause
exit /b 1

:app_error
echo.
echo ERROR: The app failed to start.
echo Please take a screenshot of the error above.
echo.
pause
exit /b 1

:end
echo.
echo Dashboard stopped.
pause
endlocal
