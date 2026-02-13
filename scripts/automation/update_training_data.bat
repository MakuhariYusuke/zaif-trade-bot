@echo off
setlocal

set DAYS=%1
if "%DAYS%"=="" set DAYS=7

echo [INFO] Updating BTC/JPY training data (source=all, days=%DAYS%)
cd /d %~dp0\..\.. || exit /b 1

if not exist ".venv\Scripts\python.exe" (
  echo [ERROR] .venv\Scripts\python.exe not found.
  echo [HINT] Create venv and install dependencies first.
  exit /b 1
)

.venv\Scripts\python.exe scripts\v456\update_data_comprehensive.py --source all --days %DAYS%
if errorlevel 1 (
  echo [ERROR] Data update failed.
  exit /b 1
)

echo [OK] Data update completed.
endlocal

