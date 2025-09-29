@echo off
REM Windows Trading Service Wrapper
REM This script provides Windows service-like functionality

setlocal enabledelayedexpansion

set "SCRIPT_DIR=%~dp0"
set "PROJECT_DIR=%SCRIPT_DIR%..\..\.."
set "SERVICE_NAME=ZaifTradingBot"
set "LOG_FILE=%PROJECT_DIR%\logs\trading_service.log"

cd /d "%PROJECT_DIR%"

echo [%DATE% %TIME%] Starting Zaif Trading Service >> "%LOG_FILE%"

:restart_loop
echo [%DATE% %TIME%] Starting trading service process >> "%LOG_FILE%"

REM Run the trading service
python -m ztb.live.service_runner --config config\production.yaml --log-level INFO

set EXIT_CODE=%errorlevel%
echo [%DATE% %TIME%] Trading service exited with code %EXIT_CODE% >> "%LOG_FILE%"

REM Check if we should restart
if %EXIT_CODE% equ 0 (
    echo [%DATE% %TIME%] Service completed normally, restarting... >> "%LOG_FILE%"
    timeout /t 5 /nobreak > nul
    goto restart_loop
) else (
    echo [%DATE% %TIME%] Service crashed with exit code %EXIT_CODE%, restarting in 30 seconds... >> "%LOG_FILE%"
    timeout /t 30 /nobreak > nul
    goto restart_loop
)

echo [%DATE% %TIME%] Service stopped >> "%LOG_FILE%"