@echo off
REM SAC v423 Training Batch Script
REM Runs SAC v423 with configurable timesteps

echo ========================================
echo SAC v423 Initial Test Training
echo ========================================
echo.

if "%1"=="" (
    echo Usage: %0 [timesteps]
    echo Example: %0 1000
    echo Default timesteps: 1000
    set TIMESTEPS=1000
) else (
    set TIMESTEPS=%1
)

echo Starting SAC v423 training with %TIMESTEPS% timesteps...
echo.

REM Run the training script
python scripts\train_sac_v423.py --timesteps %TIMESTEPS%

echo.
if %ERRORLEVEL% EQU 0 (
    echo ✅ SAC v423 training completed successfully!
    echo Check results in: results\sac_v423\
) else (
    echo ❌ Training failed with error code %ERRORLEVEL%
)

echo.
pause
