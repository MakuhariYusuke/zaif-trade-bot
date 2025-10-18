@echo off
REM SAC v423a Training Batch Script
REM Runs SAC v423a with configurable timesteps

echo ========================================
echo SAC v423a Step Test Training
echo ========================================
echo.

if "%1"=="" (
    echo Usage: %0 [timesteps]
    echo Example: %0 5000
    echo Default timesteps: 5000
    set TIMESTEPS=5000
) else (
    set TIMESTEPS=%1
)

echo Starting SAC v423a training with %TIMESTEPS% timesteps...
echo.

REM Run the training script
python scripts\train_sac_v423.py --timesteps %TIMESTEPS% --config config\sac_v423a_step_test_config.json

echo.
if %ERRORLEVEL% EQU 0 (
    echo ✅ SAC v423a training completed successfully!
    echo Check results in: reports\ and models\
) else (
    echo ❌ Training failed with error code %ERRORLEVEL%
)

echo.
pause