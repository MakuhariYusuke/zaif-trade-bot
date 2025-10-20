@echo off
echo ========================================
echo SAC v423b Step Test Training (10,000 steps)
echo ========================================

cd /d "%~dp0.."

echo Starting SAC v423b training with 10,000 timesteps...
python scripts\train_sac_v423.py --config config\sac_v423b_step_test_config.json --timesteps 10000

echo.
echo Training completed. Running analysis...
python scripts\analyze_sac_v423_series.py

echo.
echo ========================================
echo SAC v423b training and analysis complete!
echo ========================================
pause
