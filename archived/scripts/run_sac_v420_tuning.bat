@echo off
REM SAC v420 Parameter Tuning Batch Script
REM Executes systematic parameter tuning for SAC v420

echo ========================================
echo SAC v420 Parameter Tuning Suite
echo ========================================
echo.

echo Starting parameter tuning script...
echo.

REM Run the parameter tuning script
python scripts\run_sac_v420_parameter_tuning.py

echo.
echo Parameter tuning completed!
echo Check results in: results\sac_v420_tuning\tuning_summary.json

pause
