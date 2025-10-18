@echo off
REM Comprehensive analysis of v424 backtest results
set PYTHONPATH=%~dp0
python ztb/analysis/analyze_backtest.py results/backtest_v424_cost_aware.json reports/training_report_sac_sac_v424_cost_aware_20251018_174314.json
exit