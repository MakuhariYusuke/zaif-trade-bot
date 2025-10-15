@echo off
set CUDA_VISIBLE_DEVICES=-1
call venv311\Scripts\activate.bat
python scripts/train_sac_v500_equalized.py
pause