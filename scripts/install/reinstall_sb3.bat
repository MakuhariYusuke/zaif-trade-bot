@echo off
call venv311\Scripts\activate.bat
pip uninstall -y stable-baselines3
pip install stable-baselines3
pause