@echo off
call venv311\Scripts\activate.bat
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install stable-baselines3[extra]
pause