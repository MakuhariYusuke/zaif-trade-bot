import os

from stable_baselines3 import SAC

model_path = "models/sac_model.zip"
if os.path.exists(model_path):
    print(f"Loading model from {model_path}")
    model = SAC.load(model_path)
    print(f"Model Observation Space: {model.observation_space}")
else:
    print(f"Model file not found: {model_path}")
