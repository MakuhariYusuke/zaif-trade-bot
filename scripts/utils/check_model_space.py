#!/usr/bin/env python3
from stable_baselines3 import PPO

model = PPO.load("models/sac_v445.3_strong_selling_optimized_final.zip")
print("Model observation space:", model.observation_space)
print("Model action space:", model.action_space)
