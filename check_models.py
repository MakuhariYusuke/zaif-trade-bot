from pathlib import Path
from stable_baselines3 import PPO

models_dir = Path('models')
zip_files = list(models_dir.glob('*.zip'))
print(f'Found {len(zip_files)} model files')

# Check observation spaces for a few models
for i, model_path in enumerate(zip_files[:5]):  # Check first 5
    try:
        model = PPO.load(model_path)
        print(f'{model_path.name}: observation space {model.observation_space.shape}')
    except Exception as e:
        print(f'{model_path.name}: Error loading - {e}')