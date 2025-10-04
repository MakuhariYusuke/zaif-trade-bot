import zipfile

model_path = 'models/scalping_15s_balance_test12_balanced_data.zip'
with zipfile.ZipFile(model_path, 'r') as zf:
    if 'system_info.txt' in zf.namelist():
        with zf.open('system_info.txt') as f:
            content = f.read().decode('utf-8')
            print('System info:')
            print(content)
    else:
        print('No system_info.txt found')

    # Try to load the model and check observation space
    try:
        from stable_baselines3 import PPO
        model = PPO.load(model_path)
        print(f'\nModel observation space: {model.observation_space}')
        print(f'Observation space shape: {model.observation_space.shape}')
    except Exception as e:
        print(f'Error loading model: {e}')