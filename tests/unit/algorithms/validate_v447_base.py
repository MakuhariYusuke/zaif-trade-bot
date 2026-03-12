#!/usr/bin/env python3
"""Quick validation test for base config."""
import json
from pathlib import Path

config_path = Path('config/v447/sac_v447_1m_multiframe_config.json')

with open(config_path, encoding='utf-8') as f:
    config = json.load(f)

print('✓ Base config loaded successfully')
print(f"  Model: {config['training']['model_name']}")
print(f"  Timesteps: {config['training']['total_timesteps']}")
print(f"  LR: {config['training']['sac_hyperparameters']['learning_rate']}")
print(f"  Balance Penalty: {config['training']['environment']['behavior_optimization']['balance_penalty']}")
print(f"  Ent Coef: {config['training']['sac_hyperparameters']['ent_coef']}")
print(f"  Multi-timeframe: {config['training']['environment']['enable_multi_timeframe']}")
