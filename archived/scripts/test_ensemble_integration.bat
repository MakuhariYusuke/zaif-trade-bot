@echo off
cd c:\Users\Admin\dev\zaif-trade-bot
python -c "
from ztb.training.trainers.sac_trainer import SACAlgorithmTrainer
from ztb.training.core.config_manager import ConfigManager
import json

# Load config
with open('configs/sac_v428_ensemble_test.json', 'r') as f:
    config = json.load(f)

# Create trainer
config_manager = ConfigManager()
trainer = SACAlgorithmTrainer(config_manager)

# Test ensemble initialization
trainer.initialize_ensemble(config)
print(f'Ensemble enabled: {trainer.ensemble_enabled}')
print(f'Ensemble system created: {trainer.ensemble_system is not None}')
print(f'Ensemble config created: {trainer.ensemble_config is not None}')

# Test ensemble prediction (mock observation)
import numpy as np
obs = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
result = trainer.predict_with_ensemble(obs)
print(f'Ensemble prediction result: {result}')

print('SAC trainer ensemble integration test passed!')
"
