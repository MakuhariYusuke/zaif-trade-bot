import json
from ztb.training.unified_trainer import UnifiedTrainer

# Load test config
with open('sell_mitigation_test_config.json', 'r') as f:
    config = json.load(f)

# Create trainer and run short test
trainer = UnifiedTrainer(config)
print('Starting SELL mitigation integration test...')
try:
    model = trainer.train()
    print('SUCCESS: SELL mitigation integration test passed!')
    print(f'Model type: {type(model)}')
except Exception as e:
    print(f'FAILED: {e}')
    import traceback
    traceback.print_exc()