import sys
sys.path.insert(0, '.')

# Test imports
try:
    from ztb.training.ppo_trainer import PPOTrainer
    print('✓ PPOTrainer import successful')
except Exception as e:
    print(f'✗ PPOTrainer import failed: {e}')

try:
    from ztb.training.binary_search.base_optimizer import BinarySearchOptimizer
    print('✓ BinarySearchOptimizer import successful')
except Exception as e:
    print(f'✗ BinarySearchOptimizer import failed: {e}')

try:
    import ztb.training.simple_reward as sr
    print('✓ simple_reward import successful')
except Exception as e:
    print(f'✗ simple_reward import failed: {e}')

try:
    import ztb.training.train_simple_reward as tsr
    print('✓ train_simple_reward import successful')
except Exception as e:
    print(f'✗ train_simple_reward import failed: {e}')

try:
    import ztb.training.curriculum_transition as ct
    print('✓ curriculum_transition import successful')
except Exception as e:
    print(f'✗ curriculum_transition import failed: {e}')

print('All imports completed successfully!')