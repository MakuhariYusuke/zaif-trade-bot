import sys
sys.path.insert(0, '.')

# Test imports
try:
    from ztb.training.training_utils import setup_project_path, create_ppo_model, load_training_data, save_model_with_path, evaluate_model, print_training_results
    print('✓ training_utils imports successful')
except Exception as e:
    print(f'✗ training_utils imports failed: {e}')

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

try:
    import ztb.training.bayes_opt_ppo as bop
    print('✓ bayes_opt_ppo import successful')
except Exception as e:
    print(f'✗ bayes_opt_ppo import failed: {e}')

try:
    import ztb.training.ensemble_models as em
    print('✓ ensemble_models import successful')
except Exception as e:
    print(f'✗ ensemble_models import failed: {e}')

print('All imports completed successfully!')