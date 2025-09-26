#!/usr/bin/env python3
"""
Test script for Pydantic validation models.
"""

from config.validation import ExperimentConfigModel, validate_experiment_config

def test_validation():
    """Test Pydantic validation functionality"""

    print("Testing Pydantic validation...")

    # Test valid config
    config_dict = {
        'steps': 1000,
        'strategy': 'generalization',
        'learning_rate': 0.01,
        'batch_size': 64
    }

    try:
        model = validate_experiment_config(config_dict)
        print('✓ Valid config accepted:', model.model_dump())
    except Exception as e:
        print('✗ Valid config rejected:', e)

    # Test invalid config
    invalid_config = {
        'steps': -1,  # Invalid: negative steps
        'strategy': 'invalid_strategy',  # Invalid: not in allowed strategies
        'learning_rate': 2.0  # Invalid: > 1
    }

    try:
        model = validate_experiment_config(invalid_config)
        print('✗ Invalid config accepted:', model.model_dump())
    except Exception as e:
        print('✓ Invalid config rejected:', str(e))

    # Test field validators
    print("\nTesting field validators...")

    # Test strategy validation
    try:
        ExperimentConfigModel(steps=100, strategy='aggressive')
        print('✓ Valid strategy accepted')
    except Exception as e:
        print('✗ Valid strategy rejected:', e)

    try:
        ExperimentConfigModel(steps=100, strategy='invalid')
        print('✗ Invalid strategy accepted')
    except Exception as e:
        print('✓ Invalid strategy rejected:', str(e))

    # Test split validation
    try:
        ExperimentConfigModel(steps=100, strategy='generalization', test_split=0.5)
        print('✓ Valid split accepted')
    except Exception as e:
        print('✗ Valid split rejected:', e)

    try:
        ExperimentConfigModel(steps=100, strategy='generalization', test_split=1.5)
        print('✗ Invalid split accepted')
    except Exception as e:
        print('✓ Invalid split rejected:', str(e))

if __name__ == '__main__':
    test_validation()