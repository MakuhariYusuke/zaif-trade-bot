#!/usr/bin/env python3
"""Test script for causal inference feature selection."""

import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from ztb.features.causal_inference import CausalFeatureSelector, CausalInferenceEngine, create_causal_engine
from ztb.features.adaptive_selection import AdaptiveFeatureSelector


def test_causal_feature_selector():
    """Test causal feature selector."""
    print("Testing Causal Feature Selector...")

    # Create selector
    selector = CausalFeatureSelector(treatment_threshold=0.05, min_samples=50)

    # Create test data with some causal relationships
    np.random.seed(42)
    n_samples = 200

    # Create features with different causal effects on reward
    data = {
        'feature_strong': np.random.randn(n_samples) * 2,  # Strong effect
        'feature_medium': np.random.randn(n_samples) * 1.5,  # Medium effect
        'feature_weak': np.random.randn(n_samples) * 0.5,  # Weak effect
        'feature_noise': np.random.randn(n_samples) * 0.1,  # Noise
        'confounder_price': np.random.randn(n_samples),  # Confounder
        'confounder_volume': np.random.randn(n_samples),  # Confounder
    }

    # Create reward with causal relationships
    reward = (
        0.8 * data['feature_strong'] +
        0.4 * data['feature_medium'] +
        0.1 * data['feature_weak'] +
        0.05 * data['feature_noise'] +
        0.2 * data['confounder_price'] +
        np.random.randn(n_samples) * 0.1  # Noise
    )
    data['reward'] = reward

    df = pd.DataFrame(data)
    features = ['feature_strong', 'feature_medium', 'feature_weak', 'feature_noise']
    confounders = ['confounder_price', 'confounder_volume']

    # Test causal effect estimation
    effect_result = selector.estimate_causal_effect(df, 'feature_strong', 'reward', confounders)
    print(f"Causal effect for feature_strong: {effect_result}")

    # Test feature selection
    selected_features, results = selector.select_features_causal(df, features, 'reward', confounders)
    print(f"Selected features: {selected_features}")
    print(f"Selection results: {list(results.keys())}")

    # Test importance
    importance = selector.get_feature_importance()
    print(f"Feature importance: {importance}")

    print("✓ Causal Feature Selector test passed!")
    return True


def test_causal_inference_engine():
    """Test causal inference engine."""
    print("Testing Causal Inference Engine...")

    # Create engine
    config = {
        'treatment_threshold': 0.05,
        'min_samples': 50,
        'max_features': 3
    }
    engine = create_causal_engine(config)

    # Create test data
    np.random.seed(42)
    n_samples = 150

    data = {
        'feature_a': np.random.randn(n_samples),
        'feature_b': np.random.randn(n_samples),
        'feature_c': np.random.randn(n_samples),
        'price': np.random.randn(n_samples),
        'volume': np.random.randn(n_samples),
        'reward': np.random.randn(n_samples)
    }

    df = pd.DataFrame(data)
    features = ['feature_a', 'feature_b', 'feature_c']

    # Test analysis
    result = engine.analyze_causal_relationships(df, features, 'reward')
    print(f"Analysis result keys: {list(result.keys())}")
    print(f"Selected features: {result['selected_features']}")

    print("✓ Causal Inference Engine test passed!")
    return True


def test_adaptive_selector_with_causal():
    """Test adaptive selector with causal inference."""
    print("Testing Adaptive Selector with Causal Inference...")

    # Create selector
    selector = AdaptiveFeatureSelector()

    # Initialize with causal enabled
    n_features = 10
    config = {
        'enabled': False,  # Attention disabled for this test
        'causal_enabled': True,
        'causal_config': {
            'treatment_threshold': 0.05,
            'min_samples': 50,
            'max_features': 5
        }
    }
    selector.initialize_attention_trainer(n_features, config)

    # Create test dataframe
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='h')
    df = pd.DataFrame({
        'timestamp': dates,
        'close': 50000 + np.cumsum(np.random.normal(0, 100, 100)),
        'adx': np.random.uniform(15, 35, 100),
        'atr': np.random.uniform(50, 200, 100),
    })

    # Add features and reward
    feature_names = [f'feature_{i}' for i in range(n_features)]
    for name in feature_names:
        df[name] = np.random.randn(100)

    # Create reward with some causal relationships
    df['reward'] = (
        0.5 * df['feature_0'] +
        0.3 * df['feature_1'] +
        0.1 * df['feature_2'] +
        np.random.randn(100) * 0.1
    )

    # Test causal feature selection
    selected_features, stats = selector.select_features(df, feature_names, use_causal=True, outcome_feature='reward')
    print(f"Causal selection: {len(selected_features)} features selected")
    print(f"Selection method: {stats.get('selection_method', 'unknown')}")

    # Test adaptive selection
    selected_adaptive, stats_adaptive = selector.select_features(df, feature_names, use_causal=False)
    print(f"Adaptive selection: {len(selected_adaptive)} features selected")
    print(f"Selection method: {stats_adaptive.get('selection_method', 'unknown')}")

    print("✓ Adaptive Selector with Causal test passed!")
    return True


if __name__ == "__main__":
    try:
        test_causal_feature_selector()
        print()
        test_causal_inference_engine()
        print()
        test_adaptive_selector_with_causal()
        print("\n✓ All causal inference tests passed!")
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)