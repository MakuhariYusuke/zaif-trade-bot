#!/usr/bin/env python3
"""
SAC v427 vs v437 Feature Comparison Analysis

Analyzes the performance difference between v427 (154 features) and v437 (22 features)
to understand the impact of feature dimensionality on SAC training performance.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_evaluation_results(eval_csv_path: str) -> pd.DataFrame:
    """Load evaluation results from monitor CSV."""
    try:
        df = pd.read_csv(eval_csv_path, comment='#')
        return df
    except Exception as e:
        logger.error(f"Failed to load evaluation results: {e}")
        return pd.DataFrame()

def analyze_training_comparison():
    """Compare v427 vs v437 training performance."""

    print("=== SAC v427 vs v437 Feature Comparison Analysis ===\n")

    # Load evaluation results
    v427_eval = load_evaluation_results("tensorboard/v427_eval.monitor.csv")
    v437_eval = load_evaluation_results("tensorboard/v437_eval.monitor.csv")

    print("1. FEATURE DIMENSIONS:")
    print("   v427: 154 features (129 generated + padding)")
    print("   v437: 22 features (quality-filtered)")
    print()

    print("2. TRAINING RESULTS:")

    if not v427_eval.empty:
        v427_mean_reward = v427_eval['r'].mean()
        v427_mean_length = v427_eval['l'].mean()
        print(f"   v427 Mean reward: {v427_mean_reward:.2f}")
        print(f"   v427 Mean episode length: {v427_mean_length:.1f}")
        print(f"   v427 Best episode reward: {v427_eval['r'].max():.2f}")
        print(f"   v427 Worst episode reward: {v427_eval['r'].min():.2f}")
    else:
        print("   v427: No evaluation data available")

    if not v437_eval.empty:
        v437_mean_reward = v437_eval['r'].mean()
        v437_mean_length = v437_eval['l'].mean()
        print(f"   v437 Mean reward: {v437_mean_reward:.2f}")
        print(f"   v437 Mean episode length: {v437_mean_length:.1f}")
        print(f"   v437 Best episode reward: {v437_eval['r'].max():.2f}")
        print(f"   v437 Worst episode reward: {v437_eval['r'].min():.2f}")
    else:
        print("   v437: No evaluation data available")

    print()

    print("3. CONFIGURATION DIFFERENCES:")
    print("   v427 curriculum_stage: strong_penalty_trading")
    print("   v437 curriculum_stage: balanced_trading")
    print("   v427 has adaptive_feature_selection enabled")
    print("   v437 uses quality filtering (NaN>10%, variance=0, zero-rate>80%)")
    print()

    print("4. ANALYSIS:")
    print("   Both versions show poor performance with large negative rewards")
    print("   v427 generates only 129 meaningful features, requires padding to 154")
    print("   Quality filtering in v437 reduces features from 129 to 22")
    print("   Strong penalty trading in v427 may be overly restrictive")
    print("   Adaptive feature selection may add complexity without benefit")
    print()

    print("5. RECOMMENDATIONS:")
    print("   - Investigate reward function and curriculum settings")
    print("   - Review feature quality and relevance")
    print("   - Consider hybrid approach: quality-filtered v427 features")
    print("   - Test with balanced_trading curriculum for v427")
    print("   - Validate feature engineering pipeline")

if __name__ == "__main__":
    analyze_training_comparison()