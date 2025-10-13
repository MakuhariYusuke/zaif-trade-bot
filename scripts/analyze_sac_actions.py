#!/usr/bin/env python3
"""
Analyze SAC action distribution to identify bias causing BUY/SELL imbalance.
"""

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import logging

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import SAC
from ztb.trading.environment import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig, RewardSettings
from ztb.utils.logging_utils import get_logger

def analyze_sac_action_distribution(model_path: str, num_samples: int = 10000):
    """
    Analyze the action distribution of a trained SAC model.

    Args:
        model_path: Path to the SAC model
        num_samples: Number of action samples to analyze
    """
    logger = get_logger("sac_action_analysis")
    logger.info(f"Analyzing SAC action distribution from {model_path}")

    # Load the SAC model
    try:
        model = SAC.load(model_path)
        logger.info("Model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    # Create environment for testing
    config = EnvironmentConfig()
    reward_settings = RewardSettings()

    # Load data for environment
    import pandas as pd
    data_path = project_root / "btc_jpy_real_dataset.csv"
    df = pd.read_csv(data_path)

    # Use default config for testing
    env = HeavyTradingEnv(df=df, config=config)

    # Collect actions
    actions_continuous = []
    actions_discrete = []
    observations = []

    logger.info(f"Collecting {num_samples} action samples...")

    obs, _ = env.reset()
    for i in range(num_samples):
        # Get action from SAC
        action_continuous, _ = model.predict(obs, deterministic=True)
        actions_continuous.append(float(action_continuous[0]))

        # Convert to discrete using current logic
        if action_continuous[0] > 0.15:
            discrete_action = 1  # BUY
        elif action_continuous[0] < -0.15:
            discrete_action = 2  # SELL
        else:
            discrete_action = 0  # HOLD

        actions_discrete.append(discrete_action)
        observations.append(obs.copy())

        # Take step in environment
        obs, reward, terminated, truncated, info = env.step(action_continuous)

        if terminated or truncated:
            obs, _ = env.reset()

        if (i + 1) % 1000 == 0:
            logger.info(f"Processed {i + 1}/{num_samples} samples")

    # Analyze continuous actions
    actions_continuous = np.array(actions_continuous)
    actions_discrete = np.array(actions_discrete)

    print("\n" + "="*60)
    print("SAC ACTION DISTRIBUTION ANALYSIS")
    print("="*60)

    print("\nCONTINUOUS ACTIONS STATISTICS:")
    print(f"  Mean: {actions_continuous.mean():.4f}")
    print(f"  Std:  {actions_continuous.std():.4f}")
    print(f"  Min:  {actions_continuous.min():.4f}")
    print(f"  Max:  {actions_continuous.max():.4f}")
    print(f"  Median: {np.median(actions_continuous):.4f}")

    # Check distribution symmetry
    positive_actions = actions_continuous[actions_continuous > 0]
    negative_actions = actions_continuous[actions_continuous < 0]

    print("\nDISTRIBUTION ANALYSIS:")
    print(f"  Positive actions: {len(positive_actions)} ({len(positive_actions)/len(actions_continuous)*100:.1f}%)")
    print(f"  Negative actions: {len(negative_actions)} ({len(negative_actions)/len(actions_continuous)*100:.1f}%)")
    print(f"  Zero actions: {np.sum(actions_continuous == 0.0)} ({np.sum(actions_continuous == 0.0)/len(actions_continuous)*100:.1f}%)")

    # Check threshold coverage
    buy_threshold = 0.15
    sell_threshold = -0.15

    buy_actions = actions_continuous[actions_continuous > buy_threshold]
    sell_actions = actions_continuous[actions_continuous < sell_threshold]
    hold_actions = actions_continuous[(actions_continuous >= sell_threshold) & (actions_continuous <= buy_threshold)]

    print("\nTHRESHOLD ANALYSIS (current: buy>0.15, sell<-0.15):")
    print(f"  BUY actions: {len(buy_actions)} ({len(buy_actions)/len(actions_continuous)*100:.1f}%)")
    print(f"  SELL actions: {len(sell_actions)} ({len(sell_actions)/len(actions_continuous)*100:.1f}%)")
    print(f"  HOLD actions: {len(hold_actions)} ({len(hold_actions)/len(actions_continuous)*100:.1f}%)")

    # Check for potential bias
    print("\nPOTENTIAL BIAS ANALYSIS:")
    if abs(actions_continuous.mean()) > 0.05:
        print(f"  ⚠️  SIGNIFICANT MEAN BIAS: {actions_continuous.mean():.4f}")
        if actions_continuous.mean() > 0:
            print("     → SAC tends to output POSITIVE actions (BUY bias)")
        else:
            print("     → SAC tends to output NEGATIVE actions (SELL bias)")
    else:
        print("  ✅ Mean is approximately zero (good balance)")

    # Check standard deviation
    if actions_continuous.std() < 0.5:
        print(f"  ⚠️  LOW VARIANCE: {actions_continuous.std():.4f} (SAC may be too conservative)")
    elif actions_continuous.std() > 1.2:
        print(f"  ⚠️  HIGH VARIANCE: {actions_continuous.std():.4f} (SAC may be too random)")
    else:
        print(f"  ✅ Good variance: {actions_continuous.std():.4f}")

    # Suggest threshold adjustments
    print("\nTHRESHOLD RECOMMENDATIONS:")

    # Calculate percentiles for balanced distribution
    p33 = np.percentile(actions_continuous, 33.33)
    p67 = np.percentile(actions_continuous, 66.67)

    print(f"  For balanced BUY/SELL/HOLD (33%/33%/33%):")
    print(f"    BUY threshold: > {p67:.3f}")
    print(f"    SELL threshold: < {p33:.3f}")

    # For current distribution, suggest symmetric thresholds
    mean_abs = abs(actions_continuous.mean())
    suggested_buy = mean_abs + actions_continuous.std() * 0.5
    suggested_sell = -mean_abs - actions_continuous.std() * 0.5

    print(f"  For symmetric thresholds around mean:")
    print(f"    BUY threshold: > {suggested_buy:.3f}")
    print(f"    SELL threshold: < {suggested_sell:.3f}")

    # Plot histogram if matplotlib available
    try:
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.hist(actions_continuous, bins=50, alpha=0.7, color='blue')
        plt.axvline(x=buy_threshold, color='red', linestyle='--', label=f'BUY threshold ({buy_threshold})')
        plt.axvline(x=sell_threshold, color='green', linestyle='--', label=f'SELL threshold ({sell_threshold})')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.5, label='Zero')
        plt.title('Continuous Action Distribution')
        plt.xlabel('Action Value')
        plt.ylabel('Frequency')
        plt.legend()

        plt.subplot(2, 2, 2)
        discrete_counts = Counter(actions_discrete)
        labels = ['HOLD', 'BUY', 'SELL']
        counts = [discrete_counts.get(0, 0), discrete_counts.get(1, 0), discrete_counts.get(2, 0)]
        plt.bar(labels, counts, color=['gray', 'green', 'red'])
        plt.title('Discrete Action Distribution')
        plt.ylabel('Count')

        plt.subplot(2, 2, 3)
        plt.hist(actions_continuous, bins=50, alpha=0.7, color='blue', cumulative=True, density=True)
        plt.axvline(x=buy_threshold, color='red', linestyle='--')
        plt.axvline(x=sell_threshold, color='green', linestyle='--')
        plt.title('Cumulative Distribution')
        plt.xlabel('Action Value')
        plt.ylabel('Cumulative Probability')

        plt.subplot(2, 2, 4)
        # Show action distribution by ranges
        ranges = [(-1, sell_threshold), (sell_threshold, buy_threshold), (buy_threshold, 1)]
        range_labels = ['SELL zone', 'HOLD zone', 'BUY zone']
        range_counts = [
            np.sum((actions_continuous >= r[0]) & (actions_continuous < r[1])) for r in ranges
        ]
        plt.bar(range_labels, range_counts, color=['red', 'gray', 'green'])
        plt.title('Actions by Decision Zones')
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig('sac_action_distribution_analysis.png', dpi=150, bbox_inches='tight')
        print("\n📊 Plot saved as 'sac_action_distribution_analysis.png'")
    except ImportError:
        print("\n⚠️  Matplotlib not available for plotting")
    except Exception as e:
        print(f"\n⚠️  Plotting failed: {e}")

    env.close()

if __name__ == "__main__":
    # Analyze the latest SAC model
    model_path = "checkpoints/sac_session/sac_v406_sell_bonus_adjusted_final.zip"

    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        # Try the previous version
        model_path = "checkpoints/sac_session/sac_v404_extreme_win_rate_final.zip"
        if not Path(model_path).exists():
            print(f"Fallback model not found: {model_path}")
            sys.exit(1)

    analyze_sac_action_distribution(model_path, num_samples=5000)