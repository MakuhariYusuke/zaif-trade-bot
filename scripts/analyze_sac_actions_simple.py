#!/usr/bin/env python3
"""
Analyze SAC action distribution to identify bias causing BUY/SELL imbalance.
"""

import sys
from pathlib import Path
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from stable_baselines3 import SAC

def analyze_sac_action_distribution_simple(model_path: str, num_samples: int = 10000):
    """
    Analyze the action distribution of a trained SAC model using dummy observations.

    Args:
        model_path: Path to the SAC model
        num_samples: Number of action samples to analyze
    """
    print(f"Analyzing SAC action distribution from {model_path}")

    # Load the SAC model
    try:
        model = SAC.load(model_path)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    # Create dummy observations (typical market data)
    # Using realistic BTC/JPY price data ranges
    dummy_obs = np.random.rand(num_samples, model.observation_space.shape[0])

    # Normalize to typical ranges
    # Price data (indices 0-3): around 5M-6M JPY
    dummy_obs[:, :4] = dummy_obs[:, :4] * 1000000 + 5000000
    # Technical indicators (other indices): -1 to 1 range
    dummy_obs[:, 4:] = (dummy_obs[:, 4:] - 0.5) * 2

    actions_continuous = []

    print(f"Collecting {num_samples} action samples...")

    for i in range(num_samples):
        # Get action from SAC (try both deterministic and stochastic)
        action_det, _ = model.predict(dummy_obs[i], deterministic=True)
        action_stoch, _ = model.predict(dummy_obs[i], deterministic=False)
        actions_continuous.append(float(action_stoch[0]))  # Use stochastic for analysis

        if (i + 1) % 1000 == 0:
            print(f"Processed {i + 1}/{num_samples} samples")

    # Analyze continuous actions
    actions_continuous = np.array(actions_continuous)

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

    # Check threshold coverage with current settings
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

    # Analyze action distribution by ranges
    print("\nACTION RANGE ANALYSIS:")
    ranges = [
        (-1.0, -0.8), (-0.8, -0.6), (-0.6, -0.4), (-0.4, -0.2),
        (-0.2, 0.0), (0.0, 0.2), (0.2, 0.4), (0.4, 0.6),
        (0.6, 0.8), (0.8, 1.0)
    ]

    print("  Range      | Count | Percentage")
    print("  -----------|-------|-----------")
    for r in ranges:
        count = np.sum((actions_continuous >= r[0]) & (actions_continuous < r[1]))
        pct = count / len(actions_continuous) * 100
        print("5.1f")

if __name__ == "__main__":
    # Try different SAC models to find one that works
    models_to_try = [
        "checkpoints/sac_session/sac_v406_sell_bonus_adjusted_final.zip",
        "checkpoints/sac_session/sac_v404_extreme_win_rate_final.zip",
        "checkpoints/sac_session/sac_v403_high_win_rate_final.zip",
        "checkpoints/sac_session/sac_v399_balanced_reward_final.zip"
    ]

    for model_path in models_to_try:
        if Path(model_path).exists():
            print(f"\n{'='*80}")
            print(f"Testing model: {model_path}")
            print(f"{'='*80}")
            analyze_sac_action_distribution_simple(model_path, num_samples=2000)
            break
    else:
        print("No valid SAC models found!")
        sys.exit(1)