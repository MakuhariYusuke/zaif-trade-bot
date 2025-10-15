#!/usr/bin/env python3
"""
Ultimate SAC SELL Bias Investigation - Action Distribution Deep Analysis

Investigates the fundamental cause of SELL bias by analyzing SAC's action distribution
in detail, including continuous action patterns and conversion thresholds.
"""

import sys
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from stable_baselines3 import SAC

from ztb.trading.constants import ACTION_HOLD, ACTION_BUY, ACTION_SELL

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

def load_model_and_data():
    """Load SAC model and sample data."""
    print("Loading SAC model and data...")

    # Load model
    model_path = "checkpoints/sac_session/sac_v406_moderate_rewards_final.zip"
    model = SAC.load(model_path)
    print(f"Model loaded from: {model_path}")

    # Load data
    data_path = "btc_jpy_real_dataset.csv"
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    print(f"Data loaded: {len(df)} rows")

    return model, df

def create_dummy_observation(df, step=100, obs_space=None):
    """Create a dummy observation for testing."""
    # Use actual data to create realistic observation
    row = df.iloc[step]

    # Create observation array (simplified for testing)
    # This should match the environment's observation space
    obs = np.array([
        row['close'],  # price
        row['volume'] if 'volume' in row else 1000,  # volume
        0.5,  # normalized features
        0.5,
        0.0,  # position
    ], dtype=np.float32)

    # If we have observation space info, adjust size
    if obs_space is not None:
        expected_size = obs_space.shape[0]
        if len(obs) < expected_size:
            obs = np.pad(obs, (0, expected_size - len(obs)), 'constant')
        elif len(obs) > expected_size:
            obs = obs[:expected_size]

    return obs

def analyze_action_distribution(model, df, n_samples=10000):
    """Analyze SAC's action distribution in detail."""

    print(f"\nAnalyzing action distribution with {n_samples} samples...")

    # Create a temporary environment to get observation space
    from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
    from ztb.trading.environment.utils.config import EnvironmentConfig

    temp_config = EnvironmentConfig(
        max_position_size=0.01,
        reward_scaling=8000.0,
        reward_settings={"use_simple_reward": True}
    )

    temp_env = HeavyTradingEnv(df=df.head(1000), config=temp_config)  # Small dataset for temp env
    obs_space = temp_env.observation_space

    print(f"Observation space shape: {obs_space.shape}")

    actions = []
    action_histogram = defaultdict(int)

    # Initialize all action counters
    action_histogram[0] = 0  # HOLD
    action_histogram[1] = 0  # BUY
    action_histogram[2] = 0  # SELL

    # Current thresholds from config
    buy_threshold = 0.05
    sell_threshold = -0.3

    for i in range(n_samples):
        # Create observation
        step = np.random.randint(100, len(df) - 100)  # Avoid edges
        obs = create_dummy_observation(df, step, obs_space)

        # Get action from model
        action_continuous, _ = model.predict(obs, deterministic=True)
        action_value = float(action_continuous[0])

        actions.append(action_value)

        # Convert to discrete and count
        if action_value > buy_threshold:
            discrete_action = ACTION_BUY  # BUY
        elif action_value < sell_threshold:
            discrete_action = ACTION_SELL  # SELL
        else:
            discrete_action = ACTION_HOLD  # HOLD

        action_histogram[discrete_action] += 1

    actions = np.array(actions)

    # Analysis results
    results = {
        "n_samples": n_samples,
        "action_stats": {
            "mean": float(np.mean(actions)),
            "std": float(np.std(actions)),
            "min": float(np.min(actions)),
            "max": float(np.max(actions)),
            "median": float(np.median(actions)),
            "q25": float(np.percentile(actions, 25)),
            "q75": float(np.percentile(actions, 75)),
        },
        "thresholds": {
            "buy_threshold": buy_threshold,
            "sell_threshold": sell_threshold,
        },
        "discrete_distribution": {
        0: action_histogram[0],  # HOLD
        1: action_histogram[1],  # BUY
        2: action_histogram[2],  # SELL
    },
        "bias_analysis": {}
    }

    # Analyze bias
    total_actions = sum(action_histogram.values())
    buy_ratio = action_histogram[1] / total_actions
    sell_ratio = action_histogram[2] / total_actions
    hold_ratio = action_histogram[0] / total_actions

    results["bias_analysis"] = {
        "buy_ratio": buy_ratio,
        "sell_ratio": sell_ratio,
        "hold_ratio": hold_ratio,
        "buy_sell_ratio": buy_ratio / sell_ratio if sell_ratio > 0 else float('inf'),
        "sell_buy_ratio": sell_ratio / buy_ratio if buy_ratio > 0 else float('inf'),
        "is_sell_biased": sell_ratio > buy_ratio * 1.5,  # SELL is 50% more than BUY
        "is_buy_biased": buy_ratio > sell_ratio * 1.5,   # BUY is 50% more than BUY
    }

    # Analyze action ranges
    results["range_analysis"] = {
        "strong_buy_range": f"({buy_threshold}, {np.max(actions):.3f}]",
        "buy_count": action_histogram[1],
        "hold_range": f"[{sell_threshold}, {buy_threshold}]",
        "hold_count": action_histogram[0],
        "sell_range": f"[{np.min(actions):.3f}, {sell_threshold})",
        "sell_count": action_histogram[2],
    }

    return results, actions

def plot_action_distribution(actions, results):
    """Create detailed plots of action distribution."""

    print("\nGenerating action distribution plots...")

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    # 1. Full action distribution histogram
    ax1.hist(actions, bins=100, alpha=0.7, color='blue', edgecolor='black')
    ax1.axvline(results["thresholds"]["buy_threshold"], color='green', linestyle='--',
                label=f'BUY threshold ({results["thresholds"]["buy_threshold"]})')
    ax1.axvline(results["thresholds"]["sell_threshold"], color='red', linestyle='--',
                label=f'SELL threshold ({results["thresholds"]["sell_threshold"]})')
    ax1.axvline(0, color='black', linestyle='-', alpha=0.5, label='Zero')
    ax1.set_title('SAC Action Distribution (Continuous)')
    ax1.set_xlabel('Action Value')
    ax1.set_ylabel('Frequency')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Discrete action distribution
    discrete_actions = ['HOLD', 'BUY', 'SELL']
    counts = [results["discrete_distribution"].get(0, 0),
              results["discrete_distribution"].get(1, 0),
              results["discrete_distribution"].get(2, 0)]
    colors = ['gray', 'green', 'red']
    bars = ax2.bar(discrete_actions, counts, color=colors, alpha=0.7)
    ax2.set_title('Discrete Action Distribution')
    ax2.set_ylabel('Count')
    ax2.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + max(counts)*0.01,
                f'{count:,}\n({count/sum(counts)*100:.1f}%)',
                ha='center', va='bottom', fontsize=10)

    # 3. Box plot
    ax3.boxplot(actions, vert=False, patch_artist=True,
                boxprops=dict(facecolor='lightblue', color='blue'),
                medianprops=dict(color='red', linewidth=2))
    ax3.set_title('Action Value Distribution (Box Plot)')
    ax3.set_xlabel('Action Value')
    ax3.grid(True, alpha=0.3)

    # 4. Cumulative distribution
    sorted_actions = np.sort(actions)
    yvals = np.arange(len(sorted_actions))/float(len(sorted_actions))
    ax4.plot(sorted_actions, yvals, 'b-', linewidth=2)
    ax4.axvline(results["thresholds"]["buy_threshold"], color='green', linestyle='--')
    ax4.axvline(results["thresholds"]["sell_threshold"], color='red', linestyle='--')
    ax4.axvline(0, color='black', linestyle='-', alpha=0.5)
    ax4.set_title('Cumulative Distribution Function')
    ax4.set_xlabel('Action Value')
    ax4.set_ylabel('Cumulative Probability')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/sac_action_distribution_detailed.png', dpi=300, bbox_inches='tight')
    plt.close()

    print("Plots saved to: results/sac_action_distribution_detailed.png")

def investigate_bias_causes(results, actions):
    """Investigate potential causes of SELL bias."""

    print("\n" + "="*80)
    print("BIAS CAUSE INVESTIGATION")
    print("="*80)

    bias_analysis = results["bias_analysis"]
    thresholds = results["thresholds"]

    print("Current Thresholds:")
    print(f"  BUY threshold:  {thresholds['buy_threshold']}")
    print(f"  SELL threshold: {thresholds['sell_threshold']}")
    print(f"  HOLD range:     [{thresholds['sell_threshold']}, {thresholds['buy_threshold']}]")

    print(f"\nAction Distribution:")
    print(f"  BUY:  {bias_analysis['buy_ratio']*100:.1f}% ({results['discrete_distribution'][1]})")
    print(f"  SELL: {bias_analysis['sell_ratio']*100:.1f}% ({results['discrete_distribution'][2]})")
    print(f"  HOLD: {bias_analysis['hold_ratio']*100:.1f}% ({results['discrete_distribution'][0]})")

    # Analyze the action range distribution
    action_stats = results["action_stats"]
    print(f"\nContinuous Action Statistics:")
    print(f"  Mean:   {action_stats['mean']:.4f}")
    print(f"  Std:    {action_stats['std']:.4f}")
    print(f"  Median: {action_stats['median']:.4f}")
    print(f"  Q25:    {action_stats['q25']:.4f}")
    print(f"  Q75:    {action_stats['q75']:.4f}")
    print(f"  Min:    {action_stats['min']:.4f}")
    print(f"  Max:    {action_stats['max']:.4f}")

    # Identify bias causes
    print(f"\nBias Analysis:")

    if bias_analysis["is_sell_biased"]:
        print("⚠️  SELL BIAS DETECTED!")
        sell_buy_ratio = bias_analysis['sell_buy_ratio']
        print(f"   SELL ratio is {sell_buy_ratio:.2f}x higher than BUY")

        # Analyze why SELL bias occurs
        sell_range_width = abs(thresholds['sell_threshold'] - action_stats['min'])
        buy_range_width = abs(action_stats['max'] - thresholds['buy_threshold'])
        hold_range_width = thresholds['buy_threshold'] - thresholds['sell_threshold']

        print(f"\nRange Analysis:")
        print(f"  SELL range width: {sell_range_width:.4f} (from {action_stats['min']:.4f} to {thresholds['sell_threshold']})")
        print(f"  HOLD range width: {hold_range_width:.4f} (from {thresholds['sell_threshold']} to {thresholds['buy_threshold']})")
        print(f"  BUY range width:  {buy_range_width:.4f} (from {thresholds['buy_threshold']} to {action_stats['max']:.4f})")

        # Check if thresholds are the problem
        if sell_range_width > buy_range_width * 1.5:
            print("🔍 CAUSE IDENTIFIED: SELL range is significantly wider than BUY range")
            print("   This makes SELL actions much more likely due to threshold asymmetry")

        # Check if mean is biased toward negative
        if action_stats['mean'] < -0.1:
            print("🔍 CAUSE IDENTIFIED: Action mean is negatively biased")
            print(f"   Mean action value ({action_stats['mean']:.4f}) favors SELL range")

        # Check if distribution is skewed
        if action_stats['median'] < action_stats['mean'] - 0.05:
            print("🔍 CAUSE IDENTIFIED: Left-skewed distribution")
            print("   More actions fall in negative (SELL) range")

    elif bias_analysis["is_buy_biased"]:
        print("⚠️  BUY BIAS DETECTED!")
        buy_sell_ratio = bias_analysis['buy_sell_ratio']
        print(f"   BUY ratio is {buy_sell_ratio:.2f}x higher than SELL")
    else:
        print("✅ No significant bias detected in action distribution")

    # Suggest fixes
    print(f"\nSuggested Fixes:")
    if bias_analysis["is_sell_biased"]:
        optimal_buy_threshold = action_stats['q75']  # 75th percentile
        optimal_sell_threshold = action_stats['q25']  # 25th percentile

        print(f"  1. Adjust thresholds for balance:")
        print(f"     BUY threshold: {thresholds['buy_threshold']} → {optimal_buy_threshold:.4f}")
        print(f"     SELL threshold: {thresholds['sell_threshold']} → {optimal_sell_threshold:.4f}")

        print(f"  2. Alternative: Make thresholds symmetric around median ({action_stats['median']:.4f})")
        symmetric_range = min(abs(action_stats['median'] - action_stats['q25']),
                             abs(action_stats['q75'] - action_stats['median']))
        print(f"     Symmetric thresholds: ±{symmetric_range:.4f}")

def main():
    """Main analysis function."""

    print("=" * 80)
    print("ULTIMATE SAC SELL BIAS INVESTIGATION")
    print("=" * 80)

    # Load model and data
    model, df = load_model_and_data()

    # Analyze action distribution
    results, actions = analyze_action_distribution(model, df, n_samples=10000)

    # Create plots
    plot_action_distribution(actions, results)

    # Investigate bias causes
    investigate_bias_causes(results, actions)

    # Save detailed results
    output_file = "results/sac_bias_ultimate_analysis.json"
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nDetailed results saved to: {output_file}")
    print("Analysis complete!")

if __name__ == "__main__":
    main()