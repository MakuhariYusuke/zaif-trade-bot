import sys
sys.path.insert(0, r'c:\Users\Admin\dev\zaif-trade-bot')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
from stable_baselines3 import SAC
from ztb.trading.environment.heavy_trading_env import HeavyTradingEnv
from ztb.trading.environment.components.reward_calculator import RewardCalculator
from ztb.trading.environment.components.action_converter import ActionConverter
from ztb.trading.environment.components.observation_builder import ObservationBuilder
from ztb.trading.environment.components.portfolio_manager import PortfolioManager
from ztb.trading.environment.components.data_manager import DataManager
from ztb.trading.environment.components.position_manager import PositionManager
from ztb.trading.environment.components.risk_manager import RiskManager
from ztb.trading.environment.components.market_data_provider import MarketDataProvider
from ztb.trading.environment.components.indicator_calculator import IndicatorCalculator
from ztb.trading.environment.components.trading_rules import TradingRules
from ztb.trading.environment.components.performance_tracker import PerformanceTracker
from ztb.trading.environment.components.config_manager import ConfigManager

def analyze_sac_actions():
    """Analyze SAC model actions for balanced trading."""

    # Load config
    config_path = Path("config/sac_v414_balanced_trading_config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Initialize components
    config_manager = ConfigManager(config)
    data_manager = DataManager(config_manager)
    market_data_provider = MarketDataProvider(data_manager)
    indicator_calculator = IndicatorCalculator(market_data_provider, config_manager)
    observation_builder = ObservationBuilder(indicator_calculator, config_manager)
    portfolio_manager = PortfolioManager(config_manager)
    position_manager = PositionManager(config_manager)
    risk_manager = RiskManager(config_manager)
    reward_calculator = RewardCalculator(config_manager)
    action_converter = ActionConverter(config_manager)
    trading_rules = TradingRules(config_manager)
    performance_tracker = PerformanceTracker(config_manager)

    # Create environment
    env = HeavyTradingEnv(
        config_manager=config_manager,
        data_manager=data_manager,
        market_data_provider=market_data_provider,
        indicator_calculator=indicator_calculator,
        observation_builder=observation_builder,
        portfolio_manager=portfolio_manager,
        position_manager=position_manager,
        risk_manager=risk_manager,
        reward_calculator=reward_calculator,
        action_converter=action_converter,
        trading_rules=trading_rules,
        performance_tracker=performance_tracker
    )

    # Load trained model
    model_path = Path("models/sac_v414_balanced_trading")
    model = SAC.load(str(model_path))

    # Sample actions
    num_samples = 5000
    continuous_actions = []
    discrete_actions = []

    obs = env.reset()
    for _ in range(num_samples):
        action, _ = model.predict(obs, deterministic=True)
        continuous_actions.append(action[0])

        # Convert to discrete using threshold
        threshold = config_manager.get_setting_float("continuous_to_discrete_threshold", 0.1)
        if action[0] > threshold:
            discrete_action = 1  # BUY
        elif action[0] < -threshold:
            discrete_action = 2  # SELL
        else:
            discrete_action = 0  # HOLD
        discrete_actions.append(discrete_action)

        # Step environment
        obs, _, done, _ = env.step(action)
        if done:
            obs = env.reset()

    # Analyze continuous actions
    continuous_actions = np.array(continuous_actions)
    print(f"Continuous Actions Statistics:")
    print(f"Mean: {continuous_actions.mean():.4f}")
    print(f"Std: {continuous_actions.std():.4f}")
    print(f"Min: {continuous_actions.min():.4f}")
    print(f"Max: {continuous_actions.max():.4f}")

    # Analyze discrete actions
    discrete_actions = np.array(discrete_actions)
    unique, counts = np.unique(discrete_actions, return_counts=True)
    action_dist = dict(zip(unique, counts))
    total = sum(counts)

    print(f"\nDiscrete Actions Distribution ({total} samples):")
    for action_code, count in action_dist.items():
        if action_code == 0:
            action_name = "HOLD"
        elif action_code == 1:
            action_name = "BUY"
        elif action_code == 2:
            action_name = "SELL"
        else:
            action_name = f"UNKNOWN_{action_code}"
        percentage = (count / total) * 100
        print(f"{action_name}: {count} ({percentage:.1f}%)")

    # Save results
    results = {
        "continuous_stats": {
            "mean": float(continuous_actions.mean()),
            "std": float(continuous_actions.std()),
            "min": float(continuous_actions.min()),
            "max": float(continuous_actions.max())
        },
        "discrete_distribution": {
            "HOLD": int(action_dist.get(0, 0)),
            "BUY": int(action_dist.get(1, 0)),
            "SELL": int(action_dist.get(2, 0)),
            "total_samples": total
        },
        "percentages": {
            "HOLD": (action_dist.get(0, 0) / total) * 100,
            "BUY": (action_dist.get(1, 0) / total) * 100,
            "SELL": (action_dist.get(2, 0) / total) * 100
        }
    }

    with open("sac_v414_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Plot continuous action distribution
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.hist(continuous_actions, bins=50, alpha=0.7, edgecolor='black')
    plt.axvline(x=threshold, color='red', linestyle='--', label=f'BUY threshold ({threshold})')
    plt.axvline(x=-threshold, color='blue', linestyle='--', label=f'SELL threshold ({-threshold})')
    plt.xlabel('Continuous Action Value')
    plt.ylabel('Frequency')
    plt.title('Continuous Action Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    labels = ['HOLD', 'BUY', 'SELL']
    sizes = [action_dist.get(0, 0), action_dist.get(1, 0), action_dist.get(2, 0)]
    colors = ['gray', 'green', 'red']
    plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.title('Discrete Action Distribution')
    plt.axis('equal')

    plt.tight_layout()
    plt.savefig('sac_v414_action_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\nResults saved to sac_v414_analysis_results.json")
    print(f"Plot saved to sac_v414_action_analysis.png")

if __name__ == "__main__":
    analyze_sac_actions()