#!/usr/bin/env python3
"""
SAC v438 - Enhanced Short Position Rewards for Bear Market Performance

Improves SAC v437 by addressing bear market trading weaknesses:
- Increases short position reward multipliers
- Reduces short position penalty multipliers
- Adds bear market specific features
- Balances long/short position incentives
"""

import json
import os
from pathlib import Path
from typing import Any, Dict


def create_v438_reward_config():
    """Create enhanced reward configuration for better bear market performance."""

    # Base v437 config with improvements
    reward_config = {
        "reward_function": {
            # Enhanced asymmetric scaling for better bear market performance
            "long_position_reward_multiplier": 1.3,    # Reduced from 1.5 (was too aggressive)
            "short_position_reward_multiplier": 1.1,   # Increased from 0.7 (major improvement)
            "long_position_penalty_multiplier": 0.9,   # Increased from 0.8 (slight penalty increase)
            "short_position_penalty_multiplier": 0.95, # Decreased from 1.2 (major penalty reduction)

            # Enhanced profit/loss bonuses
            "profit_bonus_multiplier": 1.2,
            "loss_penalty_multiplier": 0.8,

            # Bear market specific incentives
            "bear_market_bonus": 0.1,  # Additional reward for correct short positions in downtrends
            "bull_market_bonus": 0.05, # Reduced bull market bonus for balance

            # Risk management
            "max_drawdown_penalty": 0.3,
            "volatility_penalty": 0.1,

            # Position size incentives
            "optimal_position_bonus": 0.05,
            "over_position_penalty": 0.1,
            "under_position_penalty": 0.05
        },

        "training": {
            "total_timesteps": 500000,
            "learning_rate": 3e-4,
            "batch_size": 256,
            "buffer_size": 1000000,
            "gamma": 0.99,
            "tau": 0.005,
            "ent_coef": 0.01,
            "target_update_interval": 1
        },

        "features": {
            # Enhanced bear market features
            "bear_market_indicators": [
                "rsi_bearish_divergence",
                "macd_bearish_crossover",
                "volume_bearish_confirmation",
                "trend_bearish_strength",
                "momentum_bearish_acceleration"
            ],

            # Market regime awareness
            "regime_features": [
                "market_regime_bear",
                "market_regime_bull",
                "market_regime_sideways",
                "regime_confidence_score"
            ]
        }
    }

    return reward_config

def create_v438_feature_config():
    """Create enhanced feature configuration with bear market focus."""

    feature_config = {
        "feature_sets": {
            "v438_enhanced": {
                "description": "Enhanced features for SAC v438 with bear market focus",

                "technical_indicators": {
                    # Enhanced RSI features for bear markets
                    "rsi_bearish_divergence": {
                        "enabled": True,
                        "description": "RSI bearish divergence detection"
                    },

                    # Enhanced MACD for bear signals
                    "macd_bearish_crossover": {
                        "enabled": True,
                        "description": "MACD bearish crossover signals"
                    },

                    # Volume confirmation for bear moves
                    "volume_bearish_confirmation": {
                        "enabled": True,
                        "description": "Volume confirmation for bearish moves"
                    },

                    # Trend strength for bear markets
                    "trend_bearish_strength": {
                        "enabled": True,
                        "description": "Bear trend strength indicators"
                    }
                },

                "market_regime": {
                    "bear_market_detection": {
                        "enabled": True,
                        "thresholds": {
                            "strong_bear": -0.02,  # 2% daily decline
                            "moderate_bear": -0.01, # 1% daily decline
                            "weak_bear": -0.005    # 0.5% daily decline
                        }
                    },

                    "regime_persistence": {
                        "enabled": True,
                        "lookback_periods": [5, 10, 20]
                    }
                },

                "risk_adjusted_features": {
                    "bear_market_risk_adjustment": {
                        "enabled": True,
                        "description": "Risk-adjusted features for bear markets"
                    }
                }
            }
        }
    }

    return feature_config

def save_v438_configs():
    """Save v438 configuration files."""

    # Create config directory if it doesn't exist
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)

    # Save reward config
    reward_config = create_v438_reward_config()
    reward_path = config_dir / "sac_v438_reward_config.json"

    with open(reward_path, 'w', encoding='utf-8') as f:
        json.dump(reward_config, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved v438 reward config: {reward_path}")

    # Save feature config
    feature_config = create_v438_feature_config()
    feature_path = config_dir / "sac_v438_feature_config.json"

    with open(feature_path, 'w', encoding='utf-8') as f:
        json.dump(feature_config, f, indent=2, ensure_ascii=False)

    print(f"✅ Saved v438 feature config: {feature_path}")

    return reward_path, feature_path

def create_v438_training_script():
    """Create training script for v438."""

    training_script = '''#!/usr/bin/env python3
"""
SAC v438 Training - Enhanced Bear Market Performance

Trains SAC v438 with improved reward function for better bear market trading.
"""

import json
import os
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ztb.training.sac_trainer import SACTrainer
from ztb.trading.environment.heavy_trading_env import HeavyTradingEnv
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

def load_v438_config():
    """Load v438 configuration."""
    config_dir = Path("config")

    # Load reward config
    reward_config_path = config_dir / "sac_v438_reward_config.json"
    with open(reward_config_path, 'r', encoding='utf-8') as f:
        reward_config = json.load(f)

    # Load feature config
    feature_config_path = config_dir / "sac_v438_feature_config.json"
    with open(feature_config_path, 'r', encoding='utf-8') as f:
        feature_config = json.load(f)

    return reward_config, feature_config

def create_v438_environment(reward_config, feature_config):
    """Create HeavyTradingEnv with v438 enhancements."""

    env_config = {
        "reward_function": reward_config["reward_function"],
        "features": feature_config["feature_sets"]["v438_enhanced"],
        "training_mode": True,
        "max_episode_length": 5000,
        "initial_balance": 100000,
        "transaction_fee": 0.001,  # 0.1%
        "slippage": 0.0005,  # 0.05%
    }

    env = HeavyTradingEnv(config=env_config)
    return env

def train_v438():
    """Train SAC v438 model."""

    logger.info("🚀 Starting SAC v438 training for enhanced bear market performance")

    # Load configurations
    reward_config, feature_config = load_v438_config()

    # Create environment
    env = create_v438_environment(reward_config, feature_config)

    # Training configuration
    training_config = reward_config["training"]

    # Create trainer
    trainer = SACTrainer(
        env=env,
        model_name="sac_v438_bear_enhanced",
        **training_config
    )

    # Train model
    logger.info("🎯 Training SAC v438 with enhanced bear market rewards...")
    trainer.train()

    # Save model
    model_path = trainer.save_model()
    logger.info(f"💾 Model saved to: {model_path}")

    # Evaluate model
    logger.info("📊 Evaluating trained model...")
    eval_results = trainer.evaluate(episodes=10)

    logger.info("✅ SAC v438 training completed!")
    logger.info(f"📈 Evaluation results: {eval_results}")

    return model_path, eval_results

if __name__ == "__main__":
    train_v438()
'''

    script_path = Path("train_sac_v438.py")
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(training_script)

    print(f"✅ Created v438 training script: {script_path}")
    return script_path

def create_v438_backtest_script():
    """Create backtest script for v438."""

    backtest_script = '''#!/usr/bin/env python3
"""
SAC v438 Backtest - Enhanced Bear Market Performance Test

Backtests SAC v438 model with improved bear market handling.
"""

import json
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ztb.backtest.sac_backtester import SACBacktester
from ztb.trading.environment.heavy_trading_env import HeavyTradingEnv
from ztb.utils.logging_utils import get_logger

logger = get_logger(__name__)

def load_v438_config():
    """Load v438 configuration."""
    config_dir = Path("config")

    # Load reward config
    reward_config_path = config_dir / "sac_v438_reward_config.json"
    with open(reward_config_path, 'r', encoding='utf-8') as f:
        reward_config = json.load(f)

    return reward_config

def run_v438_backtest():
    """Run backtest for SAC v438."""

    logger.info("🔍 Running SAC v438 backtest with bear market enhancements")

    # Load configuration
    reward_config = load_v438_config()

    # Model path (update this with actual trained model path)
    model_path = "models/sac_v438_bear_enhanced.zip"

    if not Path(model_path).exists():
        logger.error(f"Model not found: {model_path}")
        logger.info("Please train the model first using train_sac_v438.py")
        return None

    # Create backtester
    backtester = SACBacktester(
        model_path=model_path,
        reward_config=reward_config["reward_function"]
    )

    # Run backtest
    logger.info("📈 Running backtest...")
    results = backtester.run_backtest()

    # Analyze bear market performance specifically
    bear_performance = analyze_bear_market_performance(results)

    # Save results
    results_dir = Path("backtest_results") / "v438_bear_enhanced"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save main results
    results_file = results_dir / "backtest_results.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Save bear market analysis
    bear_file = results_dir / "bear_market_analysis.json"
    with open(bear_file, 'w', encoding='utf-8') as f:
        json.dump(bear_performance, f, indent=2, ensure_ascii=False)

    logger.info("✅ SAC v438 backtest completed!")
    logger.info(f"📁 Results saved to: {results_dir}")
    logger.info(f"🐻 Bear market analysis: {bear_performance}")

    return results, bear_performance

def analyze_bear_market_performance(results):
    """Analyze performance specifically in bear market conditions."""

    # This is a simplified analysis - in practice you'd need market regime detection
    analysis = {
        "short_position_win_rate": 0.0,
        "bear_market_adaptation_score": 0.0,
        "reward_function_effectiveness": 0.0,
        "recommendations": []
    }

    # Calculate short position performance
    if "trades" in results:
        trades_df = pd.DataFrame(results["trades"])
        short_trades = trades_df[trades_df["position"] < 0]
        if len(short_trades) > 0:
            short_wins = len(short_trades[short_trades["reward"] > 0])
            analysis["short_position_win_rate"] = short_wins / len(short_trades)

    # Basic scoring (would be more sophisticated in practice)
    analysis["bear_market_adaptation_score"] = min(1.0, analysis["short_position_win_rate"] * 1.2)
    analysis["reward_function_effectiveness"] = analysis["bear_market_adaptation_score"]

    # Generate recommendations
    if analysis["short_position_win_rate"] < 0.4:
        analysis["recommendations"].append("Further increase short position reward multipliers")
    if analysis["short_position_win_rate"] > 0.6:
        analysis["recommendations"].append("Bear market adaptation successful - consider production deployment")

    return analysis

if __name__ == "__main__":
    run_v438_backtest()
'''

    script_path = Path("backtest_sac_v438.py")
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(backtest_script)

    print(f"✅ Created v438 backtest script: {script_path}")
    return script_path

def main():
    """Main function to create v438 enhancement package."""

    print("🐻 SAC v438 - Enhanced Bear Market Performance Setup")
    print("=" * 60)

    # Create configuration files
    print("📝 Creating configuration files...")
    reward_config_path, feature_config_path = save_v438_configs()

    # Create training script
    print("🎯 Creating training script...")
    training_script_path = create_v438_training_script()

    # Create backtest script
    print("🔍 Creating backtest script...")
    backtest_script_path = create_v438_backtest_script()

    print("\n✅ SAC v438 enhancement package created successfully!")
    print("\n📋 Next steps:")
    print("1. Review and adjust reward multipliers in config/sac_v438_reward_config.json")
    print("2. Run training: python train_sac_v438.py")
    print("3. Run backtest: python backtest_sac_v438.py")
    print("4. Compare results with v437 baseline")

    print("\n🎯 Key improvements in v438:")
    print("• Short position reward multiplier: 0.7 → 1.1 (57% increase)")
    print("• Short position penalty multiplier: 1.2 → 0.95 (21% reduction)")
    print("• Added bear market specific features and incentives")
    print("• Balanced long/short position incentives")

if __name__ == "__main__":
    main()
'''

    script_path = Path("create_sac_v438_enhancement.py")
    with open(script_path, 'w', encoding='utf-8') as f:
        f.write(script_content)

    print(f"✅ Created v438 enhancement setup script: {script_path}")
    return script_path

def main():
    """Main function to create v438 enhancement package."""

    print("🐻 SAC v438 - Enhanced Bear Market Performance Setup")
    print("=" * 60)

    # Create configuration files
    print("📝 Creating configuration files...")
    reward_config_path, feature_config_path = save_v438_configs()

    # Create training script
    print("🎯 Creating training script...")
    training_script_path = create_v438_training_script()

    # Create backtest script
    print("🔍 Creating backtest script...")
    backtest_script_path = create_v438_backtest_script()

    # Create main setup script
    print("🚀 Creating main setup script...")
    setup_script_path = create_v438_setup_script()

    print("\n✅ SAC v438 enhancement package created successfully!")
    print("\n📋 Next steps:")
    print("1. Run: python create_sac_v438_enhancement.py")
    print("2. Review configurations in config/ directory")
    print("3. Train model: python train_sac_v438.py")
    print("4. Backtest: python backtest_sac_v438.py")

    print("\n🎯 Key improvements in v438:")
    print("• Short position reward multiplier: 0.7 → 1.1 (+57%)")
    print("• Short position penalty multiplier: 1.2 → 0.95 (-21%)")
    print("• Added bear market specific features")
    print("• Balanced long/short incentives for better market adaptation")

if __name__ == "__main__":
    main()
