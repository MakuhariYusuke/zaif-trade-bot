import logging

import numpy as np
import pandas as pd
from stable_baselines3 import SAC

from backtest.data_generator import generate_synthetic_data
from ztb.features.unified_feature import UnifiedFeatureEngineer as V4FeatureExtractor
from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.training.environments.heavy_trading_env import HeavyTradingEnv

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_experiment_env(adaptive_mode, volatility_multiplier=1.0, steps=5000):
    """Sets up the environment with specific threshold parameters."""

    # 1. Generate Data (High Volatility to test adaptive threshold)
    logger.info("📊 Generating synthetic market data (High Volatility)...")
    # Using higher volatility to make the adaptive threshold work harder
    df = generate_synthetic_data(n_periods=steps + 1000, volatility=1000)

    # 2. Feature Engineering
    logger.info("🔧 Generating features...")
    feature_config = {"feature_set": "full"}
    feature_extractor = V4FeatureExtractor(config=feature_config)
    enhanced_df = feature_extractor.generate_features(
        df, feature_set="full", model_type="sac"
    )

    # Manually add ATR-14 for ThresholdManager if missing
    if "atr_14" not in enhanced_df.columns:
        high = enhanced_df["high"]
        low = enhanced_df["low"]
        close = enhanced_df["close"]
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        enhanced_df["atr_14"] = tr.rolling(window=14).mean().fillna(0)
        logger.info("✅ Added calculated atr_14 column")

    enhanced_df = enhanced_df.ffill().fillna(0)

    # Debug: Check for volatility columns
    vol_cols = [
        c
        for c in enhanced_df.columns
        if "atr" in c.lower() or "volatility" in c.lower()
    ]
    logger.info(f"Found volatility columns: {vol_cols}")

    # 3. Environment Config
    env_config = EnvironmentConfig(
        transaction_cost=0.001,
        max_position_size=0.1,
        feature_names=list(enhanced_df.columns),
        reward_scaling=1e-4,
        max_steps=steps,
        use_continuous_actions=True,
        # Threshold Settings
        continuous_to_discrete_threshold=0.01,  # Base threshold
        adaptive_threshold_mode=adaptive_mode,
        threshold_volatility_multiplier=volatility_multiplier,
        min_action_threshold=0.001,
        max_action_threshold=0.05,
        initial_portfolio_value=1000000,
    )

    # 4. Create Environment
    env = HeavyTradingEnv(
        data=enhanced_df, config=env_config, feature_columns=list(enhanced_df.columns)
    )

    return env


def run_experiment(
    name, adaptive_mode, volatility_multiplier=1.0, total_timesteps=5000
):
    logger.info(
        f"\n🧪 Starting Experiment: {name} (Adaptive: {adaptive_mode}, Multiplier: {volatility_multiplier})"
    )

    # Setup Env
    env = setup_experiment_env(
        adaptive_mode, volatility_multiplier, steps=total_timesteps
    )

    # Setup Model
    model = SAC(
        "MlpPolicy",
        env,
        verbose=0,  # Reduce noise
        learning_rate=3e-4,
        buffer_size=10000,
        batch_size=256,
        ent_coef="auto",
        train_freq=1,
        gradient_steps=1,
        learning_starts=100,
    )

    # Train
    logger.info(f"🏋️ Training for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps)

    # Evaluate
    logger.info("📝 Evaluating...")
    obs, info = env.reset()
    done = False
    total_reward = 0
    actions = []
    thresholds = []

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        # Record action type
        # Note: We need to manually convert to discrete to check what happened,
        # or rely on info if we added it there.
        # HeavyTradingEnv doesn't return the discrete action in info by default,
        # but we can infer it from position changes or just log the raw action.

        # Let's use the helper to see what it WOULD be
        # But wait, the env uses the dynamic threshold internally.
        # We can check 'threshold_suppressed_actions' in info if available,
        # or just look at the raw action vs the threshold if we knew it.

        # Fortunately, we added 'action_threshold' to the env instance,
        # so we can access it if we have the env instance.
        # But here we are outside the step.

        # Let's just record the raw action and the threshold used (if we can get it)
        current_threshold = env.action_threshold
        thresholds.append(current_threshold)

        # Determine discrete action
        raw_action = float(action[0])
        if abs(raw_action) < current_threshold:
            act_type = "HOLD"
        elif raw_action > 0:
            act_type = "BUY"
        else:
            act_type = "SELL"
        actions.append(act_type)

    # Analyze Results
    action_counts = pd.Series(actions).value_counts()
    hold_ratio = action_counts.get("HOLD", 0) / len(actions)
    buy_ratio = action_counts.get("BUY", 0) / len(actions)
    sell_ratio = action_counts.get("SELL", 0) / len(actions)

    avg_threshold = np.mean(thresholds)

    logger.info(f"📊 Results for {name}:")
    logger.info(f"  Total Reward: {total_reward:.2f}")
    logger.info(f"  Final Balance: {env.balance:.2f}")
    logger.info(f"  Trades: {env.trades_count}")
    logger.info(
        f"  Action Dist: HOLD={hold_ratio:.2%}, BUY={buy_ratio:.2%}, SELL={sell_ratio:.2%}"
    )
    logger.info(f"  Avg Threshold: {avg_threshold:.4f}")

    return {
        "name": name,
        "total_reward": total_reward,
        "final_balance": env.balance,
        "trades": env.trades_count,
        "hold_ratio": hold_ratio,
        "avg_threshold": avg_threshold,
    }


if __name__ == "__main__":
    # 1. Baseline (Static Threshold)
    result_baseline = run_experiment(
        name="Baseline (Static 0.01)", adaptive_mode=False, total_timesteps=3000
    )

    # 2. Adaptive (Dynamic Threshold)
    result_adaptive = run_experiment(
        name="Adaptive (Multiplier 1.0)",
        adaptive_mode=True,
        volatility_multiplier=1.0,
        total_timesteps=3000,
    )

    # 3. Adaptive High Sensitivity
    result_adaptive_high = run_experiment(
        name="Adaptive (Multiplier 2.0)",
        adaptive_mode=True,
        volatility_multiplier=2.0,
        total_timesteps=3000,
    )

    print("\n" + "=" * 50)
    print("🏆 FINAL COMPARISON")
    print("=" * 50)

    results = [result_baseline, result_adaptive, result_adaptive_high]
    df_res = pd.DataFrame(results)
    print(df_res.to_string(index=False))
