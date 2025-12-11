import logging

from stable_baselines3 import SAC

from backtest.data_generator import generate_synthetic_data
from ztb.features.unified_feature import UnifiedFeatureEngineer as V4FeatureExtractor
from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.environment.constants import continuous_to_discrete_action
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.environment.utils.config import EnvironmentConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def setup_experiment_env(penalty_scale, steps=5000):
    """Sets up the environment with specific curriculum parameters."""

    # 1. Generate Data
    logger.info("📊 Generating synthetic market data...")
    df = generate_synthetic_data(n_periods=steps + 1000)  # Extra buffer

    # 2. Feature Engineering
    logger.info("🔧 Generating features...")
    # Create a dummy config for feature extractor
    feature_config = {"feature_set": "full"}
    feature_extractor = V4FeatureExtractor(config=feature_config)
    enhanced_df = feature_extractor.generate_features(
        df, feature_set="full", model_type="sac"
    )
    enhanced_df = enhanced_df.ffill().fillna(0)

    # 3. Environment Config
    env_config = EnvironmentConfig(
        transaction_cost=0.001,
        max_position_size=0.1,
        feature_names=list(enhanced_df.columns),
        reward_scaling=1e-4,  # Scaled for SAC stability
        max_steps=steps,
        use_continuous_actions=True,  # Enable continuous actions for SAC
        continuous_to_discrete_threshold=0.05,  # Explicitly set threshold
        curriculum_learning={
            "enabled": True,
            "stages": {
                "forced_balance": {
                    "steps": steps,  # Apply for full duration of test
                    "hold_ratio": 0.33,
                    "buy_ratio": 0.33,
                    "sell_ratio": 0.33,
                    "penalty_scale": penalty_scale,
                }
            },
        },
    )

    # 4. Create Environment
    env = HeavyTradingEnv(df=enhanced_df, config=env_config, initial_balance=1000000)

    return env


def run_experiment(name, penalty_scale, total_timesteps=5000):
    logger.info(f"\n🧪 Starting Experiment: {name} (Penalty Scale: {penalty_scale})")

    # Setup Env
    env = setup_experiment_env(penalty_scale, steps=total_timesteps)

    # Setup Model
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=10000,
        batch_size=256,
        ent_coef="auto",
        train_freq=1,
        gradient_steps=1,
        learning_starts=100,  # Start learning quickly
    )

    # Train
    logger.info(f"🏋️ Training for {total_timesteps} steps...")
    model.learn(total_timesteps=total_timesteps)

    # Evaluate (In-sample backtest on the same env for simplicity of checking behavior)
    logger.info("📝 Evaluating...")
    obs, info = env.reset()
    done = False
    total_reward = 0
    actions = []
    portfolio_values = [info.get("portfolio_value", 1000000)]

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        total_reward += reward

        # Record action
        action_val = action.item() if hasattr(action, "item") else float(action)

        # Use shared constant function for consistency
        discrete_act = continuous_to_discrete_action(action_val, threshold=0.05)
        actions.append(discrete_act)

        portfolio_values.append(info.get("portfolio_value", portfolio_values[-1]))

    # Analyze Actions
    buy_count = actions.count(ACTION_BUY)
    hold_count = actions.count(ACTION_HOLD)
    sell_count = actions.count(ACTION_SELL)
    total = len(actions)

    results = {
        "name": name,
        "penalty_scale": penalty_scale,
        "total_reward": total_reward,
        "final_return": (portfolio_values[-1] - 1000000) / 1000000 * 100,
        "action_dist": {
            "BUY": buy_count / total * 100,
            "HOLD": hold_count / total * 100,
            "SELL": sell_count / total * 100,
        },
    }

    logger.info(f"✅ Experiment {name} Completed")
    return results


if __name__ == "__main__":
    # Run AB Test
    results_a = run_experiment("A (Baseline)", penalty_scale=2.0, total_timesteps=5000)
    results_b = run_experiment("B (Relaxed)", penalty_scale=0.5, total_timesteps=5000)

    print("\n" + "=" * 60)
    print("📊 AB TEST RESULTS (5000 Steps Training)")
    print("=" * 60)

    for res in [results_a, results_b]:
        print(f"\nExperiment: {res['name']}")
        print(f"  Penalty Scale: {res['penalty_scale']}")
        print(f"  Total Reward:  {res['total_reward']:.2f}")
        print(f"  Final Return:  {res['final_return']:.2f}%")
        print(
            f"  Action Dist:   BUY={res['action_dist']['BUY']:.1f}%, HOLD={res['action_dist']['HOLD']:.1f}%, SELL={res['action_dist']['SELL']:.1f}%"
        )

    print("\n" + "=" * 60)
