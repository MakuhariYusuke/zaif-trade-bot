import sys
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from ztb.trading.environment.utils.config import RewardSettings
from ztb.utils.logging_utils import setup_logging
from ztb.training.utils.v457_config_utils import extract_env_config, load_config_dict

def run_dry_run():
    setup_logging()
    logger = logging.getLogger("v457_dry_run")
    
    config_path = project_root / "config" / "v457" / "base" / "config.yaml"
    logger.info(f"Loading config from {config_path}")
    
    try:
        config_dict = load_config_dict(config_path)
    except Exception as e:
        logger.error(f"Failed to load config {config_path}: {e}")
        return

    env_config_dict = extract_env_config(config_dict)
    
    # Load Execution Model config
    execution_config = env_config_dict.get("execution_model")
    
    # Prepare Reward Settings
    reward_settings_dict = env_config_dict.get("reward_settings", {})
    # Inject type explicitly to bypass standard loading filter if needed (though we hacked config.py)
    # The hack in config.py handles custom_reward_params
    
    reward_settings = RewardSettings.from_dict(reward_settings_dict)
    
    # Manually ensure type is passed if it got filtered out (just in case)
    if "type" in reward_settings_dict:
        if not hasattr(reward_settings, "custom_reward_params"):
            reward_settings.custom_reward_params = {}
        reward_settings.custom_reward_params["type"] = reward_settings_dict["type"]

    logger.info(f"Reward Type: {reward_settings.custom_reward_params.get('type')}")

    # Filter valid keys for EnvironmentConfig
    from ztb.trading.environment.utils.config import EnvironmentConfig
    valid_keys = {f.name for f in EnvironmentConfig.__dataclass_fields__.values()}
    
    env_kwargs = {k: v for k, v in env_config_dict.items() if k in valid_keys}
    
    # Handle aliases and manual mapping
    env_kwargs["transaction_cost"] = env_config_dict.get("transaction_cost", 0.0005)
    env_kwargs["max_position_size"] = env_config_dict.get("max_position_size", 1.0)
    env_kwargs["timeframe"] = env_config_dict.get("timeframe", "1m")
    
    # Execution model needs special handling if it's passed as object or dict
    if execution_config:
         env_kwargs["execution_model"] = execution_config

    env_config = EnvironmentConfig(**env_kwargs)
    
    # Dynamic attributes for v457 (hacks)
    if "dynamic_threshold_mode" in env_config_dict:
        env_config.dynamic_threshold_mode = env_config_dict["dynamic_threshold_mode"]
        
    # Inject reward_settings into env_config as HeavyTradingEnv expects it there
    env_config.reward_settings = reward_settings

    logger.info("Initializing HeavyTradingEnv...")
    
    try:
        import pandas as pd
        import numpy as np
        
        # Create dummy data
        dates = pd.date_range(start="2024-01-01", periods=1000, freq="1min")
        df = pd.DataFrame({
            "timestamp": dates,
            "open": np.random.rand(1000) * 100 + 1000,
            "high": np.random.rand(1000) * 100 + 1100,
            "low": np.random.rand(1000) * 100 + 900,
            "close": np.random.rand(1000) * 100 + 1000,
            "volume": np.random.rand(1000) * 100,
        })
        
        # initial_balance is usually passed as initial_portfolio_value to Env init
        initial_balance = env_config_dict.get("initial_balance", 1000000.0)

        env = HeavyTradingEnv(
            config=env_config,
            # reward_settings arg is NOT accepted by HeavyTradingEnv __init__, it uses config.reward_settings
            initial_portfolio_value=initial_balance,
            df=df 
        )
        
        # Check if correct request calculator is loaded
        logger.info(f"Reward Calculator Class: {type(env.reward_calculator).__name__}")
        
        if type(env.reward_calculator).__name__ == "V457RewardCalculator":
            logger.info("SUCCESS: V457RewardCalculator is active.")
        else:
            logger.error(f"FAILURE: Active calculator is {type(env.reward_calculator).__name__}")
            
        # Test Step
        obs, info = env.reset()
        logger.info("Environment reset.")
        
        # Take a Buy Action
        # Use simple int for discrete, array for continuous
        if hasattr(env.action_space, "n"):
            # Discrete
            action = 1 
        else:
            # Continuous
            action = np.array([1.0], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        logger.info(f"Step 1 (Buy): Reward={reward:.4f}, Info={info.keys()}")
        
        # Take a Hold Action
        if hasattr(env.action_space, "n"):
            action = 0
        else:
            action = np.array([0.0], dtype=np.float32)

        obs, reward, terminated, truncated, info = env.step(action)
        logger.info(f"Step 2 (Hold): Reward={reward:.4f}")

    except Exception as e:
        logger.exception("Dry run failed")

if __name__ == "__main__":
    run_dry_run()
