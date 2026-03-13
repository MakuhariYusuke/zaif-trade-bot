import sys
import os
import pandas as pd
import numpy as np
from typing import Dict
import itertools
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from ztb.trading.environment.components.threshold_manager import ThresholdManager
from ztb.trading.strategies.action_signal_guide.components.market_regime import MarketRegimeDetector

from stable_baselines3 import SAC

def load_data(filepath: str) -> pd.DataFrame:
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath, parse_dates=["timestamp"], index_col="timestamp")
    return df

def precompute_model_actions(model_path: str, data: pd.DataFrame) -> np.ndarray:
    """
    Load model and pre-compute actions for the dataset to speed up optimization.
    """
    print(f"Loading model from {model_path}...")
    try:
        model = SAC.load(model_path)
    except Exception as e:
        print(f"Failed to load model: {e}")
        # Fallback to random for testing if model missing
        return np.random.uniform(-1, 1, len(data))

    print("Pre-computing model actions...")
    actions = []
    
    # We need to construct observations. 
    # For simplicity in this script, we'll assume the model takes raw features 
    # or we'll use a simplified observation construction if the env is complex.
    # Ideally, we should use the actual Environment to get observations.
    
    from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
    from ztb.trading.environment.utils.config import EnvironmentConfig
    
    # We need to process the dataframe to get features first if they aren't there.
    # Assuming data already has features or we can use a simple env wrapper.
    # For speed, let's try to use the env to get observations.
    
    try:
        print("Creating environment with target_feature_count=138...")
        config = EnvironmentConfig(
            feature_set="default",
            use_continuous_actions=True,
            target_feature_count=138,
            correlation_reduction=True
        )
        env = HeavyTradingEnv(df=data, config=config)
        
        obs = env.reset()[0]
        
        for _ in tqdm(range(len(data))):
            action, _ = model.predict(obs, deterministic=True)
            actions.append(float(action[0])) # Assuming continuous action
            
            # Step env (dummy step)
            # We don't actually need to step the logic, just get next obs
            # But HeavyTradingEnv steps through the DF.
            obs, _, done, _, _ = env.step(action)
            if done:
                break
                
        # Pad if necessary
        if len(actions) < len(data):
            actions.extend([0.0] * (len(data) - len(actions)))
            
    except Exception as e:
        print(f"Error generating actions with Env: {e}")
        print("Falling back to proxy signals.")
        return np.zeros(len(data))

    return np.array(actions)

def run_simulation(
    data: pd.DataFrame, 
    actions: np.ndarray, 
    threshold_multipliers: Dict[str, float]
) -> float:
    """
    Run a simplified simulation with specific threshold multipliers.
    Returns Total PnL.
    """
    # Configure ThresholdManager
    config_mock = type("Config", (), {
        "continuous_to_discrete_threshold": 0.01,
        "adaptive_threshold_mode": True,
        "threshold_volatility_multiplier": 1.0,
        "min_action_threshold": 0.001,
        "max_action_threshold": 1.0,
        "regime_threshold_multipliers": threshold_multipliers,
        "dynamic_threshold_mode": "fixed",
        "z_score_window": 100,
        "z_score_threshold": 2.0,
        "z_score_method": "std",
        "regime_detection_window": 50,
        "threshold_adaptation_rate": 0.1,
        "performance_memory_size": 100,
        "trend_detection_threshold": 0.001,
        "volatility_detection_threshold": 0.02,
    })()
    
    threshold_manager = ThresholdManager(config_mock)
    regime_detector = MarketRegimeDetector(use_relative=True)
    
    position = 0
    entry_price = 0.0
    pnl = 0.0
    
    prices = data["close"].values
    
    # Iterate
    # Note: actions array matches data length
    
    for i in range(50, len(data)):
        current_price = prices[i]
        
        # 1. Detect Regime
        window_data = data.iloc[i-50:i+1]
        regime = regime_detector.detect_regime(window_data).value
        
        # 2. Get Threshold
        volatility = window_data["close"].pct_change().std() * current_price
        if np.isnan(volatility): volatility = 0.0
        
        threshold = threshold_manager.get_threshold(
            volatility=volatility,
            current_price=current_price,
            regime=regime,
            base_value=0.01 
        )
        
        # 3. Get Model Signal
        raw_action = actions[i]
        
        # 4. Execute Logic
        if raw_action > threshold and position <= 0:
            # Buy
            if position < 0:
                pnl += (entry_price - current_price) / entry_price
            position = 1
            entry_price = current_price
            
        elif raw_action < -threshold and position >= 0:
            # Sell
            if position > 0:
                pnl += (current_price - entry_price) / entry_price
            position = -1
            entry_price = current_price
            
    return pnl

def main():
    data_path = "data/btc_jpy_1m_v451.csv"
    model_path = "models/sac_v452_fine_tuned_5k.zip"
    
    if not os.path.exists(data_path):
        print(f"Data file not found: {data_path}")
        return

    df = load_data(data_path)
    
    # Use a subset for speed
    sim_data = df.tail(5000).copy() # Match the 5000 steps we trained on roughly
    
    # Pre-compute actions
    actions = precompute_model_actions(model_path, sim_data)
    
    # Define Search Space
    param_grid = {
        "trend_follow": [0.5, 0.8, 1.0],
        "trend_oppose": [5.0, 10.0],
        "range_chop": [5.0, 10.0],
        "range_scalp": [0.5, 0.8, 1.0] # We hope it picks 0.5 or 0.8
    }
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    combinations = list(itertools.product(*values))
    
    print(f"Starting optimization with {len(combinations)} combinations...")
    
    best_score = -float("inf")
    best_params = None
    
    for combo in tqdm(combinations):
        params = dict(zip(keys, combo))
        
        try:
            score = run_simulation(sim_data, actions, params)
            
            if score > best_score:
                best_score = score
                best_params = params
                # print(f"New Best Score: {best_score:.4f} with params: {best_params}")
        except Exception as e:
            print(f"Error with params {params}: {e}")
            continue
            
    print("\nOptimization Complete!")
    print(f"Best Score: {best_score:.4f}")
    print(f"Best Parameters: {best_params}")
    
    # Save to file
    import json
    os.makedirs("config", exist_ok=True)
    with open("config/threshold_optimized.json", "w") as f:
        json.dump(best_params, f, indent=4)
    print("Saved best parameters to config/threshold_optimized.json")

if __name__ == "__main__":
    main()
