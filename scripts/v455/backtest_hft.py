import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from ztb.features.hft_proxies import add_hft_features
from ztb.trading.environment.fast_intraday_env import FastIntradayEnv
from ztb.utils.logging_utils import setup_logging, get_logger
from ztb.utils import format_number

setup_logging()
logger = get_logger(__name__)

class SequentialFastIntradayEnv(FastIntradayEnv):
    """
    Subclass of FastIntradayEnv that allows setting the start step
    for sequential backtesting.
    """
    def set_start_step(self, step: int):
        self.force_start_step = step
        
    def reset(self, seed=None, options=None):
        # Call parent reset to initialize basics
        obs, info = super().reset(seed=seed, options=options)
        
        # Override current_step if forced
        if hasattr(self, 'force_start_step'):
            self.current_step = self.force_start_step
            # Re-do prewarm with correct data
            self.scaler.n = 0
            self.scaler.mean[:] = 0
            self.scaler.M2[:] = 0
            self.scaler.var[:] = 1
            
            # Feed prewarm data
            start_idx = max(0, self.current_step - self.prewarm_steps)
            prewarm_data = self.features_data[start_idx : self.current_step]
            self.scaler.batch_update(prewarm_data)
            
            # Re-get observation
            return self._get_observation(), {}
            
        return obs, info

def main():
    # Configuration
    DATA_PATH = "data/btc_jpy_1m_v454.csv"
    MODEL_PATH = "models/v455_hft_main/sac_hft_final.zip"
    VEC_NORM_PATH = "models/v455_hft_main/vec_normalize.pkl"
    OUTPUT_DIR = "backtest_results/v455_hft_main"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load Data
    logger.info(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH, parse_dates=True, index_col=0)
    df = add_hft_features(df)
    
    feature_columns = ["clv", "vol_pressure", "impact_proxy", "vol_regime", "trend_persistence"]
    
    # Create Environment
    # Must match training config exactly
    def make_env():
        env = SequentialFastIntradayEnv(
            df=df,
            feature_columns=feature_columns,
            initial_balance=100_000.0,
            max_position=0.01,
            max_steps=1000000, # Large number to cover full dataset if needed, or we chunk it
            prewarm_steps=100,
            max_ttl_steps=60,
            cooldown_steps=5,
            reward_params={
                "alpha": 0.5,
                "beta": 0.02,
                "min_edge_mult": 1.5,
                "edge_penalty_rate": 1.0,
                "vol_floor": 0.002,
                "vol_floor_penalty": 50.0,
                "hold_grace": 10,
                "hold_ramp": 0.01
            }
        )
        return env

    # We need to wrap in DummyVecEnv and VecNormalize to match training
    # But we need to load the statistics from training!
    env = DummyVecEnv([make_env])
    env = VecNormalize.load(VEC_NORM_PATH, env)
    env.training = False # Don't update stats during backtest
    env.norm_reward = False # Don't normalize rewards for backtest reporting
    
    # Load Model
    logger.info(f"Loading model from {MODEL_PATH}...")
    model = SAC.load(MODEL_PATH, env=env)
    
    # Run Backtest
    # We will run one long continuous episode covering the last 20% of data (Test Set equivalent)
    # Or just run the whole thing? Let's run the last 10,000 steps as a sample.
    # The user asked for "Backtest", usually implies unseen data.
    # Let's pick a range.
    
    total_len = len(df)
    test_len = 20000 # Last 20k minutes (~14 days)
    start_step = total_len - test_len
    
    logger.info(f"Starting backtest from step {start_step} to {total_len} ({test_len} steps)...")
    
    # Access the inner env to set start step
    env.envs[0].set_start_step(start_step)
    env.envs[0].max_steps = test_len # Limit episode length
    
    obs = env.reset()
    
    history = {
        "balance": [],
        "position": [],
        "price": [],
        "action_target": [],
        "pnl": [],
        "trade_cost": []
    }
    
    done = False
    step_count = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        info_dict = info[0]
        history["balance"].append(info_dict["balance"])
        history["position"].append(info_dict["position"])
        history["price"].append(env.envs[0].close_prices[env.envs[0].current_step - 1])
        history["action_target"].append(action[0][0]) # Raw action
        history["pnl"].append(info_dict.get("pnl", 0)) # Need to ensure pnl is in info?
        # FastIntradayEnv info doesn't have raw PnL per step explicitly named 'pnl' usually, 
        # but it has balance change.
        # Let's rely on balance.
        history["trade_cost"].append(info_dict.get("trade_cost", 0))
        
        step_count += 1
        if step_count % 1000 == 0:
            print(f"Step {step_count}/{test_len} | Balance: {info_dict['balance']:.0f}")
            
    # Analysis
    df_res = pd.DataFrame(history)
    df_res["cum_pnl"] = df_res["balance"] - df_res["balance"].iloc[0]
    
    final_balance = df_res["balance"].iloc[-1]
    initial_balance = df_res["balance"].iloc[0]
    profit = final_balance - initial_balance
    ret = profit / initial_balance
    
    logger.info(f"Backtest Complete.")
    logger.info(f"Initial Balance: {initial_balance:,.0f}")
    logger.info(f"Final Balance: {final_balance:,.0f}")
    logger.info(f"Profit: {profit:,.0f} ({ret:.2%})")
    
    # Plot
    plt.figure(figsize=(12, 8))
    
    plt.subplot(3, 1, 1)
    plt.plot(df_res["balance"], label="Balance")
    plt.title(f"Balance (Profit: {ret:.2%})")
    plt.grid(True)
    
    plt.subplot(3, 1, 2)
    plt.plot(df_res["price"], label="Price", color="gray", alpha=0.5)
    plt.twinx()
    plt.plot(df_res["position"], label="Position", color="orange", alpha=0.8)
    plt.title("Price vs Position")
    
    plt.subplot(3, 1, 3)
    plt.plot(df_res["trade_cost"].cumsum(), label="Cum Trade Cost", color="red")
    plt.title("Cumulative Trade Cost")
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "backtest_summary.png"))
    logger.info(f"Plot saved to {OUTPUT_DIR}/backtest_summary.png")
    
    # Save CSV
    df_res.to_csv(os.path.join(OUTPUT_DIR, "backtest_timeseries.csv"))

if __name__ == "__main__":
    main()
