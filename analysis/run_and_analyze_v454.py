#!/usr/bin/env python3
"""
Run and Analyze SAC v454 Backtest
Runs a detailed backtest for SAC v454 and generates analysis reports and plots.
"""
import json
import logging
import sys
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from stable_baselines3 import SAC

# Add project root to path
project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

# Suppress warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="pandas")
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

from ztb.trading.constants import ACTION_BUY, ACTION_HOLD, ACTION_SELL
from ztb.trading.environment.constants import continuous_to_discrete_action
from ztb.utils.logging_utils import setup_logging
from ztb.config.unified_config import UnifiedConfig
from ztb.trading.environment.utils.config import EnvironmentConfig
from ztb.trading.environment.heavy_env.core import HeavyTradingEnv
from backtest.data_generator import generate_synthetic_data

# Setup logging
setup_logging(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_detailed_backtest(model_name, config_path, output_dir):
    """Runs a detailed backtest and saves per-step data."""
    logger.info(f"🚀 Starting Detailed Backtest for {model_name}")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load Config
    try:
        unified_config = UnifiedConfig.from_file(config_path)
        config = unified_config.to_dict()
        logger.info("✅ Config loaded")
    except Exception as e:
        logger.error(f"❌ Failed to load config: {e}")
        return None

    # Load Data
    data_file = "data/btc_jpy_1m_v454.csv"
    if not Path(data_file).exists():
        logger.warning(f"Data file {data_file} not found, using synthetic data")
        data_file = "data/btc_jpy_real_dataset.csv"
        if not Path(data_file).exists():
             synthetic_df = generate_synthetic_data(n_periods=5000)
             synthetic_df.to_csv(data_file)
    
    df = pd.read_csv(data_file)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    logger.info(f"✅ Data loaded: {len(df)} rows")

    # Prepare Features
    if "v454" in data_file:
        exclude_cols = ['timestamp', 'date', 'time']
        available_features = [col for col in df.columns if col not in exclude_cols]
        featured_df = df.copy()
        logger.info(f"✅ Using v454 features: {len(available_features)} columns")
    else:
        # Fallback for other datasets
        featured_df = df.copy()

    # Load Model
    model_path = f"models/{model_name}.zip"
    logger.info(f"Loading model from {model_path}")
    try:
        model = SAC.load(model_path)
        logger.info(f"✅ Model loaded")
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        return None

    # Setup Environment
    env_config = config.get("environment", {})
    env_config_obj = EnvironmentConfig.from_dict(env_config)
    
    reward_settings = config.get("reward_settings", {})
    if reward_settings:
        env_config_obj.reward_settings = reward_settings
    
    # Force target feature count to match model observation space (166)
    env_config_obj.target_feature_count = 166

    env = HeavyTradingEnv(
        df=featured_df,
        config=env_config_obj,
    )

    # Run Backtest Loop
    obs, _ = env.reset()
    env.current_step = 99 # Skip warmup
    
    history = []
    initial_balance = 10000
    balance = initial_balance
    position = 0
    
    logger.info("Running backtest loop...")
    
    # Determine range
    start_idx = 100
    end_idx = len(df)
    
    for i in range(start_idx, end_idx):
        action, _ = model.predict(obs, deterministic=True) # Deterministic for backtest
        action_raw = float(action[0])
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Record step data
        current_price = df["close"].iloc[i]
        timestamp = df["timestamp"].iloc[i] if "timestamp" in df.columns else i
        
        step_data = {
            "timestamp": timestamp,
            "price": current_price,
            "action_raw": action_raw,
            "balance": info.get("balance", balance),
            "position": info.get("position", position),
            "portfolio_value": info.get("portfolio_value", balance),
            "reward": reward
        }
        history.append(step_data)
        
        balance = info.get("balance", balance)
        position = info.get("position", position)
        
        if i % 1000 == 0:
            logger.info(f"Processed {i}/{end_idx} steps. PV: {step_data['portfolio_value']:.2f}")

    # Save History
    history_df = pd.DataFrame(history)
    csv_path = output_dir / "backtest_detailed_results.csv"
    history_df.to_csv(csv_path, index=False)
    logger.info(f"✅ Detailed results saved to {csv_path}")
    
    return history_df

def analyze_results(df, output_dir):
    """Analyzes the backtest results and generates plots."""
    logger.info("📊 Analyzing results...")
    output_dir = Path(output_dir)
    
    # 1. PnL Curve
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], df['portfolio_value'], label='Portfolio Value')
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Time')
    plt.ylabel('Value (JPY)')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / "pnl_curve.png")
    plt.close()
    
    # 2. Action Distribution
    plt.figure(figsize=(10, 6))
    plt.hist(df['action_raw'], bins=50, alpha=0.7, color='blue')
    plt.title('Action Distribution (Raw)')
    plt.xlabel('Action Value [-1, 1]')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig(output_dir / "action_distribution.png")
    plt.close()
    
    # 3. Drawdown
    peak = df['portfolio_value'].cummax()
    drawdown = (df['portfolio_value'] - peak) / peak
    
    plt.figure(figsize=(12, 6))
    plt.plot(df['timestamp'], drawdown, label='Drawdown', color='red')
    plt.title('Drawdown Over Time')
    plt.xlabel('Time')
    plt.ylabel('Drawdown %')
    plt.fill_between(df['timestamp'], drawdown, 0, color='red', alpha=0.3)
    plt.grid(True)
    plt.savefig(output_dir / "drawdown.png")
    plt.close()
    
    # Metrics
    initial_pv = df['portfolio_value'].iloc[0]
    final_pv = df['portfolio_value'].iloc[-1]
    total_return = (final_pv - initial_pv) / initial_pv * 100
    max_dd = drawdown.min() * 100
    
    # Sharpe Ratio (assuming 1m data)
    returns = df['portfolio_value'].pct_change().dropna()
    sharpe = returns.mean() / returns.std() * np.sqrt(252 * 24 * 60) if returns.std() != 0 else 0
    
    report = f"""
# Backtest Analysis Report - SAC v454

## Performance Metrics
- **Initial Balance**: {initial_pv:.2f} JPY
- **Final Balance**: {final_pv:.2f} JPY
- **Total Return**: {total_return:.2f}%
- **Max Drawdown**: {max_dd:.2f}%
- **Sharpe Ratio**: {sharpe:.4f}

## Trade Statistics
- **Total Steps**: {len(df)}
- **Buy Actions**: {len(df[df['action_raw'] > 0.33])}
- **Sell Actions**: {len(df[df['action_raw'] < -0.33])}
- **Hold Actions**: {len(df[(df['action_raw'] >= -0.33) & (df['action_raw'] <= 0.33)])}

## Plots
- [PnL Curve](pnl_curve.png)
- [Action Distribution](action_distribution.png)
- [Drawdown](drawdown.png)
    """
    
    with open(output_dir / "analysis_report.md", "w") as f:
        f.write(report)
    
    logger.info(f"✅ Analysis complete. Report saved to {output_dir / 'analysis_report.md'}")
    print(report)

def main():
    model_name = "sac_v454_inverse_confidence"
    config_path = "config/v454/sac_v454_config.json"
    output_dir = "backtest_results/v454_detailed"
    
    df = run_detailed_backtest(model_name, config_path, output_dir)
    if df is not None:
        analyze_results(df, output_dir)
    else:
        logger.error("Backtest failed to produce results.")

if __name__ == "__main__":
    main()
